"""Measured resource use and explicit incomplete API-usage accounting."""
import math
import resource
import sys
from time import perf_counter, process_time
from statistics import mean
from eicr.contracts import require


def environment_identity():
    import platform
    from importlib.metadata import version, PackageNotFoundError
    packages={}
    for name in ('numpy','scipy','tiktoken','tokenizers'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),'packages':packages}


class Meter:
    def __init__(self):
        self.wall, self.cpu = perf_counter(), process_time()
    def result(self):
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {'wall_seconds':perf_counter()-self.wall, 'cpu_seconds':process_time()-self.cpu,
                'process_peak_rss_bytes':int(rss if sys.platform=='darwin' else rss*1024),
                'memory_scope':'process lifetime high-water mark; not a per-query incremental peak'}


def usage(calls):
    inputs=outputs=0; missing=0
    for call in calls:
        u=call.get('usage') or {}
        i=u.get('prompt_tokens',u.get('input_tokens')); o=u.get('completion_tokens',u.get('output_tokens'))
        if type(i) is not int or type(o) is not int or i<0 or o<0:
            missing+=1; continue
        inputs+=i; outputs+=o
    return {'calls':len(calls),'failed_calls':sum(c.get('status')!='ok' for c in calls),
            'known_input_tokens':inputs,'known_output_tokens':outputs,
            'total_tokens':inputs+outputs if missing==0 else None,'usage_complete':missing==0,'missing_usage_calls':missing}


def cost_ledger(run, prices=None):
    stages={k:usage(v) for k,v in run['calls'].items()}
    total=usage([c for values in run['calls'].values() for c in values])
    total['usd']=None
    if prices is not None:
        require(set(prices)=={'input_per_million','output_per_million'} and
                all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in prices.values()), 'Invalid price table')
        if total['usage_complete'] and not any(c.get('billing_kind') in ('rerank_tokens','embedding_tokens') for calls in run['calls'].values() for c in calls):
            total['usd']=(total['known_input_tokens']*prices['input_per_million']+
                          total['known_output_tokens']*prices['output_per_million'])/1e6
    billing = {}
    for calls in run['calls'].values():
        for c in calls: billing.setdefault(c.get('billing_kind','chat_tokens'), []).append(c)
    return {'stages':stages,'by_billing_kind':{k:usage(v) for k,v in billing.items()},'independent_deployment':total,
            'experimental_answer_reused':run.get('answer_reused_in_experiment',False),
            'unpriced_rerank':any(c.get('billing_kind') == 'rerank_tokens' for calls in run['calls'].values() for c in calls),
            'unpriced_embeddings':any(c.get('billing_kind') == 'embedding_tokens' for calls in run['calls'].values() for c in calls),
            'note':'Token totals mix billing units; inspect by_billing_kind for comparisons. Reused traces retain deployment cost. A chat-model price must not be applied to reranking or embedding tokens; USD is unknown until priced separately.'}


def percentile(values,p):
    if not values: return None
    xs=sorted(values); i=(len(xs)-1)*p; lo=int(i); hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(i-lo)


def break_even(added_build_cost, baseline_query_cost, method_query_cost):
    require(all(type(v) in (float,int) and math.isfinite(v) and v>=0
                for v in (added_build_cost,baseline_query_cost,method_query_cost)), 'Invalid lifecycle cost')
    saved=baseline_query_cost-method_query_cost
    return {'added_build_cost':added_build_cost,'saved_per_query':saved,
            'break_even_queries':math.ceil(added_build_cost/saved) if saved>0 else None,
            'amortizable':saved>0}


def cost_report(runs, prices=None, offline=None):
    grouped={}
    for run in runs:
        grouped.setdefault(run['arm'],[]).append(run)
    report={}
    for arm, rows in grouped.items():
        full_ledgers=[cost_ledger(r,prices) for r in rows]
        ledgers=[l['independent_deployment'] for l in full_ledgers]
        kinds=sorted({k for ledger in full_ledgers for k in ledger['by_billing_kind']})
        kind_summary={}
        for kind in kinds:
            entries=[ledger['by_billing_kind'].get(kind,usage([])) for ledger in full_ledgers]
            kind_summary[kind]={'mean_tokens':mean(e['total_tokens'] for e in entries) if all(e['usage_complete'] for e in entries) else None,
                                'calls':sum(e['calls'] for e in entries),'missing_usage_runs':sum(not e['usage_complete'] for e in entries)}
        # Never turn shared-setup replay fragments into end-to-end latency.
        measured=[r for r in rows if r.get('benchmark',{}).get('independent_request')
                  and not r.get('benchmark',{}).get('rerank_cache_reused')
                  and not (r.get('semantic_rerank') or {}).get('cache_reused')]
        durations=[r['benchmark']['wall_seconds'] for r in measured if r['benchmark'].get('has_answer_call')]
        retrieval=[r['timing_seconds']['retrieval_total'] for r in rows if r.get('benchmark',{}).get('independent_request')]
        complete=all(l['usage_complete'] for l in ledgers)
        report[arm]={'runs':len(rows),'by_billing_kind':kind_summary,'mean_online_tokens':mean(l['total_tokens'] for l in ledgers) if complete else None,
            'mean_usd':mean(l['usd'] for l in ledgers) if all(l['usd'] is not None for l in ledgers) else None,
            'missing_usage_runs':sum(not l['usage_complete'] for l in ledgers),
            'native_protocol_error_runs':sum(r.get('native_loop_result',{}).get('status')=='native_loop_error' for r in rows),
            'frontend_error_runs':sum(bool(r.get('query',{}).get('frontend_errors')) or
                r.get('query',{}).get('plan',{}).get('fallback_reason') in ('frontend_unavailable','invalid_plan') for r in rows),
            'answer_error_runs':sum(bool(r.get('answer_error')) for r in rows),
            'total_calls':sum(l['calls'] for l in ledgers),'failed_calls':sum(l['failed_calls'] for l in ledgers),
            'independent_latency_runs':len(measured),'end_to_end_latency_runs':len(durations),
            'frontend_modes':sorted({r['benchmark']['frontend_mode'] for r in measured}),
            'live_frontend_included':all(r['benchmark']['frontend_mode']=='live' or arm=='P0' for r in measured) if measured else None,
            'mean_end_to_end_seconds':mean(durations) if durations else None,
            'end_to_end_p50_seconds':percentile(durations,.5),
            'end_to_end_p95_seconds':percentile(durations,.95),'retrieval_p50_seconds':percentile(retrieval,.5),
            'retrieval_p95_seconds':percentile(retrieval,.95),
            'mean_context_tokens':mean(r['evidence_pack']['used_tokens'] for r in rows)}
    lifecycle={}
    if offline is not None:
        require(set(offline)=={'baseline_arm','method_arm','added_build_seconds','added_build_usd','queries_per_snapshot'}, 'Invalid lifecycle inputs')
        a,b=report[offline['baseline_arm']],report[offline['method_arm']]
        for unit,field,build in (('time','mean_end_to_end_seconds','added_build_seconds'),('money','mean_usd','added_build_usd')):
            if a[field] is not None and b[field] is not None:
                lifecycle[unit]=break_even(offline[build],a[field],b[field])
                n=offline['queries_per_snapshot'];require(type(n) is int and n>0,'Invalid update horizon')
                lifecycle[unit]['net_saving_per_snapshot']=n*(a[field]-b[field])-offline[build]
    return {'arms':report,'lifecycle':lifecycle,'offline_inputs':offline,
            'latency_rule':'Only independently timed requests contribute; replay timings are excluded.'}
