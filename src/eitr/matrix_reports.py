"""Build accounting, per-snapshot amortization and paired repeat summaries."""
from collections import defaultdict
import math
import random
from statistics import mean
from eicr.contracts import digest, require, read_json
from .accounting import cost_ledger, percentile
from .pricing import price_calls, validate_prices


def build_accounting(metrics, prices=None, compute_usd_per_cpu_hour=None):
    require(metrics.get('schema')=='eitr-matrix-build-metrics-1' and metrics.get('metrics_hash')==digest(
        {k:v for k,v in metrics.items() if k!='metrics_hash'}),'Build metrics hash/schema mismatch')
    require(compute_usd_per_cpu_hour is None or (type(compute_usd_per_cpu_hour) in (int,float) and
            math.isfinite(compute_usd_per_cpu_hour) and compute_usd_per_cpu_hour>=0),'Invalid compute rate')
    require(all(type(metrics['resources'].get(k)) in (int,float) and math.isfinite(metrics['resources'][k]) and
                metrics['resources'][k]>=0 for k in ('cpu_seconds','wall_seconds')), 'Invalid build resource measurement')
    require(type(metrics.get('embedding_reused')) is bool, 'Build reuse flag required')
    require(type(metrics.get('embedding_stage_seconds')) in (int,float) and math.isfinite(metrics['embedding_stage_seconds']) and
            0<=metrics['embedding_stage_seconds']<=metrics['resources']['wall_seconds'], 'Invalid embedding stage timing')
    deployment=price_calls(metrics['offline_embedding_calls'],prices)
    actual=price_calls(metrics['actual_embedding_calls'],prices)
    compute=metrics['resources']['cpu_seconds']*compute_usd_per_cpu_hour/3600 if compute_usd_per_cpu_hour is not None else None
    return {'index_hash':metrics['index_hash'],'embedding_record_hash':metrics['embedding_record_hash'],
            'embedding_pricing':deployment,'actual_embedding_pricing':actual,'compute_usd':compute,
            'deployment_usd':deployment['usd']+compute if deployment['usd'] is not None and compute is not None else None,
            'actual_usd':actual['usd']+compute if actual['usd'] is not None and compute is not None else None,
            'measured_wall_seconds':metrics['resources']['wall_seconds'],
            'seconds_eligible_for_lifecycle':metrics['resources']['wall_seconds'] if not metrics['embedding_reused'] else None,
            'non_embedding_wall_seconds':max(0.,metrics['resources']['wall_seconds']-metrics['embedding_stage_seconds']),
            'snapshot_hash':metrics['snapshot_hash'],
            'time_note':'A reused/imported embedding build is not a complete cold build; only measured time is reported.'}


def lifecycle_report(runs,spec,prices=None):
    """Compare incremental build cost and paired online requests per snapshot.

    Shared components in both arms cancel through subtraction. Each arm's list
    is an explicit costed asset inventory; no automatic double-counting of a
    component repeated in that list. Missing prices/timings remain unknown.
    """
    validate_prices(prices)
    require(isinstance(spec,dict) and set(spec)=={'comparisons'} and isinstance(spec['comparisons'],list),'Invalid lifecycle spec')
    result=[];seen=set()
    for c in spec['comparisons']:
        require(isinstance(c,dict) and set(c)=={'owner_id','snapshot_id','baseline_arm','method_arm','baseline_builds',
                'method_builds','queries_per_snapshot','compute_usd_per_cpu_hour'},'Invalid lifecycle comparison')
        key=tuple(c[k] for k in ('owner_id','snapshot_id','baseline_arm','method_arm'))
        require(key not in seen and c['baseline_arm']!=c['method_arm'],'Duplicate/self lifecycle comparison');seen.add(key)
        n=c['queries_per_snapshot'];require(type(n) is int and n>0,'Invalid amortization horizon')
        builds={};totals={}
        for side in ('baseline','method'):
            paths=c[side+'_builds'];require(isinstance(paths,list) and paths and len(set(paths))==len(paths),'Pin unique build components')
            records=[read_json(p) for p in paths]
            require(len({r['metrics_hash'] for r in records})==len(records),'Duplicate build receipts')
            require(all(r['owner_id']==c['owner_id'] and r['snapshot_id']==c['snapshot_id'] for r in records),'Build/snapshot mismatch')
            # A complete matrix build includes embeddings. Listing two builds of
            # the same embedding double-counts it; compare them as alternatives.
            require(len({r['embedding_record_hash'] for r in records})==len(records),'Duplicate embedding component in build inventory')
            builds[side]=[build_accounting(r,prices,c['compute_usd_per_cpu_hour']) for r in records]
            totals[side]={field:sum(r[field] for r in builds[side]) if all(r[field] is not None for r in builds[side]) else None
                          for field in ('deployment_usd','seconds_eligible_for_lifecycle')}
        require(len({r['snapshot_hash'] for side in builds.values() for r in side})==1,
                'Lifecycle compares different memory contents')
        shared_embedding_time_excluded=False
        # When the identical document embedding component is shared, its unknown
        # historical latency cancels. Report non-embedding build totals explicitly.
        if {r['embedding_record_hash'] for r in builds['baseline']}=={r['embedding_record_hash'] for r in builds['method']}:
            if any(totals[s]['seconds_eligible_for_lifecycle'] is None for s in totals):
                shared_embedding_time_excluded=True
                for s in totals:totals[s]['seconds_eligible_for_lifecycle']=sum(r['non_embedding_wall_seconds'] for r in builds[s])
        chosen={}
        for side in ('baseline','method'):
            selected={}
            for r in runs:
                j=r.get('job',{});b=r.get('benchmark',{})
                if b.get('warmup') or (j.get('owner_id'),j.get('snapshot_id'),r['arm'])!=(c['owner_id'],c['snapshot_id'],c[side+'_arm']):continue
                require(r.get('index_hash') in {b['index_hash'] for b in builds[side]}, 'Online run/build index mismatch')
                k=(j['question_id'],j['trial']);require(k not in selected,'Duplicate lifecycle sample');selected[k]=r
            chosen[side]=selected
        require(chosen['baseline'] and set(chosen['baseline'])==set(chosen['method']),'Lifecycle needs identical paired question/trial samples')
        # Online dollars use all attempts (including failures); latency uses only
        # successful paired uncached complete requests, with exclusions reported.
        def online_cost(r):
            api=cost_ledger(r,prices)['independent_deployment']['usd'];rate=c['compute_usd_per_cpu_hour']
            cpu=r.get('resources_independent',{}).get('cpu_seconds')
            if api is None or rate is None or (rate>0 and (type(cpu) not in (int,float) or not math.isfinite(cpu) or cpu<0)):return None
            return api+(cpu*rate/3600 if rate else 0.)
        online_usd={s:[online_cost(r) for r in chosen[s].values()] for s in chosen}
        eligible=lambda r:(r.get('benchmark',{}).get('independent_request') and r['benchmark'].get('has_answer_call') and
            not r.get('answer_error') and not any(r['benchmark'].get(k) for k in ('frontend_cache_reused','embedding_cache_reused','rerank_cache_reused')))
        paired=[k for k in chosen['baseline'] if all(eligible(chosen[s][k]) for s in chosen)]
        row={'scope':{k:c[k] for k in ('owner_id','snapshot_id','baseline_arm','method_arm')},'queries_per_snapshot':n,
             'shared_embedding_time_excluded':shared_embedding_time_excluded,
             'paired_attempts':len(chosen['baseline']),'paired_latency_samples':len(paired),
             'builds':builds,'totals':totals,'units':{},
             'quality_note':'Amortization is economic arithmetic, not quality equivalence. Compare quality independently.'}
        for unit,field in (('usd','deployment_usd'),('seconds','seconds_eligible_for_lifecycle')):
            online={}
            for s in chosen:
                vals=online_usd[s] if unit=='usd' else [chosen[s][k]['benchmark']['wall_seconds'] for k in paired]
                online[s]=mean(vals) if vals and all(v is not None for v in vals) else None
            base,method=totals['baseline'][field],totals['method'][field]
            if None in (base,method,online['baseline'],online['method']):
                row['units'][unit]={'complete':False,'reason':'missing build cost/full-build timing, rate, usage or paired independent latency'};continue
            added=method-base;saved=online['baseline']-online['method']
            point=(0 if added<=0 and saved>=0 else math.ceil(added/saved) if added>0 and saved>0 else None)
            row['units'][unit]={'complete':True,'added_build_cost':added,'saved_per_query':saved,
                'break_even_queries':point,'net_saving_at_horizon':n*saved-added,
                'baseline_total_at_horizon':base+n*online['baseline'],'method_total_at_horizon':method+n*online['method'],
                'online_means':online}
        result.append(row)
    return {'schema':'eitr-matrix-lifecycle-1','prices_hash':digest(prices) if prices is not None else None,'comparisons':result}


def summarize_repeats(inputs, *, draws=2000, seed=0):
    require(isinstance(inputs,list) and inputs and type(draws) is int and draws>=100 and type(seed) is int,'Invalid repeat summary')
    samples={};profiles=set();bytrial=defaultdict(set)
    for entry in inputs:
        require(set(entry)=={'trial','rows'} and type(entry['trial']) is int and entry['trial']>=0,'Invalid repeat input')
        rows=read_json(entry['rows']);require(isinstance(rows,list) and rows,'Empty graded repeat')
        for r in rows:
            require(type(r['correct']) is bool,'Invalid repeat outcome')
            key=(r['owner_id'],r['question_id'],r['arm'],entry['trial'])
            require(key not in samples,'Duplicate repeated sample');samples[key]=r
            profiles.add((r.get('judge_profile','semantic_strict'),r.get('judge_profile_hash')))
            bytrial[entry['trial']].add(key[:3])
    require(len(profiles)==1 and all(keys==next(iter(bytrial.values())) for keys in bytrial.values()),'Mixed rubric or incomplete repeat coverage')
    averaged=defaultdict(list)
    for (owner,q,arm,trial),r in samples.items():averaged[(owner,q,arm)].append(int(r['correct']))
    means={k:mean(v) for k,v in averaged.items()};arms=sorted({k[2] for k in means});pairs={};rng=random.Random(seed)
    for i,a in enumerate(arms):
        for b in arms[i+1:]:
            ka={(o,q) for o,q,arm in means if arm==a};kb={(o,q) for o,q,arm in means if arm==b}
            require(ka==kb,'Unpaired repeated arms')
            byowner=defaultdict(list)
            for o,q in sorted(ka):byowner[o].append(means[o,q,b]-means[o,q,a])
            owners=sorted(byowner);boot=[]
            if len(owners)>1:
                for _ in range(draws):
                    chosen=[rng.choice(owners) for _ in owners]
                    boot.append(mean(d for o in chosen for d in byowner[o]))
            pairs[a+' vs '+b]={'delta_b_minus_a':mean(d for vs in byowner.values() for d in vs),
                'paired_questions':len(ka),'owners':len(owners),'percentile_95':[percentile(boot,.025),percentile(boot,.975)] if boot else None,
                'few_owner_warning':len(owners)<10}
    validity={}
    if all('answer_contract_valid' in r for r in samples.values()):
        for a in arms:
            group=[r for (o,q,arm,t),r in samples.items() if arm==a]
            validity[a]={'mean_answer_contract_valid':mean(r['answer_contract_valid'] for r in group),
                         'mean_valid_and_correct':mean(r['correct'] and r['answer_contract_valid'] for r in group)}
    return {'validity':validity,'judge_profile_hash':next(iter(profiles))[1],
            'schema':'eitr-matrix-repeated-summary-1','repeats':len(bytrial),'judge_profile':next(iter(profiles))[0],
            'arms':{a:{'questions':sum(arm==a for o,q,arm in means),'mean_question_accuracy':mean(v for (o,q,arm),v in means.items() if arm==a)} for a in arms},
            'pairs':pairs,'draws':draws,'seed':seed,'noninferiority_established':False,
            'method':'Average repetitions per question before paired owner-cluster bootstrap; repeats do not increase question count.'}
