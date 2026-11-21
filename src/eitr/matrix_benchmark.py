"""Independent complete requests with pinned order, failures, warmups and spend.

The base corpus/index may be preloaded (warm) or loaded per request (cold).
Provider server caches cannot be disabled by this client; reported provider
cached-token usage is retained separately from local replay caches.
"""
from copy import deepcopy
from pathlib import Path
import random
from time import perf_counter
from eicr.contracts import digest, require, read_json, write_json
from .accounting import Meter, cost_ledger, cost_report
from .pricing import validate_prices, price_calls
from .evidence import Corpus
from .experiment import tokenizer_from, model_backend
from .matrix_index import load_index
from .matrix_embeddings import EmbeddingBackend
from .matrix_experiment import freeze, verify_protocol
from .matrix_retrieval import MatrixConfig
from .matrix_pipeline import prepare_query, retrieve
from .rerank import backend_from_spec
from .pipeline import runtime_source_hash


def freeze_benchmark(spec):
    require(isinstance(spec,dict) and set(spec)=={'experiment','settings','prices'},'Pin benchmark experiment/settings/prices')
    settings=spec['settings']
    require(isinstance(settings,dict) and set(settings)=={'repeats','warmup_repeats','seed','cache_policy','load_mode'},'Invalid benchmark settings')
    require(type(settings['repeats']) is int and settings['repeats']>0 and
            type(settings['warmup_repeats']) is int and settings['warmup_repeats']>=0 and
            type(settings['seed']) is int and settings['cache_policy'] in ('bypass','reuse','only') and
            settings['load_mode'] in ('warm','cold'),'Invalid benchmark policy')
    validate_prices(spec['prices'])
    effective=deepcopy(spec['experiment']);only=settings['cache_policy']=='only'
    effective['embedding']['cache_only']=only;effective['plan_cache_only']=only
    if effective['reranker']:effective['reranker']['cache_only']=only
    base=freeze(effective,inspect_caches=settings['cache_policy']!='bypass')
    questions=read_json(effective['questions']);jobs=[];rng=random.Random(settings['seed'])
    # Shuffle complete owner/question/trial blocks, then rotate arm order in each.
    for warmup,count in ((True,settings['warmup_repeats']),(False,settings['repeats'])):
        blocks=[(trial,q) for trial in range(count) for q in questions];rng.shuffle(blocks)
        for trial,q in blocks:
            arms=list(effective['arms']);rng.shuffle(arms)
            for arm in arms:
                jobs.append({'owner_id':q['owner_id'],'snapshot_id':q['snapshot_id'],'question_id':q['question_id'],
                             'trial':trial,'warmup':warmup,'arm':arm})
    body={'schema':'eitr-matrix-benchmark-protocol-1','spec':deepcopy(spec),'experiment_protocol':base,'jobs':jobs,
          'runtime_source_hash':runtime_source_hash()}
    return {**body,'protocol_hash':digest(body)}


def verify_benchmark(protocol):
    require(protocol.get('schema')=='eitr-matrix-benchmark-protocol-1' and protocol.get('protocol_hash')==digest(
        {k:v for k,v in protocol.items() if k!='protocol_hash'}),'Benchmark protocol hash mismatch')
    verify_protocol(protocol['experiment_protocol'])
    current=freeze_benchmark(protocol['spec'])
    # Only new cache pins are allowed to differ after a legitimate prior run.
    current['experiment_protocol']=protocol['experiment_protocol']
    current['protocol_hash']=digest({k:v for k,v in current.items() if k!='protocol_hash'})
    require(current==protocol,'Benchmark code/settings/order changed')
    return protocol


def independent_request(index,corpus,question, *, config,arm,embedding_backend,frontend_backend=None,
                        answer_backend=None,reranker=None,supplied_plan=None,cache_policy='bypass',
                        plan_cache_dir=None,rerank_cache_dir=None,loader=None):
    """One timed request; optional loader is inside the measured interval.

    Returns a normal replayable run, or an explicit failure receipt. Unexpected
    programming exceptions are propagated, never disguised as model failures.
    """
    from eicr.provider import ProviderTransportError
    from eicr.contracts import ContractError
    require(cache_policy in ('bypass','reuse','only'),'Invalid benchmark cache policy')
    backends={'frontend':frontend_backend,'query_embedding':embedding_backend,'rerank':reranker,'answer':answer_backend}
    starts={k:len(b.calls) if b else 0 for k,b in backends.items()}
    tick=perf_counter();meter=Meter();prepared=None;progress={};run=None;stage='load';load_seconds=0.;prep_seconds=0.
    try:
        if loader:index,corpus=loader()
        load_seconds=perf_counter()-tick
        stage='prepare';prep_tick=perf_counter()
        prepared=prepare_query(index,question,embedding_backend,config=config,frontend_backend=frontend_backend,
            supplied_plan=supplied_plan,plan_cache_dir=plan_cache_dir if cache_policy!='bypass' else None,
            plan_cache_only=cache_policy=='only' and frontend_backend is not None,
            embedding_options={'cache_only':cache_policy=='only','cache_bypass':cache_policy=='bypass'},progress=progress)
        prep_seconds=perf_counter()-prep_tick;stage='retrieve'
        run=retrieve(index,question,corpus,prepared,config=config,arm=arm,reranker=reranker,
            rerank_cache_dir=rerank_cache_dir if cache_policy!='bypass' else None,
            rerank_cache_only=cache_policy=='only',answer_backend=answer_backend)
    except (ContractError,ProviderTransportError) as error:
        require(index is not None,'Cannot record request scope without an index')
        run={'schema':'eitr-matrix-failure-1','arm':arm,'question':question,'index_hash':index.index_hash,
             'snapshot_hash':index.snapshot['content_hash'],'prepared_query':prepared or progress or None,
             'evidence_pack':{'spans':[],'atom_ids':[],'used_tokens':0},'answer':None,'raw_answer':None,
             'answer_error':type(error).__name__,'failure_stage':stage,'timing_seconds':{},
             'manifest':{'status':'execution_failed','config':config.to_dict(),'runtime_source_hash':runtime_source_hash()}}
    wall=perf_counter()-tick
    actual={k:list(b.calls[starts[k]:]) if b else [] for k,b in backends.items()}
    # Successful runs retain cache-origin calls for deployment pricing. Failures
    # retain successful cached preparation plus actual later calls when available.
    if run['schema']=='eitr-matrix-failure-1':
        run['calls']=dict(actual)
        if 'plan' in progress:run['calls']['frontend']=progress['plan']['calls']
        if 'embedding' in progress:run['calls']['query_embedding']=progress['embedding']['calls']
    front_cached=bool(progress.get('plan',{}).get('calls') and not actual['frontend'])
    embedding_cached=bool(progress.get('embedding',{}).get('calls') and
                          len(actual['query_embedding'])<len(progress['embedding']['calls']))
    rerank_cached=bool((run.get('semantic_rerank') or {}).get('cache_reused'))
    run['actual_calls']=actual
    run['benchmark']={'independent_request':True,'wall_seconds':wall,'load_seconds':load_seconds,
        'prepare_seconds':prep_seconds,'load_mode':'cold' if loader else 'warm','cache_policy':cache_policy,
        'frontend_mode':'live' if frontend_backend else ('supplied' if supplied_plan is not None else 'whole_query'),
        'frontend_cache_reused':front_cached,'embedding_cache_reused':embedding_cached,'rerank_cache_reused':rerank_cached,
        'has_answer_call':bool(actual['answer']),'has_answer_backend':answer_backend is not None,
        'provider_cache_control':'Server prompt caches are not disabled; consult usage.cached_input_tokens.',
        'cold_scope':'Reload Python index/corpus objects; OS page cache and provider caches are not flushed.',
        'scope':'index/corpus load if cold + plan + query embedding + retrieval + rerank + packing + answer; excludes journal IO'}
    run['resources_independent']=meter.result();run['cost']=cost_ledger(run)
    run['run_hash']=digest({k:v for k,v in run.items() if k!='run_hash'})
    return run


def load_benchmark_runs(root):
    root=Path(root);manifest=read_json(root/'manifest.json')
    require(manifest.get('schema')=='eitr-matrix-benchmark-1' and manifest.get('status') in ('complete','complete_with_failures'),
            'Benchmark is incomplete')
    protocol=read_json(root/'protocol.json')
    require(protocol.get('protocol_hash')==manifest['protocol_hash'] and protocol.get('protocol_hash')==digest(
        {k:v for k,v in protocol.items() if k!='protocol_hash'}),'Benchmark receipt protocol mismatch')
    require(len(manifest['completed'])==len(protocol['jobs']),'Benchmark job coverage mismatch')
    assets={(a['owner_id'],a['snapshot_id']):a['index_hash'] for a in protocol['experiment_protocol']['asset_reports']}
    rows=[]
    for i,item in enumerate(manifest['completed']):
        require(item['file']==f'jobs/{i:06d}.json','Invalid benchmark job path')
        run=read_json(root/item['file'])
        require(item['content_hash']==digest(run) and run['run_hash']==digest({k:v for k,v in run.items() if k!='run_hash'}),
                'Benchmark job changed')
        job=protocol['jobs'][i]
        require(run.get('job')==job and run['arm']==job['arm'] and
                run['index_hash']==assets[(job['owner_id'],job['snapshot_id'])] and
                run['benchmark']['trial']==job['trial'] and run['benchmark']['warmup']==job['warmup'],
                'Benchmark job order/scope mismatch');rows.append(run)
    return rows


def benchmark_report(runs,prices=None):
    require(isinstance(runs,list) and runs,'No benchmark runs')
    result=cost_report(runs,prices)
    actual=[c for r in runs for calls in r.get('actual_calls',{}).values() for c in calls]
    result.update(schema='eitr-matrix-benchmark-report-1',actual_experimental_spend=price_calls(actual,prices),
        warmup_requests=sum(r['benchmark'].get('warmup',False) for r in runs),
        measured_requests=sum(not r['benchmark'].get('warmup',False) for r in runs),
        failed_requests=sum(bool(r.get('answer_error')) for r in runs),
        failed_measured_requests=sum(bool(r.get('answer_error')) and not r['benchmark'].get('warmup',False) for r in runs),
        failed_warmup_requests=sum(bool(r.get('answer_error')) and r['benchmark'].get('warmup',False) for r in runs),
        cost_scope='USD totals price provider API usage; lifecycle reports can additionally price measured CPU time.',
        interpretation='Latency quantiles describe successful uncached independent requests; failures and warmup spend remain reported.')
    return result


def run_benchmark(protocol,output_dir):
    verify_benchmark(protocol);out=Path(output_dir);require(not out.exists(),'Benchmark output exists');out.mkdir(parents=True)
    write_json(out/'protocol.json',protocol)
    spec=protocol['experiment_protocol']['spec'];settings=protocol['spec']['settings'];prices=protocol['spec']['prices']
    cfg=MatrixConfig.from_dict(spec['config']);tok=tokenizer_from(spec['tokenizer'])
    assets={};cold={}
    for asset in spec['assets']:
        index=load_index(asset['index']);key=(index.snapshot['owner_id'],index.snapshot['snapshot_id'])
        cold[key]=asset
        assets[key]=(index,Corpus.load(asset['corpus'],index.snapshot,tok,cfg) if settings['load_mode']=='warm' else None)
    questions={(q['owner_id'],q['snapshot_id'],q['question_id']):q for q in read_json(spec['questions'])}
    journal={'schema':'eitr-matrix-benchmark-1','protocol_hash':protocol['protocol_hash'],'status':'running','completed':[]}
    write_json(out/'manifest.json',journal);runs=[]
    try:
        for i,job in enumerate(protocol['jobs']):
            key=(job['owner_id'],job['snapshot_id']);q=questions[(*key,job['question_id'])];asset=cold[key]
            eb=EmbeddingBackend(spec['embedding']);fb=model_backend(spec['model']) if spec['frontend']=='llm' else None
            ab=model_backend(spec['model']) if spec['generate'] else None
            rb=backend_from_spec(spec['reranker']) if job['arm']!='matrix_embedding' else None
            def loader():
                idx=load_index(asset['index']);return idx,Corpus.load(asset['corpus'],idx.snapshot,tok,cfg)
            run=independent_request(*assets[key],q['question'],config=cfg,arm=job['arm'],embedding_backend=eb,
                frontend_backend=fb,answer_backend=ab,reranker=rb,supplied_plan=q.get('plan'),
                cache_policy=settings['cache_policy'],plan_cache_dir=spec['plan_cache_dir'],
                rerank_cache_dir=spec['reranker']['cache_dir'] if rb else None,loader=loader if settings['load_mode']=='cold' else None)
            run['job']=job;run['benchmark']['warmup']=job['warmup'];run['benchmark']['trial']=job['trial']
            run['cost']=cost_ledger(run,prices);run['run_hash']=digest({k:v for k,v in run.items() if k!='run_hash'})
            filename=f'jobs/{i:06d}.json';write_json(out/filename,run);runs.append(run)
            journal['completed'].append({'file':filename,'content_hash':digest(run)});write_json(out/'manifest.json',journal)
        # Export separate per-snapshot/trial grading batches; never count repeats as new questions.
        batches={}
        for run in runs:
            j=run['job']
            if j['warmup']:continue
            key=(j['owner_id'],j['snapshot_id'],j['trial']);batches.setdefault(key,{}).setdefault(j['question_id'],[]).append(run)
        journal['grading_batches']=[]
        for key,cases in batches.items():
            folder='grading/'+digest(list(key))[:24];batch={'schema':'eitr-matrix-batch-1','status':'complete',
                'index_hash':assets[key[:2]][0].index_hash,'arms':spec['arms'],'completed':[]}
            for qid,rows in cases.items():
                rows.sort(key=lambda r:spec['arms'].index(r['arm']))
                case={'question_id':qid,'question':questions[(*key[:2],qid)]['question'],'index_hash':batch['index_hash'],'rows':rows}
                filename=digest(qid)[:24]+'.json';write_json(out/folder/filename,case)
                batch['completed'].append({'question_id':qid,'file':filename,'content_hash':digest(case)})
            write_json(out/folder/'manifest.json',batch)
            journal['grading_batches'].append({'owner_id':key[0],'snapshot_id':key[1],'trial':key[2],'path':folder})
        journal['status']='complete_with_failures' if any(r.get('answer_error') for r in runs) else 'complete'
        write_json(out/'summary.json',benchmark_report(runs,prices))
    except Exception as error:
        journal.update(status='failed',error_type=type(error).__name__)
        if runs:write_json(out/'partial-summary.json',benchmark_report(runs,prices))
        raise
    finally:write_json(out/'manifest.json',journal)
    return journal
