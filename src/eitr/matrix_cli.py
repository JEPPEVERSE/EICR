"""Explicit v3 commands. Historical eitr commands retain their original meaning."""
from pathlib import Path
from time import perf_counter
from eicr.contracts import read_json, write_json, require, digest
from .accounting import Meter, environment_identity
from .evidence import Corpus
from .experiment import tokenizer_from, model_backend
from .matrix_embeddings import EmbeddingBackend, embedding_options
from .matrix_index import build_index, load_index, embed_snapshot, compare_graphs
from .matrix_query import plan_record
from .matrix_retrieval import MatrixConfig, ARMS
from .matrix_pipeline import prepare_query, retrieve, replay
from .matrix_experiment import freeze, verify_protocol, run_protocol
from .rerank import backend_from_spec


def add_parsers(sub):
    p=sub.add_parser('matrix-build',help='Build fact-row/tag-membership index and optional auxiliary graph')
    p.add_argument('--snapshot',required=True,help='Canonical snapshot or historical index containing a snapshot')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--embedding-spec');g.add_argument('--embedding-record')
    p.add_argument('--graph-mode',choices=('exact','shortlist'),default='exact')
    p.add_argument('--graph-top-k',type=int,default=0);p.add_argument('--graph-shortlist',type=int,default=32)
    p.add_argument('--block',type=int,default=128);p.add_argument('--out',required=True)
    p=sub.add_parser('matrix-compare-graphs')
    for arg in ('exact-index','approximate-index','out'):p.add_argument('--'+arg,required=True)
    p=sub.add_parser('matrix-benchmark-freeze')
    for arg in ('spec','out'):p.add_argument('--'+arg,required=True)
    for command in ('matrix-benchmark-preflight','matrix-benchmark'):
        p=sub.add_parser(command)
        for arg in ('protocol','out'):p.add_argument('--'+arg,required=True)
    p=sub.add_parser('matrix-cost-report')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--benchmark');g.add_argument('--benchmarks',nargs='+');g.add_argument('--runs')
    p.add_argument('--prices');p.add_argument('--builds-spec');p.add_argument('--out',required=True)
    p=sub.add_parser('matrix-build-cost')
    for arg in ('metrics','prices','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--compute-usd-per-cpu-hour',type=float)
    p=sub.add_parser('matrix-summarize-repeats')
    for arg in ('inputs','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=0)
    p=sub.add_parser('matrix-prepare-corpus')
    for arg in ('index','tokenizer-spec','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--config')
    p=sub.add_parser('matrix-prepare-query')
    for arg in ('index','question','embedding-spec','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--config');p.add_argument('--model-spec');p.add_argument('--plan')
    p.add_argument('--plan-cache-dir');p.add_argument('--plan-cache-only',action='store_true')
    p=sub.add_parser('matrix-retrieve')
    for arg in ('index','corpus','tokenizer-spec','question','prepared','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--config');p.add_argument('--arm',choices=ARMS,default='matrix_rerank')
    p.add_argument('--reranker-spec');p.add_argument('--answer-model-spec')
    p=sub.add_parser('matrix-replay')
    for arg in ('index','corpus','tokenizer-spec','run','out'):p.add_argument('--'+arg,required=True)
    for name in ('matrix-freeze','matrix-preflight','matrix-experiment'):
        p=sub.add_parser(name);p.add_argument('--spec' if name=='matrix-freeze' else '--protocol',required=True)
        p.add_argument('--out',required=True)
    p=sub.add_parser('matrix-grade')
    for arg in ('index','batch','labels','model-spec','out'):p.add_argument('--'+arg,required=True)
    p.add_argument('--judge-profile',choices=('semantic_strict','mragent_lenient'),default='semantic_strict')


def run(args):
    cmd=args.command;require(not Path(args.out).exists(),'Output exists; choose a new path')
    meter=Meter()
    cfg=MatrixConfig.from_dict(read_json(args.config)) if getattr(args,'config',None) else MatrixConfig()
    if cmd=='matrix-build':
        data=read_json(args.snapshot);snapshot=data.get('snapshot',data)
        require(not Path(args.out+'.metrics.json').exists(),'Build metrics output exists')
        backend=None;start=perf_counter()
        if args.embedding_record: record=read_json(args.embedding_record)
        else:
            backend=EmbeddingBackend(read_json(args.embedding_spec));record=embed_snapshot(snapshot,backend,**embedding_options(backend))
        embedding_stage_seconds=perf_counter()-start;start=perf_counter()
        index=build_index(snapshot,record,graph_top_k=args.graph_top_k,graph_shortlist=args.graph_shortlist,
                          block=args.block,graph_mode=args.graph_mode)
        graph_stage_seconds=perf_counter()-start
        index.save(args.out)
        metrics={'schema':'eitr-matrix-build-metrics-1','index_hash':index.index_hash,'resources':meter.result(),
            'owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'snapshot_hash':snapshot['content_hash'],
            'embedding_record_hash':record['record_hash'],'embedding_stage_seconds':embedding_stage_seconds,
            'graph_stage_seconds':graph_stage_seconds,'environment':environment_identity(),
            'offline_embedding_calls':record['calls'],'actual_embedding_calls':list(backend.calls) if backend else [],
            'embedding_reused':backend is None or not backend.calls or len(backend.calls)<len(record['calls']),
            'facts':len(index.facts),'tags':len(index.tag_ids),'dimension':index.X.shape[1],
            'graph_config':index.graph_config,'graph_edges':sum(map(len,index.graph.values())),
            'note':'Measured build wall/CPU excludes corpus preparation. Reused embedding traces price deployment; actual calls price this execution.'}
        metrics['metrics_hash']=digest(metrics);write_json(args.out+'.metrics.json',metrics)
        return 0
    if cmd=='matrix-compare-graphs': result=compare_graphs(load_index(args.exact_index),load_index(args.approximate_index))
    elif cmd=='matrix-benchmark-freeze':
        from .matrix_benchmark import freeze_benchmark
        result=freeze_benchmark(read_json(args.spec))
    elif cmd=='matrix-benchmark-preflight':
        from .matrix_benchmark import verify_benchmark
        p=verify_benchmark(read_json(args.protocol));result={'verified':True,'protocol_hash':p['protocol_hash'],'requests':len(p['jobs'])}
    elif cmd=='matrix-benchmark':
        from .matrix_benchmark import run_benchmark
        run_benchmark(read_json(args.protocol),args.out);return 0
    elif cmd=='matrix-cost-report':
        from .matrix_benchmark import load_benchmark_runs, benchmark_report
        from .matrix_reports import lifecycle_report
        from .accounting import cost_report
        roots=args.benchmarks or ([args.benchmark] if args.benchmark else [])
        tables=[read_json(Path(root)/'protocol.json')['spec']['prices'] for root in roots]
        require(args.prices or len({digest(t) for t in tables})<=1,'Merged benchmarks need identical frozen prices or explicit repricing')
        prices=read_json(args.prices) if args.prices else (tables[0] if tables else None)
        runs=[r for root in roots for r in load_benchmark_runs(root)] if roots else read_json(args.runs)
        if roots:
            keys=[tuple(r['job'][k] for k in ('owner_id','snapshot_id','question_id','trial','arm','warmup')) for r in runs]
            require(len(keys)==len(set(keys)),'Duplicate benchmark samples in merged reports')
        result=benchmark_report(runs,prices) if roots else cost_report(runs,prices)
        if args.builds_spec:result['lifecycle']=lifecycle_report(runs,read_json(args.builds_spec),prices)
    elif cmd=='matrix-build-cost':
        from .matrix_reports import build_accounting
        result=build_accounting(read_json(args.metrics),read_json(args.prices),args.compute_usd_per_cpu_hour)
    elif cmd=='matrix-summarize-repeats':
        from .matrix_reports import summarize_repeats
        result=summarize_repeats(read_json(args.inputs),draws=args.draws,seed=args.seed)
    elif cmd=='matrix-freeze': result=freeze(read_json(args.spec))
    elif cmd=='matrix-preflight':
        p=verify_protocol(read_json(args.protocol));result={'verified':True,'protocol_hash':p['protocol_hash'],'question_count':p['question_count']}
    elif cmd=='matrix-experiment':run_protocol(read_json(args.protocol),args.out);return 0
    else:
        index=load_index(args.index)
        if cmd=='matrix-grade':
            from .evaluate import grade_batch
            grade_batch(args.batch,read_json(args.labels),index,model_backend(read_json(args.model_spec)),args.out,
                        judge_profile=args.judge_profile);return 0
        if cmd=='matrix-prepare-query':
            require(not(args.model_spec and args.plan),'Choose live or supplied plan')
            backend=EmbeddingBackend(read_json(args.embedding_spec))
            result=prepare_query(index,args.question,backend,config=cfg,
                frontend_backend=model_backend(read_json(args.model_spec)) if args.model_spec else None,
                supplied_plan=read_json(args.plan) if args.plan else None,
                plan_cache_dir=args.plan_cache_dir,plan_cache_only=args.plan_cache_only,
                embedding_options=embedding_options(backend))
        else:
            tok_spec=read_json(args.tokenizer_spec);tok=tokenizer_from(tok_spec)
            if cmd=='matrix-prepare-corpus': Corpus(index.snapshot,tok,cfg).save(args.out);return 0
            if cmd=='matrix-replay':
                prior=read_json(args.run);cfg=MatrixConfig.from_dict(prior['manifest']['config'])
                corpus=Corpus.load(args.corpus,index.snapshot,tok,cfg);result=replay(index,corpus,prior)
            else:
                corpus=Corpus.load(args.corpus,index.snapshot,tok,cfg)
                rs=read_json(args.reranker_spec) if args.reranker_spec else None
                require(tok_spec['kind']!='characters' or not(args.answer_model_spec or (rs and not rs['cache_only'])),
                        'Live answer/rerank calls require a real evidence tokenizer')
                result=retrieve(index,args.question,corpus,read_json(args.prepared),config=cfg,arm=args.arm,
                    reranker=backend_from_spec(rs) if rs else None,rerank_cache_dir=rs['cache_dir'] if rs else None,
                    rerank_cache_only=rs['cache_only'] if rs else False,
                    answer_backend=model_backend(read_json(args.answer_model_spec)) if args.answer_model_spec else None)
    write_json(args.out,result);return 0
