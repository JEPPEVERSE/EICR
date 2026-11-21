"""Explicit v3 commands. Historical eitr commands retain their original meaning."""
from pathlib import Path
from eicr.contracts import read_json, write_json, require
from .accounting import Meter, environment_identity
from .evidence import Corpus
from .experiment import tokenizer_from, model_backend
from .matrix_embeddings import EmbeddingBackend, embedding_options
from .matrix_index import build_index, load_index, embed_snapshot
from .matrix_query import plan_record
from .matrix_retrieval import MatrixConfig, ARMS
from .matrix_pipeline import prepare_query, retrieve, replay
from .matrix_experiment import freeze, verify_protocol, run_protocol
from .rerank import backend_from_spec


def add_parsers(sub):
    p=sub.add_parser('matrix-build',help='Build fact-row/tag-membership index and optional auxiliary graph')
    p.add_argument('--snapshot',required=True,help='Canonical snapshot or historical index containing a snapshot')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--embedding-spec');g.add_argument('--embedding-record')
    p.add_argument('--graph-top-k',type=int,default=0);p.add_argument('--graph-shortlist',type=int,default=32)
    p.add_argument('--block',type=int,default=128);p.add_argument('--out',required=True)
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
        if args.embedding_record: record=read_json(args.embedding_record)
        else:
            backend=EmbeddingBackend(read_json(args.embedding_spec));record=embed_snapshot(snapshot,backend,**embedding_options(backend))
        index=build_index(snapshot,record,graph_top_k=args.graph_top_k,graph_shortlist=args.graph_shortlist,block=args.block)
        index.save(args.out)
        write_json(args.out+'.metrics.json',{'index_hash':index.index_hash,'resources':meter.result(),
            'environment':environment_identity(),'offline_embedding_calls':record['calls'],
            'facts':len(index.facts),'tags':len(index.tag_ids),'dimension':index.X.shape[1],
            'graph_config':index.graph_config,'graph_edges':sum(map(len,index.graph.values())),
            'note':'Embedding traces include original cached cost; not necessarily new calls in this build.'})
        return 0
    if cmd=='matrix-freeze': result=freeze(read_json(args.spec))
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
