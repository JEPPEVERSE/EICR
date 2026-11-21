"""Explicit offline preparation, frozen experiments, replay and blinded grading."""
import argparse
import json
from pathlib import Path
import sys
from eicr.contracts import ContractError, digest, read_json, require, write_json
from eicr.provider import ChatBackend
from .config import Config
from .native import canonicalize, import_pack
from .index import build_index, save_index, load_index
from .evidence import Corpus
from .pipeline import retrieve, compare, RUN_ARMS
from .explorer import explore
from .evaluate import summarize, paired_intervals, grade_batch, JUDGE_PROFILES
from .experiment import tokenizer_from, freeze, verify_protocol, run_protocol, benchmark, replay
from .native_runtime import pin_runtime, load_runtime
from .tool_backend import ToolBackend
from .accounting import cost_report, Meter, environment_identity


def token_args(p):
    p.add_argument('--tokenizer',required=True,help='characters (tests), json, or an explicit tiktoken encoding')
    p.add_argument('--tokenizer-file');p.add_argument('--tokenizer-sha256')


def model_args(p):
    p.add_argument('--api-base');p.add_argument('--model');p.add_argument('--key-env',default='OPENAI_API_KEY')
    p.add_argument('--max-output-tokens',type=int,default=2048)
    p.add_argument('--enable-thinking',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--json-mode',action=argparse.BooleanOptionalAction,default=True)


def parser():
    cli=argparse.ArgumentParser(prog='eitr');sub=cli.add_subparsers(dest='command',required=True)
    from .matrix_cli import add_parsers
    add_parsers(sub)
    imp=sub.add_parser('import-mragent')
    for name in ('pack','manifest','out'): imp.add_argument('--'+name,required=True)
    canon=sub.add_parser('canonicalize')
    for name in ('snapshot','out'): canon.add_argument('--'+name,required=True)
    build=sub.add_parser('build');build.add_argument('--snapshot',required=True);build.add_argument('--config')
    build.add_argument('--fact-vectors');build.add_argument('--out',required=True)
    audit=sub.add_parser('audit');audit.add_argument('--index',required=True)
    prep=sub.add_parser('prepare-corpus');prep.add_argument('--index',required=True);prep.add_argument('--config');prep.add_argument('--out',required=True);token_args(prep)
    pin=sub.add_parser('pin-mragent');pin.add_argument('--root',required=True);pin.add_argument('--out',required=True)
    for name in ('freeze','preflight','experiment'):
        p=sub.add_parser(name);p.add_argument('--spec' if name=='freeze' else '--protocol',required=True)
        if name!='preflight':p.add_argument('--out',required=True)
    align=sub.add_parser('align-fact-vectors')
    for name in ('index','vectors','out'):align.add_argument('--'+name,required=True)
    dates=sub.add_parser('attest-dates')
    for name in ('snapshot','conversation','out'):dates.add_argument('--'+name,required=True)
    for name in ('retrieve','compare','explore-proxy','batch','benchmark','replay'):
        p=sub.add_parser(name);p.add_argument('--index',required=True);p.add_argument('--config');p.add_argument('--out',required=True)
        p.add_argument('--corpus');token_args(p)
        if name=='replay':p.add_argument('--run',required=True);continue
        p.add_argument('--questions' if name in ('batch','benchmark') else '--question',required=True)
        p.add_argument('--frontend');p.add_argument('--llm-frontend',action='store_true');p.add_argument('--generate',action='store_true')
        p.add_argument('--native-root');p.add_argument('--native-manifest');model_args(p)
        p.add_argument('--native-frontend', action='store_true', help='MRAgent-style keyword alternatives + legal tag scoring')
        p.add_argument('--reranker-spec', help='Explicit semantic model/cache JSON, required for C_* arms')
        if name=='retrieve':p.add_argument('--arm',choices=RUN_ARMS,default='P4')
        if name in ('compare','batch','benchmark'):p.add_argument('--arms',nargs='+',choices=RUN_ARMS,default=['P0','P1','P2','P3','P4'])
        if name=='benchmark':p.add_argument('--repeats',type=int,default=1)
    grade=sub.add_parser('grade')
    for name in ('batch','labels','index','out'):grade.add_argument('--'+name,required=True)
    model_args(grade)
    grade.add_argument('--judge-profile', choices=JUDGE_PROFILES, default='semantic_strict')
    evaluation=sub.add_parser('summarize');evaluation.add_argument('--rows',required=True);evaluation.add_argument('--out',required=True)
    evaluation.add_argument('--bootstrap',type=int,default=0);evaluation.add_argument('--seed',type=int,default=0)
    cost=sub.add_parser('cost-report');cost.add_argument('--runs',required=True);cost.add_argument('--prices');cost.add_argument('--offline');cost.add_argument('--out',required=True)
    return cli


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if getattr(args,'out',None):require(not Path(args.out).exists(),'Output exists; choose a new path')
        cmd=args.command
        if cmd.startswith('matrix-'):
            from .matrix_cli import run
            return run(args)
        offline_meter=Meter()
        if cmd=='import-mragent':result=import_pack(read_json(args.pack),read_json(args.manifest))
        elif cmd=='canonicalize':result=canonicalize(read_json(args.snapshot))
        elif cmd=='build':
            require(not Path(args.out+'.metrics.json').exists(),'Build metrics output exists')
            cfg=Config.from_dict(read_json(args.config)) if args.config else Config()
            index=build_index(read_json(args.snapshot),cfg.top_k,read_json(args.fact_vectors) if args.fact_vectors else None)
            save_index(index,args.out)
            write_json(args.out+'.metrics.json',{'resources':offline_meter.result(),'index_hash':index.index_hash,
                'environment':environment_identity(),'scope':'input load + validation + index construction + serialization',
                'new_embedding_API_calls':0,'historical_embedding_cost':'external; not assumed free'})
            return 0
        elif cmd=='audit':
            index=load_index(args.index)
            print(json.dumps({'index_hash':index.index_hash,'representation':index.representation,'owner':index.snapshot['owner_id'],
                              'snapshot':index.snapshot['snapshot_id'],'report':index.report},ensure_ascii=False,indent=2));return 0
        elif cmd=='pin-mragent':result=pin_runtime(args.root)
        elif cmd=='freeze':result=freeze(read_json(args.spec))
        elif cmd=='preflight':
            protocol=verify_protocol(read_json(args.protocol));print(json.dumps({'verified':True,'protocol_hash':protocol['protocol_hash'],'questions':protocol['question_count']}));return 0
        elif cmd=='experiment':run_protocol(read_json(args.protocol),args.out);return 0
        elif cmd=='align-fact-vectors':
            index=load_index(args.index);data=read_json(args.vectors)
            require(set(data)=={'encoder','fact_ids','vectors'} and isinstance(data['fact_ids'],list)
                    and len(data['fact_ids'])==len(set(data['fact_ids']))==len(data['vectors']),'Invalid fact vector export')
            mapping=dict(zip(data['fact_ids'],data['vectors']));ids=[f['fact_id'] for f in index.snapshot['facts']]
            require(set(ids)==set(mapping),'Fact vector IDs must exactly match the snapshot')
            result={'snapshot_hash':index.snapshot['content_hash'],'encoder':data['encoder'],'fact_ids':ids,'vectors':[mapping[f] for f in ids]}
            build_index(index.snapshot,index.top_k,result)  # Validate every vector before exporting.
        elif cmd=='attest-dates':
            from .dates import attest_dates
            result=attest_dates(read_json(args.snapshot),read_json(args.conversation))
        elif cmd=='summarize':
            rows=read_json(args.rows);result=summarize(rows)
            if args.bootstrap:result['intervals']=paired_intervals(rows,args.bootstrap,args.seed)
        elif cmd=='cost-report':result=cost_report(read_json(args.runs),read_json(args.prices) if args.prices else None,read_json(args.offline) if args.offline else None)
        elif cmd=='grade':
            require(args.api_base and args.model,'Explicit judge API base/model required')
            judge=ChatBackend(args.api_base,args.model,args.key_env,max_output_tokens=args.max_output_tokens,temperature=0,
                              enable_thinking=args.enable_thinking,json_mode=args.json_mode)
            grade_batch(args.batch,read_json(args.labels),load_index(args.index),judge,args.out,judge_profile=args.judge_profile);return 0
        else:
            index=load_index(args.index);cfg=Config.from_dict(read_json(args.config)) if args.config else Config()
            tok=tokenizer_from({'kind':args.tokenizer,**({'path':args.tokenizer_file,'sha256':args.tokenizer_sha256} if args.tokenizer=='json' else {})})
            if cmd=='prepare-corpus':
                require(not Path(args.out+'.metrics.json').exists(),'Corpus metrics output exists')
                prepared=Corpus(index.snapshot,tok,cfg);prepared.save(args.out)
                write_json(args.out+'.metrics.json',{'resources':offline_meter.result(),'corpus_hash':prepared.identity,
                    'environment':environment_identity(),'scope':'input load + tokenizer init + atom/BM25 construction + serialization'})
                return 0
            corpus=Corpus.load(args.corpus,index.snapshot,tok,cfg) if args.corpus else None
            if cmd=='replay':
                require(corpus is not None,'Replay requires the pinned corpus cache')
                result=replay(read_json(args.run),index,corpus)
            else:
                arms=getattr(args,'arms',[getattr(args,'arm','P4')]);p5='P5' in arms
                require(not((args.llm_frontend or args.native_frontend) and args.frontend), 'Choose cached or live frontend')
                require(not(args.llm_frontend and args.native_frontend), 'Choose literal or native frontend')
                live=args.llm_frontend or args.native_frontend or args.generate or p5 or cmd=='explore-proxy'
                if live:require(args.api_base and args.model and args.tokenizer!='characters','Live calls require explicit model/base and actual tokenizer')
                if cmd=='compare':require(not args.generate,'compare is retrieval-only; use batch --generate')
                def backend(cls=ChatBackend):return cls(args.api_base,args.model,args.key_env,max_output_tokens=args.max_output_tokens,temperature=0,
                                                       enable_thinking=args.enable_thinking,json_mode=args.json_mode)
                runtime=None;exploration=None
                if p5:
                    require(args.native_root and args.native_manifest,'P5 needs pinned native checkout')
                    runtime=load_runtime(args.native_root,read_json(args.native_manifest));exploration=backend(ToolBackend)
                front=backend() if args.llm_frontend or args.native_frontend else None;answer=backend() if args.generate else None
                from .rerank import backend_from_spec
                rspec=read_json(args.reranker_spec) if args.reranker_spec else None
                rerank_backend=backend_from_spec(rspec) if rspec else None
                require(not (rspec and not rspec['cache_only'] and args.tokenizer=='characters'),
                        'Live reranking requires a real evidence tokenizer')
                extra={'frontend_mode':'mragent' if args.native_frontend else 'literal',
                       'reranker':rerank_backend,'rerank_cache_dir':rspec['cache_dir'] if rspec else None,
                       'rerank_cache_only':rspec['cache_only'] if rspec else False}
                require(not any(a.startswith('C_') for a in arms) or rerank_backend is not None, 'C_* arms require --reranker-spec')
                require(not (cmd=='explore-proxy' and args.native_frontend), 'Native frontend is available through retrieval/batch/benchmark, not legacy proxy')
                mentions=None
                if args.frontend:
                    require(cmd not in ('batch','benchmark'),'Batch mentions belong in each question row')
                    supplied=read_json(args.frontend)
                    require(supplied.get('question_hash')==digest(args.question) and supplied.get('snapshot_hash')==index.snapshot['content_hash'],'Cached frontend scope mismatch')
                    mentions=supplied['mentions']
                if cmd=='retrieve':
                    if p5:
                        from .controlled import run_controlled
                        from .pipeline import make_query
                        query=make_query(index,args.question,cfg,mentions,front,extra['frontend_mode'])
                        result=run_controlled(index,args.question,tok,runtime,exploration,config=cfg,query=query,answer_backend=answer,corpus=corpus)
                    else:result=retrieve(index,args.question,tok,config=cfg,arm=args.arm,mentions=mentions,frontend_backend=front,answer_backend=answer,corpus=corpus,**extra)
                elif cmd=='explore-proxy':result=explore(index,args.question,tok,backend(),config=cfg,mentions=mentions,frontend_backend=front,answer_backend=answer)
                elif cmd=='compare':result=compare(index,args.question,tok,config=cfg,arms=arms,mentions=mentions,frontend_backend=front,corpus=corpus,native_runtime=runtime,exploration_backend=exploration,**extra)
                elif cmd=='benchmark':
                    require(corpus is not None,'Benchmark requires prepared corpus; record cold loading separately')
                    result=benchmark(index,read_json(args.questions),tok,config=cfg,arms=arms,corpus=corpus,repeats=args.repeats,
                        frontend_backend=front,answer_backend=answer,native_runtime=runtime,exploration_backend=exploration,output_path=args.out,**extra)
                else:
                    from .batch import run_batch
                    run_batch(index,read_json(args.questions),tok,config=cfg,arms=arms,frontend_backend=front,answer_backend=answer,
                        output_dir=args.out,corpus=corpus,native_runtime=runtime,exploration_backend=exploration,**extra);return 0
        write_json(args.out,result);return 0
    except (ContractError,ValueError,OSError,KeyError,TypeError) as error:
        print('eitr: '+(str(error) if isinstance(error,ContractError) else type(error).__name__),file=sys.stderr);return 2
