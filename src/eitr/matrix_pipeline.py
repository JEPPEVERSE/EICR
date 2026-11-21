"""Complete matrix retrieval -> mature reranker -> raw-source answering chain."""
from time import perf_counter
from eicr.contracts import digest, require, ContractError
from eicr.pipeline import answer_from_sources
from eicr.provider import ProviderTransportError
from .accounting import Meter, cost_ledger
from .pipeline import runtime_source_hash
from .matrix_embeddings import embed_record, validate_record
from .matrix_query import plan_record, verify_plan_record, query_texts, PlannedAnswer, ANSWER_GUIDANCE
from .matrix_retrieval import MatrixConfig, candidates, select, ARMS
from .rerank import rerank_pool


def query_embedding_request(index,question,plan):
    texts=query_texts(question,plan)
    scope={'index_hash':index.index_hash,'question_hash':digest(question),'plan_hash':digest(plan)}
    return texts,scope,{'schema':'eitr-matrix-embedding-request-1','backend':index.embedding['request']['backend'],
                      'role':'query','scope':scope,'texts_hash':digest(texts),'count':len(texts)}


def prepare_query(index,question,embedding_backend, *, frontend_backend=None,supplied_plan=None,
                  plan_cache_dir=None,plan_cache_only=False,embedding_options=None,config=MatrixConfig(),progress=None):
    require(embedding_backend.identity==index.embedding['request']['backend'],'Document/query encoder identity mismatch')
    plan=plan_record(question,backend=frontend_backend,supplied=supplied_plan,cache_dir=plan_cache_dir,
                     cache_only=plan_cache_only,max_requirements=config.max_requirements)
    if progress is not None: progress['plan']=plan
    texts,scope,_=query_embedding_request(index,question,plan['plan'])
    emb=embed_record(texts,'query',scope,embedding_backend,**(embedding_options or {}))
    if progress is not None: progress['embedding']=emb
    return {'plan':plan,'embedding':emb}


def retrieve(index,question,corpus,prepared, *, config=MatrixConfig(),arm='matrix_rerank',reranker=None,
             rerank_cache_dir=None,rerank_cache_only=False,answer_backend=None,semantic_record=None):
    meter=Meter(); tick=perf_counter()
    require(arm in ARMS,'Unknown matrix arm')
    plan=verify_plan_record(prepared['plan'],question)
    require(len(plan['requirements'])<=config.max_requirements,'Plan exceeds frozen requirement cap')
    _,_,request=query_embedding_request(index,question,plan)
    vectors=validate_record(prepared['embedding'],request)
    pool,scores,matching=candidates(index,corpus,vectors,config,arm)
    routed=perf_counter()
    semantic=None
    if arm!='matrix_embedding':
        if semantic_record is not None:
            # Replay backend returns the pinned outputs only after exact request validation.
            require(semantic_record.get('record_hash')==digest({k:v for k,v in semantic_record.items()
                    if k not in ('record_hash','cache_reused','cost_note')}),'Semantic replay checksum mismatch')
            class PinnedReranker:
                identity=semantic_record['request']['backend']
                calls=[]
                def score(self,question,documents):
                    require(digest(documents)==semantic_record['request']['documents_hash'],'Semantic replay documents changed')
                    self.calls=list(semantic_record['calls'])
                    return [semantic_record['scores'][a] for a in sorted(pool['atom_ids'])]
            semantic=rerank_pool(question,corpus,pool,PinnedReranker())
            require(semantic['request']==semantic_record['request'] and semantic['record_hash']==semantic_record['record_hash'],
                    'Semantic replay scope changed')
        else:
            require(reranker is not None and config.max_candidate_atoms<=reranker.max_documents,
                    'Reranker missing or candidate cap exceeds single-request limit')
            semantic=rerank_pool(question,corpus,pool,reranker,cache_dir=rerank_cache_dir,cache_only=rerank_cache_only)
        relevance=semantic['scores']
    else: relevance=scores[0]
    ranked=perf_counter()
    packet=select(corpus,pool,relevance,scores[1:],config)
    packed=perf_counter(); answer=None; raw_answer=None; answer_error=None; answer_calls=[]
    if answer_backend:
        start=len(answer_backend.calls); wrapper=PlannedAnswer(answer_backend,plan)
        try: answer=answer_from_sources(question,packet,wrapper)
        except (ContractError,ProviderTransportError) as error: answer_error=type(error).__name__
        finally: raw_answer=wrapper.raw;answer_calls=list(answer_backend.calls[start:])
    calls={'frontend':prepared['plan']['calls'],'query_embedding':prepared['embedding']['calls'],
           'rerank':semantic['calls'] if semantic else [],'answer':answer_calls}
    run={'schema':'eitr-matrix-run-1','arm':arm,'question':question,'index_hash':index.index_hash,
         'snapshot_hash':index.snapshot['content_hash'],'prepared_query':prepared,'query':{'plan':plan},
         'matching':matching,'candidates':pool,'semantic_rerank':semantic,'relevance':relevance,
         'evidence_pack':packet,'answer':answer,'raw_answer':raw_answer,'answer_error':answer_error,'calls':calls,
         'timing_seconds':{'matrix_routing':routed-tick,'rerank':ranked-routed,'packing':packed-ranked,
                           'retrieval_total':packed-tick,'answer':perf_counter()-packed},'resources':meter.result(),
         'manifest':{'config':config.to_dict(),'runtime_source_hash':runtime_source_hash(),
                     'tokenizer':corpus.tokenizer.identity,'answer_guidance_hash':digest(ANSWER_GUIDANCE),
                     'status':'answer_failed' if answer_error else ('answered' if answer else 'evidence_only'),
                     'timing_scope':'prepared query excluded; batch logs preparation separately',
                     'tag_graph_role':'auxiliary ablation','hard_metadata_filter':False}}
    run['cost']=cost_ledger(run)
    run['run_hash']=digest(run)
    return run


def replay(index,corpus,run):
    require(run.get('schema') in ('eitr-matrix-run-1','eitr-matrix-failure-1') and run.get('run_hash')==digest(
            {k:v for k,v in run.items() if k!='run_hash'}),'Run hash/schema mismatch')
    require(run['index_hash']==index.index_hash and run['manifest']['runtime_source_hash']==runtime_source_hash(),
            'Replay index/code mismatch')
    if run['schema']=='eitr-matrix-failure-1':
        return {'schema':'eitr-matrix-replay-1','verified':True,'run_hash':run['run_hash'],
                'execution_succeeded':False,'reconstructed':[], 'failure_stage':run['failure_stage'],
                'note':'Only failure receipt integrity checked; failed retrieval cannot be reconstructed.'}
    result=retrieve(index,run['question'],corpus,run['prepared_query'],config=MatrixConfig.from_dict(run['manifest']['config']),
                    arm=run['arm'],semantic_record=run['semantic_rerank'])
    for key in ('matching','candidates','relevance','evidence_pack'):
        require(result[key]==run[key],'Replay changed '+key)
    return {'schema':'eitr-matrix-replay-1','verified':True,'run_hash':run['run_hash'],
            'reconstructed':['query_fact_scores','query_tag_scores','witnesses','candidate_pool','selection','raw_packet'],
            'not_rerun':['provider embeddings','provider reranking','answer generation'],
            'packet_hash':result['evidence_pack']['text_hash']}
