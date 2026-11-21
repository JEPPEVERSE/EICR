"""Frozen multi-owner protocols and label-free matrix experiments."""
import hashlib
from pathlib import Path
from time import perf_counter
from eicr.contracts import digest, require, read_json, write_json
from .accounting import environment_identity
from .experiment import tokenizer_from, model_backend
from .evidence import Corpus
from .matrix_embeddings import EmbeddingBackend, embedding_options
from .matrix_index import load_index
from .matrix_retrieval import MatrixConfig, ARMS
from .matrix_pipeline import prepare_query, retrieve
from .pipeline import runtime_source_hash
from .rerank import backend_from_spec


def freeze(spec):
    require(isinstance(spec,dict) and set(spec)=={'config','arms','tokenizer','assets','questions','embedding',
            'frontend','model','plan_cache_dir','plan_cache_only','reranker','generate'}, 'Pin complete matrix protocol')
    cfg=MatrixConfig.from_dict(spec['config']); arms=spec['arms']
    require(isinstance(arms,list) and arms and len(set(arms))==len(arms) and set(arms)<=set(ARMS),'Invalid matrix arms')
    require(spec['frontend'] in ('llm','supplied','whole_query') and type(spec['generate']) is bool and
            type(spec['plan_cache_only']) is bool and isinstance(spec['plan_cache_dir'],str) and
            Path(spec['plan_cache_dir']).is_absolute(),'Invalid matrix frontend')
    embedding=EmbeddingBackend(spec['embedding']); tok=tokenizer_from(spec['tokenizer'])
    if spec['frontend']=='llm' or spec['generate']: model_backend(spec['model'])
    if any(a!='matrix_embedding' for a in arms):
        rb=backend_from_spec(spec['reranker'])
        require(cfg.max_candidate_atoms<=rb.max_documents,'Candidate cap exceeds reranker limit')
    live=(not spec['embedding']['cache_only'] or spec['generate'] or
          (spec['frontend']=='llm' and not spec['plan_cache_only']) or
          (any(a!='matrix_embedding' for a in arms) and not spec['reranker']['cache_only']))
    require(not live or spec['tokenizer']['kind']!='characters','Live calls require a real evidence tokenizer')
    files={}; indexes={}; reports=[]
    def pin(path):
        require(isinstance(path,str) and Path(path).is_absolute(),'Frozen paths must be absolute')
        p=Path(path).resolve(); files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    require(isinstance(spec['assets'],list) and spec['assets'],'No matrix assets')
    for asset in spec['assets']:
        require(isinstance(asset,dict) and set(asset)=={'index','corpus'},'Invalid matrix asset')
        index=load_index(asset['index']); key=(index.snapshot['owner_id'],index.snapshot['snapshot_id'])
        require(key not in indexes and index.embedding['request']['backend']==embedding.identity,'Duplicate scope/encoder mismatch')
        if 'matrix_graph_rerank' in arms: require(index.graph_config['top_k']>0 and cfg.graph_hops>0,'Graph ablation requires graph')
        corpus=Corpus.load(asset['corpus'],index.snapshot,tok,cfg)
        indexes[key]=index;pin(asset['index']);pin(asset['corpus'])
        reports.append({'owner_id':key[0],'snapshot_id':key[1],'index_hash':index.index_hash,'corpus_hash':corpus.identity,
                        'facts':len(index.facts),'tags':len(index.tag_ids),'offline_embedding_calls':index.embedding['calls']})
    questions=read_json(spec['questions']);pin(spec['questions']);seen=set();scopes=set()
    require(isinstance(questions,list) and questions,'No matrix questions')
    for q in questions:
        require(isinstance(q,dict) and set(q)<={'question_id','owner_id','snapshot_id','question','plan'},'Labels/unknown fields forbidden in retrieval inputs')
        key=(q.get('owner_id'),q.get('snapshot_id'))
        require(key in indexes and isinstance(q.get('question_id'),str) and q['question_id'] and
                (*key,q['question_id']) not in seen and isinstance(q.get('question'),str) and q['question'].strip(),'Invalid question identity/scope')
        require(('plan' in q)==(spec['frontend']=='supplied'),'Supplied plan mode mismatch')
        if 'plan' in q:
            from .matrix_query import validate_plan
            validate_plan(q['plan'],q['question'],cfg.max_requirements)
        seen.add((*key,q['question_id']));scopes.add(key)
    require(scopes==set(indexes),'Each asset must have questions')
    if spec['tokenizer']['kind']=='json': pin(spec['tokenizer']['path'])
    body={'schema':'eitr-matrix-protocol-1','spec':spec,'files':files,'runtime_source_hash':runtime_source_hash(),
          'environment':environment_identity(),'asset_reports':reports,'questions_hash':digest(questions),
          'question_count':len(questions),'claim':'engineering validation; no quality or cost savings assumed'}
    return {**body,'protocol_hash':digest(body)}


def verify_protocol(protocol):
    require(protocol.get('schema')=='eitr-matrix-protocol-1' and protocol.get('protocol_hash')==digest(
        {k:v for k,v in protocol.items() if k!='protocol_hash'}),'Matrix protocol hash mismatch')
    require(freeze(protocol['spec'])==protocol,'Matrix protocol assets/code/environment changed')
    return protocol


def run_protocol(protocol,output_dir):
    verify_protocol(protocol);spec=protocol['spec'];out=Path(output_dir)
    require(not out.exists(),'Matrix experiment output exists')
    out.mkdir(parents=True);write_json(out/'protocol.json',protocol)
    cfg=MatrixConfig.from_dict(spec['config']);tok=tokenizer_from(spec['tokenizer'])
    eb=EmbeddingBackend(spec['embedding']);fb=model_backend(spec['model']) if spec['frontend']=='llm' else None
    ab=model_backend(spec['model']) if spec['generate'] else None
    rb=backend_from_spec(spec['reranker']) if any(a!='matrix_embedding' for a in spec['arms']) else None
    journal={'schema':'eitr-matrix-experiment-1','protocol_hash':protocol['protocol_hash'],'status':'running','completed':[]}
    write_json(out/'manifest.json',journal)
    questions=read_json(spec['questions']);batch=None;root=None
    try:
        for asset in spec['assets']:
            index=load_index(asset['index']);corpus=Corpus.load(asset['corpus'],index.snapshot,tok,cfg)
            owner,snapshot=index.snapshot['owner_id'],index.snapshot['snapshot_id']
            name=digest([owner,snapshot])[:24];root=out/name;root.mkdir()
            batch={'schema':'eitr-matrix-batch-1','status':'running','index_hash':index.index_hash,'completed':[],'arms':spec['arms']}
            write_json(root/'manifest.json',batch)
            for q in (q for q in questions if (q['owner_id'],q['snapshot_id'])==(owner,snapshot)):
                tick=perf_counter()
                prepared=prepare_query(index,q['question'],eb,frontend_backend=fb,supplied_plan=q.get('plan'),
                    plan_cache_dir=spec['plan_cache_dir'],plan_cache_only=spec['plan_cache_only'],
                    embedding_options=embedding_options(eb),config=cfg)
                prep_seconds=perf_counter()-tick;rows=[]
                case={'question_id':q['question_id'],'question':q['question'],'index_hash':index.index_hash,'rows':rows,
                      'preparation_seconds':prep_seconds,'prepared_query':prepared,
                      'timing_note':'Shared preparation measured once; each arm retains full deployment call accounting.'}
                filename=digest(q['question_id'])[:24]+'.json'
                for arm in spec['arms']:
                    row=retrieve(index,q['question'],corpus,prepared,config=cfg,arm=arm,reranker=rb,
                        rerank_cache_dir=spec['reranker']['cache_dir'] if rb else None,
                        rerank_cache_only=spec['reranker']['cache_only'] if rb else False,answer_backend=ab)
                    rows.append(row);write_json(root/filename,case)
                batch['completed'].append({'question_id':q['question_id'],'file':filename,'content_hash':digest(case)})
                write_json(root/'manifest.json',batch)
            batch['status']='complete';write_json(root/'manifest.json',batch)
            journal['completed'].append({'owner_id':owner,'snapshot_id':snapshot,'batch':name})
            write_json(out/'manifest.json',journal)
        journal['status']='complete'
    except Exception as error:
        failure={'error_type':type(error).__name__,'calls':{name:list(b.calls) if b else []
                 for name,b in [('embedding',eb),('frontend',fb),('rerank',rb),('answer',ab)]}}
        journal.update(status='failed',failure=failure)
        if batch is not None:
            batch.update(status='failed',failure=failure);write_json(root/'manifest.json',batch)
        raise
    finally: write_json(out/'manifest.json',journal)
    return journal
