"""Matrix v3 invariants, adversarial composition, replay and executable lifecycle."""
from copy import deepcopy
import json
import numpy as np
import pytest
from eicr.contracts import ContractError, digest, read_json, write_json
from eicr.packer import CharacterTokenizer
from eitr.evidence import Corpus
from eitr.matrix_embeddings import EmbeddingBackend, embed_record, validate_record, unit_rows
from eitr.matrix_index import build_index, embed_snapshot, load_index, balanced_weights, relation
from eitr.matrix_query import validate_plan, plan_record, verify_plan_record
from eitr.matrix_retrieval import MatrixConfig, candidates, match, select, ARMS
from eitr.matrix_pipeline import prepare_query, retrieve, replay
from eitr.matrix_experiment import freeze, verify_protocol, run_protocol
from eitr.accounting import cost_ledger
from eitr.evaluate import diagnose, grade_batch, summarize


class Embeddings:
    identity={'provider':'synthetic-test-only','model':'one-hot','revision':'v1','dimension':4,
              'query_prefix':'','document_prefix':''}
    def __init__(self): self.calls=[]
    def embed(self,texts,role):
        self.calls.append({'model':'one-hot','status':'ok','billing_kind':'embedding_tokens',
                           'usage':{'input_tokens':len(texts),'output_tokens':0}})
        if role=='document': return [[1 if j=={'Bought spices':0,'Used herbs':1,'Bought snacks':2,'Train late':3}[t] else 0 for j in range(4)] for t in texts]
        return [[.1,1.,.0,.0] if 'herb' in t.lower() else [.8,.2,.1,.1] for t in texts]


class Reranker:
    identity={'provider':'synthetic-test-only','model':'raw-herbs'}
    max_documents=500
    def __init__(self): self.calls=[]
    def score(self,q,documents):
        self.calls.append({'model':'raw-herbs','status':'ok','billing_kind':'rerank_tokens','usage':{'input_tokens':len(documents),'output_tokens':0}})
        return [.9 if 'For halloumi salad' in d.split('ADJACENT CONTEXT')[0] else .1 for d in documents]


@pytest.fixture
def built(snapshot):
    eb=Embeddings();record=embed_snapshot(snapshot,eb)
    return build_index(snapshot,record,graph_top_k=2,graph_shortlist=3,block=1),eb


def config(**kw):
    return MatrixConfig(**{'evidence_budget':1000,'context_each_side':0,**kw})


def test_exact_relation_against_dense_and_asymmetric():
    x=unit_rows([[1,0],[0,1],[1,0]],3,2);y=unit_rows([[1,0]],1,2)
    wx=np.array([.25,.5,.25]);wy=np.array([1.]);k=np.maximum(0,x@y.T)
    actual=relation(x,y,wx,wy,block=1)
    assert actual['average']==pytest.approx(wx@k@wy)
    assert actual['coverage_forward']==pytest.approx(.5)
    assert actual['coverage_reverse']==pytest.approx(1)
    assert actual['bridge']==1 and actual['witness_rows']==[0,0]
    assert actual==relation(x,y,wx,wy,block=128)


def test_source_balance_duplicate_and_multimodal():
    fs=[{'origins':['s1']},{'origins':['s1']},{'origins':['s2']}]
    assert balanced_weights(fs).tolist()==[.25,.25,.5]
    assert balanced_weights([]).size==0
    # Mean collapses orthogonal modes; row maximum preserves a strong witness.
    x=np.eye(2); assert max(x@x[0])==1 and np.mean(x,axis=0)@x[0]==.5


def test_index_roundtrip_and_witnesses(built,tmp_path):
    index,_=built;path=tmp_path/'index.json';index.save(path);loaded=load_index(path)
    assert loaded.index_hash==index.index_hash
    assert loaded.B.shape==(4,4) and loaded.B.nnz==4
    a,r,details=match(loaded,[[0,1,0,0]])
    pos=next(j for j,t in enumerate(index.tag_ids) if 1 in index.members[t])
    assert r[0,pos]==1 and details['witness_fact_ids'][0][pos]=='f2'
    body=read_json(path);body['embedding']['vectors'][0][0]=.2;write_json(path,body)
    with pytest.raises(ContractError):load_index(path)


@pytest.mark.parametrize('values',[[[0,0]],[[float('nan'),1]],[[1]],[[True,'bad']]])
def test_invalid_vectors(values):
    with pytest.raises((ContractError,ValueError)):unit_rows(values,1,2)


def test_embedding_cache_scope_order_and_accounting(tmp_path):
    b=Embeddings();texts=['Bought spices','Used herbs'];scope={'owner':'a'}
    first=embed_record(texts,'document',scope,b,cache_dir=tmp_path,batch_size=1)
    assert len(b.calls)==2
    second=embed_record(texts,'document',scope,b,cache_dir=tmp_path,cache_only=True)
    assert second==first and len(b.calls)==2
    with pytest.raises(ContractError):embed_record(texts[::-1],'document',scope,b,cache_dir=tmp_path,cache_only=True)
    with pytest.raises(ContractError):embed_record(texts,'document',{'owner':'b'},b,cache_dir=tmp_path,cache_only=True)
    req=deepcopy(first['request']);req['scope']={'owner':'b'}
    with pytest.raises(ContractError):validate_record(first,req)


def compound_plan(operator='intersection'):
    return {'requirements':[{'text':'What animal does Nate like?','anchors':['Nate','animal']},
                            {'text':'What animal does Joanna like?','anchors':['Joanna','animal']}],
            'operator':operator,'operator_anchor':'both' if operator=='intersection' else 'respectively',
            'subjects':['Nate','Joanna'],'constraints':[{'kind':'relation','quote':'like'}]}


def test_composition_and_grounding():
    q='What animal do both Nate and Joanna like?'
    p=validate_plan(compound_plan(),q);assert p['operator']=='intersection'
    other='What animal do Nate and Joanna respectively like?'
    assert validate_plan(compound_plan('group_by_subject'),other)['operator']=='group_by_subject'
    bad=compound_plan();bad['subjects'].append('Alice')
    with pytest.raises(ContractError):validate_plan(bad,q)
    assert plan_record(q,supplied=bad)['plan']['fallback_reason']=='invalid_plan'
    bad=compound_plan();bad['requirements'].pop()
    assert plan_record(q,supplied=bad)['plan']['status']=='fallback'
    bad=compound_plan();bad['constraints'][0]['quote']='2024-01-01'
    assert plan_record(q,supplied=bad)['plan']['status']=='fallback'
    record=plan_record(q,supplied=compound_plan());assert verify_plan_record(record,q)['status']=='parsed'
    with pytest.raises(ContractError):verify_plan_record(record,other)


@pytest.mark.parametrize('arm',ARMS)
def test_end_to_end_replay_budget_and_diagnosis(built,arm):
    index,b=built;cfg=config(evidence_budget=220,max_candidate_atoms=4,max_core_atoms=4)
    corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    prepared=prepare_query(index,'What herbs?',b,config=cfg)
    run=retrieve(index,'What herbs?',corpus,prepared,config=cfg,arm=arm,reranker=Reranker())
    assert run['evidence_pack']['used_tokens']<=220
    assert len(run['candidates']['atom_ids'])<=4
    assert run['calls']['query_embedding']
    assert replay(index,corpus,run)['verified']
    d=diagnose(run,['D1:2'],index);assert d['source_visibility_is_answer_sufficiency'] is False
    assert d['stage_source_recall']['candidate']==1
    broken=deepcopy(run);broken['candidates']['dropped_cores']+=1;broken['run_hash']=digest({k:v for k,v in broken.items() if k!='run_hash'})
    with pytest.raises(ContractError):replay(index,corpus,broken)


def test_no_graph_requires_explicit_rebuild(built):
    index,b=built;plain=build_index(index.snapshot,index.embedding);cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    with pytest.raises(ContractError): candidates(plain,corpus,[[1,0,0,0]],cfg,'matrix_graph_rerank')
    alien=Embeddings();alien.identity={**alien.identity,'revision':'v2'}
    with pytest.raises(ContractError):prepare_query(index,'herbs',alien)


def test_candidate_context_is_indivisible(built):
    index,b=built;cfg=config(context_each_side=1,max_candidate_atoms=2,max_core_atoms=2)
    corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    pool,_,_=candidates(index,corpus,[[0,1,0,0]],cfg,'matrix_rerank')
    assert len(pool['atom_ids'])<=2
    for action in pool['actions']: assert set(action['atoms'])==set(corpus.contexts[action['core']])
    cfg2=config(evidence_budget=1);corpus2=Corpus(index.snapshot,CharacterTokenizer(),cfg2)
    run=retrieve(index,'herbs',corpus2,prepare_query(index,'herbs',b),config=cfg2,reranker=Reranker())
    assert run['evidence_pack']['atom_ids']==[]


def test_answer_receives_plan_and_invalid_citation_preserved(built):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    class Answer:
        def __init__(self):self.calls=[]
        def chat_json(self,system,user):
            assert 'For intersection/both' in system and 'Advisory retrieval plan' in user
            assert 'Raw memory sources' in user
            self.calls.append({'status':'ok','usage':{'input_tokens':12,'output_tokens':3}})
            return {'answer_text':'herbs','resolution_status':'supported','citations':[{'source_id':'missing','start':0,'end':1}],'unsupported_premises':[]}
    run=retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=Reranker(),answer_backend=Answer())
    assert run['answer'] is None and run['raw_answer']['answer_text']=='herbs'
    assert run['answer_error']=='ContractError' and run['calls']['answer']
    ledger=cost_ledger(run,{'input_per_million':1,'output_per_million':2})
    assert ledger['independent_deployment']['usd'] is None
    assert set(ledger['by_billing_kind'])=={'chat_tokens','embedding_tokens','rerank_tokens'}


def test_empty_tags_direct_rescue(raw_memory):
    from eitr.native import canonicalize
    for f in raw_memory['evidence']: f['tags']=[]
    snapshot=canonicalize(raw_memory);b=Embeddings();index=build_index(snapshot,embed_snapshot(snapshot,b));cfg=config()
    corpus=Corpus(snapshot,CharacterTokenizer(),cfg)
    run=retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=Reranker())
    assert run['matching']['tag_ids']==[] and run['evidence_pack']['atom_ids']


def test_live_embedding_transport_reorders_indices(monkeypatch,tmp_path):
    from eitr import matrix_embeddings as mod
    spec=embedding_spec(tmp_path)
    backend=EmbeddingBackend(spec)
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return json.dumps({'data':[{'index':1,'embedding':[0,1,0,0]},
            {'index':0,'embedding':[1,0,0,0]}],'usage':{'prompt_tokens':4}}).encode()
    monkeypatch.setenv('MATRIX_TEST_KEY','secret-test-only')
    def open_response(request, **kwargs):
        payload=json.loads(request.data)
        assert payload['dimensions']==4 and payload['input']==['a','b']
        return Response()
    monkeypatch.setattr(mod,'_open_no_redirect',open_response)
    assert backend.embed(['a','b'],'document')==[[1.,0,0,0],[0,1.,0,0]]
    assert backend.calls[0]['billing_kind']=='embedding_tokens'
    assert 'secret-test-only' not in json.dumps(backend.calls)


def embedding_spec(tmp_path):
    return {'endpoint':'https://example.invalid/v1/embeddings','model':'test','revision':'test-v1','key_env':'MATRIX_TEST_KEY',
            'dimension':4,'query_prefix':'','document_prefix':'','batch_size':64,
            'cache_dir':str(tmp_path/'embeddings'),'cache_only':True}


def test_frozen_protocol_cli_batch_grade_and_replay(snapshot,tmp_path,monkeypatch):
    from eitr.cli import main
    from eitr import matrix_embeddings as mod
    spec_e=embedding_spec(tmp_path);backend=EmbeddingBackend(spec_e)
    fake=Embeddings();monkeypatch.setattr(mod.EmbeddingBackend,'embed',lambda self,texts,role:fake.embed(texts,role))
    record=embed_snapshot(snapshot,backend,cache_dir=spec_e['cache_dir'],cache_only=False)
    index=build_index(snapshot,record);ip=tmp_path/'index.json';index.save(ip)
    cfg=config();cp=tmp_path/'corpus.json';Corpus(snapshot,CharacterTokenizer(),cfg).save(cp)
    question={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'What herbs?'}
    qp=tmp_path/'questions.json';write_json(qp,[question]);tp=tmp_path/'tokenizer.json';write_json(tp,{'kind':'characters'})
    cfgp=tmp_path/'config.json';write_json(cfgp,cfg.to_dict())
    # Populate query caches explicitly, then prove experiment makes no provider calls.
    prepare_query(index,question['question'],backend,config=cfg,embedding_options={'cache_dir':spec_e['cache_dir'],'cache_only':False})
    spec={'config':cfg.to_dict(),'arms':['matrix_embedding'],'tokenizer':{'kind':'characters'},
          'assets':[{'index':str(ip),'corpus':str(cp)}],'questions':str(qp),'embedding':spec_e,
          'frontend':'whole_query','model':None,'plan_cache_dir':str(tmp_path/'plans'),'plan_cache_only':False,
          'reranker':None,'generate':False}
    protocol=freeze(spec);assert verify_protocol(protocol)==protocol
    sp=tmp_path/'spec.json';pp=tmp_path/'protocol.json';write_json(sp,spec)
    assert main(['matrix-freeze','--spec',str(sp),'--out',str(pp)])==0
    monkeypatch.setattr(mod.EmbeddingBackend,'embed',lambda *a,**k:pytest.fail('Unexpected live embedding'))
    out=tmp_path/'experiment';assert main(['matrix-experiment','--protocol',str(pp),'--out',str(out)])==0
    journal=read_json(out/'manifest.json');assert journal['status']=='complete'
    batch=out/journal['completed'][0]['batch'];manifest=read_json(batch/'manifest.json')
    path=batch/manifest['completed'][0]['file'];case=read_json(path);run=case['rows'][0]
    rp=tmp_path/'run.json';write_json(rp,run)
    assert main(['matrix-replay','--index',str(ip),'--corpus',str(cp),'--tokenizer-spec',str(tp),
                 '--run',str(rp),'--out',str(tmp_path/'replay.json')])==0
    # Grade uses a separate process/data contract; fabricate a valid test answer only here.
    span=run['evidence_pack']['spans'][0]
    run['answer']={'answer_text':'herbs','resolution_status':'supported','citations':[{k:span[k] for k in ('source_id','start','end')}],'unsupported_premises':[]}
    run['run_hash']=digest({k:v for k,v in run.items() if k!='run_hash'})
    write_json(path,case);manifest['completed'][0]['content_hash']=digest(case);write_json(batch/'manifest.json',manifest)
    class Judge:
        calls=[]
        def chat_json(self,system,user): return {'correct':True,'reason':'test only'}
    labels=[{'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],
             'references':['herbs'],'source_ids':['D1:2']}]
    rows=grade_batch(batch,labels,index,Judge(),tmp_path/'graded');assert summarize(rows)['arms']['matrix_embedding']['accuracy']==1
    leaked=deepcopy(question);leaked['answer']='herbs';write_json(qp,[leaked])
    with pytest.raises(ContractError):freeze(spec)
    with pytest.raises(ContractError):verify_protocol(protocol)


def test_matrix_routes_non_top_fact_and_graph_adds_query_conditioned_tag(raw_memory):
    from eitr.native import canonicalize
    # f1 is query-best; its tag also exposes f2 outside the direct top-1.
    raw_memory['evidence'][1]['tags']=['shopping']
    snap=canonicalize(raw_memory)
    b=Embeddings();rec=embed_snapshot(snap,b)
    rec['vectors']=[[1,0,0,0],[.7,.7,0,0],[.7,0,.7,0],[0,0,0,1]]
    rec['record_hash']=digest({k:v for k,v in rec.items() if k!='record_hash'})
    index=build_index(snap,rec,graph_top_k=2,graph_shortlist=3)
    cfg=config(direct_facts=1,tags_per_requirement=1,facts_per_tag=2,max_core_atoms=4,max_candidate_atoms=4)
    corpus=Corpus(snap,CharacterTokenizer(),cfg)
    pool,_,_=candidates(index,corpus,[[1,0,0,0]],cfg,'matrix_rerank')
    assert 'D1:2' in pool['source_stages']['candidate']
    assert 'D1:10' not in pool['source_stages']['candidate']
    graph,_,_=candidates(index,corpus,[[1,0,0,0]],cfg,'matrix_graph_rerank')
    assert 'D1:10' in graph['source_stages']['candidate']
    assert graph['graph_routes'][0]
    direct,_,matching=candidates(index,corpus,[[1,0,0,0]],cfg,'direct_rerank')
    assert matching['tag_ids']==[] and 'D1:10' in direct['source_stages']['candidate']


def test_long_source_late_atom_not_lost(raw_memory):
    from eitr.native import canonicalize
    raw_memory['sources'][1]['text']='First sentence. '*12+'End: the answer is herbs.'
    snap=canonicalize(raw_memory);b=Embeddings();index=build_index(snap,embed_snapshot(snap,b))
    cfg=config(atom_tokens=32,max_candidate_atoms=40,max_core_atoms=40)
    corpus=Corpus(snap,CharacterTokenizer(),cfg)
    pool,_,_=candidates(index,corpus,[[0,1,0,0]],cfg,'matrix_rerank')
    tail=corpus.source_atoms['D1:2'][-1]
    assert tail in pool['atom_ids'] and 'herbs' in corpus.atoms[tail]['text']


def test_plan_transport_fallback_and_failed_calls():
    from eicr.provider import ProviderTransportError
    class Bad:
        identity={'model':'broken'}
        calls=[]
        def chat_json(self,*args):
            self.calls.append({'status':'error','usage':{}})
            raise ProviderTransportError('test transport')
    rec=plan_record('What animal?',backend=Bad())
    assert rec['plan']['status']=='fallback' and rec['calls'][0]['status']=='error'


def test_reranker_must_not_split_pool(built):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    rb=Reranker();rb.max_documents=2
    with pytest.raises(ContractError):retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=rb)
    assert not rb.calls


def test_valid_answer_and_paired_subjects_reach_model(built):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    q='What animal do both Nate and Joanna like?'
    prepared=prepare_query(index,q,b,supplied_plan=compound_plan(),config=cfg)
    reference=retrieve(index,q,corpus,prepared,config=cfg,reranker=Reranker())
    span=reference['evidence_pack']['spans'][0]
    class Answer:
        calls=[]
        def chat_json(self,system,user):
            assert 'intersection' in user and 'Nate' in user and 'Joanna' in user
            return {'answer_text':'Synthetic answer, not a quality result','citations':[{k:span[k] for k in ('source_id','start','end')}],
                    'unsupported_premises':[],'resolution_status':'supported'}
    result=retrieve(index,q,corpus,prepared,config=cfg,reranker=Reranker(),answer_backend=Answer())
    assert result['manifest']['status']=='answered' and result['answer_error'] is None
    assert len(result['evidence_pack']['requirement_max_similarity'])==2


def test_generation_failure_is_not_dropped_from_grading(built,tmp_path):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    from eicr.provider import ProviderTransportError
    class FailedAnswer:
        calls=[]
        def chat_json(self,*args):
            self.calls.append({'status':'error','usage':{}})
            raise ProviderTransportError('test')
    run=retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=Reranker(),answer_backend=FailedAnswer())
    case={'question_id':'q','question':'herbs','index_hash':index.index_hash,'rows':[run]}
    root=tmp_path/'batch';filename=digest('q')[:24]+'.json';write_json(root/filename,case)
    write_json(root/'manifest.json',{'status':'complete','index_hash':index.index_hash,
        'completed':[{'question_id':'q','file':filename,'content_hash':digest(case)}]})
    labels=[{'question_id':'q','owner_id':index.snapshot['owner_id'],'snapshot_id':index.snapshot['snapshot_id'],
             'references':['herbs'],'source_ids':['D1:2']}]
    class NoJudge:
        calls=[]
        def chat_json(self,*args):pytest.fail('Generation failure must not be sent to judge')
    rows=grade_batch(root,labels,index,NoJudge(),tmp_path/'grades')
    assert len(rows)==1 and rows[0]['correct'] is False and rows[0]['answer_contract_valid'] is False
    assert rows[0]['online_tokens'] is None


def test_graph_expansion_can_discover_zero_query_similarity_tag(raw_memory):
    from eitr.native import canonicalize
    # Two modes in the seed tag bridge to a target orthogonal to the query.
    raw_memory['evidence'][1]['tags']=['shopping']
    snap=canonicalize(raw_memory);b=Embeddings();record=embed_snapshot(snap,b)
    record['vectors']=[[1,0,0,0],[.6,.8,0,0],[0,1,0,0],[0,0,0,1]]
    record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'})
    index=build_index(snap,record,graph_top_k=2,graph_shortlist=3)
    cfg=config(direct_facts=1,tags_per_requirement=1,graph_tags_per_requirement=1,max_core_atoms=4,max_candidate_atoms=4)
    corpus=Corpus(snap,CharacterTokenizer(),cfg)
    plain,_,_=candidates(index,corpus,[[1,0,0,0]],cfg,'matrix_rerank')
    pool,_,_=candidates(index,corpus,[[1,0,0,0]],cfg,'matrix_graph_rerank')
    assert 'D1:10' not in plain['source_stages']['candidate']
    assert 'D1:10' in pool['source_stages']['candidate']
    assert 'D2:1' not in pool['source_stages']['candidate']


def test_embedding_backend_options_are_honored_by_prepare_query(built,tmp_path):
    index,_=built
    class Backend:
        identity=index.embedding['request']['backend']
        spec={'cache_dir':str(tmp_path/'missing-cache'),'cache_only':True,'batch_size':1}
        calls=[]
        def embed(self,*args): pytest.fail('cache_only backend must not make a live call')
    with pytest.raises(ContractError,match='Required embedding cache missing'):
        prepare_query(index,'herbs',Backend())


@pytest.fixture
def frozen_assets(snapshot,tmp_path,monkeypatch):
    from eitr import matrix_embeddings as mod
    es=embedding_spec(tmp_path);backend=EmbeddingBackend(es);fake=Embeddings()
    monkeypatch.setattr(mod.EmbeddingBackend,'embed',lambda self,ts,role:fake.embed(ts,role))
    rec=embed_snapshot(snapshot,backend,cache_only=False)
    index=build_index(snapshot,rec);ip=tmp_path/'index.json';index.save(ip)
    cfg=config();cp=tmp_path/'corpus.json';Corpus(snapshot,CharacterTokenizer(),cfg).save(cp)
    question={'question_id':'q','question':'What herbs?','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id']}
    qp=tmp_path/'questions.json';write_json(qp,[question])
    prepared=prepare_query(index,question['question'],backend,config=cfg,embedding_options={'cache_only':False})
    spec={'config':cfg.to_dict(),'arms':['matrix_embedding'],'tokenizer':{'kind':'characters'},
          'assets':[{'index':str(ip),'corpus':str(cp)}],'questions':str(qp),'embedding':es,
          'frontend':'whole_query','model':None,'plan_cache_dir':str(tmp_path/'plans'),'plan_cache_only':True,
          'reranker':None,'generate':False}
    return spec,index,prepared


def test_preflight_rejects_missing_and_changed_embedding_cache(frozen_assets,tmp_path):
    spec,index,prepared=frozen_assets
    protocol=freeze(spec)
    path=tmp_path/'embeddings'/(digest(prepared['embedding']['request'])+'.json')
    assert str(path) in protocol['files']
    original=path.read_bytes();path.unlink()
    with pytest.raises(ContractError,match='embedding cache missing'):freeze(spec)
    path.write_bytes(original)
    record=read_json(path);record['vectors'][0]=[0,0,1,0]
    record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'});write_json(path,record)
    with pytest.raises(ContractError,match='cache changed'):verify_protocol(protocol)


def test_preflight_missing_rerank_cache_and_first_arm_failure_journal(frozen_assets,tmp_path,monkeypatch):
    from eitr import matrix_experiment as experiment
    spec,index,prepared=frozen_assets
    spec['arms']=['matrix_rerank']
    spec['reranker']={'provider':'dashscope','endpoint':'https://example.invalid/rerank','model':'test','key_env':'TEST_KEY',
                     'max_documents':500,'cache_dir':str(tmp_path/'rerank'),'cache_only':True}
    with pytest.raises(ContractError,match='rerank cache missing'):freeze(spec)
    spec['arms']=['matrix_embedding'];spec['reranker']=None
    protocol=freeze(spec)
    def fail(*args,**kwargs):raise ContractError('test first-arm error')
    monkeypatch.setattr(experiment,'retrieve',fail)
    out=tmp_path/'failed-run'
    with pytest.raises(ContractError,match='first-arm'):run_protocol(protocol,out)
    journal=read_json(out/'manifest.json')
    assert journal['status']=='failed' and journal['failure']['question_id']=='q'
    assert journal['failure']['arm']=='matrix_embedding' and journal['failure']['prepared_query']
    case=read_json(out/digest([index.snapshot['owner_id'],index.snapshot['snapshot_id']])[:24]/(digest('q')[:24]+'.json'))
    assert case['prepared_query']==prepared and case['rows']==[]
    # Runtime's deterministic plan cache is newly created, not a stale protocol.
    assert verify_protocol(protocol)==protocol


def test_preflight_plan_cache_missing(frozen_assets,tmp_path):
    spec,_,_=frozen_assets
    spec['frontend']='llm'
    spec['model']={'api_base':'https://example.invalid/v1','model':'test','key_env':'TEST_KEY','max_output_tokens':128}
    with pytest.raises(ContractError,match='plan cache missing'):freeze(spec)


def test_empty_pool_cache_only_does_not_call_provider(built,tmp_path):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    pool,_,_=candidates(index,corpus,[[-1,-1,-1,-1]],cfg,'matrix_rerank')
    assert not pool['atom_ids']
    from eitr.rerank import rerank_pool
    rb=Reranker();result=rerank_pool('nothing',corpus,pool,rb,cache_dir=tmp_path,cache_only=True)
    assert result['scores']=={} and result['calls']==[] and rb.calls==[]


def test_plan_record_must_match_raw_response():
    q='What animal do both Nate and Joanna like?';record=plan_record(q,supplied=compound_plan())
    record['plan']['requirements'][0]['text']='Nate likes imagined animals'
    record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'})
    with pytest.raises(ContractError,match='raw frontend'):verify_plan_record(record,q)


def test_mixed_cost_summary_and_failure_counters(built):
    from eitr.accounting import cost_report
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    run=retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=Reranker())
    result=cost_report([run])['arms']['matrix_rerank']
    assert result['by_billing_kind']['embedding_tokens']['calls']==1
    assert result['by_billing_kind']['rerank_tokens']['calls']==1
    assert result['frontend_error_runs']==0
    assert result['mean_end_to_end_seconds'] is None  # No fake independent timing.


@pytest.mark.parametrize('fault',['missing_arm','wrong_question','empty_rows'])
def test_grading_rejects_incomplete_or_mismatched_runs_before_judge(built,tmp_path,fault):
    index,b=built;cfg=config();corpus=Corpus(index.snapshot,CharacterTokenizer(),cfg)
    run=retrieve(index,'herbs',corpus,prepare_query(index,'herbs',b),config=cfg,reranker=Reranker())
    span=run['evidence_pack']['spans'][0]
    run['answer']={'answer_text':'herbs','resolution_status':'supported',
                  'citations':[{k:span[k] for k in ('source_id','start','end')}],'unsupported_premises':[]}
    if fault=='wrong_question':run['question']='different question'
    run['run_hash']=digest({k:v for k,v in run.items() if k!='run_hash'})
    case={'question_id':'q','question':'herbs','index_hash':index.index_hash,'rows':[] if fault=='empty_rows' else [run]}
    root=tmp_path/'batch';filename=digest('q')[:24]+'.json';write_json(root/filename,case)
    write_json(root/'manifest.json',{'schema':'eitr-matrix-batch-1','status':'complete','index_hash':index.index_hash,
        'arms':['matrix_rerank','direct_rerank'] if fault=='missing_arm' else ['matrix_rerank'],
        'completed':[{'question_id':'q','file':filename,'content_hash':digest(case)}]})
    labels=[{'question_id':'q','owner_id':index.snapshot['owner_id'],'snapshot_id':index.snapshot['snapshot_id'],
             'references':['herbs'],'source_ids':[]}]
    class NoJudge:
        calls=[]
        def chat_json(self,*args):pytest.fail('Invalid batch must fail before judge')
    with pytest.raises(ContractError):grade_batch(root,labels,index,NoJudge(),tmp_path/'grades')
    assert not (tmp_path/'grades').exists()


def test_supplied_plan_has_no_mutable_alias():
    q='What animal do both Nate and Joanna like?';supplied=compound_plan()
    record=plan_record(q,supplied=supplied)
    supplied['requirements'][0]['text']='external mutation'
    assert verify_plan_record(record,q)['requirements'][0]['text']=='What animal does Nate like?'
    record['plan']['requirements'][0]['text']='plan mutation'
    assert record['raw_response']['requirements'][0]['text']=='What animal does Nate like?'


def test_random_blocked_relations_against_dense():
    rng=np.random.default_rng(281)
    for n,m,d in [(1,7,4),(9,2,3),(13,17,8)]:
        x=unit_rows(rng.normal(size=(n,d)),n,d);y=unit_rows(rng.normal(size=(m,d)),m,d)
        wx=rng.random(n);wx/=wx.sum();wy=rng.random(m);wy/=wy.sum()
        k=np.clip(x@y.T,0,1)
        for block in (1,3,32):
            got=relation(x,y,wx,wy,block)
            assert got['average']==pytest.approx(wx@k@wy)
            assert got['coverage_forward']==pytest.approx(wx@k.max(axis=1))
            assert got['coverage_reverse']==pytest.approx(wy@k.max(axis=0))
            assert got['bridge']==pytest.approx(k.max())
            assert k[tuple(got['witness_rows'])]==pytest.approx(k.max())
