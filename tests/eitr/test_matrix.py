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
    monkeypatch.setattr(mod,'_open_no_redirect',lambda *a,**k:Response())
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
    record=embed_snapshot(snapshot,backend,cache_dir=spec_e['cache_dir'])
    index=build_index(snapshot,record);ip=tmp_path/'index.json';index.save(ip)
    cfg=config();cp=tmp_path/'corpus.json';Corpus(snapshot,CharacterTokenizer(),cfg).save(cp)
    question={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'What herbs?'}
    qp=tmp_path/'questions.json';write_json(qp,[question]);tp=tmp_path/'tokenizer.json';write_json(tp,{'kind':'characters'})
    cfgp=tmp_path/'config.json';write_json(cfgp,cfg.to_dict())
    # Populate query caches explicitly, then prove experiment makes no provider calls.
    prepare_query(index,question['question'],backend,config=cfg,embedding_options={'cache_dir':spec_e['cache_dir']})
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
