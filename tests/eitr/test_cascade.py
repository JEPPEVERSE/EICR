"""Fixed-pool reranking, provider contracts, cache isolation, and MMR scope."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from eicr.contracts import ContractError, digest, read_json, write_json
from eicr.packer import CharacterTokenizer
from eitr.candidates import candidate_pool, enumerate_graph
from eitr.cascade import MMRScheduler, select_cascade
from eitr.config import Config
from eitr.evidence import Corpus
from eitr.index import build_index
from eitr.native import canonicalize
from eitr.pipeline import compare, retrieve
from eitr.query import map_groups
from eitr.rerank import DashScopeReranker, rerank_documents, rerank_pool


class LocalReranker:
    identity={'provider':'test-only','model':'deterministic-focus-v1'}
    def __init__(self, preferred=None): self.calls=[]; self.preferred=preferred
    def score(self, question, documents):
        self.calls.append({'status':'ok','usage':{'input_tokens':19,'output_tokens':0,'total_tokens':19}})
        return [1. if self.preferred and self.preferred in doc.split('ADJACENT CONTEXT')[0]
                else .01 for doc in documents]


def setup(snapshot, **kwargs):
    cfg=Config(context_each_side=0,direct_fraction=0,seed_fraction=0,**kwargs)
    idx=build_index(snapshot);corpus=Corpus(snapshot,CharacterTokenizer(),cfg)
    question='herbs halloumi shop train'
    query=map_groups(idx,question,cfg)
    scores=corpus.scores(question)
    pool=candidate_pool(idx,query,enumerate_graph(idx,query,cfg),corpus,scores,cfg)
    return cfg,idx,corpus,query,pool


def test_semantic_relevance_changes_selection_on_identical_real_pool(snapshot):
    cfg,idx,corpus,query,pool=setup(snapshot,evidence_budget=1000,max_actions=1)
    packets=[]
    for sid in ('D1:1','D2:1'):
        result=compare(idx,query['question'],CharacterTokenizer(),config=cfg,corpus=corpus,
                       arms=['P3','C_rerank'],reranker=LocalReranker('['+sid+':'))
        base,changed=result['rows']
        assert base['candidates']['pool_hash']==changed['candidates']['pool_hash']==pool['pool_hash']
        assert {s['source_id'] for s in changed['evidence_pack']['spans']}=={sid}
        assert changed['calls']['rerank'][0]['usage']['total_tokens']==19
        packets.append(changed['evidence_pack']['text_hash'])
    assert packets[0]!=packets[1]


def test_focus_document_retains_context_without_faking_focus(snapshot):
    cfg=Config(context_each_side=1);corpus=Corpus(snapshot,CharacterTokenizer(),cfg)
    aid=corpus.source_atoms['D1:2'][0]; adjacent=corpus.source_atoms['D1:1'][0]
    doc=rerank_documents(corpus,[aid,adjacent])[0]
    focus,context=doc.split('ADJACENT CONTEXT\n')
    assert '[D1:2:' in focus and '[D1:1:' not in focus
    assert '[D1:1:' in context and '[D1:2:' not in context
    assert 'speaker' in focus and 'session_datetime' in focus


def test_reranker_does_not_read_neighbors_outside_admitted_pool(snapshot):
    cfg=Config(context_each_side=1);corpus=Corpus(snapshot,CharacterTokenizer(),cfg)
    aid=corpus.source_atoms['D1:2'][0]
    assert len(corpus.contexts[aid])>1
    focus,context=rerank_documents(corpus,[aid])[0].split('ADJACENT CONTEXT\n')
    assert '[D1:2:' in focus and context==''


def test_paired_semantic_arms_share_one_backend_call_without_disk_cache(snapshot):
    cfg,idx,corpus,query,pool=setup(snapshot,evidence_budget=650)
    backend=LocalReranker('[D1:2:')
    result=compare(idx,query['question'],CharacterTokenizer(),config=cfg,corpus=corpus,
                   arms=['C_rerank','C_mmr'],reranker=backend)
    left,right=result['rows']
    assert len(backend.calls)==1
    assert left['semantic_rerank']['scores']==right['semantic_rerank']['scores']
    assert left['calls']['rerank']==right['calls']['rerank']
    assert right['semantic_rerank']['cache_reused']
    assert left['candidates']['pool_hash']==right['candidates']['pool_hash']


def test_cache_reuse_keeps_original_independent_deployment_cost(snapshot,tmp_path):
    cfg,idx,corpus,query,pool=setup(snapshot)
    backend=LocalReranker(); first=rerank_pool(query['question'],corpus,pool,backend,cache_dir=tmp_path)
    second=rerank_pool(query['question'],corpus,pool,backend,cache_dir=tmp_path,cache_only=True)
    assert len(backend.calls)==1 and second['cache_reused'] and not first['cache_reused']
    assert first['scores']==second['scores'] and first['calls']==second['calls']
    assert second['calls'][0]['usage']['total_tokens']==19


def test_wrong_question_and_changed_pool_cannot_reuse_cache(snapshot,tmp_path):
    cfg,idx,corpus,query,pool=setup(snapshot)
    backend=LocalReranker();rerank_pool(query['question'],corpus,pool,backend,cache_dir=tmp_path)
    with pytest.raises(ContractError,match='cache record missing'):
        rerank_pool('different question',corpus,pool,backend,cache_dir=tmp_path,cache_only=True)
    smaller=deepcopy(pool);smaller['atom_ids']=smaller['atom_ids'][:-1]
    smaller['pool_hash']=digest({k:v for k,v in smaller.items() if k!='pool_hash'})
    with pytest.raises(ContractError,match='cache record missing'):
        rerank_pool(query['question'],corpus,smaller,backend,cache_dir=tmp_path,cache_only=True)
    assert len(backend.calls)==1


@pytest.mark.parametrize('damage', ['score','question','score_keys'])
def test_corrupt_or_rebound_cache_rejected(snapshot,tmp_path,damage):
    cfg,idx,corpus,query,pool=setup(snapshot)
    backend=LocalReranker();result=rerank_pool(query['question'],corpus,pool,backend,cache_dir=tmp_path)
    path=next(tmp_path.glob('*.json'));record=read_json(path)
    if damage=='score': record['scores'][next(iter(record['scores']))]=.7
    elif damage=='question':
        record['request']['question']='wrong question'
        record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'})
    else:
        record['scores'].pop(next(iter(record['scores'])))
        record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'})
    write_json(path,record)
    with pytest.raises(ContractError,match='cache|scores'):
        rerank_pool(query['question'],corpus,pool,backend,cache_dir=tmp_path,cache_only=True)
    assert len(backend.calls)==1


def mock_transport(monkeypatch, response):
    import eitr.rerank as module
    monkeypatch.setenv('TEST_RERANK_KEY','test-secret-never-real')
    calls=[]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,count):return response if isinstance(response,bytes) else json.dumps(response).encode()
    def request(req,timeout):
        calls.append(json.loads(req.data));return Response()
    monkeypatch.setattr(module,'_open_no_redirect',request)
    return calls


def provider(**kwargs):
    return DashScopeReranker('https://example.invalid/api/v1/services/rerank/text-rerank/text-rerank',
                             'pinned-test-reranker','TEST_RERANK_KEY',**kwargs)


def test_provider_returns_document_order_and_single_request(monkeypatch):
    calls=mock_transport(monkeypatch,{'output':{'results':[{'index':1,'relevance_score':.9},
                                                          {'index':0,'relevance_score':.2}]},
                                      'usage':{'total_tokens':15}})
    backend=provider();assert backend.score('question',['one','two'])==[.2,.9]
    assert len(calls)==1
    assert calls[0]['parameters']['top_n']==2 and calls[0]['input']['documents']==['one','two']
    assert backend.calls[0]['status']=='ok' and backend.calls[0]['usage']['total_tokens']==15


@pytest.mark.parametrize('results', [
    [{'index':0,'relevance_score':.2},{'index':0,'relevance_score':.9}],
    [{'index':0,'relevance_score':.2}],
    [{'index':True,'relevance_score':.2},{'index':0,'relevance_score':.9}],
    [{'index':2,'relevance_score':.2},{'index':0,'relevance_score':.9}],
    [{'index':0,'relevance_score':float('nan')},{'index':1,'relevance_score':.9}],
    [{'index':0,'relevance_score':float('inf')},{'index':1,'relevance_score':.9}],
    [{'index':0,'relevance_score':True},{'index':1,'relevance_score':.9}],
    [{'index':0,'relevance_score':1.01},{'index':1,'relevance_score':.9}],
    [{'index':0,'relevance_score':-.01},{'index':1,'relevance_score':.9}],
    [None,{'index':1,'relevance_score':.9}],
])
def test_provider_rejects_duplicate_missing_or_nonfinite_results(monkeypatch,results):
    calls=mock_transport(monkeypatch,{'output':{'results':results}});backend=provider()
    with pytest.raises(ContractError):backend.score('question',['one','two'])
    assert len(calls)==1 and backend.calls[0]['status']=='error'


def test_provider_does_not_silently_split_large_pool(monkeypatch):
    calls=mock_transport(monkeypatch,{})
    backend=provider(max_documents=2)
    with pytest.raises(ContractError,match='never compare scores across batches'):
        backend.score('question',['a','b','c'])
    assert not calls and not backend.calls


@pytest.mark.parametrize('speaker,session,date,expect_penalty',[
    ('A','D1','2024-09-15',True),
    ('B','D1','2024-09-15',False),
    ('A','D2','2024-09-15',False),
    ('A','D1','2024-10-15',False),
    ('unknown','D1','2024-09-15',False),
])
def test_mmr_does_not_suppress_cross_person_or_cross_session_evidence(raw_memory,speaker,session,date,expect_penalty):
    raw=deepcopy(raw_memory)
    for src in raw['sources']:
        src['text']='Both pets could not enter the cafe.'
        src.update(speaker='A',session_datetime='2024-09-15',datetime_provenance='native_verified')
    raw['sources'][1].update(speaker=speaker,session_id=session,session_datetime=date)
    snap=canonicalize(raw);cfg,idx,corpus,query,pool=setup(snap,evidence_budget=2000)
    # Equal source text is intentional: attribution, not lexical overlap, must decide duplication.
    scores={a:.9 for a in pool['atom_ids']};state=MMRScheduler(corpus,pool,scores,[],cfg)
    left=corpus.source_atoms['D1:1'][0];right=corpus.source_atoms['D1:2'][0]
    action=next(a for a in pool['actions'] if a['core']==left)
    state.commit(state.evaluate(action,1.,False),'test')
    assert (state.redundancy(right)>0)==expect_penalty


@pytest.mark.parametrize('arm',['C_rerank','C_mmr'])
def test_cascade_respects_full_render_budget_and_overlap(snapshot,arm):
    cfg=Config(evidence_budget=650,context_each_side=1,direct_fraction=0,seed_fraction=0)
    idx=build_index(snapshot);corpus=Corpus(snapshot,CharacterTokenizer(),cfg)
    query=map_groups(idx,'herbs halloumi shop train',cfg)
    scores=corpus.scores(query['question']);pool=candidate_pool(idx,query,enumerate_graph(idx,query,cfg),corpus,scores,cfg)
    packet=select_cascade(corpus,pool,{a:.9 for a in pool['atom_ids']},cfg,arm,True)
    assert packet['atom_ids']
    assert packet['used_tokens']==len(packet['text'])<=650
    assert sum(step['cost'] for step in packet['ledger'])==packet['used_tokens']
    fresh=[a for step in packet['ledger'] for a in step['fresh_atoms']]
    assert len(fresh)==len(set(fresh))
    selected=set(packet['atom_ids'])
    for step in packet['ledger']:assert set(corpus.contexts[step['action_id']])<=selected


def test_seed_reserve_never_admits_negative_mmr_priority(raw_memory):
    raw=deepcopy(raw_memory)
    for src in raw['sources']:
        if src['session_id']=='D1':
            src.update(text='Both pets could not enter the cafe.', speaker='A',
                       session_datetime='2024-09-15', datetime_provenance='native_verified')
    snap=canonicalize(raw)
    cfg=Config(evidence_budget=2000,context_each_side=0,direct_fraction=0,seed_fraction=1,
               mmr_lambda=.85)
    idx=build_index(snap);corpus=Corpus(snap,CharacterTokenizer(),cfg)
    query=map_groups(idx,'herbs halloumi shop train',cfg)
    pool=candidate_pool(idx,query,enumerate_graph(idx,query,cfg),corpus,
                        corpus.scores(query['question']),cfg)
    packet=select_cascade(corpus,pool,{a:.01 for a in pool['atom_ids']},cfg,'C_mmr',True)
    # This is an affordable seed action, not an over-budget path or an unused flag.
    assert any(a.get('phase')=='seed' and a.get('priority',0)<0 for a in packet['alternatives'])
    assert packet['ledger'] and all(step['priority']>=0 for step in packet['ledger'])
    assert sum(s['source_id'].startswith('D1:') for s in packet['spans'])==1
    assert packet['used_tokens']<cfg.evidence_budget


@pytest.mark.parametrize('budget,context',[(350,0),(650,0),(650,1)])
def test_mmr_lambda_one_matches_semantic_greedy_even_with_duplicates(raw_memory,budget,context):
    raw=deepcopy(raw_memory)
    for src in raw['sources']:
        if src['session_id']=='D1':
            src.update(text='Both pets could not enter the cafe.', speaker='A',
                       session_datetime='2024-09-15', datetime_provenance='native_verified')
    snap=canonicalize(raw)
    cfg=Config(evidence_budget=budget,context_each_side=context,mmr_lambda=1.)
    idx=build_index(snap);corpus=Corpus(snap,CharacterTokenizer(),cfg)
    query=map_groups(idx,'herbs halloumi shop train',cfg)
    pool=candidate_pool(idx,query,enumerate_graph(idx,query,cfg),corpus,
                        corpus.scores(query['question']),cfg)
    scores={a:.3+.1*i for i,a in enumerate(sorted(pool['atom_ids']))}
    ordinary=select_cascade(corpus,pool,scores,cfg,'C_rerank',True)
    mmr=select_cascade(corpus,pool,scores,cfg,'C_mmr',True)
    assert ordinary['atom_ids'] and ordinary['atom_ids']==mmr['atom_ids']
    assert ordinary['text_hash']==mmr['text_hash']
    assert ordinary['used_tokens']==mmr['used_tokens']<=budget
    assert [(x['action_id'],x['cost'],x['priority']) for x in ordinary['ledger']]==[
        (x['action_id'],x['cost'],x['priority']) for x in mmr['ledger']]
