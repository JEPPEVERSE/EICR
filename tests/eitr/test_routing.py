from dataclasses import replace
import pytest
from eicr.contracts import ContractError, digest
from eicr.packer import CharacterTokenizer
from eitr.config import Config
from eitr.index import build_index
from eitr.query import map_groups
from eitr.candidates import enumerate_graph, candidate_pool
from eitr.evidence import Corpus
from eitr.pipeline import compare, retrieve


def test_shared_pool_and_determinism(snapshot):
    index=build_index(snapshot); config=Config(evidence_budget=650)
    one=compare(index,'Which herbs were used for halloumi?',CharacterTokenizer(),config=config)
    two=compare(index,'Which herbs were used for halloumi?',CharacterTokenizer(),config=config)
    pools=[r['candidates']['pool_hash'] for r in one['rows'] if r['arm'] in ('P2','P3','P4')]
    assert len(set(pools))==1
    assert [r['evidence_pack']['text_hash'] for r in one['rows']]==[r['evidence_pack']['text_hash'] for r in two['rows']]
    for r in one['rows']:
        assert r['evidence_pack']['used_tokens']<=650
        assert r['calls']['exploration']==[]
    assert one['rows'][3]['work']['bm25_queries']==1
    assert one['rows'][4]['work']['bm25_queries']>1


def test_no_mapping_uses_exact_flat_policy(snapshot):
    index=build_index(snapshot)
    kwargs=dict(config=Config(evidence_budget=650),mentions=[])
    flat=retrieve(index,'pinch of herbs',CharacterTokenizer(),arm='P0',**kwargs)
    full=retrieve(index,'pinch of herbs',CharacterTokenizer(),arm='P4',**kwargs)
    assert flat['evidence_pack']['text']==full['evidence_pack']['text']
    assert full['evidence_pack']['fallback']
    assert full['work']['bm25_queries']==1
    strict=retrieve(index,'pinch of herbs',CharacterTokenizer(),arm='strict_seed',**kwargs)
    assert strict['evidence_pack']['text']==''


def test_seed_scope_does_not_expand_to_whole_tag(raw_memory):
    raw_memory['evidence'][2]['tags']=['shopping']
    from eitr.native import canonicalize
    snapshot=canonicalize(raw_memory)
    index=build_index(snapshot); config=Config(context_each_side=0)
    q=map_groups(index,'paprika',config)
    graph=enumerate_graph(index,q,config)
    corpus=Corpus(snapshot,CharacterTokenizer(),config)
    pool=candidate_pool(index,q,graph,corpus,corpus.scores('paprika'),config)
    t=next(t['tag_id'] for t in snapshot['tags'] if t['text']=='shopping')
    assert all(corpus.atoms[a['core']]['source_id']!='D1:10' for a in pool['actions'] if t in a['tags'])


def test_group_max_dedup_and_seed_cap(snapshot):
    index=build_index(snapshot); config=Config(max_seed_tags=1)
    mentions=[{'surface_text':'herbs'},{'surface_text':'halloumi'}]
    q=map_groups(index,'herbs halloumi',config,mentions)
    assert len({t for g in q['groups'] for t in g['seed_tags']})<=1
    assert sum(g['support_weight'] for g in q['groups'])==pytest.approx(1)
    graph=enumerate_graph(index,q,config)
    assert all(0<=v<=1 for v in graph['support'].values())


def test_shorter_path_not_lost_and_no_output_bridge(snapshot):
    index=build_index(snapshot); config=Config(path_decay=1.)
    ids=[t['tag_id'] for t in snapshot['tags']]; a,b,c,d=ids
    index.neighbors={a:[{'target':b,'score':.9},{'target':c,'score':.5}], b:[{'target':c,'score':.9}], c:[{'target':d,'score':.9}],d:[]}
    q={'index_hash':index.index_hash,'groups':[{'group_id':'g','seed_tags':[a],'support_weight':1.}]}
    graph=enumerate_graph(index,q,config)
    assert graph['group_support']['g'][c]==pytest.approx(.81)
    assert graph['group_support']['g'][d]==pytest.approx(.45)
    assert graph['depths'][c]==1


def test_edge_cap_is_full_row(snapshot):
    index=build_index(snapshot); config=Config(max_edge_evaluations=1)
    ids=[t['tag_id'] for t in snapshot['tags']]; a,b,c,d=ids
    index.neighbors={a:[{'target':b,'score':.8},{'target':c,'score':.7}],b:[],c:[],d:[]}
    q={'index_hash':index.index_hash,'groups':[{'group_id':'g','seed_tags':[a],'support_weight':1.}]}
    graph=enumerate_graph(index,q,config)
    assert graph['edge_evaluations']==0
    assert graph['stop']=='edge_cap_full_row'


def test_whole_context_admission(snapshot):
    config=Config(max_core_atoms=1,max_candidate_atoms=1,direct_core_reservation=1)
    index=build_index(snapshot); corpus=Corpus(snapshot,CharacterTokenizer(),config)
    q=map_groups(index,'herbs',config)
    graph=enumerate_graph(index,q,config)
    pool=candidate_pool(index,q,graph,corpus,corpus.scores('herbs'),config)
    assert len(pool['atom_ids'])<=1
    for action in pool['actions']:
        assert action['atoms']==sorted(corpus.contexts[action['core']])
    assert pool['omitted']


def test_natural_source_order_and_real_metadata(snapshot):
    corpus=Corpus(snapshot,CharacterTokenizer(),Config())
    packet=corpus.render(set(corpus.atoms))
    assert packet['text'].index('[D1:2:')<packet['text'].index('[D1:10:')
    assert '2024-09-15' in packet['text']
    assert packet['used_tokens']==len(packet['text'])
    for span in packet['spans']:
        assert span['end']<=len(corpus.sources[span['source_id']]['text'])


def test_bad_config_and_scope(snapshot):
    with pytest.raises(ContractError): Config(eta=float('nan'))
    with pytest.raises(ContractError): Config.from_dict({'schema':'design'})
    index=build_index(snapshot)
    with pytest.raises(ContractError): retrieve(index,'herbs',CharacterTokenizer(),config=Config(top_k=8))
