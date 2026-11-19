"""Small independent exhaustive references, not copies of implementation helpers."""
import random
import numpy as np
import pytest
from eicr.packer import CharacterTokenizer
from eitr.native import canonicalize
from eitr.index import build_index
from eitr.config import Config
from eitr.query import map_groups
from eitr.candidates import enumerate_graph
from eitr.pipeline import retrieve
from eitr.evidence import Corpus


def test_sparse_profiles_match_direct_distribution_formula(raw_memory):
    raw_memory['evidence'][1]['tags'].append('shopping')
    snapshot=canonicalize(raw_memory);idx=build_index(snapshot)
    tags=[t['tag_id'] for t in snapshot['tags']];cues=[c['cue_id'] for c in snapshot['cues']]
    expected=np.zeros((len(tags),len(cues)))
    for ti,tid in enumerate(tags):
        fs=[f for f in snapshot['facts'] if tid in f['tag_ids']]
        origins={tuple(f['origins']) for f in fs}
        distributions=[]
        for origin in origins:
            rows=[]
            for f in fs:
                if tuple(f['origins'])!=origin:continue
                cs={c for c,fid in snapshot['feature_edges'] if fid==f['fact_id']}
                if cs:rows.append(np.array([int(c in cs)/len(cs) for c in cues]))
            if rows:distributions.append(np.mean(rows,axis=0))
        if distributions:expected[ti]=np.mean(distributions,axis=0)
    assert np.allclose(expected,idx.profiles.toarray())
    weights=np.log((sum(expected.sum(axis=1)>0)+1)/((expected>0).sum(axis=0)+1))
    vectors=np.sqrt(expected*weights)
    for row in vectors:
        if np.linalg.norm(row):row/=np.linalg.norm(row)
    assert np.allclose(vectors,idx.vectors.toarray())
    for i,tid in enumerate(tags):
        for edge in idx.neighbors[tid]:
            assert edge['score']==pytest.approx(float(vectors[i]@vectors[tags.index(edge['target'])]))


def test_enumeration_matches_all_simple_paths_to_depth_two(snapshot):
    idx=build_index(snapshot);tags=[t['tag_id'] for t in snapshot['tags']];rng=random.Random(821)
    cfg=Config(path_decay=.8,max_edge_evaluations=10000,max_state_pops=10000)
    for _ in range(15):
        idx.neighbors={a:[{'target':b,'score':rng.uniform(.1,1)} for b in tags if b!=a and rng.random()<.7] for a in tags}
        groups=[{'group_id':'g0','seed_tags':tags[:2],'support_weight':.5}, {'group_id':'g1','seed_tags':tags[2:],'support_weight':.5}]
        result=enumerate_graph(idx,{'index_hash':idx.index_hash,'groups':groups},cfg)
        expected={}
        for group in groups:
            best={t:1. for t in group['seed_tags']}
            for seed in group['seed_tags']:
                for e in idx.neighbors[seed]:
                    b=e['target'];best[b]=max(best.get(b,0),.8*e['score'])
                    for e2 in idx.neighbors[b]:
                        c=e2['target']
                        if c==seed:continue
                        best[c]=max(best.get(c,0),.8**2*e['score']*e2['score'])
            expected[group['group_id']]=best
        for group,best in expected.items():
            assert result['group_support'][group]==pytest.approx(best)
        for tag,support in result['support'].items():
            assert support==pytest.approx(sum(d.get(tag,0)*.5 for d in expected.values()))


def test_each_dynamic_choice_matches_exhaustive_marginal_objective(snapshot):
    idx=build_index(snapshot);cfg=Config(evidence_budget=1000,direct_fraction=0,seed_fraction=0)
    tok=CharacterTokenizer();corpus=Corpus(snapshot,tok,cfg)
    run=retrieve(idx,'herbs train halloumi',tok,config=cfg,corpus=corpus)
    whole=run['relevance']['whole_query'];cues=[c['scores'] for c in run['relevance']['cue_groups']]
    selected=set()
    for actual in run['evidence_pack']['ledger']:
        assert actual['phase']=='shared'
        choices=[]
        for action in run['candidates']['actions']:
            new=set(action['atoms'])-selected
            if not new:continue
            proposed=corpus.render(selected|new)
            if proposed['used_tokens']>cfg.evidence_budget:continue
            cost=proposed['used_tokens']-corpus.render(selected)['used_tokens']
            coverage_before=sum(max((s[a] for a in selected),default=0) for s in cues)/len(cues)
            coverage_after=sum(max((s[a] for a in selected|new),default=0) for s in cues)/len(cues)
            rel=sum(corpus.atoms[a]['text_tokens']*whole[a] for a in sorted(new))
            numerator=cfg.eta*rel+(1-cfg.eta)*128*(coverage_after-coverage_before)
            priority=(.1+.9*action['support'])*numerator/max(1,cost)
            if priority>0:choices.append(((-priority,-numerator,cost,action['action_id']),action))
        expected=min(choices,key=lambda x:x[0])[1]
        assert expected['action_id']==actual['action_id']
        selected.update(expected['atoms'])
    assert selected==set(run['evidence_pack']['atom_ids'])
