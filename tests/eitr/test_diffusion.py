from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from eicr.contracts import ContractError
from eicr.packer import CharacterTokenizer
from eitr.config import Config
from eitr.diffusion import enumerate_ppr, reciprocal_rank_fusion
from eitr.candidates import candidate_pool
from eitr.evidence import Corpus
from eitr.index import build_index
from eitr.native import canonicalize
from eitr.query import map_groups


def graph(edges, seeds=None):
    ids = sorted(edges)
    index = SimpleNamespace(index_hash="fixture-index", snapshot={"owner_id": "owner-A",
                            "snapshot_id": "snapshot-A", "tags": [{"tag_id": t} for t in ids]},
                            neighbors={s: [{"target": t, "score": w} for t, w in row] for s, row in edges.items()})
    seeds = seeds or [[ids[0]]]
    query = {"index_hash": index.index_hash, "groups": [
        {"group_id": str(i), "seed_tags": tags, "support_weight": 1 / len(seeds)}
        for i, tags in enumerate(seeds)]}
    return index, query


def exact_ppr(index, seeds, restart):
    """Independent dense linear-system oracle, including dangling row repair."""
    ids = sorted(index.neighbors)
    p = np.array([float(t in seeds) / len(seeds) for t in ids])
    transition = np.zeros((len(ids), len(ids)))
    for i, source in enumerate(ids):
        row = index.neighbors[source]
        if not row:
            transition[i] = p
        else:
            total = sum(e['score'] for e in row)
            for edge in row:
                transition[i, ids.index(edge['target'])] = edge['score'] / total
    solution = np.linalg.solve(np.eye(len(ids)) - (1-restart)*transition.T, restart*p)
    return dict(zip(ids, solution))


def test_ppr_matches_independent_linear_system_with_dangling_and_multi_seeds():
    index, query = graph({'a': [('b', .8), ('c', .2)], 'b': [('a', .4)], 'c': [], 'z': []},
                         seeds=[['a', 'b'], ['c']])
    result = enumerate_ppr(index, query, Config(), restart_probability=.3,
                           tolerance=1e-11, max_iterations=200)
    assert result['ppr']['converged']
    for group in query['groups']:
        expected = exact_ppr(index, group['seed_tags'], .3)
        actual = result['group_support'][group['group_id']]
        assert sum(abs(actual.get(t, 0) - v) for t, v in expected.items()) <= 1e-11
        assert sum(actual.values()) == pytest.approx(1.)
    assert 'z' not in result['support']
    assert sum(result['support'].values()) == pytest.approx(1.)
    assert result['support'] == pytest.approx({t: .5*(result['group_support']['0'].get(t, 0)
                + result['group_support']['1'].get(t, 0)) for t in result['support']})


def test_empty_and_no_expand_and_restart_one():
    index, query = graph({'a': [('b', 1.)], 'b': []})
    no_expand = enumerate_ppr(index, query, Config(), expand=False)
    assert no_expand['support'] == {'a': 1.}
    assert no_expand['stop'] == 'seed_only' and no_expand['edge_evaluations'] == 0
    restart = enumerate_ppr(index, query, Config(), restart_probability=1., max_edge_evaluations=0)
    assert restart['support'] == {'a': 1.}
    assert restart['ppr']['converged'] and restart['edge_evaluations'] == 0
    query['groups'] = []
    empty = enumerate_ppr(index, query, Config())
    assert empty['support'] == {} and empty['candidate_order'] == [] and empty['stop'] == 'no_seeds'


def test_iteration_caps_do_not_commit_partial_probability_vectors():
    index, query = graph({'a': [('b', .8), ('c', .2)], 'b': [('a', 1.)], 'c': []})
    blocked = enumerate_ppr(index, query, Config(), max_edge_evaluations=3)
    assert blocked['stop'] == 'edge_cap_before_iteration'
    assert blocked['edge_evaluations'] == 0 and blocked['support'] == {'a': 1.}
    partial = enumerate_ppr(index, query, Config(), max_edge_evaluations=4)
    assert partial['ppr']['iterations'] == 1 and not partial['ppr']['converged']
    assert partial['support'] == pytest.approx({'a': .15, 'b': .68, 'c': .17})
    assert partial['edge_evaluations'] == 4 and partial['state_pops'] == 1
    once = enumerate_ppr(index, query, Config(), max_iterations=1)
    assert once['stop'] == 'iteration_cap' and once['support'] == partial['support']


def test_ppr_output_cap_is_after_diffusion_and_preserves_seed_scope():
    index, query = graph({'a': [('b', 1.)], 'b': [('c', 1.)], 'c': [('b', 1.)], 'z': []})
    config = Config(max_reached_tags=2, max_seed_tags=1)
    capped = enumerate_ppr(index, query, config, max_iterations=200)
    full = enumerate_ppr(index, query, Config(), max_iterations=200)
    assert 'a' in capped['support'] and len(capped['support']) == 2
    assert capped['support'] == {t: full['support'][t] for t in capped['support']}
    assert capped['ppr']['retained_mass'] < 1.
    assert capped['ppr']['walk_reached_tags'] == 3
    assert capped['candidate_order'] == sorted(capped['support'], key=lambda t: (-capped['support'][t], t))
    assert capped['candidate_order'][0] == 'b'
    assert capped['new_tag_rejections'] == 1


def test_determinism_and_work_accounting():
    index, query = graph({'a': [('b', .6), ('c', .4)], 'b': [('c', 1.)], 'c': [('a', 1.)]},
                         seeds=[['a', 'b'], ['c']])
    one = enumerate_ppr(index, query, Config(), max_iterations=2)
    index.neighbors = {t: list(reversed(edges)) for t, edges in reversed(list(index.neighbors.items()))}
    query['groups'].reverse()
    query['groups'][1]['seed_tags'].reverse()
    assert enumerate_ppr(index, query, Config(), max_iterations=2) == one
    # First multiply visits 3 edges in group 0, 1 in group 1; second visits 4 and 3.
    assert one['ppr']['normalization_edge_evaluations'] == 4
    assert one['ppr']['propagation_edge_evaluations'] == 11
    assert one['edge_evaluations'] == 15
    assert one['state_pops'] == 8


@pytest.mark.parametrize('change', ['hash', 'owner', 'seed', 'unadmitted', 'weight', 'duplicate_group', 'edge'])
def test_scope_errors_fail_closed(change):
    index, query = graph({'a': [('b', 1.)], 'b': []})
    if change == 'hash': query['index_hash'] = 'other-index'
    if change == 'owner': query['owner_id'] = 'other-owner'
    if change == 'seed': query['groups'][0]['seed_tags'] = ['foreign-tag']
    if change == 'unadmitted': query['groups'][0]['candidate_tags'] = ['b']
    if change == 'weight': query['groups'][0]['support_weight'] = float('nan')
    if change == 'duplicate_group': query['groups'].append(deepcopy(query['groups'][0]))
    if change == 'edge': index.neighbors['a'][0]['target'] = 'foreign-tag'
    with pytest.raises(ContractError): enumerate_ppr(index, query, Config())


@pytest.mark.parametrize('kw', [dict(restart_probability=0), dict(restart_probability=float('nan')),
    dict(tolerance=0), dict(tolerance=float('inf')), dict(max_iterations=True),
    dict(max_edge_evaluations=-1)])
def test_invalid_solver_settings(kw):
    index, query = graph({'a': []})
    with pytest.raises(ContractError): enumerate_ppr(index, query, Config(), **kw)


def test_existing_candidate_pool_retains_exact_seed_fact_scope(raw_memory):
    raw_memory['evidence'][2]['tags'] = ['shopping']
    snapshot = canonicalize(raw_memory)
    index = build_index(snapshot)
    config = Config(context_each_side=0)
    query = map_groups(index, 'paprika', config)
    result = enumerate_ppr(index, query, config)
    corpus = Corpus(snapshot, CharacterTokenizer(), config)
    pool = candidate_pool(index, query, result, corpus, corpus.scores('paprika'), config)
    seed = next(t['tag_id'] for t in snapshot['tags'] if t['text'] == 'shopping')
    assert all(corpus.atoms[a['core']]['source_id'] != 'D1:10'
               for a in pool['actions'] if seed in a['tags'])


def test_rrf_rank_fusion_without_score_calibration_or_double_counting():
    result = reciprocal_rank_fusion([['a', 'b'], ['b', 'c']], rank_constant=60)
    assert [r['id'] for r in result] == ['b', 'a', 'c']
    assert result[0]['score'] == pytest.approx(1/62 + 1/61)
    assert result[0]['ranks'] == [2, 1]
    assert reciprocal_rank_fusion([['z'], ['a']])[0]['id'] == 'a'
    assert reciprocal_rank_fusion([['a']], limit=0) == []
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([['a'], ['b']], weights=[0., 1.])[0]['id'] == 'b'
    with pytest.raises(ContractError): reciprocal_rank_fusion([['a', 'a']])
    with pytest.raises(ContractError): reciprocal_rank_fusion([['a']], weights=[float('nan')])
