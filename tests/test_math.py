import copy

import numpy as np
import pytest

from eicr import IndexConfig, build_index
from eicr.contracts import ContractError


def test_sparse_math_matches_dense_and_hellinger(snapshot):
    index = build_index(snapshot, IndexConfig(structure=True, lambda_=0.2))
    x, y = index.X.toarray(), index.Y.toarray()
    valid = np.flatnonzero(index.norm_x)
    nx = x[valid] / index.norm_x[valid, None]
    ny = y[valid] / index.norm_y[valid, None]
    v = np.hstack([np.sqrt(0.8)*nx, np.sqrt(0.2)*ny])
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1)
    gram = v @ v.T
    assert np.linalg.eigvalsh(gram).min() > -1e-10
    for a, i in enumerate(valid):
        for b, j in enumerate(valid):
            scores = index.similarity(index.cue_ids[i], index.cue_ids[j])
            assert scores["combined"] == pytest.approx(gram[a, b])
            p = index.weights * index.R[i].toarray()[0] / index.mass[i]
            q = index.weights * index.R[j].toarray()[0] / index.mass[j]
            assert scores["direct"] == pytest.approx(np.sqrt(p*q).sum())
            if scores["direct"] == 0:
                assert scores["combined"] <= 0.2 + 1e-10
    assert index.similarity("empty", "empty")["combined"] is None
    assert index.joint([])["normalized"] is None


def test_weak_strength_survives_normalization(snapshot):
    index = build_index(snapshot, IndexConfig(structure=True, lambda_=0.2))
    assert index.report["structure_active"]
    for i in range(len(index.fact_ids)):
        assert index.P[i].sum() == pytest.approx(1)
        assert index.T[i].sum() == pytest.approx(1)
        assert float(index.T[i].sum() - index.T[i, i]) <= 0.0625 + 1e-12


def test_no_topic_identity_and_duplicate_edge(snapshot):
    for fact in snapshot["evidence"]:
        fact["topics"] = []
    baseline = build_index(snapshot)
    snapshot["edges"].append(copy.deepcopy(snapshot["edges"][0]))
    structural = build_index(snapshot, IndexConfig(structure=True, lambda_=0.2))
    assert not structural.report["structure_active"]
    for a in baseline.cue_ids:
        for b in baseline.cue_ids:
            assert baseline.similarity(a,b)["combined"] == pytest.approx(structural.similarity(a,b)["combined"])


def test_containment_is_not_conflict(snapshot):
    index = build_index(snapshot)
    assert index.coverage("halloumi", "owner") == pytest.approx(1)
    assert index.similarity("halloumi", "owner")["direct"] < 1


def test_invalid_config():
    for values in ({"beta": 1}, {"lambda_": 0.2}, {"neighbors": 0}, {"topic_strength": float("nan")}):
        with pytest.raises(ContractError):
            IndexConfig(**values)


def test_coordinate_mass_conserving_split():
    r = np.array([[1, 1], [1, 0]], dtype=float)
    w = np.array([0.6, 0.4])
    old = (r*np.sqrt(w)) @ (r*np.sqrt(w)).T
    split_r = np.array([[1,1,1,1], [1,1,1,0]], dtype=float)
    split_w = np.array([0.2,0.2,0.2,0.4])
    new = (split_r*np.sqrt(split_w)) @ (split_r*np.sqrt(split_w)).T
    np.testing.assert_allclose(old, new)


def test_pairwise_overlap_has_no_joint(snapshot):
    snapshot["edges"] = [{"cue_id": c, "evidence_id": "e"+str(i)} for c, ids in {"halloumi":[0,1], "pinch":[1,2], "paprika":[0,2]}.items() for i in ids]
    index = build_index(snapshot)
    for a,b in [("halloumi","pinch"),("pinch","paprika"),("paprika","halloumi")]:
        assert index.similarity(a,b)["combined"] > 0
    assert index.joint(["halloumi","pinch","paprika"])["mass"] == 0
