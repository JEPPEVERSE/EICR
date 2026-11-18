import copy

import numpy as np
import pytest

from eicr import build_index, load_index, save_index
from eicr.contracts import ContractError
from eicr.import_mragent import import_pack


def test_exact_dedup_and_same_text_different_sources(snapshot):
    copy_fact = copy.deepcopy(snapshot["evidence"][0])
    copy_fact["evidence_id"] = "copy"
    snapshot["evidence"].append(copy_fact)
    snapshot["edges"].append({"cue_id":"halloumi", "evidence_id":"copy"})
    index = build_index(snapshot)
    assert len(index.fact_ids) == 5
    assert index.report["exact_duplicates"] == 1
    snapshot["evidence"][-1]["origins"] = ["s4"]
    assert len(build_index(snapshot).fact_ids) == 6


def test_future_filter_before_stats_and_alias_rejection(snapshot):
    prefix = copy.deepcopy(snapshot)
    prefix["provenance"]["allowed_source_ids"].remove("s4")
    index = build_index(prefix)
    prefix["evidence"][-1]["topics"] = ["future-only-topic"]
    prefix["sources"][-1]["text"] = "future changed"
    again = build_index(prefix)
    assert index.index_hash == again.index_hash
    prefix["cues"][0]["aliases"] = [{"text":"future", "source_ids":["s4"]}]
    with pytest.raises(ContractError, match="alias"):
        build_index(prefix)


def test_dangling_id_fails(snapshot):
    snapshot["edges"].append({"cue_id":"halloumi", "evidence_id":"missing"})
    with pytest.raises(ContractError, match="Dangling"):
        build_index(snapshot)


def test_overlapping_origins_share_family(snapshot):
    snapshot["evidence"][0]["origins"].append("s1")
    index = build_index(snapshot)
    f = index.records["source_families"]
    assert f["s0"] == f["s1"]
    assert len(index.report["family_sizes"]) == 4
    assert index.weights.sum() == pytest.approx(1)


def test_index_roundtrip_and_tamper(snapshot, tmp_path):
    index = build_index(snapshot)
    path = tmp_path/"index"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.index_hash == index.index_hash
    np.testing.assert_array_equal(loaded.X.toarray(), index.X.toarray())
    (path/"records.json").write_text("{}")
    with pytest.raises(ContractError, match="hash"):
        load_index(path)


def test_native_import_joins_ids_not_sentence_order():
    payload = {"topics": {"T1":"cooking"}, "sentence": [
        {"id":"f1","text":"cook","origin":"D1:1","tag":"food","topic":["T1"]},
        {"id":"f2","text":"shop","origin":"D1:2","tag":"shopping","topic":[]}
    ]}
    pack = {"raw_text":{"D1":{"D1:1":"user:cook","D1:2":"user:shop"}}, "rewrites":[{"session_1":payload}],
            "keywords":[{"sentence":[{"sentence_id":"f2","keyword":["shop"]},{"sentence_id":"f1","keyword":["cook"]}]}]}
    manifest = {"owner_id":"x", "snapshot_id":"p", "allowed_source_ids":["D1:1","D1:2"], "extraction_prefix_verified":True}
    canonical = import_pack(pack, manifest)
    index = build_index(canonical)
    cook = next(c["cue_id"] for c in index.records["cues"] if c["text"] == "cook")
    fact = index.records["evidence"][index.R[index.cue_pos[cook]].indices[0]]
    assert fact["origins"] == ["D1:1"]
    assert fact["metadata"]["time_provenance"] == "legacy_unverified_rewrite"


def test_tag_does_not_create_direct_edges(snapshot):
    for fact in snapshot["evidence"]:
        fact["tags"] = ["same-wide-tag"]
    index = build_index(snapshot)
    assert index.similarity("halloumi","shopping")["combined"] == 0
