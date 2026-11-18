"""Regression cases from the 2026-11-18 review, independent of benchmark gold."""

import copy
import io
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np
import pytest

from eicr import IndexConfig, build_index, load_index, save_index
from eicr.contracts import ContractError, digest
from eicr.diagnostics import diagnose
from eicr.packer import CharacterTokenizer, chunk_sources
from eicr.pipeline import answer_from_sources, run
from eicr.policy import choose_route, freeze_policy
from eicr.provider import ChatBackend
from eicr.provider import ProviderTransportError, _NoRedirect
from eicr.query_map import map_query
from eicr.retrieval import SharedRetriever, collect_candidates


def test_structure_index_saves_and_loads(snapshot, tmp_path):
    index = build_index(snapshot, IndexConfig(structure=True, lambda_=0.2))
    save_index(index, tmp_path / "structural")
    loaded = load_index(tmp_path / "structural")
    assert loaded.report["structure_active"]
    np.testing.assert_array_equal(index.Y.toarray(), loaded.Y.toarray())


def test_structural_index_audit_across_hash_seeds(tmp_path):
    n = 80
    facts = [{"evidence_id": f"e{i}", "text": f"fact {i}", "origins": [f"s{i}"], "topics": []} for i in range(n)]
    for j in range(31):
        for i in [0, 1] + list(range(2, 2 + j % 3)):
            facts[i]["topics"].append("topic-" + str(j))
    snapshot = {
        "owner_id": "o", "snapshot_id": "s",
        "sources": [{"source_id": f"s{i}", "text": f"fact {i}"} for i in range(n)],
        "cues": [{"cue_id": "c", "text": "cue"}], "evidence": facts,
        "edges": [{"cue_id": "c", "evidence_id": "e0"}],
        "provenance": {"allowed_source_ids": [f"s{i}" for i in range(n)], "extraction_prefix_verified": True}
    }
    path = tmp_path / "index"
    save_index(build_index(snapshot, IndexConfig(structure=True, lambda_=0.2)), path)
    for seed in ("1", "2", "3"):
        result = subprocess.run([sys.executable, "-m", "eicr", "audit-index", "--index", str(path)],
                                env={**os.environ, "PYTHONHASHSEED": seed}, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["status"] == "passed"


@pytest.mark.parametrize("table", ["sources", "cues", "evidence", "edges"])
def test_declared_foreign_owner_is_rejected(snapshot, table):
    snapshot[table][0]["owner_id"] = "foreign-owner"
    with pytest.raises(ContractError, match="owner"):
        build_index(snapshot)


def test_missing_origin_is_not_silently_treated_as_future(snapshot):
    snapshot["evidence"][0]["origins"] = ["typo-not-a-real-source"]
    with pytest.raises(ContractError, match="source"):
        build_index(snapshot)


@pytest.mark.parametrize("bad", [None, 17, "not-an-object", [], {"role": "entry"}])
def test_malformed_frontend_entries_fall_back(snapshot, bad):
    class Frontend:
        def __init__(self):
            self.calls = []

        def chat_json(self, *args):
            self.calls.append({"status": "invalid-format"})
            return {"mentions": [bad]}

    backend = Frontend()
    out = run(build_index(snapshot), "halloumi", CharacterTokenizer(), frontend_backend=backend)
    assert out["route"]["mode"] == "ORIGINAL"
    assert out["query_map"]["frontend_error"]
    assert len(backend.calls) == 2


def test_diagnostics_cannot_be_reused_for_another_question(snapshot):
    index = build_index(snapshot)
    first = map_query(index, "halloumi shopping")
    second = map_query(index, "pinch paprika")
    with pytest.raises(ContractError, match="[Qq]uery"):
        choose_route(index, second, diagnose(index, first))


def test_fuzzy_match_to_scope_does_not_hide_unresolved_entry(snapshot):
    snapshot["cues"].append({"cue_id": "person", "text": "owner person", "scope_role": "scope"})
    index = build_index(snapshot)
    query = map_query(index, "person", [{"span_start": 0, "span_end": 6, "surface_text": "person"}])
    assert query["mentions"][0]["status"] == "ambiguous"
    assert query["mentions"][0]["role"] == "entry"
    assert diagnose(index, query)["unresolved_entries"] == ["person"]


def test_normalized_scan_preserves_original_offsets(snapshot):
    snapshot["cues"] += [{"cue_id": "road", "text": "Straße"}, {"cue_id": "space", "text": "halloumi salad"},
                          {"cue_id": "accent", "text": "café"}]
    index = build_index(snapshot)
    question = "STRASSE halloumi   salad cafe\u0301"
    result = map_query(index, question)
    ids = {m["candidate_cue_ids"][0] for m in result["mentions"] if m["status"] == "unique"}
    assert {"road", "space", "accent"} <= ids
    for mention in result["mentions"]:
        assert question[mention["span_start"]:mention["span_end"]] == mention["surface_text"]


def test_nonmonotone_tokenizer_sentence_boundary_remains_within_cap():
    class Nonmonotone:
        identity = "test-counter"

        def count(self, text):
            # A shorter sentence-ending prefix can tokenize into more pieces.
            return 15 if text == "abcde. " else len(text)

    chunks = chunk_sources([{"source_id": "s", "text": "abcde. xyz more text"}], Nonmonotone(), 10, 0)
    assert all(Nonmonotone().count(c["text"]) <= 10 for c in chunks)


def test_large_finite_dense_values_do_not_overflow_norm():
    chunks = [{"span_id": "x", "text": "x"}, {"span_id": "y", "text": "y"}]
    retriever = SharedRetriever(chunks, [[1e308, 0], [0, 1e308]], "synthetic")
    np.testing.assert_allclose(np.linalg.norm(retriever.dense, axis=1), 1)
    assert retriever.rank("", [0, 1e308])[0]["span_id"] == "y"


def test_answer_source_text_is_verbatim_and_route_metadata_stays_private(snapshot):
    index = build_index(snapshot)
    out = run(index, "halloumi", CharacterTokenizer())

    class Backend:
        def chat_json(self, system, user):
            assert user.endswith(out["evidence_pack"]["text"])
            assert '"queues"' not in user
            return {"answer_text": "Insufficient.", "citations": [], "unsupported_premises": [], "resolution_status": "insufficient"}

    answer_from_sources("halloumi", out["evidence_pack"], Backend())


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_provider_never_follows_redirects_with_credentials(code):
    request = urllib.request.Request("https://api.example.test/v1/chat/completions", data=b"private-memory",
                                     headers={"Authorization": "Bearer synthetic-test-value"})
    with pytest.raises(ProviderTransportError, match="redirect"):
        _NoRedirect().redirect_request(request, None, code, "Found", {}, "https://different.example.test/collect")


@pytest.mark.parametrize("body", [b"[]", b"null", b'{"choices":[null]}',
                                  b'{"choices":[{"finish_reason":"length"}]}', b'{"value":NaN}'])
def test_malformed_provider_payloads_are_redacted_and_logged(monkeypatch, body):
    monkeypatch.setenv("EICR_TEST_KEY", "synthetic")
    monkeypatch.setattr("eicr.provider._open_no_redirect", lambda *args, **kwargs: io.BytesIO(body))
    backend = ChatBackend("https://example.test/v1", "model", key_env="EICR_TEST_KEY")
    with pytest.raises(ContractError):
        backend.chat_json("system", "q")
    assert backend.calls[-1]["status"] == "error"


def test_provider_response_cap_and_usage_allowlist(monkeypatch):
    monkeypatch.setenv("EICR_TEST_KEY", "synthetic")
    body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{}'}}],
                       "usage": {"total_tokens": 4, "echoed_private_input": "private"}}).encode()
    monkeypatch.setattr("eicr.provider._open_no_redirect", lambda *args, **kwargs: io.BytesIO(body))
    backend = ChatBackend("https://example.test/v1", "m", key_env="EICR_TEST_KEY")
    assert backend.chat_json("s", "q") == {}
    assert backend.calls[0]["usage"] == {"total_tokens": 4}
    capped = ChatBackend("https://example.test/v1", "m", key_env="EICR_TEST_KEY", max_response_bytes=10)
    with pytest.raises(ContractError, match="byte cap"):
        capped.chat_json("s", "q")


def test_frontend_transport_error_is_not_retried_as_format(snapshot):
    class Backend:
        def __init__(self):
            self.calls = []

        def chat_json(self, *args):
            self.calls.append({"status": "error"})
            raise ProviderTransportError("test transport failure")

    backend = Backend()
    out = run(build_index(snapshot), "halloumi", CharacterTokenizer(), frontend_backend=backend)
    assert len(backend.calls) == 1
    assert out["query_map"]["frontend_error"] == "ProviderTransportError"


def test_distinct_facts_in_one_turn_do_not_manufacture_core_consensus(snapshot):
    snapshot["edges"] = [{"cue_id": c, "evidence_id": "e" + str(i)} for c, i in
                         [("halloumi", 0), ("pinch", 1), ("paprika", 2), ("shopping", 3)]]
    for fact in snapshot["evidence"][:3]:
        fact["origins"] = ["s0"]
    index = build_index(snapshot)
    query = map_query(index, "halloumi pinch paprika shopping")
    diagnostics = diagnose(index, query)
    route = choose_route(index, query, diagnostics)
    # Exercise candidate collection separately from the trigger classifier.
    route.update(mode="SINGLE_CUE_CHECK", selected_cue_id="shopping")
    retriever = SharedRetriever(chunk_sources(index.records["sources"], CharacterTokenizer()))
    queues, _ = collect_candidates(index, query, diagnostics, route, retriever)
    assert queues["core"]
    for candidate in queues["core"]:
        assert candidate["support_count"] == 1
        assert set(candidate["source_cue_union"]) == {"halloumi", "pinch", "paprika"}


def test_legacy_index_requires_rebuild(snapshot, tmp_path):
    path = tmp_path / "legacy"
    save_index(build_index(snapshot), path)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema"] = "eicr-index-1"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="rebuild"):
        load_index(path)


def test_malformed_policy_and_legacy_policy_cannot_enable_routing(snapshot):
    index = build_index(snapshot)
    query = map_query(index, "halloumi shopping")
    diagnostics = diagnose(index, query)
    for policy in ([], "invalid", {"enabled": "false"}):
        route = choose_route(index, query, diagnostics, policy)
        assert route["mode"] == "ORIGINAL"
        assert route["reasons"] == ["POLICY_INCOMPATIBLE"]
    body = {"schema": "eicr-policy-1", "enabled": True, "index_hash": index.index_hash,
            "thresholds": {"tau_delta": 0, "tau_core": 0, "tau_margin": 0, "tau_pair": 1}}
    policy = {**body, "policy_hash": digest(body)}
    assert choose_route(index, query, diagnostics, policy)["reasons"] == ["POLICY_INCOMPATIBLE"]


def test_retrieval_rejects_route_from_different_query(snapshot):
    index = build_index(snapshot)
    first = map_query(index, "halloumi shopping")
    second = map_query(index, "pinch paprika")
    route = choose_route(index, first, diagnose(index, first))
    retriever = SharedRetriever(chunk_sources(index.records["sources"], CharacterTokenizer()))
    with pytest.raises(ContractError, match="Query"):
        collect_candidates(index, second, diagnose(index, second), route, retriever)


def test_import_prefix_validates_missing_raw_origins(snapshot):
    from eicr.import_mragent import import_pack
    payload = {"topics": {}, "sentence": [{"id": "e", "text": "fact", "origin": "D1:999", "topic": []}]}
    pack = {"raw_text": {"D1": {"D1:1": "real"}}, "rewrites": [{"session_1": payload}],
            "keywords": [{"sentence": [{"sentence_id": "e", "keyword": ["fact"]}]}]}
    manifest = {"owner_id": "o", "snapshot_id": "s", "allowed_source_ids": ["D1:1"], "extraction_prefix_verified": True}
    with pytest.raises(ContractError, match="missing raw source"):
        import_pack(pack, manifest)
