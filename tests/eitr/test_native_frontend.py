from copy import deepcopy
import json

import pytest

from eicr.packer import CharacterTokenizer
from eitr.config import Config
from eitr.index import build_index
from eitr.native import canonicalize
from eitr.native_frontend import (
    NativeFrontendError, NativeFrontendOptions, map_native_groups, static_tag_catalog,
)


def keywords(*items):
    return {"question_time": "", "keywords": [
        {"id": name, "alternatives": alternatives} for name, alternatives in items
    ]}


class Backend:
    def __init__(self, extraction=None, score=None, corrupt=None, fail=None):
        self.extraction = extraction or keywords(("herb", ["herbs"]))
        self.score = score or (lambda tag: 1.)
        self.corrupt = corrupt
        self.fail = fail
        self.calls = []
        self.requests = []

    def chat_json(self, system, user):
        payload = json.loads(user)
        self.requests.append(payload)
        self.calls.append({"status": "error" if self.fail else "ok",
                           "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        if self.fail:
            raise RuntimeError(self.fail)
        if "max_keywords" in payload:
            return deepcopy(self.extraction)
        response = {"groups": [
            {"group_id": group["group_id"], "tag_scores": {
                tag["tag_id"]: self.score(tag) for tag in group["tags"]
            }} for group in payload["groups"]
        ]}
        if self.corrupt:
            self.corrupt(response)
        return response


def test_alternative_matching_static_reuse_and_query_contract(snapshot):
    from eitr.pipeline import retrieve
    original = deepcopy(snapshot)
    index = build_index(snapshot)
    cfg = Config()
    backend = Backend()
    question = "Which herb went in the salad?"
    query = map_native_groups(index, question, cfg, backend=backend)
    group = query["groups"][0]
    assert group["key_ids"] == ["herbs"]
    assert group["key_matching"] == [{"cue_id": "herbs", "method": "alternative_exact",
                                       "matched_expression": "herbs"}]
    assert group["mentions"] == []
    assert set(fid for ids in group["facts_by_tag"].values() for fid in ids) == {"f1", "f2"}
    assert len(query["frontend_calls"]) == len(backend.calls) == 2
    assert [o["stage"] for o in query["frontend_outputs"]] == ["keywords", "tag_scoring"]
    assert query["frontend_request"]["status"] == "mapped"
    assert snapshot == original
    row = retrieve(index, question, CharacterTokenizer(), config=cfg, query=query, arm="P3")
    assert row["evidence_pack"]["atom_ids"]
    assert row["calls"]["frontend"] == backend.calls


def test_offline_replay_does_not_require_backend(snapshot):
    index = build_index(snapshot)
    backend = Backend()
    live = map_native_groups(index, "herb", Config(), backend=backend)
    offline = map_native_groups(index, "herb", Config(),
        keyword_response=live["frontend_outputs"][0]["response"],
        tag_response=live["frontend_outputs"][1]["response"])
    assert offline["groups"] == live["groups"]
    assert offline["frontend_calls"] == []
    assert offline["frontend_request"] == live["frontend_request"]


def test_tag_cap_applies_after_question_conditioned_scores(snapshot):
    backend = Backend(score=lambda tag: float(tag["text"] == "cooking"))
    query = map_native_groups(build_index(snapshot), "herbs shopping", Config(max_tags_per_key=1), backend=backend)
    group = query["groups"][0]
    assert len(backend.requests[1]["groups"][0]["tags"]) == 2
    assert len(group["seed_tags"]) == 1
    assert list(group["facts_by_tag"].values()) == [["f2"]]


def test_key_tag_cap_does_not_admit_facts_through_full_key(raw_memory):
    # a -> {shopping, cooking}; b -> {shopping}; choosing cooking for a
    # must not leak a's shopping fact when b later admits shopping.
    raw_memory["cues"] += [{"cue_id": "pair_a", "text": "pair a", "aliases": []},
                           {"cue_id": "pair_b", "text": "pair b", "aliases": []}]
    raw_memory["evidence"].append({"evidence_id": "f5", "text": "Different shopping fact",
                                    "origins": ["D1:10"], "tags": ["shopping"]})
    raw_memory["edges"] += [{"cue_id": key, "evidence_id": fact, "edge_origin": "native_keyword"}
                           for key, fact in (("pair_a", "f1"), ("pair_a", "f2"), ("pair_b", "f5"))]
    index = build_index(canonicalize(raw_memory))
    backend = Backend(keywords(("pair", [])), score=lambda tag: 1. if tag["text"] == "cooking" else .5)
    query = map_native_groups(index, "pair", Config(max_tags_per_key=1), backend=backend)
    group = query["groups"][0]
    assert all(len(tags) == 1 for tags in group["selected_key_tags"].values())
    assert set(fid for ids in group["facts_by_tag"].values() for fid in ids) == {"f2", "f5"}


@pytest.mark.parametrize("corrupt", [
    lambda response: response["groups"][0]["tag_scores"].update({"invented": .9}),
    lambda response: response["groups"][0]["tag_scores"].pop(next(iter(response["groups"][0]["tag_scores"]))),
    lambda response: response["groups"][0].update(group_id="invented"),
    lambda response: response["groups"].append(deepcopy(response["groups"][0])),
    lambda response: response["groups"][0]["tag_scores"].update({
        next(iter(response["groups"][0]["tag_scores"])): True}),
    lambda response: response["groups"][0]["tag_scores"].update({
        next(iter(response["groups"][0]["tag_scores"])): float("nan")}),
])
def test_invalid_model_ids_or_scores_are_rejected_with_trace(snapshot, corrupt):
    backend = Backend(corrupt=corrupt)
    with pytest.raises(NativeFrontendError) as caught:
        map_native_groups(build_index(snapshot), "herb", Config(), backend=backend)
    error = caught.value
    assert error.stage == "tag_scoring"
    assert len(error.frontend_calls) == len(error.frontend_outputs) == 2


def test_transport_error_has_no_retry_or_silent_fallback(snapshot):
    backend = Backend(fail="connection refused")
    with pytest.raises(NativeFrontendError, match="connection refused") as caught:
        map_native_groups(build_index(snapshot), "herb", Config(), backend=backend)
    assert caught.value.stage == "keywords"
    assert len(caught.value.frontend_calls) == 1
    assert caught.value.frontend_outputs == []


def test_unmapped_keyword_is_explicit_and_skips_tag_call(snapshot):
    backend = Backend(keywords(("astronaut", [])))
    query = map_native_groups(build_index(snapshot), "astronaut", Config(), backend=backend)
    assert len(query["frontend_calls"]) == 1
    assert query["frontend_request"]["status"] == "no_mapped_seeds"
    assert query["groups"][0]["seed_tags"] == []
    assert query["query_map"]["unresolved_spans"] == ["astronaut"]


@pytest.mark.parametrize("extraction,config,options", [
    (keywords(("herb", []), ("train", [])), Config(max_query_cues=1), NativeFrontendOptions()),
    (keywords(("herb", ["herbs", "seasoning"])), Config(), NativeFrontendOptions(max_alternatives=1)),
    (keywords(("what", [])), Config(), NativeFrontendOptions()),
    (keywords(("herb", ["the"])), Config(), NativeFrontendOptions()),
])
def test_invalid_keyword_or_extraction_budget_is_explicit(snapshot, extraction, config, options):
    backend = Backend(extraction)
    with pytest.raises(NativeFrontendError) as caught:
        map_native_groups(build_index(snapshot), "herb train", config, backend=backend, options=options)
    assert caught.value.stage == "keywords"
    assert len(caught.value.frontend_calls) == 1


def test_prompt_budget_blocks_call_before_transport(snapshot):
    backend = Backend()
    with pytest.raises(NativeFrontendError, match="character cap") as caught:
        map_native_groups(build_index(snapshot), "herb", Config(), backend=backend,
                          options=NativeFrontendOptions(max_prompt_chars=10))
    assert caught.value.frontend_calls == []
    assert backend.calls == []


def test_candidate_and_global_seed_caps_are_logged(snapshot):
    backend = Backend(keywords(("herbs", []), ("shop", []), ("train", [])))
    query = map_native_groups(build_index(snapshot), "herbs shop train", Config(max_seed_tags=1),
        backend=backend, options=NativeFrontendOptions(max_scored_tag_pairs=2))
    assert query["frontend_request"]["scored_tag_pairs"] == 2
    assert sum(len(group["offered_tag_ids"]) for group in query["groups"]) == 2
    assert len({t for group in query["groups"] for t in group["seed_tags"]}) == 1
    assert any(group["tag_truncations"] for group in query["groups"])
    assert query["groups"][-1]["seed_tags"] == []


def test_static_catalog_preserves_links_and_omits_scope_only_key(raw_memory):
    next(cue for cue in raw_memory["cues"] if cue["cue_id"] == "I")["scope_role"] = "scope"
    snapshot = canonicalize(raw_memory)
    original = deepcopy(snapshot)
    catalog = static_tag_catalog(build_index(snapshot))
    assert catalog["tags"] == {t["tag_id"]: t["text"] for t in snapshot["tags"]}
    assert "I" not in catalog["key_tags"]
    for key, tags in catalog["key_tags"].items():
        for tag, facts in tags.items():
            assert set(facts) == {link["fact_id"] for link in snapshot["links"]
                                  if link["cue_id"] == key and link["tag_id"] == tag}
    assert snapshot == original


def test_live_and_supplied_responses_cannot_mix(snapshot):
    backend = Backend()
    with pytest.raises(NativeFrontendError, match="Choose live or supplied"):
        map_native_groups(build_index(snapshot), "herb", Config(), backend=backend,
                          keyword_response=keywords(("herb", [])))
    assert not backend.calls
