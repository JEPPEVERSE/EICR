import copy

import pytest

from eicr import build_index
from eicr.contracts import ContractError
from eicr.diagnostics import diagnose
from eicr.packer import CharacterTokenizer, chunk_sources, pack_sources
from eicr.pipeline import answer_from_sources, run
from eicr.policy import choose_route, freeze_policy
from eicr.query_map import map_query
from eicr.retrieval import SharedRetriever, collect_candidates


def policy_for(index):
    return freeze_policy(index, {"tau_delta":0.1,"tau_core":0.3,"tau_margin":0.1,"tau_pair":0.1},
                         {"status":"development_selected","development_split_hash":"synthetic-test-only", "selection_rule":"unit-test-fixture",
                          "grouping":"memory-and-question-family","confirmation_untouched":True})


def test_single_check_and_default_disabled(snapshot):
    index = build_index(snapshot)
    query = map_query(index, "halloumi pinch paprika shopping")
    diagnostics = diagnose(index, query)
    assert choose_route(index, query, diagnostics)["mode"] == "ORIGINAL"
    route = choose_route(index, query, diagnostics, policy_for(index))
    assert route["mode"] == "SINGLE_CUE_CHECK"
    assert route["selected_cue_id"] == "shopping"
    assert sum(route["quotas"].values()) == 6000


def test_two_cues_symmetric_and_unknown_not_wrong(snapshot):
    index = build_index(snapshot)
    query = map_query(index, "halloumi shopping")
    assert choose_route(index, query, diagnose(index,query), policy_for(index))["mode"] == "SYMMETRIC_CHECK"
    mention = {"span_start":0,"span_end":7,"surface_text":"unknown"}
    query = map_query(index,"unknown halloumi", [mention])
    diagnostics = diagnose(index, query)
    assert diagnostics["unresolved_entries"]
    assert choose_route(index, query, diagnostics, policy_for(index))["mode"] == "ORIGINAL"


def test_containment_and_multi_target_do_not_select(snapshot):
    index = build_index(snapshot)
    query = map_query(index,"halloumi paprika")
    route = choose_route(index,query,diagnose(index,query),policy_for(index))
    assert route["reasons"] == ["CONTAINMENT_AMBIGUITY"]
    query = map_query(index,"halloumi pinch paprika shopping", query_shape="explicit_multi_target")
    assert choose_route(index,query,diagnose(index,query),policy_for(index))["mode"] == "ORIGINAL"


def test_identical_rows_not_extra_votes(snapshot):
    snapshot["cues"].append({"cue_id":"salad", "text":"salad"})
    snapshot["edges"] += [{"cue_id":"salad","evidence_id":"e0"},{"cue_id":"salad","evidence_id":"e1"}]
    index = build_index(snapshot)
    query = map_query(index,"halloumi salad halloumi shopping")
    diag = diagnose(index,query)
    assert diag["effective_groups"] == 2
    assert any(set(g["members"]) == {"halloumi","salad"} for g in diag["groups"])


def test_ambiguous_casefold_does_not_merge_identity(snapshot):
    snapshot["cues"].append({"cue_id":"H2","text":"HALLOUMI"})
    index = build_index(snapshot)
    query = map_query(index,"halloumi")
    assert query["mentions"][0]["status"] == "ambiguous"


def test_offset_validation(snapshot):
    index = build_index(snapshot)
    with pytest.raises(ContractError, match="offset"):
        map_query(index,"halloumi",[{"span_start":0,"span_end":2,"surface_text":"halloumi"}])


def test_packer_merges_selected_only_and_charges_uncovered():
    source = {"source_id":"s","text":"x"*500}
    chunks = [{"span_id":str(start),"source_id":"s","start":start,"end":start+100,"text":"x"*100} for start in range(0,401,80)]
    packet = pack_sources({"original":chunks},[source],CharacterTokenizer(),{"original":250},"q")
    assert packet["used_tokens"] <= 250
    assert 100 < packet["spans"][0]["end"] < 500
    assert sum(entry["new_rendered_tokens"] for entry in packet["ledger"]) == packet["used_tokens"]
    assert packet["omitted"]


def test_duplicate_text_counted_once():
    source = {"source_id":"s", "text":"a"*60}
    span={"span_id":"a","source_id":"s","start":0,"end":60,"text":source["text"]}
    packet=pack_sources({"original":[span],"core":[span]},[source],CharacterTokenizer(),{"original":100,"core":100},"q")
    assert packet["spans"][0]["queues"] == ["core","original"]
    assert packet["ledger"][-1]["new_rendered_tokens"] == 0


def test_independent_queue_not_conditioned_on_core(snapshot):
    index=build_index(snapshot)
    query=map_query(index,"halloumi pinch paprika shopping",answer_need="salad")
    diag=diagnose(index,query)
    route=choose_route(index,query,diag,policy_for(index))
    retriever=SharedRetriever(chunk_sources(index.records["sources"],CharacterTokenizer()))
    queues,trace=collect_candidates(index,query,diag,route,retriever)
    assert any("snacks" in s["text"] for s in queues["independent_cue"])
    assert trace["retriever"] == "bm25_only"


def test_pipeline_no_api_and_citation_validation(snapshot):
    index=build_index(snapshot)
    result=run(index,"halloumi pinch paprika shopping",CharacterTokenizer(),policy=policy_for(index))
    assert result["answer"] is None
    assert result["evidence_pack"]["used_tokens"]<=6000
    class Backend:
        def chat_json(self,*args):
            return {"answer_text":"wrong","citations":[{"source_id":"absent","start":0,"end":2}], "unsupported_premises":[],"resolution_status":"supported"}
    with pytest.raises(ContractError, match="outside"):
        answer_from_sources("q",result["evidence_pack"],Backend())


def test_wrong_policy_index_falls_back(snapshot):
    index=build_index(snapshot)
    policy=policy_for(index)
    policy["index_hash"]="different"
    query=map_query(index,"halloumi shopping")
    assert choose_route(index,query,diagnose(index,query),policy)["reasons"]==["POLICY_INCOMPATIBLE"]
