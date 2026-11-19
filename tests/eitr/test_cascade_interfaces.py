"""Public integration contracts for semantic ranking and the native-style frontend."""
from copy import deepcopy
import json

import pytest

from eicr.contracts import ContractError, digest, read_json, write_json
from eicr.packer import CharacterTokenizer
from eitr.accounting import cost_ledger, cost_report
from eitr.batch import run_batch
from eitr.cascade import CASCADE_ARMS
from eitr.cli import main, parser
from eitr.config import Config
from eitr.evidence import Corpus
from eitr.experiment import benchmark, freeze, model_backend, replay, run_protocol, verify_protocol
from eitr.index import build_index, save_index
from eitr.native_frontend import NativeFrontendError
from eitr.pipeline import compare, retrieve
from eitr.rerank import backend_from_spec


class Reranker:
    def __init__(self, identity=None):
        self.identity = identity or {"provider": "fake-interface-test", "model": "test"}
        self.calls = []

    def score(self, question, documents):
        self.calls.append({"status": "ok", "billing_kind": "rerank_tokens",
                           "usage": {"input_tokens": 31, "output_tokens": 0}})
        return [.9 if "pinch" in text else .2 for text in documents]


class NativeFront:
    def __init__(self, *, invalid_score=False):
        self.calls = []
        self.invalid_score = invalid_score

    def chat_json(self, system, user):
        self.calls.append({"status": "ok", "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
        payload = json.loads(user)
        if "max_keywords" in payload:
            return {"question_time": "", "keywords": [{"id": "herb", "alternatives": ["herbs"]}]}
        return {"groups": [{"group_id": group["group_id"], "tag_scores": {
            tag["tag_id"]: 2. if self.invalid_score else .8 for tag in group["tags"]
        }} for group in payload["groups"]]}


class Answer:
    def __init__(self): self.calls = []

    def chat_json(self, system, user):
        self.calls.append({"status": "ok", "usage": {"prompt_tokens": 11, "completion_tokens": 2}})
        return {"answer_text": "herbs", "citations": [], "unsupported_premises": [],
                "resolution_status": "insufficient"}


def question(snapshot, *, supplied=False):
    return {"question_id": "q", "owner_id": snapshot["owner_id"], "snapshot_id": snapshot["snapshot_id"],
            "question": "Which herb was used?", **({"mentions": []} if supplied else {})}


def reranker_spec(tmp_path):
    return {"provider": "dashscope", "endpoint": "https://example.invalid/rerank", "model": "test-reranker",
            "key_env": "EITR_TEST_UNUSED", "max_documents": 500,
            "cache_dir": str(tmp_path / "rerank-cache"), "cache_only": True}


def frozen_spec(snapshot, tmp_path):
    index = build_index(snapshot)
    tok = CharacterTokenizer()
    corpus = Corpus(snapshot, tok, Config())
    save_index(index, tmp_path / "index.json")
    corpus.save(tmp_path / "corpus.json")
    write_json(tmp_path / "questions.json", [question(snapshot, supplied=True)])
    spec = {"config": Config().to_dict(), "arms": list(CASCADE_ARMS), "tokenizer": {"kind": "characters"},
            "assets": [{"index": str(tmp_path / "index.json"), "corpus": str(tmp_path / "corpus.json")}],
            "questions": str(tmp_path / "questions.json"), "native": None,
            "model": {"api_base": "https://example.invalid", "model": "test", "key_env": "EITR_TEST_UNUSED",
                      "max_output_tokens": 20},
            "frontend": "supplied", "generate": False, "require_dates": False, "prices": None}
    return spec, index, tok, corpus


@pytest.mark.parametrize("arm", CASCADE_ARMS)
def test_parser_exposes_cascade_and_native_options(arm):
    args = parser().parse_args(["retrieve", "--index", "index.json", "--out", "out.json",
        "--question", "herb", "--tokenizer", "json", "--arm", arm, "--native-frontend",
        "--reranker-spec", "rerank.json"])
    assert args.arm == arm and args.native_frontend and args.reranker_spec == "rerank.json"


@pytest.mark.parametrize("model_flags,thinking,json_mode", [
    ([], None, True), (["--no-enable-thinking", "--no-json-mode"], False, False),
])
def test_cli_routes_native_frontend_and_semantic_backend(snapshot, tmp_path, monkeypatch,
                                                        model_flags, thinking, json_mode):
    import eitr.cli as cli
    import eitr.rerank as rerank_module
    index = build_index(snapshot)
    save_index(index, tmp_path / "index.json")
    spec = reranker_spec(tmp_path)
    spec["cache_only"] = False
    write_json(tmp_path / "reranker.json", spec)
    front, semantic = NativeFront(), Reranker()
    constructors = []
    def make_front(*args, **kwargs):
        constructors.append(kwargs)
        return front
    monkeypatch.setattr(cli, "ChatBackend", make_front)
    # No model tokenizer or model is downloaded: this test verifies routing.
    monkeypatch.setattr(cli, "tokenizer_from", lambda spec: CharacterTokenizer())
    monkeypatch.setattr(rerank_module, "backend_from_spec", lambda settings: semantic)
    output = tmp_path / "run.json"
    assert main(["retrieve", "--index", str(tmp_path / "index.json"), "--out", str(output),
        "--question", "Which herb was used?", "--tokenizer", "test-contract", "--arm", "C_rerank",
        "--native-frontend", "--reranker-spec", str(tmp_path / "reranker.json"),
        "--api-base", "https://example.invalid", "--model", "test"] + model_flags) == 0
    row = read_json(output)
    assert row["query"]["frontend_request"]["mode"] == "native_style_v1"
    assert row["query"]["groups"][0]["key_ids"] == ["herbs"]
    assert row["manifest"]["frontend_mode"] == "mragent"
    assert len(row["calls"]["frontend"]) == len(front.calls) == 2
    assert row["calls"]["rerank"] == semantic.calls
    assert row["semantic_rerank"]["request"]["backend"] == semantic.identity
    assert constructors == [{"max_output_tokens": 2048, "temperature": 0,
                             "enable_thinking": thinking, "json_mode": json_mode}]


def test_cli_semantic_arm_requires_explicit_backend(snapshot, tmp_path, capsys):
    save_index(build_index(snapshot), tmp_path / "index.json")
    assert main(["retrieve", "--index", str(tmp_path / "index.json"), "--out", str(tmp_path / "out.json"),
        "--question", "herbs", "--tokenizer", "characters", "--arm", "C_mmr"]) == 2
    assert "reranker-spec" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_cli_judge_profile_reaches_actual_grading(snapshot, tmp_path, monkeypatch):
    import eitr.cli as cli
    from eitr.evaluate import MRAGENT_ACCURACY_PROMPT
    index = build_index(snapshot)
    save_index(index, tmp_path / "index.json")
    run_batch(index, [question(snapshot)], CharacterTokenizer(), config=Config(), arms=["P3"],
              answer_backend=Answer(), output_dir=tmp_path / "batch")
    write_json(tmp_path / "labels.json", [{"question_id": "q", "owner_id": snapshot["owner_id"],
        "snapshot_id": snapshot["snapshot_id"], "references": ["herbs"], "source_ids": ["D1:2"]}])
    class Judge:
        def __init__(self): self.calls = []; self.prompts = []
        def chat_json(self, system, user):
            self.prompts.append(user)
            self.calls.append({"status": "ok"})
            return {"label": "CORRECT"}
    judge = Judge()
    monkeypatch.setattr(cli, "ChatBackend", lambda *args, **kwargs: judge)
    assert main(["grade", "--batch", str(tmp_path / "batch"), "--labels", str(tmp_path / "labels.json"),
        "--index", str(tmp_path / "index.json"), "--out", str(tmp_path / "grades"),
        "--api-base", "https://example.invalid", "--model", "test", "--judge-profile", "mragent_lenient"]) == 0
    assert judge.prompts == [MRAGENT_ACCURACY_PROMPT.format(question="Which herb was used?",
                                                         gold_answer="herbs", generated_answer="herbs")]
    assert read_json(tmp_path / "grades" / "manifest.json")["judge_profile"]["name"] == "mragent_lenient"


def test_frozen_semantic_protocol_requires_and_uses_explicit_reranker(snapshot, tmp_path):
    spec, index, tok, corpus = frozen_spec(snapshot, tmp_path)
    with pytest.raises(ContractError, match="frozen reranker"):
        freeze(spec)
    spec["reranker"] = reranker_spec(tmp_path)
    backend = Reranker(backend_from_spec(spec["reranker"]).identity)
    compare(index, question(snapshot)["question"], tok, config=Config(), arms=CASCADE_ARMS,
            mentions=[], corpus=corpus, reranker=backend, rerank_cache_dir=spec["reranker"]["cache_dir"])
    protocol = freeze(spec)
    assert verify_protocol(protocol) == protocol
    journal = run_protocol(protocol, tmp_path / "experiment")
    assert journal["status"] == "complete"
    batch_dir = tmp_path / "experiment" / journal["completed"][0]["batch"]
    manifest = read_json(batch_dir / "manifest.json")
    rows = read_json(batch_dir / manifest["completed"][0]["file"])["rows"]
    assert {row["arm"] for row in rows} == set(CASCADE_ARMS)
    assert all(row["semantic_rerank"]["cache_reused"] for row in rows)
    assert all(row["semantic_rerank"]["request"]["backend"] == backend.identity for row in rows)
    assert all(row["calls"]["rerank"] for row in rows)


@pytest.mark.parametrize("model_options", [{}, {"enable_thinking": False, "json_mode": False}])
def test_frozen_native_frontend_reaches_batch_without_literal_mentions(snapshot, tmp_path, monkeypatch,
                                                                     model_options):
    import eitr.experiment as experiment
    spec, index, tok, corpus = frozen_spec(snapshot, tmp_path)
    spec.update(frontend="mragent", tokenizer={"kind": "test-native-tokenizer"}, arms=["C_rerank"])
    spec["model"].update(model_options)
    spec["reranker"] = reranker_spec(tmp_path)
    write_json(tmp_path / "questions.json", [question(snapshot)])
    front = NativeFront()
    constructors = []
    def make_front(*args, **kwargs):
        constructors.append(kwargs)
        return front
    # The fake backends remain fully offline; only public protocol wiring is tested.
    monkeypatch.setattr(experiment, "tokenizer_from", lambda settings: tok)
    monkeypatch.setattr(experiment, "ChatBackend", make_front)
    backend = Reranker(backend_from_spec(spec["reranker"]).identity)
    compare(index, question(snapshot)["question"], tok, config=Config(), arms=["C_rerank"], corpus=corpus,
            frontend_backend=NativeFront(), frontend_mode="mragent", reranker=backend,
            rerank_cache_dir=spec["reranker"]["cache_dir"])
    protocol = freeze(spec)
    journal = run_protocol(protocol, tmp_path / "native-experiment")
    batch_dir = tmp_path / "native-experiment" / journal["completed"][0]["batch"]
    manifest = read_json(batch_dir / "manifest.json")
    row = read_json(batch_dir / manifest["completed"][0]["file"])["rows"][0]
    assert manifest["frontend_mode"] == row["manifest"]["frontend_mode"] == "mragent"
    assert row["query"]["frontend_request"]["mode"] == "native_style_v1"
    assert row["calls"]["frontend"] == front.calls and len(front.calls) == 2
    assert row["semantic_rerank"]["cache_reused"]
    assert constructors == [{"max_output_tokens": 20, "temperature": 0,
                             "enable_thinking": model_options.get("enable_thinking"),
                             "json_mode": model_options.get("json_mode", True)}]


def test_model_option_defaults_match_existing_chat_and_tool_backends():
    from eicr.provider import ChatBackend
    from eitr.tool_backend import ToolBackend
    model = {"api_base": "https://example.invalid", "model": "test", "key_env": "EITR_TEST_UNUSED",
             "max_output_tokens": 20}
    old = ChatBackend(model["api_base"], model["model"], model["key_env"], max_output_tokens=20, temperature=0)
    current = model_backend(model)
    assert current.generation_options == old.generation_options == {"temperature": 0}
    assert current.json_mode is old.json_mode is True
    assert current.max_output_tokens == old.max_output_tokens == 20
    explicit = dict(model, enable_thinking=False, json_mode=False)
    for cls in (ChatBackend, ToolBackend):
        backend = model_backend(explicit, cls)
        assert backend.generation_options == {"temperature": 0, "enable_thinking": False}
        assert backend.json_mode is False
        assert not backend.calls


def test_freeze_pins_generation_options_and_detects_changes(snapshot, tmp_path):
    spec, _, _, _ = frozen_spec(snapshot, tmp_path)
    spec["arms"] = ["P3"]
    original = freeze(deepcopy(spec))
    spec["model"].update(enable_thinking=False, json_mode=False)
    explicit = freeze(deepcopy(spec))
    assert explicit["spec"]["model"]["enable_thinking"] is False
    assert explicit["spec"]["model"]["json_mode"] is False
    assert original["protocol_hash"] != explicit["protocol_hash"]
    assert verify_protocol(explicit) == explicit
    changed = deepcopy(explicit)
    changed["spec"]["model"]["json_mode"] = True
    with pytest.raises(ContractError, match="checksum"):
        verify_protocol(changed)


@pytest.mark.parametrize("field,value", [
    ("enable_thinking", "false"), ("enable_thinking", 0),
    ("json_mode", "false"), ("json_mode", 0), ("json_mode", None),
])
def test_freeze_rejects_invalid_generation_option_types(snapshot, tmp_path, field, value):
    spec, _, _, _ = frozen_spec(snapshot, tmp_path)
    spec["arms"] = ["P3"]
    spec["model"][field] = value
    with pytest.raises(ContractError, match=field + " must be a boolean"):
        freeze(spec)


@pytest.mark.parametrize("arm", CASCADE_ARMS)
def test_semantic_replay_verifies_values_and_original_inputs(snapshot, arm):
    index, cfg, tok = build_index(snapshot), Config(), CharacterTokenizer()
    corpus = Corpus(snapshot, tok, cfg)
    row = retrieve(index, "herbs salad", tok, config=cfg, corpus=corpus, arm=arm, reranker=Reranker())
    assert replay(row, index, corpus)["verified"]
    changed = deepcopy(row)
    aid = next(iter(changed["relevance"]["whole_query"]))
    changed["relevance"]["whole_query"][aid] = .1234
    with pytest.raises(ContractError, match="semantic input/score"):
        replay(changed, index, corpus)
    changed = deepcopy(row)
    changed["semantic_rerank"]["scores"][aid] = .1234
    with pytest.raises(ContractError, match="semantic input/score"):
        replay(changed, index, corpus)
    changed = deepcopy(row)
    # Even a correctly rehashed trace must still match the actual rerank input.
    changed["semantic_rerank"]["request"]["documents_hash"] = digest("changed input")
    sem = changed["semantic_rerank"]
    sem["record_hash"] = digest({key: sem[key] for key in ("request", "scores", "calls")})
    with pytest.raises(ContractError, match="semantic input/score"):
        replay(changed, index, corpus)
    changed = deepcopy(row)
    changed["question"] = "different question"
    with pytest.raises(ContractError, match="semantic input/score"):
        replay(changed, index, corpus)


def test_benchmark_cache_replay_retains_cost_but_is_not_deployment_latency(snapshot, tmp_path):
    index, cfg, tok = build_index(snapshot), Config(), CharacterTokenizer()
    rows = benchmark(index, [question(snapshot, supplied=True)], tok, config=cfg, arms=["C_rerank"],
        corpus=Corpus(snapshot, tok, cfg), repeats=2, reranker=Reranker(),
        rerank_cache_dir=tmp_path / "cache", answer_backend=Answer())
    assert [row["semantic_rerank"]["cache_reused"] for row in rows] == [False, True]
    assert [row["benchmark"]["independent_request"] for row in rows] == [True, False]
    assert cost_ledger(rows[0])["independent_deployment"]["total_tokens"] == 44
    assert cost_ledger(rows[1])["independent_deployment"]["total_tokens"] == 44
    report = cost_report(rows)["arms"]["C_rerank"]
    assert report["independent_latency_runs"] == report["end_to_end_latency_runs"] == 1
    assert report["retrieval_p50_seconds"] == rows[0]["timing_seconds"]["retrieval_total"]
    # Older records may have the wrong independent flag; explicit cache status wins.
    rows[1]["benchmark"]["independent_request"] = True
    report = cost_report(rows)["arms"]["C_rerank"]
    assert report["independent_latency_runs"] == report["end_to_end_latency_runs"] == 1


def test_batch_native_failure_preserves_both_paid_frontend_calls(snapshot, tmp_path):
    front, reranker = NativeFront(invalid_score=True), Reranker()
    with pytest.raises(NativeFrontendError) as caught:
        run_batch(build_index(snapshot), [question(snapshot)], CharacterTokenizer(), config=Config(),
                  arms=["C_rerank"], frontend_backend=front, frontend_mode="mragent", reranker=reranker,
                  output_dir=tmp_path / "failed")
    manifest = read_json(tmp_path / "failed" / "manifest.json")
    assert manifest["status"] == "failed" and manifest["error_type"] == "NativeFrontendError"
    assert manifest["failure_call_traces"]["frontend"] == caught.value.frontend_calls == front.calls
    assert len(front.calls) == 2 and not reranker.calls
    assert cost_ledger({"calls": manifest["failure_call_traces"]})["independent_deployment"]["total_tokens"] == 20
