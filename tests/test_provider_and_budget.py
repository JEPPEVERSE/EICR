import io
import json
import random
import urllib.error

import pytest

from eicr.contracts import ContractError, tokens
from eicr.packer import CharacterTokenizer, chunk_sources, pack_sources
from eicr.provider import ChatBackend


def test_provider_explicit_call_and_redacted_trace(monkeypatch):
    monkeypatch.setenv("EICR_TEST_KEY", "test-secret-do-not-log")
    requests=[]
    def fake_urlopen(request,timeout):
        requests.append(request)
        body={"choices":[{"finish_reason":"stop","message":{"content":json.dumps({"ok":True})}}],"usage":{"total_tokens":12}}
        return io.BytesIO(json.dumps(body).encode())
    monkeypatch.setattr("urllib.request.urlopen",fake_urlopen)
    backend=ChatBackend("https://example.test/v1","explicit-test-model",key_env="EICR_TEST_KEY")
    assert backend.calls==[]
    assert backend.chat_json("system","untrusted question")=={"ok":True}
    payload=json.loads(requests[0].data)
    assert payload["model"]=="explicit-test-model" and payload["max_tokens"]==2048
    assert "test-secret" not in json.dumps(backend.calls)
    assert backend.calls[0]["usage"]["total_tokens"]==12


def test_provider_error_does_not_expose_body(monkeypatch):
    monkeypatch.setenv("EICR_TEST_KEY","secret")
    def fail(*args,**kwargs):
        raise urllib.error.URLError("provider-response-with-secret")
    monkeypatch.setattr("urllib.request.urlopen",fail)
    backend=ChatBackend("https://example.test/v1","model",key_env="EICR_TEST_KEY")
    with pytest.raises(ContractError) as error:
        backend.chat_json("s","q")
    assert "provider-response" not in str(error.value)
    assert backend.calls[0]["status"]=="error"


def test_insecure_remote_endpoint_rejected():
    with pytest.raises(ContractError):
        ChatBackend("http://example.test/v1","m")
    with pytest.raises(ContractError):
        ChatBackend("https://user:secret@example.test/v1","m")


def test_budget_random_overlaps_and_source_integrity():
    rng=random.Random(18)
    tokenizer=CharacterTokenizer()
    sources=[{"source_id":"s"+str(i),"text":"abcdefghijklmnopqrstuvwxyz"*30} for i in range(3)]
    chunks=chunk_sources(sources,tokenizer,max_tokens=100,overlap_tokens=32)
    for _ in range(20):
        original=rng.sample(chunks,len(chunks))
        independent=rng.sample(chunks,len(chunks)//2)
        quota=rng.randint(120,900)
        packet=pack_sources({"original":original,"independent_cue":independent},sources,tokenizer,
                            {"original":quota,"independent_cue":quota},"random-fixed-seed")
        assert packet["used_tokens"]==tokenizer.count(packet["text"])<=2*quota
        by_id={s["source_id"]:s for s in sources}
        for span in packet["spans"]:
            assert by_id[span["source_id"]]["text"][span["start"]:span["end"]] in packet["text"]
        for first,second in zip(packet["spans"],packet["spans"][1:]):
            assert first["source_id"]!=second["source_id"] or first["end"]<second["start"]


def test_multilingual_chunking_and_lexical_units():
    text="我做了沙拉。然后加入一撮红椒粉。 Another sentence. "*20
    chunks=chunk_sources([{"source_id":"cn","text":text}],CharacterTokenizer(),max_tokens=50,overlap_tokens=10)
    assert all(len(c["text"])<=50 for c in chunks)
    assert chunks[0]["start"]==0 and chunks[-1]["end"]==len(text)
    assert all(a["end"]>=b["start"] for a,b in zip(chunks,chunks[1:]))
    assert tokens("我在做halloumi沙拉")==["我","在","做","halloumi","沙","拉"]
