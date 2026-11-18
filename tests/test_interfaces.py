import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from eicr import IndexConfig, build_index
from eicr.contracts import ContractError, digest
from eicr.evaluate import summarize, validate_splits
from eicr.packer import CharacterTokenizer, chunk_sources
from eicr.pipeline import run
from eicr.retrieval import SharedRetriever


def test_permutation_invariant(snapshot):
    a=build_index(snapshot,IndexConfig(structure=True,lambda_=0.2))
    for key in ("cues","sources","edges","evidence"):
        snapshot[key].reverse()
    b=build_index(snapshot,IndexConfig(structure=True,lambda_=0.2))
    assert a.index_hash == b.index_hash
    for x in a.cue_ids:
        for y in a.cue_ids:
            assert a.similarity(x,y) == b.similarity(x,y)


def test_hub_topic_excluded_whole(snapshot):
    for e in snapshot["evidence"]:
        e["topics"]=["hub"]
    index=build_index(snapshot,IndexConfig(structure=True,lambda_=0.2))
    assert not index.report["structure_active"]
    assert index.report["excluded_hub_topics"] == 1
    assert (index.T.toarray() == np.eye(len(index.fact_ids))).all()


def test_empty_index():
    snapshot={"owner_id":"empty","snapshot_id":"v1","sources":[],"cues":[],"evidence":[],"edges":[],
              "provenance":{"extraction_prefix_verified":True,"allowed_source_ids":[]}}
    index=build_index(snapshot)
    result=run(index,"anything",CharacterTokenizer())
    assert result["route"]["mode"]=="ORIGINAL"
    assert result["evidence_pack"]["used_tokens"]==0


def test_dense_manifest_alignment(snapshot):
    index=build_index(snapshot)
    tokenizer=CharacterTokenizer()
    chunks=chunk_sources(index.records["sources"],tokenizer)
    vectors=np.ones((len(chunks),3))
    manifest={"index_hash":index.index_hash,"tokenizer":tokenizer.identity,"chunk_order_hash":digest([c["span_id"] for c in chunks]),
              "question_hash":digest("halloumi"),"dense_identity":"synthetic-vectors"}
    out=run(index,"halloumi",tokenizer,dense_vectors=vectors,query_vector=[1,1,1],dense_identity="synthetic-vectors",dense_manifest=manifest)
    assert out["candidate_trace"]["retriever"]=="dense_bm25_rrf"
    manifest["question_hash"]="other-question"
    with pytest.raises(ContractError,match="aligned"):
        run(index,"halloumi",tokenizer,dense_vectors=vectors,query_vector=[1,1,1],dense_identity="synthetic-vectors",dense_manifest=manifest)


def test_frontend_format_retry_and_safe_fallback(snapshot):
    class BrokenFrontend:
        calls=[]
        def chat_json(self,*args):
            self.calls.append({"status":"invalid-format"})
            return {"mentions":[{}]}
    frontend=BrokenFrontend()
    result=run(build_index(snapshot),"halloumi",CharacterTokenizer(),frontend_backend=frontend)
    assert len(frontend.calls)==2
    assert result["query_map"]["frontend_error"]
    assert result["route"]["mode"]=="ORIGINAL"


def test_split_leakage_and_independent_outcome_counts():
    with pytest.raises(ContractError,match="leaks"):
        validate_splits([{"memory_id":"m","question_family_id":"q1","split":"dev"},
                         {"memory_id":"m","question_family_id":"q2","split":"test"}])
    row={"example_id":"x","method":"original","condition":"clean","correct":True,"memory_id":"m"}
    assert summarize([row])["outcomes"][0]["accuracy"]==1
    with pytest.raises(ContractError,match="Duplicate"):
        summarize([row,row])


def test_cli_smoke_and_failure_exit(snapshot,tmp_path):
    source=tmp_path/"snapshot.json"
    source.write_text(json.dumps(snapshot))
    path=tmp_path/"index"
    def cli(*args):
        return subprocess.run([sys.executable,"-m","eicr",*args],capture_output=True,text=True)
    built=cli("build","--snapshot",str(source),"--out",str(path))
    assert built.returncode==0,built.stderr
    assert cli("audit-index","--index",str(path)).returncode==0
    result=tmp_path/"result.json"
    executed=cli("run","--index",str(path),"--question","halloumi paprika","--tokenizer","characters","--out",str(result))
    assert executed.returncode==0,executed.stderr
    assert json.loads(result.read_text())["run_manifest"]["status"]=="evidence_only"
    failed=cli("run","--index",str(path),"--question","q","--tokenizer","characters","--out",str(result))
    assert failed.returncode==2
