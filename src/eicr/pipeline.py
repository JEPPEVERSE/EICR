"""Executable orchestration: map, diagnose, route, retrieve, pack, optionally answer."""

import json

from .contracts import ContractError, digest, require
from .diagnostics import diagnose
from .packer import chunk_sources, pack_sources
from .policy import choose_route
from .provider import ANSWER_PROMPT, FRONTEND_PROMPT
from .query_map import map_query
from .retrieval import SharedRetriever, collect_candidates


def answer_from_sources(question, packet, backend):
    result = backend.chat_json(ANSWER_PROMPT, json.dumps({"question": question, "sources": packet["text"], "allowed_spans": packet["spans"]}, ensure_ascii=False))
    require(isinstance(result.get("answer_text"), str), "Answer lacks text")
    require(isinstance(result.get("citations"), list) and isinstance(result.get("unsupported_premises"), list), "Invalid answer lists")
    require(result.get("resolution_status") in ("supported", "conflicting", "insufficient"), "Invalid resolution status")
    for citation in result["citations"]:
        require(all(k in citation for k in ("source_id", "start", "end")), "Citation lacks span")
        require(type(citation["start"]) is int and type(citation["end"]) is int and citation["start"] < citation["end"], "Invalid citation coordinates")
        require(any(s["source_id"] == citation["source_id"] and s["start"] <= citation["start"] < citation["end"] <= s["end"] for s in packet["spans"]), "Citation is outside visible evidence")
    if result["resolution_status"] == "supported":
        require(bool(result["citations"]), "Supported answer requires a visible citation")
    return result


def run(index, question, tokenizer, *, mentions=None, answer_need="", query_shape="unknown", question_id=None,
        policy=None, frontend_backend=None, answer_backend=None, dense_vectors=None, query_vector=None, dense_identity=None,
        dense_manifest=None):
    frontend_error = None
    frontend_call_start = len(frontend_backend.calls) if frontend_backend else 0
    if frontend_backend is not None:
        for attempt in range(2):
            try:
                parsed = frontend_backend.chat_json(FRONTEND_PROMPT, json.dumps({"question": question}, ensure_ascii=False))
                require(isinstance(parsed.get("mentions"), list), "Frontend lacks mentions")
                require(isinstance(parsed.get("answer_need", ""), str), "Invalid frontend answer_need")
                query = map_query(index, question, parsed["mentions"], question_id=question_id,
                                  answer_need=parsed.get("answer_need", ""), query_shape=parsed.get("query_shape", "unknown"))
                query["frontend_basis"] = "explicit_llm_spans"
                break
            except (ContractError, KeyError, TypeError) as error:
                frontend_error = type(error).__name__
        else:
            query = map_query(index, question, [], question_id=question_id)
            query["frontend_error"] = frontend_error
    else:
        query = map_query(index, question, mentions, question_id=question_id, answer_need=answer_need, query_shape=query_shape)
    frontend_calls = list(frontend_backend.calls[frontend_call_start:]) if frontend_backend else []
    diagnostics = diagnose(index, query)
    route = choose_route(index, query, diagnostics, policy)
    chunks = chunk_sources(index.records["sources"], tokenizer)
    if dense_vectors is not None:
        require(bool(dense_manifest), "Dense vectors require a frozen alignment manifest")
        expected = {"index_hash": index.index_hash, "tokenizer": tokenizer.identity,
                    "chunk_order_hash": digest([c["span_id"] for c in chunks]),
                    "question_hash": digest(question), "dense_identity": dense_identity}
        require(all(dense_manifest.get(k) == value for k, value in expected.items()), "Dense bundle is not aligned with index/chunks/question")
    retriever = SharedRetriever(chunks, dense_vectors, dense_identity)
    queues, retrieval_trace = collect_candidates(index, query, diagnostics, route, retriever, query_vector)
    packet = pack_sources(queues, index.records["sources"], tokenizer, route["quotas"], query["question_id"])
    answer_call_start = len(answer_backend.calls) if answer_backend else 0
    answer = answer_from_sources(question, packet, answer_backend) if answer_backend is not None else None
    calls = {"frontend": frontend_calls,
             "answer": list(answer_backend.calls[answer_call_start:]) if answer_backend else []}
    return {"schema": "eicr-run-1", "index_hash": index.index_hash, "query_map": query,
            "diagnostics": diagnostics, "route": route, "candidate_trace": retrieval_trace,
            "candidates": queues, "evidence_pack": packet, "answer": answer, "calls": calls,
            "run_manifest": {"tokenizer": tokenizer.identity, "answer_prompt_hash": digest(ANSWER_PROMPT),
                             "frontend_prompt_hash": digest(FRONTEND_PROMPT),
                             "dense_manifest": dense_manifest,
                             "status": "answered" if answer else "evidence_only"}}
