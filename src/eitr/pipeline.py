"""No-network by default orchestration and paired retrieval comparisons."""
from time import perf_counter
from pathlib import Path
from eicr.contracts import digest, require
from eicr.pipeline import answer_from_sources
from eicr.provider import ANSWER_PROMPT, FRONTEND_PROMPT
from .config import Config
from .query import map_groups, mapping_config_hash
from .candidates import enumerate_graph, candidate_pool
from .evidence import Corpus
from .scheduler import Scheduler, ARMS, build_interaction_matrix
from .accounting import Meter, environment_identity

from .cascade import CASCADE_ARMS, select_cascade
from .rerank import rerank_pool

RUN_ARMS = ARMS + CASCADE_ARMS + ("P5",)


def make_query(index, question, config, mentions=None, backend=None, frontend_mode="literal"):
    require(frontend_mode in ("literal", "mragent"), "Unknown frontend mode")
    if frontend_mode == "mragent":
        require(backend is not None and mentions is None, "MRAgent frontend needs its explicit backend, without literal mentions")
        from .native_frontend import map_native_groups
        return map_native_groups(index, question, config, backend=backend)
    return map_groups(index, question, config, mentions, backend)



def runtime_source_hash():
    # Matching, packing and model contracts are shared with the legacy package.
    # Pin both packages so changes to those dependencies invalidate a run identity.
    root = Path(__file__).parent.parent
    return digest({str(p.relative_to(root)): digest(p.read_text())
                   for package in ("eitr", "eicr")
                   for p in sorted((root / package).glob("*.py"))})


def retrieve(index, question, tokenizer, *, config=None, arm="P4", mentions=None,
             frontend_backend=None, answer_backend=None, query=None, corpus=None, score_cache=None,
             frontend_mode="literal", reranker=None, rerank_cache_dir=None, rerank_cache_only=False):
    config = config or Config()
    meter = Meter()
    require(arm in ARMS + CASCADE_ARMS, "Use the separately labelled explorer runner for an LLM control")
    require(arm not in CASCADE_ARMS or reranker is not None, "Semantic arms require an explicit reranker")
    require(config.top_k == index.top_k, "Graph top_k differs from run configuration")
    times, started = {}, perf_counter()
    score_cache = {} if score_cache is None else score_cache
    actual_score_queries = 0
    if corpus is None:
        corpus = Corpus(index.snapshot, tokenizer, config)
    require(corpus.snapshot_hash == index.snapshot["content_hash"] and corpus.tokenizer.identity == tokenizer.identity
            and corpus.atom_config == (config.atom_tokens, config.context_each_side),
            "Corpus scope or tokenizer mismatch")
    times["corpus_setup"] = perf_counter()-started
    def score(text):
        nonlocal actual_score_queries
        key = (corpus.identity, text)
        if key not in score_cache:
            score_cache[key] = corpus.scores(text)
            actual_score_queries += 1
        return score_cache[key]
    stage = perf_counter()
    if arm == "P0":
        require(frontend_backend is None, "Flat retrieval must not invoke an unnecessary frontend")
        query = map_groups(index, question, config, [])
        frontend_shared = False
    elif query is None:
        query = make_query(index, question, config, mentions, frontend_backend, frontend_mode)
        frontend_shared = False
    else:
        require(query["index_hash"] == index.index_hash and query["question"] == question, "Shared query mismatch")
        require(query.get("mapping_config_hash") == mapping_config_hash(config), "Shared frontend config mismatch")
        require(frontend_backend is None and mentions is None, "Shared query cannot override frontend")
        frontend_shared = True
    times["frontend"] = perf_counter()-stage
    has_seeds = any(g["seed_tags"] for g in query["groups"])
    mode = "flat" if arm == "P0" or (not has_seeds and arm != "strict_seed") else (
        "strict_seed" if arm == "strict_seed" else "seed" if arm == "P1" else "graph")
    stage = perf_counter()
    if arm.startswith("C_ppr"):
        from .diffusion import enumerate_ppr
        graph = enumerate_ppr(index, query, config, expand=mode == "graph",
            restart_probability=config.ppr_restart_probability, tolerance=config.ppr_tolerance,
            max_iterations=config.ppr_max_iterations, max_edge_evaluations=config.ppr_max_edge_evaluations)
    else:
        graph = enumerate_graph(index, query, config, expand=mode == "graph")
    times["graph"] = perf_counter()-stage
    stage = perf_counter()
    whole = score(question)
    times["whole_query_scoring"] = perf_counter()-stage
    stage = perf_counter()
    pool = candidate_pool(index, query, graph, corpus, whole, config, mode)
    times["candidate_pool"] = perf_counter()-stage
    stage = perf_counter()
    # Do not charge the cheap control for cue queries it never uses.
    needs_coverage = arm in ("P4", "P4_no_support", "P4_active") and has_seeds
    cue_scores = [score(g["text"]) for g in query["groups"]] if needs_coverage else []
    cue_interactions = (build_interaction_matrix(index, query, config.interaction_top_edges)
                        if arm == "P4_active" and has_seeds else [])
    deploy_score_queries = len({question} | ({g["text"] for g in query["groups"]} if needs_coverage else set()))
    times["cue_scoring"] = perf_counter()-stage
    stage = perf_counter()
    semantic = None
    selection_scores = whole
    if arm in CASCADE_ARMS:
        require(reranker is not None, "Semantic arms require an explicit reranker")
        semantic_key = ("semantic", corpus.identity, question, digest(pool["atom_ids"]), digest(reranker.identity))
        if semantic_key in score_cache:
            semantic = {**score_cache[semantic_key], "cache_reused": True}
        else:
            semantic = rerank_pool(question, corpus, pool, reranker, cache_dir=rerank_cache_dir, cache_only=rerank_cache_only)
            score_cache[semantic_key] = semantic
        selection_scores = semantic["scores"]
        times["semantic_rerank"] = perf_counter()-stage
        stage = perf_counter()
        packet = select_cascade(corpus, pool, selection_scores, config, arm, has_seeds)
    else:
        packet = Scheduler(corpus, pool, whole, cue_scores, config,
                           interaction_matrix=cue_interactions).select(arm, has_seeds)
    times["selection"] = perf_counter()-stage
    times["retrieval_total"] = perf_counter()-started
    answer_start = len(answer_backend.calls) if answer_backend else 0
    stage = perf_counter()
    answer = answer_from_sources(question, packet, answer_backend) if answer_backend else None
    times["answer"] = perf_counter()-stage
    return {"schema": "eitr-run-1", "arm": arm, "question": question,
            "index_hash": index.index_hash, "snapshot_hash": index.snapshot["content_hash"],
            "query": query, "graph": graph, "candidates": pool, "evidence_pack": packet,
            "semantic_rerank": semantic,
            "relevance": {"whole_query": {a:selection_scores[a] for a in pool["atom_ids"]},
                          "candidate_bm25": {a:whole[a] for a in pool["atom_ids"]},
                          "cue_groups": [{"group_id":g["group_id"], "text":g["text"],
                                          "scores":{a:values[a] for a in pool["atom_ids"]}}
                                         for g, values in zip(query["groups"], cue_scores)],
                          "interaction_matrix": cue_interactions,
                          "normalization": "single_request_semantic_scores" if semantic else "max_nonnegative_over_full_legal_snapshot"},
            "answer": answer, "calls": {"frontend": query["frontend_calls"], "exploration": [],
                **({"rerank": semantic["calls"]} if semantic else {}),
                "answer": list(answer_backend.calls[answer_start:]) if answer_backend else []},
            "timing_seconds": times,
            "resources": meter.result(),
            "work": {"raw_atom_count": len(corpus.atoms), "bm25_queries": deploy_score_queries,
                     "bm25_actual_queries_this_replay": actual_score_queries,
                     "bm25_document_score_evaluations": len(corpus.atoms)*deploy_score_queries,
                     "graph_edge_evaluations": graph["edge_evaluations"],
                     "scheduler_action_evaluations": packet["action_evaluations"]},
            "manifest": {"config": config.to_dict(), "config_hash": digest(config.to_dict()),
                "tokenizer": tokenizer.identity, "corpus_hash": corpus.identity,
                "corpus_cache_hash": corpus.cache_hash,
                "frontend_shared": frontend_shared, "frontend_mode": frontend_mode,
                "frontend_cost_note": "Shared frontend traces represent one deploy-time extraction per request; setup reuse is separate.",
                "answer_prompt_hash": digest(ANSWER_PROMPT),
                "frontend_prompt_hash": query.get("frontend_request", {}).get("prompt_hash", digest(FRONTEND_PROMPT)),
                "status": "answered" if answer else "retrieval_only",
                "runtime_source_hash": runtime_source_hash(),
                "environment": environment_identity(),
                "cost_unit_warning": "character tokenizer is test-only; final packet budget excludes model prompt/output tokens"}}


def compare(index, question, tokenizer, *, config=None, arms=("P0", "P1", "P2", "P3", "P4"),
            mentions=None, frontend_backend=None, corpus=None, native_runtime=None, exploration_backend=None,
            frontend_mode="literal", reranker=None, rerank_cache_dir=None, rerank_cache_only=False):
    config = config or Config()
    require(arms and len(set(arms)) == len(arms) and set(arms) <= set(RUN_ARMS), "Invalid comparison arms")
    require(not set(arms) & set(CASCADE_ARMS) or reranker is not None, "Semantic arms require an explicit reranker")
    require(config.top_k == index.top_k, "Graph top_k differs from run configuration")
    if "P5" in arms:
        require(native_runtime is not None and exploration_backend is not None
                and "native_auxiliary" in index.snapshot, "P5 requires pinned native runtime, backend and auxiliary graph")
    start = perf_counter()
    corpus = corpus or Corpus(index.snapshot, tokenizer, config)
    require(corpus.snapshot_hash == index.snapshot["content_hash"] and corpus.tokenizer.identity == tokenizer.identity
            and corpus.atom_config == (config.atom_tokens, config.context_each_side), "Shared corpus/config mismatch")
    query = make_query(index, question, config, mentions, frontend_backend, frontend_mode) if any(a != "P0" for a in arms) else None
    shared_seconds = perf_counter()-start
    rows, score_cache = [], {}
    for arm in arms:
        if arm == "P5":
            from .controlled import run_controlled
            rows.append(run_controlled(index, question, tokenizer, native_runtime, exploration_backend,
                                       config=config, query=query, corpus=corpus))
        else:
            rows.append(retrieve(index, question, tokenizer, config=config, arm=arm, query=query if arm != "P0" else None, corpus=corpus, score_cache=score_cache, frontend_mode=frontend_mode,
                reranker=reranker, rerank_cache_dir=rerank_cache_dir, rerank_cache_only=rerank_cache_only))
    matched = [r for r in rows if r["arm"] in ("P2", "P3", "P4", "P4_no_support", "P4_no_coverage", "P4_active", "C_rerank", "C_mmr")]
    require(len({r["candidates"]["pool_hash"] for r in matched}) <= 1, "Scheduler arms saw different candidates")
    return {"schema": "eitr-comparison-1", "question": question, "index_hash": index.index_hash,
            "shared_setup_seconds": shared_seconds, "frontend_calls_once_in_experiment": query["frontend_calls"] if query is not None else [],
            "rows": rows, "same_pool_verified": True,
            "same_pool_arms": [r["arm"] for r in matched],
            "candidate_changing_arms": [r["arm"] for r in rows if r["arm"].startswith("C_ppr")],
            "timing_note": "Per-arm durations exclude shared setup/frontend; do not present these as independent end-to-end latency."}
