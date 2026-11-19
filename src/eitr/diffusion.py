"""Optional, bounded personalized PageRank for tag candidate generation.

This is a standard graph retrieval control, not an answer-probability model.
It changes candidate generation; it is not a reranker of a frozen evidence pool.
Only admitted seed tags personalize the walk. The existing candidate_pool must
continue enforcing each seed's exact key/tag fact scope when it reads evidence.
"""
from collections import defaultdict
import math

from eicr.contracts import require


def enumerate_ppr(index, query, config, expand=True, *, restart_probability=0.15,
                  tolerance=1e-8, max_iterations=100,
                  max_edge_evaluations=50_000_000):
    """Return enumerate_graph-compatible fields plus an explicit candidate_order.

    For each active cue group, p is uniform over its admitted seeds and
    x_next = restart*p + (1-restart)*(P.T*x + dangling_mass*p).
    Work uses sparse dictionaries and visited adjacency rows, never an N x N
    dense matrix. A budget that cannot fund a whole synchronous iteration stops
    before that iteration, leaving the previous probability vector intact.

    edge_evaluations counts row-normalization edge inspections plus propagation
    operations (one per nonzero group/source/outgoing-edge in each iteration).
    state_pops counts group/source states propagated, including dangling nodes.
    Convergence means the contraction bound on every group's stationary L1
    error is <= tolerance, not merely that the top-k order stopped changing.

    Output support/group_support contain *unrenormalized PPR mass* for admitted
    output tags. They are not the old max-path support. The depth-prioritized
    default candidate_pool ordering is not a PPR rank: callers should explicitly
    consume candidate_order. paths/depths are deterministic discovery witnesses,
    not a decomposition of PPR mass or independent evidentiary support.
    """
    require(query.get("index_hash") == index.index_hash, "Query/index mismatch")
    for field in ("owner_id", "snapshot_id"):
        require(field not in query or query[field] == index.snapshot[field],
                "PPR query scope mismatch")
    require(type(expand) is bool, "Invalid expansion flag")
    require(type(restart_probability) in (int, float)
            and math.isfinite(restart_probability) and 0 < restart_probability <= 1
            and 1. - restart_probability < 1.,
            "Invalid PPR restart probability")
    require(type(tolerance) in (int, float) and math.isfinite(tolerance)
            and tolerance > 0, "Invalid PPR tolerance")
    require(type(max_iterations) is int and max_iterations > 0,
            "Invalid PPR iteration cap")
    require(type(max_edge_evaluations) is int and max_edge_evaluations >= 0,
            "Invalid PPR edge cap")
    require(type(config.max_reached_tags) is int and config.max_reached_tags > 0,
            "Invalid PPR output tag cap")
    tags = {t["tag_id"] for t in index.snapshot["tags"]}
    require(len(tags) == len(index.snapshot["tags"]) and set(index.neighbors) == tags,
            "PPR graph tag scope mismatch")
    groups = query.get("groups")
    require(isinstance(groups, list), "Missing PPR query groups")
    active, seen = [], set()
    for group in groups:
        require(isinstance(group, dict), "Invalid PPR group")
        gid = group.get("group_id")
        require(isinstance(gid, str) and gid and gid not in seen, "Invalid PPR group identity")
        seen.add(gid)
        seeds = group.get("seed_tags")
        require(isinstance(seeds, list) and all(isinstance(t, str) for t in seeds)
                and len(set(seeds)) == len(seeds) and set(seeds) <= tags,
                "PPR seed outside graph scope or duplicated")
        if "candidate_tags" in group:
            require(set(seeds) <= set(group["candidate_tags"]), "PPR seed was not admitted")
        if "facts_by_tag" in group:
            require(set(group["facts_by_tag"]) == set(seeds), "PPR seed fact scope mismatch")
        weight = group.get("support_weight", 0.)
        require(type(weight) in (int, float) and math.isfinite(weight)
                and (0 < weight <= 1 if seeds else weight == 0), "Invalid PPR group weight")
        if seeds:
            active.append((gid, sorted(seeds), float(weight)))
    active.sort()
    require(not active or math.isclose(math.fsum(w for _, _, w in active), 1.,
                                      rel_tol=0., abs_tol=1e-12), "PPR group weights must sum to one")
    seed_set = {t for _, seeds, _ in active for t in seeds}
    require(len(seed_set) <= config.max_reached_tags, "PPR output cap cannot hold all admitted seeds")
    personal = {gid: {t: 1. / len(seeds) for t in seeds} for gid, seeds, _ in active}
    scores = {gid: dict(p) for gid, p in personal.items()}
    paths = {gid: {t: [t] for t in seeds} for gid, seeds, _ in active}
    transition, trace = {}, []
    normalization_work = propagation_work = state_work = 0
    last_delta = error_bound = None
    converged = False
    stop = "no_seeds" if not active else "seed_only" if not expand else "iteration_cap"
    alpha = 1. - restart_probability
    for iteration in range(1, max_iterations + 1) if active and expand else ():
        states = [(gid, tag, value) for gid in sorted(scores)
                  for tag, value in sorted(scores[gid].items()) if value > 0]
        needed = sorted({tag for _, tag, _ in states} - transition.keys()) if alpha else []
        require(all(isinstance(index.neighbors[tag], list) for tag in needed), "Invalid PPR adjacency row")
        # Degree lookup is not an edge scan; no truncated stochastic rows.
        normalize_cost = sum(len(index.neighbors[tag]) for tag in needed)
        propagate_cost = sum(len(index.neighbors[tag]) for _, tag, _ in states) if alpha else 0
        if normalization_work + propagation_work + normalize_cost + propagate_cost > max_edge_evaluations:
            stop = "edge_cap_before_iteration"
            break
        for tag in needed:
            edges, targets = [], set()
            for edge in index.neighbors[tag]:
                target, weight = edge.get("target"), edge.get("score")
                require(isinstance(target, str) and target in tags and target != tag and target not in targets
                        and type(weight) in (int, float) and math.isfinite(weight)
                        and 0 < weight <= 1, "Invalid or out-of-scope PPR edge")
                targets.add(target)
                edges.append((target, float(weight)))
            edges.sort()
            total = math.fsum(w for _, w in edges)
            transition[tag] = [(target, weight / total) for target, weight in edges]
        normalization_work += normalize_cost
        nxt = {gid: defaultdict(float, {tag: restart_probability * value for tag, value in p.items()})
               for gid, p in personal.items()}
        dangling = defaultdict(float)
        for gid, source, value in states:
            if not alpha:
                continue
            if not transition[source]:
                dangling[gid] += value
            elif alpha:
                for target, probability in transition[source]:
                    mass = alpha * value * probability
                    if mass:
                        nxt[gid][target] += mass
                        if target not in paths[gid]:
                            paths[gid][target] = paths[gid][source] + [target]
        for gid, mass in dangling.items():
            for seed, probability in personal[gid].items():
                nxt[gid][seed] += alpha * mass * probability
        propagation_work += propagate_cost
        state_work += len(states)
        last_delta = max(math.fsum(abs(nxt[gid].get(tag, 0.) - scores[gid].get(tag, 0.))
                                    for tag in sorted(nxt[gid].keys() | scores[gid].keys()))
                         for gid in scores)
        # The PageRank map contracts L1 by alpha, including dangling repair.
        error_bound = min(2., alpha / restart_probability * last_delta)
        scores = {gid: dict(sorted(values.items())) for gid, values in nxt.items()}
        require(all(math.isclose(math.fsum(v.values()), 1., rel_tol=0., abs_tol=1e-10)
                    for v in scores.values()), "PPR probability mass was lost")
        trace.append({"iteration": iteration, "max_group_l1_step": last_delta,
                      "stationary_l1_error_bound": error_bound, "states": len(states),
                      "normalization_edges": normalize_cost, "propagation_edges": propagate_cost})
        if error_bound <= tolerance:
            stop, converged = "converged", True
            break
    aggregate = defaultdict(float)
    for gid, _, weight in active:
        for tag, value in scores[gid].items():
            aggregate[tag] += weight * value
    all_ranked = sorted(aggregate, key=lambda tag: (-aggregate[tag], tag))
    extras = [t for t in all_ranked if t not in seed_set]
    retained = seed_set | set(extras[:config.max_reached_tags - len(seed_set)])
    order = [t for t in all_ranked if t in retained]
    output_paths = {gid: {t: path for t, path in p.items() if t in retained} for gid, p in paths.items()}
    depths = {tag: min(len(p[tag]) - 1 for p in paths.values() if tag in p) for tag in retained}
    rejected = [{"target": tag, "reach": aggregate[tag], "reason": "ppr_output_tag_cap"}
                for tag in all_ranked if tag not in retained]
    return {"support": {t: aggregate[t] for t in sorted(retained)},
            "group_support": {gid: {t: values[t] for t in sorted(retained & values.keys())}
                              for gid, values in scores.items()},
            "paths": output_paths, "depths": depths, "candidate_order": order,
            "trace": trace, "state_pops": state_work,
            "edge_evaluations": normalization_work + propagation_work,
            "new_tag_rejections": len(rejected), "rejected_tags": rejected,
            "stop": stop, "frontier": [], "remaining_states": 0,
            "policy": "personalized_pagerank", "owner_id": index.snapshot["owner_id"],
            "snapshot_id": index.snapshot["snapshot_id"], "index_hash": index.index_hash,
            "score_semantics": "unrenormalized_stationary_mass_not_path_support",
            "ppr": {"restart_probability": restart_probability, "tolerance": tolerance,
                    "max_iterations": max_iterations, "max_edge_evaluations": max_edge_evaluations,
                    "iterations": len(trace), "converged": converged,
                    "max_group_l1_step": last_delta, "stationary_l1_error_bound": error_bound,
                    "normalization_edge_evaluations": normalization_work,
                    "propagation_edge_evaluations": propagation_work,
                    "walk_reached_tags": len(aggregate), "retained_mass": math.fsum(aggregate[t] for t in retained),
                    "personalization": personal,
                    "depth_note": "discovery distance; PPR is not bounded by config.max_hops",
                    "frontier_note": "power iteration has no path frontier; incomplete status is in converged/stop"}}


def reciprocal_rank_fusion(rankings, *, rank_constant=60., weights=None, limit=None):
    """Fuse explicitly supplied, same-scope ID rankings; no retriever is invoked.

    RRF score is sum(weight / (rank_constant + 1-based rank)). Duplicates within
    one ranking are rejected so a channel cannot vote twice for the same item.
    Missing IDs contribute zero. Ties are resolved by ID. This helper is optional
    and does not manufacture a dense ranking or silently alter PPR candidates.
    """
    require(isinstance(rankings, (list, tuple)), "RRF rankings must be a sequence")
    require(type(rank_constant) in (int, float) and math.isfinite(rank_constant)
            and rank_constant > 0, "Invalid RRF rank constant")
    require(limit is None or type(limit) is int and limit >= 0, "Invalid RRF limit")
    weights = [1.] * len(rankings) if weights is None else weights
    require(isinstance(weights, (list, tuple)) and len(weights) == len(rankings)
            and all(type(w) in (int, float) and math.isfinite(w) and w >= 0 for w in weights),
            "Invalid RRF weights")
    scores, ranks = defaultdict(float), {}
    for channel, (ranking, weight) in enumerate(zip(rankings, weights)):
        require(isinstance(ranking, (list, tuple)) and all(isinstance(item, str) and item for item in ranking)
                and len(ranking) == len(set(ranking)), "Invalid or duplicated RRF ranking IDs")
        for rank, item in enumerate(ranking, 1):
            if weight == 0:
                continue
            scores[item] += weight / (rank_constant + rank)
            ranks.setdefault(item, [None] * len(rankings))[channel] = rank
    return [{"id": item, "score": scores[item], "ranks": ranks[item]}
            for item in sorted(scores, key=lambda item: (-scores[item], item))[:limit]]
