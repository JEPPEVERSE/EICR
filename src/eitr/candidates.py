"""Bounded group-wise graph enumeration, then a frozen whole-action pool."""
from collections import defaultdict, deque
import heapq
from eicr.contracts import digest, require


def enumerate_graph(index, query, config, expand=True):
    require(query["index_hash"] == index.index_hash, "Query/index mismatch")
    active = [g for g in query["groups"] if g["seed_tags"]]
    queues, best, support, paths = {}, {}, {}, {}
    reached = {t for g in active for t in g["seed_tags"]}
    depths = {t: 0 for t in reached}
    for g in active:
        gid = g["group_id"]
        queues[gid] = [( -1., 0, t, (t,)) for t in g["seed_tags"]]
        heapq.heapify(queues[gid])
        best[gid] = {(t, 0): 1. for t in g["seed_tags"]}
        support[gid] = {t: 1. for t in g["seed_tags"]}
        paths[gid] = {t: [t] for t in g["seed_tags"]}
    trace, popped, evaluations, rejected = [], 0, 0, []
    stop = "exhausted"
    cycle = deque(g["group_id"] for g in active)
    while cycle and expand:
        gid = cycle.popleft()
        queue = queues[gid]
        if not queue:
            continue
        neg, depth, tag, path = heapq.heappop(queue)
        score = -neg
        if score < best[gid].get((tag, depth), -1):
            if queue: cycle.append(gid)
            continue
        if any(d < depth and t == tag and s >= score for (t, d), s in best[gid].items()):
            if queue: cycle.append(gid)
            continue
        if popped >= config.max_state_pops:
            stop = "state_cap"; heapq.heappush(queue, (neg, depth, tag, path)); break
        edges = index.neighbors[tag] if depth < config.max_hops else []
        if evaluations + len(edges) > config.max_edge_evaluations:
            stop = "edge_cap_full_row"; heapq.heappush(queue, (neg, depth, tag, path)); break
        popped += 1; evaluations += len(edges)
        trace.append({"group": gid, "tag": tag, "depth": depth, "reach": score, "path": list(path)})
        for edge in edges:
            target = edge["target"]
            if target in path:
                continue
            value = score * config.path_decay * edge["score"]
            if value <= 0:
                continue
            if target not in reached and len(reached) >= config.max_reached_tags:
                rejected.append({"group": gid, "from": tag, "target": target, "depth": depth+1,
                                 "reach": value, "path": list(path + (target,)), "reason": "tag_cap"})
                continue
            reached.add(target)
            depths[target] = min(depths.get(target, depth+1), depth+1)
            if value > support[gid].get(target, 0):
                support[gid][target] = value; paths[gid][target] = list(path + (target,))
            state = (target, depth+1)
            if value > best[gid].get(state, 0):
                best[gid][state] = value
                heapq.heappush(queue, (-value, depth+1, target, path + (target,)))
        if queue: cycle.append(gid)
    aggregate = {t: sum(g["support_weight"] * support[g["group_id"]].get(t, 0) for g in active)
                 for t in sorted(reached)}
    return {"support": aggregate, "group_support": support, "paths": paths, "depths": depths,
            "trace": trace, "state_pops": popped, "edge_evaluations": evaluations,
            "new_tag_rejections": len(rejected), "rejected_tags": rejected, "stop": stop,
            "frontier": [{"group": gid, "reach": -neg, "depth": d, "tag": t, "path": list(p)}
                         for gid in sorted(queues) for neg, d, t, p in sorted(queues[gid])] if expand else [],
            "remaining_states": sum(map(len, queues.values())) if expand else 0}


def candidate_pool(index, query, graph, corpus, whole_scores, config, mode="graph"):
    require(mode in ("graph", "seed", "flat", "strict_seed"), "Unknown pool mode")
    require(corpus.snapshot_hash == index.snapshot["content_hash"], "Corpus/index mismatch")
    facts = {f["fact_id"]: f for f in index.snapshot["facts"]}
    seed_facts = defaultdict(set)
    group_sources = {}
    for g in query["groups"]:
        origins = set()
        for t, fids in g["facts_by_tag"].items():
            seed_facts[t].update(fids)
            origins.update(o for fid in fids for o in facts[fid]["origins"])
        group_sources[g["group_id"]] = origins
    tag_facts = defaultdict(set)
    for fid, fact in facts.items():
        for t in fact["tag_ids"]:
            tag_facts[t].add(fid)
    legal_tags = [] if mode == "flat" else sorted(graph["support"], key=lambda t: (
        graph["depths"][t], -graph["support"][t], t))
    per_tag, atom_tags, truncations = {}, defaultdict(set), []
    reachable_sources, shortlisted_sources = set(), set()
    for t in legal_tags:
        fids = seed_facts[t] if t in seed_facts else tag_facts[t]
        origins = {o for fid in fids for o in facts[fid]["origins"]}
        reachable_sources.update(origins)
        ids = {a for o in origins for a in corpus.source_atoms[o]}
        for a in ids:
            atom_tags[a].add(t)
        ordered = sorted(ids, key=lambda a: (-whole_scores[a], a))
        per_tag[t] = ordered[:config.max_core_atoms_per_tag]
        shortlisted_sources.update(corpus.atoms[a]["source_id"] for a in per_tag[t])
        if len(ordered) > len(per_tag[t]):
            truncations.append({"tag": t, "reason": "per_tag_core_cap", "omitted_core_ids": ordered[len(per_tag[t]):]})
    direct_order = sorted((a for a in corpus.atoms if whole_scores[a] > 0), key=lambda a: (-whole_scores[a], a))
    if mode == "strict_seed": direct_order = []
    direct = direct_order[:config.direct_core_reservation]
    actions, atoms, omitted = {}, set(), []
    def admit(a):
        if a in actions: return
        context = set(corpus.contexts[a])
        if len(actions) >= config.max_core_atoms or len(atoms | context) > config.max_candidate_atoms:
            omitted.append({"core": a, "reason": "whole_action_candidate_cap"}); return
        tags = sorted(atom_tags[a])
        sid = corpus.atoms[a]["source_id"]
        actions[a] = {"action_id": a, "core": a, "atoms": sorted(context), "tags": tags,
                      "direct": a in direct,
                      "seed_groups": sorted(g for g, origins in group_sources.items() if sid in origins),
                      "support": max((graph["support"][t] for t in tags), default=0.)}
        atoms.update(context)
    for a in direct: admit(a)
    for pos in range(max(map(len, per_tag.values()), default=0)):
        for t in legal_tags:
            if pos < len(per_tag[t]): admit(per_tag[t][pos])
    result = {"actions": [actions[a] for a in sorted(actions)], "atom_ids": sorted(atoms),
              "tag_order": legal_tags, "mode": mode, "omitted": omitted,
              "per_tag_truncations": truncations,
              "direct_truncation": {"reason": "direct_core_cap", "positive_core_count": len(direct_order),
                                     "omitted_core_ids": direct_order[len(direct):]},
              "source_stages": {"seed": sorted({o for origins in group_sources.values() for o in origins}),
                  "graph_reachable": sorted(reachable_sources), "graph_shortlist": sorted(shortlisted_sources),
                  "direct_before_cap": sorted({corpus.atoms[a]["source_id"] for a in direct_order}),
                  "direct": sorted({corpus.atoms[a]["source_id"] for a in direct}),
                  "before_global_cap": sorted({corpus.atoms[c]["source_id"]
                       for a in set(direct) | {a for ids in per_tag.values() for a in ids} for c in corpus.contexts[a]}),
                  "candidate": sorted({corpus.atoms[a]["source_id"] for a in atoms})},
              "index_hash": index.index_hash,
              "query_hash": digest({k:query.get(k) for k in ("index_hash","question","groups","mapping_config_hash")}),
              "corpus_hash": corpus.identity,
              "graph_hash": digest(graph), "config_hash": digest(config.to_dict())}
    result["pool_hash"] = digest(result)
    return result
