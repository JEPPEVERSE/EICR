"""Bounded one-step evidence neighborhoods with absolute weak-edge attenuation."""

import math
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, eye


def build_transition(records, config):
    facts = records["evidence"]
    n = len(facts)
    positions = {f["evidence_id"]: i for i, f in enumerate(facts)}
    identity = eye(n, format="csr", dtype=np.float64)
    if not config.structure:
        return identity, identity, [], {"structure_active": False, "eligible_topics": 0, "pruned_candidates": 0}
    members = defaultdict(set)
    for i, fact in enumerate(facts):
        for topic in set(fact.get("topics", [])):
            members[topic].add(i)
    bound = min(config.topic_max_df, max(2, math.floor(config.topic_max_fraction * n)))
    eligible = {t for t, ids in members.items() if config.topic_min_df <= len(ids) <= bound}
    retained = []
    ignored_topics = 0
    for fact in facts:
        possible = sorted(set(fact.get("topics", [])) & eligible, key=lambda t: (len(members[t]), t))
        retained.append(set(possible[:config.max_topics]))
        ignored_topics += max(0, len(possible) - config.max_topics)
    idf = {t: math.log((n + 1) / (len(members[t]) + 1)) + 1 for t in eligible}
    explicit = defaultdict(dict)
    for link in records.get("context_links", []):
        if link["kind"] in config.explicit_link_types:
            i, j = positions[link["from"]], positions[link["to"]]
            if i != j:
                explicit[i].setdefault(j, []).append(link)
    rows, cols, values, trace = [], [], [], []
    pruned = 0
    for i in range(n):
        candidates = set()
        for topic in retained[i]:
            candidates.update(members[topic])
        explicit_targets = sorted(explicit[i], key=lambda j: facts[j]["evidence_id"])
        candidates.update(explicit_targets[:config.max_explicit_candidates])
        permitted_explicit = set(explicit_targets[:config.max_explicit_candidates])
        pruned += max(0, len(explicit_targets) - config.max_explicit_candidates)
        candidates.discard(i)
        scores = []
        for j in candidates:
            common = retained[i] & retained[j]
            union = retained[i] | retained[j]
            jac = sum(idf[t] for t in common) / sum(idf[t] for t in union) if union else 0.0
            is_explicit = j in permitted_explicit
            strength = max(1.0 if is_explicit else 0.0, config.topic_strength * jac)
            if strength >= config.min_strength and strength > 0:
                scores.append((j, strength, sorted(common), is_explicit))
        scores.sort(key=lambda item: (-item[1], facts[item[0]]["evidence_id"]))
        keep = scores[:config.neighbors]
        pruned += len(candidates) - len(keep)
        r = max((s[1] for s in keep), default=0.0)
        total = sum(s[1] for s in keep)
        rows.append(i); cols.append(i); values.append(1.0 - r)
        for rank, (j, strength, common, is_explicit) in enumerate(scores):
            selected = rank < config.neighbors
            q = strength / total if selected else None
            if selected:
                rows.append(i); cols.append(j); values.append(r * q)
            trace.append({"from": facts[i]["evidence_id"], "to": facts[j]["evidence_id"],
                          "g": strength, "q": q, "r": r, "retained": selected,
                          "topic_ids": common, "edge_kind": "explicit_context" if is_explicit else "native_topic",
                          "links": explicit[i].get(j, []) if is_explicit else []})
    p = csr_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float64)
    p.eliminate_zeros()
    transition = ((1 - config.beta) * identity + config.beta * p).tocsr()
    active = any(row["retained"] for row in trace)
    return p, transition, trace, {
        "structure_active": active, "eligible_topics": len(eligible),
        "topic_df_bound": bound, "excluded_hub_topics": sum(len(ids) > bound for ids in members.values()),
        "ignored_topic_memberships": ignored_topics, "pruned_candidates": pruned,
        "expanded_evidence_rows": sum(p.indptr[i + 1] - p.indptr[i] > 1 for i in range(n))
    }
