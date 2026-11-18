"""Evidence association diagnostics, not an event or truth classifier."""

from collections import defaultdict

from .contracts import digest, require


def diagnose(index, query_map):
    require(query_map["index_hash"] == index.index_hash, "Query/index mismatch")
    observed, unresolved = [], []
    for mention in query_map["mentions"]:
        if mention["role"] != "entry":
            continue
        if mention["status"] != "unique":
            unresolved.append(mention["surface_text"])
            continue
        cid = mention["candidate_cue_ids"][0]
        require(cid in index.cue_pos, "Query references an unknown cue")
        i = index.cue_pos[cid]
        cue = index.records["cues"][i]
        if cue.get("scope_role") == "scope":
            continue
        if index.norm_x[i] == 0:
            unresolved.append(mention["surface_text"])
        elif cid not in observed:
            observed.append(cid)
    groups_by_row = defaultdict(list)
    profiles = []
    total = float(index.weights.sum())
    for cid in sorted(observed):
        i = index.cue_pos[cid]
        omega = max(0.0, 1 - float(index.mass[i]) / total) if total else 0.0
        row = index.R[i]
        profiles.append({"cue_id": cid, "mass": float(index.mass[i]), "weight": omega,
                         "facts": row.nnz, "families": len({index.records["evidence"][j]["family_id"] for j in row.indices}),
                         "tag_mass": index.facet_profile(cid)})
        if omega > 1e-12:
            groups_by_row[tuple(row.indices)].append(cid)
    groups = []
    profile_map = {p["cue_id"]: p for p in profiles}
    for members in sorted(groups_by_row.values(), key=lambda ids: ids[0]):
        groups.append({"members": members, "representative": members[0], "weight": max(profile_map[c]["weight"] for c in members)})
    pairs = []
    for i, a in enumerate(groups):
        for j in range(i + 1, len(groups)):
            b = groups[j]
            pairs.append({"i": i, "j": j, **index.similarity(a["representative"], b["representative"]),
                          "cover_i_j": index.coverage(a["representative"], b["representative"]),
                          "cover_j_i": index.coverage(b["representative"], a["representative"]),
                          "contributions": index.explain(a["representative"], b["representative"])})
    scores = []
    if len(groups) >= 3:
        for i, group in enumerate(groups):
            remaining = [p for p in pairs if p["i"] != i and p["j"] != i]
            den = sum(groups[p["i"]]["weight"] * groups[p["j"]]["weight"] for p in remaining)
            cohesion = sum(groups[p["i"]]["weight"] * groups[p["j"]]["weight"] * p["combined"] for p in remaining) / den
            peers = [p for p in pairs if p["i"] == i or p["j"] == i]
            affinity = sum(groups[p["j"] if p["i"] == i else p["i"]]["weight"] * p["combined"] for p in peers) / sum(groups[j]["weight"] for j in range(len(groups)) if j != i)
            core_ids = [g["representative"] for j, g in enumerate(groups) if j != i]
            scores.append({"group": i, "cohesion": cohesion, "affinity": affinity, "check_priority": cohesion - affinity,
                           "core_joint": index.joint(core_ids)})
    joint = index.joint([g["representative"] for g in groups])
    flags = []
    if joint["mass"] == 0:
        flags.append("NO_JOINT_WITNESS")
    if any(len(g["members"]) > 1 for g in groups):
        flags.append("IDENTICAL_ROWS_GROUPED")
    entries = [m for m in query_map["mentions"] if m["role"] == "entry"]
    coverage = sum(m["status"] == "unique" for m in entries) / len(entries) if entries else None
    return {"index_hash": index.index_hash, "query_map_hash": digest(query_map),
            "profiles": profiles, "groups": groups, "pairs": pairs,
            "scores": scores, "joint": joint, "flags": flags, "unresolved_entries": unresolved,
            "observed_mapping_coverage": coverage, "effective_groups": len(groups),
            "association_is_probability": False}
