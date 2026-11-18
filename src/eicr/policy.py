"""Frozen development-selected thresholds govern bounded checks; default is off."""

from copy import deepcopy

from .contracts import digest, finite_unit, require

POLICY_SCHEMA = "eicr-policy-2"

QUOTAS = {
    "ORIGINAL": {"original": 6000},
    "SYMMETRIC_CHECK": {"original": 4000, "side_a": 1000, "side_b": 1000},
    "SINGLE_CUE_CHECK": {"original": 3000, "independent_cue": 1000, "core": 1000, "answer_need": 1000}
}


def freeze_policy(index, thresholds, selection_manifest):
    require(isinstance(thresholds, dict) and isinstance(selection_manifest, dict), "Policy inputs must be objects")
    require(selection_manifest.get("status") == "development_selected", "Policy needs a development-selection manifest")
    for key in ("development_split_hash", "selection_rule", "grouping", "confirmation_untouched"):
        require(bool(selection_manifest.get(key)), "Missing policy provenance: " + key)
    require(selection_manifest["confirmation_untouched"] is True, "Confirmation set must remain untouched")
    require(set(thresholds) == {"tau_delta", "tau_core", "tau_margin", "tau_pair"}, "Incomplete policy thresholds")
    for key, value in thresholds.items():
        finite_unit(value, key)
    value = {"schema": POLICY_SCHEMA, "enabled": True, "index_hash": index.index_hash,
             "thresholds": deepcopy(thresholds), "selection_manifest": deepcopy(selection_manifest)}
    return {**value, "policy_hash": digest(value)}


def choose_route(index, query_map, diagnostics, policy=None):
    require(diagnostics["index_hash"] == index.index_hash == query_map["index_hash"], "Incompatible route inputs")
    require(diagnostics.get("query_map_hash") == digest(query_map), "Query map does not match diagnostics")
    result = {"question_id": query_map["question_id"], "snapshot_id": query_map["snapshot_id"],
              "index_hash": index.index_hash, "policy_hash": None, "mode": "ORIGINAL", "selected_cue_id": None,
              "query_map_hash": digest(query_map), "diagnostics_hash": digest(diagnostics),
              "group_ids": [], "reasons": [], "observed_mapping_coverage": diagnostics["observed_mapping_coverage"]}
    def finish(reason):
        result["reasons"].append(reason)
        result["quotas"] = dict(QUOTAS[result["mode"]])
        result["evidence_budget_tokens"] = 6000
        return result
    if policy is not None and not isinstance(policy, dict):
        return finish("POLICY_INCOMPATIBLE")
    if not policy or policy.get("enabled") is False:
        return finish("POLICY_DISABLED")
    if policy.get("enabled") is not True:
        return finish("POLICY_INCOMPATIBLE")
    body = {k: v for k, v in policy.items() if k != "policy_hash"}
    if policy.get("schema") != POLICY_SCHEMA or policy.get("index_hash") != index.index_hash or digest(body) != policy.get("policy_hash"):
        return finish("POLICY_INCOMPATIBLE")
    result["policy_hash"] = policy["policy_hash"]
    thresholds = policy.get("thresholds", {})
    if not isinstance(thresholds, dict):
        return finish("POLICY_INCOMPATIBLE")
    if any(thresholds.get(k) is None for k in ("tau_delta", "tau_core", "tau_margin", "tau_pair")):
        return finish("POLICY_UNCONFIGURED")
    if set(thresholds) != {"tau_delta", "tau_core", "tau_margin", "tau_pair"}:
        return finish("POLICY_INCOMPATIBLE")
    try:
        for key, value in thresholds.items():
            finite_unit(value, key)
    except ValueError:
        return finish("POLICY_INCOMPATIBLE")
    if query_map.get("parse_truncated") or query_map.get("frontend_error") or diagnostics["unresolved_entries"]:
        return finish("INCOMPLETE_MAPPING")
    if query_map["query_shape"] == "explicit_multi_target":
        return finish("MULTI_TARGET")
    groups = diagnostics["groups"]
    if len(groups) < 2:
        return finish("INSUFFICIENT_GROUPS")
    def contained(pair):
        return pair["cover_i_j"] >= 1 - 1e-10 or pair["cover_j_i"] >= 1 - 1e-10
    if len(groups) == 2:
        pair = diagnostics["pairs"][0]
        if contained(pair):
            return finish("CONTAINMENT_AMBIGUITY")
        if pair["combined"] < thresholds["tau_pair"]:
            result.update(mode="SYMMETRIC_CHECK", group_ids=[0, 1])
            return finish("PAIR_NEEDS_SYMMETRIC_CHECK")
        return finish("NO_CHECK_SIGNAL")
    scores = sorted(diagnostics["scores"], key=lambda s: -s["check_priority"])
    first, second = scores[:2]
    gap = first["check_priority"] - second["check_priority"]
    if gap <= 1e-10:
        return finish("TIED_CHECK_PRIORITY")
    i = first["group"]
    if len(groups[i]["members"]) > 1:
        return finish("CANNOT_LOCALIZE_IDENTICAL_ROWS")
    if any(contained(pair) for pair in diagnostics["pairs"] if pair["i"] == i or pair["j"] == i):
        return finish("CONTAINMENT_AMBIGUITY")
    if first["check_priority"] < thresholds["tau_delta"] or first["cohesion"] < thresholds["tau_core"] or gap < thresholds["tau_margin"]:
        return finish("NO_CHECK_SIGNAL")
    result.update(mode="SINGLE_CUE_CHECK", selected_cue_id=groups[i]["representative"], group_ids=[i])
    return finish("SINGLE_CUE_NEEDS_CHECK")
