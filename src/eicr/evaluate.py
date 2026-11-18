"""Offline paired outcome summaries; gold never enters online method functions."""

from collections import defaultdict

from .contracts import require


def validate_splits(rows):
    assignments = defaultdict(set)
    for row in rows:
        for kind, value in (("memory", row["memory_id"]), ("question", row["question_family_id"])):
            assignments[(kind, value)].add(row["split"])
    require(all(len(splits) == 1 for splits in assignments.values()), "Memory/question family leaks across splits")
    return {"status": "passed", "groups": len(assignments)}


def summarize(rows):
    """Aggregate externally judged correctness; do not substitute lexical overlap for QA truth."""
    buckets = defaultdict(list)
    seen = set()
    for row in rows:
        identity = (row["example_id"], row["method"])
        require(identity not in seen, "Duplicate scored outcome")
        seen.add(identity)
        require(type(row["correct"]) is bool, "Correctness must be an independent boolean judgment")
        buckets[(row["method"], row["condition"])].append(row)
    result = []
    for (method, condition), values in sorted(buckets.items()):
        result.append({"method": method, "condition": condition, "count": len(values),
                       "accuracy": sum(r["correct"] for r in values) / len(values),
                       "memory_count": len({r["memory_id"] for r in values})})
    return {"outcomes": result, "note": "Descriptive counts; no independent-sample significance or calibration claim."}
