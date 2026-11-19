"""Post-retrieval diagnostics only. Gold data never enters candidate generation."""
from collections import defaultdict
from statistics import mean, median
from eicr.contracts import require


def source_metrics(run, gold_source_ids, corpus=None):
    gold = set(gold_source_ids)
    visible = {s["source_id"] for s in run["evidence_pack"]["spans"]}
    result = {"gold_count": len(gold), "packet_source_recall": len(gold & visible)/len(gold) if gold else None,
              "all_required_sources": gold <= visible if gold else None}
    if corpus is not None and "candidates" in run:
        candidates = {corpus.atoms[a]["source_id"] for a in run["candidates"]["atom_ids"]}
        result["candidate_source_recall"] = len(gold & candidates)/len(gold) if gold else None
    return result


def summarize(rows):
    require(isinstance(rows, list) and rows, "Need scored rows")
    grouped, seen = defaultdict(list), set()
    for row in rows:
        require(all(k in row for k in ("question_id", "owner_id", "arm", "correct", "online_tokens")), "Invalid scored row")
        key = (row["owner_id"], row["question_id"], row["arm"])
        require(key not in seen, "Aggregate repeated samples per question before summary")
        require(type(row["correct"]) is bool and type(row["online_tokens"]) is int and row["online_tokens"] >= 0,
                "Invalid outcome or token count")
        seen.add(key); grouped[row["arm"]].append(row)
    summary = {}
    for arm, items in grouped.items():
        by_owner = defaultdict(list)
        for r in items: by_owner[r["owner_id"]].append(r["correct"])
        summary[arm] = {"questions": len(items), "accuracy": mean(r["correct"] for r in items),
                        "mean_online_tokens": mean(r["online_tokens"] for r in items),
                        "median_online_tokens": median(r["online_tokens"] for r in items),
                        "owner_accuracy": {o: mean(v) for o, v in by_owner.items()}}
    pairs = {}
    arms = sorted(grouped)
    for i, a in enumerate(arms):
        left = {(r["owner_id"], r["question_id"]): r for r in grouped[a]}
        for b in arms[i+1:]:
            right = {(r["owner_id"], r["question_id"]): r for r in grouped[b]}
            common = sorted(left.keys() & right.keys())
            pairs[a+" vs "+b] = {"paired_questions": len(common),
                "a_wins": sum(left[k]["correct"] and not right[k]["correct"] for k in common),
                "b_wins": sum(right[k]["correct"] and not left[k]["correct"] for k in common),
                "unpaired_left": len(left.keys()-right.keys()), "unpaired_right": len(right.keys()-left.keys())}
    return {"arms": summary, "pairs": pairs,
            "statistical_claim": "Descriptive paired diagnostics only; no noninferiority or acceptance-probability claim."}
