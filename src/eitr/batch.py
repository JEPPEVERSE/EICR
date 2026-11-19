"""Frozen unlabeled batch runner with per-question output and paired packet reuse."""
from pathlib import Path
from time import perf_counter
from eicr.contracts import digest, require, write_json
from eicr.pipeline import answer_from_sources
from .pipeline import compare


def run_batch(index, questions, tokenizer, *, config, arms, frontend_backend=None, answer_backend=None, output_dir):
    require(isinstance(questions, list) and questions, "Questions must be a nonempty list")
    allowed = {"question_id", "owner_id", "snapshot_id", "question", "mentions"}
    ids = set()
    for q in questions:
        require(isinstance(q, dict) and set(q) <= allowed, "Question contains unknown fields (gold must stay outside runtime)")
        require(isinstance(q.get("question_id"), str) and q["question_id"] and q["question_id"] not in ids,
                "Missing or duplicate question ID")
        require(q.get("owner_id") == index.snapshot["owner_id"] and q.get("snapshot_id") == index.snapshot["snapshot_id"],
                "Question owner/snapshot mismatch")
        require(isinstance(q.get("question"), str) and q["question"].strip(), "Invalid question")
        require(frontend_backend is None or "mentions" not in q, "Cannot mix live and supplied mentions")
        ids.add(q["question_id"])
    out = Path(output_dir)
    require(not out.exists(), "Batch output exists; never silently overwrite/resume")
    out.mkdir(parents=True)
    manifest = {"schema": "eitr-batch-1", "status": "running", "index_hash": index.index_hash,
                "questions_hash": digest(questions), "config": config.to_dict(), "arms": list(arms),
                "tokenizer": tokenizer.identity, "completed": [],
                "answer_reuse_policy": "same question + identical rendered packet; independent deployment usage retained"}
    write_json(out / "manifest.json", manifest)
    for q in questions:
        qid = q["question_id"]
        filename = digest(qid)[:24] + ".json"  # Untrusted question IDs are never paths.
        try:
            result = compare(index, q["question"], tokenizer, config=config, arms=arms,
                             mentions=q.get("mentions"), frontend_backend=frontend_backend)
            answers = {}
            for row in result["rows"]:
                packet_hash = row["evidence_pack"]["text_hash"]
                if answer_backend is not None:
                    if packet_hash not in answers:
                        start, tick = len(answer_backend.calls), perf_counter()
                        answer = answer_from_sources(q["question"], row["evidence_pack"], answer_backend)
                        answers[packet_hash] = (answer, list(answer_backend.calls[start:]), perf_counter()-tick)
                        reused = False
                    else:
                        reused = True
                    answer, calls, elapsed = answers[packet_hash]
                    row["answer"] = answer
                    row["calls"]["answer"] = calls
                    row["answer_reused_in_experiment"] = reused
                    row["timing_seconds"]["answer_measured_on_original_call"] = elapsed
                    row["manifest"]["status"] = "answered"
            result.update(question_id=qid, owner_id=q["owner_id"])
            write_json(out / filename, result)
            manifest["completed"].append({"question_id": qid, "file": filename})
            write_json(out / "manifest.json", manifest)
        except Exception as error:
            manifest.update(status="failed", failed_question=qid, error_type=type(error).__name__)
            manifest["failure_call_traces"] = {
                "frontend": list(frontend_backend.calls) if frontend_backend else [],
                "answer": list(answer_backend.calls) if answer_backend else []}
            write_json(out / "manifest.json", manifest)
            raise
    manifest["status"] = "complete"
    write_json(out / "manifest.json", manifest)
    return manifest
