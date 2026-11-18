"""Frozen chunks and a shared exact-rendered-token source budget."""

import hashlib
import re
from dataclasses import dataclass

from .contracts import digest, require


@dataclass
class CharacterTokenizer:
    """Deterministic codepoint counter for tests/demos, never a model token claim."""
    identity: str = "unicode-codepoints-test-only-v1"

    def count(self, text):
        return len(text)


class TiktokenTokenizer:
    def __init__(self, encoding):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding)
        # Content fingerprint pins the effective vocabulary, not a moving model alias.
        vocab = sorted((key.hex(), rank) for key, rank in self.encoding._mergeable_ranks.items())
        self.identity = "tiktoken:" + encoding + ":" + digest([vocab, self.encoding._special_tokens, self.encoding._pat_str])

    def count(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))


def chunk_sources(sources, tokenizer, max_tokens=384, overlap_tokens=32):
    require(type(max_tokens) is int and max_tokens > 0 and 0 <= overlap_tokens < max_tokens, "Invalid chunk limits")
    chunks = []
    for source in sources:
        text, sid = source["text"], source["source_id"]
        start = 0
        while start < len(text):
            # All tested prefix lengths are actually counted. Monotonic token counts
            # are not assumed; the final bound is checked for each emitted chunk.
            lo, hi = start + 1, len(text)
            if tokenizer.count(text[start:lo]) > max_tokens:
                raise ValueError("A Unicode character exceeds the chunk token cap")
            best = lo
            while lo <= hi:
                mid = (lo + hi) // 2
                if tokenizer.count(text[start:mid]) <= max_tokens:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            end = best
            if end < len(text):
                boundaries = [m.end() + start for m in re.finditer(r"[.!?](?:\s|$)|[。！？\n]", text[start:end])]
                for boundary in reversed(boundaries):
                    if boundary > start + (end-start)//2 and tokenizer.count(text[start:boundary]) <= max_tokens:
                        end = boundary
                        break
            require(end > start and tokenizer.count(text[start:end]) <= max_tokens, "Chunk cap violated")
            chunks.append({"span_id": digest([sid, start, end])[:24], "source_id": sid,
                           "start": start, "end": end, "text": text[start:end]})
            if end == len(text):
                break
            next_start = end
            while next_start > start + 1 and tokenizer.count(text[next_start-1:end]) <= overlap_tokens:
                next_start -= 1
            start = max(start + 1, next_start)
    return chunks


def _merge_selected(selected, candidate):
    out = [dict(span, queues=list(span["queues"])) for span in selected]
    same, other = [], []
    for span in out:
        (same if span["source_id"] == candidate["source_id"] else other).append(span)
    same.append({"source_id": candidate["source_id"], "start": candidate["start"], "end": candidate["end"], "queues": [candidate["queue"]]})
    merged = []
    for span in sorted(same, key=lambda s: (s["start"], s["end"])):
        if merged and span["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            merged[-1]["queues"] = sorted(set(merged[-1]["queues"]) | set(span["queues"]))
        else:
            merged.append(span)
    return sorted(other + merged, key=lambda s: (s["source_id"], s["start"]))


def render_sources(selected, source_map):
    return "\n\n".join("[" + s["source_id"] + ":" + str(s["start"]) + "-" + str(s["end"]) + "]\n" + source_map[s["source_id"]]["text"][s["start"]:s["end"]] for s in selected)


def pack_sources(queues, sources, tokenizer, quotas, question_id):
    require(set(queues) == set(quotas), "Queue/quota mismatch")
    require("original" in queues and set(queues) <= {"original", "core", "answer_need", "independent_cue", "side_a", "side_b"}, "Unknown evidence queues")
    require(("side_a" in queues) == ("side_b" in queues), "Symmetric sides must occur together")
    require("side_a" not in queues or set(queues) == {"original", "side_a", "side_b"}, "Symmetric mode cannot contain other queues")
    require(all(type(q) is int and q >= 0 for q in quotas.values()), "Invalid quotas")
    source_map = {s["source_id"]: s for s in sources}
    # Validate every candidate before any budget filtering, including skipped ones.
    for name, candidates in queues.items():
        for span in candidates:
            require(span["source_id"] in source_map, "Source outside snapshot")
            text = source_map[span["source_id"]]["text"]
            require(type(span["start"]) is int and type(span["end"]) is int and 0 <= span["start"] < span["end"] <= len(text), "Invalid source span")
            require(span["text"] == text[span["start"]:span["end"]], "Source span text changed")
    budget = sum(quotas.values())
    selected, ledger, omitted = [], [], []
    charged = {q: 0 for q in quotas}
    done = set()
    current_tokens = 0
    def attempt(name, pos, cap, phase):
        nonlocal selected, current_tokens
        candidate = {**queues[name][pos], "queue": name}
        proposed = _merge_selected(selected, candidate)
        rendered = render_sources(proposed, source_map)
        count = tokenizer.count(rendered)
        delta = count - current_tokens
        if max(0, delta) > cap or count > budget:
            omitted.append({"queue": name, "span_id": candidate["span_id"], "reason": "token_quota", "phase": phase})
            return
        was_covered = all(any(s["source_id"] == candidate["source_id"] and s["start"] <= offset < s["end"] for s in selected) for offset in (candidate["start"], candidate["end"]-1))
        # Endpoints alone are not sufficient when there is a gap; record true union coverage.
        covered_chars = sum(max(0, min(s["end"], candidate["end"]) - max(s["start"], candidate["start"])) for s in selected if s["source_id"] == candidate["source_id"])
        was_covered = was_covered and covered_chars >= candidate["end"]-candidate["start"]
        selected, current_tokens = proposed, count
        charged[name] += max(0, delta)
        done.add((name, pos))
        ledger.append({"queue": name, "span_id": candidate["span_id"], "phase": phase,
                       "new_rendered_tokens": delta, "already_covered": was_covered})
    order = [q for q in ("original", "independent_cue", "core", "answer_need") if q in queues]
    if "side_a" in queues:
        sides = ["side_a", "side_b"]
        if int(hashlib.sha256(question_id.encode()).hexdigest(), 16) % 2:
            sides.reverse()
        order = ["original"] + sides
    def schedule():
        if "side_a" not in queues:
            return [(name, pos) for name in order for pos in range(len(queues[name]))]
        items = [("original", pos) for pos in range(len(queues["original"]))]
        for pos in range(max(len(queues[s]) for s in sides)):
            items.extend((s, pos) for s in sides if pos < len(queues[s]))
        return items
    for name, pos in schedule():
        attempt(name, pos, max(0, quotas[name] - charged[name]), "reserved")
    for name, pos in schedule():
        if (name, pos) not in done:
            attempt(name, pos, max(0, budget-current_tokens), "shared_leftover")
    rendered = render_sources(selected, source_map)
    actual = tokenizer.count(rendered)
    require(actual <= budget and actual == current_tokens, "Rendered source budget violated")
    final_omitted = [item for item in omitted if not any(name == item["queue"] and queues[name][pos]["span_id"] == item["span_id"] for name, pos in done)]
    return {"text": rendered, "text_hash": digest(rendered), "spans": selected,
            "budget_tokens": budget, "used_tokens": actual, "unused_tokens": budget-actual,
            "tokenizer": tokenizer.identity, "quotas": quotas, "charged_positive_increments": charged,
            "ledger": ledger, "omitted": final_omitted,
            "accounting_note": "Only final rendered tokens are a total; negative merge deltas and shared text are explicit."}
