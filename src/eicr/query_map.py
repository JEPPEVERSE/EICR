"""Map question spans to existing nodes while preserving ambiguous observations."""

import re
import unicodedata
from collections import defaultdict

from .contracts import digest, lookup, require, tokens


def ground_mentions(question, mentions):
    """Locate exact model-copied surfaces; never guess repeated occurrences.

    This repairs coordinates only, not semantic labels or lexical mismatches.
    Supplied correct coordinates win; otherwise a single exact occurrence is
    required. Original model observations can be retained separately by callers.
    """
    require(isinstance(question, str) and isinstance(mentions, list), "Invalid grounding inputs")
    grounded = []
    for mention in mentions:
        require(isinstance(mention, dict), "Each mention must be an object")
        surface = mention.get("surface_text")
        require(isinstance(surface, str) and bool(surface), "Mention needs a nonempty literal surface")
        start, end = mention.get("span_start"), mention.get("span_end")
        if type(start) is int and type(end) is int and 0 <= start < end <= len(question) and question[start:end] == surface:
            grounded.append(dict(mention))
            continue
        start = question.find(surface)
        require(start >= 0 and question.find(surface, start + 1) == -1, "Surface absent or repeated without valid offsets")
        grounded.append({**mention, "span_start": start, "span_end": start + len(surface)})
    return grounded


def _scan_text(text):
    """NFD + casefold + whitespace normalization with original codepoint spans.

    Canonical combining clusters keep a single original span, so normalization
    may expand/contract characters without fabricating offsets into the question.
    """
    clusters = []
    current, start, end = "", 0, 0
    for pos, char in enumerate(text):
        for part in unicodedata.normalize("NFD", char):
            if current and not unicodedata.combining(part):
                clusters.append((current, start, end))
                current = ""
            if not current:
                start = pos
            current += part
            end = pos + 1
    if current:
        clusters.append((current, start, end))
    chars, spans = [], []
    for cluster, start, end in clusters:
        normalized = unicodedata.normalize("NFD", unicodedata.normalize("NFC", cluster).casefold())
        for char in normalized:
            if char.isspace():
                if not chars:
                    continue
                if chars[-1] == " ":
                    spans[-1] = (spans[-1][0], end)
                    continue
                char = " "
            chars.append(char)
            spans.append((start, end))
    if chars and chars[-1] == " ":
        chars.pop(); spans.pop()
    return "".join(chars), spans


def map_query(index, question, mentions=None, *, question_id=None, answer_need="", query_shape="unknown"):
    require(isinstance(question, str), "Question must be a string")
    require(isinstance(answer_need, str), "answer_need must be a string")
    require(question_id is None or isinstance(question_id, str), "question_id must be a string")
    require(query_shape in ("single_lookup", "explicit_multi_target", "unknown"), "Invalid query shape")
    registry, aliases = defaultdict(set), defaultdict(set)
    for cue in index.records["cues"]:
        registry[lookup(cue["text"])].add(cue["cue_id"])
        for alias in cue.get("aliases", []):
            aliases[lookup(alias["text"])].add(cue["cue_id"])
    # A runnable conservative default, not a complete natural-language parser.
    automatic = mentions is None
    if automatic:
        mentions = []
        normalized_question, offsets = _scan_text(question)
        for surface in sorted(set(registry) | set(aliases), key=lambda s: (-len(s), s)):
            normalized_surface = unicodedata.normalize("NFD", surface)
            for match in re.finditer(re.escape(normalized_surface), normalized_question):
                start, end = offsets[match.start()][0], offsets[match.end()-1][1]
                if lookup(question[start:end]) != surface:
                    continue
                # Avoid substring matches inside Latin words, but permit CJK adjacency.
                if start and question[start-1].isascii() and question[start-1].isalnum() and surface[0].isascii() and surface[0].isalnum():
                    continue
                if end < len(question) and question[end].isascii() and question[end].isalnum() and surface[-1].isascii() and surface[-1].isalnum():
                    continue
                mentions.append({"span_start": start, "span_end": end, "surface_text": question[start:end], "role": "entry"})
        # Longest exact surface wins when proposed spans overlap; report parser basis.
        selected = []
        for mention in sorted(mentions, key=lambda m: (-(m["span_end"]-m["span_start"]), m["span_start"])):
            if not any(mention["span_start"] < m["span_end"] and mention["span_end"] > m["span_start"] for m in selected):
                selected.append(mention)
        mentions = sorted(selected, key=lambda m: m["span_start"])
    require(isinstance(mentions, list), "Mentions must be a list")
    for mention in mentions:
        require(isinstance(mention, dict), "Each mention must be an object")
        require(all(k in mention for k in ("span_start", "span_end", "surface_text")), "Mention lacks required span fields")
        require(isinstance(mention["surface_text"], str), "Mention surface must be a string")
    mapped = []
    entry_count = sum(m.get("role", "entry") == "entry" for m in mentions)
    truncated = entry_count > 24
    kept_entries = 0
    for mention in mentions:
        start, end = mention["span_start"], mention["span_end"]
        require(type(start) is int and type(end) is int and 0 <= start < end <= len(question), "Invalid Unicode span")
        require(question[start:end] == mention["surface_text"], "Mention offset/text mismatch")
        role = mention.get("role", "entry")
        require(role in ("entry", "scope", "answer_request"), "Invalid mention role")
        if role == "entry":
            kept_entries += 1
            if kept_entries > 24:
                continue
        key = lookup(mention["surface_text"])
        exact, alias = registry[key], aliases[key]
        # Alias collisions with native text are ambiguous rather than hidden.
        candidates = sorted(exact | alias)
        candidate_count = len(candidates)
        method = "exact" if exact else "explicit_alias" if alias else "none"
        status = "unique" if len(candidates) == 1 else "ambiguous" if candidates else "unmapped"
        if not candidates:
            query_tokens = set(tokens(mention["surface_text"]))
            ranked = []
            for cue in index.records["cues"]:
                candidate_tokens = set(tokens(cue["text"]))
                union = candidate_tokens | query_tokens
                score = len(candidate_tokens & query_tokens) / len(union) if union else 0
                if score > 0:
                    ranked.append((-score, cue["cue_id"]))
            candidates = [cid for _, cid in sorted(ranked)[:3]]
            candidate_count = len(ranked)
            if candidates:
                status, method = "ambiguous", "lexical_fuzzy"
        if status == "unique":
            cue = index.records["cues"][index.cue_pos[candidates[0]]]
            if cue.get("scope_role") == "scope":
                role = "scope"
        mapped.append({"span_start": start, "span_end": end, "surface_text": mention["surface_text"],
                       "role": role, "candidate_cue_ids": candidates[:3], "match_method": method, "status": status,
                       "candidate_cue_count": candidate_count, "candidate_cues_truncated": candidate_count > 3})
    return {"question_id": question_id or digest(question)[:16], "original_question": question,
            "snapshot_id": index.records["snapshot_id"], "index_hash": index.index_hash,
            "answer_need": answer_need, "query_shape": query_shape, "mentions": mapped,
            "unresolved_spans": [m["surface_text"] for m in mapped if m["status"] != "unique"],
            "parse_truncated": truncated, "frontend_error": None,
            "frontend_basis": "conservative_exact_registry_scan" if automatic else "supplied_spans",
            "semantic_completeness_verified": False}
