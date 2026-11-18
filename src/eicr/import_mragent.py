"""Import frozen MRAgent packs without importing or executing Agent.

The caller must supply a prefix-verification manifest. IDs, rather than sentence
list positions, determine keyword joins. Native speaker injection is not added.
"""

import re

from .contracts import identifier, require


def import_pack(pack, manifest):
    require(manifest.get("extraction_prefix_verified") is True, "Missing extraction provenance attestation")
    owner, snapshot = manifest["owner_id"], manifest["snapshot_id"]
    allowed = set(manifest["allowed_source_ids"])
    sources = []
    for session, rows in pack["raw_text"].items():
        for sid, text in rows.items():
            if sid in allowed:
                sources.append({"source_id": sid, "text": text, "session_id": session})
    require({s["source_id"] for s in sources} == allowed, "Manifest references missing raw sources")
    facts, known = [], set()
    # Keywords may be plain session records or explicitly keyed session records.
    keyword_records = pack.get("keywords", [])
    require(len(keyword_records) == len(pack["rewrites"]), "Session cache count mismatch")
    session_facts = []
    for record in pack["rewrites"]:
        require(isinstance(record, dict) and len(record) == 1, "Expected one keyed rewrite session")
        session, payload = next(iter(record.items()))
        local = set()
        if payload is not None:
            topic_registry = payload.get("topics") or {}
            for fact in payload.get("sentence") or []:
                fid = fact["id"]
                require(fid not in known, "Duplicate native sentence ID")
                known.add(fid)
                local.add(fid)
                origins = fact["origin"]
                if isinstance(origins, str):
                    origins = [x.strip() for x in origins.split(",") if x.strip()]
                require(all(re.fullmatch(r"[^\s,]+:\d+", o) for o in origins), "Unrecognized native origin")
                topics = fact.get("topic") or []
                if isinstance(topics, str):
                    topics = [topics]
                require(all(t in topic_registry for t in topics), "Dangling native topic")
                tag = fact.get("tag")
                tags = tag if isinstance(tag, list) else ([tag] if tag else [])
                facts.append({
                    "evidence_id": fid, "text": fact["text"], "origins": origins,
                    "topics": [identifier("topic", [owner, session, t]) for t in topics],
                    "tags": tags,
                    "metadata": {"session_id": session, "native_time": fact.get("time"),
                                 "time_provenance": "legacy_unverified_rewrite",
                                 "conversation_time": payload.get("conversation_time")}
                })
        session_facts.append((session, local))
    cues, edges = {}, []
    for pos, record in enumerate(keyword_records):
        session, local = session_facts[pos]
        if record is None:
            continue
        if session in record:
            require(len(record) == 1, "Ambiguous keyed keyword session")
            record = record[session]
        require(record is not None and isinstance(record.get("sentence"), list), "Invalid keyword session")
        for row in record["sentence"]:
            fid = row["sentence_id"]
            require(fid in local, "Keyword sentence does not belong to corresponding rewrite session")
            for text in row.get("keyword") or []:
                require(isinstance(text, str) and text.strip(), "Invalid keyword")
                cid = identifier("cue", [owner, text])
                if cid not in cues:
                    cues[cid] = {"cue_id": cid, "text": text, "native_key": text,
                                 "aliases": [], "scope_role": "scope" if text in manifest.get("scope_cue_texts", []) else "entry"}
                edges.append({"cue_id": cid, "evidence_id": fid, "edge_origin": "native_keyword"})
    return {
        "owner_id": owner, "snapshot_id": snapshot, "sources": sources,
        "cues": list(cues.values()), "evidence": facts, "edges": edges,
        "provenance": {"allowed_source_ids": sorted(allowed), "extraction_prefix_verified": True,
                       "attestation": manifest.get("attestation", "caller_supplied")}
    }
