"""Validated native snapshot adapter. No code or models from MRAgent are executed."""
from copy import deepcopy
from eicr.contracts import digest, identifier, require
from eicr.import_mragent import import_pack as import_legacy_pack

SCHEMA = "eitr-snapshot-1"


def import_pack(pack, manifest):
    """Require explicit time/source attestations rather than trust rewrite timestamps."""
    legacy = import_legacy_pack(pack, manifest)
    times = manifest.get("session_datetimes", {})
    require(isinstance(times, dict), "session_datetimes must be an object")
    require(set(times) <= {s["session_id"] for s in legacy["sources"]},
            "Date map must use raw session IDs, not rewrite session keys")
    # Keywords loaded from the cache, never the runtime-injected speaker list.
    require(manifest.get("keyword_cache_is_pre_injection") is True,
            "Attest keyword cache is before runtime speaker injection")
    for s in legacy["sources"]:
        value = times.get(s["session_id"])
        require(value is None or isinstance(value, str), "Invalid session datetime")
        s["session_datetime"] = value
        s["datetime_provenance"] = "manifest_verified" if value else "unavailable"
        s["speaker"] = s["text"].split(":", 1)[0] if ":" in s["text"] else "unknown"
    legacy["provenance"]["keyword_cache_is_pre_injection"] = True
    return canonicalize(legacy)


def canonicalize(raw):
    require(isinstance(raw, dict), "Snapshot must be an object")
    for name in ("owner_id", "snapshot_id"):
        require(isinstance(raw.get(name), str) and raw[name], "Missing " + name)
    owner, snap = raw["owner_id"], raw["snapshot_id"]
    provenance = raw.get("provenance", {})
    require(provenance.get("extraction_prefix_verified") is True, "Unverified memory horizon")
    sources = deepcopy(raw.get("sources", []))
    require(isinstance(sources, list), "sources must be a list")
    source_map = {}
    for s in sources:
        require(isinstance(s, dict), "Invalid source")
        for name in ("source_id", "session_id", "text"):
            require(isinstance(s.get(name), str) and bool(s[name]), "Invalid source " + name)
        require(s["source_id"] not in source_map, "Duplicate source ID")
        require(not any(c in s["source_id"] for c in "[]\r\n"), "Unsafe source ID header")
        for name, expected in (("owner_id", owner), ("snapshot_id", snap)):
            require(name not in s or s[name] == expected, "Cross-scope source")
        value = s.get("session_datetime")
        require(value is None or (isinstance(value, str) and bool(value)), "Invalid source date")
        if value:
            require(s.get("datetime_provenance") in ("manifest_verified", "native_verified"),
                    "Source date is not attested")
        s.setdefault("session_datetime", None)
        s.setdefault("datetime_provenance", "unavailable")
        s.setdefault("speaker", "unknown")
        require(isinstance(s["speaker"], str), "Invalid speaker")
        source_map[s["source_id"]] = s
    allowed = provenance.get("allowed_source_ids")
    require(isinstance(allowed, list) and len(set(allowed)) == len(allowed)
            and set(allowed) == set(source_map), "Allowed sources mismatch")
    cues = deepcopy(raw.get("cues", []))
    cue_map = {}
    for c in cues:
        require(isinstance(c, dict) and isinstance(c.get("cue_id"), str)
                and c["cue_id"] and isinstance(c.get("text"), str) and c["text"].strip(), "Invalid cue")
        require(c["cue_id"] not in cue_map, "Duplicate cue ID")
        for alias in c.get("aliases", []):
            require(isinstance(alias, dict) and isinstance(alias.get("text"), str)
                    and bool(alias["text"].strip()), "Invalid explicit alias")
            require(isinstance(alias.get("source_ids"), list) and alias["source_ids"]
                    and set(alias["source_ids"]) <= set(source_map), "Alias lacks in-snapshot provenance")
        cue_map[c["cue_id"]] = c
    facts, tags, fact_map = [], {}, {}
    for f in raw.get("evidence", []):
        require(isinstance(f, dict) and isinstance(f.get("evidence_id"), str)
                and f["evidence_id"] and isinstance(f.get("text"), str), "Invalid fact")
        fid = f["evidence_id"]
        require(fid not in fact_map, "Duplicate fact ID")
        origins = f.get("origins")
        require(isinstance(origins, list) and origins and all(isinstance(o, str) for o in origins)
                and set(origins) <= set(source_map), "Fact has missing or out-of-horizon origins")
        native_tags = f.get("tags", [])
        require(isinstance(native_tags, list) and all(isinstance(t, str) and t.strip() for t in native_tags),
                "Fact tags must be native strings")
        tids = []
        for text in sorted(set(native_tags)):
            tid = identifier("tag", [owner, snap, text])
            tags[tid] = {"tag_id": tid, "text": text}
            tids.append(tid)
        fact = {"fact_id": fid, "text": f["text"], "origins": sorted(set(origins)),
                "tag_ids": tids, "metadata": deepcopy(f.get("metadata", {}))}
        facts.append(fact); fact_map[fid] = fact
    links, feature_edges = set(), set()
    for edge in raw.get("edges", []):
        cid, fid = edge.get("cue_id"), edge.get("evidence_id")
        require(cid in cue_map and fid in fact_map, "Dangling native edge")
        origin = edge.get("edge_origin")
        require(origin in ("native_keyword", "injected_speaker"), "Unknown edge origin")
        if origin == "native_keyword":
            feature_edges.add((cid, fid))
        for tid in fact_map[fid]["tag_ids"]:
            links.add((cid, tid, fid, origin))
    body = {"schema": SCHEMA, "owner_id": owner, "snapshot_id": snap,
            "sources": sorted(sources, key=lambda s: s["source_id"]),
            "cues": sorted(cues, key=lambda c: c["cue_id"]),
            "facts": sorted(facts, key=lambda f: f["fact_id"]),
            "tags": sorted(tags.values(), key=lambda t: t["tag_id"]),
            "links": [{"cue_id": c, "tag_id": t, "fact_id": f, "edge_origin": o}
                      for c, t, f, o in sorted(links)],
            "feature_edges": [list(x) for x in sorted(feature_edges)],
            "provenance": deepcopy(provenance)}
    return {**body, "content_hash": digest(body)}


def validate_snapshot(snapshot):
    require(snapshot.get("schema") == SCHEMA, "Not an EITR snapshot")
    body = {k: v for k, v in snapshot.items() if k != "content_hash"}
    require(digest(body) == snapshot.get("content_hash"), "Snapshot hash mismatch")
    # Reconstruct and revalidate the underlying native relationships, not only a checksum.
    tag_names = {t["tag_id"]: t["text"] for t in snapshot["tags"]}
    raw = {k: deepcopy(snapshot[k]) for k in ("owner_id", "snapshot_id", "sources", "cues", "provenance")}
    raw["evidence"] = [{"evidence_id": f["fact_id"], "text": f["text"], "origins": f["origins"],
                        "tags": [tag_names[t] for t in f["tag_ids"]], "metadata": f["metadata"]}
                       for f in snapshot["facts"]]
    edges = {(l["cue_id"], l["fact_id"], l["edge_origin"]) for l in snapshot["links"]}
    edges.update((c, f, "native_keyword") for c, f in snapshot["feature_edges"])
    raw["edges"] = [{"cue_id": c, "evidence_id": f, "edge_origin": o} for c, f, o in sorted(edges)]
    rebuilt = canonicalize(raw)
    require(rebuilt == snapshot, "Invalid canonical native relationships")
    return snapshot
