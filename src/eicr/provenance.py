"""Snapshot validation, exact deduplication and explicit source-family accounting."""

from collections import defaultdict
from copy import deepcopy

from .contracts import digest, identifier, normalize, require


class UnionFind:
    def __init__(self, values):
        self.parent = {v: v for v in values}

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def prepare_snapshot(snapshot):
    """Return immutable-by-convention canonical records; never mutate caller data.

    The allowed source manifest and extraction attestation are supplied by the
    dataset owner. This checks consistency; it cannot verify an untrue attestation.
    """
    data = deepcopy(snapshot)
    for key in ("owner_id", "snapshot_id", "sources", "cues", "evidence", "edges", "provenance"):
        require(key in data, "Missing snapshot field: " + key)
    provenance = data["provenance"]
    require(provenance.get("extraction_prefix_verified") is True, "Extraction prefix is not verified")
    allowed = set(provenance.get("allowed_source_ids", []))
    sources = {}
    for source in data["sources"]:
        sid = source["source_id"]
        require(sid not in sources, "Duplicate source ID: " + sid)
        require(source.get("owner_id", data["owner_id"]) == data["owner_id"], "Cross-owner source")
        require(isinstance(source.get("text"), str), "Source text must be a string")
        sources[sid] = source
    require(allowed <= sources.keys(), "Allowed source is missing")
    sources = {k: v for k, v in sources.items() if k in allowed}
    cues = {}
    for cue in data["cues"]:
        cid = cue["cue_id"]
        require(cid not in cues, "Duplicate cue ID: " + cid)
        require(isinstance(cue.get("text"), str) and normalize(cue["text"]), "Empty cue")
        require(cue.get("owner_id", data["owner_id"]) == data["owner_id"], "Cross-owner cue")
        require(set(cue.get("extraction_source_ids", [])) <= allowed, "Cue derived from future source")
        # Aliases are explicit records with their own allowed provenance.
        for alias in cue.get("aliases", []):
            require(isinstance(alias, dict) and alias.get("text"), "Alias requires text and provenance")
            require(bool(alias.get("source_ids")) and set(alias["source_ids"]) <= allowed, "Illegal alias provenance")
        cues[cid] = cue
    evidence, native_map, seen_native, dedup = {}, {}, set(), {}
    excluded = []
    for fact in data["evidence"]:
        eid = fact["evidence_id"]
        require(eid not in seen_native, "Duplicate native fact ID: " + eid)
        seen_native.add(eid)
        origins = sorted(set(fact.get("origins", [])))
        require(origins, "Fact has no source: " + eid)
        if not set(origins) <= allowed:
            excluded.append(eid)
            continue
        require(isinstance(fact.get("text"), str) and normalize(fact["text"]), "Empty rewrite")
        metadata = fact.get("metadata", {})
        key = digest([origins, normalize(fact["text"]), metadata, sorted(fact.get("tags", [])), sorted(fact.get("topics", []))])
        if key not in dedup:
            new_id = identifier("e", [data["owner_id"], key])
            dedup[key] = new_id
            evidence[new_id] = {**fact, "evidence_id": new_id, "origins": origins, "native_ids": []}
        new_id = dedup[key]
        evidence[new_id]["native_ids"].append(eid)
        native_map[eid] = new_id
    edges = {}
    for edge in data["edges"]:
        cid, native = edge["cue_id"], edge["evidence_id"]
        require(cid in cues, "Dangling cue edge: " + cid)
        require(native in seen_native, "Dangling fact edge: " + native)
        require(edge.get("weight", 1) == 1, "v1 direct incidence must be binary")
        require(edge.get("edge_origin", "native_keyword") in ("native_keyword", "owner_injected"), "Derived edge in direct R")
        if native not in native_map:
            continue
        eid = native_map[native]
        key = (cid, eid)
        if key not in edges:
            edges[key] = {"cue_id": cid, "evidence_id": eid, "weight": 1, "origins": []}
        edges[key]["origins"].append(edge.get("edge_origin", "native_keyword"))
    # Remove future-only native nodes, but retain explicitly declared zero nodes.
    for edge in edges.values():
        edge["origins"] = sorted(set(edge["origins"]))
    linked = {c for c, _ in edges}
    future_only = {e["cue_id"] for e in data["edges"] if e["evidence_id"] in excluded} - linked
    cues = {c: v for c, v in cues.items() if c not in future_only}
    source_links = data.get("source_links", [])
    uf = UnionFind(sources)
    accepted_source_links = []
    for link in source_links:
        require(link.get("kind") in ("explicit_copy", "explicit_derivation"), "Unknown source root relationship")
        a, b = link["from"], link["to"]
        require(a in sources and b in sources, "Source relationship outside snapshot")
        uf.union(a, b)
        accepted_source_links.append(link)
    root_members = defaultdict(list)
    for sid in sources:
        root_members[uf.find(sid)].append(sid)
    source_roots = {sid: identifier("root", sorted(root_members[uf.find(sid)])) for sid in sources}
    family_uf = UnionFind(set(source_roots.values()))
    for fact in evidence.values():
        roots = [source_roots[s] for s in fact["origins"]]
        for root in roots[1:]:
            family_uf.union(roots[0], root)
    components = defaultdict(list)
    for sid, root in source_roots.items():
        components[family_uf.find(root)].append(sid)
    source_families = {sid: identifier("family", sorted(components[family_uf.find(root)])) for sid, root in source_roots.items()}
    for fact in evidence.values():
        fact["family_id"] = source_families[fact["origins"][0]]
        fact["native_ids"].sort()
    explicit = []
    for link in data.get("context_links", []):
        require(link["from"] in seen_native and link["to"] in seen_native, "Dangling context link")
        require(bool(link.get("source_ids")) and set(link["source_ids"]) <= allowed, "Unproven context link")
        if link["from"] in native_map and link["to"] in native_map:
            explicit.append({**link, "from": native_map[link["from"]], "to": native_map[link["to"]]})
    result = {
        "owner_id": data["owner_id"], "snapshot_id": data["snapshot_id"],
        "sources": [sources[k] for k in sorted(sources)],
        "cues": [cues[k] for k in sorted(cues)],
        "evidence": [evidence[k] for k in sorted(evidence)],
        "edges": [edges[k] for k in sorted(edges)],
        "context_links": sorted(explicit, key=digest),
        "source_roots": source_roots, "source_families": source_families,
        "source_links": sorted(accepted_source_links, key=digest), "provenance": provenance,
        "import_report": {"excluded_future_facts": sorted(excluded), "exact_duplicates": len(native_map) - len(evidence)}
    }
    return result
