"""Sparse cue-profile graph and aligned fact-vector control."""
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import sparse
from eicr.contracts import digest, require, read_json, write_json
from .native import validate_snapshot


@dataclass
class Index:
    snapshot: dict
    profiles: sparse.csr_matrix
    vectors: sparse.csr_matrix
    neighbors: dict
    report: dict
    index_hash: str
    top_k: int
    representation: str
    representation_identity: str

    @property
    def records(self):
        # Narrow adapter for the existing, grounded mention matcher.
        return {"cues": self.snapshot["cues"], "snapshot_id": self.snapshot["snapshot_id"]}

    @property
    def cue_pos(self):
        return {c["cue_id"]: i for i, c in enumerate(self.snapshot["cues"])}


def payload(index):
    def rows(matrix):
        return [[[int(j), float(v)] for j, v in zip(matrix.getrow(i).indices, matrix.getrow(i).data)]
                for i in range(matrix.shape[0])]
    return {"schema": "eitr-index-1", "snapshot": index.snapshot,
            "profiles": {"shape": list(index.profiles.shape), "rows": rows(index.profiles)},
            "vectors": {"shape": list(index.vectors.shape), "rows": rows(index.vectors)},
            "neighbors": index.neighbors, "top_k": index.top_k,
            "representation": index.representation, "representation_identity": index.representation_identity}


def build_index(snapshot, top_k=16, fact_vectors=None):
    start = perf_counter()
    validate_snapshot(snapshot)
    require(type(top_k) is int and top_k > 0, "Invalid neighbor count")
    tags, cues = snapshot["tags"], snapshot["cues"]
    tp = {t["tag_id"]: i for i, t in enumerate(tags)}
    cp = {c["cue_id"]: i for i, c in enumerate(cues)}
    fact_cues = defaultdict(set)
    for c, f in snapshot["feature_edges"]:
        fact_cues[f].add(cp[c])
    grouped = defaultdict(lambda: defaultdict(list))
    for f in snapshot["facts"]:
        for tid in f["tag_ids"]:
            grouped[tid][tuple(f["origins"])].append(f)
    row, col, val = [], [], []
    usable_groups = {}
    for tid in tp:
        groups = []
        for facts in grouped[tid].values():
            valid = [f for f in facts if fact_cues[f["fact_id"]]]
            if valid:
                profile = defaultdict(float)
                for f in valid:
                    coords = fact_cues[f["fact_id"]]
                    for c in coords:
                        profile[c] += 1 / (len(valid) * len(coords))
                groups.append(profile)
        usable_groups[tid] = len(groups)
        merged = defaultdict(float)
        for group in groups:
            for c, v in group.items():
                merged[c] += v / len(groups)
        for c, v in sorted(merged.items()):
            row.append(tp[tid]); col.append(c); val.append(v)
    H = sparse.csr_matrix((val, (row, col)), shape=(len(tags), len(cues)))
    V = H.copy()
    m = sum(n > 0 for n in usable_groups.values())
    df = np.asarray((H > 0).sum(axis=0)).ravel()
    idf = np.log((m + 1) / (df + 1))
    V.data = np.sqrt(V.data * idf[V.indices])
    V.eliminate_zeros()
    representation, identity = "R_cue", "native-cue-source-balanced-idf-sqrt-v2"
    if fact_vectors is not None:
        require(fact_vectors.get("snapshot_hash") == snapshot["content_hash"], "Fact vectors scope mismatch")
        require(isinstance(fact_vectors.get("encoder"), str) and fact_vectors["encoder"], "Missing encoder pin")
        ids = [f["fact_id"] for f in snapshot["facts"]]
        require(fact_vectors.get("fact_ids") == ids, "Fact vectors order mismatch")
        x = np.asarray(fact_vectors.get("vectors"), dtype=np.float64)
        require(x.ndim == 2 and x.shape[0] == len(ids) and x.shape[1] > 0 and np.isfinite(x).all(),
                "Invalid fact vectors")
        norms = np.max(np.abs(x), axis=1)
        require((norms > 0).all(), "Zero fact vector")
        x = x / norms[:, None]; x /= np.linalg.norm(x, axis=1)[:, None]
        fp = {fid: i for i, fid in enumerate(ids)}
        aggregated = []
        for t in tags:
            means = [np.mean([x[fp[f["fact_id"]]] for f in fs], axis=0)
                     for fs in grouped[t["tag_id"]].values()]
            aggregated.append(np.mean(means, axis=0))
        V = sparse.csr_matrix(np.asarray(aggregated).reshape(len(tags), x.shape[1]))
        representation, identity = "R_fact", fact_vectors["encoder"] + ":" + digest(fact_vectors)
    norms = np.sqrt(np.asarray(V.multiply(V).sum(axis=1)).ravel())
    inv = np.divide(1., norms, out=np.zeros_like(norms), where=norms > 0)
    V = (sparse.diags(inv) @ V).tocsr()
    neighbors = {}
    # One row at a time: bounded temporary similarity row, never M x M dense storage.
    for i, t in enumerate(tags):
        similarity = (V.getrow(i) @ V.T).tocsr()
        pairs = [(j, min(1., max(0., float(s)))) for j, s in zip(similarity.indices, similarity.data)
                 if j != i and s > 0]
        pairs.sort(key=lambda p: (-p[1], tags[p[0]]["tag_id"]))
        edges = []
        for j, score in pairs[:top_k]:
            contributions = V.getrow(i).multiply(V.getrow(j)).tocoo()
            features = sorted(zip(contributions.col, contributions.data), key=lambda p: (-p[1], p[0]))[:5]
            edges.append({"target": tags[j]["tag_id"], "score": score,
                          "features": [{"coordinate": cues[c]["cue_id"] if representation == "R_cue" else int(c),
                                        "contribution": float(v)} for c, v in features]})
        neighbors[t["tag_id"]] = edges
    report = {"tags": len(tags), "facts": len(snapshot["facts"]), "cues": len(cues),
              "valid_profile_tags": m, "zero_vectors": int(sum(norms == 0)),
              "profile_nnz": H.nnz, "vector_nnz": V.nnz,
              "edges": sum(map(len, neighbors.values())), "usable_source_groups": usable_groups,
              "build_seconds": perf_counter() - start}
    index = Index(snapshot, H, V, neighbors, report, "", top_k, representation, identity)
    index.index_hash = digest(payload(index))
    return index


def save_index(index, path):
    write_json(path, {**payload(index), "index_hash": index.index_hash, "report": index.report})


def load_index(path):
    data = read_json(path)
    require(data.get("schema") == "eitr-index-1", "Wrong index schema")
    body = {k: v for k, v in data.items() if k not in ("index_hash", "report")}
    require(digest(body) == data.get("index_hash"), "Index content hash mismatch")
    validate_snapshot(data["snapshot"])
    def matrix(obj):
        shape = obj["shape"]
        require(len(shape) == 2 and all(type(n) is int and n >= 0 for n in shape), "Invalid matrix shape")
        require(len(obj["rows"]) == shape[0], "Matrix row mismatch")
        ii, jj, vv = [], [], []
        for i, row in enumerate(obj["rows"]):
            seen = set()
            for j, v in row:
                require(type(j) is int and 0 <= j < shape[1] and j not in seen
                        and type(v) in (float, int) and np.isfinite(v), "Invalid sparse coordinate")
                seen.add(j); ii.append(i); jj.append(j); vv.append(v)
        return sparse.csr_matrix((vv, (ii, jj)), shape=shape)
    H, V = matrix(data["profiles"]), matrix(data["vectors"])
    tags = {t["tag_id"] for t in data["snapshot"]["tags"]}
    require(H.shape == (len(tags), len(data["snapshot"]["cues"])) and V.shape[0] == len(tags), "Index shape mismatch")
    require(set(data["neighbors"]) == tags, "Graph tag mismatch")
    for tid, edges in data["neighbors"].items():
        seen = set()
        require(len(edges) <= data["top_k"], "Too many edges")
        for edge in edges:
            require(edge["target"] in tags - {tid} and edge["target"] not in seen
                    and type(edge["score"]) in (float, int) and 0 < edge["score"] <= 1, "Invalid edge")
            seen.add(edge["target"])
    return Index(data["snapshot"], H, V, data["neighbors"], data["report"], data["index_hash"],
                 data["top_k"], data["representation"], data["representation_identity"])
