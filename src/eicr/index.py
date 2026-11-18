"""Sparse evidence-induced representations, provenance artifacts and verified I/O."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from .config import IndexConfig
from .contracts import digest, read_json, require, write_json
from .neighbors import build_transition
from .provenance import prepare_snapshot


def row_norms(matrix):
    return np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())


@dataclass
class Index:
    records: dict
    config: IndexConfig
    R: csr_matrix
    X: csr_matrix
    Y: csr_matrix
    P: csr_matrix
    T: csr_matrix
    weights: np.ndarray
    neighbors: list
    report: dict
    index_hash: str

    def __post_init__(self):
        self.cue_ids = [c["cue_id"] for c in self.records["cues"]]
        self.cue_pos = {c: i for i, c in enumerate(self.cue_ids)}
        self.fact_ids = [f["evidence_id"] for f in self.records["evidence"]]
        self.norm_x, self.norm_y = row_norms(self.X), row_norms(self.Y)
        self.mass = np.asarray(self.R @ self.weights).ravel()

    def similarity(self, a, b):
        i, j = self.cue_pos[a], self.cue_pos[b]
        if self.norm_x[i] == 0 or self.norm_x[j] == 0:
            return {"direct": None, "expanded": None, "combined": None}
        direct = float(self.X[i].multiply(self.X[j]).sum() / (self.norm_x[i] * self.norm_x[j]))
        expanded = float(self.Y[i].multiply(self.Y[j]).sum() / (self.norm_y[i] * self.norm_y[j]))
        direct, expanded = float(np.clip(direct, 0, 1)), float(np.clip(expanded, 0, 1))
        return {"direct": direct, "expanded": expanded,
                "combined": (1 - self.config.lambda_) * direct + self.config.lambda_ * expanded}

    def coverage(self, a, b):
        i, j = self.cue_pos[a], self.cue_pos[b]
        if self.mass[i] == 0:
            return None
        return float(self.R[i].minimum(self.R[j]).dot(self.weights)[0] / self.mass[i])

    def joint(self, cue_ids):
        if not cue_ids or any(self.mass[self.cue_pos[c]] == 0 for c in cue_ids):
            return {"mass": None, "normalized": None}
        row = self.R[self.cue_pos[cue_ids[0]]]
        for cid in cue_ids[1:]:
            row = row.minimum(self.R[self.cue_pos[cid]])
        mass = float(row.dot(self.weights)[0])
        return {"mass": mass, "normalized": mass / min(self.mass[self.cue_pos[c]] for c in cue_ids)}

    def facet_profile(self, cue_id):
        i = self.cue_pos[cue_id]
        masses = Counter()
        for j in self.R[i].indices:
            tags = sorted(set(self.records["evidence"][j].get("tags", []))) or ["unclassified"]
            for tag in tags:
                masses[tag] += float(self.weights[j]) / len(tags)
        return dict(sorted(masses.items()))

    def explain(self, a, b, limit=8):
        """Top coordinate contributions; expanded overlap is a derived feature."""
        i, j = self.cue_pos[a], self.cue_pos[b]
        if not self.norm_x[i] or not self.norm_x[j]:
            return {"direct": [], "expanded": [], "status": "unavailable"}
        out = {}
        for name, matrix, norms in (("direct", self.X, self.norm_x), ("expanded", self.Y, self.norm_y)):
            products = matrix[i].multiply(matrix[j]).tocsr()
            values = [(int(k), float(value/(norms[i]*norms[j]))) for k, value in zip(products.indices, products.data) if value > 0]
            values.sort(key=lambda item: (-item[1], self.fact_ids[item[0]]))
            out[name] = [{"evidence_id": self.fact_ids[k], "contribution": value,
                          "origins": self.records["evidence"][k]["origins"],
                          "family_id": self.records["evidence"][k]["family_id"]} for k, value in values[:limit]]
            out[name+"_omitted_contribution"] = sum(value for _, value in values[limit:])
        out["expanded_is_independent_witness"] = False
        return out


def build_index(snapshot, config=None):
    config = config or IndexConfig()
    records = prepare_snapshot(snapshot)
    return _build_records(records, config)


def _build_records(records, config):
    cues, facts = records["cues"], records["evidence"]
    cp = {c["cue_id"]: i for i, c in enumerate(cues)}
    ep = {e["evidence_id"]: i for i, e in enumerate(facts)}
    r = csr_matrix((np.ones(len(records["edges"])),
                    ([cp[e["cue_id"]] for e in records["edges"]],
                     [ep[e["evidence_id"]] for e in records["edges"]])), shape=(len(cues), len(facts)), dtype=np.float64)
    r.sum_duplicates()
    r.data[:] = 1
    counts = Counter(f["family_id"] for f in facts)
    weights = np.array([1 / (len(counts) * counts[f["family_id"]]) for f in facts], dtype=np.float64)
    if config.weight_mode == "uniform" and facts:
        weights[:] = 1 / len(facts)
    x = r.multiply(np.sqrt(weights)).tocsr()
    p, t, neighbors, graph_report = build_transition(records, config)
    y = (x @ t).tocsr()
    y.sort_indices()
    report = {**records["import_report"], **graph_report, "status": "ok" if facts else "empty_index",
              "cue_count": len(cues), "evidence_count": len(facts), "source_count": len(records["sources"]),
              "family_sizes": dict(sorted(counts.items())), "nnz_R": r.nnz, "nnz_Y": y.nnz,
              "expansion_ratio": y.nnz / r.nnz if r.nnz else 0.0,
              "zero_cues": int(np.count_nonzero(row_norms(x) == 0))}
    identity = digest({"schema": "eicr-index-1", "records": records, "config": config.to_dict()})
    return Index(records, config, r, x, y, p, t, weights, neighbors, report, identity)


def audit_index(index):
    """Fail closed on changed matrices, records or configuration."""
    require(index.index_hash == digest({"schema": "eicr-index-1", "records": index.records, "config": index.config.to_dict()}), "Index identity mismatch")
    expected = _build_records(index.records, index.config)
    for name in ("R", "X", "Y", "P", "T"):
        matrix = getattr(index, name)
        oracle = getattr(expected, name)
        require(matrix.shape == oracle.shape, name + " shape mismatch")
        require(np.isfinite(matrix.data).all() and (matrix.data >= 0).all(), name + " has invalid entries")
        difference = matrix - oracle
        require(not difference.nnz or np.max(np.abs(difference.data)) <= 1e-10, name + " differs from frozen inputs")
    require(np.array_equal(index.weights, expected.weights), "Coordinate weights changed")
    require(index.neighbors == expected.neighbors and index.report == expected.report, "Audit metadata changed")
    if len(index.fact_ids):
        require(np.allclose(np.asarray(index.P.sum(axis=1)).ravel(), 1), "P is not stochastic")
        require(np.allclose(np.asarray(index.T.sum(axis=1)).ravel(), 1), "T is not stochastic")
        require(np.isclose(index.weights.sum(), 1), "Weight sum is not one")
    return {"status": "passed", "index_hash": index.index_hash, **index.report}


def save_index(index, folder):
    folder = Path(folder)
    require(not folder.exists(), "Output index already exists; use a new immutable directory")
    audit_index(index)
    folder.mkdir(parents=True)
    write_json(folder / "records.json", index.records)
    write_json(folder / "neighbors.json", index.neighbors)
    write_json(folder / "build_report.json", index.report)
    for name in ("R", "X", "Y", "P", "T"):
        save_npz(folder / (name + ".npz"), getattr(index, name))
    np.save(folder / "weights.npy", index.weights, allow_pickle=False)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}
    write_json(folder / "manifest.json", {"schema": "eicr-index-1", "index_hash": index.index_hash,
                                         "config": index.config.to_dict(), "files_sha256": files})


def load_index(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    require(manifest.get("schema") == "eicr-index-1", "Unknown index schema")
    expected_files = {"records.json", "neighbors.json", "build_report.json", "weights.npy", "R.npz", "X.npz", "Y.npz", "P.npz", "T.npz"}
    require(set(manifest["files_sha256"]) == expected_files, "Incomplete index file manifest")
    for name, sha in manifest["files_sha256"].items():
        require(hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha, "Artifact hash mismatch: " + name)
    matrices = {name: load_npz(folder / (name + ".npz")).tocsr() for name in ("R", "X", "Y", "P", "T")}
    index = Index(read_json(folder / "records.json"), IndexConfig.from_dict(manifest["config"]),
                  **matrices, weights=np.load(folder / "weights.npy", allow_pickle=False),
                  neighbors=read_json(folder / "neighbors.json"), report=read_json(folder / "build_report.json"), index_hash=manifest["index_hash"])
    audit_index(index)
    return index
