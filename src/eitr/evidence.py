"""Frozen non-overlapping raw atoms, full-context actions, and exact token budgets."""
from collections import defaultdict, Counter
from pathlib import Path
import hashlib
import json
import re
from time import perf_counter
from eicr.contracts import digest, require, read_json, write_json
from eicr.packer import chunk_sources
from eicr.retrieval import SharedRetriever


def natural(value):
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in re.split(r"(\d+)", str(value)))


class JsonTokenizer:
    def __init__(self, path, expected_sha256):
        from pathlib import Path
        from tokenizers import Tokenizer
        raw = Path(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Tokenizer SHA256 mismatch")
        self.tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        self.identity = "tokenizer-json-sha256:" + expected_sha256

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


class Corpus:
    def __init__(self, snapshot, tokenizer, config):
        start = perf_counter()
        self.tokenizer = tokenizer
        self.atom_config = (config.atom_tokens, config.context_each_side)
        self.snapshot_hash = snapshot["content_hash"]
        self.sources = {s["source_id"]: s for s in snapshot["sources"]}
        self.atoms = {}
        self.source_atoms = defaultdict(list)
        sessions = defaultdict(list)
        ordered = sorted(snapshot["sources"], key=lambda s: (natural(s["session_id"]), natural(s["source_id"])))
        for atom in chunk_sources(ordered, tokenizer, config.atom_tokens, 0):
            sid = atom["source_id"]
            aid = digest([self.snapshot_hash, sid, atom["start"], atom["end"], tokenizer.identity])[:24]
            atom.update(atom_id=aid, span_id=aid, text_tokens=tokenizer.count(atom["text"]))
            require(atom["text_tokens"] > 0, "Nonempty raw text has zero tokens")
            self.atoms[aid] = atom
            self.source_atoms[sid].append(aid)
            sessions[self.sources[sid]["session_id"]].append(aid)
        self.contexts = {}
        for ids in sessions.values():
            for i, aid in enumerate(ids):
                self.contexts[aid] = ids[max(0, i-config.context_each_side):i+config.context_each_side+1]
        documents = []
        for aid, atom in self.atoms.items():
            src = self.sources[atom["source_id"]]
            # Time is queryable but remains separate from raw quoted character offsets.
            documents.append({"span_id": aid, "text": atom["text"] + "\n" +
                              (src.get("session_datetime") or "") + "\n" + src.get("speaker", "unknown")})
        self.order = list(self.atoms)
        self.retriever = SharedRetriever(documents)
        self.identity = digest([self.snapshot_hash, tokenizer.identity, config.atom_tokens,
                                config.context_each_side, documents])
        self.build_seconds = perf_counter() - start
        self.load_seconds = 0.
        self.cache_hash = None

    @staticmethod
    def implementation_hash():
        import eicr.packer, eicr.retrieval, eicr.contracts
        return digest({m.__name__: Path(m.__file__).read_text()
                       for m in (eicr.packer, eicr.retrieval, eicr.contracts)} |
                      {"evidence": Path(__file__).read_text()})

    def save(self, path):
        body = {"schema": "eitr-corpus-1", "snapshot_hash": self.snapshot_hash,
                "tokenizer": self.tokenizer.identity, "atom_config": list(self.atom_config),
                "implementation_hash": self.implementation_hash(), "identity": self.identity,
                "atoms": [self.atoms[a] for a in self.order], "contexts": self.contexts,
                "documents": self.retriever.chunks,
                "term_counts": [dict(c) for c in self.retriever.term_counts],
                "df": dict(self.retriever.df), "avg_len": self.retriever.avg_len,
                "build_seconds": self.build_seconds}
        write_json(path, {**body, "cache_hash": digest(body)})

    @classmethod
    def load(cls, path, snapshot, tokenizer, config):
        start = perf_counter()
        body = read_json(path)
        cache_hash = body.pop("cache_hash", None)
        require(digest(body) == cache_hash and body.get("schema") == "eitr-corpus-1", "Corpus checksum/schema mismatch")
        require(body["snapshot_hash"] == snapshot["content_hash"] and body["tokenizer"] == tokenizer.identity
                and body["atom_config"] == [config.atom_tokens, config.context_each_side]
                and body["implementation_hash"] == cls.implementation_hash(), "Stale or incompatible corpus cache")
        obj = cls.__new__(cls)
        obj.tokenizer, obj.snapshot_hash = tokenizer, snapshot["content_hash"]
        obj.atom_config = (config.atom_tokens, config.context_each_side)
        obj.sources = {s["source_id"]: s for s in snapshot["sources"]}
        obj.atoms, obj.source_atoms = {}, defaultdict(list)
        sessions, offsets = defaultdict(list), defaultdict(int)
        for atom in body["atoms"]:
            sid, aid = atom["source_id"], atom["atom_id"]
            require(sid in obj.sources and aid not in obj.atoms and atom["start"] == offsets[sid]
                    and atom["end"] > atom["start"] and atom["text"] == obj.sources[sid]["text"][atom["start"]:atom["end"]]
                    and 0 < atom["text_tokens"] <= config.atom_tokens, "Invalid cached raw atom")
            require(aid == digest([obj.snapshot_hash, sid, atom["start"], atom["end"], tokenizer.identity])[:24], "Invalid cached atom ID")
            offsets[sid] = atom["end"]
            obj.atoms[aid] = atom; obj.source_atoms[sid].append(aid)
            sessions[obj.sources[sid]["session_id"]].append(aid)
        require(all(offsets[sid] == len(s["text"]) for sid, s in obj.sources.items()), "Cached atoms do not cover raw sources")
        obj.order = list(obj.atoms)
        obj.contexts = {aid: ids[max(0, i-config.context_each_side):i+config.context_each_side+1]
                        for ids in sessions.values() for i, aid in enumerate(ids)}
        require(obj.contexts == body["contexts"], "Invalid cached context")
        documents = [{"span_id": a, "text": obj.atoms[a]["text"] + "\n" +
                      (obj.sources[obj.atoms[a]["source_id"]].get("session_datetime") or "") + "\n" +
                      obj.sources[obj.atoms[a]["source_id"]].get("speaker", "unknown")} for a in obj.order]
        require(documents == body["documents"] and len(body["term_counts"]) == len(documents), "Invalid cached BM25 documents")
        obj.retriever = SharedRetriever.__new__(SharedRetriever)
        obj.retriever.chunks = documents
        obj.retriever.term_counts = [Counter(c) for c in body["term_counts"]]
        require(all(isinstance(k, str) and type(v) is int and v > 0 for c in obj.retriever.term_counts for k, v in c.items()), "Invalid cached term counts")
        obj.retriever.df = Counter(term for c in obj.retriever.term_counts for term in c)
        obj.retriever.avg_len = sum(sum(c.values()) for c in obj.retriever.term_counts) / max(1, len(documents))
        require(dict(obj.retriever.df) == body["df"] and obj.retriever.avg_len == body["avg_len"], "Invalid cached BM25 statistics")
        obj.retriever.dense = obj.retriever.dense_identity = None
        obj.identity = digest([obj.snapshot_hash, tokenizer.identity, config.atom_tokens, config.context_each_side, documents])
        require(obj.identity == body["identity"], "Corpus identity mismatch")
        obj.build_seconds, obj.load_seconds, obj.cache_hash = body["build_seconds"], perf_counter()-start, cache_hash
        return obj

    def scores(self, query):
        values = self.retriever.lexical_scores(query)
        scale = max(values, default=0)
        return {aid: max(0., score) / scale if scale > 0 else 0.
                for aid, score in zip(self.order, values)}

    def render(self, selected):
        spans = []
        ordered = sorted((self.atoms[a] for a in selected), key=lambda a: (
            natural(self.sources[a["source_id"]]["session_id"]), natural(a["source_id"]), a["start"]))
        for atom in ordered:
            if spans and spans[-1]["source_id"] == atom["source_id"] and spans[-1]["end"] == atom["start"]:
                spans[-1]["end"] = atom["end"]
            else:
                spans.append({k: atom[k] for k in ("source_id", "start", "end")})
        sections = []
        for span in spans:
            src = self.sources[span["source_id"]]
            header = "[{}:{}-{}]".format(span["source_id"], span["start"], span["end"])
            meta = {"session_datetime": src.get("session_datetime"),
                    "datetime_provenance": src.get("datetime_provenance", "unavailable"),
                    "speaker": src.get("speaker", "unknown")}
            sections.append(header + "\n" + json.dumps(meta, ensure_ascii=False, sort_keys=True) +
                            "\n" + src["text"][span["start"]:span["end"]])
        text = "\n\n".join(sections)
        return {"text": text, "text_hash": digest(text), "spans": spans,
                "atom_ids": sorted(selected), "used_tokens": self.tokenizer.count(text),
                "tokenizer": self.tokenizer.identity}
