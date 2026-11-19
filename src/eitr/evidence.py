"""Frozen non-overlapping raw atoms, full-context actions, and exact token budgets."""
from collections import defaultdict
import hashlib
import json
import re
from time import perf_counter
from eicr.contracts import digest, require
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
