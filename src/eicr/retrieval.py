"""Shared lexical/hybrid ranking plus bounded graph-origin evidence queues."""

import math
from collections import Counter, defaultdict

import numpy as np

from .contracts import require, tokens


class SharedRetriever:
    def __init__(self, chunks, dense_vectors=None, dense_identity=None):
        self.chunks = chunks
        self.term_counts = [Counter(tokens(c["text"])) for c in chunks]
        self.df = Counter(term for counts in self.term_counts for term in counts)
        self.avg_len = sum(sum(c.values()) for c in self.term_counts) / max(1, len(chunks))
        self.dense = None
        self.dense_identity = dense_identity
        if dense_vectors is not None:
            array = np.asarray(dense_vectors, dtype=np.float64)
            require(array.ndim == 2 and len(array) == len(chunks) and np.isfinite(array).all(), "Invalid dense chunk matrix")
            norms = np.linalg.norm(array, axis=1)
            require((norms > 0).all() and bool(dense_identity), "Dense vectors need nonzero norms and a model identity")
            self.dense = array / norms[:, None]

    def lexical_scores(self, query):
        terms = set(tokens(query))
        scores = []
        n = len(self.chunks)
        for counts in self.term_counts:
            length, score = sum(counts.values()), 0.0
            for term in terms:
                tf = counts[term]
                if tf:
                    idf = math.log(1 + (n-self.df[term]+0.5)/(self.df[term]+0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / (self.avg_len or 1)))
            scores.append(score)
        return scores

    def rank(self, query, query_vector=None, limit=256):
        lexical = self.lexical_scores(query)
        lex_order = sorted((i for i, s in enumerate(lexical) if s > 0), key=lambda i: (-lexical[i], self.chunks[i]["span_id"]))
        if query_vector is None:
            require(self.dense is None, "Dense index requires an explicitly supplied query vector")
            return [dict(self.chunks[i], retrieval_score=lexical[i]) for i in lex_order[:limit]]
        require(self.dense is not None, "Query vector without a dense chunk index")
        vector = np.asarray(query_vector, dtype=np.float64)
        require(vector.shape == (self.dense.shape[1],) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0, "Invalid query vector")
        similarity = self.dense @ (vector / np.linalg.norm(vector))
        dense_order = sorted(range(len(self.chunks)), key=lambda i: (-similarity[i], self.chunks[i]["span_id"]))[:limit]
        fused = defaultdict(float)
        for order in (lex_order[:limit], dense_order):
            for rank, i in enumerate(order, 1):
                fused[i] += 1/(60 + rank)
        order = sorted(fused, key=lambda i: (-fused[i], self.chunks[i]["span_id"]))
        return [dict(self.chunks[i], retrieval_score=fused[i]) for i in order[:limit]]


def collect_candidates(index, query_map, diagnostics, route, retriever, query_vector=None):
    require(route["index_hash"] == index.index_hash, "Route/index mismatch")
    question = query_map["original_question"]
    queues = {"original": retriever.rank(question, query_vector)}
    trace = {"retriever": "dense_bm25_rrf" if retriever.dense is not None else "bm25_only", "online_graph_hops": 0,
             "candidate_limit": 256, "dense_identity": retriever.dense_identity}
    if route["mode"] == "ORIGINAL":
        return queues, trace
    facts = index.records["evidence"]
    source_families = index.records["source_families"]
    def direct_sources(cue_ids):
        ids = set()
        for cid in cue_ids:
            ids.update(index.R[index.cue_pos[cid]].indices)
        return {o for j in ids for o in facts[j]["origins"]}
    def independent(cue_ids):
        source_ids = direct_sources(cue_ids)
        texts = []
        for cid in cue_ids:
            cue = index.records["cues"][index.cue_pos[cid]]
            texts.append(cue["text"])
            texts.extend(a["text"] for a in cue.get("aliases", []))
        scores = retriever.lexical_scores(" ".join(texts))
        families = defaultdict(list)
        for i, chunk in enumerate(retriever.chunks):
            if chunk["source_id"] in source_ids:
                families[source_families[chunk["source_id"]]].append((i, scores[i]))
        for family in families:
            families[family].sort(key=lambda row: (-row[1], retriever.chunks[row[0]]["span_id"]))
        order = sorted(families, key=lambda family: (-families[family][0][1], family))
        out = []
        for rank in range(max((len(v) for v in families.values()), default=0)):
            for family in order:
                if rank < len(families[family]):
                    i, score = families[family][rank]
                    out.append(dict(retriever.chunks[i], retrieval_score=score, family_id=family))
        return out[:256]
    groups = diagnostics["groups"]
    if route["mode"] == "SYMMETRIC_CHECK":
        queues["side_a"] = independent(groups[0]["members"])
        queues["side_b"] = independent(groups[1]["members"])
    else:
        selected = route["selected_cue_id"]
        queues["independent_cue"] = independent([selected])
        remaining = [g for g in groups if selected not in g["members"]]
        source_support = {}
        for j, fact in enumerate(facts):
            supporters = [g for g in remaining if index.R[index.cue_pos[g["representative"]], j] > 0]
            if supporters:
                for origin in fact["origins"]:
                    # Union group support at raw origin; do not count multiple facts as multiple votes.
                    state = source_support.setdefault(origin, {})
                    state.update({g["representative"]: g["weight"] for g in supporters})
        lexical = retriever.lexical_scores(query_map.get("answer_need", ""))
        core = []
        for i, chunk in enumerate(retriever.chunks):
            support = source_support.get(chunk["source_id"], {})
            if support:
                core.append(dict(chunk, support_cues=sorted(support), support_count=len(support), support_weight=sum(support.values()), lexical_score=lexical[i]))
        core.sort(key=lambda c: (-c["support_count"], -c["support_weight"], -c["lexical_score"], c["span_id"]))
        queues["core"] = core[:256]
        scores = retriever.lexical_scores(query_map.get("answer_need", ""))
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], retriever.chunks[i]["span_id"]))
        queues["answer_need"] = [dict(retriever.chunks[i], retrieval_score=scores[i]) for i in order[:256]]
    trace["queue_sizes"] = {key: len(value) for key, value in queues.items()}
    return queues, trace
