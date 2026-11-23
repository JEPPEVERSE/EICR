"""Scoped evidence views, fixed top-three cosine, and teacher-scale calibration."""
from dataclasses import asdict, dataclass
from collections import defaultdict
from copy import deepcopy
import math
import time
import numpy as np
from eicr.contracts import digest, require


@dataclass(frozen=True)
class Scope:
    owner: str
    snapshot: str
    graph_hash: str

    def __post_init__(self):
        require(all(isinstance(v, str) and v for v in asdict(self).values()), 'Incomplete graph scope')


@dataclass(frozen=True)
class EmbeddingIdentity:
    provider: str
    model: str
    revision: str
    dimension: int
    preprocessing: str = 'replace_newlines_strip'
    normalization: str = 'l2'

    def __post_init__(self):
        require(all(isinstance(getattr(self, k), str) and getattr(self, k)
                    for k in ('provider', 'model', 'revision')), 'Unverified embedding identity')
        require('REPLACE' not in self.revision.upper(), 'Replace the embedding revision template with a verified identity')
        require(type(self.dimension) is int and self.dimension > 0, 'Invalid dimension')
        require(self.preprocessing in ('replace_newlines_strip', 'identity') and self.normalization == 'l2',
                'Unsupported embedding convention')

    def input_text(self, text):
        return text.replace('\n', ' ').strip() if self.preprocessing == 'replace_newlines_strip' else text


def unit(vector, dimension):
    v = np.asarray(vector, dtype=np.float64)
    require(v.shape == (dimension,) and np.isfinite(v).all(), 'Invalid embedding vector/dimension')
    norm = float(np.linalg.norm(v))
    require(norm > 0, 'Zero embedding is an identity/asset error')
    return v / norm


def source_ids(origin):
    values = origin.split(',') if isinstance(origin, str) else origin
    require(isinstance(values, (list, tuple)) and bool(values), 'Missing source identity')
    result = tuple(sorted(set(x.strip() for x in values)))
    require(all(result), 'Empty source identity')
    return result


class EvidenceView:
    """One vector table; native (key,tag) links, never a global tag union.

    Duplicate full source sets use the lexicographically first event ID in each
    candidate view. This fixed, query-independent convention is part of v1.
    Native events and tools are never mutated or deduplicated.
    """
    def __init__(self, memory, scope, identity, encoded_texts, *, provenance):
        require(provenance.get('identity_verified') is True, 'Embedding identity must be attested')
        require(provenance.get('input_texts_verified') is True, 'Embedding input identity must be attested')
        self.scope, self.identity = scope, identity
        self.rows, self.sources, self.texts, self.tool_texts = {}, {}, {}, {}
        require(set(encoded_texts) == set(memory.episode_events), 'Incomplete embedding input manifest')
        for eid, event in memory.episode_events.items():
            require(isinstance(encoded_texts[eid], str), 'Missing encoded text')
            self.rows[eid] = unit(event.embedding, identity.dimension)
            self.sources[eid] = source_ids(event.origin)
            self.texts[eid] = encoded_texts[eid]
            self.tool_texts[eid] = event.text
        self.links = {}
        for key in memory.keys:
            for tag in memory.get_tag_list(key):
                _, _, ids = memory.event_by_tag(key, tag)
                require(set(ids) <= set(self.rows), 'Dangling native event link')
                seen, representatives = set(), []
                for eid in sorted(set(ids)):
                    if self.sources[eid] not in seen:
                        seen.add(self.sources[eid]); representatives.append(eid)
                self.links[key, tag] = tuple(representatives)
        self.identity_hash = digest({'scope': asdict(scope), 'embedding': asdict(identity),
            'inputs': self.texts, 'tool_texts': self.tool_texts, 'sources': self.sources,
            'rows': {k: v.tolist() for k, v in self.rows.items()},
            'links': [[k, t, list(v)] for (k, t), v in self.links.items()]})
        self.last_dot_products = 0

    def validate_request(self, key, tags, scope):
        require(scope == self.scope, 'Cross-owner/snapshot/graph request')
        require(len(tags) == len(set(tags)), 'Duplicate candidate tag')
        require(all((key, tag) in self.links for tag in tags), 'Illegal native key/tag candidate')

    def score(self, vector, key, tags, scope, identity):
        self.validate_request(key, tags, scope)
        require(identity == self.identity, 'Query/event embedding identity mismatch')
        q = unit(vector, identity.dimension)
        ids = sorted({eid for tag in tags for eid in self.links[key, tag]})
        dots = {eid: max(0., min(1., float(np.dot(q, self.rows[eid])))) for eid in ids}
        self.last_dot_products = len(dots)
        return {tag: float(np.mean(sorted((dots[e] for e in self.links[key, tag]), reverse=True)[:3]))
                if self.links[key, tag] else 0. for tag in tags}


@dataclass(frozen=True)
class Isotonic:
    x: tuple
    y: tuple

    def __post_init__(self):
        require(len(self.x) == len(self.y) > 0, 'Empty calibration')
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in (*self.x, *self.y)), 'Invalid calibration knots')
        require(all(a < b for a, b in zip(self.x, self.x[1:])), 'Calibration x must increase')
        require(all(a <= b for a, b in zip(self.y, self.y[1:])), 'Calibration must be monotone')

    def predict(self, value):
        require(math.isfinite(value), 'Nonfinite raw score')
        return float(np.clip(np.interp(value, self.x, self.y), 0, 1))

    @classmethod
    def fit(cls, points):
        by_x = defaultdict(lambda: [0., 0.])
        for x, y, weight in points:
            require(all(math.isfinite(v) for v in (x, y, weight)) and 0 <= x <= 1 and 0 <= y <= 1
                    and weight > 0, 'Invalid calibration point')
            by_x[x][0] += weight; by_x[x][1] += weight*y
        require(by_x, 'No calibration points')
        xs, blocks = sorted(by_x), []
        for i, x in enumerate(xs):
            w, wy = by_x[x]
            blocks.append([i, i, w, wy])
            while len(blocks) > 1 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
                b, a = blocks.pop(), blocks.pop()
                blocks.append([a[0], b[1], a[2]+b[2], a[3]+b[3]])
        ys = [0.]*len(xs)
        for start, end, weight, weighted_y in blocks:
            ys[start:end+1] = [weighted_y/weight]*(end-start+1)
        return cls(tuple(xs), tuple(ys))


def fit_calibration(records, development_ids, evaluation_ids=(), development_questions=None):
    require(0 < len(development_ids) <= 16 and len(set(development_ids)) == len(development_ids),
            'Freeze at most 16 distinct development questions')
    require(not set(development_ids) & set(evaluation_ids), 'Calibration/evaluation leakage')
    groups, seen = defaultdict(list), set()
    for r in records:
        require(set(r) <= {'request_id', 'question_id', 'stage', 'raw', 'teacher', 'view_hash', 'teacher_calls', 'embedding_identity'},
                'Calibration contains unapproved fields (gold/labels are forbidden)')
        require(r['question_id'] in development_ids and r['request_id'] not in seen, 'Invalid/duplicate teacher request')
        seen.add(r['request_id'])
        require(r['stage'] in ('initial', 'expansion'), 'Unknown stage')
        require(r['raw'] and set(r['raw']) == set(r['teacher']), 'Incomplete teacher response')
        # Validate before counting support.
        Isotonic.fit([(x, r['teacher'][t], 1/len(r['raw'])) for t, x in r['raw'].items()])
        groups[r['stage']].append(r)
    def supported(rows):
        bins = {('zero' if y == 0 else 'high' if y >= .7 else 'low')
                for r in rows for y in r['teacher'].values()}
        return len(rows) >= 20 and bins == {'zero', 'low', 'high'}
    def fit(rows):
        return asdict(Isotonic.fit([(x, r['teacher'][t], 1/len(r['raw']))
                                   for r in rows for t, x in r['raw'].items()]))
    require(len({digest(r['embedding_identity']) for r in records}) <= 1, 'Mixed calibration embedding identities')
    maps, sources = {}, {}
    for stage in ('initial', 'expansion'):
        rows = groups[stage]
        if supported(rows):
            maps[stage], sources[stage] = fit(rows), stage
        elif supported(records):
            maps[stage], sources[stage] = fit(records), 'pooled'
        else:
            maps[stage], sources[stage] = None, 'insufficient_support'
    body = {'schema': 'native-tag-calibration-1', 'development_ids': list(development_ids),
            'embedding_identities': [dict(x) for x in sorted({tuple(sorted(r['embedding_identity'].items())) for r in records})],
            'development_text_hashes': sorted({digest(q.strip()) for q in (development_questions or [])}),
            'records_hash': digest(records), 'view_hashes': sorted({r['view_hash'] for r in records}),
            'maps': maps, 'sources': sources, 'counts': {s: len(groups[s]) for s in ('initial', 'expansion')},
            'ready': all(maps.values()), 'interpolation': 'linear_clamped_endpoints_float64',
            'tie_order': 'native_response_insertion_order; cheap=candidate_order',
            'weights': 'each_request_total_weight_one'}
    return {**body, 'hash': digest(body)}


class EncoderUnavailable(RuntimeError):
    """Only availability errors may trigger the explicitly billed native fallback."""


class Selector:
    def __init__(self, mode, view, encoder, calibration=None, *, shadow=False, replay=None, collect_raw=False):
        require(mode in ('native', 'cheap_all', 'cheap_initial'), 'Unknown selector mode')
        require(encoder.identity == view.identity, 'Incompatible query encoder')
        self.mode, self.view, self.encoder = mode, view, encoder
        self.calibration, self.shadow, self.replay = calibration, shadow, replay
        self.collect_raw = collect_raw
        self.decisions = []
        self.existing_query = None
        if calibration is not None:
            body = {k:v for k,v in calibration.items() if k != 'hash'}
            require(digest(body) == calibration['hash'], 'Calibration hash mismatch')
            require(asdict(view.identity) in calibration['embedding_identities'], 'Calibration/embedding mismatch')
        if (mode != 'native' or shadow) and replay is None:
            require(calibration is not None and calibration['ready'], 'Calibration not ready')

    def score_tags(self, context_text, key_id, ordered_tag_ids, stage, graph_scope, native):
        require(stage in ('initial', 'expansion'), 'Unknown scoring stage')
        tags = list(ordered_tag_ids)
        self.view.validate_request(key_id, tags, graph_scope)
        record = {'context_text': context_text, 'key_id': key_id, 'ordered_tag_ids': tags,
                  'stage': stage, 'graph_scope': asdict(graph_scope), 'view_hash': self.view.identity_hash,
                  'fallback': False, 'mode': self.mode}
        self.decisions.append(record)
        cheap = self.mode == 'cheap_all' or (self.mode == 'cheap_initial' and stage == 'initial')
        if self.replay is not None and cheap:
            result = self.replay(record)
            record['response'] = deepcopy(result)
            record['replay'] = True
            return result
        if not cheap:
            result = native()
            record['response'] = deepcopy(result)
            record['teacher_response'] = deepcopy(result)
            if not self.shadow and not self.collect_raw:
                return result
        started, cpu = time.perf_counter(), time.process_time()
        try:
            if self.existing_query is not None and self.existing_query[0] == context_text:
                vector = self.existing_query[1]; record['query_reuse'] = 'same_request_exact_context'
            else:
                vector = self.encoder.encode(context_text); record['query_reuse'] = 'none'
            raw = self.view.score(vector, key_id, tags, graph_scope, self.encoder.identity)
            record['raw'] = raw
            record['dot_products'] = self.view.last_dot_products
            if self.calibration is None:
                return result
            mapping = Isotonic(**self.calibration['maps'][stage])
            mapped = {'tag_scores': {t: mapping.predict(raw[t]) for t in tags}}
            if self.shadow:
                record['shadow_response'] = mapped
            elif cheap:
                result = mapped; record['response'] = deepcopy(result)
        except EncoderUnavailable as exc:
            record['fallback_reason'] = str(exc)
            if cheap:
                record['fallback'] = True
                result = native(); record['response'] = deepcopy(result)
            else:
                record['shadow_error'] = str(exc)
        finally:
            record['scoring_seconds'] = time.perf_counter()-started
            record['scoring_cpu_seconds'] = time.process_time()-cpu
        return result


def decision_sets(response, limit):
    scores = (response or {}).get('tag_scores') or {}
    ordered = sorted(((t, s) for t, s in scores.items() if s != 0), key=lambda x:x[1], reverse=True)[:limit]
    return {'selected': [t for t, _ in ordered], 'early': [t for t, s in ordered if s >= .7],
            'zero': [t for t, s in scores.items() if s == 0]}
