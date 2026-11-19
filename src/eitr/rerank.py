"""Explicit semantic reranking with content-addressed, auditable replay.

DashScope scores are request-relative: never split a candidate pool into batches
and compare their raw scores. No network calls occur unless a backend is passed.
"""
import json
import math
import os
from pathlib import Path
from time import perf_counter
import urllib.error
import urllib.request
from urllib.parse import urlparse
from eicr.contracts import ContractError, digest, require, read_json, write_json
from eicr.provider import ProviderTransportError, _open_no_redirect, _strict_json


class DashScopeReranker:
    def __init__(self, endpoint, model, key_env, *, max_documents=500, timeout=60):
        parsed = urlparse(endpoint)
        require(parsed.scheme == 'https' and parsed.hostname and not any(
            (parsed.username, parsed.password, parsed.query, parsed.fragment)), 'Invalid rerank endpoint')
        require(isinstance(model, str) and model and isinstance(key_env, str) and key_env,
                'Explicit rerank model and credential variable required')
        require(type(max_documents) is int and 0 < max_documents <= 500, 'Invalid rerank document cap')
        require(type(timeout) in (float, int) and math.isfinite(timeout) and 0 < timeout <= 60,
                'Rerank timeout must be in (0, 60]')
        self.endpoint, self.model, self.key_env = endpoint, model, key_env
        self.max_documents, self.timeout = max_documents, timeout
        self.identity = {'provider': 'dashscope-text-rerank', 'endpoint_hash': digest(endpoint),
                         'model': model, 'max_documents': max_documents,
                         'score_contract': 'single-request-relative-[0,1]',
                         'version_pinning': 'hosted alias; exact scores pinned in cache'}
        self.calls = []

    def score(self, question, documents):
        require(0 < len(documents) <= self.max_documents,
                'Pool exceeds single-request rerank cap; freeze a smaller common pool, never compare scores across batches')
        key = os.environ.get(self.key_env)
        require(key, 'Missing rerank credential variable: ' + self.key_env)
        payload = {'model': self.model, 'input': {'query': question, 'documents': documents},
                   'parameters': {'top_n': len(documents), 'return_documents': False}}
        trace = {'model': self.model, 'endpoint_hash': digest(self.endpoint), 'request_hash': digest(payload),
                 'documents': len(documents), 'status': 'started', 'billing_kind': 'rerank_tokens'}
        self.calls.append(trace)
        tick = perf_counter()
        try:
            request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            with _open_no_redirect(request, timeout=self.timeout) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
            require(len(raw) <= 4 * 1024 * 1024, 'Rerank response exceeds byte cap')
            data = _strict_json(raw)
            require(isinstance(data, dict) and not data.get('code'), 'Rerank provider reported failure')
            values = data.get('output', {}).get('results')
            require(isinstance(values, list) and len(values) == len(documents), 'Incomplete rerank results')
            scores = {}
            for row in values:
                i, s = row.get('index'), row.get('relevance_score')
                require(type(i) is int and 0 <= i < len(documents) and i not in scores,
                        'Invalid/duplicate rerank result index')
                require(type(s) in (float, int) and math.isfinite(s) and 0 <= s <= 1, 'Invalid rerank score')
                scores[i] = float(s)
            u = data.get('usage') or {}
            total = u.get('total_tokens')
            trace['usage'] = ({'input_tokens': total, 'output_tokens': 0, 'total_tokens': total}
                              if type(total) is int and total >= 0 else {})
            trace.update(status='ok', response_hash=digest(data), request_id=data.get('request_id'))
            return [scores[i] for i in range(len(documents))]
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            trace['status'] = 'error'
            raise ProviderTransportError('Rerank transport failed (' + type(error).__name__ + ')') from None
        except (ContractError, ValueError, KeyError, TypeError, AttributeError) as error:
            trace['status'] = 'error'
            if isinstance(error, ContractError):
                raise
            raise ContractError('Invalid rerank response (' + type(error).__name__ + ')') from None
        finally:
            trace['seconds'] = perf_counter() - tick


def rerank_documents(corpus, atom_ids):
    """Score a focal utterance with bounded conversational context for pronouns.

    Speaker/date are retained. An explicit focus marker distinguishes the scored
    atom from helpful adjacent utterances. Gold answers never enter this input.
    """
    admitted = set(atom_ids)
    return ['FOCUS PASSAGE\n' + corpus.render({aid})['text'] +
            '\nADJACENT CONTEXT\n' + corpus.render((set(corpus.contexts[aid]) & admitted) - {aid})['text']
            for aid in atom_ids]


def rerank_pool(question, corpus, pool, backend, *, cache_dir=None, cache_only=False):
    require(backend is not None, 'Semantic arms require an explicit reranker')
    require(pool['corpus_hash'] == corpus.identity and pool['pool_hash'] == digest(
        {k: v for k, v in pool.items() if k != 'pool_hash'}), 'Invalid rerank pool')
    ids = sorted(pool['atom_ids'])
    documents = rerank_documents(corpus, ids)
    request = {'schema': 'eitr-rerank-request-1', 'question': question,
               'corpus_hash': corpus.identity, 'atom_ids': ids, 'documents_hash': digest(documents),
               'backend': backend.identity}
    key = digest(request)
    path = Path(cache_dir) / (key + '.json') if cache_dir else None
    if path is not None and path.exists():
        result = read_json(path)
        require(result.get('request') == request and result.get('record_hash') == digest(
            {k: v for k, v in result.items() if k != 'record_hash'}), 'Stale/corrupt rerank cache')
        reused = True
    else:
        require(not cache_only, 'Required rerank cache record missing')
        start = len(backend.calls)
        scores = backend.score(question, documents) if ids else []
        require(len(scores) == len(ids), 'Incomplete semantic scores')
        # Validate scores before writing even for injected backends.
        require(all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in scores),
                'Invalid semantic scores')
        result = {'request': request, 'scores': dict(zip(ids, scores)), 'calls': backend.calls[start:]}
        result['record_hash'] = digest(result)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, result)
        reused = False
    require(set(result['scores']) == set(ids) and all(type(v) in (float, int) and math.isfinite(v)
            and 0 <= v <= 1 for v in result['scores'].values()), 'Invalid cached semantic scores')
    return {**result, 'cache_reused': reused,
            'cost_note': 'Retain original rerank calls as independent deployment cost; cache replay is not a free deployment.'}


def backend_from_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'provider', 'endpoint', 'model', 'key_env',
            'max_documents', 'cache_dir', 'cache_only'}, 'Pin all reranker settings')
    require(spec['provider'] == 'dashscope' and type(spec['cache_only']) is bool
            and isinstance(spec['cache_dir'], str) and Path(spec['cache_dir']).is_absolute(), 'Invalid reranker spec')
    return DashScopeReranker(spec['endpoint'], spec['model'], spec['key_env'], max_documents=spec['max_documents'])
