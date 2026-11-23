"""Explicit embedding identity and complete provider attempt receipts."""
from copy import deepcopy
import os
import time
from eicr.contracts import require
from .core import EmbeddingIdentity, EncoderUnavailable, unit


class Encoder:
    def __init__(self, spec, trace):
        self.identity = EmbeddingIdentity(**spec['identity'])
        self.spec, self.trace = spec, trace
        self.client = self.local = None

    def encode(self, text):
        identity, spec = self.identity, self.spec
        prepared = identity.input_text(text)
        if identity.provider == 'local':
            started, cpu = time.perf_counter(), time.process_time()
            row = {'stage': self.trace.stage, 'billing_kind': 'local_embedding', 'provider': 'local',
                   'model': identity.model, 'request': {'input': prepared}, 'status': 'error',
                   'embedding_identity': spec['identity'], 'cache_hit': False}
            self.trace.calls.append(row)
            self.trace.flush()
            try:
                if self.local is None:
                    from sentence_transformers import SentenceTransformer
                    self.local = SentenceTransformer(identity.model, revision=identity.revision,
                                                     device=spec.get('device', 'cpu'))
                vector = self.local.encode([prepared], convert_to_numpy=True, normalize_embeddings=True)[0]
                row['status'] = 'ok'; row['response'] = vector.tolist()
            except (ImportError, OSError, RuntimeError) as exc:
                row['error'] = str(exc); raise EncoderUnavailable(str(exc)) from exc
            finally:
                row['wall_seconds'] = time.perf_counter()-started; row['cpu_seconds'] = time.process_time()-cpu
                self.trace.flush()
            return unit(vector, identity.dimension)
        from openai import OpenAI, OpenAIError
        if self.client is None:
            key = os.environ.get(spec['api_key_env'])
            if not key: raise EncoderUnavailable('Missing embedding API key')
            self.client = OpenAI(api_key=key, base_url=spec['endpoint'], timeout=120, max_retries=0)
        request = {'model': identity.model, 'input': [prepared]}
        if spec.get('send_dimensions'): request['dimensions'] = identity.dimension
        for attempt in range(1, 4):
            started, cpu = time.perf_counter(), time.process_time()
            row = {'stage': self.trace.stage, 'billing_kind': 'embedding_tokens', 'provider': identity.provider,
                   'model': identity.model, 'endpoint': spec['endpoint'], 'request': deepcopy(request),
                   'attempt': attempt, 'status': 'error', 'usage': None, 'cache_hit': False,
                   'embedding_identity': spec['identity']}
            self.trace.calls.append(row)
            self.trace.flush()
            try:
                response = self.client.embeddings.create(**request)
                row['response'] = response.model_dump(); row['usage'] = row['response'].get('usage')
                require(len(response.data) == 1 and response.data[0].index == 0, 'Invalid embedding response indices')
                vector = unit(response.data[0].embedding, identity.dimension)
                row['status'] = 'ok'; return vector
            except (OpenAIError, TimeoutError) as exc:
                row['error'] = str(exc)
                if attempt == 3: raise EncoderUnavailable(str(exc)) from exc
            finally:
                row['wall_seconds'] = time.perf_counter()-started; row['cpu_seconds'] = time.process_time()-cpu
                self.trace.flush()
            time.sleep(1.5**attempt)


def validate_prices(prices):
    if prices is None: return
    require(set(prices) == {'currency','effective_date','source','rates'} and
            all(prices[k] for k in ('currency','effective_date','source')), 'Incomplete price provenance')
    seen = set()
    import math
    for row in prices['rates']:
        require(set(row) == {'provider','model','endpoint','billing_kind','input_per_million',
                             'output_per_million','cached_input_per_million'}, 'Incomplete rate identity')
        key = tuple(row[k] for k in ('provider','model','endpoint','billing_kind'))
        require(key not in seen and all(key), 'Duplicate or empty rate identity'); seen.add(key)
        for k in ('input_per_million','output_per_million','cached_input_per_million'):
            v = row[k]
            require((k == 'cached_input_per_million' and v is None) or
                    (type(v) in (int,float) and math.isfinite(v) and v >= 0), 'Invalid rate')


def bill(calls, prices):
    validate_prices(prices)
    rates = {tuple(r[k] for k in ('provider','model','endpoint','billing_kind')):r
             for r in prices['rates']} if prices else {}
    rows = []
    for call in calls:
        amount, reason = None, None
        if not isinstance(call,dict):
            rows.append({'stage':'unclassified','amount':None,'unknown_reason':'malformed_call'})
            continue
        if call.get('billing_kind') == 'local_embedding':
            amount = 0.  # API-only scope; CPU time is reported separately.
        else:
            rate = rates.get(tuple(call.get(k) for k in ('provider','model','endpoint','billing_kind')))
            usage = call.get('usage') or {}
            malformed = not isinstance(usage, dict)
            if malformed: usage = {}
            inp = usage.get('prompt_tokens', usage.get('input_tokens'))
            out = usage.get('completion_tokens', usage.get('output_tokens'))
            if call.get('billing_kind') == 'embedding_tokens' and inp is not None: out = 0
            details = usage.get('prompt_tokens_details') or {}
            malformed = malformed or not isinstance(details, dict)
            cached = usage.get('cached_input_tokens', details.get('cached_tokens') if isinstance(details, dict) else None)
            if rate is None: reason = 'missing_price'
            elif malformed: reason = 'malformed_usage'
            elif not all(type(v) is int and v >= 0 for v in (inp,out)): reason = 'missing_usage'
            elif rate['cached_input_per_million'] is not None and not (type(cached) is int and 0 <= cached <= inp):
                reason = 'missing_cache_usage'
            else:
                cached = cached if rate['cached_input_per_million'] is not None else 0
                amount = ((inp-cached)*rate['input_per_million']+out*rate['output_per_million']+
                          cached*(rate['cached_input_per_million'] or 0))/1e6
        rows.append({'stage':call.get('stage','unclassified'), 'amount':amount, 'unknown_reason':reason})
    known = sum(r['amount'] for r in rows if r['amount'] is not None)
    complete = all(r['amount'] is not None for r in rows)
    return {'currency': prices['currency'] if prices else None, 'complete':complete,
            'amount':known if complete else None, 'known_subtotal':known,
            'unknown_calls':sum(r['amount'] is None for r in rows), 'calls':rows}
