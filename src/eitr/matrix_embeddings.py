"""Explicit embedding identity, bounded transport, and content-addressed replay."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse
import numpy as np
from eicr.contracts import digest, require, read_json, write_json, ContractError
from eicr.provider import _open_no_redirect, _strict_json, ProviderTransportError


def unit_rows(values, count, dimension):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (count, dimension) and np.isfinite(x).all(), 'Invalid embedding shape/nonfinite values')
    norms = np.linalg.norm(x, axis=1)
    require(np.isfinite(norms).all() and (norms > 1e-12).all(), 'Zero/overflow embedding row')
    return x / norms[:, None]


class EmbeddingBackend:
    """OpenAI-compatible /embeddings endpoint; revision is a user attestation.

    Hosted aliases cannot pin server weights. Stored vectors pin actual results.
    Query/document prefixes are part of identity, including empty prefixes.
    """
    def __init__(self, spec):
        require(isinstance(spec, dict) and set(spec) == {
            'endpoint', 'model', 'revision', 'key_env', 'dimension', 'query_prefix',
            'document_prefix', 'batch_size', 'cache_dir', 'cache_only'}, 'Pin all embedding settings')
        p = urlparse(spec['endpoint'])
        require(p.scheme == 'https' and p.hostname and not any((p.username,p.password,p.query,p.fragment)), 'Invalid embedding endpoint')
        require(all(isinstance(spec[k], str) and spec[k] for k in ('model','revision','key_env')), 'Missing embedding identity')
        require(all(isinstance(spec[k], str) for k in ('query_prefix','document_prefix')), 'Invalid embedding prefixes')
        require(type(spec['dimension']) is int and spec['dimension'] > 0 and
                type(spec['batch_size']) is int and 0 < spec['batch_size'] <= 256, 'Invalid embedding limits')
        require(type(spec['cache_only']) is bool and isinstance(spec['cache_dir'],str)
                and Path(spec['cache_dir']).is_absolute(), 'Pin absolute embedding cache')
        self.spec, self.calls = dict(spec), []
        self.identity = {k:spec[k] for k in ('model','revision','dimension','query_prefix','document_prefix')}
        self.identity.update(provider='openai-compatible-embeddings', endpoint_hash=digest(spec['endpoint']))

    def embed(self, texts, role):
        key = os.environ.get(self.spec['key_env'])
        require(key, 'Missing embedding credential variable: '+self.spec['key_env'])
        payload = {'model':self.spec['model'], 'input':[self.spec[role+'_prefix']+t for t in texts], 'encoding_format':'float', 'dimensions':self.spec['dimension']}
        trace = {'model':self.spec['model'], 'billing_kind':'embedding_tokens', 'status':'started', 'request_hash':digest(payload)}
        self.calls.append(trace); tick = perf_counter()
        try:
            req = urllib.request.Request(self.spec['endpoint'], data=json.dumps(payload).encode(),
                headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            with _open_no_redirect(req, timeout=60) as response:
                raw = response.read(32*1024*1024+1)
            require(len(raw) <= 32*1024*1024, 'Embedding response too large')
            data = _strict_json(raw)
            rows = data.get('data')
            require(isinstance(rows,list) and len(rows)==len(texts), 'Incomplete embedding response')
            by_id = {}
            for row in rows:
                i = row.get('index')
                require(type(i) is int and 0 <= i < len(texts) and i not in by_id, 'Invalid embedding response indices')
                by_id[i] = row['embedding']
            vectors = unit_rows([by_id[i] for i in range(len(texts))],len(texts),self.identity['dimension'])
            u = data.get('usage') or {}; n = u.get('prompt_tokens',u.get('total_tokens'))
            trace.update(status='ok', usage={'input_tokens':n,'output_tokens':0} if type(n) is int and n>=0 else {}, response_hash=digest(data))
            return vectors.tolist()
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            trace['status']='error'
            raise ProviderTransportError('Embedding transport failed ('+type(error).__name__+')') from None
        except (ContractError,ValueError,TypeError,KeyError,AttributeError) as error:
            trace['status']='error'
            if isinstance(error,ContractError): raise
            raise ContractError('Invalid embedding response ('+type(error).__name__+')') from None
        finally: trace['seconds']=perf_counter()-tick


def embed_record(texts, role, scope, backend, *, cache_dir=None, cache_only=None, batch_size=None):
    defaults = getattr(backend, 'spec', {})
    cache_dir = cache_dir if cache_dir is not None else defaults.get('cache_dir')
    cache_only = cache_only if cache_only is not None else defaults.get('cache_only', False)
    batch_size = batch_size if batch_size is not None else defaults.get('batch_size', 64)
    require(type(cache_only) is bool, 'Invalid embedding cache mode')
    require(role in ('query','document') and isinstance(texts,list) and texts and
            all(isinstance(t,str) and t.strip() for t in texts), 'Invalid embedding input')
    require(type(batch_size) is int and 0 < batch_size <= 256, 'Invalid embedding batch size')
    request = {'schema':'eitr-matrix-embedding-request-1', 'backend':backend.identity,
               'role':role, 'scope':scope, 'texts_hash':digest(texts), 'count':len(texts)}
    path = Path(cache_dir)/(digest(request)+'.json') if cache_dir else None
    if path and path.exists():
        record = read_json(path); validate_record(record,request)
        return record
    require(not cache_only, 'Required embedding cache missing')
    vectors=[]; start=len(backend.calls)
    for offset in range(0,len(texts),batch_size):
        part=texts[offset:offset+batch_size]
        vectors.extend(unit_rows(backend.embed(part,role),len(part),backend.identity['dimension']).tolist())
    record={'request':request,'vectors':vectors,'calls':list(backend.calls[start:])}
    record['record_hash']=digest(record)
    validate_record(record,request)
    if path: write_json(path,record)
    return record


def validate_record(record, request):
    require(record.get('request')==request and record.get('record_hash')==digest(
        {k:v for k,v in record.items() if k!='record_hash'}), 'Embedding record scope/hash mismatch')
    require(isinstance(record.get('calls'),list), 'Embedding record lacks accounting')
    return unit_rows(record['vectors'],request['count'],request['backend']['dimension'])


def embedding_options(backend):
    return {k:backend.spec[k] for k in ('cache_dir','cache_only','batch_size')}
