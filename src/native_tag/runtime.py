"""Freeze local native sources and execute their full classes in a private namespace.

Only two exact AST-identified call expressions are substituted. No native reader,
answer prompt, tool, branch, sort or budget is reimplemented here. Credential and
CLI initialization are excluded; native definitions and configured constants are
compiled intact. The original and instrumented variants share this loader.
"""
import ast
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime
import difflib
import hashlib
import json
import logging
import math
import os
import uuid
from pathlib import Path
import random
import re
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from eicr.contracts import digest, require
from .core import Scope, EmbeddingIdentity, EvidenceView

FILES = ('common/config.py', 'common/model_config.py', 'common/utils.py', 'prompts/prompts.py',
         'prompts/schema.py', 'memory/system.py', 'memory/controller.py', 'agent/tools.py',
         'llm/controller.py', 'agent/agent.py', 'llm/embeddings.py', 'llm/rag_utils.py',
         'data/embed_rewrite.py')
LIMITS = ('K1', 'K2', 'TAG_MAX', 'TAG_LIMIT', 'TIME_EVENT_LIMIT', 'TOPIC_K', 'RERANK_LIMIT',
          'MAX_ROUNDS', 'MAX_TOOL_CALLS')
TARGETS = {'agent/agent.py': ('select_key_tag', 'question', 'initial'),
           'memory/controller.py': ('query_event_keywords_single', 'text', 'expansion')}
SOURCE_READERS = {'event_by_tag','query_event_context','query_conversation_time','query_topic_events',
                  'query_personal_information','query_personal_aspect'}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('w') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def patch_source(source, filename):
    if filename not in TARGETS:
        return source
    method, context, stage = TARGETS[filename]
    tree = ast.parse(source)
    matches = [n for c in tree.body if isinstance(c, ast.ClassDef)
               for f in c.body if isinstance(f, ast.FunctionDef) and f.name == method
               for n in ast.walk(f) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'key_out' for t in n.targets)]
    require(len(matches) == 1, 'Native scoring call site changed: '+filename)
    call = matches[0].value
    require(isinstance(call, ast.Call) and ast.unparse(call.func) == 'self.llm.chat_text',
            'Native tag call no longer matches explicit interface')
    lines = source.splitlines(keepends=True)
    # AST column offsets are UTF-8 byte offsets.
    start = sum(len(x.encode()) for x in lines[:call.lineno-1])+call.col_offset
    end = sum(len(x.encode()) for x in lines[:call.end_lineno-1])+call.end_col_offset
    original = source.encode()[start:end].decode()
    replacement = f'_tag_score(self, {context}, key, tag, "{stage}", lambda: {original})'
    return (source.encode()[:start]+replacement.encode()+source.encode()[end:]).decode()


def freeze_sources(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists(), 'Refusing to overwrite frozen runtime')
    require(not destination.is_relative_to(source), 'Keep experimental copy outside upstream')
    hashes, patched, diffs = {}, {}, []
    for name in FILES:
        text = (source/name).read_text(); adapted = patch_source(text, name)
        for subdir, data in (('reference', text), ('instrumented', adapted)):
            target = destination/subdir/name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(data)
        hashes[name] = file_hash(destination/'reference'/name)
        patched[name] = file_hash(destination/'instrumented'/name)
        diffs.extend(difflib.unified_diff(text.splitlines(True), adapted.splitlines(True),
                                       fromfile='reference/'+name, tofile='instrumented/'+name))
    def git(*args):
        return subprocess.run(['git', '-C', str(source), *args], capture_output=True, text=True, check=True).stdout
    (destination/'local-tracked.diff').write_text(git('diff', 'HEAD', '--', *FILES))
    (destination/'selector.diff').write_text(''.join(diffs))
    constants = {}
    for node in ast.parse((source/'common/config.py').read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in LIMITS:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    require(set(constants) == set(LIMITS), 'Cannot freeze native runtime limits')
    body = {'schema': 'native-tag-runtime-1', 'upstream': 'https://github.com/Ji-shuo/MRAgent',
            'head': git('rev-parse', 'HEAD').strip(), 'source_root': str(source),
            'reference': hashes, 'instrumented': patched, 'limits': constants,
            'tracked_diff_sha256': file_hash(destination/'local-tracked.diff'),
            'selector_diff_sha256': file_hash(destination/'selector.diff'),
            'untracked_runtime_files': [n for n in git('ls-files', '--others', '--exclude-standard', '--', *FILES).splitlines()],
            'loader_policy': 'full_native_definitions; isolated_config; two_explicit_tag_call_sites',
            'sdk_retries': 0, 'retry_policy': 'native_outer_retries_only; every_HTTP_attempt_logged',
            'reproduction': 'frozen_local_reference_with_disclosed_local_changes_not_clean_upstream'}
    manifest = {**body, 'hash': digest(body)}
    write(destination/'runtime_manifest.json', manifest)
    return manifest


def verify_sources(root):
    root = Path(root); manifest = json.loads((root/'runtime_manifest.json').read_text())
    require(digest({k:v for k,v in manifest.items() if k != 'hash'}) == manifest['hash'], 'Runtime manifest hash mismatch')
    for variant in ('reference', 'instrumented'):
        require(set(manifest[variant]) == set(FILES), 'Runtime file set mismatch')
        for name, sha in manifest[variant].items():
            require(file_hash(root/variant/name) == sha, 'Frozen source changed: '+name)
    for name, field in (('local-tracked.diff', 'tracked_diff_sha256'), ('selector.diff', 'selector_diff_sha256')):
        require(file_hash(root/name) == manifest[field], 'Frozen diff changed')
    for name in FILES:
        require(patch_source((root/'reference'/name).read_text(), name) == (root/'instrumented'/name).read_text(),
                'Unexpected downstream change: '+name)
    return manifest


def load_runtime(root, model, *, instrumented=True):
    root = Path(root); manifest = verify_sources(root)
    require(set(model) >= {'provider', 'model', 'endpoint'}, 'Pin chat provider/model/endpoint')
    config = SimpleNamespace(**manifest['limits'], MODEL=model['model'], RE_MODEL=model['model'],
                             MODEL_SORT=model['model'], MODEL_NAME=model.get('alias', model['model']),
                             PROVIDER=model['provider'], dataset='locomo')
    ns = {'__name__': 'native_tag.frozen', 'config': config, 'np': np, 'json': json, 're': re,
          'random': random, 'time': time, 'logging': logging, 'logger': logging.getLogger('native_tag.frozen'),
          'datetime': datetime, 'date': date, 'defaultdict': defaultdict,
          'API_STATS': dict.fromkeys(('calls', 'successful_calls', 'errors', 'input_tokens', 'output_tokens',
                                     'total_tokens', 'latency_seconds', 'rounds', 'tool_calls'), 0)}
    try:
        from openai import APIStatusError, APIConnectionError, APIResponseValidationError
        ns.update(APIStatusError=APIStatusError, APIConnectionError=APIConnectionError,
                  APIResponseValidationError=APIResponseValidationError)
    except ImportError:
        # Offline replay never creates provider exceptions. Live mode requires SDK.
        for name in ('APIStatusError', 'APIConnectionError', 'APIResponseValidationError'):
            ns[name] = type(name, (Exception,), {})
    def tag_score(obj, context, key, tags, stage, native):
        selector = getattr(obj, 'tag_selector', None)
        require(selector is not None, 'Explicit tag selector must be installed')
        trace = obj.llm.trace
        begin, position, started = len(trace.calls), len(selector.decisions), time.perf_counter()
        state = {'events':deepcopy(getattr(obj,'queried_event',getattr(getattr(obj,'memory_controller',None),'queried_event',[]))),
                 'keywords':deepcopy(getattr(obj,'queried_keyword',getattr(getattr(obj,'memory_controller',None),'queried_keyword',[])))}
        with trace.phase(stage+'_tag'):
            try:
                return selector.score_tags(context, key, tags, stage, obj.graph_scope, native)
            finally:
                if len(selector.decisions) > position:
                    selector.decisions[position]['call_indices'] = list(range(begin, len(trace.calls)))
                    selector.decisions[position]['wall_seconds'] = time.perf_counter()-started
                    selector.decisions[position]['state_before'] = state
    ns['_tag_score'] = tag_score
    variant = 'instrumented' if instrumented else 'reference'
    def install(name, only=None, target=None):
        tree = ast.parse((root/variant/name).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Assign))
                 and (only is None or getattr(n, 'name', None) in only
                      or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in only for t in n.targets)))]
        # Skip only process configuration/API counters, never class or function bodies.
        if name == 'llm/controller.py': nodes = [n for n in nodes if not isinstance(n, ast.Assign)]
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]+nodes,
                            type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(root/variant/name), 'exec'), target if target is not None else ns)
    install('common/model_config.py', {'prepare_chat_request'})
    install('common/utils.py')
    install('prompts/prompts.py')
    schema_ns = dict(ns); install('prompts/schema.py', target=schema_ns)
    ns['json_scheme'] = SimpleNamespace(**{k:v for k,v in schema_ns.items() if not k.startswith('__')})
    install('memory/system.py')
    install('llm/controller.py')
    install('memory/controller.py')
    install('agent/tools.py')
    install('agent/agent.py')
    return SimpleNamespace(**{k:v for k,v in ns.items() if not k.startswith('__')}, manifest=manifest)


class Trace:
    def __init__(self, path=None, metadata=None):
        self.stage = 'unclassified'; self.calls = []; self.events = []
        self.path, self.metadata = path, metadata or {}
        self.ledger_id = uuid.uuid4().hex
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.measurements = []
        if path is not None:
            require(not Path(path).exists(), 'Refusing to overwrite an attempt ledger')
            self.flush()

    def receipt(self, status='running', **extra):
        body = {'schema':'native-tag-ledger-2','ledger_id':self.ledger_id, **self.metadata,
                'status':status,'calls':clean(self.calls),'measurements':clean(self.measurements),
                'wall_seconds':time.perf_counter()-self.started,
                'cpu_seconds':time.process_time()-self.cpu_started, **extra}
        return {**body,'hash':digest(body)}

    def flush(self, status='running', **extra):
        receipt = self.receipt(status, **extra)
        if self.path is not None: write(self.path, receipt)
        return receipt

    @contextmanager
    def phase(self, stage):
        previous = self.stage; self.stage = stage
        started, cpu = time.perf_counter(), time.process_time()
        try: yield
        finally:
            self.measurements.append({'stage':stage,'wall_seconds':time.perf_counter()-started,
                                      'cpu_seconds':time.process_time()-cpu})
            self.stage = previous
            self.flush()


class Completion:
    """SDK-compatible response for deterministic offline external-response replay."""
    def __init__(self, data):
        self.data = deepcopy(data)
        self.usage = SimpleNamespace(**data['usage']) if data.get('usage') else None
        self.choices = [SimpleNamespace(message=Message(c['message'])) for c in data.get('choices', [])]

    def model_dump(self, **kwargs): return deepcopy(self.data)


class Message:
    def __init__(self, data):
        self.data = data; self.content = data.get('content')
    def model_dump(self, **kwargs):
        def dump(value):
            if isinstance(value, dict):
                return {k:dump(v) for k,v in value.items() if v is not None or not kwargs.get('exclude_none')}
            if isinstance(value, list): return [dump(v) for v in value]
            return deepcopy(value)
        return dump(self.data)


class ReplayTransport:
    def __init__(self, tape): self.tape = deepcopy(tape); self.position = 0
    def __call__(self, request):
        require(self.position < len(self.tape), 'Replay exhausted')
        row = self.tape[self.position]; self.position += 1
        require(row['request'] == request, 'Native request replay mismatch')
        require(row.get('status', 'ok') == 'ok', 'Error tape needs a provider-specific exception fixture')
        return Completion(row['response'])
    def finish(self): require(self.position == len(self.tape), 'Replay left unused responses')


def tool_event_failed(event):
    """Match the native tool error envelope, never words in successful text."""
    if 'error' in event:
        return True
    if event.get('method') != 'call':
        return False
    for message in event.get('result', []):
        content = message.get('content')
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            try:
                payload = ast.literal_eval(content)
            except (ValueError, SyntaxError, TypeError):
                continue
        if isinstance(payload, dict) and 'error' in payload:
            return True
    return False


def make_llm(runtime, transport, model, trace):
    llm = runtime.LLM.__new__(runtime.LLM)
    llm.trace = trace; llm.model = model['model']
    def create(**request):
        start, cpu = time.perf_counter(), time.process_time()
        row = {'attempt': len(trace.calls), 'stage': trace.stage, 'billing_kind': 'chat_tokens',
               'provider': model['provider'], 'model': request['model'], 'endpoint': model['endpoint'],
               'request': deepcopy(request), 'status': 'error', 'usage': None, 'response': None}
        trace.calls.append(row)
        trace.flush()
        try:
            response = transport(deepcopy(request))
            row['response'] = response.model_dump()
            row['usage'] = row['response'].get('usage'); row['status'] = 'ok'
            return response
        except Exception as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            row['wall_seconds'] = time.perf_counter()-start; row['cpu_seconds'] = time.process_time()-cpu
            trace.flush()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return llm


def live_transport(model):
    import os
    from openai import OpenAI
    key = os.environ.get(model['api_key_env'])
    require(bool(key), 'Missing configured API key environment variable')
    client = OpenAI(api_key=key, base_url=model['endpoint'], timeout=120, max_retries=0)
    return lambda request: client.chat.completions.create(**request)


def build_memory(runtime, asset):
    """Use the native builder, including its topic, persona, time and speaker links."""
    require(asset.get('schema') == 'native-tag-assets-1', 'Not native assets; canonical EITR graph is not interchangeable')
    require(digest({k:v for k,v in asset.items() if k != 'hash'}) == asset['hash'], 'Asset hash mismatch')
    require(asset['provenance'].get('identity_verified') is True and
            asset['provenance'].get('input_texts_verified') is True, 'Unverified native embedding cache')
    identity = EmbeddingIdentity(**asset['embedding_identity'])
    rewrite_texts = {}
    expected_topics = []
    for i, record in enumerate(asset['rewrites'], 1):
        require(isinstance(record, dict) and len(record)==1, 'Invalid native rewrite record')
        session = next(iter(record.values()))
        if session is None: continue
        for event in session.get('sentence', []):
            require(event['id'] not in rewrite_texts, 'Duplicate event input ID')
            rewrite_texts[event['id']] = event['text']
        expected_topics.extend(f'D{i}:'+tid for tid in session.get('topics', {}))
    require(rewrite_texts == asset['event_input_texts'], 'Encoded text must match native rewritten sentence identity')
    require(set(rewrite_texts) == set(asset['event_embeddings']), 'Native event vector table mismatch')
    require(expected_topics == asset['topic_ids'] and len(expected_topics) == len(asset['topic_embeddings']),
            'Native topic embedding identity mismatch')
    from .core import unit
    for vector in asset['topic_embeddings']: unit(vector, identity.dimension)
    memory = runtime.MemorySystem()
    agent = runtime.Agent(None, memory, runtime.MemoryController(memory, None))
    agent.store_raw_text(deepcopy(asset['raw_text']),
                         {k: np.asarray(v) for k,v in asset['event_embeddings'].items()},
                         deepcopy(asset['topic_ids']), np.asarray(asset['topic_embeddings']))
    require(len(asset['rewrites']) == len(asset['keywords']), 'Native cache session mismatch')
    for i, (rewrite, keys) in enumerate(zip(asset['rewrites'], asset['keywords']), 1):
        agent.store_event_new(deepcopy(next(iter(rewrite.values()))), deepcopy(keys), i)
    scope = Scope(asset['owner'], asset['snapshot'], asset['hash'])
    view = EvidenceView(memory, scope, identity, asset['event_input_texts'], provenance=asset['provenance'])
    return memory, view


def install_trace(agent, controller, trace):
    def wrap(obj, name, stage=None):
        original = getattr(obj, name)
        def invoke(*args, **kwargs):
            before = {'events': deepcopy(controller.queried_event), 'keywords': deepcopy(controller.queried_keyword)}
            record = {'method': name, 'args': clean(args), 'kwargs': clean(kwargs), 'before': before}
            trace.events.append(record)
            with trace.phase(stage or trace.stage):
                try:
                    result = original(*args, **kwargs); record['result'] = clean(result)
                    source_candidates = set(re.findall(r'\bD\d+:\d+\b', json.dumps(record['result'],ensure_ascii=False))) if name in SOURCE_READERS else set()
                    known = {sid for session in controller.memory.raw_text.values() for sid in session}
                    record['returned_source_ids'] = sorted(source_candidates & known)
                    return result
                except Exception as exc:
                    record['error'] = str(exc); raise
                finally:
                    record['after'] = {'events': deepcopy(controller.queried_event),
                                       'keywords': deepcopy(controller.queried_keyword)}
        setattr(obj, name, invoke)
    for name, stage in [('extract_question_keys','question_parse'), ('get_time','question_time'),
                        ('select_finegrained_sentence','initial_fact_select'),
                        ('select_finegrained_sentence_sort','initial_fact_rerank'),
                        ('select_key_tag','initial_tag'), ('_chat_with_tools','tool_loop_and_answer')]:
        wrap(agent, name, stage)
    for name in ('event_by_tag','query_event_keywords','query_event_context','query_conversation_time',
                 'query_topic_events','query_personal_information','query_personal_aspect','evaluate_relations_over_graph'):
        wrap(controller, name)
    wrap(agent.tool_bridge, 'call')


def clean(value):
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (set, tuple, list)): return [clean(v) for v in value]
    return value


def graph_digest(memory):
    """Full native object state, including all auxiliary indices; read-only."""
    def encode(value):
        if isinstance(value, np.ndarray): return {'array': value.tolist(), 'dtype': str(value.dtype)}
        if isinstance(value, dict): return {'dict': [[encode(k), encode(v)] for k,v in value.items()]}
        if isinstance(value, set): return {'set': sorted((encode(v) for v in value), key=lambda v:json.dumps(v,sort_keys=True))}
        if isinstance(value, (list,tuple)): return [encode(v) for v in value]
        if isinstance(value, (date,datetime)): return {'date':str(value)}
        if hasattr(value, '__dict__'): return {'class':type(value).__name__, 'state':encode(vars(value))}
        return value
    return digest(encode(memory))
