"""Frozen sequential, fully independent native-agent trials and fail-closed gates."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math
from decimal import Decimal, ROUND_CEILING
import platform
import sys
import importlib.metadata
from pathlib import Path
import random
import time
from eicr.contracts import digest, require, ContractError
from .core import Selector, decision_sets, EmbeddingIdentity, Isotonic, fit_calibration, unit
from .runtime import (build_memory, clean, file_hash, install_trace, live_transport, load_runtime,
                      make_llm, Trace, verify_sources, write, graph_digest, tool_event_failed)
from .transport import bill, Encoder, validate_prices

QUESTION_FIELDS = {'question_id','owner','snapshot','category','question'}
CHECKS = {'native_identity_replay','selector_passthrough_replay','native_tool_outputs',
          'scope_isolation','score_boundaries','query_identity','ledger_completeness','all_native_tools',
          'independent_native_loader','protocol_provenance','symmetric_adjudication','offline_and_source_reports'}


def sealed(body): return {**body, 'hash': digest(body)}


def verify(value):
    require(isinstance(value,dict), 'Missing or invalid artifact')
    require(value.get('hash') == digest({k:v for k,v in value.items() if k != 'hash'}), 'Artifact hash mismatch')
    return value


def question_key(q): return (q['owner'],q['snapshot'],q['question_id'])


def sample_design(owners, design=None):
    require(len(owners)==len(set(owners))==4, 'Primary design requires four owners')
    if design is None:
        design={'rationale':'default four owners x four categories x two questions',
                'strata':[{'owner':o,'category':c,'count':2} for o in owners for c in (1,2,3,4)]}
    require(design.get('rationale') and len(design['strata'])==16, 'Document all strata and the revised design rationale')
    counts={(s['owner'],s['category']):s['count'] for s in design['strata']}
    require(set(counts)=={(o,c) for o in owners for c in (1,2,3,4)} and
            all(type(n) is int and n>=0 for n in counts.values()) and sum(counts.values())==32 and
            all(sum(counts[o,c] for c in (1,2,3,4))>0 for o in owners), 'Invalid revised 32-question design')
    return design,counts


def select_questions(pool, excluded_ids, excluded_texts, *, owners, salt, evidence_by_scope, design=None):
    """Gold eligibility is used only here; runtime receives a label-free projection."""
    groups, candidates, exclusions = defaultdict(list), [], []
    seen = set()
    for q in pool:
        key = question_key(q)
        require(key not in seen, 'Duplicate candidate question identity'); seen.add(key)
        text_hash = digest(q['question'].strip())
        reason = None
        if q['question_id'] in excluded_ids or text_hash in excluded_texts: reason = 'history_or_development'
        elif q['owner'] not in owners or q['category'] not in (1,2,3,4): reason = 'outside_strata'
        elif not str(q.get('reference','')).strip(): reason = 'empty_answer'
        elif not q.get('evidence') or not set(q['evidence']) <= set(evidence_by_scope.get((q['owner'],q['snapshot']),())):
            reason = 'evidence_outside_snapshot'
        record = {'key':list(key),'text_hash':text_hash,'selection_hash':digest([salt,*key,q['question']]),'excluded':reason}
        candidates.append(record)
        if reason: exclusions.append(record)
        else: groups[q['owner'],q['category']].append((record['selection_hash'],q))
    selected, labels = [], []
    design,counts = sample_design(owners,design)
    for owner in owners:
        for category in (1,2,3,4):
            group = sorted(groups[owner,category], key=lambda x:x[0])
            count=counts[owner,category]
            require(len(group) >= count, f'Insufficient stratum {owner}/{category}; freeze a revised design before running')
            for _, q in group[:count]:
                selected.append({k:q[k] for k in QUESTION_FIELDS})
                labels.append({k:deepcopy(v) for k,v in q.items() if k in QUESTION_FIELDS | {'reference','evidence','review_flags'}})
    manifest = sealed({'schema':'native-tag-sample-1','salt':salt,'owners':owners,'candidates':candidates,
                       'excluded_ids':sorted(excluded_ids),'excluded_text_hashes':sorted(excluded_texts),
                       'questions_hash':digest(selected),'labels_hash':digest(labels),'design':design,
                       'exposure_limit':'public previously inspected conversations; exact ID/text exclusions only'})
    return selected, labels, manifest


def audit_history(paths):
    ids, texts, files = set(), set(), []
    def visit(value):
        if isinstance(value, dict):
            for k in ('question_id','qid'):
                if isinstance(value.get(k),str): ids.add(value[k])
            for k in ('question','condition_question','original_question'):
                if isinstance(value.get(k),str): texts.add(digest(value[k].strip()))
            for v in value.values(): visit(v)
        elif isinstance(value,list):
            for v in value: visit(v)
    for p in sorted(set(Path(p).resolve() for p in paths)):
        require(p.suffix in ('.json','.jsonl'), 'History inputs must be explicit JSON/JSONL records')
        data = p.read_text()
        values = [json.loads(line) for line in data.splitlines() if line.strip()] if p.suffix == '.jsonl' else [json.loads(data)]
        for value in values: visit(value)
        files.append({'path':str(p),'sha256':file_hash(p)})
    return sealed({'files':files,'question_ids':sorted(ids),'question_text_hashes':sorted(texts),
                   'scope':'only listed files; no claim of zero historical exposure'})


def environment_identity():
    versions = {}
    for name in ('numpy','openai','sentence-transformers'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    import os
    return {'python':sys.version,'platform':platform.platform(),'dependencies':versions,
            'python_hash_seed':os.environ.get('PYTHONHASHSEED'),
            # Record actual interpreter state too: setting the environment after
            # startup does not change Python's hash secret or native set order.
            'python_hash_probe':[hash(s) for s in ('native-tag', 'Maria', 'D1:1-1')]}


def implementation_hashes():
    root = Path(__file__).parent
    files = list(root.glob('*.py')) + [root.parent/'eitr/evaluate.py', root.parent/'eicr/contracts.py']
    return {str(p.relative_to(root.parent)):file_hash(p) for p in sorted(files)}


def execution_identity(spec):
    """Identity shared by development, primary and confirmation; graphs may be new."""
    runtime = verify_sources(spec['runtime'])
    return {'runtime_hash':runtime['hash'], 'limits':runtime['limits'],
            'implementation':implementation_hashes(), 'model':spec['model'],
            'embedding':spec['embedding'], 'environment':environment_identity()}


def validate_run_set(protocol, runs):
    verify(protocol)
    expected = {(question_key(j['question']),j['trial'],j['arm']) for j in protocol['jobs']}
    require(len(runs) == len(expected), 'Incomplete run denominator')
    actual = set()
    for row in runs:
        verify(row)
        require(row['protocol_hash'] == protocol['hash'], 'Mixed protocol runs')
        actual.add((question_key(row['question']),row['trial'],row['arm']))
    require(actual == expected and len(actual) == len(runs), 'Run pairing mismatch')


def development_provenance(protocol, runs):
    validate_run_set(protocol, runs)
    require(protocol['phase'] == 'development' and all(r['arm']=='native' for r in runs), 'Use frozen development runs')
    require(not any(r.get('adaptation_error') for r in runs), 'Development adaptation error')
    return {'protocol_hash':protocol['hash'], 'execution':protocol['execution_identity'],
            'questions_hash':digest(protocol['questions']), 'assets':{k:v['hash'] for k,v in protocol['assets'].items()},
            'run_hashes':[r['hash'] for r in runs]}


def calibrate(protocol, runs):
    provenance = development_provenance(protocol, runs)
    require(not protocol.get('shadow'), 'Calibrate from original teacher collection, not a shadow rerun')
    start, cpu = time.perf_counter(), time.process_time()
    records = calibration_records(runs)
    cal = fit_calibration(records, [q['question_id'] for q in protocol['questions']],
                          development_questions=[q['question'] for q in protocol['questions']])
    cal = sealed({**{k:v for k,v in cal.items() if k!='hash'}, 'provenance':provenance})
    receipt = sealed({'schema':'native-tag-ledger-2','ledger_id':'calibration:'+protocol['hash'],
        'components':['teacher_calibration'],'status':'complete','calibration_hash':cal['hash'],
        'calls':[c for r in runs for c in r['calls']],
        'wall_seconds':sum(r['wall_seconds'] for r in runs)+time.perf_counter()-start,
        'cpu_seconds':sum(r['cpu_seconds'] for r in runs)+time.process_time()-cpu,
        'cost_policy':'full_development_collection_once_including_stage_a; do_not_add_same_runs_again'})
    return cal, records, receipt


def shadow_acceptance(protocol, runs):
    provenance = development_provenance(protocol, runs)
    require(protocol.get('shadow') and protocol.get('calibration'), 'Use a frozen shadow development protocol')
    decisions = [d for r in runs for d in r['decisions']]
    accepted = bool(decisions) and not any(r['failed'] for r in runs) and all('shadow_response' in d for d in decisions)
    return sealed({'accepted':accepted,'provenance':provenance,'calibration_hash':protocol['calibration']['hash'],
        'states':[state_signature(d) for d in decisions],
        'differences':[{'stage':d['stage'],'sets':d.get('shadow_differences'),
                        'native_sources':d.get('native_reachable_sources'),
                        'shadow_sources':d.get('shadow_reachable_sources')} for d in decisions]})


def state_signature(d):
    return digest({k:d.get(k) for k in ('stage','context_text','key_id','ordered_tag_ids','graph_scope','state_before')})


def freeze_protocol(spec):
    require(spec['phase'] in ('development','evaluation','confirmation','diagnostic'), 'Unknown phase')
    runtime = verify_sources(spec['runtime'])
    questions = spec['questions']
    require(questions and len({question_key(q) for q in questions}) == len(questions), 'Empty/duplicate questions')
    for q in questions:
        require(set(q) == QUESTION_FIELDS and q['question'].strip() and q['category'] in (1,2,3,4), 'Invalid/unblinded runtime question')
    phase = spec['phase']
    identity = execution_identity(spec)
    if phase in ('development','diagnostic'): require(len(questions) <= 16, 'Development maximum is 16')
    else:
        require(len(questions) == 32 and len({q['owner'] for q in questions}) == 4, 'Primary design is 32 questions / 4 owners')
        require(spec['repeats'] == 2, 'Two full pipeline repeats required')
        sample = verify(spec['sample_manifest'])
        _,counts=sample_design(sample['owners'],sample['design'])
        actual=Counter((q['owner'],q['category']) for q in questions)
        require(all(actual[k]==n for k,n in counts.items()), 'Frozen primary strata mismatch')
        require(sample['questions_hash'] == digest(questions), 'Sample manifest mismatch')
        require(spec['stage_a']['decision'] == 'continue' and spec['stage_a']['hash'] ==
                digest({k:v for k,v in spec['stage_a'].items() if k != 'hash'}) and spec['stage_a']['runtime_hashes']==[runtime['hash']], 'Stage A has not passed')
        require(spec['stage_a'].get('provenance',{}).get('execution') == identity and
                spec['stage_a'].get('prices_hash') == digest(spec.get('prices')), 'Stage A model/configuration/prices changed')
        checks = verify(spec['checks'])
        require(checks['runtime_hash'] == runtime['hash'] and set(checks['checks']) == CHECKS and
                all(v is True for v in checks['checks'].values()) and checks['implementation'] == implementation_hashes(), 'Stage B adaptation checks incomplete')
        from .grading import rubric_manifest
        rubric_receipt = verify(spec['rubric_acceptance'])
        require(rubric_receipt['accepted'] and rubric_receipt['rubric_hash'] == rubric_manifest()['hash']
                and rubric_receipt['model'] == spec['judge'], 'Judge rubric not accepted on development cases')
        if phase == 'confirmation':
            previous = verify(spec['stage_c'])
            parent = verify(spec['previous_protocol'])
            require(parent['phase']=='evaluation' and previous['protocol_hash']==parent['hash'] and
                    previous['decision'] == 'continue', 'Stage C must pass under the referenced primary protocol')
            require(not {q['question_id'] for q in questions} & set(previous['question_ids']) and
                    not {digest(q['question'].strip()) for q in questions} & set(previous['question_text_hashes']), 'Confirmation reuse')
    arms = ['native'] if phase == 'development' else ['native','cheap_all']
    if phase == 'diagnostic':
        parent = verify(spec['previous_protocol']); previous = verify(spec['stage_c'])
        require(parent['phase']=='evaluation' and previous['protocol_hash']==parent['hash'], 'Diagnostic primary provenance mismatch')
        require(questions == parent.get('diagnostic_questions') and questions, 'Use the diagnostic subset frozen before Stage C')
        require(previous['aggregate']['cheap_all']['accuracy'] < previous['aggregate']['native']['accuracy'],
                'Diagnostic requires observed primary quality loss')
        arms = ['native','cheap_initial']
    require(type(spec['repeats']) is int and spec['repeats'] > 0, 'Invalid repeat count')
    validate_prices(spec.get('prices'))
    if phase in ('evaluation','confirmation'):
        require(spec.get('prices') is not None, 'Freeze actual rates before paid comparison')
    if spec.get('prices') is not None:
        expected = [(spec['model']['provider'],spec['model']['model'],spec['model']['endpoint'],'chat_tokens')]
        if spec['embedding']['identity']['provider'] != 'local':
            expected.append((spec['embedding']['identity']['provider'],spec['embedding']['identity']['model'],
                             spec['embedding']['endpoint'],'embedding_tokens'))
        rates = {tuple(r[k] for k in ('provider','model','endpoint','billing_kind')) for r in spec['prices']['rates']}
        require(set(expected) <= rates, 'Missing rate for frozen online model/endpoint')
    assets = {}
    for path in spec['assets']:
        p = Path(path).resolve(); asset = verify(json.loads(p.read_text()))
        key = asset['owner']+'/'+asset['snapshot']
        require(key not in assets, 'Duplicate graph asset')
        require(asset['embedding_identity'] == spec['embedding']['identity'], 'Mixed embedding identities')
        require(asset['runtime_reference_hashes'] == runtime['reference'], 'Graph builder source identity mismatch')
        # Validate native construction, all scoped links and vector rows before any model requests.
        build_memory(load_runtime(spec['runtime'],spec['model']),asset)
        assets[key] = {'path':str(p),'sha256':file_hash(p),'hash':asset['hash']}
    require(all(q['owner']+'/'+q['snapshot'] in assets for q in questions), 'Missing question graph')
    calibration = spec.get('calibration')
    shadow = spec.get('shadow',False)
    require(type(shadow) is bool and (not shadow or phase=='development'), 'Shadow is a development-only native trajectory')
    if shadow: require(calibration and calibration['ready'], 'Shadow requires supported calibration')
    if calibration is not None:
        verify(calibration)
        require(calibration.get('provenance',{}).get('execution') == identity, 'Teacher model/runtime/configuration mismatch')
        for mapping in calibration['maps'].values():
            if mapping is not None: Isotonic(**mapping)
        if shadow:
            require(digest(questions)==calibration['provenance']['questions_hash'], 'Shadow must use the frozen development questions')
            require({k:v['hash'] for k,v in assets.items()}==calibration['provenance']['assets'], 'Shadow development graphs changed')
    if phase != 'development':
        verify(calibration)
        require(calibration['ready'], 'Calibration is underpowered')
        if phase != 'diagnostic':
            require(len(calibration['development_text_hashes']) == len(calibration['development_ids']) and
                    not set(calibration['development_text_hashes']) & {digest(q['question'].strip()) for q in questions},
                    'Missing development text exposure audit or duplicate evaluation text')
        require(not set(calibration['development_ids']) & {q['question_id'] for q in questions}
                or phase == 'diagnostic', 'Development/evaluation leakage')
        require(asdict(EmbeddingIdentity(**spec['embedding']['identity'])) in calibration['embedding_identities'], 'Calibration identity mismatch')
    if phase in ('evaluation','confirmation'):
        require(spec['stage_a']['provenance']==calibration['provenance'], 'Stage A/calibration development provenance mismatch')
        shadow_receipt = verify(spec['shadow_acceptance'])
        require(shadow_receipt['accepted'] and shadow_receipt['calibration_hash']==calibration['hash'] and
                shadow_receipt['provenance']['execution']==identity and
                shadow_receipt['provenance']['questions_hash']==calibration['provenance']['questions_hash'],
                'Missing or incompatible development shadow acceptance')
    diagnostic_questions = spec.get('diagnostic_questions',[])
    if diagnostic_questions:
        require(phase=='evaluation' and len(diagnostic_questions)<=16 and calibration, 'Invalid preregistered diagnostic subset')
        require(all(set(q)==QUESTION_FIELDS and q['question_id'] in calibration['development_ids'] and
                    digest(q['question'].strip()) in calibration['development_text_hashes'] for q in diagnostic_questions),
                'Diagnostic subset must be frozen development questions')
    rng = random.Random(spec['seed'])
    groups = [(q,r) for q in questions for r in range(spec['repeats'])]; rng.shuffle(groups)
    jobs = []
    for q, trial in groups:
        order = list(arms); rng.shuffle(order)
        for arm in order: jobs.append({'question':q,'trial':trial,'arm':arm})
    body = {'schema':'native-tag-protocol-2', 'phase':phase, 'runtime':str(Path(spec['runtime']).resolve()),
            'runtime_hash':runtime['hash'], 'limits':runtime['limits'], 'questions':questions,
            'repeats':spec['repeats'],'seed':spec['seed'],'arms':arms,'jobs':jobs,'assets':assets,
            'model':spec['model'],'embedding':spec['embedding'],'prices':spec.get('prices'),
            'judge':spec.get('judge'),'rubric_acceptance':spec.get('rubric_acceptance'),
            'calibration':calibration,'shadow':shadow,'implementation':implementation_hashes(),'environment':environment_identity(),
            'execution_identity':identity, 'diagnostic_questions':diagnostic_questions,
            'cache_policy':'no_cross_request_query_or_chat_cache; exact_context_within_request_reuse',
            'measurement':'sequential_wall_and_cpu; includes_graph_load_and_encoder_initialization; service_cache_reported',
            'gates':{'tag_fraction':.10,'saving':.15,'quality_delta':0.,'owners_saving':3,
                     'failure_delta':0.,'fallback':.10,'missing_cost':.05},
            'audit':{k:spec.get(k) for k in ('sample_manifest','stage_a','checks','stage_c','shadow_acceptance')}}
    if phase in ('confirmation','diagnostic'):
        for key in ('execution_identity','model','embedding','prices','judge','rubric_acceptance',
                    'calibration','implementation','environment','limits','gates','cache_policy','measurement','repeats','seed'):
            require(body[key]==parent[key], 'Frozen primary configuration changed: '+key)
        body['previous_protocol_hash'] = parent['hash']
        if phase=='confirmation':
            def shape(sample):
                _,counts=sample_design(sample['owners'],sample['design'])
                return sorted(tuple(counts[o,c] for c in (1,2,3,4)) for o in sample['owners'])
            require(shape(body['audit']['sample_manifest'])==shape(parent['audit']['sample_manifest']), 'Confirmation sampling design changed')
    return sealed(body)


def preflight(protocol):
    verify(protocol)
    require(protocol['environment'] == environment_identity(), 'Execution environment changed')
    require(protocol['implementation'] == implementation_hashes(), 'Implementation changed after freezing')
    require(verify_sources(protocol['runtime'])['hash'] == protocol['runtime_hash'], 'Runtime changed')
    if protocol.get('rubric_acceptance'):
        from .grading import rubric_manifest
        require(protocol['rubric_acceptance']['rubric_hash']==rubric_manifest()['hash'], 'Frozen rubric changed')
    for ref in protocol['assets'].values():
        require(file_hash(ref['path']) == ref['sha256'], 'Frozen graph asset changed')
    return {'ready':True,'protocol_hash':protocol['hash'],'jobs':len(protocol['jobs'])}


def run_one(protocol, job, *, transport_factory=live_transport, encoder_factory=Encoder,
            instrumented=True, selector_replay=None, trace=None):
    trace = trace if trace is not None else Trace()
    start, cpu = time.perf_counter(), time.process_time()
    q = job['question']; asset_ref = protocol['assets'][q['owner']+'/'+q['snapshot']]
    rt = load_runtime(protocol['runtime'], protocol['model'], instrumented=instrumented)
    memory, view = build_memory(rt,json.loads(Path(asset_ref['path']).read_text()))
    graph_before = graph_digest(memory)
    llm = make_llm(rt, transport_factory(protocol['model']), protocol['model'], trace)
    controller = rt.MemoryController(memory,llm); agent = rt.Agent(llm,memory,controller)
    encoder = encoder_factory(protocol['embedding'],trace)
    selector = Selector(job['arm'],view,encoder,protocol.get('calibration'),replay=selector_replay,
                        collect_raw=protocol['phase'] == 'development',shadow=protocol.get('shadow',False))
    for obj in (agent,controller): obj.tag_selector=selector; obj.graph_scope=view.scope
    install_trace(agent,controller,trace)
    row = {'schema':'native-tag-run-1','protocol_hash':protocol['hash'],'question':q,'trial':job['trial'],
           'runtime_hash':rt.manifest['hash'],'arm':job['arm'],'answer':None,'supports':[],'error':None,'graph_scope':asdict(view.scope),
           'view_hash':view.identity_hash,'calls':trace.calls,'events':trace.events,'decisions':selector.decisions,
           'query_cache':'bypass','backend':'full_frozen_local_native'}
    try:
        formatted = agent.question_format('locomo',{'question':q['question'],'category':q['category']})
        row['formatted_question'] = formatted
        with trace.phase('native_question_embedding'): vector = encoder.encode(q['question'])
        unit(vector,encoder.identity.dimension)  # Validate before native readers or paid chat requests.
        selector.existing_query = (q['question'], vector)
        answer, supports = agent.answer_question(formatted,q['category'],vector)
        row['answer'],row['supports'] = answer,supports
    except Exception as exc:
        row['error'] = {'type':type(exc).__name__,'message':str(exc)}
        row['adaptation_error'] = isinstance(exc, ContractError)
    row['graph_unchanged'] = graph_before == graph_digest(memory)
    if not row['graph_unchanged']:
        row['adaptation_error'] = True
        row['error'] = row['error'] or {'type':'GraphMutation','message':'Native graph mutated during request'}
    row['wall_seconds'] = time.perf_counter()-start; row['cpu_seconds'] = time.process_time()-cpu
    row['state'] = {'queried_event':clean(controller.queried_event),'queried_keyword':clean(controller.queried_keyword)}
    row['native_stats'] = deepcopy(rt.API_STATS)
    row['budget_exceeded'] = (rt.API_STATS['tool_calls'] > protocol['limits']['MAX_TOOL_CALLS'] or
                              rt.API_STATS['rounds'] > protocol['limits']['MAX_ROUNDS'])
    # Check the native output contract, not the former matrix source-character contract.
    loop_messages = [c['response']['choices'][0]['message'] for c in trace.calls
                     if c['stage'] == 'tool_loop_and_answer' and c.get('response') and c['response'].get('choices')]
    final = None
    for msg in loop_messages:
        try:
            value = rt.extract_json_from_content(msg.get('content') or '')
            if value.get('mode') == 'answer': final=value
        except (ValueError,AttributeError,TypeError): pass
    row['native_answer_object'] = final
    row['answer_format_valid'] = bool(final and isinstance(final.get('answer'),str) and final['answer'].strip())
    row['sources_valid'] = (set(row['supports'] or []) <= {s for values in view.sources.values() for s in values})
    row['source_supports_answer'] = None  # semantic support needs separate blind assessment.
    all_sources = {sid:text for session in memory.raw_text.values() for sid,text in session.items()}
    row['support_texts'] = {s:all_sources.get(s) for s in (row['supports'] or [])}
    row['visited_event_ids'] = sorted(set(controller.queried_event))
    row['visited_source_ids'] = sorted({s for eid in controller.queried_event for s in view.sources.get(eid,())} |
                                     {s for e in trace.events for s in e.get('returned_source_ids',[])})
    row['tool_failures'] = [e for e in trace.events if tool_event_failed(e)]
    row['failed'] = row['error'] is not None or not row['answer_format_valid'] or bool(row['tool_failures']) or not row['graph_unchanged'] or row['budget_exceeded']
    row['billing'] = bill(trace.calls,protocol.get('prices'))
    row['measurements'] = clean(trace.measurements)
    for d in selector.decisions:
        try:
            d['sets'] = decision_sets(d.get('response'),protocol['limits']['TAG_LIMIT'])
        except (ValueError,TypeError,AttributeError):
            d['sets'] = {'selected':[],'early':[],'zero':[]}
            d['decision_parse_failed'] = True
            row['failed'] = True
        if 'shadow_response' in d:
            shadow_sets = decision_sets(d['shadow_response'],protocol['limits']['TAG_LIMIT'])
            d['shadow_sets'] = shadow_sets
            d['shadow_differences'] = {k:sorted(set(d['sets'][k]) ^ set(shadow_sets[k])) for k in ('selected','early','zero')}
            # Native model output can include tags absent from this key's graph.
            # The native reader returns no edges for them. Preserve the response
            # and selection, and record the anomaly without breaking diagnostics.
            d['native_unlinked_selected_tags'] = [tag for tag in d['sets']['selected']
                                                 if (d['key_id'],tag) not in view.links]
            def reachable(tags):
                return sorted({s for tag in tags for eid in view.links.get((d['key_id'],tag),()) for s in view.sources[eid]})
            d['native_reachable_sources'] = reachable(d['sets']['selected'])
            d['shadow_reachable_sources'] = reachable(shadow_sets['selected'])
        if d['stage'] == 'expansion':
            d['sets']['early'] = []  # native expansion has no early-read branch.
            if 'shadow_sets' in d:
                d['shadow_sets']['early'] = []; d['shadow_differences']['early'] = []
    return sealed(clean(row))


def run_protocol(protocol, output, **kwargs):
    preflight(protocol)
    output = Path(output); require(not output.exists(), 'New output directory required; no overwrite/resume ambiguity')
    output.mkdir(parents=True); write(output/'protocol.json',protocol)
    completed = []
    abort_reason = None
    for i, job in enumerate(protocol['jobs']):
        write(output/'journal.json', {'status':'running','job':i,'completed':completed,'protocol_hash':protocol['hash']})
        trace = Trace(output/f'ledgers/{i:04d}.json', {'protocol_hash':protocol['hash'],'job':i,'components':['online']})
        start, cpu = time.perf_counter(), time.process_time()
        try:
            if abort_reason: raise ContractError('Not executed after adaptation failure: '+abort_reason)
            row = run_one(protocol,job,trace=trace,**kwargs)
        except Exception as exc:
            row = sealed({'schema':'native-tag-run-1','protocol_hash':protocol['hash'],
                'runtime_hash':protocol['runtime_hash'],**job,'answer':None,'supports':[],
                'error':{'type':type(exc).__name__,'message':str(exc),
                         'phase':'aborted' if abort_reason else 'postprocess' if trace.calls else 'setup'},
                'adaptation_error':isinstance(exc,ContractError) or bool(trace.calls),'failed':True,'answer_format_valid':False,
                'calls':clean(trace.calls),'events':clean(trace.events),'decisions':[],
                'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                'billing':bill(trace.calls,protocol.get('prices')),
                'backend':'full_frozen_local_native','sources_valid':False,'source_supports_answer':None})
        if row.get('adaptation_error') and abort_reason is None:
            abort_reason = row['error']['message']
        trace.flush('failed' if row['failed'] else 'complete', run_hash=row['hash'], error=row.get('error'))
        name = f'runs/{i:04d}.json'; write(output/name,row)
        completed.append({'path':name,'hash':row['hash']})
        print(f'{i+1}/{len(protocol["jobs"])} {job["arm"]} failed={row["failed"]}',flush=True)
    result = sealed({'status':'aborted' if abort_reason else 'complete','protocol_hash':protocol['hash'],
                     'abort_reason':abort_reason,'runs':completed})
    write(output/'manifest.json',result)
    write(output/'journal.json', {'status':result['status'],'completed':completed,
                                 'protocol_hash':protocol['hash'],'abort_reason':abort_reason})
    return result


def load_runs(directory):
    directory = Path(directory); p = verify(json.loads((directory/'protocol.json').read_text()))
    m = verify(json.loads((directory/'manifest.json').read_text()))
    require(m['status'] in ('complete','aborted') and m['protocol_hash'] == p['hash'] and len(m['runs']) == len(p['jobs']),
            'Incomplete run denominator')
    runs = []
    for ref, job in zip(m['runs'],p['jobs']):
        row = verify(json.loads((directory/ref['path']).read_text()))
        require(row['hash'] == ref['hash'] and row['protocol_hash'] == p['hash'] and
                row['question'] == job['question'] and row['trial'] == job['trial'] and row['arm'] == job['arm'],
                'Run does not match planned job')
        runs.append(row)
    return p,runs


def calibration_records(runs):
    result = []
    for run in runs:
        require(run['arm'] == 'native', 'Teacher must be native')
        identity = next(c['embedding_identity'] for c in run['calls'] if 'embedding_identity' in c)
        for i,d in enumerate(run['decisions']):
            teacher = (d.get('teacher_response') or {}).get('tag_scores')
            if 'raw' not in d or not isinstance(teacher,dict) or not teacher or set(teacher) != set(d['raw']): continue
            if not all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1 for v in teacher.values()): continue
            result.append({'request_id':digest([run['hash'],i]),'question_id':run['question']['question_id'],
                           'stage':d['stage'],'raw':d['raw'],'teacher':teacher,'view_hash':d['view_hash'],
                           'embedding_identity':identity,
                           'teacher_calls':[run['calls'][j] for j in d['call_indices'] if run['calls'][j]['billing_kind']=='chat_tokens']})
    return result


def stage_a(runs, prices, *, protocol):
    provenance = development_provenance(protocol,runs)
    require(not protocol.get('shadow'), 'Use the original native collection for Stage A')
    require(runs and all(r['arm'] == 'native' and r['backend']=='full_frozen_local_native' for r in runs),
            'Stage A requires full native traces')
    totals, tags, per_run = [], [], []
    for r in runs:
        # Raw-score collection is calibration expenditure, not part of the native baseline.
        calls = [c for c in r['calls'] if not (c['billing_kind'] != 'chat_tokens' and c['stage'] in ('initial_tag','expansion_tag'))]
        total = bill(calls,prices); tagged = bill([c for c in calls if c['stage'] in ('initial_tag','expansion_tag')],prices)
        totals.append(total['amount']); tags.append(tagged['amount'])
        per_run.append({'question_id':r['question']['question_id'],'total':total,'tag':tagged,
                        'failed':r['failed'],'wall_seconds':r['wall_seconds'],
                        'native_wall_excluding_raw_scoring':max(0.,r['wall_seconds']-sum(d.get('scoring_seconds',0) for d in r['decisions'])),
                        'tag_http_seconds':sum(c['wall_seconds'] for c in calls if c['stage'] in ('initial_tag','expansion_tag'))})
    trigger = sum(bool(r['decisions']) for r in runs)
    complete = all(x is not None for x in totals+tags)
    fraction = sum(tags)/sum(totals) if complete and sum(totals)>0 else None
    decision = 'stop' if trigger == 0 else 'inconclusive' if fraction is None else 'continue' if fraction >= .1 else 'stop'
    stages = {}
    for s in ('initial','expansion'):
        decisions = [d for r in runs for d in r['decisions'] if d['stage']==s]
        stages[s] = {'triggered_runs':sum(any(d['stage']==s for d in r['decisions']) for r in runs),
                     'calls':len(decisions),'candidate_counts':[len(d['ordered_tag_ids']) for d in decisions],
                     'selected_counts':[len(d['sets']['selected']) for d in decisions],
                     'early_reads':sum(len(d['sets']['early']) for d in decisions)}
    return sealed({'schema':'native-tag-stage-a-2','decision':decision,'runs':len(runs), 'triggered_runs':trigger,
                   'provenance':provenance,'prices_hash':digest(prices),
                   'failure_rate':sum(r['failed'] for r in runs)/len(runs),
                   'tag_fraction_native_wall':sum(x['tag_http_seconds'] for x in per_run)/sum(x['native_wall_excluding_raw_scoring'] for x in per_run) if sum(x['native_wall_excluding_raw_scoring'] for x in per_run)>0 else None,
                   'tag_http_fraction':sum(x['tag_http_seconds'] for x in per_run)/sum(c.get('wall_seconds',0) for r in runs for c in r['calls'] if c['billing_kind']=='chat_tokens') if sum(c.get('wall_seconds',0) for r in runs for c in r['calls'] if c['billing_kind']=='chat_tokens') else None,
                   'latency_note':'HTTP ratio excludes raw scoring; collection wall includes instrumentation and offline calibration work',
                   'downstream_cost':bill([c for r in runs for c in r['calls'] if c['stage']=='tool_loop_and_answer'],prices),
                   'tag_cost_fraction_all_runs':fraction,'stages':stages,'per_run':per_run,
                   'native_run_hashes':[r['hash'] for r in runs],'runtime_hashes':sorted({r['runtime_hash'] for r in runs}),
                   'triggered_subset_tag_fraction':(sum(tags[i] for i,r in enumerate(runs) if r['decisions'])/sum(totals[i] for i,r in enumerate(runs) if r['decisions'])) if complete and trigger and sum(totals[i] for i,r in enumerate(runs) if r['decisions'])>0 else None,
                   'interpretation':'conditional savings space if downstream trajectory stays unchanged, not a hard bound'})


def analyze(protocol, runs, grade_artifact, *, offline=None, source_review=None, offline_cost=None):
    verify(protocol)
    from .grading import validate_grades, SOURCE_RULE
    from .reporting import mechanism_report
    grades = validate_grades(protocol,runs,grade_artifact)
    if protocol['phase'] in ('evaluation','confirmation'):
        require(offline is not None and offline_cost is None, 'Primary analysis requires receipt-based offline expenditure')
    if offline is not None:
        verify(offline)
        require(offline['currency']==protocol['prices']['currency'], 'Offline/online currency mismatch')
        require({a['hash'] for a in protocol['assets'].values()} <= set(offline['asset_hashes']) and
                protocol['calibration']['hash'] in offline['calibration_hashes'], 'Offline receipts do not cover frozen assets/calibration')
        ledger_ids={r['ledger_id'] for r in offline['receipts']}
        require('calibration:'+protocol['calibration']['provenance']['protocol_hash'] in ledger_ids,
                'Missing development calibration expenditure')
        shadow_receipt=protocol.get('audit',{}).get('shadow_acceptance')
        if shadow_receipt:
            require('shadow:'+shadow_receipt['provenance']['protocol_hash'] in ledger_ids, 'Missing shadow expenditure')
        offline_cost=offline['incremental_api_cost']
    if offline_cost is not None: require(math.isfinite(offline_cost) and offline_cost>=0,'Invalid offline cost')
    support_by_run={}
    if source_review is not None:
        verify(source_review)
        require(source_review['protocol_hash']==protocol['hash'] and source_review['rule']==SOURCE_RULE and
                source_review['run_hashes']==sorted(r['hash'] for r in runs),'Source review provenance mismatch')
        for item in source_review['assessments']:
            require(item['verdict'] in ('supported','unsupported','unresolved') and item.get('reviewer'), 'Invalid source assessment')
            for rh in item['run_hashes']:
                require(rh not in support_by_run and rh in source_review['run_hashes'],'Duplicate/unexpected source review')
                run=next(r for r in runs if r['hash']==rh)
                require(item['payload_hash']==digest({'question':run['question']['question'],
                        'answer':run['answer'] or '', 'sources':run.get('support_texts',{})}), 'Source assessment text mismatch')
                support_by_run[rh]=item['verdict']
    require(len(runs)==len(protocol['jobs']), 'Missing planned runs')
    for run in runs:
        verify(run)
        require(run['protocol_hash'] == protocol['hash'], 'Mixed protocol runs')
    indexed = {(question_key(r['question']),r['trial'],r['arm']):r for r in runs}
    require(len(indexed)==len(runs), 'Duplicate runs')
    for job in protocol['jobs']:
        require((question_key(job['question']),job['trial'],job['arm']) in indexed, 'Missing planned pairing')
    grade_map = {(g['run_hash'],g['rubric']):g for g in grades}
    require(len(grade_map)==len(grades), 'Duplicate grades')
    arms, per_q = protocol['arms'], []
    require(arms == ['native','cheap_all'], 'Primary analysis only N/S')
    def mean(values): return sum(values)/len(values)
    for q in protocol['questions']:
        row = {'question_id':q['question_id'],'owner':q['owner'],'category':q['category']}
        for arm in arms:
            rr = [indexed[question_key(q),t,arm] for t in range(protocol['repeats'])]
            gg = []
            for r in rr:
                g = grade_map.get((r['hash'],'strict'))
                require(g is not None and g.get('score') in (0,1), 'Missing strict grade; preserve failures in denominator')
                gg.append(g)
            costs = [r['billing']['amount'] for r in rr]
            row[arm] = {'accuracy':mean([g['score'] for g in gg]),
                        'lower':mean([g['score'] if not g.get('unresolved') else 0 for g in gg]),
                        'upper':mean([g['score'] if not g.get('unresolved') else 1 for g in gg]),
                        'cost':mean(costs) if all(c is not None for c in costs) else None,
                        'failure':mean([int(r['failed']) for r in rr]),
                        'wall_seconds':mean([r['wall_seconds'] for r in rr]),
                        'repeat_scores':[g['score'] for g in gg],
                        'triggered':any(r['decisions'] for r in rr)}
        delta = row['cheap_all']['accuracy']-row['native']['accuracy']
        row['outcome'] = 'win' if delta>0 else 'loss' if delta<0 else 'tie'
        row['paired_cost_delta'] = row['cheap_all']['cost']-row['native']['cost'] if row['cheap_all']['cost'] is not None and row['native']['cost'] is not None else None
        per_q.append(row)
    # Each identical question/reference/answer cache key has ONE unresolved verdict.
    # Its contributions on the two arms cancel when both returned the same text.
    fixed_delta, coefficients = 0., defaultdict(float)
    weight = 1/(len(protocol['questions'])*protocol['repeats'])
    for run in runs:
        grade = grade_map[run['hash'],'strict']
        sign = 1 if run['arm']=='cheap_all' else -1
        if grade.get('unresolved'):
            coefficients[grade['cache_key']] += sign*weight
        else:
            fixed_delta += sign*weight*grade['score']
    delta_bounds = [fixed_delta+sum(min(0.,v) for v in coefficients.values()),
                    fixed_delta+sum(max(0.,v) for v in coefficients.values())]
    aggregate = {a:{k:mean([r[a][k] for r in per_q]) if all(r[a][k] is not None for r in per_q) else None
                    for k in ('accuracy','lower','upper','cost','failure','wall_seconds')} for a in arms}
    baseline, method = aggregate['native'],aggregate['cheap_all']
    complete = baseline['cost'] is not None and method['cost'] is not None
    saving = 1-method['cost']/baseline['cost'] if complete and baseline['cost']>0 else None
    owners = {}
    for owner in sorted({q['owner'] for q in per_q}):
        rows = [q for q in per_q if q['owner']==owner]
        owners[owner] = {a:mean([q[a]['cost'] for q in rows]) if complete else None for a in arms}
    method_decisions = [d for r in runs if r['arm']=='cheap_all' for d in r['decisions']]
    fallback = sum(d['fallback'] for d in method_decisions)/len(method_decisions) if method_decisions else 0.
    missing = sum(not r['billing']['complete'] for r in runs)/len(runs)
    checks = {'cost_saving':saving is not None and saving>=.15,
              'quality':method['accuracy']>=baseline['accuracy'],
              'quality_robust_to_unresolved':delta_bounds[0]>=-1e-12,
              'owners_saving':sum(v['cheap_all']<v['native'] for v in owners.values())>=3 if complete else False,
              'failure':method['failure']<=baseline['failure'], 'fallback':fallback<=.1,
              'complete_economics':complete, 'missing_fraction':missing<=.05,
              'tag_triggered':bool(method_decisions),
              'offline_accounted':offline_cost is not None if protocol['phase'] in ('evaluation','confirmation') else True,
              'adaptation_integrity':not any(r.get('adaptation_error') for r in runs)}
    uncertain = not complete or not checks['quality_robust_to_unresolved'] and any(g.get('unresolved') for g in grades) or not checks['fallback'] or not checks['adaptation_integrity'] or not checks['offline_accounted']
    decision = 'inconclusive' if uncertain else 'continue' if all(checks.values()) else 'stop'
    per_query_saving = baseline['cost']-method['cost'] if complete else None
    break_even = int((Decimal(str(offline_cost))/(Decimal(str(baseline['cost']))-Decimal(str(method['cost'])))).to_integral_value(rounding=ROUND_CEILING)) if offline_cost is not None and per_query_saving is not None and per_query_saving>0 else None
    sensitivity = None
    if complete and len(per_q)>1:
        expensive = max(per_q,key=lambda r:r['native']['cost']+r['cheap_all']['cost'])
        remaining = [q for q in per_q if q is not expensive]
        n,s = (mean([q[a]['cost'] for q in remaining]) for a in arms)
        sensitivity = {'excluded_question_id':expensive['question_id'],'cost_reduction':1-s/n if n else None}
    groups = {}
    for label, subset in [('triggered',[q for q in per_q if q['native']['triggered'] or q['cheap_all']['triggered']]),
                          ('not_triggered',[q for q in per_q if not q['native']['triggered'] and not q['cheap_all']['triggered']])]+[
                          (f'category_{c}',[q for q in per_q if q['category']==c]) for c in (1,2,3,4)]:
        groups[label] = {'questions':len(subset), 'accuracy':{a:mean([q[a]['accuracy'] for q in subset]) if subset else None for a in arms},
                         'cost':{a:mean([q[a]['cost'] for q in subset]) if subset and all(q[a]['cost'] is not None for q in subset) else None for a in arms},
                         'failure':{a:mean([q[a]['failure'] for q in subset]) if subset else None for a in arms}}
    for owner, values in owners.items():
        subset=[q for q in per_q if q['owner']==owner]
        values['accuracy']={a:mean([q[a]['accuracy'] for q in subset]) for a in arms}
        values['repeat_disagreement']={a:sum(len(set(q[a]['repeat_scores']))>1 for q in subset)/len(subset) for a in arms}
    diagnostics = {}
    import numpy as np
    for arm in arms:
        rr = [r for r in runs if r['arm']==arm]
        ok = [r for r in rr if not r['failed']]
        decision_rows = [d for r in rr for d in r.get('decisions',[])]
        lenient = [grade_map.get((r['hash'],'lenient')) for r in rr]
        diagnostics[arm] = {
            'requests':len(rr),'successful_requests':len(ok),
            'success_conditional_cost':mean([r['billing']['amount'] for r in ok]) if ok and all(r['billing']['complete'] for r in ok) else None,
            'wall_all_requests':{str(p):float(np.percentile([r['wall_seconds'] for r in rr],p)) for p in (50,95)},
            'latency_note':'sequential independent requests; small-sample p95 descriptive only',
            'lenient_accuracy':mean([g['score'] for g in lenient]) if all(g is not None for g in lenient) else None,
            'failures_by_type':dict(Counter((r.get('error') or {}).get('type','tool_or_answer_format') for r in rr if r['failed'])),
            'tag_calls':len(decision_rows),
            'tool_batches':sum(e.get('method')=='call' for r in rr for e in r.get('events',[])),
            'early_reads':sum(len(d.get('sets',{}).get('early',[])) for d in decision_rows),
            'native_rounds':sum(r.get('native_stats',{}).get('rounds',0) for r in rr),
            'stages':{stage:{'calls':sum(d.get('stage')==stage for d in decision_rows),
                            'triggered_runs':sum(any(d.get('stage')==stage for d in r['decisions']) for r in rr)}
                      for stage in ('initial','expansion')}}
    decomposition = {}
    for arm in arms:
        rr = [r for r in runs if r['arm']==arm]
        phases = defaultdict(list)
        for r in rr:
            for call, priced in zip(r['calls'],r['billing']['calls']):
                phases[call['stage']].append(priced['amount'])
        decomposition[arm] = {phase:sum(values)/len(rr) if all(v is not None for v in values) else None
                              for phase,values in phases.items()}
    return sealed({'schema':'native-tag-analysis-1','decision':decision,'protocol_hash':protocol['hash'],
                   'grades_hash':grade_artifact['hash'],
                   'question_ids':[q['question_id'] for q in per_q],
                   'question_text_hashes':[digest(q['question'].strip()) for q in protocol['questions']],'per_question':per_q,'aggregate':aggregate,
                   'owners':owners,'groups':groups,'checks':checks,'symmetric_quality_delta_bounds':delta_bounds,'cost_reduction':saving,
                   'fallback_fraction':fallback,'missing_cost_run_fraction':missing,
                   'cost_by_stage_per_request':decomposition,'diagnostics':diagnostics,
                   'mechanisms':mechanism_report(protocol,runs,grade_artifact['labels']),
                   'source_assessment':{a:{'valid_source_fraction':mean([int(r.get('sources_valid',False)) for r in runs if r['arm']==a]),
                       'support_verdicts':dict(Counter(support_by_run.get(r['hash'],'unresolved') for r in runs if r['arm']==a))} for a in arms},
                   'source_review_hash':source_review['hash'] if source_review else None,
                   'offline_incremental_cost':offline_cost,'break_even_queries':break_even,
                   'offline_report':offline,
                   'saving_after_projected_rebuilds':per_query_saving-offline['projected_rebuild_cost_per_query'] if offline and per_query_saving is not None and offline['projected_rebuild_cost_per_query'] is not None else None,
                   'drop_highest_cost_question_sensitivity':sensitivity,
                   'claim':'investment gate only; no statistical noninferiority or cross-user generalization claim'})
