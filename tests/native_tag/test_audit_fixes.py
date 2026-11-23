"""Regression coverage for the seven audit defects and missing protocol/report paths."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from eicr.contracts import ContractError,digest
from native_tag import experiment as ex
from native_tag import cli
from native_tag.core import Selector,fit_calibration,EncoderUnavailable
from native_tag.runtime import Trace,build_memory,load_runtime,write,graph_digest,clean,make_llm,Completion
from native_tag.grading import (grade_runs,apply_adjudication,rubric_manifest,
                                export_source_review,apply_source_review)
from native_tag.assets import prepare_assets
from native_tag.reporting import offline_costs
from test_native import FakeEncoder,Script,teacher_records,comp,semantic
from test_protocol import synthetic_analysis,synthetic_grades
from conftest import MODEL,IDENTITY,AdaptiveScript


def reseal(value, **updates):
    return ex.sealed({**{k:v for k,v in value.items() if k!='hash'},**updates})


def test_replay_message_preserves_sdk_exclude_none():
    payload = {'role':'assistant', 'content':None, 'refusal':None,
               'tool_calls':[{'id':'call1', 'type':'function', 'index':None,
                              'function':{'name':'query_event_context', 'arguments':'{}'}}]}
    completion = Completion({'choices':[{'message':payload}]})
    message = completion.choices[0].message
    assert message.model_dump() == payload
    assert message.model_dump(exclude_none=True) == {
        'role':'assistant', 'tool_calls':[{'id':'call1', 'type':'function',
                                         'function':{'name':'query_event_context', 'arguments':'{}'}}]}
    assert completion.model_dump()['choices'][0]['message'] == payload


@pytest.mark.parametrize('collect_raw,shadow',[(False,False),(True,False),(True,True)])
def test_native_never_returns_calibrated_teacher(frozen,asset,collect_raw,shadow):
    _,view=build_memory(load_runtime(frozen,MODEL),asset)
    selector=Selector('native',view,FakeEncoder({'identity':IDENTITY},Trace()),
                      fit_calibration(teacher_records(),['dev']),collect_raw=collect_raw,shadow=shadow)
    teacher={'tag_scores':{'t0':0.0}}
    assert selector.score_tags('context','Alice',['t0'],'initial',view.scope,lambda:deepcopy(teacher))==teacher
    d=selector.decisions[0]
    assert d['teacher_response']==d['response']==teacher
    if shadow: assert d['shadow_response']!=teacher


def test_malformed_usage_keeps_attempts(protocol,tmp_path):
    class BadUsage(Script):
        def __call__(self,req):
            response=super().__call__(req)
            response.data['usage']['prompt_tokens_details']='bad nested metadata'
            return response
    p=reseal(protocol,prices={'currency':'USD','effective_date':'2026-11-22','source':'fixture','rates':[
        {'provider':'offline','model':'fixture','endpoint':'offline://chat','billing_kind':'chat_tokens',
         'input_per_million':1,'output_per_million':1,'cached_input_per_million':None}]})
    ex.run_protocol(p,tmp_path/'usage',transport_factory=lambda _:BadUsage(),encoder_factory=FakeEncoder)
    _,rows=ex.load_runs(tmp_path/'usage');row=rows[0]
    assert len([c for c in row['calls'] if c['billing_kind']=='chat_tokens'])==6
    assert row['billing']['amount'] is None and not row['billing']['complete']
    assert {c['unknown_reason'] for c in row['billing']['calls']} >= {'malformed_usage'}
    receipt=ex.verify(json.loads((tmp_path/'usage/ledgers/0000.json').read_text()))
    assert receipt['calls']==row['calls'] and receipt['wall_seconds']>0


def test_postprocessing_failure_retains_attempt_ledger(protocol,tmp_path):
    calls=0
    def fail_after_execution(memory):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('postprocessing failed')
        return graph_digest(memory)
    with patch.object(ex,'graph_digest',fail_after_execution):
        ex.run_protocol(protocol,tmp_path/'post',transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    _,rows=ex.load_runs(tmp_path/'post');row=rows[0]
    assert row['error']['phase']=='postprocess' and len(row['calls'])>6
    assert row['wall_seconds']>0 and row['billing']['amount'] is None
    assert json.loads((tmp_path/'post/ledgers/0000.json').read_text())['status']=='failed'


def test_shadow_preserves_native_unknown_response_tags(protocol):
    """Provider-added tags have no native edges; shadow diagnostics must not crash."""
    class ExtraTagScript(Script):
        def __call__(self, request):
            response = super().__call__(request)
            if self.n in (2, 4):
                message = response.data['choices'][0]['message']
                payload = json.loads(message['content'])
                payload['tag_scores']['not-in-graph'] = 1.0
                return comp(payload)
            return response

    baseline = ex.run_one(protocol, protocol['jobs'][0],
                          transport_factory=lambda _: ExtraTagScript(),
                          encoder_factory=FakeEncoder, instrumented=False)
    assert not baseline['failed']
    shadow_protocol = reseal(protocol, shadow=True,
                             calibration=fit_calibration(teacher_records(), ['dev']))
    shadow = ex.run_one(shadow_protocol, shadow_protocol['jobs'][0],
                        transport_factory=lambda _: ExtraTagScript(), encoder_factory=FakeEncoder)
    assert semantic(shadow) == semantic(baseline)
    assert len(shadow['decisions']) == 2
    for decision in shadow['decisions']:
        assert 'not-in-graph' in decision['sets']['selected']
        assert 'not-in-graph' not in decision['ordered_tag_ids']
        assert decision['native_unlinked_selected_tags'] == ['not-in-graph']
        assert set(decision['native_reachable_sources']) <= {f'D1:{i}' for i in range(1, 37)}
    assert ex.calibration_records([shadow]) == []


def test_stage_a_calibration_shadow_identity_gates(primary_spec):
    spec,_,_=primary_spec
    assert ex.freeze_protocol(spec)
    development={k:deepcopy(spec[k]) for k in ('runtime','assets','model','embedding','seed','prices')}
    development.update(phase='development',questions=spec['diagnostic_questions'],repeats=1)
    baseline=ex.freeze_protocol(development)
    calibrated=ex.freeze_protocol({**development,'calibration':spec['calibration']})
    rows=[ex.run_one(p,p['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
          for p in (baseline,calibrated)]
    assert semantic(rows[0])==semantic(rows[1])
    assert [d['teacher_response'] for d in rows[0]['decisions']]==[d['teacher_response'] for d in rows[1]['decisions']]
    for key in ('model','embedding'):
        changed=deepcopy(spec)
        if key=='model': changed['model']={**changed['model'],'model':'different'}
        else: changed['embedding']={**changed['embedding'],'device':'other'}
        with pytest.raises(ContractError): ex.freeze_protocol(changed)
    changed=deepcopy(spec);changed['shadow_acceptance']=reseal(changed['shadow_acceptance'],accepted=False)
    with pytest.raises(ContractError,match='shadow'): ex.freeze_protocol(changed)
    changed=deepcopy(spec);changed['calibration']=reseal(changed['calibration'],provenance={})
    with pytest.raises(ContractError,match='Teacher'): ex.freeze_protocol(changed)


def test_analysis_rejects_unbound_judge_rubric_and_rows(protocol):
    p,runs,grades=synthetic_analysis(protocol)
    with pytest.raises(ContractError,match='complete'): ex.analyze(p,runs,grades['grades'])
    with pytest.raises(ContractError,match='model'): ex.analyze(p,runs,reseal(grades,model={**p['judge'],'model':'other'}))
    changed=reseal(grades,rubrics=reseal(grades['rubrics'],strict='changed rubric'))
    with pytest.raises(ContractError,match='rubric'): ex.analyze(p,runs,changed)
    changed=deepcopy(grades);changed['grades'][1]['score']=0
    with pytest.raises(ContractError,match='inconsistent'): ex.analyze(p,runs,reseal(changed))
    assert 'eitr/evaluate.py' in ex.implementation_hashes() and 'eicr/contracts.py' in ex.implementation_hashes()


def primary_result(spec,labels,costs,tmp_path):
    p=ex.freeze_protocol(spec);runs=[]
    for job in p['jobs']:
        amount=1. if job['arm']=='native' else .8
        runs.append(ex.sealed({**job,'protocol_hash':p['hash'],'answer':'Yoga','supports':['D1:1'],
            'answer_format_valid':True,'sources_valid':True,'support_texts':{'D1:1':'Alice does Yoga'},
            'failed':False,'calls':[{'stage':'tool_loop_and_answer'}],
            'billing':{'amount':amount,'complete':True,'calls':[{'amount':amount}]},
            'wall_seconds':1.,'decisions':[{'fallback':False,'stage':'initial','ordered_tag_ids':[],'mode':job['arm']}]}))
    def judge(system,user):
        return {'correct':True,'reason':'fixture','partial':False,'temporal_ambiguity':False,'gold_concern':False} if 'Return JSON:' in system else {'label':'CORRECT'}
    grades=grade_runs(runs,labels,judge,p['judge'],tmp_path/'valid-grades',protocol=p)
    return p,runs,grades,ex.analyze(p,runs,grades,offline=costs)


def test_confirmation_and_diagnostic_freezes(primary_spec,tmp_path):
    spec,labels,costs=primary_spec
    parent,runs,grades,result=primary_result(spec,labels,costs,tmp_path)
    assert result['decision']=='continue'
    confirm=deepcopy(spec);confirm.update(phase='confirmation',stage_c=result,previous_protocol=parent,diagnostic_questions=[])
    pool=[{**q,'question_id':'new-'+q['question_id'],'question':'New '+q['question']} for q in labels]
    confirm['questions'],_,confirm['sample_manifest']=ex.select_questions(pool,set(),set(),owners=spec['sample_manifest']['owners'],
        salt='confirmation',evidence_by_scope={(q['owner'],q['snapshot']):['D1:1'] for q in pool})
    assert ex.freeze_protocol(confirm)['previous_protocol_hash']==parent['hash']
    for field,value in [('model',{**MODEL,'model':'other'}),('seed',99),('calibration',None)]:
        changed={**confirm,field:value}
        with pytest.raises((ContractError,TypeError)): ex.freeze_protocol(changed)
    diag={**spec,'phase':'diagnostic','previous_protocol':parent,'stage_c':result,
          'questions':spec['diagnostic_questions'],'diagnostic_questions':[]}
    with pytest.raises(ContractError,match='quality loss'): ex.freeze_protocol(diag)
    loss=reseal(result,aggregate={**result['aggregate'],'cheap_all':{**result['aggregate']['cheap_all'],'accuracy':.9}})
    diag['stage_c']=loss
    assert ex.freeze_protocol(diag)['arms']==['native','cheap_initial']
    with pytest.raises(ContractError,match='subset'): ex.freeze_protocol({**diag,'questions':[{**diag['questions'][0],'question':'posthoc'}]})


def test_incremental_review_is_globally_symmetric(protocol,tmp_path):
    row=ex.run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    pair=[row,reseal(row,arm='cheap_all')]
    def judge(system,user):
        return {'correct':False,'reason':'partial','partial':True,'temporal_ambiguity':False,'gold_concern':False} if 'Return JSON:' in system else {'label':'WRONG'}
    g=grade_runs(pair,[{**row['question'],'reference':'Yoga and strength'}],judge,MODEL,tmp_path/'review')
    items=json.loads((tmp_path/'review/adjudication/blind.json').read_text())['items']
    mapping=json.loads((tmp_path/'review/adjudication/private-mapping.json').read_text())
    first=apply_adjudication(g,mapping,[{**items[0],'score':1,'reason':'complete'}],reviewer='reviewer')
    strict=[x for x in first['grades'] if x['rubric']=='strict']
    assert all(x['score']==1 and not x['unresolved'] and x['raw_score']==0 for x in strict)
    with pytest.raises(ContractError,match='symmetric'):
        apply_adjudication(first,mapping,[{**items[1],'score':0,'reason':'conflicting'}],reviewer='reviewer')
    again=apply_adjudication(first,mapping,[{**items[1],'score':1,'reason':'agrees'}],reviewer='reviewer')
    assert all(x['raw_score']==0 for x in again['grades'] if x['rubric']=='strict')


def test_grade_failure_persists_attempts(protocol,tmp_path):
    p=reseal(protocol,judge=MODEL)
    row=ex.run_one(p,p['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    write(tmp_path/'judge.json',MODEL);write(tmp_path/'labels.json',[{**row['question'],'reference':'Yoga'}])
    calls=[]
    def judge(request):
        calls.append(request)
        return comp({'correct':True,'reason':'ok','partial':False,'temporal_ambiguity':False,'gold_concern':False} if len(calls)==1 else {'label':'INVALID'})
    with patch.object(cli,'load_runs',return_value=(p,[row])),patch.object(cli,'live_transport',return_value=judge):
        status=cli.main(['grade','--runs','unused','--labels',str(tmp_path/'labels.json'),'--judge',str(tmp_path/'judge.json'),'--out',str(tmp_path/'grades')])
    ledger=ex.verify(json.loads((tmp_path/'grades.judge-calls.json').read_text()))
    assert status==2 and len(calls)==len(ledger['calls'])==2 and ledger['status']=='failed'
    assert ledger['calls'][0]['usage']['prompt_tokens']==10


def test_reembedding_failure_persists_attempts(frozen,asset,tmp_path):
    paths={}
    for key in ('raw_text','rewrites','keywords'):
        path=tmp_path/(key+'.json');write(path,asset[key]);paths[key]=str(path)
    class FailingEncoder(FakeEncoder):
        def encode(self,text):
            vector=super().encode(text)
            if len(self.trace.calls)==2:
                self.trace.calls[-1]['status']='error'
                raise EncoderUnavailable('fixture outage')
            return vector
    spec={'runtime':str(frozen),**paths,'owner':'owner1','snapshot':'snap1','embedding':{'identity':IDENTITY}}
    with pytest.raises(EncoderUnavailable): prepare_assets(spec,reembed=True,encoder_factory=FailingEncoder,receipt_path=tmp_path/'receipt.json')
    receipt=ex.verify(json.loads((tmp_path/'receipt.json').read_text()))
    assert receipt['status']=='failed' and len(receipt['calls'])==2 and receipt['cpu_seconds']>0


def test_source_review_and_offline_reporting(primary_spec,tmp_path):
    spec,labels,costs=primary_spec
    p,runs,grades,_=primary_result(spec,labels,costs,tmp_path)
    bundle=export_source_review(p,runs,tmp_path/'source-review')
    items=json.loads((tmp_path/'source-review/blind.json').read_text())['items']
    assert all('arm' not in x and 'cost' not in x for x in items)
    review=apply_source_review(bundle,[{**x,'verdict':'supported','reason':'source entails the answer'} for x in items],reviewer='independent')
    result=ex.analyze(p,runs,grades,offline=costs,source_review=review)
    assert result['source_assessment']['native']['support_verdicts']=={'supported':64}
    assert result['offline_incremental_cost']>0 and result['break_even_queries']>0
    assert result['saving_after_projected_rebuilds'] is not None
    with pytest.raises(ContractError): ex.analyze(p,runs,grades,offline=reseal(costs,asset_hashes=[]))
    with pytest.raises(ContractError): ex.analyze(p,runs,grades,offline=costs,source_review=reseal(review,protocol_hash='other'))


def test_offline_unknown_duplicate_and_rebuild_policy():
    prices={'currency':'USD','effective_date':'2026-11-22','source':'test','rates':[]}
    r=ex.sealed({'ledger_id':'a','status':'failed','components':['embedding','index','teacher_calibration'],
        'calls':[{'stage':'offline_embedding','billing_kind':'embedding_tokens','usage':None}],
        'wall_seconds':1.,'cpu_seconds':.5})
    policy={'trigger':'new snapshot','period':'month','expected_builds':None,'queries_per_period':None}
    result=offline_costs([r],prices,policy)
    assert result['incremental_api_cost'] is None and result['projected_rebuild_cost_per_query'] is None
    with pytest.raises(ContractError,match='Duplicate'): offline_costs([r,r],prices,policy)
    with pytest.raises(ContractError): offline_costs([r],prices,{})


def test_adaptation_error_aborts_remaining_jobs(protocol,tmp_path):
    p=reseal(protocol,repeats=2,jobs=[protocol['jobs'][0],{**protocol['jobs'][0],'trial':1}])
    class WrongVector(FakeEncoder):
        def encode(self,text):
            super().encode(text)
            return np.array([1.,0.])
    result=ex.run_protocol(p,tmp_path/'abort',transport_factory=lambda _:Script(),encoder_factory=WrongVector)
    _,rows=ex.load_runs(tmp_path/'abort')
    assert result['status']=='aborted' and rows[0]['adaptation_error']
    assert rows[1]['error']['phase']=='aborted' and rows[1]['calls']==[]


def test_revised_sampling_design_is_frozen():
    owners=['a','b','c','d'];pool=[]
    for o in owners:
        for c in (1,2,3,4):
            for i in range(3):pool.append({'owner':o,'snapshot':'s','question_id':f'{o}{c}{i}',
                'question':f'question {o}{c}{i}','category':c,'reference':'answer','evidence':['D1:1']})
    design,_=ex.sample_design(owners);design=deepcopy(design);design['rationale']='declared stratum shortage'
    design['strata'][0]['count']=1;design['strata'][1]['count']=3
    q,_,m=ex.select_questions(pool,set(),set(),owners=owners,salt='frozen',evidence_by_scope={(o,'s'):['D1:1'] for o in owners},design=design)
    assert len(q)==32 and m['design']==design and sum(x['owner']=='a' and x['category']==1 for x in q)==1


def test_long_native_readers_and_temporal_bypass(frozen,asset):
    """Exercise previously unvisited K2/RERANK_LIMIT and time-only branches."""
    results=[]
    for instrumented in (False,True):
        rt=load_runtime(frozen,MODEL,instrumented=instrumented);memory,view=build_memory(rt,asset)
        ids=list(memory.episode_events);texts=[memory.episode_events[i].text for i in ids]
        vectors=[memory.episode_events[i].embedding for i in ids];trace=Trace();requests=[]
        responses=iter([{'events':ids[:25]},{'relevance_scores':{eid:1-i/100 for i,eid in enumerate(ids)}}])
        def transport(req): requests.append(req);return comp(next(responses))
        llm=make_llm(rt,transport,MODEL,trace);controller=rt.MemoryController(memory,llm);agent=rt.Agent(llm,memory,controller)
        controller.question_emb=np.array([1.,0.,0.]);memory.keys['Alice'].tag_dict['t0']=ids
        reads=controller.event_by_tag('Alice','t0','read matching sources')
        assert len(reads[2])==rt.config.K2 and len(controller.queried_event)==rt.config.K2
        first=agent.select_finegrained_sentence('q',controller.question_emb,texts,ids,vectors)
        second=agent.select_finegrained_sentence_sort('q',controller.question_emb,texts,ids,vectors)
        assert len(first[0])==len(second[0])==rt.config.K2
        temporal=agent.answer_question_with_time('q','2026-01-01, 2026-01-01',{'keywords':[{'id':'Alice'}]},2)
        assert temporal['similar_sentence'] and len(requests)==2
        results.append(clean({'reads':reads,'first':first,'second':second,'temporal':temporal,'requests':requests}))
    assert results[0]==results[1]


def test_independent_native_module_loading_matches_runner(protocol,monkeypatch):
    """Execute unmodified reference modules, independently of the AST definition loader.

    Only process config, dotenv and SDK construction are substituted for offline replay.
    Native module imports, assignments, function defaults and class bodies execute intact.
    """
    import sys
    from types import ModuleType,SimpleNamespace
    from native_tag.runtime import ReplayTransport,verify_sources
    root=Path(protocol['runtime'])/'reference';manifest=verify_sources(protocol['runtime'])
    modules={}
    for name in ('common','prompts','memory','llm','agent'):
        package=ModuleType(name);package.__path__=[str(root/name)];modules[name]=package
    config=ModuleType('common.config')
    for key,value in {**manifest['limits'],'MODEL':MODEL['model'],'RE_MODEL':MODEL['model'],
                      'MODEL_SORT':MODEL['model'],'MODEL_NAME':MODEL['model'],'PROVIDER':MODEL['provider'],'dataset':'locomo'}.items():
        setattr(config,key,value)
    modules['common.config']=config;modules['common'].config=config
    sdk=ModuleType('openai');sdk.OpenAI=lambda **kw:pytest.fail('Unexpected SDK construction')
    for name in ('APIStatusError','APIConnectionError','APIResponseValidationError'):setattr(sdk,name,type(name,(Exception,),{}))
    modules['openai']=sdk
    dotenv=ModuleType('dotenv');dotenv.load_dotenv=lambda *a,**k:None;modules['dotenv']=dotenv
    old_path=list(sys.path)
    with patch.dict(sys.modules,modules):
        try:
            loaded={}
            for name in ('common.model_config','common.utils','prompts.prompts','prompts.schema',
                         'memory.system','llm.controller','memory.controller','agent.tools','agent.agent'):
                path=root/(name.replace('.','/')+'.py');module=ModuleType(name);module.__file__=str(path)
                sys.modules[name]=module;setattr(sys.modules[name.rsplit('.',1)[0]],name.rsplit('.',1)[1],module)
                exec(compile(path.read_text(),str(path),'exec'),module.__dict__);loaded[name]=module
            rt=SimpleNamespace(Agent=loaded['agent.agent'].Agent,MemoryController=loaded['memory.controller'].MemoryController,
                MemorySystem=loaded['memory.system'].MemorySystem,LLM=loaded['llm.controller'].LLM,
                API_STATS=loaded['llm.controller'].API_STATS,extract_json_from_content=loaded['common.utils'].extract_json_from_content,
                manifest=manifest)
            with patch.object(ex,'load_runtime',return_value=rt):
                baseline=ex.run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder,instrumented=False)
        finally: sys.path[:]=old_path
    assert not baseline['failed']
    replay=ReplayTransport([c for c in baseline['calls'] if c['billing_kind']=='chat_tokens'])
    actual=ex.run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:replay,encoder_factory=FakeEncoder)
    replay.finish()
    assert semantic(actual)==semantic(baseline)
