from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace
import numpy as np
import pytest
from eicr.contracts import ContractError,digest
from native_tag.core import (Scope,EmbeddingIdentity,EvidenceView,Isotonic,Selector,EncoderUnavailable,
                             fit_calibration,decision_sets)
from native_tag.runtime import (build_memory,load_runtime,Trace,Completion,ReplayTransport,
                                make_llm,install_trace,patch_source,verify_sources)
from native_tag.experiment import run_one,run_protocol,load_runs,stage_a,calibration_records,sealed
from native_tag.transport import bill
from conftest import MODEL,IDENTITY

class FakeEncoder:
    def __init__(self,spec,trace): self.identity=EmbeddingIdentity(**spec['identity']);self.trace=trace
    def encode(self,text):
        self.trace.calls.append({'billing_kind':'local_embedding','stage':self.trace.stage,'model':'fixture',
           'provider':'local','embedding_identity':asdict(self.identity),'status':'ok','request':{'input':text},
           'response':[1.,0.,0.],'wall_seconds':0.,'cpu_seconds':0.})
        return np.array([1.,0.,0.])


def comp(content=None,tools=None):
    message={'role':'assistant','content':json.dumps(content) if content is not None else None}
    if tools: message['tool_calls']=tools
    return Completion({'choices':[{'message':message}], 'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})


def tc(name,**args): return {'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}

class Script:
    def __init__(self):self.n=0
    def __call__(self,request):
        self.n+=1
        if self.n==1:return comp({'keywords':[{'id':'Alice','alternatives':[]}],'question_time':''})
        if self.n in (2,4):return comp({'tag_scores':{f't{i}':([0.,.699999,.7,.9][i%4]) for i in range(18)}})
        if self.n==3:return comp(tools=[tc('query_event_keywords',event_id='D1:19')])
        if self.n==5:return comp(tools=[tc('edges_by_tag',key='Bob',tag='t3',note='Read matching event'),
            tc('query_event_context',event_id='D1:19'),tc('query_conversation_time',event_id='D1:19'),
            tc('query_topic_events',topic='D1:t1'),tc('query_personal_information',person='Alice'),
            tc('query_personal_aspect',person='Alice',aspect='hobby')])
        return comp({'mode':'answer','answer':'Yoga','supports':['D1:1-1']})


def semantic(row):return {k:row[k] for k in ('answer','supports','events','state','native_answer_object','failed')}


def test_complete_native_replay_and_passthrough(protocol):
    baseline=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder,instrumented=False)
    assert baseline['error'] is None,baseline['error']
    assert not baseline['failed'],baseline['tool_failures']
    tape=[c for c in baseline['calls'] if c['billing_kind']=='chat_tokens']
    replay=ReplayTransport(tape)
    native=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:replay,encoder_factory=FakeEncoder)
    replay.finish()
    assert semantic(native)==semantic(baseline)
    assert [d['stage'] for d in native['decisions']]==['initial','expansion']
    assert native['decisions'][1]['context_text']=='Bob: original event 19.'
    assert set(e['method'] for e in native['events']) >= {'query_personal_aspect','query_topic_events','query_event_context','query_conversation_time'}
    native_calls=[c for c in native['calls'] if c['billing_kind']=='chat_tokens']
    transport=ReplayTransport([c for c in native_calls if c['stage'] not in ('initial_tag','expansion_tag')])
    responses=iter([d['response'] for d in native['decisions']])
    cheap=run_one(protocol,{**protocol['jobs'][0],'arm':'cheap_all'},transport_factory=lambda _:transport,
                  encoder_factory=FakeEncoder,selector_replay=lambda _:next(responses))
    transport.finish()
    assert semantic(cheap)==semantic(native)
    assert len(cheap['calls'])<len(native['calls'])


def test_scopes_dedup_top_three(frozen,asset):
    rt=load_runtime(frozen,MODEL); memory,view=build_memory(rt,asset)
    identity=view.identity
    assert view.links['Alice','t0']!=view.links['Bob','t0']
    with pytest.raises(ContractError):view.score([1,0,0],'Alice',['t0'],Scope('other','snap1',asset['hash']),identity)
    with pytest.raises(ContractError):view.score([1,0],'Alice',['t0'],view.scope,identity)
    with pytest.raises(ContractError):view.score([1,0,0],'Alice',['missing'],view.scope,identity)
    # Full source sets, not merely first source; shared pair dots computed once.
    e=memory.episode_events['D1:1-1']; e.origin='D1:1,D1:2'
    for eid,origin,vector in [('D1:2-1','D1:2,D1:1',[0,1,0]),('D1:3-1','D1:1,D1:3',[0,1,0]),
                               ('D1:4-1','D1:4',[-1,0,0]),('D1:5-1','D1:5',[1,0,0])]:
        memory.episode_events[eid].origin=origin;memory.episode_events[eid].embedding=vector
    memory.keys['Alice'].tag_dict['t0']=['D1:1-1','D1:2-1','D1:3-1','D1:4-1','D1:5-1']
    before=deepcopy(memory.event_by_tag('Alice','t0'))
    view=EvidenceView(memory,view.scope,identity,asset['event_input_texts'],provenance=asset['provenance'])
    scores=view.score([1,0,0],'Alice',['t0','t4'],view.scope,identity)
    assert len(view.links['Alice','t0'])==4
    assert scores['t0']==pytest.approx((1+float(np.dot(view.rows['D1:1-1'],[1,0,0])))/3)
    assert view.last_dot_products==4
    assert memory.event_by_tag('Alice','t0')==before


def teacher_records(n=20):
    return [{'request_id':str(i),'question_id':'dev','stage':'initial','raw':{'a':0.,'b':.4,'c':1.},
             'teacher':{'a':0.,'b':.3,'c':.9},'view_hash':'v','embedding_identity':IDENTITY} for i in range(n)]


def test_pava_weights_and_stage_support():
    m=Isotonic.fit([(0.,1.,1.),(.5,0.,3.),(1.,1.,1.)])
    assert m.y==(.25,.25,1.)
    assert m.predict(-10)==.25 and m.predict(10)==1.
    c=fit_calibration(teacher_records(),['dev'])
    assert c['ready'] and c['sources']=={'initial':'initial','expansion':'pooled'}
    assert not fit_calibration(teacher_records(19),['dev'])['ready']
    with pytest.raises(ContractError):fit_calibration(teacher_records(),['dev'],['dev'])
    rows=teacher_records();rows[0]['gold_answer']='leak'
    with pytest.raises(ContractError):fit_calibration(rows,['dev'])
    # One one-tag request and one three-tag request have equal total weight.
    a=[{'request_id':'a','question_id':'dev','stage':'initial','raw':{'a':.5},'teacher':{'a':1},'view_hash':'v','embedding_identity':IDENTITY},
       {'request_id':'b','question_id':'dev','stage':'initial','raw':{'a':.5,'b':.5,'c':.5},'teacher':{'a':0,'b':0,'c':0},'view_hash':'v','embedding_identity':IDENTITY}]
    points=[(x,r['teacher'][t],1/len(r['raw'])) for r in a for t,x in r['raw'].items()]
    assert Isotonic.fit(points).predict(.5)==pytest.approx(.5)


def test_fallback_identity_and_exact_cache(frozen,asset):
    rt=load_runtime(frozen,MODEL);_,view=build_memory(rt,asset);trace=Trace()
    enc=FakeEncoder({'identity':IDENTITY},trace);cal=fit_calibration(teacher_records(),['dev'])
    s=Selector('cheap_all',view,enc,cal);called=[]
    def native():called.append(1);return {'tag_scores':{'t0':.7}}
    s.score_tags('q','Alice',['t0'],'initial',view.scope,native)
    assert called==[]
    s.existing_query=('exact',[1,0,0]);n=len(trace.calls)
    s.score_tags('exact','Alice',['t0'],'initial',view.scope,native)
    assert len(trace.calls)==n
    s.score_tags('exact ','Alice',['t0'],'initial',view.scope,native)
    assert len(trace.calls)==n+1
    def unavailable(_):raise EncoderUnavailable('offline')
    enc.encode=unavailable
    assert s.score_tags('new','Alice',['t0'],'initial',view.scope,native)=={'tag_scores':{'t0':.7}}
    assert called==[1] and s.decisions[-1]['fallback']
    def broken(_):return [1,0]
    enc.encode=broken
    with pytest.raises(ContractError):s.score_tags('new','Alice',['t0'],'initial',view.scope,native)
    assert called==[1]


@pytest.mark.parametrize('size',[0,1,15])
def test_small_lists_do_not_trigger(frozen,asset,size):
    rt=load_runtime(frozen,MODEL);memory,view=build_memory(rt,asset)
    memory.keys['Alice'].tag_list=memory.keys['Alice'].tag_list[:size]
    llm=make_llm(rt,lambda _:pytest.fail('Unexpected model call'),MODEL,Trace())
    controller=rt.MemoryController(memory,llm);agent=rt.Agent(llm,memory,controller)
    agent.tag_selector=SimpleNamespace(score_tags=lambda *a:pytest.fail('Selector on small list'))
    agent.graph_scope=view.scope
    result=agent.select_key_tag('q',{'keywords':[{'id':'Alice'}]},[])
    assert result[0]==([{'key':'Alice','tags':[f't{i}' for i in range(size)]}] if size else [])


def test_threshold_and_tie_order():
    result=decision_sets({'tag_scores':{'z':0,'a':.699999,'b':.7,'c':.7,'d':1}},3)
    assert result=={'selected':['d','b','c'],'early':['d','b','c'],'zero':['z']}


def test_complete_runner_and_calibration_receipts(protocol,tmp_path):
    out=tmp_path/'runs'
    run_protocol(protocol,out,transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    p,rows=load_runs(out)
    assert len(rows)==1 and len(calibration_records(rows))==2
    audit=stage_a(rows,None,protocol=p)
    assert audit['decision']=='inconclusive' and audit['stages']['expansion']['calls']==1
    with pytest.raises(ContractError):run_protocol(protocol,out)


def test_unknown_cost_and_cache_discount():
    c={'stage':'initial_tag','provider':'p','model':'m','endpoint':'e','billing_kind':'chat_tokens','usage':{'prompt_tokens':100,'completion_tokens':10}}
    prices={'currency':'USD','effective_date':'2026-11-22','source':'fixture','rates':[
        {'provider':'p','model':'m','endpoint':'e','billing_kind':'chat_tokens','input_per_million':1.,'output_per_million':2.,'cached_input_per_million':.1}]}
    assert bill([c],prices)['amount'] is None
    c['usage']['prompt_tokens_details']={'cached_tokens':50}
    assert bill([c],prices)['amount']==pytest.approx(.000075)
    assert bill([c,{**c,'usage':None}],prices)['amount'] is None


def test_retry_attempt_receipts(frozen):
    rt=load_runtime(frozen,MODEL);trace=Trace();attempts=[]
    def transport(request):
        attempts.append(request)
        if len(attempts)==1:raise RuntimeError('transient')
        return comp({'ok':True})
    llm=make_llm(rt,transport,MODEL,trace)
    response=llm.chat_with_tool(messages=[{'role':'user','content':'hello'}],backoff=0.)
    assert response.choices and [r['status'] for r in trace.calls]==['error','ok']
    assert bill(trace.calls,None)['amount'] is None


def test_native_threshold_branches(frozen,asset):
    rt=load_runtime(frozen,MODEL);memory,view=build_memory(rt,asset);trace=Trace()
    response={'tag_scores':{f't{i}':(0. if i==0 else .699999 if i==1 else .7) for i in range(18)}}
    llm=make_llm(rt,lambda _:comp(response),MODEL,trace)
    controller=rt.MemoryController(memory,llm);agent=rt.Agent(llm,memory,controller)
    agent.tag_selector=Selector('native',view,FakeEncoder({'identity':IDENTITY},trace))
    agent.graph_scope=view.scope
    reads=[];original=controller.event_by_tag
    def reader(*args):reads.append(args[:2]);return original(*args)
    controller.event_by_tag=reader
    result=agent.select_key_tag('question',{'keywords':[{'id':'Alice'}]},[])
    assert result[0][0]['tags']==[f't{i}' for i in range(2,12)]
    assert len(reads)==10 and all(t!='t0' and t!='t1' for _,t in reads)


def test_expansion_native_in_initial_only_mode(frozen,asset):
    rt=load_runtime(frozen,MODEL);_,view=build_memory(rt,asset);trace=Trace()
    s=Selector('cheap_initial',view,FakeEncoder({'identity':IDENTITY},trace),fit_calibration(teacher_records(),['dev']))
    called=[]
    out=s.score_tags('event','Bob',['t0'],'expansion',view.scope,lambda:called.append(1) or {'tag_scores':{'t0':0}})
    assert out=={'tag_scores':{'t0':0}} and called==[1] and trace.calls==[]


def test_missing_graph_identity_fails_preflight(protocol,tmp_path):
    from native_tag.experiment import preflight
    ref=next(iter(protocol['assets'].values()));path=__import__('pathlib').Path(ref['path'])
    path.write_text(path.read_text()+' ')
    with pytest.raises(ContractError):preflight(protocol)


def test_malformed_teacher_keeps_spent_calls(protocol):
    class BadTeacher(Script):
        def __call__(self,request):
            if self.n==1:
                self.n+=1;return comp({'tag_scores':{'t0':'bad-score'}})
            return super().__call__(request)
    row=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:BadTeacher(),encoder_factory=FakeEncoder)
    assert row['failed'] and row['error'] is not None
    assert len([c for c in row['calls'] if c['billing_kind']=='chat_tokens'])==2
    assert row['decisions'][0]['decision_parse_failed']


def test_native_safety_limit_retains_failure(protocol):
    class TooManyTools(Script):
        def __call__(self,request):
            if self.n==2:
                self.n+=1
                return comp(tools=[{**tc('query_conversation_time',event_id='D1:1'),'id':str(i)} for i in range(51)])
            if self.n==3:
                self.n+=1;return comp({'mode':'answer','answer':'Yoga','supports':['D1:1-1']})
            return super().__call__(request)
    row=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:TooManyTools(),encoder_factory=FakeEncoder)
    assert row['answer']=='Yoga' and row['budget_exceeded'] and row['failed']
    calls=[e for e in row['events'] if e['method']=='call']
    assert len(calls[0]['args'][0])==50 # native truncation, not a new budget implementation
