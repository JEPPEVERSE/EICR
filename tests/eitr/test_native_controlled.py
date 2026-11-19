"""Run against the user's pinned upstream checkout, never a rewritten fake loop.
Set EITR_NATIVE_ROOT to enable these integration checks on another machine.
"""
import json
import os
from pathlib import Path
import pytest
from eicr.contracts import ContractError
from eicr.packer import CharacterTokenizer
from eitr.native import canonicalize, import_pack
from eitr.index import build_index
from eitr.config import Config
from eitr.query import map_groups
from eitr.evidence import Corpus
from eitr.native_runtime import pin_runtime, load_runtime
from eitr.controlled import run_controlled, ControlledController
from eitr.pipeline import compare


@pytest.fixture
def runtime():
    root=os.environ.get('EITR_NATIVE_ROOT')
    if not root:pytest.skip('Set EITR_NATIVE_ROOT for actual upstream integration')
    return load_runtime(root,pin_runtime(root))


@pytest.fixture
def native_index(raw_memory):
    raw_memory['native_auxiliary']={'topics':[{'topic_id':'D1:t1','text':'cooking','fact_ids':['f1','f2']}],
        'personal':[{'personal_id':'D1:p1','person':'A','aspect':'cooking','origins':['D1:2']}]}
    return build_index(canonicalize(raw_memory))


class ScriptedTools:
    model='scripted-contract-test'
    def __init__(self,responses):self.responses=list(responses);self.calls=[];self.inputs=[]
    def chat_tools(self,messages,tools):
        self.inputs.append(messages)
        self.calls.append({'status':'ok','usage':{'prompt_tokens':10,'completion_tokens':3}})
        return self.responses.pop(0)


def tool(name,args,id='call-1'):
    return {'role':'assistant','tool_calls':[{'id':id,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}


def stop():return {'role':'assistant','content':json.dumps({'mode':'answer','answer':'native stop output','supports':[]})}


def test_actual_native_loop_executes_tool_and_shared_answer(runtime,native_index):
    backend=ScriptedTools([tool('edges_by_tag',{'key':'herbs','tag':'cooking','note':'Read herbs evidence'}),stop()])
    class Answer:
        def __init__(self):self.calls=[]
        def chat_json(self,system,user):
            assert 'native stop output' not in user
            self.calls.append({'status':'ok','usage':{'prompt_tokens':10,'completion_tokens':3}})
            return {'answer_text':'herbs','citations':[],'unsupported_premises':[],'resolution_status':'insufficient'}
    answer=Answer()
    row=run_controlled(native_index,'Which herbs?',CharacterTokenizer(),runtime,backend,answer_backend=answer)
    assert row['arm']=='P5'
    assert row['manifest']['baseline_identity']=='controlled_MRAgent_pinned_native_loop_and_seven_tools'
    assert len(runtime.tools)==7
    assert row['native_loop_result']['answer_discarded']=='native stop output'
    assert len(backend.calls)==2 and len(answer.calls)==1
    assert row['native_tools'][0]['calls'][0]['function']['name']=='edges_by_tag'
    assert any(m['role']=='tool' for m in backend.inputs[1])
    assert row['evidence_pack']['used_tokens']<=6000


def test_all_seven_native_tools_bridge_to_valid_raw_sources(runtime,native_index):
    cfg=Config();tok=CharacterTokenizer();corpus=Corpus(native_index.snapshot,tok,cfg)
    ctrl=ControlledController(native_index,map_groups(native_index,'herbs',cfg),corpus,cfg)
    ctrl.read_origins({'D1:1','D1:2'},'test-read')
    calls=[('edges_by_tag',{'key':'herbs','tag':'cooking','note':'Read source'}),
        ('query_conversation_time',{'event_id':'D1:2'}),('query_event_keywords',{'event_id':'D1:2'}),
        ('query_event_context',{'event_id':'D1:2'}),('query_personal_information',{'person':'A'}),
        ('query_personal_aspect',{'person':'A','aspect':'cooking'}),('query_topic_events',{'topic':'D1:t1'})]
    for i,(name,args) in enumerate(calls):
        result=runtime.bridge(ctrl).call(tool(name,args,str(i))['tool_calls'])
        assert len(result)==1 and "'error':" not in result[0]['content'],(name,result)
    assert ctrl.state.packet['used_tokens']<=cfg.evidence_budget
    with pytest.raises(ContractError):ctrl.query_conversation_time('D99:1')


def test_controlled_batch_uses_same_initial_frontend(runtime,native_index):
    b=ScriptedTools([stop()])
    result=compare(native_index,'herbs',CharacterTokenizer(),arms=['P0','P3','P4','P5'],native_runtime=runtime,exploration_backend=b)
    rows={r['arm']:r for r in result['rows']}
    assert rows['P3']['query']==rows['P4']['query']==rows['P5']['query']
    assert rows['P3']['candidates']['pool_hash']==rows['P4']['candidates']['pool_hash']


def test_invalid_native_tool_does_not_execute(runtime,native_index):
    b=ScriptedTools([tool('query_topic_events',{'topic':'D1:t1','extra':'bad'})])
    row=run_controlled(native_index,'herbs',CharacterTokenizer(),runtime,b)
    assert row['native_loop_result']['status']=='native_loop_error'
    assert row['native_tools']==[]


def test_source_pins_reject_changed_upstream(tmp_path):
    root=os.environ.get('EITR_NATIVE_ROOT')
    if not root:pytest.skip('Set EITR_NATIVE_ROOT')
    manifest=pin_runtime(root);manifest['files']['agent/tools.py']='changed'
    with pytest.raises(ContractError):load_runtime(root,manifest)


def test_import_preserves_auxiliary_and_filters_future_sources():
    pack={'raw_text':{'D1':{'D1:1':'A: herbs','D1:2':'A: future'}},'rewrites':[{'session_1':{
        'topics':{'t1':'herbs','t2':'future'},'personal_sentences':[{'id':'p1','person':'A','tag':'likes','origin':'D1:1','text':'herbs'}],
        'sentence':[{'id':'f1','text':'herbs','origin':['D1:1'],'tag':'food','topic':['t1']},
                    {'id':'f2','text':'future','origin':['D1:2'],'tag':'future','topic':['t2']}]
    }}],'keywords':[{'sentence':[{'sentence_id':'f1','keyword':['herbs']},{'sentence_id':'f2','keyword':['future']}]}]}
    m={'owner_id':'a','snapshot_id':'s','allowed_source_ids':['D1:1'],'extraction_prefix_verified':True,'keyword_cache_is_pre_injection':True}
    snapshot=import_pack(pack,m)
    assert snapshot['native_auxiliary']['topics']==[{'topic_id':'D1:t1','text':'herbs','fact_ids':['f1']}]
    assert snapshot['native_auxiliary']['personal'][0]['origins']==['D1:1']


def test_native_speaker_navigation_is_preserved_but_never_a_feature():
    pack={'raw_text':{'D1':{'D1:1':'Alice: hello'}},'rewrites':[{'session_1':{
        'sentence':[{'id':'f1','text':'hello','origin':['D1:1'],'tag':'greeting'}]}}],
        'keywords':[{'sentence':[{'sentence_id':'f1','keyword':['hello']}]}]}
    manifest={'owner_id':'a','snapshot_id':'s','allowed_source_ids':['D1:1'],'extraction_prefix_verified':True,'keyword_cache_is_pre_injection':True}
    snapshot=import_pack(pack,manifest)
    speaker=next(c['cue_id'] for c in snapshot['cues'] if c['text']=='Alice')
    assert any(l['cue_id']==speaker and l['edge_origin']=='injected_speaker' for l in snapshot['links'])
    assert all(c!=speaker for c,f in snapshot['feature_edges'])
    index=build_index(snapshot)
    assert map_groups(index,'Alice',Config())['groups'][0]['seed_tags']
    assert index.profiles[:,index.cue_pos[speaker]].nnz==0
