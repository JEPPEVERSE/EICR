import json
import pytest
from eicr.contracts import ContractError
from eicr.packer import CharacterTokenizer
from eitr.tool_backend import ToolBackend
from eitr.index import build_index
from eitr.evidence import Corpus
from eitr.config import Config
from eitr.pipeline import compare
from eitr.batch import run_batch


class Response:
    def __init__(self,data):self.raw=json.dumps(data).encode()
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,size):return self.raw[:size]


def test_tool_transport_keeps_native_schema_and_usage(monkeypatch):
    monkeypatch.setenv('EITR_TEST_KEY','fake-contract-test-key')
    import eitr.tool_backend as module
    def respond(req,timeout):
        payload=json.loads(req.data)
        assert 'response_format' not in payload
        assert payload['tools']==[{'type':'function','function':{'name':'test'}}]
        return Response({'model':'resolved-model','system_fingerprint':'fp-test','usage':{'prompt_tokens':11,'completion_tokens':3},
            'choices':[{'finish_reason':'tool_calls','message':{'role':'assistant','tool_calls':[{'id':'x','type':'function',
                'function':{'name':'test','arguments':'{}'}}]}}]})
    monkeypatch.setattr(module,'_open_no_redirect',respond)
    b=ToolBackend('https://example.invalid','requested','EITR_TEST_KEY')
    out=b.chat_tools([{'role':'user','content':'test'}],[{'type':'function','function':{'name':'test'}}])
    assert out['tool_calls'][0]['id']=='x'
    assert b.calls[0]['response_model']=='resolved-model' and b.calls[0]['usage']['prompt_tokens']==11
    assert 'fake-contract-test-key' not in json.dumps(b.calls)


def test_truncated_tool_response_is_a_charged_failure(monkeypatch):
    monkeypatch.setenv('EITR_TEST_KEY','fake')
    import eitr.tool_backend as module
    monkeypatch.setattr(module,'_open_no_redirect',lambda *a,**k:Response({'usage':{'prompt_tokens':10,'completion_tokens':20},
        'choices':[{'finish_reason':'length','message':{'role':'assistant','content':'partial'}}]}))
    b=ToolBackend('https://example.invalid','test','EITR_TEST_KEY')
    with pytest.raises(ContractError):b.chat_tools([],[])
    assert b.calls[0]['status']=='error' and b.calls[0]['usage']['completion_tokens']==20


def test_bad_shared_corpus_rejected_before_frontend(snapshot):
    class Front:
        calls=[]
        def chat_json(self,*args):pytest.fail('A mismatched corpus must fail before calling a model')
    tok=CharacterTokenizer();wrong=Corpus(snapshot,tok,Config(atom_tokens=16))
    with pytest.raises(ContractError):compare(build_index(snapshot),'herbs',tok,corpus=wrong,frontend_backend=Front())


def test_answer_failure_keeps_retrieved_packet_and_usage(snapshot,tmp_path):
    class Broken:
        def __init__(self):self.calls=[]
        def chat_json(self,*args):
            self.calls.append({'status':'error','usage':{'prompt_tokens':7,'completion_tokens':1}})
            raise ContractError('test failure')
    q={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'herbs'}
    out=tmp_path/'batch'
    with pytest.raises(ContractError):run_batch(build_index(snapshot),[q],CharacterTokenizer(),config=Config(),arms=['P4'],answer_backend=Broken(),output_dir=out)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['status']=='failed' and manifest['failure_call_traces']['answer'][0]['usage']['prompt_tokens']==7
    packets=[p for p in out.glob('*.json') if p.name!='manifest.json']
    assert len(packets)==1
    assert json.loads(packets[0].read_text())['rows'][0]['evidence_pack']['atom_ids']
