import json
from dataclasses import replace
import pytest
from eicr.contracts import ContractError, digest, write_json
from eicr.packer import CharacterTokenizer
from eitr.cli import main
from eitr.index import build_index, save_index
from eitr.config import Config
from eitr.pipeline import retrieve, compare
from eitr.batch import run_batch
from eitr.explorer import explore
from eitr.evaluate import summarize
from eitr.evidence import Corpus


class FakeBackend:
    def __init__(self, kind): self.kind=kind; self.calls=[]
    def chat_json(self, system, user):
        self.calls.append({'status':'ok','usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}})
        if self.kind=='frontend': return {'mentions':[{'surface_text':'herbs'}]}
        if self.kind=='invalid_frontend': return {'mentions':[{'surface_text':'not in question'}]}
        if self.kind=='stop': return {'reads':[], 'stop':True}
        if self.kind=='invent': return {'reads':['invented'], 'stop':False}
        return {'answer_text':'Insufficient evidence.', 'citations':[], 'unsupported_premises':[], 'resolution_status':'insufficient'}


def test_one_frontend_and_answer_no_exploration(snapshot):
    index=build_index(snapshot); f=FakeBackend('frontend');a=FakeBackend('answer')
    row=retrieve(index,'Which herbs?',CharacterTokenizer(),frontend_backend=f,answer_backend=a)
    assert len(f.calls)==len(a.calls)==1
    assert row['calls']['exploration']==[]
    assert row['answer']['resolution_status']=='insufficient'
    assert row['manifest']['status']=='answered'


def test_bad_frontend_bounded_retry_and_fallback(snapshot):
    f=FakeBackend('invalid_frontend')
    row=retrieve(build_index(snapshot),'Which herbs?',CharacterTokenizer(),frontend_backend=f)
    assert len(f.calls)==2
    assert row['evidence_pack']['fallback']
    assert len(row['query']['frontend_errors'])==2


def test_flat_baseline_does_not_pay_frontend(snapshot):
    f=FakeBackend('frontend')
    result=compare(build_index(snapshot),'Which herbs?',CharacterTokenizer(),frontend_backend=f)
    assert len(f.calls)==1
    assert result['rows'][0]['calls']['frontend']==[]


def test_flat_only_compare_never_calls_frontend(snapshot):
    f = FakeBackend('frontend')
    result = compare(build_index(snapshot), 'Which herbs?', CharacterTokenizer(),
                     arms=['P0'], frontend_backend=f)
    assert f.calls == result['frontend_calls_once_in_experiment'] == []


def test_config_mismatch_fails_before_paid_frontend(snapshot):
    f = FakeBackend('frontend')
    with pytest.raises(ContractError):
        compare(build_index(snapshot), 'Which herbs?', CharacterTokenizer(),
                config=Config(top_k=8), frontend_backend=f)
    assert f.calls == []


def test_nonobject_frontend_retries_then_falls_back(snapshot):
    class NonobjectBackend(FakeBackend):
        def chat_json(self, system, user):
            self.calls.append({'status': 'ok'})
            return []
    f = NonobjectBackend('invalid')
    row = retrieve(build_index(snapshot), 'Which herbs?', CharacterTokenizer(), frontend_backend=f)
    assert len(f.calls) == 2
    assert row['evidence_pack']['fallback']


def test_batch_scope_gold_isolation_and_safe_filenames(snapshot,tmp_path):
    index=build_index(snapshot);tokenizer=CharacterTokenizer();cfg=Config()
    q={'question_id':'../../escape','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'Which herbs?'}
    with pytest.raises(ContractError):
        run_batch(index,[{**q,'gold_answer':'herbs'}],tokenizer,config=cfg,arms=['P3','P4'],output_dir=tmp_path/'bad')
    manifest=run_batch(index,[q],tokenizer,config=cfg,arms=['P3','P4'],answer_backend=FakeBackend('answer'),output_dir=tmp_path/'good')
    assert manifest['status']=='complete'
    assert '/' not in manifest['completed'][0]['file']
    row=json.loads((tmp_path/'good'/manifest['completed'][0]['file']).read_text())
    assert row['same_pool_verified']
    assert len(row['rows'])==2


def test_cli_roundtrip_no_network(snapshot,tmp_path):
    path=tmp_path/'snap.json';write_json(path,snapshot)
    idx=tmp_path/'index.json'
    assert main(['build','--snapshot',str(path),'--out',str(idx)])==0
    assert main(['audit','--index',str(idx)])==0
    out=tmp_path/'compare.json'
    args=['compare','--index',str(idx),'--question','Which herbs?','--tokenizer','characters','--out',str(out)]
    assert main(args)==0
    assert main(args)==2
    assert json.loads(out.read_text())['same_pool_verified']
    assert main(['retrieve','--index',str(idx),'--question','herbs','--tokenizer','characters','--generate',
                 '--api-base','https://example.invalid','--model','test','--out',str(tmp_path/'live.json')])==2


def test_control_loop_is_explicitly_not_native_mragent(snapshot):
    index=build_index(snapshot)
    stop=FakeBackend('stop')
    row=explore(index,'herbs halloumi',CharacterTokenizer(),stop,config=Config(evidence_budget=700))
    assert row['arm']=='P5_proxy'
    assert row['manifest']['baseline_identity'].endswith('not_native_MRAgent')
    assert row['evidence_pack']['used_tokens']<=700
    assert len(stop.calls)<=1
    bad=FakeBackend('invent')
    row=explore(index,'herbs halloumi',CharacterTokenizer(),bad)
    assert row['stop']=='explorer_error'


def test_reject_cross_config_corpus(snapshot):
    index=build_index(snapshot);tokenizer=CharacterTokenizer()
    corpus=Corpus(snapshot,tokenizer,Config(atom_tokens=20))
    with pytest.raises(ContractError): retrieve(index,'herbs',tokenizer,corpus=corpus)


def test_summary_counts_unique_questions():
    rows=[{'question_id':'q','owner_id':'a','arm':'P3','correct':False,'online_tokens':10},
          {'question_id':'q','owner_id':'a','arm':'P4','correct':True,'online_tokens':10}]
    result=summarize(rows)
    assert result['pairs']['P3 vs P4']['b_wins']==1
    with pytest.raises(ContractError): summarize(rows+rows)
