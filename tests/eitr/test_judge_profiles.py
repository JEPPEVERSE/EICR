"""Judge rubric identity, fair reuse, and independent citation validity."""
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from eicr.contracts import ContractError, digest, read_json, write_json
from eicr.packer import CharacterTokenizer
from eitr.batch import run_batch
from eitr.config import Config
from eitr.evaluate import (JUDGE_PROMPT, MRAGENT_ACCURACY_PROMPT, MRAGENT_JUDGE_SOURCE,
                           answer_validity, grade_batch, judge_profile_metadata, summarize)
from eitr.index import build_index


class Answer:
    def __init__(self): self.calls=[]
    def chat_json(self, system, user):
        self.calls.append({'status':'ok','usage':{'prompt_tokens':7,'completion_tokens':3}})
        return {'answer_text':'herbs', 'citations':[], 'unsupported_premises':[],
                'resolution_status':'insufficient'}


class Judge:
    def __init__(self, result): self.calls=[];self.requests=[];self.result=result
    def chat_json(self, system, user):
        self.requests.append((system,user));self.calls.append({'status':'ok'})
        return deepcopy(self.result)


def batch(tmp_path, snapshot):
    idx=build_index(snapshot)
    q={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'Which herbs?'}
    labels=[{k:q[k] for k in ('question_id','owner_id','snapshot_id')}|{'references':['herbs'],'source_ids':['D1:2']}]
    run_batch(idx,[q],CharacterTokenizer(),config=Config(),arms=['P3','P4'],
              answer_backend=Answer(),output_dir=tmp_path/'batch')
    return idx,labels


def test_strict_default_keeps_original_prompt_and_blind_reuse(snapshot,tmp_path):
    idx,labels=batch(tmp_path,snapshot);judge=Judge({'correct':True,'reason':'equivalent answer'})
    rows=grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'strict')
    assert len(judge.requests)==1
    system,user=judge.requests[0];assert system==JUDGE_PROMPT
    assert json.loads(user)=={'question':'Which herbs?','references':['herbs'],'candidate':'herbs'}
    assert all(r['judge_profile']=='semantic_strict' and r['correct'] and r['answer_contract_valid'] for r in rows)
    assert rows[1]['judge_reused']


def test_lenient_uses_exact_pinned_rubric_for_every_arm(snapshot,tmp_path):
    idx,labels=batch(tmp_path,snapshot);judge=Judge({'label':' CORRECT '})
    rows=grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'lenient',judge_profile='mragent_lenient')
    assert len(judge.requests)==1
    assert judge.requests[0][1]==MRAGENT_ACCURACY_PROMPT.format(question='Which herbs?',gold_answer='herbs',generated_answer='herbs')
    assert all(r['correct'] and r['judge']['label']=='CORRECT' and r['judge_profile']=='mragent_lenient' for r in rows)
    journal=read_json(tmp_path/'lenient/manifest.json')
    assert journal['judge_profile']==judge_profile_metadata('mragent_lenient')
    assert journal['judge_profile']['source']['commit']==MRAGENT_JUDGE_SOURCE['commit']
    assert journal['judge_profile']['prompt_sha256']==hashlib.sha256(MRAGENT_ACCURACY_PROMPT.encode()).hexdigest()
    with pytest.raises(ContractError,match='exists'):
        grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'lenient',judge_profile='mragent_lenient')


def test_invalid_citations_are_not_made_valid_by_correct_judge(snapshot,tmp_path):
    idx,labels=batch(tmp_path,snapshot);m=read_json(tmp_path/'batch/manifest.json');item=m['completed'][0]
    r=read_json(tmp_path/'batch'/item['file']);r['rows'][0]['answer'].update(
        resolution_status='supported', citations=[{'source_id':'invented','start':0,'end':2}])
    write_json(tmp_path/'batch'/item['file'],r);item['content_hash']=digest(r);write_json(tmp_path/'batch/manifest.json',m)
    judge=Judge({'label':'CORRECT'})
    rows=grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'grades',judge_profile='mragent_lenient')
    assert rows[0]['correct'] and rows[0]['judge_correct']
    assert not rows[0]['answer_contract_valid'] and not rows[0]['citations_valid'] and not rows[0]['valid_and_correct']
    assert rows[0]['citation_errors']==['0:outside_legal_source']
    assert rows[1]['answer_contract_valid'] and rows[1]['valid_and_correct']
    assert rows[1]['judge_reused']  # Semantic judgement shared; provenance audited per arm.


def test_visibility_and_contract_checks_are_separate(snapshot):
    idx=build_index(snapshot); source=snapshot['sources'][0]
    run={'evidence_pack':{'spans':[{'source_id':source['source_id'],'start':0,'end':4}]},
         'answer':{'answer_text':'herbs','citations':[{'source_id':source['source_id'],'start':4,'end':5}],
                   'unsupported_premises':[],'resolution_status':'supported'}}
    assert answer_validity(run,idx)['citation_errors']==['0:outside_visible_evidence']
    run['answer']['citations']=[]
    result=answer_validity(run,idx)
    assert result['citations_valid'] and not result['answer_contract_valid']
    assert result['answer_contract_errors']==['supported_answer_without_citations']
    run['answer']['citations']=[{'source_id':source['source_id'],'start':True,'end':3}]
    assert not answer_validity(run,idx)['citations_valid']


@pytest.mark.parametrize('profile,verdict', [('semantic_strict',{'correct':'true','reason':'x'}),
                                           ('mragent_lenient',{'label':'yes'})])
def test_profile_rejects_invalid_output(snapshot,tmp_path,profile,verdict):
    idx,labels=batch(tmp_path,snapshot)
    with pytest.raises(ContractError,match='Invalid'):
        grade_batch(tmp_path/'batch',labels,idx,Judge(verdict),tmp_path/'invalid',judge_profile=profile)
    assert read_json(tmp_path/'invalid/manifest.json')['status']=='failed'


def test_unknown_profile_and_multi_gold_rejected_before_paid_call(snapshot,tmp_path):
    idx,labels=batch(tmp_path,snapshot);judge=Judge({'label':'CORRECT'})
    with pytest.raises(ContractError,match='Unknown judge profile'):
        grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'unknown',judge_profile='unknown')
    labels[0]['references'].append('paprika')
    with pytest.raises(ContractError,match='one gold'):
        grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'multi',judge_profile='mragent_lenient')
    assert not judge.calls


def test_summary_cannot_pool_rubrics():
    rows=[{'question_id':'q','owner_id':'a','arm':'P3','correct':True,'online_tokens':2},
          {'question_id':'q','owner_id':'a','arm':'P4','correct':True,'online_tokens':2,'judge_profile':'mragent_lenient'}]
    with pytest.raises(ContractError,match='mixed judge profiles'):summarize(rows)


def test_original_rubric_matches_pinned_upstream_when_available():
    path=os.environ.get('EITR_NATIVE_ROOT')
    if not path:pytest.skip('Set EITR_NATIVE_ROOT for pinned upstream rubric verification')
    for filename,expected in MRAGENT_JUDGE_SOURCE['files_sha256'].items():
        raw=subprocess.check_output(['git','-C',str(Path(path).resolve()),'show',MRAGENT_JUDGE_SOURCE['commit']+':'+filename])
        assert hashlib.sha256(raw).hexdigest()==expected
        if filename=='eval/judge.py':
            statement=next(x for x in ast.parse(raw).body if isinstance(x,ast.Assign)
                           and any(isinstance(t,ast.Name) and t.id=='ACCURACY_PROMPT' for t in x.targets))
            assert ast.literal_eval(statement.value)==MRAGENT_ACCURACY_PROMPT
