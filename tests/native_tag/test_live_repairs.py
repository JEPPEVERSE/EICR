"""Regressions from the paid diagnostic: aborts, process identity and workflow state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from eicr.contracts import ContractError
from native_tag import experiment as ex, workflow, cli
from native_tag.runtime import write, tool_event_failed
from test_native import FakeEncoder, Script


@pytest.mark.parametrize('content',[
    "['A mix of excitement and terror.']", '["An error in a screenplay, not a failed tool."]',
    '{"text":"error","metadata":{"error":"quoted source"}}',
    '__import__("os").system("false")',
])
def test_tool_failure_detection_does_not_match_source_text(content):
    assert not tool_event_failed({'method':'call','result':[{'content':content}]})


@pytest.mark.parametrize('event',[
    {'method':'query_event_context','error':"'D26:t3'"},
    {'method':'query_event_context','error':''},
    {'method':'call','result':[{'content':"{'error': \"'D26:t3'\"}"}]},
    {'method':'call','result':[{'content':'{"error":null}'}]},
    {'method':'call','result':[{'content':'[]'},{'content':"{'error': 'unknown op'}"}]},
])
def test_tool_failure_detection_keeps_structured_errors(event):
    assert tool_event_failed(event)


def test_native_successful_source_with_terror_is_not_failure(protocol,monkeypatch):
    original=ex.install_trace
    def inject_source_text(agent,controller,trace):
        original(agent,controller,trace)
        trace.events.append({'method':'call','result':[{'role':'tool','content':"['Excitement and terror.']"}]})
    monkeypatch.setattr(ex,'install_trace',inject_source_text)
    row=ex.run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    assert not row['failed'] and row['tool_failures']==[]


def test_batch_abort_keeps_first_error_and_terminal_journal(protocol, tmp_path, monkeypatch):
    p=ex.sealed({**{k:v for k,v in protocol.items() if k!='hash'},
                 'jobs':[{**protocol['jobs'][0],'trial':i} for i in range(3)]})
    executions=[]
    def broken_run(*args, **kwargs):
        executions.append(True)
        raise ContractError('original scope mismatch')
    monkeypatch.setattr(ex,'run_one',broken_run)
    out=tmp_path/'aborted'
    result=ex.run_protocol(p,out)
    _,rows=ex.load_runs(out)
    assert len(executions)==1
    assert result['status']=='aborted' and result['abort_reason']=='original scope mismatch'
    assert rows[0]['error']['phase']=='setup'
    assert all(r['error']['phase']=='aborted' and not r['calls'] for r in rows[1:])
    assert {r['error']['message'] for r in rows[1:]}=={
        'Not executed after adaptation failure: original scope mismatch'}
    journal=json.loads((out/'journal.json').read_text())
    assert journal['status']=='aborted' and journal['completed']==result['runs']
    assert journal['abort_reason']=='original scope mismatch'


def test_successful_batch_has_terminal_journal(protocol,tmp_path):
    out=tmp_path/'complete'
    result=ex.run_protocol(protocol,out,transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    journal=json.loads((out/'journal.json').read_text())
    assert journal['status']=='complete' and journal['abort_reason'] is None
    assert journal['completed']==result['runs']


def test_run_cli_reports_adaptation_abort_as_failure(protocol,tmp_path,monkeypatch):
    path=tmp_path/'protocol.json';write(path,protocol)
    monkeypatch.setattr(cli,'run_protocol',lambda *a:{'status':'aborted','abort_reason':'scope mismatch'})
    assert cli.main(['run','--protocol',str(path),'--out',str(tmp_path/'runs')])==2


def test_cli_freeze_and_native_replay_are_stable_across_processes(protocol,tmp_path):
    root=Path(__file__).resolve().parents[2]
    env={**os.environ,'PYTHONPATH':os.pathsep.join((str(root/'src'),str(root/'tests/native_tag')))}
    env.pop('PYTHONHASHSEED',None)
    spec={k:deepcopy(protocol[k]) for k in ('runtime','model','embedding','questions','repeats','seed','prices')}
    spec.update(phase='development',assets=[r['path'] for r in protocol['assets'].values()])
    spec_path=tmp_path/'spec.json';write(spec_path,spec)
    protocols=[]
    for i in range(2):
        path=tmp_path/f'protocol-{i}.json'
        result=subprocess.run([sys.executable,'-m','native_tag.cli','freeze','--spec',str(spec_path),'--out',str(path)],
                              env=env,capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        protocols.append(json.loads(path.read_text()))
    assert protocols[0]==protocols[1]
    assert protocols[0]['environment']['python_hash_seed']=='0'
    command=[sys.executable,'-m','native_tag.cli','preflight','--protocol',str(tmp_path/'protocol-0.json'),
             '--out',str(tmp_path/'preflight.json')]
    assert subprocess.run(command,env=env,capture_output=True).returncode==0
    command[-1]=str(tmp_path/'wrong-seed.json')
    wrong=subprocess.run(command,env={**env,'PYTHONHASHSEED':'1'},capture_output=True,text=True)
    assert wrong.returncode==2 and 'Execution environment changed' in wrong.stderr
    assert not (tmp_path/'wrong-seed.json').exists()
    # Execute the actual frozen native graph reader twice in independent processes.
    script='''
import json,sys
from pathlib import Path
from native_tag.experiment import preflight,run_one
from test_native import FakeEncoder,Script,semantic
p=json.loads(Path(sys.argv[1]).read_text());preflight(p)
r=run_one(p,p['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
assert not r['failed']
Path(sys.argv[2]).write_text(json.dumps(semantic(r),sort_keys=True))
'''
    traces=[]
    for i in range(2):
        path=tmp_path/f'trace-{i}.json'
        result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'protocol-0.json'),str(path)],
                              env={**env,'PYTHONHASHSEED':'0'},capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        traces.append(json.loads(path.read_text()))
    assert traces[0]==traces[1]


def prepare_workflow(tmp_path,monkeypatch):
    (tmp_path/'development-runs').mkdir()
    write(tmp_path/'development-runs/manifest.json',{})
    write(tmp_path/'development-spec.json',{})
    monkeypatch.setattr(workflow,'load_runs',lambda _:({'prices':{}},[]))
    monkeypatch.setattr(workflow,'preflight',lambda _:None)
    monkeypatch.setattr(workflow,'stage_a',lambda *a,**kw:{'decision':'continue','tag_cost_fraction_all_runs':.5})
    monkeypatch.setattr(workflow,'calibrate',lambda *a:({'ready':True},[],{}))
    monkeypatch.setattr(workflow,'freeze_protocol',lambda spec:{'fixture':True})


@pytest.mark.parametrize('failure',['subprocess','adaptation','interrupt','shadow_check'])
def test_workflow_failure_stops_dependents_and_persists_status(tmp_path,monkeypatch,failure):
    prepare_workflow(tmp_path,monkeypatch)
    commands=[]
    def run(command,**kwargs):
        stage=command[1];commands.append(stage)
        if stage=='grade':return
        if stage=='run':
            out=tmp_path/'shadow-runs';out.mkdir()
            write(out/'paid-receipt.json',{'preserved':True})
            if failure=='subprocess':raise subprocess.CalledProcessError(2,command)
            if failure=='interrupt':raise KeyboardInterrupt()
            write(out/'manifest.json',{'status':'aborted' if failure=='adaptation' else 'complete',
                                       'abort_reason':'original graph mismatch'})
        if stage=='shadow-check':raise subprocess.CalledProcessError(2,command)
    monkeypatch.setattr(workflow.subprocess,'run',run)
    with pytest.raises((ContractError,subprocess.CalledProcessError,KeyboardInterrupt)):
        workflow.run_workflow(tmp_path,cli_command=['fixture'])
    state=json.loads((tmp_path/'workflow-status.json').read_text())
    assert state['stage']==('interrupted' if failure=='interrupt' else 'failed')
    assert state['failed_stage']==('shadow_checking' if failure=='shadow_check' else 'shadow_running')
    assert state['decision']=='inconclusive'
    assert commands==(['grade','run','shadow-check'] if failure=='shadow_check' else ['grade','run'])
    assert json.loads((tmp_path/'shadow-runs/paid-receipt.json').read_text())=={'preserved':True}
    old_status=(tmp_path/'workflow-status.json').read_bytes()
    with pytest.raises((ContractError,FileExistsError)):
        workflow.run_workflow(tmp_path,cli_command=['fixture'])
    assert (tmp_path/'workflow-status.json').read_bytes()==old_status
    assert not (tmp_path/'primary-protocol.json').exists()


def test_workflow_development_timeout_is_terminal(tmp_path):
    with pytest.raises(TimeoutError):workflow.run_workflow(tmp_path,wait_seconds=0)
    state=json.loads((tmp_path/'workflow-status.json').read_text())
    assert state['stage']=='failed' and state['failed_stage']=='waiting_for_development'


def test_workflow_failed_shadow_gate_never_freezes_primary(tmp_path,monkeypatch):
    prepare_workflow(tmp_path,monkeypatch)
    def run(command,**kwargs):
        if command[1]=='run':write(tmp_path/'shadow-runs/manifest.json',{'status':'complete'})
        if command[1]=='shadow-check':write(tmp_path/'shadow-acceptance.json',{'accepted':False})
    monkeypatch.setattr(workflow.subprocess,'run',run)
    state=workflow.run_workflow(tmp_path,cli_command=['fixture'])
    assert state['stage']=='stopped_at_shadow_gate'
    assert not (tmp_path/'primary-protocol.json').exists()
