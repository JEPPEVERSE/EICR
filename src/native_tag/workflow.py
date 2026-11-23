"""Sequential diagnostic gates with durable terminal state and no implicit resume."""
import json
from pathlib import Path
import subprocess
import sys
import time

from eicr.contracts import require
from .experiment import load_runs, freeze_protocol, calibrate, stage_a, preflight
from .runtime import write


def run_workflow(directory, *, cli_command=None, wait_seconds=3600):
    """Consume prepared inputs; an existing workflow must use a new directory.

    Credentials are inherited from the launching environment. A subprocess error,
    adaptation abort or interruption stops dependent stages and preserves all
    already-written receipts. Nothing here promotes offline replay to acceptance.
    """
    root=Path(directory).resolve()
    require(root.is_dir(), 'Prepared workflow directory does not exist')
    generated=('stage-a.json','calibration.json','calibration-records.json','calibration.cost-receipt.json',
               'development-grades','development-grades.judge-calls.json','shadow-protocol.json',
               'shadow-runs','shadow-acceptance.json','shadow-acceptance.cost-receipt.json',
               'primary-protocol.json','primary-runs','primary-grades','primary-grades.judge-calls.json',
               'offline-costs-spec.json','offline-costs.json','source-review','analysis.json','analysis.md')
    require(not any((root/name).exists() for name in generated),
            'Prior workflow outputs exist; preserve them and prepare a new directory')
    status_path=root/'workflow-status.json'
    # Exclusive creation prevents accidental reruns from resetting terminal state.
    with status_path.open('x') as handle:
        json.dump({'stage':'starting'},handle)
    state={'stage':'starting'}
    command=list(cli_command) if cli_command else [sys.executable,'-m','native_tag.cli']

    def read(name): return json.loads((root/name).read_text())
    def status(stage, **fields):
        state.clear();state.update(stage=stage,**fields);write(status_path,state)
    def cli(*args):
        state['command']=args[0];write(status_path,state)
        with (root/(args[0]+'-workflow.log')).open('a') as log:
            subprocess.run([*command,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    def run_phase(phase, protocol):
        cli('run','--protocol',root/protocol,'--out',root/(phase+'-runs'))
        manifest=read(phase+'-runs/manifest.json')
        require(manifest['status']=='complete',
                phase+' run aborted: '+str(manifest.get('abort_reason')))

    try:
        status('waiting_for_development')
        deadline=time.monotonic()+wait_seconds
        while not (root/'development-runs/manifest.json').exists():
            if time.monotonic()>=deadline:
                raise TimeoutError('Development collection did not finish before the workflow deadline')
            time.sleep(min(5,max(0,deadline-time.monotonic())))
        p,runs=load_runs(root/'development-runs')
        preflight(p)
        prices=p['prices']
        gate=stage_a(runs,prices,protocol=p);write(root/'stage-a.json',gate)
        cal,records,receipt=calibrate(p,runs)
        write(root/'calibration.json',cal);write(root/'calibration-records.json',records)
        write(root/'calibration.cost-receipt.json',receipt)
        status('development_grading',stage_a=gate['decision'],
               tag_cost_fraction=gate['tag_cost_fraction_all_runs'],calibration_ready=cal['ready'])
        cli('grade','--runs',root/'development-runs','--labels',root/'development-labels.json',
            '--judge',root/'model.json','--out',root/'development-grades')
        if gate['decision']!='continue' or not cal['ready']:
            status('stopped_at_development_gate',decision=gate['decision'],calibration_ready=cal['ready'])
            return dict(state)
        spec=read('development-spec.json');spec.update(calibration=cal,shadow=True,repeats=1)
        write(root/'shadow-protocol.json',freeze_protocol(spec))
        status('shadow_running');run_phase('shadow','shadow-protocol.json')
        status('shadow_checking')
        cli('shadow-check','--runs',root/'shadow-runs','--out',root/'shadow-acceptance.json')
        shadow=read('shadow-acceptance.json')
        if not shadow['accepted']:
            status('stopped_at_shadow_gate',decision='inconclusive');return dict(state)
        spec=read('development-spec.json')
        spec.update(phase='evaluation',questions=read('questions.json'),repeats=2,
                    calibration=cal,stage_a=gate,shadow_acceptance=shadow,checks=read('checks/checks.json'),
                    sample_manifest=read('sample-manifest.json'),rubric_acceptance=read('rubric-acceptance.json'),
                    diagnostic_questions=read('development-spec.json')['questions'])
        primary=freeze_protocol(spec);write(root/'primary-protocol.json',primary)
        status('primary_running',jobs=len(primary['jobs']),protocol_hash=primary['hash'])
        run_phase('primary','primary-protocol.json')
        status('primary_grading')
        cli('grade','--runs',root/'primary-runs','--labels',root/'labels.json',
            '--judge',root/'model.json','--out',root/'primary-grades')
        receipts=sorted(root.glob('*.embedding-receipt.json'))+[root/name for name in (
            'calibration.cost-receipt.json','shadow-acceptance.cost-receipt.json','rubric-acceptance.calls.json',
            'development-grades.judge-calls.json','primary-grades.judge-calls.json')]
        write(root/'offline-costs-spec.json',{'receipts':list(map(str,receipts)),'prices':prices,
            'rebuild_policy':{'trigger':'New snapshot or encoder/calibration configuration; full bundle projection',
                              'period':'month','expected_builds':None,'queries_per_period':None}})
        status('analyzing')
        cli('offline-costs','--spec',root/'offline-costs-spec.json','--out',root/'offline-costs.json')
        cli('source-export','--runs',root/'primary-runs','--out',root/'source-review')
        cli('analyze','--runs',root/'primary-runs','--grades',root/'primary-grades/grades.json',
            '--offline',root/'offline-costs.json','--out',root/'analysis.json')
        status('complete',decision=read('analysis.json')['decision'])
        return dict(state)
    except (Exception,KeyboardInterrupt) as exc:
        previous=dict(state)
        status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
               decision='inconclusive',failed_stage=previous['stage'],command=previous.get('command'),
               error={'type':type(exc).__name__,'message':str(exc)})
        raise
