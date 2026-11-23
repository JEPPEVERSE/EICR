"""File-based native-tag experiment workflow. No provider call in freeze/check/audit."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
from eicr.contracts import require,digest
from .core import fit_calibration
from .assets import prepare_assets
from .runtime import freeze_sources,verify_sources,load_runtime,build_memory,write,Trace,make_llm,live_transport,file_hash
from .experiment import (sealed,verify,freeze_protocol,preflight,run_protocol,load_runs,calibration_records,
                         stage_a,analyze,audit_history,select_questions,implementation_hashes,CHECKS,calibrate,shadow_acceptance)
from .grading import (grade_runs,rubric_manifest,check_rubric,apply_adjudication,
                      export_source_review,apply_source_review)
from .reporting import offline_costs, render_report


def read(path): return json.loads(Path(path).read_text())


def build_assets(spec, trace=None):
    runtime=verify_sources(spec['runtime'])
    fields=('raw_text','rewrites','keywords','event_embeddings','event_input_texts','topic_ids','topic_embeddings')
    body={'schema':'native-tag-assets-1','owner':spec['owner'],'snapshot':spec['snapshot'],
          'embedding_identity':spec['embedding_identity'],'provenance':spec['provenance'],
          'runtime_reference_hashes':runtime['reference'],'input_files':{}}
    for key in fields:
        path=Path(spec[key]).resolve();body[key]=read(path)
        body['input_files'][key]={'path':str(path),'sha256':file_hash(path)}
    asset=sealed(body)
    trace=trace if trace is not None else Trace()
    with trace.phase('offline_index_validation'):
        build_memory(load_runtime(spec['runtime'],{'provider':'offline','model':'audit','endpoint':'offline'}),asset)
    return asset


def acceptance(runtime, output):
    """Generate a receipt only from successful executable tests against this snapshot."""
    output=Path(output);require(not output.exists(),'New acceptance directory required')
    output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[2]
    test_dir=root/'tests'/'native_tag'
    require(test_dir.exists(),'Acceptance needs source distribution with tests')
    env={**os.environ,'NATIVE_TAG_TEST_RUNTIME':str(Path(runtime).resolve())}
    result=subprocess.run([sys.executable,'-m','pytest',str(test_dir),'-q','--junitxml='+str(output/'junit.xml')],
                          cwd=root,env=env,capture_output=True,text=True)
    (output/'pytest.log').write_text(result.stdout+result.stderr)
    require(result.returncode==0,'Acceptance failed; inspect pytest.log')
    cases=ET.parse(output/'junit.xml').findall('.//testcase')
    require(cases and all(c.find('skipped') is None and c.find('failure') is None and c.find('error') is None for c in cases),
            'Acceptance requires all tests executed and passing')
    names={c.attrib['name'].split('[')[0] for c in cases}
    evidence={
      'native_identity_replay':['test_complete_native_replay_and_passthrough',
                                'test_replay_message_preserves_sdk_exclude_none',
                                'test_shadow_preserves_native_unknown_response_tags'],
      'selector_passthrough_replay':['test_complete_native_replay_and_passthrough'],
      'native_tool_outputs':['test_complete_native_replay_and_passthrough','test_scopes_dedup_top_three','test_long_native_readers_and_temporal_bypass'],
      'scope_isolation':['test_scopes_dedup_top_three'],
      'score_boundaries':['test_threshold_and_tie_order','test_small_lists_do_not_trigger'],
      'query_identity':['test_fallback_identity_and_exact_cache'],
      'ledger_completeness':['test_complete_runner_and_calibration_receipts','test_retry_attempt_receipts',
                             'test_malformed_usage_keeps_attempts','test_postprocessing_failure_retains_attempt_ledger',
                             'test_batch_abort_keeps_first_error_and_terminal_journal',
                             'test_successful_batch_has_terminal_journal',
                             'test_tool_failure_detection_does_not_match_source_text',
                             'test_tool_failure_detection_keeps_structured_errors',
                             'test_native_successful_source_with_terror_is_not_failure',
                             'test_workflow_failure_stops_dependents_and_persists_status',
                             'test_run_cli_reports_adaptation_abort_as_failure',
                             'test_grade_failure_persists_attempts','test_reembedding_failure_persists_attempts'],
      'all_native_tools':['test_complete_native_replay_and_passthrough'],
      'independent_native_loader':['test_independent_native_module_loading_matches_runner'],
      'protocol_provenance':['test_stage_a_calibration_shadow_identity_gates','test_confirmation_and_diagnostic_freezes',
                             'test_analysis_rejects_unbound_judge_rubric_and_rows','test_native_never_returns_calibrated_teacher',
                             'test_cli_freeze_and_native_replay_are_stable_across_processes',
                             'test_workflow_failed_shadow_gate_never_freezes_primary'],
      'symmetric_adjudication':['test_incremental_review_is_globally_symmetric'],
      'offline_and_source_reports':['test_source_review_and_offline_reporting','test_offline_unknown_duplicate_and_rebuild_policy']}
    require(all(set(v)<=names for v in evidence.values()),'Acceptance evidence missing')
    report=sealed({'runtime_hash':verify_sources(runtime)['hash'],'implementation':implementation_hashes(),
                   'checks':{k:True for k in CHECKS},'evidence':evidence,'tests':len(cases),
                   'junit_sha256':file_hash(output/'junit.xml'),'scope':'offline fixed-external-response integration; not real-model quality'})
    write(output/'checks.json',report);return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    def cmd(name,flags):
        s=sub.add_parser(name)
        for flag in flags:s.add_argument('--'+flag,required=True)
        return s
    cmd('snapshot',['source','out']);cmd('assets',['spec','out']);cmd('freeze',['spec','out'])
    workflow_parser=sub.add_parser('workflow',help='Continue a prepared diagnostic through explicit gates')
    workflow_parser.add_argument('--directory',required=True)
    cmd('import-cache',['spec','out']);cmd('reembed',['spec','out'])
    cmd('preflight',['protocol','out']);cmd('run',['protocol','out']);cmd('check',['runtime','out'])
    cmd('stage-a',['runs','out']);cmd('calibrate',['runs','out']);cmd('sample',['spec','out'])
    cmd('history',['spec','out']);cmd('grade',['runs','labels','judge','out'])
    analysis_parser=cmd('analyze',['runs','grades','offline','out']);analysis_parser.add_argument('--source-review')
    cmd('rubrics',['out']);cmd('shadow-check',['runs','out']);cmd('offline-costs',['spec','out'])
    cmd('source-export',['runs','out']);cmd('source-review',['bundle','reviewed','reviewer','out'])
    cmd('rubric-check',['runtime','judge','cases','out'])
    cmd('adjudicate',['grades','mapping','reviewed','reviewer','out'])
    args=p.parse_args(argv)
    ledger=None; ledger_status='failed'; ledger_error=None
    try:
        name=args.command
        if name=='workflow':
            from .workflow import run_workflow
            result=run_workflow(args.directory);print(json.dumps(result));return 0
        if name=='snapshot':result=freeze_sources(args.source,args.out);print(json.dumps(result));return 0
        if name=='check':result=acceptance(args.runtime,args.out);print(json.dumps(result));return 0
        require(not Path(args.out).exists(),'Refusing to overwrite output')
        if name in ('import-cache','reembed'):
            result,receipt=prepare_assets(read(args.spec),reembed=name=='reembed',
                                          receipt_path=Path(args.out).with_suffix('.embedding-receipt.json'))
        elif name=='assets':
            ledger=Trace(Path(args.out).with_suffix('.embedding-receipt.json'),
                         {'components':['embedding','index'],'new_embedding':False,'common_native_graph_extraction_cost_excluded':True})
            result=build_assets(read(args.spec),ledger);ledger.metadata['asset_hash']=result['hash'];ledger_status='complete'
        elif name=='freeze':result=freeze_protocol(read(args.spec))
        elif name=='preflight':result=preflight(read(args.protocol))
        elif name=='run':
            result=run_protocol(read(args.protocol),args.out);print(json.dumps(result))
            return 0 if result['status']=='complete' else 2
        elif name=='stage-a':
            protocol,runs=load_runs(args.runs);result=stage_a(runs,protocol['prices'],protocol=protocol)
        elif name=='calibrate':
            protocol,runs=load_runs(args.runs)
            require(protocol['phase']=='development','Calibrate only the frozen development split')
            result,records,receipt=calibrate(protocol,runs)
            write(Path(args.out).with_suffix('.records.json'),records)
            write(Path(args.out).with_suffix('.cost-receipt.json'),receipt)
        elif name=='shadow-check':
            protocol,runs=load_runs(args.runs);result=shadow_acceptance(protocol,runs)
            write(Path(args.out).with_suffix('.cost-receipt.json'),sealed({
                'schema':'native-tag-ledger-2','ledger_id':'shadow:'+protocol['hash'],'components':['shadow'],
                'status':'complete','calibration_hash':protocol['calibration']['hash'],
                'calls':[c for r in runs for c in r['calls']],
                'wall_seconds':sum(r['wall_seconds'] for r in runs),'cpu_seconds':sum(r['cpu_seconds'] for r in runs)}))
        elif name=='offline-costs':
            spec=read(args.spec);result=offline_costs([read(p) for p in spec['receipts']],spec['prices'],spec['rebuild_policy'])
        elif name=='source-export':
            protocol,runs=load_runs(args.runs);export_source_review(protocol,runs,args.out);return 0
        elif name=='source-review':
            result=apply_source_review(read(args.bundle),read(args.reviewed)['items'],reviewer=args.reviewer)
        elif name=='history':result=audit_history(read(args.spec)['paths'])
        elif name=='sample':
            spec=read(args.spec);history=verify(spec['history'])
            evidence={(a['owner'],a['snapshot']):a['source_ids'] for a in spec['scopes']}
            questions,labels,manifest=select_questions(spec['pool'],set(history['question_ids'])|set(spec['development_ids']),
                  set(history['question_text_hashes'])|{digest(t.strip()) for t in spec['development_questions']},
                  owners=spec['owners'],salt=spec['salt'],evidence_by_scope=evidence,design=spec.get('design'))
            manifest=sealed({**{k:v for k,v in manifest.items() if k!='hash'},'history_hash':history['hash']})
            write(Path(args.out)/'questions.json',questions);write(Path(args.out)/'labels.json',labels)
            write(Path(args.out)/'sample_manifest.json',manifest);return 0
        elif name=='adjudicate':
            result=apply_adjudication(read(args.grades),read(args.mapping),read(args.reviewed)['items'],reviewer=args.reviewer)
        elif name=='rubric-check':
            model=read(args.judge);rt=load_runtime(args.runtime,model)
            trace=ledger=Trace(Path(args.out).with_suffix('.calls.json'),{'components':['rubric_check']})
            llm=make_llm(rt,live_transport(model),model,trace)
            def judge(system,user):
                with trace.phase('rubric_check'):
                    return llm.chat_text(messages=[{'role':'system','content':system},{'role':'user','content':user}],model=model['model'])
            result=check_rubric(read(args.cases),judge,model)
            ledger_status='complete'
        elif name=='grade':
            protocol,runs=load_runs(args.runs);model=read(args.judge)
            preflight(protocol)
            require(model==protocol.get('judge'), 'Use the frozen judge model')
            rt=load_runtime(protocol['runtime'],model)
            trace=ledger=Trace(Path(args.out).with_suffix('.judge-calls.json'),{'components':['judge'],'protocol_hash':protocol['hash']})
            llm=make_llm(rt,live_transport(model),model,trace)
            def judge(system,user):
                with trace.phase('judge'):
                    return llm.chat_text(messages=[{'role':'system','content':system},{'role':'user','content':user}],model=model['model'])
            labels=read(args.labels)
            sample=protocol.get('audit',{}).get('sample_manifest')
            if sample is not None: require(digest(labels)==sample['labels_hash'], 'Frozen grading labels changed')
            result=grade_runs(runs,labels,judge,model,args.out,protocol=protocol)
            ledger_status='complete';return 0
        elif name=='analyze':
            protocol,runs=load_runs(args.runs);grades=verify(read(args.grades))
            result=analyze(protocol,runs,grades,offline=read(args.offline),
                           source_review=read(args.source_review) if args.source_review else None)
            report_path=Path(args.out).with_suffix('.md');require(not report_path.exists(),'Refusing to overwrite report')
            report_path.parent.mkdir(parents=True,exist_ok=True);report_path.write_text(render_report(result))
        else:result=rubric_manifest()
        write(args.out,result);print(json.dumps({'command':name,'out':args.out,'status':result.get('decision',result.get('ready','written'))}));return 0
    except (ValueError,KeyError,OSError,subprocess.SubprocessError) as exc:
        ledger_error={'type':type(exc).__name__,'message':str(exc)}
        print(f'native-tag: {exc}',file=sys.stderr);return 2
    finally:
        if ledger is not None:
            active=sys.exc_info()[1]
            if active is not None: ledger_error={'type':type(active).__name__,'message':str(active)}
            ledger.flush(ledger_status,error=ledger_error)

def entrypoint():
    """Set the native set-order seed at interpreter startup, not after import."""
    if os.environ.get('PYTHONHASHSEED') in (None, '', 'random'):
        env={**os.environ,'PYTHONHASHSEED':'0'}
        os.execve(sys.executable,[sys.executable,'-m','native_tag.cli',*sys.argv[1:]],env)
    return main()


if __name__=='__main__':raise SystemExit(entrypoint())
