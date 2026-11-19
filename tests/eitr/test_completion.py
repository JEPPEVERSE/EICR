from copy import deepcopy
from dataclasses import replace
import json
import pytest
from eicr.contracts import ContractError, digest, write_json
from eicr.packer import CharacterTokenizer
from eitr.config import Config
from eitr.native import canonicalize, import_pack
from eitr.index import build_index, save_index
from eitr.query import map_groups
from eitr.evidence import Corpus
from eitr.pipeline import retrieve, compare
from eitr.experiment import replay, freeze, verify_protocol, run_protocol, benchmark
from eitr.accounting import cost_ledger, cost_report, break_even
from eitr.evaluate import diagnose, grade_batch, summarize, paired_intervals
from eitr.batch import run_batch
from eitr.dates import attest_dates
from eitr.cli import main


def test_verified_alias_is_one_vote_and_one_coverage_group(raw_memory):
    cue=next(c for c in raw_memory['cues'] if c['text']=='herbs')
    cue['aliases']=[{'text':'mixed herbs','source_ids':['D1:1']}]
    index=build_index(canonicalize(raw_memory))
    q=map_groups(index,'herbs and mixed herbs and train',Config(),[
        {'surface_text':'herbs','span_start':0,'span_end':5},{'surface_text':'mixed herbs'},{'surface_text':'train'}])
    assert len(q['groups'])==2
    assert [g['support_weight'] for g in q['groups']]==[.5,.5]
    herb=next(g for g in q['groups'] if g['key_ids']==['herbs'])
    assert set(herb['surfaces'])=={'herbs','mixed herbs'}
    row=retrieve(index,'herbs and mixed herbs and train',CharacterTokenizer(),query=q)
    assert len(row['evidence_pack']['coverage'])==2


def test_fuzzy_collisions_do_not_merge(snapshot):
    q=map_groups(build_index(snapshot),'green herbs and dried herbs',Config(),[
        {'surface_text':'green herbs'},{'surface_text':'dried herbs'}])
    assert len(q['groups'])==2
    assert all(g['match_method']=='lexical_fuzzy' for g in q['groups'])


def test_corpus_cache_equivalence_and_stale_rejection(snapshot,tmp_path,monkeypatch):
    config=Config();tok=CharacterTokenizer();corpus=Corpus(snapshot,tok,config);path=tmp_path/'corpus.json';corpus.save(path)
    import eitr.evidence
    monkeypatch.setattr(eitr.evidence,'chunk_sources',lambda *a,**k:pytest.fail('Cache load must not recut sources'))
    loaded=Corpus.load(path,snapshot,tok,config)
    assert loaded.scores('herbs pinch')==corpus.scores('herbs pinch')
    assert loaded.render(set(loaded.atoms))==corpus.render(set(corpus.atoms))
    with pytest.raises(ContractError):Corpus.load(path,snapshot,tok,replace(config,atom_tokens=64))
    data=json.loads(path.read_text());data['term_counts'][0]['bad']=1;write_json(path,data)
    with pytest.raises(ContractError):Corpus.load(path,snapshot,tok,config)


def test_replay_recomputes_identical_decisions_and_detects_changed_scores(snapshot):
    config=Config(evidence_budget=650);index=build_index(snapshot);tok=CharacterTokenizer();corpus=Corpus(snapshot,tok,config)
    row=retrieve(index,'herbs and halloumi',tok,config=config,corpus=corpus)
    assert row['evidence_pack']['alternatives']
    assert replay(row,index,corpus)['verified']
    bad=deepcopy(row);bad['evidence_pack']['text_hash']='wrong'
    with pytest.raises(ContractError):replay(bad,index,corpus)


def test_full_truncation_records(snapshot):
    cfg=Config(max_core_atoms_per_tag=1,max_core_atoms=1,max_candidate_atoms=1,direct_core_reservation=1)
    row=retrieve(build_index(snapshot),'herbs',CharacterTokenizer(),config=cfg)
    assert row['candidates']['omitted']
    assert 'before_global_cap' in row['candidates']['source_stages']
    assert 'frontier' in row['graph'] and 'rejected_tags' in row['graph']


def test_budget_loss_is_distinguished_from_candidate_loss(snapshot):
    idx=build_index(snapshot)
    row=retrieve(idx,'halloumi herbs',CharacterTokenizer(),config=Config(evidence_budget=1))
    report=diagnose(row,['D1:2'],idx)
    assert report['losses']['D1:2']=='packet_budget_or_selection'
    assert report['packet_source_recall']==0
    with pytest.raises(ContractError):diagnose(row,['future'],idx)


def test_unknown_usage_is_never_free_and_reuse_is_charged():
    row={'calls':{'frontend':[{'status':'error'}],'answer':[{'status':'ok','usage':{'prompt_tokens':5,'completion_tokens':3}}]},'answer_reused_in_experiment':True}
    ledger=cost_ledger(row)
    assert ledger['independent_deployment']['total_tokens'] is None
    assert ledger['independent_deployment']['failed_calls']==1
    del row['calls']['frontend']
    assert cost_ledger(row)['independent_deployment']['total_tokens']==8
    assert break_even(10,2,1)['break_even_queries']==10
    assert break_even(10,1,2)['break_even_queries'] is None


def test_independent_latency_only(snapshot):
    idx=build_index(snapshot);tok=CharacterTokenizer();cfg=Config();corpus=Corpus(snapshot,tok,cfg)
    q={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'herbs','mentions':[]}
    cached=compare(idx,'herbs',tok)['rows']
    assert cost_report(cached)['arms']['P4']['end_to_end_p50_seconds'] is None
    rows=benchmark(idx,[q],tok,config=cfg,arms=['P0','P4'],corpus=corpus,repeats=2)
    assert len(rows)==4
    assert cost_report(rows)['arms']['P4']['independent_latency_runs']==2
    assert all(not r['benchmark']['has_answer_call'] for r in rows)


def test_frozen_protocol_executes_and_detects_asset_mutation(snapshot,tmp_path):
    idx=build_index(snapshot);ip=tmp_path/'index.json';save_index(idx,ip)
    cp=tmp_path/'corpus.json';Corpus(snapshot,CharacterTokenizer(),Config()).save(cp)
    qp=tmp_path/'questions.json';write_json(qp,[{'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'herbs','mentions':[]}])
    spec={'config':Config().to_dict(),'arms':['P0','P3','P4'],'tokenizer':{'kind':'characters'},
          'assets':[{'index':str(ip),'corpus':str(cp)}],'questions':str(qp),'native':None,
          'model':{'api_base':'https://example.invalid','model':'test','key_env':'TEST_UNUSED','max_output_tokens':20},
          'frontend':'supplied','generate':False,'require_dates':False,'prices':None}
    protocol=freeze(spec)
    assert verify_protocol(protocol)==protocol
    assert run_protocol(protocol,tmp_path/'run')['status']=='complete'
    questions=json.loads(qp.read_text());questions[0]['question']='changed';write_json(qp,questions)
    with pytest.raises(ContractError):verify_protocol(protocol)
    questions[0]['gold']='secret';write_json(qp,questions)
    with pytest.raises(ContractError):freeze(spec)


def test_dates_require_exact_original_text(raw_memory):
    raw_memory['sources']=raw_memory['sources'][:1];raw_memory['evidence']=raw_memory['evidence'][:1]
    raw_memory['edges']=[e for e in raw_memory['edges'] if e['evidence_id']=='f1']
    raw_memory['provenance']['allowed_source_ids']=['D1:1']
    raw_memory['sources'][0]['text']='A: original';raw_memory['sources'][0]['session_datetime']=None
    snap=canonicalize(raw_memory)
    source={'owner_id':snap['owner_id'],'conversation':{'session_1_date_time':'2024-09-15','session_1':[{'dia_id':'D1:1','speaker':'A','text':'original'}]}}
    updated=attest_dates(snap,source)
    assert updated['sources'][0]['session_datetime']=='2024-09-15'
    source['conversation']['session_1'][0]['text']='different'
    with pytest.raises(ContractError):attest_dates(snap,source)


def test_grade_is_blinded_and_outputs_diagnostics(snapshot,tmp_path):
    class Answer:
        def __init__(self):self.calls=[]
        def chat_json(self,system,user):
            self.calls.append({'status':'ok','usage':{'prompt_tokens':5,'completion_tokens':2}})
            return {'answer_text':'herbs','citations':[],'unsupported_premises':[],'resolution_status':'insufficient'}
    class Judge(Answer):
        def chat_json(self,system,user):
            parsed=json.loads(user);assert set(parsed)=={'question','references','candidate'}
            assert parsed['candidate']=='herbs'
            self.calls.append({'status':'ok'})
            return {'correct':True,'reason':'Matches reference'}
    idx=build_index(snapshot)
    q={'question_id':'q','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id'],'question':'Which herbs?'}
    run_batch(idx,[q],CharacterTokenizer(),config=Config(),arms=['P3','P4'],answer_backend=Answer(),output_dir=tmp_path/'batch')
    labels=[{k:q[k] for k in ('question_id','owner_id','snapshot_id')}|{'references':['herbs'],'source_ids':['D1:2']}]
    judge=Judge();rows=grade_batch(tmp_path/'batch',labels,idx,judge,tmp_path/'grade')
    assert len(judge.calls)==1
    assert summarize(rows)['pairs']['P3 vs P4']['paired_questions']==1
    assert paired_intervals(rows,100)['pairs']['P3 vs P4']['percentile_95'] is None


def test_prepare_cache_cli(snapshot,tmp_path):
    ip=tmp_path/'i.json';save_index(build_index(snapshot),ip)
    assert main(['prepare-corpus','--index',str(ip),'--tokenizer','characters','--out',str(tmp_path/'c.json')])==0


def test_cross_process_hash_seed_does_not_change_choices(snapshot,tmp_path):
    import os,sys,subprocess
    from pathlib import Path
    path=tmp_path/'snapshot.json';write_json(path,snapshot)
    script="""import json,sys
from eitr.index import build_index
from eitr.pipeline import retrieve
from eicr.packer import CharacterTokenizer
from eitr.config import Config
r=retrieve(build_index(json.load(open(sys.argv[1]))),'herbs halloumi train',CharacterTokenizer(),config=Config(evidence_budget=700))
print(json.dumps({'packet':r['evidence_pack']['text_hash'],'ledger':r['evidence_pack']['ledger']},sort_keys=True))
"""
    outputs=[]
    for seed in ('1','27','123'):
        env={**os.environ,'PYTHONHASHSEED':seed,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')}
        outputs.append(subprocess.check_output([sys.executable,'-c',script,str(path)],env=env,text=True))
    assert len(set(outputs))==1


def test_no_semantic_dedup_of_different_original_experiences(raw_memory):
    raw_memory['sources'][2]['text']=raw_memory['sources'][0]['text']
    index=build_index(canonicalize(raw_memory))
    row=retrieve(index,'paprika shop',CharacterTokenizer(),config=Config(evidence_budget=3000,context_each_side=0))
    assert {'D1:1','D1:10'}<={s['source_id'] for s in row['evidence_pack']['spans']}


def test_isolated_seed_survives_without_direct_relevance(snapshot):
    index=build_index(snapshot)
    row=retrieve(index,'train',CharacterTokenizer(),config=Config(evidence_budget=1000,direct_fraction=0,seed_fraction=1),arm='P4')
    assert any(s['source_id']=='D2:1' for s in row['evidence_pack']['spans'])
    assert any(step['phase']=='seed' for step in row['evidence_pack']['ledger'])
