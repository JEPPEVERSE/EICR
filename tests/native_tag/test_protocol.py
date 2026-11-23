from copy import deepcopy
import json
import pytest
from eicr.contracts import ContractError,digest
from native_tag.experiment import (sealed,select_questions,audit_history,freeze_protocol,analyze,run_one,stage_a)
from native_tag.grading import grade_runs,apply_adjudication,check_rubric
from native_tag.runtime import write
from test_native import Script,FakeEncoder


def test_sampling_exposure_and_missing_strata(tmp_path):
    pool=[];scopes={}
    for owner in ('o1','o2','o3','o4'):
        scopes[owner,'s']=['D1:1']
        for c in range(1,5):
            for i in range(3):pool.append({'question_id':f'{owner}-{c}-{i}','owner':owner,'snapshot':'s','category':c,
                  'question':f'{owner} question {c}/{i}','reference':'answer','evidence':['D1:1']})
    q,l,m=select_questions(pool,{'o1-1-0'},set(),owners=['o1','o2','o3','o4'],salt='v1',evidence_by_scope=scopes)
    assert len(q)==len(l)==32 and all('reference' not in x for x in q)
    assert all(x['question_id']!='o1-1-0' for x in q)
    assert select_questions(list(reversed(pool)),{'o1-1-0'},set(),owners=['o1','o2','o3','o4'],salt='v1',evidence_by_scope=scopes)[0]==q
    with pytest.raises(ContractError):select_questions(pool,{'o1-1-0','o1-1-1'},set(),owners=['o1','o2','o3','o4'],salt='v1',evidence_by_scope=scopes)
    path=tmp_path/'history.json';write(path,{'runs':[{'question_id':'old','question':'old q'}]})
    history=audit_history([path]);assert history['question_ids']==['old'] and history['question_text_hashes']==[digest('old q')]


def test_grading_blind_cache_and_symmetric_review(protocol,tmp_path):
    row=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    rows=[row,{**row,'hash':'other','arm':'cheap_all'}]
    labels=[{**row['question'],'reference':'Yoga, strength training','review_flags':['known partial']}]
    calls=[]
    def judge(system,user):
        calls.append((system,user));assert 'cheap_all' not in user and 'native' not in user
        return {'correct':False,'reason':'missing strength training','partial':True,'temporal_ambiguity':False,'gold_concern':False} if 'Return JSON:' in system else {'label':'CORRECT'}
    artifact=grade_runs(rows,labels,judge,{'model':'fixture'},tmp_path/'grades')
    assert len(calls)==2 and all(g['unresolved'] for g in artifact['grades'] if g['rubric']=='strict')
    public=json.loads((tmp_path/'grades/adjudication/blind.json').read_text())
    mapping=json.loads((tmp_path/'grades/adjudication/private-mapping.json').read_text())
    reviewed=[{**r,'score':0,'reason':'incomplete'} for r in public['items']]
    updated=apply_adjudication(artifact,mapping,reviewed,reviewer='independent fixture reviewer')
    assert not any(g['unresolved'] for g in updated['grades'])
    reviewed[1]['score']=1
    with pytest.raises(ContractError):apply_adjudication(artifact,mapping,reviewed,reviewer='reviewer')


def synthetic_analysis(protocol):
    from native_tag.grading import rubric_manifest
    p={k:deepcopy(v) for k,v in protocol.items() if k!='hash'}
    p['judge']=p['model'];p['rubric_acceptance']={'rubric_hash':rubric_manifest()['hash']}
    p['arms']=['native','cheap_all'];p['repeats']=2;p['questions']=[];p['jobs']=[]
    for i in range(4):
        q={**protocol['questions'][0],'question_id':str(i),'owner':f'o{i}'};p['questions'].append(q)
        for t in range(2):
            for a in p['arms']:p['jobs'].append({'question':q,'trial':t,'arm':a})
    p=sealed(p);runs=[]
    for job in p['jobs']:
        cost=1. if job['arm']=='native' else .8
        runs.append(sealed({**job,'protocol_hash':p['hash'],'billing':{'amount':cost,'complete':True,'calls':[{'amount':cost}]},
            'answer':'same answer','failed':False,'wall_seconds':1.,'calls':[{'stage':'tool_loop_and_answer'}],
            'decisions':[{'fallback':False,'stage':'initial','ordered_tag_ids':[],'mode':job['arm']}]}))
    return p,runs,synthetic_grades(p,runs)


def synthetic_grades(p,runs,unresolved=False):
    from native_tag.grading import rubric_manifest
    labels=[{**q,'reference':'same answer'} for q in p['questions']];grades=[]
    for r in runs:
        for rubric in ('strict','lenient'):
            payload={'question':r['question']['question'],'reference':'same answer','answer':r['answer']}
            grades.append({'run_hash':r['hash'],'rubric':rubric,'score':1,'unresolved':unresolved,
                'cache_key':digest([rubric_manifest()['hash'],p['judge'],rubric,payload])})
    return sealed({'protocol_hash':p['hash'],'model':p['judge'],'rubrics':rubric_manifest(),
        'run_hashes':sorted(r['hash'] for r in runs),'labels':labels,'labels_hash':digest(labels),'grades':grades})


def test_analysis_failures_unknowns_and_symmetric_bounds(protocol):
    p,runs,grades=synthetic_analysis(protocol)
    result=analyze(p,runs,grades,offline_cost=1.)
    assert result['decision']=='continue' and result['break_even_queries']==5
    grades=synthetic_grades(p,runs,True)
    result=analyze(p,runs,grades)
    assert result['decision']=='continue' and result['symmetric_quality_delta_bounds']==[0.,0.]
    r={k:v for k,v in runs[0].items() if k!='hash'};r['answer']='different answer';runs[0]=sealed(r)
    grades=synthetic_grades(p,runs,True)
    assert analyze(p,runs,grades)['decision']=='inconclusive'
    with pytest.raises(ContractError):analyze(p,runs[:-1],grades)
    r={k:v for k,v in runs[0].items() if k!='hash'};r['billing']={**r['billing'],'amount':None,'complete':False};runs[0]=sealed(r)
    grades=synthetic_grades(p,runs,True)
    result=analyze(p,runs,grades)
    assert result['aggregate']['native']['cost'] is None and result['decision']=='inconclusive'


def test_primary_cannot_run_without_gates(protocol):
    # Any missing Stage A/B or malformed primary matrix is rejected before payment.
    spec={'phase':'evaluation','runtime':protocol['runtime'],'questions':protocol['questions'],'assets':[],
          'repeats':2,'model':protocol['model'],'embedding':protocol['embedding'],'seed':0}
    with pytest.raises(ContractError):freeze_protocol(spec)


def test_original_question_embedding_separate_from_scoring_context(protocol):
    row=run_one(protocol,protocol['jobs'][0],transport_factory=lambda _:Script(),encoder_factory=FakeEncoder)
    embeddings=[c for c in row['calls'] if c['billing_kind']=='local_embedding']
    assert embeddings[0]['request']['input']==protocol['questions'][0]['question']
    assert embeddings[1]['request']['input']==row['formatted_question']
    assert row['decisions'][0]['query_reuse']=='none'


def test_native_cache_import_and_reembedding(frozen,asset,tmp_path):
    import pickle
    import numpy as np
    from native_tag.assets import prepare_assets
    paths={}
    for k in ('raw_text','rewrites','keywords'):
        p=tmp_path/(k+'.json');write(p,asset[k]);paths[k]=str(p)
    cache={'sentence_id':list(asset['event_embeddings']),'embeddings':np.asarray(list(asset['event_embeddings'].values())),
           'topic_list':asset['topic_ids'],'topic':np.asarray(asset['topic_embeddings'])}
    path=tmp_path/'cache.pkl';path.write_bytes(pickle.dumps(cache))
    spec={'runtime':str(frozen),'owner':'owner1','snapshot':'s',**paths,'embedding_cache':str(path),
          'embedding_identity':asset['embedding_identity'],'provenance':{'identity_verified':True,'input_texts_verified':True,'identity_evidence':'synthetic known encoder'},
          'embedding':{'identity':asset['embedding_identity']}}
    imported,receipt=prepare_assets(spec)
    assert imported['event_input_texts']==asset['event_input_texts'] and receipt['calls']==[]
    rebuilt,receipt=prepare_assets(spec,reembed=True,encoder_factory=FakeEncoder)
    assert len(receipt['calls'])==37 and receipt['new_embedding']
    assert rebuilt['keywords']==asset['keywords'] # native speaker injection must not mutate caches


def test_setup_failure_is_retained(protocol,tmp_path):
    from native_tag.experiment import run_protocol,load_runs
    def unavailable(_):raise OSError('provider initialization failed')
    out=tmp_path/'failure'
    run_protocol(protocol,out,transport_factory=unavailable,encoder_factory=FakeEncoder)
    _,rows=load_runs(out)
    assert len(rows)==1 and rows[0]['failed'] and rows[0]['error']['phase']=='setup'


def test_full_128_independent_trials(primary_spec,tmp_path):
    from conftest import AdaptiveScript
    from native_tag.experiment import run_protocol,load_runs
    spec,labels,costs=primary_spec
    protocol=freeze_protocol(spec);assert len(protocol['jobs'])==128
    instances=[]
    def factory(_):
        script=AdaptiveScript();instances.append(script);return script
    run_protocol(protocol,tmp_path/'matrix',transport_factory=factory,encoder_factory=FakeEncoder)
    _,runs=load_runs(tmp_path/'matrix')
    assert len(instances)==128 and all(s.rounds==3 for s in instances)
    assert not any(r['failed'] for r in runs)
    assert all(r['events'][0]['before']=={'events':[],'keywords':[]} for r in runs)
    assert all(not any(c['billing_kind']=='chat_tokens' and c['stage'] in ('initial_tag','expansion_tag')
                      for c in r['calls']) for r in runs if r['arm']=='cheap_all')
    def judge(system,user):
        return {'correct':True,'reason':'fixture','partial':False,'temporal_ambiguity':False,'gold_concern':False} if 'Return JSON:' in system else {'label':'CORRECT'}
    grades=grade_runs(runs,labels,judge,protocol['judge'],tmp_path/'grades',protocol=protocol)
    report=analyze(protocol,runs,grades,offline=costs)
    assert report['decision']=='continue' and report['cost_reduction']>0.15
    assert report['break_even_queries']>0 and report['mechanisms']['cheap_all']['stages']['initial']['replaced_calls']==64
    from native_tag.cli import main
    write(tmp_path/'offline.json',costs)
    assert main(['analyze','--runs',str(tmp_path/'matrix'),'--grades',str(tmp_path/'grades/grades.json'),
                 '--offline',str(tmp_path/'offline.json'),'--out',str(tmp_path/'analysis.json')])==0
    assert (tmp_path/'analysis.md').exists()
