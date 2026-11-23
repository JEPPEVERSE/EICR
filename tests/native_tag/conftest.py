from pathlib import Path
import pytest
from native_tag.runtime import freeze_sources,write
from native_tag.experiment import sealed,freeze_protocol

ROOT=Path(__file__).resolve().parents[4]/'Reproductions'/'MRAgent'
MODEL={'provider':'offline','model':'fixture','endpoint':'offline://chat','api_key_env':'UNUSED'}
IDENTITY={'provider':'local','model':'fixture','revision':'frozen-test','dimension':3,
          'preprocessing':'replace_newlines_strip','normalization':'l2'}

@pytest.fixture(scope='session')
def frozen(tmp_path_factory):
    import os
    if os.environ.get('NATIVE_TAG_TEST_RUNTIME'): return Path(os.environ['NATIVE_TAG_TEST_RUNTIME'])
    if not ROOT.exists(): pytest.skip('Set up the local MRAgent checkout for native integration tests')
    path=tmp_path_factory.mktemp('native')/'runtime'
    freeze_sources(ROOT,path)
    return path

@pytest.fixture
def asset(frozen):
    import json
    manifest=json.loads((frozen/'runtime_manifest.json').read_text())
    raw, sentences, keys, vectors, texts = {}, [], [], {}, {}
    for i in range(1,37):
        source=f'D1:{i}'; eid=source+'-1'; key='Alice' if i<=18 else 'Bob'
        raw[source]=f'{key}: original event {i}.'
        text=f'Rewritten event {i}.'
        sentences.append({'id':eid,'origin':source,'text':text,'tag':f't{(i-1)%18}',
                          'time':'2026-01-01','topic':['t1']})
        keys.append({'sentence_id':eid,'keyword':[key]})
        vectors[eid]=[1.,float(i%3)/4,.1]; texts[eid]=text
    return sealed({'schema':'native-tag-assets-1','owner':'owner1','snapshot':'snap1',
        'raw_text':{'D1':raw},'rewrites':[{'session_1':{'conversation_time':'2026-01-02',
         'sentence':sentences,'topics':{'t1':'Daily activities'},'personal_sentences':[
             {'id':'p1','person':'Alice','text':'likes yoga','origin':'D1:1','tag':'hobby'}]}}],
        'keywords':[{'sentence':keys}],'event_embeddings':vectors,'event_input_texts':texts,
        'topic_ids':['D1:t1'],'topic_embeddings':[[1.,0.,0.]],'embedding_identity':IDENTITY,
        'provenance':{'identity_verified':True,'input_texts_verified':True,'note':'synthetic fixture'},
        'runtime_reference_hashes':manifest['reference']})

@pytest.fixture
def protocol(frozen,asset,tmp_path):
    path=tmp_path/'asset.json';write(path,asset)
    return freeze_protocol({'phase':'development','runtime':str(frozen),'assets':[str(path)],
        'questions':[{'question_id':'q1','owner':'owner1','snapshot':'snap1','category':1,'question':'What did Alice do?'}],
        'model':MODEL,'embedding':{'identity':IDENTITY},'repeats':1,'seed':11,'prices':None})


class AdaptiveScript:
    def __init__(self): self.rounds=0
    def __call__(self,request):
        import json
        from test_native import comp,tc
        if request.get('tools'):
            self.rounds+=1
            if self.rounds==1: return comp(tools=[tc('query_event_keywords',event_id='D1:19')])
            if self.rounds==2: return comp(tools=[tc('query_conversation_time',event_id='D1:19')])
            return comp({'mode':'answer','answer':'Yoga','supports':['D1:1-1']})
        try: payload=json.loads(request['messages'][-1]['content'])
        except (ValueError,TypeError): payload={}
        if isinstance(payload,dict) and 'tags' in payload:
            return comp({'tag_scores':{t:[0.,.3,.9][i%3] for i,t in enumerate(payload['tags'])}})
        return comp({'keywords':[{'id':'Alice','alternatives':[]}],'question_time':''})


@pytest.fixture
def primary_spec(frozen,asset,tmp_path):
    from copy import deepcopy
    from native_tag.experiment import (select_questions,run_one,stage_a,calibrate,shadow_acceptance,
                                      CHECKS,implementation_hashes)
    from native_tag.runtime import verify_sources
    from native_tag.grading import rubric_manifest
    from native_tag.reporting import offline_costs
    from test_native import FakeEncoder
    rates={'currency':'USD','effective_date':'2026-11-22','source':'SYNTHETIC TEST RATES','rates':[
        {'provider':'offline','model':'fixture','endpoint':'offline://chat','billing_kind':'chat_tokens',
         'input_per_million':1.,'output_per_million':1.,'cached_input_per_million':None}]}
    assets=[];pool=[];scopes={};owners=[f'o{i}' for i in range(4)];receipts=[]
    for owner in owners:
        a={k:deepcopy(v) for k,v in asset.items() if k!='hash'};a['owner']=owner;a=sealed(a)
        path=tmp_path/(owner+'.json');write(path,a);assets.append(str(path))
        receipts.append(sealed({'ledger_id':owner,'components':['embedding','index'],'status':'complete',
            'asset_hash':a['hash'],'calls':[],'wall_seconds':0.,'cpu_seconds':0.,'new_embedding':False}))
        scopes[owner,asset['snapshot']]=['D1:1']
        for c in (1,2,3,4):
            for i in range(2):pool.append({'owner':owner,'snapshot':asset['snapshot'],'question_id':f'{owner}-{c}-{i}',
                'question':f'What did Alice do in case {owner}/{c}/{i}?','category':c,'reference':'Yoga','evidence':['D1:1']})
    questions,labels,sample=select_questions(pool,set(),set(),owners=owners,salt='fixture',evidence_by_scope=scopes)
    dev={**questions[0],'question_id':'dev','question':'known development question'}
    base={'runtime':str(frozen),'assets':assets,'model':MODEL,'embedding':{'identity':IDENTITY},'seed':3,'prices':rates}
    dp=freeze_protocol({**base,'phase':'development','questions':[dev],'repeats':20})
    runs=[run_one(dp,j,transport_factory=lambda _:AdaptiveScript(),encoder_factory=FakeEncoder) for j in dp['jobs']]
    gate=stage_a(runs,rates,protocol=dp)
    cal,_,receipt=calibrate(dp,runs);receipts.append(receipt)
    sp=freeze_protocol({**base,'phase':'development','questions':[dev],'repeats':1,'calibration':cal,'shadow':True})
    sr=[run_one(sp,j,transport_factory=lambda _:AdaptiveScript(),encoder_factory=FakeEncoder) for j in sp['jobs']]
    shadow=shadow_acceptance(sp,sr)
    receipts.append(sealed({'ledger_id':'shadow:'+sp['hash'],'components':['shadow'],'status':'complete',
        'calibration_hash':cal['hash'],'calls':[c for r in sr for c in r['calls']],
        'wall_seconds':sum(r['wall_seconds'] for r in sr),'cpu_seconds':sum(r['cpu_seconds'] for r in sr)}))
    spec={**base,'phase':'evaluation','questions':questions,'sample_manifest':sample,'repeats':2,
        'calibration':cal,'stage_a':gate,'shadow_acceptance':shadow,
        'checks':sealed({'runtime_hash':verify_sources(frozen)['hash'],'implementation':implementation_hashes(),
            'checks':{k:True for k in CHECKS}}),'judge':MODEL,
        'rubric_acceptance':sealed({'accepted':True,'model':MODEL,'rubric_hash':rubric_manifest()['hash']}),
        'diagnostic_questions':[dev]}
    costs=offline_costs(receipts,rates,{'trigger':'fixture snapshot replacement','period':'month',
                                      'expected_builds':1,'queries_per_period':1000})
    return spec,labels,costs
