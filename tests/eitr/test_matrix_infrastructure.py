"""Exact graph, independent measurement, service pricing and lifecycle contracts."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from eicr.contracts import digest, read_json, write_json, ContractError
from eicr.packer import CharacterTokenizer
from eicr.provider import ProviderTransportError
from eitr.evidence import Corpus
from eitr.matrix_index import build_index, embed_snapshot, relation, compare_graphs, MatrixIndex
from eitr.matrix_embeddings import EmbeddingBackend
from eitr.matrix_benchmark import (freeze_benchmark, verify_benchmark, independent_request,
                                   run_benchmark, load_benchmark_runs, benchmark_report)
from eitr.matrix_reports import build_accounting, lifecycle_report, summarize_repeats
from eitr.pricing import price_calls, validate_prices
from eitr.accounting import cost_ledger, cost_report
from eitr.matrix_pipeline import replay
from eitr.matrix_retrieval import MatrixConfig
from eitr.cli import main
from test_matrix import Embeddings, Reranker, config, embedding_spec


def prices_for(*backends):
    rates=[]
    for kind,model,endpoint in backends:
        rates.append({'billing_kind':kind,'model':model,'endpoint_hash':endpoint,
                      'input_per_million':2.,'output_per_million':6.,'cached_input_per_million':None})
    return {'schema':'eitr-prices-1','currency':'USD','as_of':'test-only','source':'synthetic test tariff; not market pricing','rates':rates}


def call(kind='chat_tokens',model='chat',endpoint=None,i=100,o=20):
    return {'billing_kind':kind,'model':model,'endpoint_hash':endpoint or digest('endpoint'),'status':'ok',
            'usage':{'input_tokens':i,'output_tokens':o},'seconds':.01}


@pytest.fixture
def matrix(snapshot):
    b=Embeddings();idx=build_index(snapshot,embed_snapshot(snapshot,b),graph_top_k=2,graph_mode='exact')
    cfg=config();return idx,b,cfg,Corpus(snapshot,CharacterTokenizer(),cfg)


def test_exact_graph_matches_all_pairs_dense_reference(snapshot):
    b=Embeddings();record=embed_snapshot(snapshot,b)
    rng=np.random.default_rng(73);x=rng.normal(size=(4,4));x/=np.linalg.norm(x,axis=1)[:,None]
    record['vectors']=x.tolist();record['record_hash']=digest({k:v for k,v in record.items() if k!='record_hash'})
    idx=build_index(snapshot,record,graph_top_k=2,graph_shortlist=0,graph_mode='exact',block=1)
    for tag in idx.tag_ids:
        expected=[]
        for other in idx.tag_ids:
            if tag==other:continue
            ti,ui=idx.members[tag],idx.members[other]
            edge=relation(idx.X[ti],idx.X[ui],idx.weights[tag],idx.weights[other],128)
            expected.append((other,edge))
        expected.sort(key=lambda pair:(-pair[1]['coverage_forward'],-pair[1]['bridge'],pair[0]))
        assert [e['target'] for e in idx.graph[tag]]==[t for t,e in expected[:2]]
        for actual,(_,e) in zip(idx.graph[tag],expected):
            assert actual['average']==pytest.approx(e['average'])
    approx=build_index(snapshot,record,graph_top_k=2,graph_shortlist=3,graph_mode='shortlist')
    report=compare_graphs(idx,approx);assert report['positive_edge_recall']==1
    lost=deepcopy(approx.graph)
    for t in lost:lost[t]=[]
    empty=MatrixIndex(snapshot,record,lost,approx.graph_config)
    assert compare_graphs(idx,empty)['positive_edge_recall']==0
    with pytest.raises(ContractError):compare_graphs(approx,idx)


def test_exact_no_edges_and_graph_cli_build_metrics(snapshot,tmp_path):
    b=Embeddings();record=embed_snapshot(snapshot,b)
    sp=tmp_path/'snapshot.json';ep=tmp_path/'embedding.json';ip=tmp_path/'index.json'
    write_json(sp,snapshot);write_json(ep,record)
    assert main(['matrix-build','--snapshot',str(sp),'--embedding-record',str(ep),'--graph-mode','exact',
                 '--graph-top-k','3','--graph-shortlist','0','--out',str(ip)])==0
    metrics=read_json(str(ip)+'.metrics.json')
    assert metrics['embedding_reused'] and metrics['actual_embedding_calls']==[]
    assert metrics['graph_config']['mode']=='exact' and metrics['graph_edges']==12
    assert metrics['metrics_hash']==digest({k:v for k,v in metrics.items() if k!='metrics_hash'})
    assert build_accounting(metrics,None,0)['seconds_eligible_for_lifecycle'] is None
    assert main(['matrix-compare-graphs','--exact-index',str(ip),'--approximate-index',str(ip),
                 '--out',str(tmp_path/'comparison.json')])==0


def test_multi_service_pricing_and_missing_identity():
    h=digest('endpoint');p=prices_for(('chat_tokens','chat',h),('embedding_tokens','embed',h),('rerank_tokens','rank',h))
    calls=[call(),call('embedding_tokens','embed',i=200,o=0),call('rerank_tokens','rank',i=300,o=0)]
    bill=price_calls(calls,p)
    assert bill['usd']==pytest.approx((100*2+20*6+200*2+300*2)/1e6)
    ledger=cost_ledger({'calls':{'frontend':[calls[0]],'embedding':[calls[1]],'rerank':[calls[2]]}},p)
    assert ledger['independent_deployment']['usd']==bill['usd']
    assert not ledger['unpriced_rerank'] and not ledger['unpriced_embeddings']
    wrong=deepcopy(calls);wrong[1]['endpoint_hash']=digest('other-vendor')
    partial=price_calls(wrong,p);assert partial['usd'] is None and partial['known_usd']>0
    calls[2]['usage']={};assert price_calls(calls,p)['usd'] is None
    assert price_calls([],p)['usd']==0
    assert price_calls([call('rerank_tokens','rank')],{'input_per_million':1,'output_per_million':1})['usd'] is None


def test_provider_cached_tokens_discount_is_explicit():
    h=digest('endpoint');p=prices_for(('chat_tokens','chat',h));p['rates'][0]['cached_input_per_million']=.5
    c=call();assert price_calls([c],p)['usd'] is None
    c['usage']['cached_input_tokens']=60
    assert price_calls([c],p)['usd']==pytest.approx((40*2+60*.5+20*6)/1e6)
    c['usage']['cached_input_tokens']=101;assert price_calls([c],p)['usd'] is None
    p['rates'].append(deepcopy(p['rates'][0]))
    with pytest.raises(ContractError):validate_prices(p)


class Answer:
    model='test-answer'
    def __init__(self):self.calls=[]
    def chat_json(self,system,user):
        self.calls.append(call(model=self.model))
        return {'answer_text':'Cannot establish from these sources','citations':[],'unsupported_premises':[],
                'resolution_status':'insufficient'}


def test_independent_pipeline_has_fresh_calls_and_replay(matrix,tmp_path):
    idx,b,cfg,corpus=matrix;rb=Reranker();ab=Answer()
    rows=[]
    for _ in range(2):
        run=independent_request(idx,corpus,'herbs',config=cfg,arm='matrix_rerank',embedding_backend=b,
                                reranker=rb,answer_backend=ab,cache_policy='bypass')
        assert run['benchmark']['has_answer_call']
        assert run['benchmark']['wall_seconds']>=run['benchmark']['prepare_seconds']
        assert len(run['actual_calls']['query_embedding'])==1
        assert not run['benchmark']['embedding_cache_reused']
        assert replay(idx,corpus,run)['verified'];rows.append(run)
    assert len(b.calls)==3  # offline document call + two independent query calls
    summary=benchmark_report(rows)
    assert summary['arms']['matrix_rerank']['end_to_end_latency_runs']==2
    assert summary['arms']['matrix_rerank']['mean_end_to_end_seconds']>0


def test_reused_preparation_is_not_counted_as_live_latency(matrix,tmp_path):
    idx,b,cfg,corpus=matrix
    b.spec={'cache_dir':str(tmp_path/'emb'),'cache_only':False,'batch_size':10}
    rb=Reranker();ab=Answer();runs=[]
    for _ in range(2):
        runs.append(independent_request(idx,corpus,'herbs',config=cfg,arm='matrix_rerank',embedding_backend=b,
             reranker=rb,answer_backend=ab,cache_policy='reuse',plan_cache_dir=str(tmp_path/'plans'),rerank_cache_dir=str(tmp_path/'rerank')))
    assert runs[1]['benchmark']['embedding_cache_reused'] and runs[1]['benchmark']['rerank_cache_reused']
    assert runs[1]['actual_calls']['query_embedding']==[] and runs[1]['calls']['query_embedding']
    summary=benchmark_report(runs)
    assert summary['arms']['matrix_rerank']['end_to_end_latency_runs']==1
    assert sum(len(v) for v in runs[1]['actual_calls'].values())==1  # only fresh answer
    # Explicit bypass ignores previously populated local caches.
    fresh=independent_request(idx,corpus,'herbs',config=cfg,arm='matrix_rerank',embedding_backend=b,
             reranker=rb,answer_backend=ab,cache_policy='bypass',plan_cache_dir=str(tmp_path/'plans'),rerank_cache_dir=str(tmp_path/'rerank'))
    assert not fresh['benchmark']['embedding_cache_reused'] and fresh['actual_calls']['rerank']


def test_failure_receipt_retains_spend_and_is_gradeable(matrix,tmp_path):
    from eitr.evaluate import grade_batch
    idx,b,cfg,corpus=matrix
    class Broken(Reranker):
        def score(self,*args):
            self.calls.append(call('rerank_tokens','broken',i=30,o=0))
            raise ProviderTransportError('test outage')
    run=independent_request(idx,corpus,'herbs',config=cfg,arm='matrix_rerank',embedding_backend=b,
                            reranker=Broken(),answer_backend=Answer())
    assert run['schema']=='eitr-matrix-failure-1' and run['failure_stage']=='retrieve'
    assert run['actual_calls']['query_embedding'] and run['actual_calls']['rerank']
    assert replay(idx,corpus,run)['execution_succeeded'] is False
    assert benchmark_report([run])['arms']['matrix_rerank']['end_to_end_latency_runs']==0
    case={'question_id':'q','question':'herbs','index_hash':idx.index_hash,'rows':[run]};filename=digest('q')[:24]+'.json'
    root=tmp_path/'batch';write_json(root/filename,case)
    write_json(root/'manifest.json',{'schema':'eitr-matrix-batch-1','status':'complete','index_hash':idx.index_hash,
        'arms':['matrix_rerank'],'completed':[{'question_id':'q','file':filename,'content_hash':digest(case)}]})
    labels=[{'question_id':'q','owner_id':idx.snapshot['owner_id'],'snapshot_id':idx.snapshot['snapshot_id'],'references':['herbs']}]
    class NoJudge:
        calls=[]
        def chat_json(self,*args):pytest.fail('No answer must not call judge')
    graded=grade_batch(root,labels,idx,NoJudge(),tmp_path/'grade')
    assert len(graded)==1 and not graded[0]['correct']


def test_cold_loader_is_inside_timed_request(matrix):
    idx,b,cfg,corpus=matrix;seen=[]
    def loader():seen.append(True);return idx,corpus
    run=independent_request(idx,None,'herbs',config=cfg,arm='matrix_embedding',embedding_backend=b,
                            answer_backend=Answer(),loader=loader)
    assert seen and run['benchmark']['load_mode']=='cold'
    assert run['benchmark']['load_seconds']<=run['benchmark']['wall_seconds']


@pytest.fixture
def benchmark_spec(snapshot,tmp_path,monkeypatch):
    import eitr.matrix_benchmark as bench
    import eitr.matrix_experiment as experiment
    from eitr import matrix_embeddings as emb
    es=embedding_spec(tmp_path);es['cache_only']=False
    def fake_embed(self,texts,role):
        self.calls.append(call('embedding_tokens',self.identity['model'],self.identity['endpoint_hash'],len(texts),0))
        return Embeddings().embed(texts,role)
    monkeypatch.setattr(emb.EmbeddingBackend,'embed',fake_embed)
    backend=EmbeddingBackend(es);idx=build_index(snapshot,embed_snapshot(snapshot,backend),graph_top_k=2,graph_mode='exact')
    ip=tmp_path/'index.json';idx.save(ip);cfg=config();cp=tmp_path/'corpus.json';Corpus(snapshot,CharacterTokenizer(),cfg).save(cp)
    qp=tmp_path/'questions.json';write_json(qp,[{'question_id':'q','question':'herbs','owner_id':snapshot['owner_id'],'snapshot_id':snapshot['snapshot_id']}])
    # A named mock tokenizer avoids real external tokenizer downloads in network-free integration tests.
    monkeypatch.setattr(experiment,'tokenizer_from',lambda spec:CharacterTokenizer())
    monkeypatch.setattr(bench,'tokenizer_from',lambda spec:CharacterTokenizer())
    monkeypatch.setattr(bench,'model_backend',lambda spec:Answer())
    model={'api_base':'https://example.invalid/v1','model':'test-answer','key_env':'TEST_KEY','max_output_tokens':128}
    experiment_spec={'config':cfg.to_dict(),'arms':['matrix_embedding'],'tokenizer':{'kind':'mock-test-only'},
        'assets':[{'index':str(ip),'corpus':str(cp)}],'questions':str(qp),'embedding':es,
        'frontend':'whole_query','model':model,'plan_cache_dir':str(tmp_path/'plans'),'plan_cache_only':False,
        'reranker':None,'generate':True}
    return {'experiment':experiment_spec,'settings':{'repeats':2,'warmup_repeats':1,'seed':19,'cache_policy':'bypass','load_mode':'warm'},
            'prices':prices_for(('embedding_tokens','test',backend.identity['endpoint_hash']),('chat_tokens','test-answer',digest('endpoint')))}


def test_benchmark_cli_frozen_order_warmup_spend_and_report(benchmark_spec,tmp_path):
    spec=benchmark_spec;sp=tmp_path/'spec.json';pp=tmp_path/'protocol.json';write_json(sp,spec)
    assert main(['matrix-benchmark-freeze','--spec',str(sp),'--out',str(pp)])==0
    protocol=read_json(pp);assert verify_benchmark(protocol)==protocol
    assert len(protocol['jobs'])==3 and protocol['jobs'][0]['warmup']
    out=tmp_path/'run';assert main(['matrix-benchmark','--protocol',str(pp),'--out',str(out)])==0
    runs=load_benchmark_runs(out);assert len(runs)==3
    summary=read_json(out/'summary.json')
    assert summary['warmup_requests']==1 and summary['measured_requests']==2
    assert summary['arms']['matrix_embedding']['runs']==2
    assert summary['actual_experimental_spend']['complete']
    assert len(read_json(out/'manifest.json')['grading_batches'])==2
    assert main(['matrix-cost-report','--benchmark',str(out),'--out',str(tmp_path/'report.json')])==0
    assert read_json(tmp_path/'report.json')==summary
    assert main(['matrix-cost-report','--benchmarks',str(out),str(out),'--out',str(tmp_path/'duplicates.json')])==2
    assert not (tmp_path/'duplicates.json').exists()
    # Tampering with a job is caught before aggregation.
    job=read_json(out/'jobs/000000.json');job['question']='changed';write_json(out/'jobs/000000.json',job)
    with pytest.raises(ContractError):load_benchmark_runs(out)


def receipt(tmp_path,name,idx,wall,rate_calls,reused=False):
    data={'schema':'eitr-matrix-build-metrics-1','index_hash':idx.index_hash,'owner_id':idx.snapshot['owner_id'],
          'snapshot_id':idx.snapshot['snapshot_id'],'snapshot_hash':idx.snapshot['content_hash'],'embedding_stage_seconds':0.,'embedding_record_hash':idx.embedding['record_hash'],
          'resources':{'wall_seconds':wall,'cpu_seconds':1.},'offline_embedding_calls':rate_calls,
          'actual_embedding_calls':[] if reused else rate_calls,'embedding_reused':reused}
    data['metrics_hash']=digest(data);p=tmp_path/(name+'.json');write_json(p,data);return str(p)


def test_lifecycle_uses_paired_samples_and_real_build_receipts(matrix,tmp_path):
    idx,b,cfg,corpus=matrix;h=digest('endpoint');p=prices_for(('chat_tokens','chat',h),('embedding_tokens','embed',h))
    basepath=receipt(tmp_path,'baseline',idx,5,[call('embedding_tokens','embed',i=100,o=0)])
    methodpath=receipt(tmp_path,'method',idx,15,[call('embedding_tokens','embed',i=100,o=0)])
    runs=[]
    for arm,seconds,tokens in [('baseline',3,100),('method',1,50)]:
        runs.append({'arm':arm,'index_hash':idx.index_hash,'calls':{'answer':[call(i=tokens,o=0)]},'job':{'owner_id':idx.snapshot['owner_id'],
            'snapshot_id':idx.snapshot['snapshot_id'],'question_id':'q','trial':0},
            'benchmark':{'independent_request':True,'has_answer_call':True,'wall_seconds':seconds}})
    c={'owner_id':idx.snapshot['owner_id'],'snapshot_id':idx.snapshot['snapshot_id'],'baseline_arm':'baseline','method_arm':'method',
       'baseline_builds':[basepath],'method_builds':[methodpath],'queries_per_snapshot':10,'compute_usd_per_cpu_hour':0}
    report=lifecycle_report(runs,{'comparisons':[c]},p)['comparisons'][0]
    assert report['units']['seconds']['break_even_queries']==5
    assert report['units']['seconds']['net_saving_at_horizon']==10
    assert report['units']['usd']['net_saving_at_horizon']==pytest.approx(.001)
    runs[1]['job']['question_id']='other'
    with pytest.raises(ContractError):lifecycle_report(runs,{'comparisons':[c]},p)
    record=read_json(basepath);record['resources']['wall_seconds']=0
    with pytest.raises(ContractError):build_accounting(record,p,0)


def test_repeat_summary_does_not_inflate_question_count(tmp_path):
    files=[]
    for trial in range(3):
        rows=[{'owner_id':o,'question_id':'q','arm':a,'correct':(a=='method' or trial==0)}
              for o in ('a','b') for a in ('baseline','method')]
        path=tmp_path/f'{trial}.json';write_json(path,rows);files.append({'trial':trial,'rows':str(path)})
    summary=summarize_repeats(files,draws=100)
    assert summary['repeats']==3 and summary['arms']['method']['questions']==2
    assert summary['pairs']['baseline vs method']['delta_b_minus_a']==pytest.approx(2/3)
    assert not summary['noninferiority_established']
    rows=read_json(files[0]['rows']);rows.pop();write_json(files[0]['rows'],rows)
    with pytest.raises(ContractError):summarize_repeats(files,draws=100)


def test_all_arms_benchmark_failure_is_preserved_in_complete_journal(benchmark_spec,tmp_path,monkeypatch):
    import eitr.matrix_benchmark as bench
    class Flaky(Reranker):
        def score(self,q,documents):
            self.calls.append(call('rerank_tokens','broken',i=30,o=0))
            raise ProviderTransportError('test rerank outage')
    monkeypatch.setattr(bench,'backend_from_spec',lambda spec:Flaky())
    spec=benchmark_spec;spec['settings'].update(repeats=1,warmup_repeats=0,load_mode='cold')
    spec['experiment']['arms']=['direct_rerank','matrix_rerank','matrix_graph_rerank','matrix_embedding']
    spec['experiment']['reranker']={'provider':'dashscope','endpoint':'https://example.invalid/rerank','model':'broken',
        'key_env':'TEST_KEY','max_documents':500,'cache_dir':str(tmp_path/'rerank'),'cache_only':False}
    protocol=freeze_benchmark(spec);out=tmp_path/'failed-benchmark'
    result=run_benchmark(protocol,out);runs=load_benchmark_runs(out)
    assert result['status']=='complete_with_failures' and len(runs)==4
    assert sum(r['schema']=='eitr-matrix-failure-1' for r in runs)==3
    assert len(result['grading_batches'])==1
    report=read_json(out/'summary.json')
    assert report['failed_measured_requests']==3
    batch=read_json(out/result['grading_batches'][0]['path']/'manifest.json')
    case=read_json(out/result['grading_batches'][0]['path']/batch['completed'][0]['file'])
    assert [r['arm'] for r in case['rows']]==spec['experiment']['arms']


def test_cached_frontend_cost_survives_embedding_failure(matrix,tmp_path):
    from eitr.matrix_query import plan_record
    idx,b,cfg,corpus=matrix
    class Front:
        identity={'model':'test-frontend'}
        def __init__(self):self.calls=[]
        def chat_json(self,*args):
            self.calls.append(call(model='test-frontend'))
            return {'requirements':[{'text':'herbs','anchors':['herbs']}],'operator':'single','operator_anchor':'','subjects':[],'constraints':[]}
    front=Front();plan_record('herbs',backend=front,cache_dir=tmp_path,max_requirements=cfg.max_requirements)
    class BadEmbedding:
        identity=idx.embedding['request']['backend']
        calls=[]
        def embed(self,*args):
            self.calls.append(call('embedding_tokens','failed',i=1,o=0))
            raise ProviderTransportError('embedding failure')
    run=independent_request(idx,corpus,'herbs',config=cfg,arm='matrix_embedding',embedding_backend=BadEmbedding(),
                            frontend_backend=front,cache_policy='reuse',plan_cache_dir=tmp_path)
    assert run['schema']=='eitr-matrix-failure-1' and run['failure_stage']=='prepare'
    assert run['calls']['frontend'] and not run['actual_calls']['frontend']
    assert run['benchmark']['frontend_cache_reused']
    assert 'plan' in run['prepared_query'] and run['actual_calls']['query_embedding']


def test_shared_embedding_latency_cancels_without_fabricating_cold_time(matrix,tmp_path):
    idx,b,cfg,corpus=matrix;h=digest('endpoint');p=prices_for(('chat_tokens','chat',h),('embedding_tokens','embed',h))
    paths=[receipt(tmp_path,side,idx,wall,[call('embedding_tokens','embed',i=100,o=0)],reused=True)
           for side,wall in [('base',2),('method',8)]]
    runs=[{'arm':side,'index_hash':idx.index_hash,'calls':{'answer':[call(i=10,o=0)]},
           'job':{'owner_id':idx.snapshot['owner_id'],'snapshot_id':idx.snapshot['snapshot_id'],'question_id':'q','trial':0},
           'benchmark':{'independent_request':True,'has_answer_call':True,'wall_seconds':seconds}}
          for side,seconds in [('base',2),('method',1)]]
    c={'owner_id':idx.snapshot['owner_id'],'snapshot_id':idx.snapshot['snapshot_id'],'baseline_arm':'base','method_arm':'method',
       'baseline_builds':[paths[0]],'method_builds':[paths[1]],'queries_per_snapshot':10,'compute_usd_per_cpu_hour':0}
    row=lifecycle_report(runs,{'comparisons':[c]},p)['comparisons'][0]
    assert row['shared_embedding_time_excluded'] and row['units']['seconds']['break_even_queries']==6
    c['baseline_builds']*=2
    with pytest.raises(ContractError):lifecycle_report(runs,{'comparisons':[c]},p)


def test_price_table_and_code_pins_are_verified(benchmark_spec):
    protocol=freeze_benchmark(benchmark_spec)
    bad=deepcopy(protocol);bad['spec']['prices']['rates'][0]['input_per_million']=99
    with pytest.raises(ContractError):verify_benchmark(bad)
    bad=deepcopy(protocol);bad['runtime_source_hash']='different';bad['protocol_hash']=digest({k:v for k,v in bad.items() if k!='protocol_hash'})
    with pytest.raises(ContractError):verify_benchmark(bad)


def test_embedding_build_resume_preserves_completed_batches_and_failed_spend(tmp_path):
    from eitr.matrix_embeddings import embed_record
    class Interrupted(Embeddings):
        interrupted=False
        def embed(self,texts,role):
            if texts==['Used herbs'] and not self.interrupted:
                self.interrupted=True
                self.calls.append(call('embedding_tokens','interrupted',i=1,o=0))
                raise ProviderTransportError('test mid-build failure')
            return super().embed(texts,role)
    backend=Interrupted();texts=['Bought spices','Used herbs']
    with pytest.raises(ProviderTransportError):embed_record(texts,'document',{'scope':'offline'},backend,cache_dir=tmp_path,batch_size=1)
    assert len(list((tmp_path/'parts').glob('*.json')))==1
    assert len(list((tmp_path/'failures').glob('*.json')))==1
    before=len(backend.calls)
    record=embed_record(texts,'document',{'scope':'offline'},backend,cache_dir=tmp_path,batch_size=1)
    assert len(backend.calls)-before==1 and len(record['calls'])==2
    assert record['vectors']==[[1.,0,0,0],[0,1.,0,0]]
    # Whole-record removal still permits network-free reconstruction from pinned parts.
    (tmp_path/(digest(record['request'])+'.json')).unlink()
    again=embed_record(texts,'document',{'scope':'offline'},backend,cache_dir=tmp_path,batch_size=1,cache_only=True)
    assert again==record and len(backend.calls)==before+1


def test_chat_provider_preserves_cached_token_usage_for_pricing(monkeypatch):
    import eicr.provider as provider
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return json.dumps({'usage':{'prompt_tokens':100,'completion_tokens':20,
            'prompt_tokens_details':{'cached_tokens':60}},'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}).encode()
    monkeypatch.setenv('TEST_CACHE_TOKEN_KEY','test-only')
    monkeypatch.setattr(provider,'_open_no_redirect',lambda *args,**kwargs:Response())
    backend=provider.ChatBackend('https://example.invalid/v1','chat','TEST_CACHE_TOKEN_KEY')
    assert backend.chat_json('system','question')=={'ok':True}
    assert backend.calls[0]['usage']['cached_input_tokens']==60
    p=prices_for(('chat_tokens','chat',digest('https://example.invalid/v1')))
    p['rates'][0]['cached_input_per_million']=.5
    assert price_calls(backend.calls,p)['usd']==pytest.approx(.00023)


def test_bypass_does_not_read_corrupt_local_plan_cache(benchmark_spec,tmp_path):
    from eitr.matrix_query import plan_request
    spec=benchmark_spec;request=plan_request('herbs',max_requirements=spec['experiment']['config']['max_requirements'])
    path=Path(spec['experiment']['plan_cache_dir'])/(digest(request)+'.json')
    path.parent.mkdir(parents=True);path.write_text('corrupted existing cache')
    protocol=freeze_benchmark(spec)
    assert protocol['experiment_protocol']['cache_preflight'] is False
    result=run_benchmark(protocol,tmp_path/'bypass')
    assert result['status']=='complete'
    spec['settings']['cache_policy']='reuse'
    with pytest.raises((ContractError,ValueError)):freeze_benchmark(spec)
