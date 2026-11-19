"""No-network acceptance against an existing, attested native cache.

The native loop receives scripted tool calls, NOT model predictions. This checks
interfaces/provenance/math/serialization and never reports QA effectiveness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from eicr.contracts import read_json, write_json, require, digest
from eicr.packer import TiktokenTokenizer
from eitr.native import import_pack
from eitr.dates import attest_dates
from eitr.index import build_index,save_index
from eitr.evidence import Corpus
from eitr.config import Config
from eitr.query import map_groups
from eitr.pipeline import compare,runtime_source_hash
from eitr.native_runtime import pin_runtime,load_runtime
from eitr.experiment import replay
from eitr.accounting import Meter,environment_identity


def main():
    p=argparse.ArgumentParser()
    for name in ('pack-dir','owner','scope-records','dataset','native-root','out'):p.add_argument('--'+name,required=True)
    args=p.parse_args();root=Path(args.pack_dir);out=Path(args.out)
    require(not out.exists(),'Acceptance output exists');out.mkdir(parents=True)
    manifest=read_json(root/'manifest.json')
    for name in (args.owner+'.json',args.owner+'.npz'):
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==manifest['files_sha256'][name],'Native pack/cache SHA256 mismatch')
    pack=read_json(root/(args.owner+'.json'));scope=read_json(args.scope_records)
    m={**scope['provenance'],'owner_id':scope['owner_id'],'snapshot_id':scope['snapshot_id'],'keyword_cache_is_pre_injection':True}
    require(args.owner==m['owner_id'],'Owner mismatch')
    snapshot=import_pack(pack,m)
    native=next(row for row in read_json(args.dataset) if row['sample_id']==args.owner)
    snapshot=attest_dates(snapshot,{'owner_id':args.owner,'conversation':native['conversation'],'raw_format':'mragent_artifacts_v1'})
    cfg=Config(evidence_budget=2000);tok=TiktokenTokenizer('cl100k_base')
    write_json(out/'snapshot.json',snapshot);write_json(out/'config.json',cfg.to_dict())
    meter=Meter();index=build_index(snapshot,cfg.top_k);cue_build=meter.result();save_index(index,out/'cue-index.json')
    meter=Meter();corpus=Corpus(snapshot,tok,cfg);corpus_build=meter.result();corpus.save(out/'corpus.json')
    cached=Corpus.load(out/'corpus.json',snapshot,tok,cfg)
    assert corpus.scores('Tokyo')==cached.scores('Tokyo')
    native_pin=pin_runtime(args.native_root);write_json(out/'native-pin.json',native_pin);runtime=load_runtime(args.native_root,native_pin)
    question='When did I visit Tokyo?'
    q=map_groups(index,question,cfg)
    key_text={c['cue_id']:c['text'] for c in snapshot['cues']};tag_text={t['tag_id']:t['text'] for t in snapshot['tags']}
    pairs={(l['cue_id'],l['tag_id']) for l in snapshot['links']}
    choices=sorted((k,t) for g in q['groups'] for k in g['key_ids'] for t in g['seed_tags'] if (k,t) in pairs)
    if not choices:choices=sorted(pairs)[:1]
    k,t=choices[0]
    class Scripted:
        model='scripted-interface-test-not-LLM'
        def __init__(self):self.calls=[]
        def chat_tools(self,messages,tools):
            self.calls.append({'status':'ok','backend_kind':'scripted_not_network','usage':{'prompt_tokens':0,'completion_tokens':0}})
            if len(self.calls)==1:
                return {'role':'assistant','tool_calls':[{'id':'acceptance-read','type':'function','function':{
                    'name':'edges_by_tag','arguments':json.dumps({'key':key_text[k],'tag':tag_text[t],'note':'Interface acceptance read'})}}]}
            return {'role':'assistant','content':json.dumps({'mode':'answer','answer':'scripted stop; not a prediction','supports':[]})}
    result=compare(index,question,tok,config=cfg,corpus=cached,arms=['P0','P1','P2','P3','P4','P5'],native_runtime=runtime,exploration_backend=Scripted())
    write_json(out/'comparison.json',result)
    replay_results=[replay(row,index,cached) for row in result['rows'] if row['arm']!='P5']
    assert all(row['evidence_pack']['used_tokens']<=cfg.evidence_budget for row in result['rows'])
    assert result['rows'][-1]['native_loop_result']['status']=='native_loop_completed'
    with np.load(root/(args.owner+'.npz'),allow_pickle=False) as data:
        fp={f['id']:i for i,f in enumerate(pack['corpus'])};ids=[f['fact_id'] for f in snapshot['facts']]
        require(set(ids)==set(fp),'Embedding fact IDs mismatch')
        bundle={'snapshot_hash':snapshot['content_hash'],'encoder':manifest['embedding_config']['model_id']+':'+manifest['files_sha256'][args.owner+'.npz'],
                'fact_ids':ids,'vectors':data['embeddings'][[fp[f] for f in ids]].tolist()}
    meter=Meter();fact_index=build_index(snapshot,cfg.top_k,bundle);fact_build=meter.result();save_index(fact_index,out/'fact-index.json')
    fact_result=compare(fact_index,question,tok,config=cfg,corpus=cached,arms=['P3','P4'])
    write_json(out/'fact-comparison.json',fact_result)
    for row in fact_result['rows']:assert replay(row,fact_index,cached)['verified']
    summary={'kind':'engineering_acceptance_not_effectiveness_experiment','real_API_calls':0,'question_count':1,
        'runtime_source_hash':runtime_source_hash(),'environment':environment_identity(),'native_runtime_hash':runtime.identity,
        'source_checksums':{'pack':manifest['files_sha256'][args.owner+'.json'],'vectors':manifest['files_sha256'][args.owner+'.npz'],
                            'dataset':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()},
        'dates_verified':sum(bool(s.get('session_datetime')) for s in snapshot['sources']),
        'topics':len(snapshot['native_auxiliary']['topics']),'personal':len(snapshot['native_auxiliary']['personal']),
        'cue_index':{k:index.report[k] for k in ('tags','facts','cues','edges','zero_vectors')},
        'fact_index':{k:fact_index.report[k] for k in ('tags','facts','edges','zero_vectors')},
        'offline_resources':{'cue_build':cue_build,'fact_build':fact_build,'corpus_build':corpus_build},
        'corpus_load_seconds':cached.load_seconds,'same_pool':result['same_pool_verified'],
        'exact_replay':replay_results,'arms':[{'arm':r['arm'],'tokens':r['evidence_pack']['used_tokens'],
                                             'retrieval_seconds':r['timing_seconds']['retrieval_total']} for r in result['rows']],
        'P5':'Actual pinned native loop and bridge; scripted transport only; no QA accuracy claim',
        'tokenizer_note':'cl100k_base for engineering checks only; does not assert compatibility with future answer model'}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
