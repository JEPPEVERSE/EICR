"""Frozen input protocol, independent timing and exact scheduler replay."""
from pathlib import Path
import hashlib
import json
from eicr.contracts import digest, require, read_json, write_json
from eicr.packer import CharacterTokenizer, TiktokenTokenizer
from eicr.provider import ChatBackend
from .config import Config
from .evidence import Corpus, JsonTokenizer
from .index import load_index
from .pipeline import retrieve, RUN_ARMS, runtime_source_hash
from .native_runtime import pin_runtime, load_runtime
from .tool_backend import ToolBackend
from .accounting import Meter, cost_ledger, environment_identity


def tokenizer_from(spec):
    require(isinstance(spec,dict) and set(spec)<={'kind','path','sha256'}, 'Invalid tokenizer spec')
    if spec['kind']=='characters': return CharacterTokenizer()
    if spec['kind']=='json': return JsonTokenizer(spec['path'],spec['sha256'])
    return TiktokenTokenizer(spec['kind'])


def validate_questions(questions, indexes, frontend):
    require(isinstance(questions,list) and questions, 'Empty question set')
    ids=set()
    for q in questions:
        require(isinstance(q,dict) and set(q)<={'question_id','owner_id','snapshot_id','question','mentions'}, 'Labels/unknown fields forbidden in online inputs')
        key=(q.get('owner_id'),q.get('snapshot_id'))
        require(key in indexes and isinstance(q.get('question_id'),str) and q['question_id']
                and (key,q['question_id']) not in ids and isinstance(q.get('question'),str) and q['question'].strip(), 'Invalid question scope/identity')
        ids.add((key,q['question_id']))
        if frontend=='supplied':
            require('mentions' in q, 'Supplied frontend requires explicit mentions, including empty lists')
            from eicr.query_map import ground_mentions, map_query
            map_query(indexes[key],q['question'],ground_mentions(q['question'],q['mentions']))
        else: require('mentions' not in q, 'Live frontend cannot use supplied mentions')


def freeze(spec):
    require(isinstance(spec,dict) and set(spec)=={'config','arms','tokenizer','assets','questions','native','model','frontend','generate','require_dates','prices'}, 'Protocol fields must be explicit')
    cfg=Config.from_dict(spec['config']); arms=spec['arms']
    require(isinstance(arms,list) and arms and len(set(arms))==len(arms) and set(arms)<=set(RUN_ARMS),'Invalid protocol arms')
    require(spec['frontend'] in ('llm','supplied') and type(spec['generate']) is bool and type(spec['require_dates']) is bool,'Invalid protocol modes')
    model=spec['model']
    require(isinstance(model,dict) and set(model)=={'api_base','model','key_env','max_output_tokens'}, 'Pin complete model settings')
    ChatBackend(model['api_base'],model['model'],model['key_env'],max_output_tokens=model['max_output_tokens'],temperature=0)
    tok=tokenizer_from(spec['tokenizer'])
    require(spec['tokenizer']['kind']!='characters' or not (spec['generate'] or spec['frontend']=='llm' or 'P5' in arms), 'Live requests need the actual model tokenizer')
    require(isinstance(spec['assets'],list) and spec['assets'],'No protocol assets')
    files={}; indexes={}; reports=[]
    def pin(path):
        require(Path(path).is_absolute(), 'Frozen asset paths must be absolute')
        p=Path(path).resolve();files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    for asset in spec['assets']:
        require(set(asset)=={'index','corpus'},'Invalid asset fields')
        index=load_index(asset['index']);key=(index.snapshot['owner_id'],index.snapshot['snapshot_id'])
        require(key not in indexes and index.top_k==cfg.top_k, 'Duplicate snapshot or top_k mismatch')
        corpus=Corpus.load(asset['corpus'],index.snapshot,tok,cfg)
        if spec['require_dates']:
            require(all(s.get('session_datetime') for s in index.snapshot['sources']), 'Formal protocol requires verified source dates')
        if 'P5' in arms: require('native_auxiliary' in index.snapshot, 'P5 requires reimported native auxiliary graph')
        indexes[key]=index;pin(asset['index']);pin(asset['corpus'])
        reports.append({'owner_id':key[0],'snapshot_id':key[1],'index_hash':index.index_hash,'corpus_hash':corpus.identity})
    questions=read_json(spec['questions']);validate_questions(questions,indexes,spec['frontend']);pin(spec['questions'])
    if spec['tokenizer']['kind']=='json': pin(spec['tokenizer']['path'])
    if 'P5' in arms:
        require(isinstance(spec['native'],dict) and set(spec['native'])=={'root','manifest'},'P5 requires a native source pin')
        require(Path(spec['native']['root']).is_absolute(),'Frozen native checkout path must be absolute')
        require(read_json(spec['native']['manifest'])==pin_runtime(spec['native']['root']),'Native source mismatch')
        pin(spec['native']['manifest'])
    elif spec['native'] is not None:
        require(isinstance(spec['native'],dict),'Invalid native settings')
    # Validate price fields without inventing a model price.
    cost_ledger({'calls':{}},spec['prices'])
    body={'schema':'eitr-protocol-1','spec':spec,'files':files,'runtime_source_hash':runtime_source_hash(),
          'environment':environment_identity(),
          'tokenizer_identity':tok.identity,'asset_reports':reports,'questions_hash':digest(questions),
          'question_count':len(questions),'quality_claim':'engineering pilot; no noninferiority conclusion'}
    return {**body,'protocol_hash':digest(body)}


def verify_protocol(protocol):
    require(protocol.get('schema')=='eitr-protocol-1' and protocol.get('protocol_hash')==digest({k:v for k,v in protocol.items() if k!='protocol_hash'}),'Protocol checksum mismatch')
    require(freeze(protocol['spec'])==protocol,'Frozen files/code/config changed; create a new protocol')
    return protocol


def run_protocol(protocol, output_dir):
    verify_protocol(protocol)
    spec=protocol['spec'];cfg=Config.from_dict(spec['config']);tok=tokenizer_from(spec['tokenizer'])
    runtime=load_runtime(spec['native']['root'],read_json(spec['native']['manifest'])) if 'P5' in spec['arms'] else None
    model=spec['model']
    def backend(cls=ChatBackend): return cls(model['api_base'],model['model'],model['key_env'],max_output_tokens=model['max_output_tokens'],temperature=0)
    front=backend() if spec['frontend']=='llm' else None
    answer=backend() if spec['generate'] else None
    explore=backend(ToolBackend) if runtime else None
    out=Path(output_dir);require(not out.exists(),'Experiment output exists');out.mkdir(parents=True)
    write_json(out/'protocol.json',protocol)
    journal={'schema':'eitr-experiment-1','status':'running','protocol_hash':protocol['protocol_hash'],'completed':[]}
    write_json(out/'manifest.json',journal)
    questions=read_json(spec['questions'])
    try:
        from .batch import run_batch
        for asset in spec['assets']:
            index=load_index(asset['index']);corpus=Corpus.load(asset['corpus'],index.snapshot,tok,cfg)
            subset=[q for q in questions if q['owner_id']==index.snapshot['owner_id'] and q['snapshot_id']==index.snapshot['snapshot_id']]
            if not subset: continue
            name=digest([index.snapshot['owner_id'],index.snapshot['snapshot_id']])[:24]
            run_batch(index,subset,tok,config=cfg,arms=spec['arms'],frontend_backend=front,answer_backend=answer,
                      output_dir=out/name,corpus=corpus,native_runtime=runtime,exploration_backend=explore)
            journal['completed'].append({'owner_id':index.snapshot['owner_id'],'snapshot_id':index.snapshot['snapshot_id'],'batch':name})
            write_json(out/'manifest.json',journal)
        journal['status']='complete';write_json(out/'manifest.json',journal)
    except Exception as error:
        journal.update(status='failed',error_type=type(error).__name__);write_json(out/'manifest.json',journal);raise
    return journal


def benchmark(index, questions, tokenizer, *, config, arms, corpus, repeats=1,
              frontend_backend=None, answer_backend=None,native_runtime=None,exploration_backend=None,output_path=None):
    require(type(repeats) is int and repeats>0 and arms and len(set(arms))==len(arms) and set(arms)<=set(RUN_ARMS),'Invalid benchmark settings')
    validate_questions(questions,{(index.snapshot['owner_id'],index.snapshot['snapshot_id']):index},'llm' if frontend_backend else 'supplied')
    if 'P5' in arms: require(native_runtime and exploration_backend and 'native_auxiliary' in index.snapshot,'P5 benchmark missing dependencies')
    rows=[]
    journal={'schema':'eitr-benchmark-1','status':'running','arms':list(arms),'repeats':repeats,
             'corpus_hash':corpus.identity,'runtime_source_hash':runtime_source_hash()}
    journal_path=str(output_path)+'.manifest.json' if output_path else None
    if journal_path:
        require(not Path(journal_path).exists() and not Path(output_path).exists(),'Benchmark output exists')
        write_json(journal_path,journal)
    try:
        for q in questions:
            for repeat in range(repeats):
                # Rotate order deterministically to avoid always timing one arm first.
                offset=(int(digest(q['question_id'])[:8],16)+repeat)%len(arms)
                for arm in list(arms[offset:])+list(arms[:offset]):
                    meter=Meter()
                    if arm=='P5':
                        from .controlled import run_controlled
                        row=run_controlled(index,q['question'],tokenizer,native_runtime,exploration_backend,config=config,corpus=corpus,
                            mentions=q.get('mentions'),frontend_backend=frontend_backend,answer_backend=answer_backend)
                    else:
                        row=retrieve(index,q['question'],tokenizer,config=config,arm=arm,corpus=corpus,mentions=q.get('mentions'),
                                     frontend_backend=frontend_backend if arm!='P0' else None,answer_backend=answer_backend)
                    row['benchmark']={**meter.result(),'independent_request':True,'corpus_state':'preloaded',
                        'frontend_mode':'live' if frontend_backend and arm!='P0' else 'supplied_or_none',
                        'has_answer_call':answer_backend is not None,'repeat':repeat}
                    row.update(question_id=q['question_id'],owner_id=q['owner_id'])
                    rows.append(row)
                    if output_path:write_json(output_path,rows)
    except Exception as error:
        if journal_path:
            journal.update(status='failed',error_type=type(error).__name__,completed_runs=len(rows),
                calls={'frontend':list(frontend_backend.calls) if frontend_backend else [],
                       'exploration':list(exploration_backend.calls) if exploration_backend else [],
                       'answer':list(answer_backend.calls) if answer_backend else []})
            write_json(journal_path,journal)
        raise
    if journal_path:
        journal.update(status='complete',completed_runs=len(rows));write_json(journal_path,journal)
    return rows


def replay(run,index,corpus):
    from .scheduler import Scheduler, ARMS
    require(run['arm'] in ARMS and run['index_hash']==index.index_hash and
            run['manifest']['runtime_source_hash']==runtime_source_hash(),'Replay code/index mismatch')
    cfg=Config.from_dict(run['manifest']['config']);rel=run['relevance']
    packet=Scheduler(corpus,run['candidates'],rel['whole_query'],[g['scores'] for g in rel['cue_groups']],cfg).select(
        run['arm'],any(g['seed_tags'] for g in run['query']['groups']))
    for field in ('text_hash','atom_ids','used_tokens','ledger','alternatives','coverage'):
        require(packet[field]==run['evidence_pack'][field], 'Replay mismatch: '+field)
    return {'verified':True,'arm':run['arm'],'packet_hash':packet['text_hash'],'decisions':len(packet['ledger'])}
