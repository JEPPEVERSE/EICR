"""Attach native LoCoMo session dates only after exact raw-text matching."""
from copy import deepcopy
from eicr.contracts import digest, require
from .native import validate_snapshot


def attest_dates(snapshot, source):
    validate_snapshot(snapshot)
    require(isinstance(source,dict) and {'owner_id','conversation'}<=set(source)<={'owner_id','conversation','raw_format'}
            and source['owner_id']==snapshot['owner_id'], 'Date source owner mismatch')
    raw_format=source.get('raw_format','speaker_colon_space_text')
    require(raw_format in ('speaker_colon_space_text','mragent_artifacts_v1'), 'Unknown native raw renderer')
    conv=source['conversation'];require(isinstance(conv,dict),'Invalid native conversation')
    raw={}
    for key, turns in conv.items():
        if not key.startswith('session_') or not isinstance(turns,list):continue
        date=conv.get(key+'_date_time')
        require(isinstance(date,str) and date.strip(),'Missing native session date')
        for turn in turns:
            require(isinstance(turn,dict) and all(isinstance(turn.get(k),str) for k in ('dia_id','speaker','text')),'Invalid native turn')
            sid=turn['dia_id'];require(sid not in raw,'Duplicate native source')
            text=turn['speaker']+': '+turn['text']
            if raw_format=='mragent_artifacts_v1':
                # Exact documented renderer in MRAgent experiments/entry_sensitivity/artifacts.py.
                text=turn['speaker'].strip()+':'+turn['text'].strip()
                if turn.get('blip_caption') is not None:
                    require(isinstance(turn['blip_caption'],str),'Invalid native image caption')
                    text+=' and shared '+turn['blip_caption']
            raw[sid]=(text,date,turn['speaker'])
    result=deepcopy(snapshot)
    for s in result['sources']:
        require(s['source_id'] in raw and raw[s['source_id']][0]==s['text'], 'Native date source text mismatch')
        text,date,speaker=raw[s['source_id']]
        if s.get('session_datetime') is not None:require(s['session_datetime']==date,'Conflicting attested source date')
        s.update(session_datetime=date,datetime_provenance='native_verified',speaker=speaker)
    result['provenance']['date_source_hash']=digest(source)
    result['content_hash']=digest({k:v for k,v in result.items() if k!='content_hash'})
    return validate_snapshot(result)
