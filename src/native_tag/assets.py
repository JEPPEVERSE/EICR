"""Import attested native caches or uniformly build new event/topic embeddings."""
from copy import deepcopy
import io
import json
from pathlib import Path
import pickle
import numpy as np
from eicr.contracts import require
from .runtime import file_hash,Trace,verify_sources,build_memory,load_runtime
from .experiment import sealed
from .transport import Encoder


def records(path):
    # Native rewrite/keyword caches are JSONL despite their .json extension.
    text=Path(path).read_text()
    try:
        value=json.loads(text)
        if isinstance(value,list):return value
        return [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


class NumpyCacheUnpickler(pickle.Unpickler):
    """Accept only native numeric arrays/containers, not arbitrary pickle code."""
    def find_class(self,module,name):
        if (module,name) in {('numpy','ndarray'),('numpy','dtype'),
                            ('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
                            ('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar'),
                            ('numpy.core.numeric','_frombuffer'),('numpy._core.numeric','_frombuffer')}:
            return super().find_class(module,name)
        raise ValueError('Unsupported native cache pickle type: '+module+'.'+name)


def prepare_assets(spec, *, reembed=False, encoder_factory=Encoder, receipt_path=None):
    trace=Trace(receipt_path, {'components':['embedding','index'], 'new_embedding':reembed,
                              'common_native_graph_extraction_cost_excluded':True})
    try:
        asset=_prepare_assets(spec,reembed=reembed,encoder_factory=encoder_factory,trace=trace)
    except BaseException as exc:
        trace.flush('failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    return asset,trace.flush('complete',asset_hash=asset['hash'])


def _prepare_assets(spec, *, reembed, encoder_factory, trace):
    runtime=verify_sources(spec['runtime'])
    rewrites,keywords=records(spec['rewrites']),records(spec['keywords'])
    raw=json.loads(Path(spec['raw_text']).read_text())
    texts,topic_texts={},{}
    for i,record in enumerate(rewrites,1):
        require(isinstance(record,dict) and len(record)==1,'Invalid native rewrite record')
        session=next(iter(record.values()))
        if session is None:continue
        for e in session.get('sentence',[]):
            require(e['id'] not in texts,'Duplicate rewrite ID');texts[e['id']]=e['text']
        for tid,text in session.get('topics',{}).items():topic_texts[f'D{i}:'+tid]=text
    trace.stage='offline_embedding'
    provenance=deepcopy(spec.get('provenance',{}))
    if reembed:
        encoder=encoder_factory(spec['embedding'],trace)
        identity=spec['embedding']['identity']
        vectors={eid:encoder.encode(text).tolist() for eid,text in texts.items()}
        topics={tid:encoder.encode(text).tolist() for tid,text in topic_texts.items()}
        provenance.update(identity_verified=True,input_texts_verified=True,
                          origin='uniform_new_embedding_for_all_native_event_and_topic_rows')
    else:
        require(provenance.get('identity_verified') is True and provenance.get('input_texts_verified') is True
                and provenance.get('identity_evidence'), 'Attest native model revision and input identity; otherwise use reembed')
        identity=spec['embedding_identity']
        path=Path(spec['embedding_cache'])
        cache=NumpyCacheUnpickler(io.BytesIO(path.read_bytes())).load()
        require(len(cache['sentence_id'])==len(cache['embeddings']) and len(cache['topic_list'])==len(cache['topic']),
                'Native vector table dimensions mismatch')
        require(len(set(cache['sentence_id']))==len(cache['sentence_id']) and len(set(cache['topic_list']))==len(cache['topic_list']),
                'Duplicate native vector IDs')
        vectors={eid:np.asarray(v).tolist() for eid,v in zip(cache['sentence_id'],cache['embeddings'])}
        topics={tid:np.asarray(v).tolist() for tid,v in zip(cache['topic_list'],cache['topic'])}
        require(list(topics)==list(topic_texts) and set(vectors)==set(texts), 'Cache/rewrite identity mismatch')
        provenance['native_cache_sha256']=file_hash(path)
    body={'schema':'native-tag-assets-1','owner':spec['owner'],'snapshot':spec['snapshot'],
          'raw_text':raw,'rewrites':rewrites,'keywords':keywords,'event_embeddings':vectors,'event_input_texts':texts,
          'topic_ids':list(topic_texts),'topic_embeddings':[topics[t] for t in topic_texts],
          'embedding_identity':identity,'provenance':provenance,'runtime_reference_hashes':runtime['reference'],
          'input_files':{k:{'path':str(Path(spec[k]).resolve()),'sha256':file_hash(spec[k])}
                         for k in ('raw_text','rewrites','keywords')}}
    asset=sealed(body)
    with trace.phase('offline_index_validation'):
        build_memory(load_runtime(spec['runtime'],{'provider':'offline','model':'asset-check','endpoint':'offline'}),asset)
    return asset
