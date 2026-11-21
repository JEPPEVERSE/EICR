"""Synthetic wiring demonstration ONLY; no model or quality evaluation.

Run: python examples/matrix/offline_smoke.py --out /tmp/matrix-smoke
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from eicr.contracts import read_json, write_json, tokens, require
from eicr.packer import CharacterTokenizer
from eitr.native import canonicalize
from eitr.evidence import Corpus
from eitr.matrix_index import build_index, embed_snapshot
from eitr.matrix_pipeline import prepare_query, retrieve, replay
from eitr.matrix_retrieval import MatrixConfig, ARMS


class SyntheticEmbedding:
    identity={'provider':'synthetic-smoke-only','model':'hashed-words','revision':'v1','dimension':32,
              'document_prefix':'','query_prefix':''}
    calls=[]
    def embed(self,texts,role):
        rows=[]
        for text in texts:
            v=np.zeros(32)
            for word in tokens(text):v[int(hashlib.sha256(word.encode()).hexdigest(),16)%32]+=1
            if not v.any():v[0]=1
            rows.append(v.tolist())
        return rows


class SyntheticReranker:
    identity={'provider':'synthetic-smoke-only','model':'word-overlap'}
    calls=[]
    max_documents=500
    def score(self,question,documents):
        words=set(tokens(question))
        return [len(words & set(tokens(d)))/max(1,len(words)) for d in documents]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    out=Path(args.out);require(not out.exists(),'Choose a new output directory');out.mkdir(parents=True)
    raw=read_json(Path(__file__).resolve().parents[1]/'eitr'/'raw_snapshot.json')
    snapshot=canonicalize(raw);eb=SyntheticEmbedding()
    index=build_index(snapshot,embed_snapshot(snapshot,eb),graph_top_k=2,graph_shortlist=4)
    index.save(out/'index.json');cfg=MatrixConfig(evidence_budget=1200)
    corpus=Corpus(snapshot,CharacterTokenizer(),cfg);corpus.save(out/'corpus.json')
    question='Which herbs did I use for the halloumi salad?'
    prepared=prepare_query(index,question,eb,config=cfg);rows=[]
    for arm in ARMS:
        run=retrieve(index,question,corpus,prepared,config=cfg,arm=arm,reranker=SyntheticReranker())
        write_json(out/(arm+'.json'),run)
        rows.append({'arm':arm,'replay':replay(index,corpus,run),'candidates':len(run['candidates']['atom_ids']),
                     'characters_not_model_tokens':run['evidence_pack']['used_tokens']})
    write_json(out/'summary.json',{'scope':'synthetic wiring only, no quality/cost claims','new_API_calls':0,'rows':rows})
    print(str(out/'summary.json'))


if __name__=='__main__':main()
