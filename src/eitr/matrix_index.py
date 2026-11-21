"""Fact-row matrices with source-balanced measures; graph edges retain witnesses."""
from collections import Counter
from itertools import combinations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from eicr.contracts import digest, require, read_json, write_json
from .native import validate_snapshot
from .matrix_embeddings import embed_record, validate_record


def balanced_weights(facts):
    # Exact origin sets define provenance groups; not an independence assertion.
    groups=[tuple(sorted(set(f['origins']))) for f in facts]
    counts=Counter(groups)
    return np.array([1/(len(counts)*counts[g]) for g in groups],dtype=float)


def relation(x, y, wx, wy, block=128):
    """Exact nonnegative cosine kernels, bounded temporary memory."""
    require(len(x)>0 and len(y)>0 and type(block) is int and block>0, 'Empty relation/invalid block')
    left=np.zeros(len(x)); right=np.zeros(len(y)); avg=0.; bridge=-1.; witness=None
    for i in range(0,len(x),block):
        for j in range(0,len(y),block):
            k=np.clip(x[i:i+block] @ y[j:j+block].T,0,1)
            left[i:i+len(k)]=np.maximum(left[i:i+len(k)],k.max(axis=1))
            right[j:j+k.shape[1]]=np.maximum(right[j:j+k.shape[1]],k.max(axis=0))
            avg+=float(wx[i:i+len(k)] @ k @ wy[j:j+k.shape[1]])
            a,b=np.unravel_index(np.argmax(k),k.shape); score=float(k[a,b]); pair=(i+int(a),j+int(b))
            if score>bridge or (score==bridge and pair<witness): bridge,witness=score,pair
    return {'average':avg,'coverage_forward':float(wx @ left),'coverage_reverse':float(wy @ right),
            'bridge':bridge,'witness_rows':list(witness)}


@dataclass
class MatrixIndex:
    snapshot: dict
    embedding: dict
    graph: dict
    graph_config: dict
    index_hash: str = ''

    def __post_init__(self):
        validate_snapshot(self.snapshot)
        self.facts=sorted(self.snapshot['facts'],key=lambda f:f['fact_id'])
        require(self.facts, 'Matrix index requires facts')
        request={'schema':'eitr-matrix-embedding-request-1','backend':self.embedding['request']['backend'],
            'role':'document','scope':{'snapshot_hash':self.snapshot['content_hash'],'fact_ids':[f['fact_id'] for f in self.facts]},
            'texts_hash':digest([f['text'] for f in self.facts]),'count':len(self.facts)}
        self.X=validate_record(self.embedding,request)
        self.tag_ids=sorted(t['tag_id'] for t in self.snapshot['tags'])
        lookup={t:i for i,t in enumerate(self.tag_ids)}
        rows=[]; cols=[]
        for i,f in enumerate(self.facts):
            for t in f['tag_ids']: rows.append(lookup[t]); cols.append(i)
        self.B=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(lookup),len(self.facts)))
        self.members={t:self.B.getrow(i).indices for t,i in lookup.items()}
        self.weights={t:balanced_weights([self.facts[i] for i in ids]) for t,ids in self.members.items()}
        require(set(self.graph)==set(self.tag_ids), 'Graph tag scope mismatch')
        for t,edges in self.graph.items():
            require(len(edges)<=self.graph_config['top_k'] and len({e['target'] for e in edges})==len(edges), 'Invalid graph degree')
            for e in edges:
                require(e['target'] in self.members and e['target']!=t and all(
                    type(e[k]) in (int,float) and np.isfinite(e[k]) and 0<=e[k]<=1+1e-12
                    for k in ('average','coverage_forward','coverage_reverse','bridge')), 'Invalid graph edge')
                require(e['witness'][0] in {self.facts[i]['fact_id'] for i in self.members[t]} and
                        e['witness'][1] in {self.facts[i]['fact_id'] for i in self.members[e['target']]}, 'Invalid graph witnesses')
        calculated=digest(self.body())
        require(not self.index_hash or calculated==self.index_hash,'Matrix index hash mismatch')
        self.index_hash=calculated

    def body(self):
        return {'schema':'eitr-matrix-index-1','snapshot':self.snapshot,'embedding':self.embedding,
                'graph':self.graph,'graph_config':self.graph_config}

    def save(self,path): write_json(path,{**self.body(),'index_hash':self.index_hash})


def build_index(snapshot, embedding, *, graph_top_k=0, graph_shortlist=32, block=128, graph_mode="shortlist"):
    require(type(graph_top_k) is int and graph_top_k>=0 and type(graph_shortlist) is int and
            graph_shortlist>=0 and type(block) is int and block>0, 'Invalid graph limits')
    require(graph_mode in ('shortlist','exact'), 'Unknown graph construction mode')
    require(graph_mode=='exact' or graph_shortlist>=graph_top_k, 'Shortlist is smaller than top-k')
    cfg={'top_k':graph_top_k,'shortlist':graph_shortlist if graph_mode=='shortlist' else None,'block':block,'mode':graph_mode,
         'candidate_method':('all-unordered-tag-pairs; exact directed top-k' if graph_mode=='exact' else
                             'balanced-centroid-union-knn; approximate candidate edges, exact retained relations'),
         'ranking':'coverage_forward'}
    index=MatrixIndex(snapshot,embedding,{t['tag_id']:[] for t in snapshot['tags']},cfg)
    if graph_top_k:
        ids=[t for t in index.tag_ids if len(index.members[t])]
        if graph_mode=='exact':
            pairs=combinations(range(len(ids)),2)
        else:
            centroids=np.array([index.weights[t] @ index.X[index.members[t]] for t in ids])
            if ids:
                norms=np.linalg.norm(centroids,axis=1); centroids/=np.maximum(norms[:,None],1e-12)
            pairs=set()
            # Bounded tag-block x all-tags matrix; never an all-fact-pairs allocation.
            for start in range(0,len(ids),block):
                sim=centroids[start:start+block] @ centroids.T
                for local,row in enumerate(sim):
                    i=start+local
                    for j in sorted((j for j in range(len(ids)) if j!=i),key=lambda j:(-row[j],ids[j]))[:graph_shortlist]:
                        pairs.add(tuple(sorted((i,j))))
            pairs=sorted(pairs)
        edge_key=lambda e:(-e['coverage_forward'],-e['bridge'],e['target'])
        for a,b in pairs:
            t,u=ids[a],ids[b]; ti,ui=index.members[t],index.members[u]
            edge=relation(index.X[ti],index.X[ui],index.weights[t],index.weights[u],block)
            wi,wj=edge.pop('witness_rows'); witness=[index.facts[ti[wi]]['fact_id'],index.facts[ui[wj]]['fact_id']]
            index.graph[t].append({'target':u,**edge,'witness':witness})
            index.graph[u].append({'target':t,**edge,'coverage_forward':edge['coverage_reverse'],
                                   'coverage_reverse':edge['coverage_forward'],'witness':witness[::-1]})
            # Exact mode visits all pairs but stores at most k+1 edges per tag.
            for tag in (t,u): index.graph[tag]=sorted(index.graph[tag],key=edge_key)[:graph_top_k]
        for t in index.tag_ids:
            index.graph[t]=sorted(index.graph[t],key=lambda e:(-e['coverage_forward'],-e['bridge'],e['target']))[:graph_top_k]
    return MatrixIndex(snapshot,embedding,index.graph,cfg)


def load_index(path):
    data=read_json(path)
    require(data.get('schema')=='eitr-matrix-index-1','Expected matrix index; rebuild from canonical snapshot')
    return MatrixIndex(data['snapshot'],data['embedding'],data['graph'],data['graph_config'],data['index_hash'])


def embed_snapshot(snapshot, backend, **options):
    validate_snapshot(snapshot); facts=sorted(snapshot['facts'],key=lambda f:f['fact_id'])
    return embed_record([f['text'] for f in facts],'document',
        {'snapshot_hash':snapshot['content_hash'],'fact_ids':[f['fact_id'] for f in facts]},backend,**options)


def compare_graphs(exact, approximate):
    require(exact.graph_config.get('mode')=='exact' and exact.graph_config['top_k']>0,
            'Reference graph must be exact with positive top-k')
    require(exact.snapshot['content_hash']==approximate.snapshot['content_hash'] and
            exact.embedding['record_hash']==approximate.embedding['record_hash'] and
            exact.graph_config['top_k']==approximate.graph_config['top_k'],
            'Graph comparison requires identical snapshot, embedding record and top-k')
    rows=[];total=hits=0
    for tag in exact.tag_ids:
        reference={e['target']:e for e in exact.graph[tag] if e['coverage_forward']>0}
        actual={e['target']:e for e in approximate.graph[tag] if e['coverage_forward']>0}
        shared=set(reference)&set(actual); total+=len(reference);hits+=len(shared)
        for target in shared:
            require(all(np.isclose(reference[target][k],actual[target][k],rtol=1e-10,atol=1e-12)
                        for k in ('average','coverage_forward','coverage_reverse','bridge')), 'Shared edge scores differ')
        rows.append({'tag_id':tag,'reference_positive_edges':len(reference),'recovered_edges':len(shared),
                     'recall':len(shared)/len(reference) if reference else None,
                     'missed_targets':sorted(set(reference)-set(actual)),
                     'missed_coverage_mass':sum(reference[t]['coverage_forward'] for t in set(reference)-set(actual))})
    return {'schema':'eitr-matrix-graph-comparison-1','exact_index_hash':exact.index_hash,
            'approximate_index_hash':approximate.index_hash,'positive_edge_recall':hits/total if total else None,
            'reference_positive_edges':total,'recovered_edges':hits,'rows':rows,
            'scope':'Directed top-k structural recall, not answer accuracy; zero edges excluded.'}
