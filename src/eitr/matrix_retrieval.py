"""Query-conditioned fact/tag matching, bounded routing and exact source packing."""
from dataclasses import dataclass, asdict, fields
import numpy as np
from eicr.contracts import digest, require
from .matrix_embeddings import unit_rows

ARMS=('direct_rerank','matrix_rerank','matrix_graph_rerank','matrix_embedding')


@dataclass(frozen=True)
class MatrixConfig:
    atom_tokens: int = 128
    context_each_side: int = 1
    evidence_budget: int = 2000
    max_requirements: int = 8
    tags_per_requirement: int = 8
    facts_per_tag: int = 16
    direct_facts: int = 32
    max_core_atoms: int = 200
    max_candidate_atoms: int = 480
    graph_hops: int = 1
    graph_tags_per_requirement: int = 8
    max_actions: int = 128
    max_score_evaluations: int = 100000

    def __post_init__(self):
        for f in fields(self):
            v=getattr(self,f.name)
            require(type(v) is int and v >= (0 if f.name in ('context_each_side','graph_hops') else 1),'Invalid matrix config: '+f.name)
        require(self.context_each_side<=1 and self.graph_hops<=2,'Context/hops exceed audited bounds')
        require(self.max_core_atoms<=self.max_candidate_atoms<=500 and self.max_requirements<=8,'Invalid matrix caps')

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls,data):
        require(isinstance(data,dict) and set(data)<={f.name for f in fields(cls)},'Unknown matrix settings')
        return cls(**data)


def match(index, vectors, *, with_tags=True):
    q=unit_rows(vectors,len(vectors),index.X.shape[1])
    a=np.clip(q @ index.X.T,0,1)
    tag_ids=index.tag_ids if with_tags else []
    r=np.zeros((len(q),len(tag_ids))); witnesses=[]
    for col,t in enumerate(tag_ids):
        ids=index.members[t]
        if not len(ids): witnesses.append([None]*len(q));continue
        loc=np.argmax(a[:,ids],axis=1); best=ids[loc]
        r[:,col]=a[np.arange(len(q)),best]
        witnesses.append([index.facts[i]['fact_id'] for i in best])
    return a,r,{'tag_ids':tag_ids,'scores':r.tolist(),
                'witness_fact_ids':np.array(witnesses,dtype=object).T.tolist() if witnesses else [[] for _ in q],
                'rows':'original question, then plan requirements','score_semantics':'nonnegative cosine maximum; not probability'}


def round_robin(queues):
    positions=[0]*len(queues)
    while True:
        changed=False
        for j,queue in enumerate(queues):
            if positions[j]<len(queue):
                yield j,queue[positions[j]]; positions[j]+=1; changed=True
        if not changed: break


def candidates(index,corpus,question_vectors,config,arm):
    require(arm in ARMS,'Unknown matrix arm')
    require(corpus.snapshot_hash==index.snapshot['content_hash'] and
            corpus.atom_config==(config.atom_tokens,config.context_each_side),'Matrix corpus mismatch')
    if arm=='matrix_graph_rerank': require(index.graph_config['top_k']>0 and config.graph_hops>0,'Graph arm needs built graph and positive hops')
    a,r,matching=match(index,question_vectors,with_tags=arm!='direct_rerank')
    n=len(a); tagpos={t:i for i,t in enumerate(index.tag_ids)}
    fact_order=lambda row,ids: sorted(ids,key=lambda i:(-a[row,i],index.facts[i]['fact_id']))
    # The standalone dense control gets the full core cap, not the smaller rescue-channel cap.
    direct_limit=config.max_core_atoms if arm=='direct_rerank' else config.direct_facts
    direct=[fact_order(row,range(len(index.facts)))[:direct_limit] for row in range(n)]
    routed=[]; tag_routes=[]; graph_routes=[]
    for row in range(n):
        if arm=='direct_rerank':
            routed.append([]);tag_routes.append([]);graph_routes.append([]);continue
        tags=sorted((t for t in index.tag_ids if len(index.members[t]) and r[row,tagpos[t]]>0),
                    key=lambda t:(-r[row,tagpos[t]],t))[:config.tags_per_requirement]
        extra=[]; seen=set(tags); frontier=tags
        if arm=='matrix_graph_rerank':
            for hop in range(config.graph_hops):
                scores={}
                for t in frontier:
                    for e in index.graph[t]:
                        u=e['target']
                        if u not in seen:
                            # Positive edges may bridge a query-orthogonal tag. Keep the
                            # original query*edge ordering; use edge strength for zero-score
                            # ties, never promote disconnected zero-weight edges.
                            strength=float(e['coverage_forward'])
                            if strength > 0:
                                scores[u]=max(scores.get(u,(0.,0.)),
                                              (r[row,tagpos[u]]*strength,strength))
                frontier=sorted(scores,key=lambda t:(-scores[t][0],-scores[t][1],t))[:config.graph_tags_per_requirement]
                extra.extend(frontier);seen.update(frontier)
        tag_routes.append(tags);graph_routes.append(extra)
        queues=[fact_order(row,index.members[t])[:config.facts_per_tag] for t in tags+extra]
        routed.append(list(dict.fromkeys(i for _,i in round_robin(queues))))
    # Source-level provenance is not a fact-to-character alignment. Keep every raw
    # atom as an eligible core, ordered round-robin across facts and origins.
    def atoms_for(fids,row,allow_zero=False):
        lists=[]
        for i in fids:
            if a[row,i]<=0 and not allow_zero: continue
            origins=index.facts[i]['origins']
            lists.append(list(dict.fromkeys(aid for _,aid in round_robin([corpus.source_atoms[s] for s in origins]))))
        return list(dict.fromkeys(aid for _,aid in round_robin(lists)))
    queues=[]
    for row in range(n):
        queues.append(atoms_for(direct[row],row))
        if arm!='direct_rerank': queues.append(atoms_for(routed[row],row,allow_zero=arm=='matrix_graph_rerank'))
    all_before={aid for queue in queues for aid in queue}
    admitted=set(); actions=[]; seen_core=set(); dropped=0
    for _,core in round_robin(queues):
        if core in seen_core: continue
        seen_core.add(core)
        context=set(corpus.contexts[core])
        if len(actions)>=config.max_core_atoms or len(admitted|context)>config.max_candidate_atoms:
            dropped+=1;continue
        admitted|=context
        actions.append({'action_id':core,'core':core,'atoms':sorted(context)})
    scores=[dict.fromkeys(admitted,0.) for _ in range(n)]
    # Carry every fact mapped to an admitted raw source, not only the first witness.
    for i,f in enumerate(index.facts):
        for sid in f['origins']:
            for aid in corpus.source_atoms[sid]:
                if aid in admitted:
                    for row in range(n): scores[row][aid]=max(scores[row][aid],float(a[row,i]))
    source_ids=lambda ids: sorted({corpus.atoms[x]['source_id'] for x in ids})
    pool={'schema':'eitr-matrix-candidates-1','corpus_hash':corpus.identity,'config_hash':digest(config.to_dict()),
          'arm':arm,'actions':actions,'atom_ids':sorted(admitted),'tag_routes':tag_routes,'graph_routes':graph_routes,
          'source_stages':{'candidate':source_ids(admitted),'before_global_cap':source_ids({aid for core in all_before for aid in corpus.contexts[core]}),
                           'direct':source_ids({aid for row,fids in enumerate(direct) for core in atoms_for(fids,row) for aid in corpus.contexts[core]})},
          'dropped_cores':dropped,'raw_alignment':'fact origins -> all source atoms; no fabricated span alignment'}
    pool['pool_hash']=digest(pool)
    return pool,scores,matching


def select(corpus,pool,scores,requirement_scores,config):
    """Fixed one-action-per-requirement pass, then relevance density.

    No committee or adaptive importance budgets. Requirements only order
    the initial pass; all arms share exactly the same selection rule.
    """
    ids=set(pool['atom_ids'])
    require(pool['pool_hash']==digest({k:v for k,v in pool.items() if k!='pool_hash'}) and
            pool['corpus_hash']==corpus.identity and pool['config_hash']==digest(config.to_dict()),'Invalid matrix pool')
    for values in [scores]+requirement_scores:
        require(set(values)==ids and all(type(v) in (float,int) and np.isfinite(v) and 0<=v<=1 for v in values.values()),'Invalid packing scores')
    for action in pool['actions']:
        require(action['action_id']==action['core'] and set(action['atoms'])==set(corpus.contexts[action['core']]) and
                set(action['atoms'])<=ids,'Incomplete action context')
    selected=set(); packet=corpus.render(selected); trace=[]; evaluations=0; stop='exhausted'
    def best(values,eligibility=None):
        nonlocal evaluations
        winner=None
        for action in pool['actions']:
            if evaluations>=config.max_score_evaluations: break
            fresh=set(action['atoms'])-selected
            if not fresh or (eligibility is not None and not any(eligibility[a]>0 for a in fresh)): continue
            evaluations+=1; proposed=corpus.render(selected|fresh)
            if proposed['used_tokens']>config.evidence_budget: continue
            cost=proposed['used_tokens']-packet['used_tokens']
            gain=sum(corpus.atoms[a]['text_tokens']*values[a] for a in sorted(fresh))
            if gain<=0: continue
            key=(-gain/max(1,cost),-gain,action['core'])
            if winner is None or key<winner[0]: winner=(key,action,fresh,proposed,cost)
        return winner
    def commit(choice,phase):
        nonlocal packet
        _,action,fresh,packet,cost=choice; selected.update(fresh)
        trace.append({'action_id':action['core'],'phase':phase,'fresh_atoms':sorted(fresh),
                      'cost':cost,'packet_tokens':packet['used_tokens']})
    for j,values in enumerate(requirement_scores):
        if len(trace)>=config.max_actions or evaluations>=config.max_score_evaluations: break
        # Require both reranker relevance and requirement compatibility; multiplication
        # is a ranking heuristic, never a calibrated probability.
        choice=best({a:scores[a]*values[a] for a in ids},values)
        if choice: commit(choice,'requirement:'+str(j))
    while len(trace)<config.max_actions and evaluations<config.max_score_evaluations:
        choice=best(scores)
        if not choice: break
        commit(choice,'relevance_density')
    if evaluations>=config.max_score_evaluations: stop='evaluation_cap'
    elif len(trace)>=config.max_actions: stop='action_cap'
    require(packet==corpus.render(selected) and packet['used_tokens']<=config.evidence_budget,'Matrix packing budget failure')
    return {**packet,'budget_tokens':config.evidence_budget,'unused_tokens':config.evidence_budget-packet['used_tokens'],
            'ledger':trace,'action_evaluations':evaluations,'stop':stop,'pool_hash':pool['pool_hash'],
            'requirement_max_similarity':[max((v[a] for a in selected),default=0.) for v in requirement_scores],
            'coverage_is_sufficiency':False}
