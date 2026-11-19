from copy import deepcopy
import pytest
from eicr.contracts import digest
from eicr.packer import CharacterTokenizer
from eitr.index import build_index
from eitr.config import Config
from eitr.query import map_groups
from eitr.candidates import enumerate_graph,candidate_pool
from eitr.evidence import Corpus
from eitr.scheduler import Scheduler,build_interaction_matrix,coverage_gain


def make(snapshot, **kwargs):
    config=Config(context_each_side=0, direct_fraction=0,seed_fraction=0,**kwargs)
    index=build_index(snapshot); corpus=Corpus(snapshot,CharacterTokenizer(),config)
    query=map_groups(index,'herbs halloumi shop train',config)
    pool=candidate_pool(index,query,enumerate_graph(index,query,config),corpus,corpus.scores(query['question']),config)
    ids={a['source_id']:a['atom_id'] for a in corpus.atoms.values()}
    return config,corpus,pool,ids


def test_true_coverage_marginal():
    assert coverage_gain([.9],[.8])==0
    assert coverage_gain([.2],[.8])==pytest.approx(.6)
    assert coverage_gain([1.],[1.])==0
    assert coverage_gain([],[])==0


def test_interaction_matrix_is_bounded_and_normalized(snapshot):
    config, _, _, _ = make(snapshot)
    index=build_index(snapshot)
    query=map_groups(index,'herbs halloumi shop train',config)
    matrix=build_interaction_matrix(index,query,config.interaction_top_edges)
    assert len(matrix)==len(query['groups'])
    assert all(len(row)==len(matrix) and all(0<=v<=1 for v in row) for row in matrix)
    assert all(sum(row) <= 1 + 1e-9 for row in matrix)


def test_active_interaction_rewards_complement(snapshot):
    config,corpus,pool,ids=make(snapshot,evidence_budget=2000)
    q={a:.5 for a in corpus.atoms}
    left={a:float(a==ids['D1:1']) for a in q}
    right={a:float(a==ids['D1:2']) for a in q}
    state=Scheduler(corpus,pool,q,[left,right],config,interaction_matrix=[[0.,.4],[.4,0.]])
    state.active_interaction=True
    state.phase='shared'
    actions={a['core']:a for a in pool['actions']}
    def choose(source):
        return state.evaluate(actions[ids[source]],config.eta,False)
    assert choose('D1:2')['interaction']==0
    state.commit(choose('D1:1'),'shared')
    complement=choose('D1:2')
    assert complement['cov']==.5
    assert complement['interaction']==pytest.approx(.2)
    assert complement['committee_gain']==pytest.approx(.5 + .2*config.interaction_weight)
    assert complement['support_factor']==1


def test_shared_seed_and_unmapped_groups_cannot_vote_for_each_other():
    from types import SimpleNamespace
    index=SimpleNamespace(neighbors={'a':[{'target':'b','score':.3}], 'b':[], 'shared':[]})
    groups=[{'seed_tags':['shared','a']}, {'seed_tags':['shared','b']},
            {'seed_tags':['shared','a']}, {'seed_tags':[], 'candidate_tags':['b']}]
    matrix=build_interaction_matrix(index,{'groups':groups})
    assert matrix[0]==[0.,.3,0.,0.]
    assert matrix[3]==[0.]*4


def test_active_replay_and_zero_weight_control(snapshot):
    from eitr.pipeline import compare
    from eitr.experiment import replay
    cfg=Config(interaction_weight=0.,evidence_budget=700)
    idx=build_index(snapshot); tok=CharacterTokenizer(); corpus=Corpus(snapshot,tok,cfg)
    runs=compare(idx,'herbs halloumi shop',tok,config=cfg,corpus=corpus,
                 arms=['P4_no_support','P4_active'])['rows']
    assert runs[0]['evidence_pack']['atom_ids']==runs[1]['evidence_pack']['atom_ids']
    assert replay(runs[1],idx,corpus)['verified']


def test_state_feedback_prefers_complement(snapshot):
    config,corpus,pool,ids=make(snapshot,evidence_budget=2000)
    whole={a:.5 for a in corpus.atoms}
    cue1={a:0. for a in corpus.atoms};cue2=dict(cue1)
    cue1[ids['D1:1']]=1.;cue1[ids['D1:10']]=.8;cue2[ids['D1:2']]=1.
    state=Scheduler(corpus,pool,whole,[cue1,cue2],config)
    actions={a['core']:a for a in pool['actions']}
    state.commit(state.evaluate(actions[ids['D1:1']],.5,False),'test')
    repeat=state.evaluate(actions[ids['D1:10']],.5,False)
    complement=state.evaluate(actions[ids['D1:2']],.5,False)
    assert repeat['cov']==0
    assert repeat['rel']>0  # Never shut down all evidence sharing a covered cue.
    assert complement['cov']==.5
    assert complement['priority']>repeat['priority']


def test_overlap_and_budget_count_full_render(snapshot):
    config=Config(evidence_budget=650)
    index=build_index(snapshot);corpus=Corpus(snapshot,CharacterTokenizer(),config)
    query=map_groups(index,'herbs halloumi shop',config)
    scores=corpus.scores(query['question'])
    pool=candidate_pool(index,query,enumerate_graph(index,query,config),corpus,scores,config)
    state=Scheduler(corpus,pool,scores,[scores],config)
    packet=state.select('P4',True)
    fresh=[a for step in packet['ledger'] for a in step['fresh_atoms']]
    assert len(fresh)==len(set(fresh))
    assert packet['used_tokens']==len(packet['text'])<=650
    assert sum(step['cost'] for step in packet['ledger'])==packet['used_tokens']


def test_unaffordable_does_not_hide_small_action(snapshot):
    config,corpus,pool,ids=make(snapshot,evidence_budget=250)
    q={a:1. for a in corpus.atoms}
    state=Scheduler(corpus,pool,q,[],config)
    packet=state.select('P3',True)
    assert packet['atom_ids']
    assert packet['used_tokens']<=250


def test_compute_limit_includes_reserves(snapshot):
    config,corpus,pool,ids=make(snapshot,evidence_budget=2000,max_actions=1,max_score_evaluations=2)
    q={a:1. for a in corpus.atoms}
    state=Scheduler(corpus,pool,q,[],config)
    packet=state.select('P3',True)
    assert packet['successful_actions']<=1
    assert packet['action_evaluations']<=2


def test_full_coverage_does_not_stop_related_evidence(snapshot):
    config,corpus,pool,ids=make(snapshot,evidence_budget=2000)
    q={a:1. for a in corpus.atoms}
    state=Scheduler(corpus,pool,q,[q],config)
    packet=state.select('P4',True)
    assert packet['coverage']==[1.]
    assert packet['successful_actions']>1
