"""Controlled MRAgent P5: pinned native loop/bridge, common evidence contracts.

Initial extraction/ranking, raw tool outputs, fixed context and final answer are
controlled as specified by the experiment. Native loop code and all seven tool
schemas are loaded unchanged from the pinned upstream checkout. This is not M0.
"""
from collections import defaultdict
from copy import deepcopy
import json
from time import perf_counter
from types import SimpleNamespace
from eicr.contracts import ContractError, digest, require
from eicr.pipeline import answer_from_sources
from eicr.retrieval import SharedRetriever
from .config import Config
from .evidence import Corpus
from .query import map_groups, mapping_config_hash
from .candidates import enumerate_graph, candidate_pool
from .scheduler import Scheduler
from .accounting import Meter


class ControlledController:
    def __init__(self, index, query, corpus, config):
        self.index, self.query, self.corpus, self.config = index, query, corpus, config
        self.whole = corpus.scores(query['question'])
        graph = enumerate_graph(index, query, config, expand=False)
        self.pool = candidate_pool(index, query, graph, corpus, self.whole, config, 'seed')
        self.state = Scheduler(corpus, self.pool, self.whole, [], config)
        self.state.greedy([a for a in self.state.actions if a['direct']], 'direct', cap=int(config.evidence_budget*config.direct_fraction))
        self.state.seed_round_robin(int(config.evidence_budget*config.seed_fraction))
        self.facts = {f['fact_id']:f for f in index.snapshot['facts']}
        self.cues = {c['cue_id']:c['text'] for c in index.snapshot['cues']}
        self.tags = {t['tag_id']:t['text'] for t in index.snapshot['tags']}
        self.pairs, self.fact_keys, self.key_tags = defaultdict(set), defaultdict(set), defaultdict(set)
        for link in index.snapshot['links']:
            key, tag, fid = link['cue_id'], link['tag_id'], link['fact_id']
            self.pairs[(self.cues[key], self.tags[tag])].add(fid)
            self.fact_keys[fid].add(key); self.key_tags[key].add(tag)
        self.aux = index.snapshot['native_auxiliary']
        self.candidate_atoms = set(self.pool['atom_ids'])
        self.core_ids = {a['core'] for a in self.pool['actions']}
        self.reads = []

    def visible_sources(self):
        return {self.corpus.atoms[a]['source_id'] for a in self.state.selected}

    def event_origins(self, event_id):
        origins = set(self.facts[event_id]['origins']) if event_id in self.facts else {event_id}
        require(origins <= self.visible_sources(), 'Event must already be visible')
        return origins

    def key_candidates(self, keys):
        result = []
        for key in sorted(set(keys)):
            tags = sorted(self.key_tags[key])
            ranker = SharedRetriever([{'span_id':t, 'text':self.tags[t]} for t in tags])
            scores = dict(zip(tags, ranker.lexical_scores(self.query['question'])))
            ordered = sorted(tags, key=lambda t:(-scores[t],t))[:self.config.max_tags_per_key]
            result.append({'key':self.cues[key], 'tags':[self.tags[t] for t in ordered]})
        return result

    def read_origins(self, origins, operation):
        require(set(origins) <= set(self.corpus.sources), 'Tool read crosses snapshot')
        ids = {a for o in origins for a in self.corpus.source_atoms[o]}
        order = sorted(ids, key=lambda a:(-self.whole[a],a))
        proposed = order[:self.config.max_core_atoms_per_tag]
        omitted = [{'core':a, 'reason':'per_tool_core_cap'} for a in order[len(proposed):]]
        actions = []
        for a in proposed:
            ctx = set(self.corpus.contexts[a])
            if len(self.core_ids | {a}) > self.config.max_core_atoms or len(self.candidate_atoms | ctx) > self.config.max_candidate_atoms:
                omitted.append({'core':a,'reason':'whole_action_candidate_cap'}); continue
            self.core_ids.add(a); self.candidate_atoms.update(ctx)
            actions.append({'action_id':a, 'core':a, 'atoms':sorted(ctx), 'tags':[], 'support':0., 'direct':False, 'seed_groups':[]})
        previous = set(self.state.selected)
        self.state.phase = operation
        while self.state.can_continue():
            choice = self.state.best(actions, 1., False, allow_zero=True)
            if choice is None: break
            self.state.commit(choice, operation)
        fresh = self.state.selected-previous
        self.reads.append({'operation':operation, 'requested_sources':sorted(origins), 'candidate_cores':[a['core'] for a in actions],
                           'omitted':omitted, 'fresh_atoms':sorted(fresh), 'packet_tokens':self.state.packet['used_tokens']})
        return {'raw_sources':self.corpus.render(fresh)['text'], 'visible_event_ids':sorted(self.visible_sources()),
                'packet_tokens':self.state.packet['used_tokens'], 'budget':self.config.evidence_budget,
                'read_status':'added' if fresh else 'no_new_fitting_evidence'}

    # Signatures and tuple returns match the actual upstream ToolBridge dispatch.
    def event_by_tag(self, key, tag, note):
        require(isinstance(note,str), 'Invalid note')
        require((key,tag) in self.pairs, 'Unknown native key/tag pair')
        fids = self.pairs[(key,tag)]
        origins = {o for f in fids for o in self.facts[f]['origins']}
        return self.read_origins(origins, 'edges_by_tag'), sorted(origins), sorted(fids)

    def query_conversation_time(self, event_id):
        origins = self.event_origins(event_id)
        return {'conversation_times':[{'source_id':o, 'session_datetime':self.corpus.sources[o].get('session_datetime'),
                                      'provenance':self.corpus.sources[o].get('datetime_provenance')} for o in sorted(origins)]}

    def query_event_keywords(self, event_id):
        origins = self.event_origins(event_id)
        keys = {key for f in self.facts.values() if set(f['origins']) <= origins for key in self.fact_keys[f['fact_id']]}
        return self.key_candidates(keys)

    def query_event_context(self, event_id):
        origins = self.event_origins(event_id)
        # Exactly the shared fixed atom context, never native unbounded turn text.
        return self.read_origins(origins, 'query_event_context'), sorted(origins)

    def query_topic_events(self, topic):
        node = next((t for t in self.aux['topics'] if t['topic_id']==topic),None)
        require(node is not None, 'Unknown native topic')
        origins = {o for f in node['fact_ids'] for o in self.facts[f]['origins']}
        return self.read_origins(origins, 'query_topic_events'), sorted(origins)

    def query_personal_information(self, person):
        aspects = sorted({p['aspect'] for p in self.aux['personal'] if p['person']==person})
        require(aspects, 'Unknown native person')
        return {'person':person,'aspects':aspects}

    def query_personal_aspect(self, person, aspect):
        nodes = [p for p in self.aux['personal'] if p['person']==person and p['aspect']==aspect]
        require(nodes, 'Unknown native personal aspect')
        origins = {o for p in nodes for o in p['origins']}
        return self.read_origins(origins, 'query_personal_aspect'), sorted(origins)


def run_controlled(index, question, tokenizer, runtime, backend, *, config=None, mentions=None,
                   frontend_backend=None, answer_backend=None, query=None, corpus=None):
    config = config or Config()
    meter = Meter()
    require(runtime is not None and backend is not None, 'P5 requires pinned MRAgent runtime and tool backend')
    require('native_auxiliary' in index.snapshot, 'P5 needs native topic/personal graph; reimport original pack')
    require(config.top_k==index.top_k, 'P5 index/config mismatch')
    start = perf_counter()
    corpus = corpus or Corpus(index.snapshot, tokenizer, config)
    require(corpus.snapshot_hash==index.snapshot['content_hash'] and corpus.tokenizer.identity==tokenizer.identity
            and corpus.atom_config==(config.atom_tokens,config.context_each_side), 'P5 corpus mismatch')
    shared = query is not None
    if query is None: query = map_groups(index, question, config, mentions, frontend_backend)
    require(query['index_hash']==index.index_hash and query['question']==question, 'P5 query mismatch')
    require(query.get('mapping_config_hash')==mapping_config_hash(config),'P5 frontend config mismatch')
    ctrl = ControlledController(index, query, corpus, config)
    bridge = runtime.bridge(ctrl)
    topics = ctrl.aux['topics']
    ranker = SharedRetriever([{'span_id':t['topic_id'], 'text':t['text']} for t in topics])
    topic_order = sorted(zip(topics,ranker.lexical_scores(question)), key=lambda x:(-x[1],x[0]['topic_id']))[:config.max_seed_tags]
    user = {'question':question, 'key_sentences':ctrl.state.packet['text'],
            'keys_candidates':[{'key':ctrl.cues[k], 'tags':sorted({ctrl.tags[t] for g in query['groups']
                                if k in g['key_ids'] for t in g['seed_tags'] if t in ctrl.key_tags[k]})}
                               for k in sorted({k for g in query['groups'] for k in g['key_ids']})],
            'key_tag_sentences':[], 'similar_topic':[t['topic_id']+':'+t['text'] for t,s in topic_order],
            'visible_event_ids':sorted(ctrl.visible_sources())}
    calls_start = len(backend.calls)
    messages_log, tool_log = [], []
    tool_limit = config.explorer_rounds*config.explorer_actions_per_round
    used = 0
    all_tool_ids = set()
    class NativeTransport:
        def chat_with_tool(self, *, messages, tools, **kwargs):
            nonlocal used
            tick=perf_counter()
            msg = backend.chat_tools(deepcopy(messages), deepcopy(tools))
            messages_log.append({'messages':deepcopy(messages), 'response':deepcopy(msg), 'seconds':perf_counter()-tick})
            requested = msg.get('tool_calls') or []
            require(len(requested) <= config.explorer_actions_per_round and used+len(requested)<=tool_limit,
                    'Native requested tool batch exceeds controlled budget')
            # Validate before the upstream loop can truncate or execute any call.
            allowed = {t['function']['name']:t['function']['parameters'] for t in tools}
            ids=set()
            for call in requested:
                require(isinstance(call,dict) and isinstance(call.get('id'),str) and call['id']
                        and call['id'] not in ids and call['id'] not in all_tool_ids, 'Invalid native tool identity')
                ids.add(call['id'])
                fun=call.get('function',{}); name=fun.get('name')
                require(name in allowed, 'Unknown native tool')
                args=json.loads(fun.get('arguments',''))
                schema=allowed[name]
                require(isinstance(args,dict) and set(schema.get('required',[]))<=set(args)
                        and set(args)<=set(schema['properties']) and all(isinstance(v,str) for v in args.values()), 'Invalid native tool arguments')
                for k,v in args.items():
                    rule=schema['properties'][k]
                    require(len(v)>=rule.get('minLength',0) and len(v)<=rule.get('maxLength',len(v)), 'Invalid native tool string length')
            all_tool_ids.update(ids)
            used += len(requested)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda **_:deepcopy(msg)))])
    def execute(calls):
        results = bridge.call(calls)
        tool_log.append({'calls':deepcopy(calls),'results':deepcopy(results)})
        return results
    status, error = 'native_loop_completed', None
    try:
        native_answer, supports = runtime.loop(NativeTransport(), system_prompt=runtime.prompts.ANSWER_SYSTEM_TOOL_PROMPT,
            user_obj=user, tools=runtime.tools, category=0, execute_tool=execute, max_rounds=config.explorer_rounds,
            max_tool_calls=tool_limit, model=getattr(backend,'model','controlled'), temperature=0.)
        if native_answer is None:
            status = 'native_round_cap_without_answer'
    except (ContractError, ValueError, TypeError, KeyError) as exc:
        native_answer, supports = None, []
        status, error = 'native_loop_error', type(exc).__name__
    packet = {**ctrl.state.packet, 'budget_tokens':config.evidence_budget, 'unused_tokens':config.evidence_budget-ctrl.state.packet['used_tokens'],
              'ledger':ctrl.state.trace,'alternatives':ctrl.state.alternatives,'action_evaluations':ctrl.state.evaluations,
              'successful_actions':ctrl.state.successful, 'render_seconds':ctrl.state.render_seconds}
    require(packet['used_tokens']<=config.evidence_budget, 'P5 budget contract violated')
    retrieval_seconds = perf_counter()-start
    answer_start=len(answer_backend.calls) if answer_backend else 0
    answer_tick=perf_counter()
    answer=answer_from_sources(question,packet,answer_backend) if answer_backend else None
    from .pipeline import runtime_source_hash
    return {'schema':'eitr-run-1','arm':'P5','question':question,'query':query,'index_hash':index.index_hash,
            'snapshot_hash':index.snapshot['content_hash'],'evidence_pack':packet,'answer':answer,
            'native_loop_result':{'answer_discarded':native_answer, 'supports_discarded':supports,'status':status,'error_type':error},
            'native_messages':messages_log,'native_tools':tool_log,'tool_reads':ctrl.reads,
            'candidates':{'atom_ids':sorted(ctrl.candidate_atoms), 'source_stages':{'candidate':sorted({corpus.atoms[a]['source_id'] for a in ctrl.candidate_atoms})}},
            'calls':{'frontend':query['frontend_calls'],'exploration':list(backend.calls[calls_start:]),
                     'answer':list(answer_backend.calls[answer_start:]) if answer_backend else []},
            'timing_seconds':{'retrieval_total':retrieval_seconds,'answer':perf_counter()-answer_tick,'end_to_end':perf_counter()-start},
            'resources':meter.result(),
            'manifest':{'status':'answered' if answer else 'retrieval_only','config':config.to_dict(),'config_hash':digest(config.to_dict()),
                        'tokenizer':tokenizer.identity,'corpus_hash':corpus.identity,'frontend_shared':shared,
                        'baseline_identity':'controlled_MRAgent_pinned_native_loop_and_seven_tools',
                        'native_runtime_hash':runtime.identity,'native_runtime':runtime.manifest,'runtime_source_hash':runtime_source_hash(),
                        'controlled_changes':['shared cue frontend and cheap tag/topic ranker','verified raw sources and bounded common atom context',
                          'same reserves and rendered evidence budget','common final answerer; native stop answer discarded but charged',
                          'fixed generic native category=0; no gold question categories','no native dense/LLM initial reranking'],
                        'not_native_M0':True}}
