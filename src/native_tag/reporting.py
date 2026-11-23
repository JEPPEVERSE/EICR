"""Auditable offline expenditure and descriptive mechanism diagnostics."""
from collections import Counter, defaultdict
import math
from eicr.contracts import require
from .experiment import sealed, verify, question_key, state_signature
from .transport import bill, validate_prices


def offline_costs(receipts, prices, rebuild_policy):
    validate_prices(prices)
    require(prices is not None and receipts, 'Offline costs require receipts and fixed prices')
    require(isinstance(rebuild_policy,dict) and rebuild_policy.get('trigger') and rebuild_policy.get('period'),
            'Declare snapshot rebuild trigger and reporting period')
    for field in ('expected_builds','queries_per_period'):
        value=rebuild_policy.get(field)
        require(value is None or (type(value) in (int,float) and math.isfinite(value) and value>0), 'Invalid rebuild projection')
    seen=set(); components=set(); rows=[]; assets=set(); calibrations=set()
    for receipt in receipts:
        verify(receipt)
        require(receipt['ledger_id'] not in seen,'Duplicate expenditure ledger; do not count the same attempts twice')
        seen.add(receipt['ledger_id'])
        kinds=set(receipt['components'])
        require(kinds and kinds <= {'embedding','index','teacher_calibration','shadow','judge','rubric_check'}, 'Invalid offline cost component')
        components.update(kinds)
        if receipt.get('asset_hash'): assets.add(receipt['asset_hash'])
        if receipt.get('calibration_hash'): calibrations.add(receipt['calibration_hash'])
        priced=bill(receipt['calls'],prices)
        # A running/crashed ledger may have unobserved completion or local work.
        if receipt['status'] not in ('complete','failed'):
            priced={**priced,'complete':False,'amount':None,'interrupted':True}
        rows.append({'ledger_id':receipt['ledger_id'],'receipt_hash':receipt['hash'],'components':sorted(kinds),
                     'status':receipt['status'],'billing':priced,'wall_seconds':receipt['wall_seconds'],
                     'cpu_seconds':receipt['cpu_seconds'],'measurements':receipt.get('measurements',[])})
    require({'embedding','index','teacher_calibration'} <= components, 'Account for embedding/index/calibration, including verified reuse')
    incremental=[r for r in rows if set(r['components']) & {'embedding','index','teacher_calibration','shadow'}]
    evaluation=[r for r in rows if set(r['components']) & {'judge','rubric_check'}]
    def total(values):
        return sum(r['billing']['amount'] for r in values) if all(r['billing']['complete'] for r in values) else None
    amount=total(incremental)
    builds,queries=(rebuild_policy.get(k) for k in ('expected_builds','queries_per_period'))
    recurring=amount*builds/queries if amount is not None and builds is not None and queries is not None else None
    return sealed({'schema':'native-tag-offline-costs-1','currency':prices['currency'],'prices':prices,'receipts':rows,
        'asset_hashes':sorted(assets),'calibration_hashes':sorted(calibrations),
        'incremental_api_cost':amount,'evaluation_api_cost':total(evaluation),
        'cpu_seconds':sum(r['cpu_seconds'] for r in rows),'wall_seconds':sum(r['wall_seconds'] for r in rows),
        'rebuild_policy':rebuild_policy,'projected_rebuild_cost_per_query':recurring,
        'projection_scope':'expected_builds applies to the entire incremental receipt bundle; no implied permanent reuse',
        'local_compute':'API costs exclude local CPU charges; measured CPU/wall are reported separately'})


def mechanism_report(protocol, runs, labels):
    label_map={question_key(q):q for q in labels}
    shadow_states=set((protocol.get('audit',{}).get('shadow_acceptance') or {}).get('states',[]))
    result={}
    for arm in protocol['arms']:
        rr=[r for r in runs if r['arm']==arm]; dd=[d for r in rr for d in r.get('decisions',[])]
        tool_counts=Counter(); repeated=0; event_ids=set(); source_ids=set(); recall=[]; measurements=defaultdict(list)
        for r in rr:
            scope=r['question']['owner']+'/'+r['question']['snapshot']+'/'
            event_ids.update(scope+eid for eid in r.get('visited_event_ids',[]))
            source_ids.update(scope+sid for sid in r.get('visited_source_ids',[])); returned=Counter()
            for e in r.get('events',[]):
                if e['method'] in ('event_by_tag','query_event_keywords','query_event_context','query_conversation_time',
                                   'query_topic_events','query_personal_information','query_personal_aspect','evaluate_relations_over_graph'):
                    tool_counts[e['method']]+=1
                    returned.update(e.get('returned_source_ids',[]))
            repeated+=sum(max(0,n-1) for n in returned.values())
            evidence=set(label_map[question_key(r['question'])].get('evidence',[]))
            if evidence: recall.append(len(evidence & set(r.get('visited_source_ids',[])))/len(evidence))
            for m in r.get('measurements',[]): measurements[m['stage']].append(m)
        stages={}
        for stage in ('initial','expansion'):
            ds=[d for d in dd if d['stage']==stage]
            stages[stage]={'calls':len(ds),'triggered_runs':sum(any(d['stage']==stage for d in r.get('decisions',[])) for r in rr),
                'candidates':[len(d['ordered_tag_ids']) for d in ds],
                'selected':[len(d.get('sets',{}).get('selected',[])) for d in ds],
                'zeros':[len(d.get('sets',{}).get('zero',[])) for d in ds],
                'early_reads':sum(len(d.get('sets',{}).get('early',[])) for d in ds),
                'fallback_fraction':sum(bool(d.get('fallback')) for d in ds)/len(ds) if ds else 0,
                'replaced_calls':sum(d.get('mode') in ('cheap_all','cheap_initial') and not d.get('fallback') and
                                     (d['mode']=='cheap_all' or stage=='initial') for d in ds)}
        result[arm]={'stages':stages,'tools':dict(tool_counts),'visited_event_ids':sorted(event_ids),
            'visited_source_ids':sorted(source_ids),'repeated_source_returns':repeated,
            'mean_annotated_source_recall':sum(recall)/len(recall) if recall else None,
            'source_recall_note':'diagnostic only; union of native read state and source IDs explicitly returned by tools',
            'teacher_shadow_states':sum(state_signature(d) in shadow_states for d in dd),
            'states_not_in_teacher_shadow':sum(state_signature(d) not in shadow_states for d in dd),
            'state_note':'exact scope/context/candidates/read-state match; different questions can produce entirely new states',
            'inclusive_phase_time':{stage:{k:sum(m[k] for m in values) for k in ('wall_seconds','cpu_seconds')}
                                    for stage,values in measurements.items()},
            'phase_time_note':'nested inclusive spans overlap; do not sum them to obtain request latency'}
    return result


def render_report(analysis):
    verify(analysis)
    def fmt(value): return 'unknown' if value is None else f'{value:.6g}'
    lines=['# Native tag selector experiment report','',f"Decision: **{analysis['decision']}**.",
           '',f"Protocol: `{analysis['protocol_hash']}`; grading: `{analysis['grades_hash']}`.",'',
           '| Arm | Strict accuracy | Mean API cost, all attempts | Failure rate | Mean wall seconds |',
           '|---|---:|---:|---:|---:|']
    for arm,row in analysis['aggregate'].items():
        lines.append('| '+arm+' | '+' | '.join(fmt(row[k]) for k in ('accuracy','cost','failure','wall_seconds'))+' |')
    lines.extend(['',f"Cost reduction: {fmt(analysis['cost_reduction'])}; symmetric quality delta bounds: {analysis['symmetric_quality_delta_bounds']}.",
        '',f"Incremental offline API cost: {fmt(analysis['offline_incremental_cost'])}; break-even queries: {fmt(analysis['break_even_queries'])}.",
        '',f"Projected saving per query after declared rebuilds: {fmt(analysis['saving_after_projected_rebuilds'])}.",
        '',f"Gates not satisfied: {', '.join(k for k,v in analysis['checks'].items() if not v) or 'none'}.",
        '',f"Independent source support assessments: {analysis['source_assessment']}.",
        '', 'The accompanying JSON retains per-question pairs, owner/category/triggered subsets, strict and lenient scores, failures, phase costs, CPU/HTTP timing and mechanism diagnostics.',
        '', 'Unresolved source assessments are reported separately from answer correctness. Offline unknowns are not zero. Rebuild projections use the explicitly declared receipt bundle and period.',
        '',analysis['claim'], ''])
    return '\n'.join(lines)
