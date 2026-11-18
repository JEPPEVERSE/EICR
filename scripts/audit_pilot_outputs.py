#!/usr/bin/env python3
"""Post-hoc output audit: retain primary failures; score cached raw answer content.

Never regenerate answers. An exactly rendered header in source_id can be repaired
without guessing; all other citation errors stay invalid. This is supplementary,
not a replacement for the preregistered strict end-to-end metric.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from eicr.contracts import ContractError, digest, read_json, require, write_json
from eicr.pipeline import answer_from_sources
from run_development_pilot import ARMS, JUDGE_PROMPT, CachedBackend, load_key, parallel


def repair_exact_header_ids(answer, packet):
    result = {**answer, 'citations': [dict(c) for c in answer.get('citations', [])]}
    repairs = 0
    for citation in result['citations']:
        if any(citation['source_id'] == s['source_id'] for s in packet['spans']):
            continue
        matches = [s for s in packet['spans'] if citation['source_id'] == '{}:{}-{}'.format(s['source_id'], s['start'], s['end'])]
        if len(matches) == 1:
            citation['source_id'] = matches[0]['source_id']
            repairs += 1
    return result, repairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    protocol = read_json(out/'protocol.json')
    data = args.workspace/'outputs/entry_dataset_v41flash_20261110/frozen_v0.1'
    rows = [json.loads(line) for line in (data/'evaluation_inputs.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['example_id'] in protocol['evaluation_ids']]
    require(len(rows) == 39 and all((out/'answers'/(r['example_id']+'.json')).exists() for r in rows), 'Finish all frozen answers before this audit')
    labels = {r['unit_id']: r for r in read_json(data/'dataset.json')['items']}
    cache = [read_json(f) for f in (out/'requests').glob('*.json')]
    by_prompt = {r['prompt_hash']: r for r in cache}
    by_key = {r['request_key']: r for r in cache}
    load_key(args.workspace/'main_lab/.env')
    def audit(row):
        qid = row['example_id']
        path = out/'output_audit'/(qid+'.json')
        if path.exists():
            return
        label = labels[row['unit_id']]
        answers = read_json(out/'answers'/(qid+'.json'))['arms']
        packets = read_json(out/'retrieval'/(qid+'.json'))['arms']
        primary = read_json(out/'judgments'/(qid+'.json'))['arms']
        results, seen = {}, {}
        for arm in ARMS:
            stored = answers[arm]
            raw = stored['answer']
            if raw is None:
                trace = stored['calls'][-1]
                record = by_key[trace['request_key']] if trace.get('cache_hit') else by_prompt[trace['prompt_hash']]
                raw = record['response']
            key = digest(raw)
            if key not in seen:
                if stored['answer'] is not None:
                    judged = primary[arm]
                else:
                    payload = {'question': row['question'], 'premise_perturbed': row['condition'] != 'OG',
                        'correct_slot': label['slot_original'], 'target_event': label['target_event'],
                        'required_answer_points': label['required_answer_points'], 'gold_answer': label['gold_answer'],
                        'reference_sources': [{'source_id': e['dia_id'], 'text': e['text']} for e in label['evidence_records']],
                        'candidate_answer': raw}
                    judged = CachedBackend(out, 'judge_raw_output_audit').chat_json(JUDGE_PROMPT, json.dumps(payload, ensure_ascii=False))
                    require(type(judged.get('answer_correct')) is bool, 'Invalid judge output')
                seen[key] = judged
            packet = packets[arm]['evidence_pack']
            repaired, n_repairs = repair_exact_header_ids(raw, packet)
            class StaticBackend:
                def chat_json(self, system, user):
                    return repaired
            try:
                validated = answer_from_sources(row['question'], packet, StaticBackend())
                repair_valid, remaining_error = True, None
            except ContractError as error:
                repair_valid, remaining_error = False, str(error)
            results[arm] = {'raw_answer': raw, 'primary_output_valid': stored['error'] is None,
                'header_id_repairs': n_repairs, 'valid_after_exact_header_repair': repair_valid,
                'remaining_error': remaining_error, 'semantic_judgment': seen[key],
                'semantic_correct': seen[key]['answer_correct'],
                'repaired_end_to_end_correct': repair_valid and seen[key]['answer_correct']}
        write_json(path, {'example_id': qid, 'arms': results})
    parallel(rows, audit)
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {}
        for cond in ['OG','P0','P1']:
            group = [read_json(out/'output_audit'/(r['example_id']+'.json'))['arms'][arm] for r in rows if r['condition'] == cond]
            metrics[arm][cond] = {'n': len(group), 'raw_valid': sum(r['primary_output_valid'] for r in group),
                'semantic_correct': sum(r['semantic_correct'] for r in group),
                'valid_after_exact_header_repair': sum(r['valid_after_exact_header_repair'] for r in group),
                'repaired_end_to_end_correct': sum(r['repaired_end_to_end_correct'] for r in group)}
    paired = {}
    for arm in ARMS[1:]:
        paired[arm] = {}
        for cond in ['OG','P0','P1']:
            group = [read_json(out/'output_audit'/(r['example_id']+'.json'))['arms'] for r in rows if r['condition'] == cond]
            paired[arm][cond] = {'wins': sum(g[arm]['semantic_correct'] and not g['original']['semantic_correct'] for g in group),
                'losses': sum(not g[arm]['semantic_correct'] and g['original']['semantic_correct'] for g in group)}
    result = {'status':'post_hoc_output_audit_not_primary_metric','protocol_hash':digest(protocol),'metrics':metrics,'paired_semantic':paired}
    write_json(out/'output_audit_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
