"""Export aggregate-only quality/cost data and reproducible research figures.

Default: plot the committed aggregate data (requires matplotlib and numpy).
--refresh: rebuild aggregates from the local, ignored experimental artifacts.
No provider calls, fitting of quality curves, or changes to experimental results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/data/native_tag_quality_cost_20261123.json'
FIGURES = ROOT / 'docs/figures'


def refresh():
    sources = {}
    def read(relative):
        path = ROOT / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    base = 'artifacts/native_tag_comparison_20261122/'
    analysis = read(base + 'analysis.json')
    audit = read(base + 'fairness-and-cost-audit.json')
    points = []
    for arm in ('native', 'cheap_all'):
        s = audit['summary'][arm]
        points.append({'arm': arm, 'n_requests': s['requests'], 'strict_correct': s['strict_correct'],
                       'strict_accuracy': s['strict_accuracy'], 'lenient_accuracy': s['lenient_correct']/s['requests'],
                       'online_usd_per_query': s['mean_api_cost'],
                       'uncached_repricing_usd_per_query': s['mean_api_cost_no_cache'],
                       'usd_per_automatically_correct': s['api_cost_per_raw_correct_answer'],
                       'wall_seconds_per_query': s['mean_wall_seconds']})
    n, s = points
    savings = n['online_usd_per_query'] - s['online_usd_per_query']
    quality_gap = n['strict_accuracy'] - s['strict_accuracy']
    fixed = analysis['offline_incremental_cost']
    samples = []
    for q in (16, 32, 64, 79, 128, 1000, 10000):
        c = s['online_usd_per_query'] + fixed/q
        samples.append({'queries_per_build': q, 'cheap_amortized_usd_per_query': c,
                        'native_usd_per_query': n['online_usd_per_query'],
                        'savings_fraction': 1-c/n['online_usd_per_query']})
    dev_paths = [
        ('v1', 'artifacts/native_tag_optimization_20261123/replay-analysis.json', 'evidence_complete_v1'),
        ('v2 first', 'artifacts/native_tag_optimization_20261123/v2-evaluation/replay-analysis.json', 'evidence_complete_v2'),
        ('v2 repeat', 'artifacts/native_tag_development_repeat_20261123/evaluation/replay-analysis.json', 'evidence_complete_v2'),
    ]
    dev = []
    for label, path, profile in dev_paths:
        record = read(path)
        for arm, key in [('native', 'native'), ('candidate', profile)]:
            row = record['summary'][key]
            assert row['usage_complete']
            dev.append({'round': label, 'arm': arm, 'profile': key,
                        'n_requests': row['runs'], 'strict_correct': row['strict_correct'],
                        'strict_accuracy': row['strict_correct']/row['runs'],
                        'input_tokens_per_answer': row['prompt_tokens']/row['runs'],
                        'output_tokens_per_answer': row['completion_tokens']/row['runs'],
                        'uncached_repricing_usd_per_answer':
                            (row['prompt_tokens']*.15 + row['completion_tokens']*.60)/1e6/row['runs'],
                        'valid_citations': row['valid_citations'], 'format_or_call_failures': row['runtime_failures']})
    result = {'date': '2026-11-23', 'source_sha256': sources,
              'quality_scope': 'frozen_automated_scores_not_human_adjudication',
              'end_to_end': {'questions': 32, 'owners': 4, 'repeats_per_question': 2, 'points': points,
                             'quality_delta_bounds': analysis['symmetric_quality_delta_bounds'],
                             'decision': analysis['decision'],
                             'online_saving_fraction': 1-s['online_usd_per_query']/n['online_usd_per_query'],
                             'uncached_saving_fraction': audit['no_cache_cost_reduction'],
                             'incremental_usd_per_additional_correct_native': savings/quality_gap,
                             'cost_by_stage_per_request': analysis['cost_by_stage_per_request']},
              'amortization': {'incremental_build_usd': fixed, 'break_even_queries': math.ceil(fixed/savings),
                               'formula': 'C_cheap(Q)=online_cheap+B/Q; static snapshot, common build costs excluded',
                               'samples': samples},
              'development_replay': {'questions': 16, 'fixed_contexts': 32, 'points': dev,
                                     'repricing': {'input_per_million': .15, 'output_per_million': .60,
                                                   'cache': 'all_uncached', 'currency': 'USD',
                                                   'scope': 'common historical tariff scenario, not actual current cost'},
                                     'not_comparable_to_end_to_end': True},
              'new_provider_calls': 0}
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 12, 'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {'native': '#3E0097', 'cheap_all': '#087E8B', 'candidate': '#C55A20'}
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), layout='constrained')
    ax = axes[0]
    for row in data['end_to_end']['points']:
        c = colors[row['arm']]; name = 'Native' if row['arm']=='native' else 'Cheap tags'
        x = row['online_usd_per_query']*1000
        xn = row['uncached_repricing_usd_per_query']*1000
        for key, marker in [('strict_accuracy', 'o'), ('lenient_accuracy', '^')]:
            y = row[key]*100
            ax.plot([x, xn], [y, y], ':', color=c, alpha=.55, lw=1)
            ax.scatter(x, y, color=c, marker=marker, s=75, zorder=3)
            ax.scatter(xn, y, facecolors='white', edgecolors=c, marker=marker, s=75, zorder=3)
            if key == 'strict_accuracy':
                offset = (0,12) if row['arm']=='native' else (0,-34)
                ax.annotate(f'{name}\n{x:.3f}, {y+1e-9:.2f}%', (x,y), xytext=offset,
                            textcoords='offset points', ha='center', color=c, fontsize=9)
            else:
                ax.annotate(f'{y:.2f}%', (x,y), xytext=(0,9), textcoords='offset points', ha='center',color=c,fontsize=9)
    ax.set(title='A  Full requests: two observed operating points',
           xlabel='USD per 1,000 complete requests', ylabel='Automated accuracy (%)',
           xlim=(.4,4.4), ylim=(55,96))
    ax.text(.04,.96,'Circles: strict    Triangles: lenient\nFilled: billed    Hollow: uncached repricing',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.grid(alpha=.16)
    ax = axes[1]
    markers = {'v1':'o', 'v2 first':'s', 'v2 repeat':'D'}
    for label, marker in markers.items():
        rows = [r for r in data['development_replay']['points'] if r['round']==label]
        n, c = rows
        xs = [r['uncached_repricing_usd_per_answer']*1000 for r in rows]
        ys = [r['strict_accuracy']*100 for r in rows]
        ax.plot(xs,ys,'--',color='#999999',lw=1,alpha=.7)
        for row,x,y in zip(rows,xs,ys):
            ax.scatter(x,y,color=colors[row['arm']],marker=marker,s=70,zorder=3)
            offset = (0,10) if row['arm']=='native' else (0,-17)
            ax.annotate(f'{label}: {row["strict_correct"]}/32',(x,y),xytext=offset,
                        textcoords='offset points',ha='center',fontsize=9,color=colors[row['arm']])
    ax.set(title='B  Answer-only replay: paired prompt variants',
           xlabel='USD per 1,000 answers (common uncached tariff)',
           ylabel='Automated strict accuracy (%)', xlim=(.43,.62),ylim=(68,92))
    ax.text(.04,.96,'Purple: native    Orange: candidate\nDashed pairs are NOT a fitted quality curve',
            transform=ax.transAxes,va='top',fontsize=9)
    ax.grid(alpha=.16)
    fig.suptitle('Quality vs cost | different experimental scopes must stay separate',fontsize=14)
    for ext in ('png','pdf'):
        fig.savefig(FIGURES/f'native_tag_quality_cost_20261123.{ext}',dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    n,c=data['end_to_end']['points']; B=data['amortization']['incremental_build_usd']
    q=np.geomspace(16,10000,400)
    ax.plot(q,1000*(c['online_usd_per_query']+B/q),color=colors['cheap_all'],lw=2,label='Cheap tags + incremental setup / Q')
    ax.axhline(1000*n['online_usd_per_query'],color=colors['native'],lw=2,label='Native online cost')
    ax.axhline(1000*c['online_usd_per_query'],color=colors['cheap_all'],ls=':',label='Cheap-tag online asymptote')
    threshold=data['amortization']['break_even_queries']
    ax.axvline(threshold,color='#888888',ls='--',lw=1)
    ax.annotate(f'API-cost break-even: {threshold} queries',(threshold,1000*n['online_usd_per_query']),
                xytext=(120,4.2),arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.set(xscale='log',xlabel='Queries per static snapshot / calibration build (Q)',
           ylabel='Amortized USD per 1,000 queries',ylim=(0,9),
           title=f'Cost amortization model | incremental setup B = ${B:.5f}')
    ax.text(.97,.95,'Cost recovery does not establish quality equivalence',ha='right',va='top',transform=ax.transAxes,fontsize=10)
    ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1,.86));ax.grid(alpha=.16)
    for ext in ('png','pdf'):
        fig.savefig(FIGURES/f'native_tag_amortization_20261123.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true')
    args=parser.parse_args()
    data=refresh() if args.refresh else json.loads(DATA.read_text())
    plot(data)
    print(json.dumps({'data':str(DATA),'figures':str(FIGURES),'break_even_queries':data['amortization']['break_even_queries']}))
