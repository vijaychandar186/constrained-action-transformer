"""Summarize a paired model comparison without interpreting its score as Elo."""
import argparse
import json
import random
import statistics
from pathlib import Path
from src.pipeline import write_json

def summarize(small,large,matches,output,baseline_name="small",candidate_name="large"):
    a,b,m=[json.loads(Path(p).read_text()) for p in (small,large,matches)]
    keys=('positions_sha256','seed','depth','engine','decoding','mate_score','count')
    if any(a[k]!=b[k] for k in keys):
        raise ValueError('Evaluation protocols differ.')
    if [(r['fen'],r['moves']) for r in a['rows']]!=[(r['fen'],r['moves']) for r in b['rows']]:
        raise ValueError('Sampled positions differ.')
    if m['rl_sha256']!=b['checkpoint_sha256'] or m['baseline_sha256']!=a['checkpoint_sha256']:
        raise ValueError('Match checkpoints differ from evaluation checkpoints.')
    differences=[y['centipawn_loss']-x['centipawn_loss'] for x,y in zip(a['rows'],b['rows'])]
    rng=random.Random(42)
    boot=sorted(statistics.mean(rng.choices(differences,k=len(differences))) for _ in range(5000))
    metrics={}
    for label,report in [(baseline_name,a),(candidate_name,b)]:
        losses=[r['centipawn_loss'] for r in report['rows']]
        metrics[label]=dict(acpl=statistics.mean(losses),median_cpl=statistics.median(losses),
                            fraction_cpl_over_100=sum(v>100 for v in losses)/len(losses),
                            fraction_zero_cpl=sum(v==0 for v in losses)/len(losses))
    report=dict(positions=len(differences),baseline_name=baseline_name,candidate_name=candidate_name,evaluation=metrics,
                large_minus_small_acpl=statistics.mean(differences),
                paired_position_bootstrap_95_ci=[boot[125],boot[4874]],
                bootstrap_note='Resamples paired positions; positions from the same game may be correlated.',
                matches={k:m[k] for k in ('pairs','candidate_wins','candidate_losses','draws','candidate_score_fraction','paired_bootstrap_95_ci','truncated_games','max_plies')},
                elo_measured=False)
    if baseline_name!='small' or candidate_name!='large':
        report['candidate_minus_reference_acpl']=report.pop('large_minus_small_acpl')
    write_json(output,report)
    print(json.dumps(report,indent=2))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small',required=True);p.add_argument('--large',required=True)
    p.add_argument('--matches',required=True);p.add_argument('--output',required=True)
    p.add_argument('--baseline-name',default='small');p.add_argument('--candidate-name',default='large')
    a=p.parse_args();summarize(a.small,a.large,a.matches,a.output,a.baseline_name,a.candidate_name)

if __name__=='__main__':
    main()
