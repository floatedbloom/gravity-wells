import json, gzip, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

root = Path('.')
auths = json.load(gzip.open(root/'auths_by_subr-sorted.json.gz','rt'))
agents = json.load(gzip.open(root/'agents_by_subr-par_sorted.json.gz','rt'))

rows = []
for sr in sorted(set(auths) & set(agents)):
    gt = auths[sr]          # author -> [timestamp, rank]
    sim = agents[sr]        # author -> [avg_rank, rank_order]
    common = list(set(gt) & set(sim))
    if len(common) < 5:
        continue
    gt_ranks  = [gt[a][1]   for a in common]
    sim_ranks = [sim[a][1]  for a in common]
    rho, p = spearmanr(gt_ranks, sim_ranks)
    rows.append({'subreddit': sr, 'n_common': len(common), 'spearman_rho': rho, 'p_value': p})

pd.DataFrame(rows).sort_values('spearman_rho', ascending=False).to_csv('exit_order_correlation.csv', index=False)
print('Wrote exit_order_correlation.csv')