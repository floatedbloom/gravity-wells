# Gravity Wells

Code and data for [*Gravity Well Echo Chamber Modeling With An LLM-Based Confirmation Bias Model*](https://arxiv.org/abs/2509.03832). `sr_sim_ec.py` is the original simulation; `sr_sim_ec2.py` adds confirmation bias.

## Run

Install Python and Git LFS, then run from the repository root:

```bash
git lfs pull
pip install -r requirements-simulation.txt
python reddit_2024/sr_sim_ec2.py reddit_2024/data/rcs/progressive_2019-01_sim_rc.json
```

Results go to `sim_results/`. Cluster scripts are in `reddit_2024/hpc/`. The [archive branch](https://github.com/floatedbloom/gravity-wells/tree/archive/pre-cleanup-2026-10-12) keeps the original run files.
