# Gravity Wells

Code and data for [*Gravity Well Echo Chamber Modeling With An LLM-Based Confirmation Bias Model*](https://arxiv.org/abs/2509.03832) (Jackson, Lapin, Thompson).

The project models how confirmation bias changes the pull of social media echo chambers. It contains the original subreddit simulation, the confirmation-bias variant, analysis scripts, and cluster jobs used for the paper.

## Repository guide

- `reddit_2024/sr_sim_ec.py`: original gravity-well simulation.
- `reddit_2024/sr_sim_ec2.py`: simulation with per-user confirmation-bias values.
- `reddit_2024/create_conf_bias.py`: estimates those values from Reddit comments using Ollama.
- `reddit_2024/data/rcs/`: simulation parameter files.
- `reddit_2024/data/zipped/` and `reddit_2024/data/bias/`: agent data and saved bias values.
- `reddit_2024/hpc/`: WSU cluster scripts. These contain cluster-specific paths and SLURM settings.

Most JSON files use Git LFS. Install Git LFS and run `git lfs pull` after cloning; an unresolved file beginning with `version https://git-lfs.github.com/spec/v1` is only a pointer.

## Run a simulation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-minimal.txt
python reddit_2024/sr_sim_ec2.py reddit_2024/data/rcs/progressive_2019-01_sim_rc.json
```

The confirmation-bias run writes logs and a CSV to `sim_results/`. It needs the parameter JSON, agent data, and bias JSON for the selected subreddit. `requirements-minimal.txt` also covers the data and HPC workflows.

To generate new bias values, the corresponding Reddit comment files must be placed in `reddit_2024/data/filtered_analyzable/` or `reddit_2024/data/with_likes/`; those source files are not included here. With Ollama running, invoke `python create_conf_bias.py --subreddits progressive` from `reddit_2024/`. The HPC scripts require the original cluster environment or local adaptation.
