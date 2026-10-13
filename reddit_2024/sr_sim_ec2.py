'''
Gravity well echo chamber simulation for subreddits
Fixed version with proper RNG state management
'''

from dataclasses import dataclass
import argparse
import json
import random
import math
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import sys
import pathlib
import csv
from thompson.jetlogging import *

AGENT_ATTRIBUTES = ['author', 'avg_sim', 'avg_diss', 'affinity', 'diss_affinity_ratio']
MIN_VALUE = 0.01

@dataclass(eq=False)
class Agent:
    author: str
    affinity: float
    dissimilarity: float
    bias_seeking: float = 1.0
    captured_ctr: int = 0

    @classmethod
    def from_record(cls, record: pd.Series, bias_path: str):
        bias_seeking = 1.0
        try:
            with open(bias_path, 'r') as infile:
                bias_data = json.load(infile)
            for entry in bias_data:
                if entry['author'] == record.Index:
                    bias_seeking = entry.get('bias_seeking', 1.0)
                    break
        except Exception:
            # If bias file missing or unreadable, default to 1.0
            pass

        return cls(
            author=record.Index,
            affinity=max(MIN_VALUE, record.affinity * bias_seeking),
            dissimilarity=max(MIN_VALUE, record.avg_diss),
            bias_seeking=bias_seeking
        )

    def update_affinity(self, max_affinity: float, np_rng: np.random.RandomState):
        if self.affinity >= max_affinity:
            return

        mean = self.affinity
        high = min(1.0, max_affinity)
        sd = (high - mean)/2
        low = mean - sd
        
        new_affinity = truncnorm(
            (low-mean)/sd,
            (high-mean)/sd,
            loc=mean,
            scale=sd
        ).rvs(random_state=np_rng)
        
        self.affinity = min(1.0, new_affinity * self.bias_seeking)

    def update_dissimilarity(self, mass: int, np_rng: np.random.RandomState):
        self.dissimilarity = min(1.0,
            np_rng.poisson(self.dissimilarity * mass) / mass)

class EchoChamber:
    def __init__(self, agents_df: pd.DataFrame, src_mod: float,
                 tech_mod: float, exit_frac: float, np_rng: np.random.RandomState,
                 bias_path: str):
        self.np_rng = np_rng
        random.seed(12345)
        
        self.max_affinity = agents_df.affinity.max()
        self._src_mod = src_mod
        self._tech_mod = tech_mod
        self._exit_frac = exit_frac
        
        self.captured_agents = [
            Agent.from_record(agent_record, bias_path)
            for agent_record in agents_df.itertuples()
        ]

    def capture_agent(self, agent: Agent):
        self.max_affinity = max(agent.affinity, self.max_affinity)
        self.captured_agents.append(agent)

    def release_agent(self, agent: Agent):
        self.captured_agents.remove(agent)

    def update_captured_agents(self):
        for agent in self.captured_agents:
            agent.captured_ctr += 1
            agent.update_dissimilarity(self.mass, self.np_rng)
            agent.update_affinity(self.max_affinity, self.np_rng)

    def nbr_exiting(self):
        return abs(
            random.randint(0, math.ceil(self.EXIT_FRAC * self.mass)) -
            random.randint(0, math.floor(self.EXIT_FRAC * self.mass))
        )
    
    @property
    def mass(self):
        return len(self.captured_agents)
    
    @property
    def eff_mass(self):
        return self.mass * self._tech_mod

    @property
    def affin_mod(self):
        return self._src_mod

    @property
    def EXIT_FRAC(self):
        return self._exit_frac

    @property
    def capt_DA_ratio(self):
        if not self.captured_agents:
            return (0.0, 0.0)
        avg_diss = sum(ag.dissimilarity for ag in self.captured_agents) / self.mass
        avg_aff = sum(ag.affinity * self.affin_mod for ag in self.captured_agents) / self.mass
        return (avg_diss, avg_aff)

class Simulation:
    def __init__(self, agents_df: pd.DataFrame, n_iterations: int,
                 grav_const: float, src_mod: float, tech_mod: float,
                 decay_rate: float, exit_frac: float, join_frac: float,
                 bias_path: str):
        self.np_rng = np.random.RandomState(12345)
        random.seed(12345)
        
        self._G = grav_const
        self._DECAY_RATE = decay_rate
        self.agents_df = agents_df
        self._NUM_ITERATIONS = n_iterations
        self._JOIN_FRAC = join_frac
        
        self.ec = EchoChamber(agents_df, src_mod, tech_mod, exit_frac, self.np_rng, bias_path)
        self.free_agents = []

    def run_sim(self, join_exit_fn: str, join_force_threshold: float,
               exit_force_threshold: float, rnd_exit: bool, rnd_join: bool, DEBUG: bool):
        je_f = open(join_exit_fn, 'w', newline='')
        je_file_writer = csv.writer(je_f)
        je_file_writer.writerow([
            "Epoch", "Joined", "Joining", "Join pct", "Avg Join Force",
            "Exited", "Exiting", "Exit pct", "Avg Exit Force", "EC-size-pct"
        ])

        for epoch in range(self.NUM_ITERATIONS):
            self.ec.update_captured_agents()
            self.update_free_agents()

            nbr_joining = self.nbr_joining()
            j_a_l = []
            if nbr_joining > 0:
                if rnd_join:
                    j_a_l = random.sample(self.free_agents, min(nbr_joining, len(self.free_agents)))
                else:
                    j_a_l = sorted(self.free_agents, key=lambda a: a.affinity, reverse=True)[:nbr_joining]

            nbr_joined = 0
            f_jtot = 0
            for j_a in j_a_l:
                denominator = (j_a.dissimilarity/j_a.affinity)**2 + 1e-8
                jf = self.G * self.ec.eff_mass / denominator
                f_jtot += jf
                if jf > join_force_threshold and self.freedom > 1:
                    self.ec.capture_agent(j_a)
                    self.free_agents.remove(j_a)
                    nbr_joined += 1

            nbr_exiting = self.ec.nbr_exiting()
            x_a_l = []
            if nbr_exiting > 0:
                if rnd_exit:
                    x_a_l = random.sample(self.ec.captured_agents, min(nbr_exiting, len(self.ec.captured_agents)))
                else:
                    x_a_l = sorted(self.ec.captured_agents, key=lambda a: a.dissimilarity, reverse=True)[:nbr_exiting]

            nbr_exited = 0
            f_xtot = 0
            for x_a in x_a_l:
                denominator = (x_a.dissimilarity/(x_a.affinity*self.ec.affin_mod))**2 + 1e-8
                xf = self.G * self.ec.eff_mass / denominator
                f_xtot += xf
                if xf < exit_force_threshold and self.ec.mass > 1:
                    self.free_agents.append(x_a)
                    self.ec.release_agent(x_a)
                    nbr_exited += 1

            joined_pct = 0 if nbr_joining == 0 else nbr_joined/nbr_joining
            exited_pct = 0 if nbr_exiting == 0 else nbr_exited/nbr_exiting
            je_file_writer.writerow([
                epoch, nbr_joined, nbr_joining, f"{joined_pct*100:.2f}", f"{f_jtot/max(1,nbr_joining):.3f}",
                nbr_exited, nbr_exiting, f"{exited_pct*100:.2f}", f"{f_xtot/max(1,nbr_exiting):.3f}",
                f"{self.ec.mass/len(self.agents_df)*100:.2f}"
            ])

        je_f.close()

    def update_free_agents(self):
        for agent in self.free_agents:
            if agent.captured_ctr > 0:
                agent.captured_ctr -= 1
            
            orig_diss = self.agents_df.loc[agent.author].avg_diss
            agent.dissimilarity = max(MIN_VALUE,
                orig_diss if agent.dissimilarity <= orig_diss/self.DECAY_RATE
                else agent.dissimilarity * self.DECAY_RATE
            )
            
            orig_aff = self.agents_df.loc[agent.author].affinity
            agent.affinity = max(MIN_VALUE,
                orig_aff if agent.affinity <= orig_aff/self.DECAY_RATE
                else agent.affinity * self.DECAY_RATE
            )

    def nbr_joining(self):
        return abs(
            random.randint(0, math.floor(self.JOIN_FRAC * self.freedom)) -
            random.randint(0, math.floor(self.JOIN_FRAC * self.freedom))
        )

    @property
    def freedom(self):
        return len(self.free_agents)
    
    @property
    def G(self):
        return self._G
    
    @property
    def DECAY_RATE(self):
        return self._DECAY_RATE
    
    @property
    def NUM_AGENTS(self):
        return len(self.agents_df)
    
    @property
    def NUM_ITERATIONS(self):
        return self._NUM_ITERATIONS
    
    @property
    def JOIN_FRAC(self):
        return self._JOIN_FRAC

def get_agents(agfile: str):
    try:
        with open(agfile, 'r') as f:
            return pd.read_json(f)[AGENT_ATTRIBUTES].set_index('author')
    except (FileNotFoundError, ValueError):
        with open(agfile + '.gz', 'rb') as f:
            return pd.read_json(f, compression='gzip')[AGENT_ATTRIBUTES].set_index('author')

def get_dat(rcfile: str):
    with open(rcfile, 'r') as f:
        parms = json.load(f)
    agent_file = pathlib.Path(__file__).resolve().parent / 'data' / 'zipped' / (parms['FILE_BASE'] + '_auths_agree.json')
    return parms, get_agents(str(agent_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('rcfile')
    args = parser.parse_args()
    
    # Ensure outputs (logs, CSVs) go to a dedicated directory
    out_dir = pathlib.Path('sim_results')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Log file in output directory
    log_path = out_dir / (pathlib.Path(args.rcfile).stem + '.log')
    logger = jetlogger(str(log_path), args.debug)
    parms, agents_df = get_dat(args.rcfile)
    
    # Derive subreddit from FILE_BASE to select the correct bias file
    file_base = parms.get('FILE_BASE', '')
    subreddit = file_base.split('_', 1)[0] if '_' in file_base else file_base
    derived_bias_path = pathlib.Path(__file__).resolve().parent / 'data' / 'bias' / f'bias_values_{subreddit.lower()}.json'

    sim = Simulation(
        agents_df=agents_df,
        n_iterations=parms.get('NUMBER_ITERATIONS', 90),
        grav_const=parms.get('GRAVITY_CONSTANT'),
        src_mod=parms.get('TOPIC_SOURCE_MODIFIER', 1),
        tech_mod=parms.get('TECHNOLOGY_MODIFIER', 1),
        decay_rate=parms.get('DECAY_RATE', 0.9),
        exit_frac=parms.get('EXIT_FRAC', 0.1),
        join_frac=parms.get('JOIN_FRAC', 0.1),
        bias_path=str(derived_bias_path)
    )
    
    # CSV output in output directory
    join_exit_path = out_dir / f"{parms['FILE_BASE']}_je2.csv"
    sim.run_sim(
        join_exit_fn=str(join_exit_path),
        join_force_threshold=parms.get('JOIN_FORCE_THRESHOLD', 500),
        exit_force_threshold=parms.get('EXIT_FORCE_THRESHOLD', 500),
        rnd_exit=parms.get('RANDOM_EXITERS', False),
        rnd_join=parms.get('RANDOM_JOINERS', False),
        DEBUG=args.debug
    )
