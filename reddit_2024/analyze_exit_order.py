''' 
Script to compare the order that authors leave (at least temporarily) each subreddit,
versus the order agents leave the echo chamber simulation of same subreddit. Code 
makes use of the "auths_by_subr-sorted.json.gz" file created by "get_auths_last_posts.py".
'''
import os
import sys
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from sr_sim_ec2 import *
import json     # for processing json files
import pandas as pd     # use pandas Python Data Analysis Library
import pathlib  # for current execution directory info
import shutil   # for file manipulation
from tqdm import tqdm   # Progress Bar Made Easy
import numpy as np  # for NpEncoder class to convert Numpy Ints for JSON export
import gzip     # to enable data export to compressed file
from collections import OrderedDict     # for storing sorted dictionary of times
from operator import itemgetter     # for sorting the dictionary of author times
from sklearn.metrics import mean_absolute_percentage_error  # error calc
from multiprocessing import Pool    # multiprocessor pool
from multiprocessing import cpu_count   # get cpu count
from multiprocessing import Queue   # add Queues for comm
from multiprocessing import Manager   # needed to manage queues
from queue import Empty  # for queue timeouts

# Extend the JSONEncoder class to handle Numpy values
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def sim_agt(iteration,rank_Q,rcfile_arg,debug_arg,log_path_str,bias_path_str):
    ''' function to run simulation to observe agent exiting order '''
    # Create a local logger for the worker; fall back to print if unavailable
    _worker_logger = None
    try:
        _logfile = pathlib.Path(log_path_str).with_name(pathlib.Path(log_path_str).stem + f'.worker-{iteration}.log')
        _worker_logger = jetlogger(_logfile, debug_arg)
    except Exception:
        _worker_logger = None
    def _log_debug(msg):
        if _worker_logger is not None:
            _worker_logger.debug(msg)
        else:
            print(msg)
    try:
        _log_debug(f'Worker start iteration={iteration}, rcfile={rcfile_arg}')
        (parms,agents_df) = get_dat(rcfile_arg)
        parms['TOPIC_SOURCE_MODIFIER'] = 1  # override TSM
        _log_debug(f"overrode TOPIC_SOURCE_MODIFIER = {parms.get('TOPIC_SOURCE_MODIFIER')}")
        parms['TECHNOLOGY_MODIFIER'] = 1  # override TM
        _log_debug(f"overrode TECHNOLOGY_MODIFIER = {parms.get('TECHNOLOGY_MODIFIER')}")
        parms['JOIN_FRAC'] = 0.0  #override join fraction
        _log_debug(f"overrode JOIN_FRAC = {parms.get('JOIN_FRAC')}")
        # pass parameters (or default) to new sim object
        sim = Simulation(
                agents_df,
                parms.get('NUMBER_ITERATIONS',90),    # defaults to nominal number
                parms.get('GRAVITY_CONSTANT'),
                parms.get('TOPIC_SOURCE_MODIFIER',1),   # 1 ensures agents will exit
                parms.get('TECHNOLOGY_MODIFIER',1),     # 1 ensures agents will exit
                parms.get('DECAY_RATE',0.9),     # 0.9 is a slow decay
                parms.get('EXIT_FRAC',0.1),     # 1/10th of capt members try to leave
                parms.get('JOIN_FRAC',0.1),     # JOIN_FRAC=0 ensures no agents rejoin
                bias_path=bias_path_str
                )
        # run simulation
        sim.run_sim(
                parms.get('FILE_BASE') + '_je.csv',
                parms.get('JOIN_FORCE_THRESHOLD',500), 
                parms.get('EXIT_FORCE_THRESHOLD',500),
                parms.get('RANDOM_EXITERS',False),
                parms.get('RANDOM_JOINERS',False),
                debug_arg
                )
        # put results in multiprocessing queue
        rank_Q.put((iteration, [agt.author for agt in sim.free_agents],
                [agt.author for agt in sim.ec.captured_agents], None))
    except Exception as e:
        # Propagate errors to main process
        try:
            rank_Q.put((iteration, None, None, f"worker error: {type(e).__name__}: {e}"))
        except Exception:
            pass
    return
# end of def sim_agt():

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    pdf = pd.DataFrame()
    # Resolve paths assuming this script lives in reddit_2024/
    script_dir = Path(__file__).resolve().parent  # points to reddit_2024/
    repo_root = script_dir.parent                 # points to gravity_well/
    sorted_auths_filename = str(repo_root / 'auths_by_subr-sorted.json.gz')
    sorted_agents_filename = str(repo_root / 'agents_by_subr-par_sorted.json.gz')
    #sorted_auths_filename = 'auths_by_subr_gs-sorted.json.gz'   #globeskepticism
    #sorted_agents_filename = 'agents_by_subr_gs-par_sorted.json.gz'     #globeskepticism
    subrs = ["SandersForPresident","flatearth","trump","science","cars","Republican",
            "SocialDemocracy","Freethought","travel","math","NeutralPolitics",
            "PoliticalDiscussion","democrats","hiking","republicans","mlb",
            "progressive","AmericanPolitics"]
    its = 100
    #its = 10    # test data
    agents_by_subr = {}     # dict to track agent exit ranking by subreddit
    mape_by_subr = {}   # record mean avg pct error of exit rank by subreddit

    # read sorted dictionary of exiting authors for each subreddit
    print(f"Reading auths from {sorted_auths_filename}")
    with gzip.open(sorted_auths_filename, 'rt') as filein:
        auths_by_subr = json.load(filein)

    # Filter to subreddits present in auths data
    available = set(auths_by_subr.keys())
    missing = [sr for sr in subrs if sr not in available]
    if missing:
        print(f"Skipping subreddits not in auths data: {missing}")
    subrs = [sr for sr in subrs if sr in available]
    print(f"Processing subreddits: {subrs}")

    # simulate repeatedly for each subreddit to determine common agent exiting order
    for sr in subrs:
        rcfile = f'data/rcs/{sr}_2019-01_sim_rc.json'
        if sr == "globeskepticism":
            rcfile = f'data/rcs/globeskepticism_2019-01_sim_rc.json'
        debug = False

        # Pre-check expected input files
        file_base = f"{sr}_2019-01"
        auths_agree_json = script_dir / f"data/zipped/{file_base}_auths_agree.json"
        auths_agree_gz = script_dir / f"data/zipped/{file_base}_auths_agree.json.gz"
        agree_json = script_dir / f"data/zipped/{file_base}_agree.json"
        agree_gz = script_dir / f"data/zipped/{file_base}_agree.json.gz"
        print(f"[{sr}] RC: {rcfile}")
        print(f"[{sr}] Exists(auths_agree.json): {auths_agree_json.exists()} | .gz: {auths_agree_gz.exists()}")
        print(f"[{sr}] Exists(agree.json):        {agree_json.exists()} | .gz: {agree_gz.exists()}")
        if not (auths_agree_json.exists() or auths_agree_gz.exists()):
            print(f"[{sr}] Warning: auths_agree file missing; sr_sim_ec2 will fail without fallback.")

        agent_dict = dict.fromkeys(list(auths_by_subr[sr].keys()),0)
        # Use a dedicated logs directory under reddit_2024
        log_dir = script_dir / 'exit_order_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        logfile = log_dir / (pathlib.Path(rcfile).stem + '.par-rank.log')
        # Derive bias path for this subreddit (lowercased)
        bias_path = script_dir / f"data/bias/bias_values_{sr.lower()}.json"
        if not bias_path.exists():
            print(f"[{sr}] Warning: Bias file not found at {bias_path}")
        logger = jetlogger(logfile,debug)
        logger.debug(f'Entering parallel processing for subreddit {sr}')
        # start parallel coding
        with Pool() as rnkpool:
            # create progress bar
            kwargs = {
                'total': its,
                'unit': 'it',
                'unit_scale': True,
                'leave': True,
                'colour': 'blue',
                'dynamic_ncols': True,
                'desc': f'agent sim for {sr}'
                }
            rnkbar = tqdm(**kwargs)
            rnk_m = Manager()
            rnk_Q = rnk_m.Queue()
            # Start load operations and mark each proc with iteration
            procs_rnk = {rnkpool.apply_async(sim_agt,(it,rnk_Q,rcfile,debug,str(logfile),str(bias_path))):
                    it for it in range(its)}
            completed = 0
            while completed < its:
                try:
                    msg = rnk_Q.get(timeout=6000)
                except Empty:
                    logger.warning(f"[{sr}] Timeout waiting for worker result; completed={completed}/{its}")
                    continue
                it, free_agents, captured_agents, err = msg
                if err is not None:
                    logger.error(f"[{sr}] Worker {it} error: {err}")
                    completed += 1
                    rnkbar.update(1)
                    continue
                logger.debug(f'Finished run {it} of sr_sim_ec for {sr}')
                for idx,agt in enumerate(free_agents):  # free_agents ordered by exit
                    agent_dict[agt] += idx+1     # update total exit rank
                # should only be a few agents left in the EC, so they will have highest rank
                max_rank = len(free_agents)+1
                logger.debug(f'Setting remaining {len(captured_agents)} captured agents in iteration {it} to max rank of {max_rank}')
                for agt in captured_agents:  # all agents that didn't exit
                    agent_dict[agt] += max_rank
                completed += 1
                rnkbar.update(1)
            if rnkbar.n>=its:     # done with this bar
                rnkbar.close()
        agent_dict.update((k,rank/its) for k,rank in agent_dict.items()) # avg rank
        # now sort dictionary by earliest to latest average exit order
        agent_dict = OrderedDict(sorted(agent_dict.items(), key=itemgetter(1)))
        # now rank by order of departure, then save to dict
        prnk = 0 
        ctr = 0     # create rank ctr for author departure order 
        for agt,rnk in agent_dict.items(): 
            if rnk != prnk:
                ctr+=1  # only incr ctr if curr rank differs from prev
            prnk = rnk
            agent_dict[agt]=(rnk,ctr)    # add rank ctr as part of dict
        # calculate mean absolute error of rankings
        ## list of real exit rankings
        y_true = [itemgetter(1)(item) for item in list(auths_by_subr[sr].values())]
        ## list of simulation exit rankings
        y_pred = [itemgetter(1)(agent_dict[agt]) for agt in auths_by_subr[sr]]
        mape_by_subr[sr] = mean_absolute_percentage_error(y_true, y_pred)
        logger.debug(f'{sr} ranking MAPE = {mape_by_subr[sr]}')
        print(f'{sr} ranking MAPE = {mape_by_subr[sr]}')

        # close logging for this subreddit
        logs = list(logger.handlers)
        for l in logs:
            logger.removeHandler(l)
            l.flush()
            l.close()
        # compress logfile then delete orig
        with open(logfile, 'rb') as f_in:
            with gzip.open(str(logfile) + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        pathlib.Path.unlink(logfile)    # delete orig logfile
        agents_by_subr[sr] = agent_dict     # preserve agent_dict for this subreddit
    # writing sorted and ranked dictionary of agent dictionaries to disk
    with gzip.open(sorted_agents_filename, 'w') as fileout:
        fileout.write(json.dumps(agents_by_subr,cls=NpEncoder).encode('utf-8'))
    # print Mean Absolute Percent Error to screen
    w = max([len(k) for k in mape_by_subr])
    print('\tsubreddit\t\tMAPE')
    for k,m in mape_by_subr.items():
        print(f'{k:>{w}}: \t{m}')