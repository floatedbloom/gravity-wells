''' 
Tuning for gravity well echo chamber simulation for subreddits
'''
from sr_sim_ec import *
import argparse # for parsing commandline arguments
import sys      # for sys.argv to check number of arguments
from thompson.jetlogging import * # customized logging module
from multiprocessing import Pool    # multiprocessor pool
from multiprocessing import cpu_count   # get cpu count
from multiprocessing import Queue   # add Queues for comm
from multiprocessing import Manager   # needed to manage queues
import numpy as np  # used for getting mean value
from random import randint  # used to generate random integer
from tqdm import tqdm   # Progress Bar Made Easy
import pathlib  # for current execution directory info 
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_agfile(filebase):
    ''' Ensure consistent setting of agents filename '''
    return filebase + '_auths_agree.json'

def cntr(num1,num2):
    ''' split difference between two nums, with rounding '''
    return round((num1+num2)/2)

def tune_tm(subr_fb,tm_Q=None,DEBUG=False):
    TM_NUDGE_LIM = 20
    agents_df = get_agents(get_agfile(subr_fb))
    # Create simulation. 
    tm_val = tm_floor = 1    # use tm_val for tech mod; set lower limit to tuning
    tm_ceil = TM_TUNE_MIN   # set upper limit to tuning
    tm_prev = tm_val    # use prev to track last value 
    tm_ctr = 0     # use ctr to track nbr times prev and current are equal
    tm_nudge_ctr = 0     # use nudge_ctr to track nbr times we've nudged
    tot_cnt = sys.maxsize   # ensure we enter the while loop
    tm_targ = round(TM_TUNE_MOD*len(agents_df))   # reduce exiters according to mod
    targ_upper_rng = round(1.05*tm_targ+1)    # upper bound for acceptable target range
    targ_lower_rng = round(0.9*tm_targ)      # lower bound for acceptable target range

    #iters = 0
    sim = None
    logger.debug(f'*** Tuning {subr_fb} tm to allow escapes ~' +
            f'{round(TM_TUNE_MOD*100)}% of EC ({len(agents_df)}) = {tm_targ}: ***')
    #return (randint(1,15))   #jet debug
    #while tm_targ<tot_cnt and tm_val<TM_TUNE_MIN:   # exit if tm_val gets too large
    while True:
        logger.debug(f'\tStarted {subr_fb} loop for tm val = {tm_val} for ' +
            f'~{targ_lower_rng} - {targ_upper_rng} allowed escapes')
        sim = Simulation(
                agents_df,
                NBR_ITERS,
                GC,
                TSM,    # use tsm value from rc file
                tm_val,     # use tm_val in place of tm
                DR,     # decay rate
                X_FRAC,
                J_FRAC,
                )
        sim.run_sim(
                subr_fb + '_je.csv',
                JFT,
                XFT,
                RX,
                RJ,
                targ_upper_rng,    # break early if tuning limit exceeded
                DEBUG
                )
        tot_cnt = len(sim.free_agents)
        logger.debug(f'*** Finished {subr_fb} with {tot_cnt} free agents for ' +
                f'target of {targ_lower_rng} - {targ_upper_rng} on tm val ' +
                f'= {tm_val}')
        if tot_cnt in range(targ_lower_rng,targ_upper_rng):
            break   # good enough
        elif tm_targ < tot_cnt:   # tm_val too small
            if tm_val < tm_ceil:    # retune with new floor
                tm_floor = tm_val   # reset floor
            elif tm_ceil < TM_TUNE_MIN:     # retune with reset ceiling
                tm_ceil = cntr(tm_ceil,TM_TUNE_MIN)
            else:   # ceiling reached -- no need to tune further
                tm_val = tm_ceil    # ensure tm_val is at ceiling
                break
        else:   # tm_val too large
            if tm_val > tm_floor:   # retune with new ceiling
                tm_ceil = tm_val    # reset ceiling
            elif tm_floor > 1:  # retune with reset floor 
                tm_floor = cntr(tm_floor,1)
            else:   # floor too large
                tm_val = tm_floor   # ensure tm_val is at floor
                break
        if tm_nudge_ctr>TM_NUDGE_LIM:     # too many nudges
            break
        tm_val = cntr(tm_floor,tm_ceil)
        if tm_val == tm_prev:  # nudge tm_val to change repetitive outcome
            tm_ctr += 1
            if tm_ctr > 2:
                tm_nudge_ctr += 1
                tm_val += randint(-tm_ctr,tm_ctr)    # nudge
                logger.debug(f'\t\tnudged {subr_fb} tm {tm_nudge_ctr} times')
        else:   # reset ctr
            tm_ctr = 0
            tm_prev = tm_val
    logger.debug(f'*** Tuned {subr_fb} tm to {tm_val} for ' +
            f'target escapes of ~{targ_lower_rng} - {targ_upper_rng} with ' + 
            f'actual of {tot_cnt} ***')
    if tm_Q:    # if queue present from parallel processing
        if tm_nudge_ctr>TM_NUDGE_LIM:     # too many nudges
            tm_Q.put((subr_fb,0))    # return zero to indicate failed run
        else:
            tm_Q.put((subr_fb,tm_val))    # add tm_val to queue of results
    if tm_nudge_ctr>TM_NUDGE_LIM:     # too many nudges
        return 0    # return zero to indicate failed run
    else:
        return tm_val
# end of def tune_tm(subr_fb,DEBUG=False):

def tune_tsm(subr_fb,tuned_tm,tsm_Q=None,DEBUG=False):
    TSM_NUDGE_LIM = 20
    agents_df = get_agents(get_agfile(subr_fb))
    tsm_floor = tsm_ceil = tsm_val = 1    # use tsm_val for topic source mod
    tot_cnt = sys.maxsize   # ensure we enter the while loop
    tsm_prev = tsm_val    # use prev to track last value 
    tsm_ctr = 0     # use ctr to track nbr times prev and current are equal
    tsm_nudge_ctr = 0     # use nudge_ctr to track nbr times we've nudged
    tsm_targ = round(TSM_TUNE_MOD*len(agents_df))   # reduce exiters according to mod
    targ_upper_rng = round(1.02*tsm_targ+1)    # upper bound for acceptable target range
    targ_lower_rng = round(0.96*tsm_targ)      # lower bound for acceptable target range
    sim = None
    logger.debug(f'*** Tuning {subr_fb} tsm to allow escapes ~ ' +
            f'{round(TSM_TUNE_MOD*100)}% of EC ({len(agents_df)}) = {tsm_targ}: ***')
    #return (randint(1,15))   #jet debug -- uncomment to skip this calculation
    # for tuning, TSM*LLM sets the maximum times tuning will try to converge 
    ## before aborting the tuning effort
    while True:
        logger.debug(f'\tStarted {subr_fb} loop for tsm val = {tsm_val} for ' +
            f'~{targ_lower_rng} - {targ_upper_rng} allowed escapes')
        sim = Simulation(
                agents_df,
                NBR_ITERS,
                GC,
                tsm_val,   # use tsm_val in place of tsm
                tuned_tm,
                DR,  # decay rate
                X_FRAC,
                J_FRAC,
                )
        sim.run_sim(
                subr_fb + '_je.csv',
                JFT,
                XFT,
                RX,
                RJ,
                targ_upper_rng    # allow sim to break early if tuning limit exceeded
                )
        tot_cnt = len(sim.free_agents)
        logger.debug(f'\tFinished {subr_fb} with {tot_cnt} free agents for ' +
                f'target of {tsm_targ} on tsm val = {tsm_val}')
        logger.trace(f'tsm_floor={tsm_floor},tsm_ceil={tsm_ceil},' +
                f'tsm_val={tsm_val},targ_low={targ_lower_rng},' +
                f'targ_high={ targ_upper_rng},freed={tot_cnt}')
        if tot_cnt in range(targ_lower_rng,targ_upper_rng+1):
            break   # good enough
        elif tsm_targ < tot_cnt:   # tsm_val too small
            if tsm_val < tsm_ceil:    # retune with new floor
                tsm_floor = tsm_val   # reset floor
            else:
                tsm_ceil *= 10     # retune with reset ceiling
                tsm_floor = tsm_val # raise the floor
        else:   # tsm_val too large
            if tsm_val > tsm_floor:   # retune with new ceiling
                tsm_ceil = tsm_val    # reset ceiling
            elif tsm_floor > 1:  # retune with reset floor 
                tsm_floor = cntr(tsm_floor,1)
                tsm_ceil = tsm_val    # reset ceiling
            else:   # tsm floor of '1' too large
                tsm_val = tsm_floor   # ensure tsm_val is at floor
                logger.info(f'\n\t***Unable to tune {subr_fb} tsm small enough!')
                break
        if tsm_nudge_ctr>TSM_NUDGE_LIM:     # too many nudges
            break
        tsm_val = cntr(tsm_floor,tsm_ceil)
        if tsm_val == tsm_prev:  # nudge tsm_val to change repetitive outcome
            tsm_ctr += 1
            if tsm_ctr > 2:
                tsm_nudge_ctr += 1
                tsm_val += randint(-tsm_ctr,tsm_ctr)    # nudge
                logger.debug(f'\t\tnudged {subr_fb} tsm {tsm_nudge_ctr} times')
        else:   # reset ctr
            tsm_ctr = 0
            tsm_prev = tsm_val
    logger.debug(f'*** Tuned {subr_fb} tsm to {tsm_val} for ' +
            f'target escapes of ~{targ_lower_rng} - {targ_upper_rng} with ' + 
            f'actual of {tot_cnt} ***')
    if tsm_Q:    # if queue present from parallel processing
        if tsm_nudge_ctr>TSM_NUDGE_LIM:     # too many nudges
            tsm_Q.put((subr_fb,0))    # return zero to indicate failed run
        else:
            tsm_Q.put((subr_fb,tsm_val))    # add tsm_val to queue of results
    if tsm_nudge_ctr>TSM_NUDGE_LIM:     # too many nudges
        return 0   # return zero to indicate failed run 
    else:
        return tsm_val
# end of def tune_tsm(subr_fb,tuned_tm,DEBUG=False):

if __name__ == "__main__":
    if not len(sys.argv) > 1:
        sys.exit("Must call script from commandline with rc filename")
    my_parser = argparse.ArgumentParser(
        description=__doc__, # Use auto python doc description
        formatter_class=argparse.RawDescriptionHelpFormatter)                     
    my_parser.add_argument('rcfile',
        help="specify 'run commands' file containing sim parameters ",
        type=str)
    my_args = my_parser.parse_args()
    print("Reading tuning parameters from {}/{}".format(
        pathlib.Path().resolve(),my_args.rcfile))
    with open(my_args.rcfile,'r') as f:
        tune_parms = json.load(f)
    subr_fbs = tune_parms.get("SUBREDDIT_FILEBASES")
    DEBUG = tune_parms.get("DEBUG_TUNE",False)
    PARALLEL = tune_parms.get("PAR_TUNE",True)  # execute in parallel
    LLM = tune_parms.get("LOOP_LIMIT_MULTIPLIER",5) # to limit while loops
    NBR_ITERS = tune_parms.get("NUMBER_ITERATIONS_TUNE", 90)
    GC = tune_parms.get("GRAVITY_CONSTANT_TUNE", 0.00002)
    TM = tune_parms.get('TM_MOD_TUNE',1)     # one default has no effect
    # for tuning, TSM dictates the value used while initially tuning TM
    ## in order to find a common (miniumum) TM value (specific to the 
    ## technology--in this case Reddit) to use while tuning TSM for
    ## each specific subreddit
    TSM = tune_parms.get('TSM_MOD_TUNE',1)   # one default has no effect
    DR = tune_parms.get("DECAY_RATE_TUNE", 0.9)
    X_FRAC = tune_parms.get("EXIT_FRAC_TUNE", 0.1)
    J_FRAC = tune_parms.get("JOIN_FRAC_TUNE", 0.1)
    JFT = tune_parms.get("JF_THRESHOLD_TUNE", 120)
    XFT = tune_parms.get("XF_THRESHOLD_TUNE", 20)
    RJ = tune_parms.get("RANDOM_J_TUNE", False)     # defaults to prioritized
    RX = tune_parms.get("RANDOM_X_TUNE", False)     # defaults to prioritized
    TM_TUNE_MIN = tune_parms.get("TM_TUNE_MIN",99)     # used for mult tunings
    TM_TUNE_MOD = tune_parms.get("TM_TUNE_MOD",0.1)     # default 10%
    TSM_TUNE_MOD = tune_parms.get("TSM_TUNE_MOD",0.01)     # default 1%
    # use to skip TM tuning and just use val in rc file
    TSMONLY = tune_parms.get("TSMONLY", False) 
    # use to tune TM only and update val in rc file
    TMONLY = tune_parms.get("TMONLY", False)
    SR = tune_parms.get("SIM_REPS", 3)
    COLORS = ['blue','green']
    
    logger = jetlogger(pathlib.Path(my_args.rcfile).stem+'.log',DEBUG)
    logger.info(f'\n\tFrom {my_args.rcfile}, tuning {", ".join(subr_fbs)}')
    if not TSMONLY: # tune technology modifier
        logger.debug(f'\n\tBegin tuning with current TM MIN = {TM_TUNE_MIN}')
        for key,value in tune_parms.items():
            logger.debug(f'\t{key} = {value}')
        if PARALLEL:     # execute parallel tuning
            # create all processes to run in parallel
            with Pool() as tmpool:
                tmbar = dict()
                for idx,subr in enumerate(subr_fbs):
                    # initialize tm_result with empty lists
                    tm_result = {subr: [] for subr in subr_fbs}
                    # create progress bars for each subreddit
                    kwargs = {
                        'total': SR,
                        'unit': 'it',
                        'unit_scale': True,
                        'position': idx,
                        'leave': True,
                        'colour': COLORS[idx%2],
                        'dynamic_ncols': True,
                        'desc': f'TM tuning {subr}'
                        }
                    tmbar[subr] = tqdm(**kwargs)
                tm_m = Manager()
                tm_Q = tm_m.Queue()
                # Start load operations and mark each proc with its subreddit
                procs_tm = {tmpool.apply_async(tune_tm,(subr,tm_Q,DEBUG)): 
                        subr for subr in subr_fbs for __ in range(SR)}
                # Create a (dynamic) list to iterate over
                procs_tml = [(p,sr) for p,sr in procs_tm.items()]
                for proc,subr in procs_tml:
                    try:
                        (sr,r) = tm_Q.get()
                        logger.debug(f'TM tune subreddit {sr}, result = {r}')
                        if r:   # if a valid result queued
                            logger.debug(f'TM tune {sr}, {r}')
                            tm_result[sr].append(r)
                            tmbar[sr].update(1)
                            if tmbar[sr].n==SR:     # done with this bar
                                tmbar[sr].close()
                        else:   # requeue the job
                            logger.debug(f'TM tune resubmit {sr}, result = {r}')
                            p = tmpool.apply_async(tune_tm,(sr,tm_Q,DEBUG))
                            procs_tm[p]=sr
                            procs_tml.append((p,sr))
                    except Exception as e:
                        logger.error(f'tuning tm {sr,r} generated an exception: {e}')
                #for bar in tmbar: tmbar[bar].close()
        else:   # tune serially
            tm_result = {}
            logger.warning("\n\t***Tuning TM _serially_***")
            for subr in tqdm(subr_fbs,desc='TM subreddits',colour="blue"):
                tm_result[subr] = [tune_tm(subr,DEBUG=DEBUG) 
                        for __ in tqdm(range(SR),
                            desc=f'TM reps for {subr}',colour="magenta")]
        tuned_tm = min([min(tml) for tml in tm_result.values()])   # min tm for all sims
        if tuned_tm < TM_TUNE_MIN:  # update TM_TUNE_MIN in json
            procs_keylist = list(procs_tm)
            logger.info(f'\n\tReplacing TM_TUNE_MIN {TM_TUNE_MIN} with {tuned_tm} ' +
                f'found for ' + ', '.join({key
                    for key in tm_result for i,j in enumerate(tm_result[key]) 
                    if j==tuned_tm}))
            tune_parms['TM_TUNE_MIN'] = tuned_tm
            with open(my_args.rcfile,'w') as f:
                json.dump(tune_parms,f,indent=2)    # use indent to make readable
        else:   # replace with TM_TUNE_MIN
            tuned_tm = TM_TUNE_MIN
            logger.info(f'\n\tUsing prev technology modifier = {tuned_tm}')
        sys.exit(f'JET - exiting for checking TM={tuned_tm} modification')
    else:   # replace with TM_TUNE_MIN since did not tune TM
        tuned_tm = TM_TUNE_MIN
        logger.info(f'\n\tSkipped tuning TM. Using prev TM = {tuned_tm}')
    if not TMONLY:
        # now tune topic source modifier based on tuned_tm
        if PARALLEL:     # tune in parallel
            with Pool() as tsmpool:
                tsmbar = dict()
                for idx,subr in enumerate(subr_fbs):
                    # initialize tsm_result with empty lists
                    tsm_result = {subr: [] for subr in subr_fbs}
                    # create progress bars for each subreddit
                    kwargs = {
                        'total': SR,
                        'unit': 'it',
                        'unit_scale': True,
                        'position': idx,
                        'leave': True,
                        'colour': COLORS[idx%2],
                        'dynamic_ncols': True,
                        'desc': f'TSM tuning {subr}'
                        }
                    tsmbar[subr] = tqdm(**kwargs)
                tsm_m = Manager()
                tsm_Q = tsm_m.Queue()
                # Start load operations and mark each proc with its subreddit
                procs_tsm = {tsmpool.apply_async(tune_tsm,(subr_fb,tuned_tm,
                    tsm_Q,DEBUG)): subr_fb for subr_fb in subr_fbs for __ in range(SR)}
                # Create a (dynamic) list to iterate over
                procs_tsml = [(p,sr) for p,sr in procs_tsm.items()]
                for proc,subr in procs_tsml:
                    try:
                        (sr,r) = tsm_Q.get()
                        if r:   # if a valid result queued
                            logger.debug(f'TSM tune subreddit {sr}, result = {r}')
                            tsm_result[sr].append(r)
                            tsmbar[sr].update(1)
                            if tsmbar[sr].n==SR:     # done with this bar
                                tsmbar[sr].close()
                        else:   # requeue the job
                            logger.debug(f'TSM tune resubmit {sr}, result = {r}')
                            p = tsmpool.apply_async(tune_tsm,(sr,tuned_tm,tsm_Q,DEBUG))
                            procs_tsm[p]=sr
                            procs_tsml.append((p,sr))
                    except Exception as e:
                        logger.error(f'tuning tsm {sr,r} generated an exception: {e}')
                #for bar in tsmbar: tsmbar[bar].close()
        else:   # tune serially
            tsm_result = {}
            logger.warning("/n/t***Tuning TSM _serially_***")
            for subr_fb in tqdm(subr_fbs,desc='TSM subreddits',colour="green"):
                tsm_result[subr_fb] = [tune_tsm(subr_fb,tuned_tm,DEBUG=DEBUG) 
                        for __ in tqdm(range(SR),
                            desc=f'TSM reps for {subr_fb}',colour="cyan")]
    # END OF     if not TMONLY:

# summarize tuning results
    if not TSMONLY:  # tuned technology modifier
        logger.debug(f'*** Tuned tm to allow ~{round(TM_TUNE_MOD*100)}% ' +
                'escapes of EC size: ***')
        logger.debug(f'\ttm_result = {tm_result}')
    if not TMONLY:  # tuned topic source modifier
        logger.debug(f'*** Tuned tsm to ~{round(TSM_TUNE_MOD*100)}% escapes ' +
                'of EC size: ***')
        logger.debug(f'\ttsm_result = {tsm_result}')
        logger.info(f'\n\taverage topic source modifier for:\n\t\t' +
                '\n\t\t'.join(f'{sr} = {round(np.mean(tsm_result[sr]))}' 
                    for sr in subr_fbs))
