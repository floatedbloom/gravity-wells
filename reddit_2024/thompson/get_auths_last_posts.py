''' 
Script to retrieve all authors in first mo/yr provided (hardcoded for now)
for the list of subreddits provided (again hardcoded), then process all remaining
months in that year, searching for last post made by each of the initial authors.
After which, the list of authors for each subreddit is sorted by earliest
departure (last date of posting for each author), ranked, then savd to the 
specified (hardcoded) file.
'''
import par_zst_to_df as pz     # local module for converting zst to dataframe
import argparse     # for parsing commandline arguments
import json     # for processing json files
import pandas as pd     # use pandas Python Data Analysis Library
import pathlib  # for current execution directory info
from tqdm import tqdm   # Progress Bar Made Easy
import numpy as np  # for NpEncoder class to convert Numpy Ints for JSON export
import gzip     # to enable data export to compressed file
from collections import OrderedDict     # for storing sorted dictionary of times
from operator import itemgetter     # for sorting the dictionary of author times
from multiprocessing import freeze_support

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
if __name__ == '__main__':
    freeze_support()
    pdf = pd.DataFrame()
    auths_filename = 'auths_by_subr.json.gz'
    auths_filename2 = 'auths_by_subr-sorted.json.gz'
    subrs = ["SandersForPresident","flatearth","trump","science","cars","Republican",
            "SocialDemocracy","Freethought","travel","math","NeutralPolitics",
            "PoliticalDiscussion","democrats","hiking","republicans","mlb",
            "progressive","AmericanPolitics"]
    # keep bare minimum to reduce memory usage
    keys_to_keep = ['author','created_utc','id','subreddit']
    yr = 2019
    mo = 1
    zstcf = f'RC_{yr}-{mo:02}.zst'
    zstsf = f'RS_{yr}-{mo:02}.zst'

    # get data from first month of interest
    rs_zstfp = pathlib.Path(zstsf)
    if not rs_zstfp.exists():
        raise FileNotFoundError(f'zstfile {rs_zstfp} does not exist!')
    rc_zstfp = pathlib.Path(zstcf)
    if not rc_zstfp.exists():
        raise FileNotFoundError(f'zstfile {rc_zstfp} does not exist!')
    print(f'Decompressing data from {rs_zstfp} and {rc_zstfp}.')
    for zstfp in [rs_zstfp,rc_zstfp]:
        pdf = pz.decompress_file_chunks(zstfp,pdf,subrs,keys_to_keep)
    # create dictionary of authors occurring in the first month for each subreddit
    ## store maximum (latest) created_utc date in dict
    auths_by_subr = {}
    for s in (pbar:=tqdm(subrs,desc='subreddits',colour="blue")):
        pbar.set_postfix_str(s)     # update label of progressbar
        pdf_subr = pdf[pdf.subreddit==s]
        auths_by_subr[s]={a: pdf_subr[pdf_subr.author==a].created_utc.max() 
                for a in set(pdf_subr.author)}
    # loop over remaining months in year to determine latest date each
    ## author makes a post
    for m in range(mo+1,3):
        pdf = pd.DataFrame() # reset working dataframe
        zstcf = f'RC_{yr}-{m:02}.zst'
        zstsf = f'RS_{yr}-{m:02}.zst'
        rs_zstfp = pathlib.Path(zstsf)
        if not rs_zstfp.exists():
            raise FileNotFoundError(f'zstfile {rs_zstfp} does not exist!')
        rc_zstfp = pathlib.Path(zstcf)
        if not rc_zstfp.exists():
            raise FileNotFoundError(f'zstfile {rc_zstfp} does not exist!')
        print(f'**Decompressing data from {rs_zstfp} and {rc_zstfp}.')
        for zstfp in [rs_zstfp,rc_zstfp]:
            pdf = pz.decompress_file_chunks(zstfp,pdf,subrs,keys_to_keep)
        for s in (pbar0:=tqdm(subrs,desc='subreddits',colour="blue",position=0)):
            pbar0.set_postfix_str(s)     # update label of progressbar
            pdf_subr = pdf[pdf.subreddit==s]
            # loop over the authors that appear in the new data only if they
            ## were in the original set of authors
            for auth in (pbar1:=tqdm(auths_by_subr[s].keys() & pdf_subr.author,
                    desc='authors',colour="green",position=1)):
                auths_by_subr[s][auth]=pdf_subr[pdf_subr.author==auth].created_utc.max() 
    # writing dictionary of author dictionaries to disk
    with gzip.open(auths_filename, 'w') as fileout:
        fileout.write(json.dumps(auths_by_subr,cls=NpEncoder).encode('utf-8'))
    # read dictionary back in to verify success
    with gzip.open(auths_filename, 'rt') as filein:
        auths_by_subr = json.load(filein)
    # now sort dictionaries by earliest to latest date of posting, assuming 
    ## earlier dates represent users who have departed the subreddit, at least
    ## temporarily
    for sr,dct in auths_by_subr.items():     # select each subreddit dict of auths 
        auths_by_subr[sr] = OrderedDict(sorted(dct.items(), key=itemgetter(1)))
    # now rank by order of departure, then save to dict
    for sr,dct in auths_by_subr.items():     # select each subreddit dict of auths 
        ptim = 0 	# track previous time to see if it's same as current 
        pauth = '' 	# track previous author (dict index) 
        ctr = 0     # create rank ctr for author departure order 
        for au,dtim in dct.items(): 	# select each author and latest post in dict 
            if dtim != ptim:
                ctr+=1  # only incr ctr if curr timestamp differs from prev
            ptim = dtim
            pauth = au
            auths_by_subr[sr][au]=(dtim,ctr)    # add rank as part of dict
    # writing sorted and ranked dictionary of author dictionaries to disk
    with gzip.open(auths_filename2, 'w') as fileout:
        fileout.write(json.dumps(auths_by_subr,cls=NpEncoder).encode('utf-8'))
