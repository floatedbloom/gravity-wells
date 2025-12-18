''' 
Calculate toxicity using the Google API Perspective Comment Analyzer.  
Reads Google API key from "tox_key.txt" file in current directory
'''
from googleapiclient import discovery
import json     # for json files
import pandas as pd     # for dataframes
import numpy as np  # for np.nan
import pathlib  # for path/file specification and manipulation
import argparse     # for parsing commandline arguments
import datetime     # for throttling Google App requests
from time import sleep     # time.sleep for throttling
from tqdm import tqdm   # Progress Bar Made Easy
import sys  # for sys.exit and sys.argv
from ratelimit import limits, sleep_and_retry

# Define variables to keep track of throttling
# 30 calls per minute
CALLS = 60  # number of calls per minute
RATE_LIMIT = 60     # time to wait when calls exceeded
TXT_LIMIT = 20480   # text length limit for Google API query

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def check_limit():
    ''' Empty function just to check for calls to API '''
    return

@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def get_post_attr(txt: str) -> (float,float,float):
    ''' analyze text for toxicity '''
    analyze_request = {
            'comment': { 'text': txt},
            'languages':[lang],
            'requestedAttributes': {
                'TOXICITY': {},
                'SEVERE_TOXICITY': {},
                'INSULT': {},
                'THREAT': {},
                }
            }
    # get the attributes for this text
    response = client.comments().analyze(body=analyze_request).execute() 
    # parse the results
    tox=response['attributeScores']['TOXICITY']['summaryScore']['value']
    sev_tox=response['attributeScores']['SEVERE_TOXICITY']['summaryScore']['value']
    insult=response['attributeScores']['INSULT']['summaryScore']['value']
    threat=response['attributeScores']['THREAT']['summaryScore']['value']
    return (tox,sev_tox,insult,threat)

with open('tox_key.txt') as f:
    API_KEY = f.read()
lang = "en"

if __name__ == "__main__":
    if not len(sys.argv) > 1:
        sys.exit("Must call script from commandline with filename as parameter.")

    else:
# Use auto python doc description
        my_parser = argparse.ArgumentParser(description=__doc__, 
            formatter_class=argparse.RawDescriptionHelpFormatter)                     
        my_parser.add_argument('filename',
            help="specify 'filename' from which to retrieve posts",
            type=str)
        my_parser.add_argument('-t', '--test',dest='test',action='store_true')
        my_parser.add_argument('-nt','--no-test',dest='test',action='store_false')
        my_parser.set_defaults(test=False)
        my_args = my_parser.parse_args()

        fp = pathlib.Path(my_args.filename)
        with open(fp, 'rb') as f:  # need 'rb' for zipped files!
            posts_df = pd.read_json(f,compression='gzip')
            print(f'\tRead {posts_df.shape} dataframe from json file {fp}.')
        if my_args.test:    # truncate posts_df for testing
            posts_df = posts_df[:100]
# Limit text column to limit required by Google API
        posts_df['text']=posts_df['text'].str.slice(0,TXT_LIMIT)
        new_cols = ['tox','sev_tox','insult','threat']
        posts_df[new_cols] = np.nan     # add new columns

# create client connection to Google API Perspective library
        client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=API_KEY,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/" + \
                        "$discovery/rest?version=v1alpha1",
                static_discovery=False,
        )
# calculate toxicity for every posting in dataframe
        tqdm.pandas(desc='Get toxicity',colour='blue')   # construct progress bar
# use progress_apply from tqdm to show progress
        posts_df.update(posts_df['text'].progress_apply(lambda txt: 
                pd.Series(dict(zip(new_cols,get_post_attr(txt))))))
# write updated dataframe to file
        if my_args.test:    # write test results to another file
            fp = pathlib.Path('test_tox.json.gz')
        posts_df.to_json(fp, orient='records', compression='infer')
