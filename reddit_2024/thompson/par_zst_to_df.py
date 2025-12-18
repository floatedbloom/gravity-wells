import zstandard as zstd    # for .zst compressed files
import argparse     # for parsing commandline arguments
import json     # for processing json files
import pandas as pd     # use pandas Python Data Analysis Library
import pathlib  # for current execution directory info
from multiprocessing import Pool    # multiprocessor pool
from multiprocessing import cpu_count   # get cpu count
import sys      # for sys.exit and sys.argv to check number of arguments
import datetime as dt
from itertools import repeat    # ease replicating parameters in zip function

def process_line(line, subrs, keys):
    '''Method to process a line from decompressed .zst file contents'''
    try:
        obj = json.loads(line)
        keys = keys & obj.keys()    # ensure keys are in the returned object
        # ensure it is a text post if it is a submission (vs comment)
        txt_bool = obj['is_self'] if 'is_self' in obj else True
        
        if (obj['subreddit'] in subrs and \
            obj['author'] != '[deleted]' and \
            txt_bool):
            return pd.DataFrame([{k: obj[k] for k in keys}])
        return None
    except Exception as e:
        # Return None on error instead of crashing
        return None

def decompress_file_chunks(zstfp,posts_df,subrs,keys_to_keep):
    ''' 
    Method to read in file chunks from compressed .zst file and store in
    specified dataframe. Dataframe is updated, retaining any data it contains.
    Code snippet from Watchful at:
    https://www.reddit.com/r/pushshift/comments/ajmcc0/information_and_code_examples_on_how_to_use_the/ef012vk/
    '''
    processes = cpu_count()     # identify nbr of processors available
    print(f'\nUtilizing {processes} cores')
    print(f'\n\tStarted with {posts_df.shape} dataframe')
    pool = Pool(processes)      # create pool of processes
    with open(zstfp, 'rb') as fh:
        try:
            dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
            with dctx.stream_reader(fh) as reader:
                chunk_ctr = 1   # show progress when processing chunks
                previous_line = ""
                print(f'\tProcessing chunks for {zstfp} file ... ' + 
                        dt.datetime.now().strftime( "%Y/%m/%d %H:%M:%S"))
                #ctr = 0     #jet debug
                #while ctr<5000000:  #jet debug
                #if True:    # added to leave indents with 'while' commented out
                while True:
                    try:
                        chunk = reader.read(2**24)  # 16mb chunks
                        if not chunk:
                            break
                        
                        print(f'\t\tProcessing chunk number: {chunk_ctr:10d}', end='\r')
                        chunk_ctr += 1
                        string_data = chunk.decode('utf-8')
                        lines = string_data.split("\n")
                        lines[0] = previous_line + lines[0]
                        
                        # Filter out None results from failed processing
                        results = [r for r in pool.starmap(process_line, 
                                    zip(lines[:-1], repeat(subrs), repeat(keys_to_keep))) 
                                 if r is not None]
                        
                        if results:
                            posts_df = pd.concat([posts_df] + results, ignore_index=True)
                        
                        previous_line = lines[-1]
                        
                    except Exception as e:
                        print(f"\n\tError processing chunk {chunk_ctr}: {str(e)}")
                        continue
        except Exception as e:
            print(f"\n\tError opening/reading file: {str(e)}")
            pool.close()
            pool.join()
            return posts_df
            
    pool.close()
    pool.join()
    print(f'\n\tProcessed {posts_df.shape} dataframe from file chunks.')
    print("\n\tFinished chunks: " + 
            dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    return posts_df
