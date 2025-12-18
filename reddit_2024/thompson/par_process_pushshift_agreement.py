'''
Parallel process specified month and year .zst comment and submission archive 
files for specified subreddit, examining (dis)agreement among author posts and 
comments with entire subreddit. Search .zst archive files for subreddit posts 
and comments, then evaluate (dis)agreement among all posts for that month for 
that subreddit. If json file already exists, reads processed data from file.
Must call script from commandline with year, month, and subreddit name as 
parameters. The script expects to run in same directory where files will be 
accessed and stored. This script is a precursor to the "sr_sim_ec.py" script.
'''

import os
import thompson.par_zst_to_df as pz     # local module for converting zst to dataframe
import praw     # for accessing reddit feeds
import zstandard as zstd    # for .zst compressed files
import argparse     # for parsing commandline arguments
import json     # for processing json files
import pandas as pd     # use pandas Python Data Analysis Library
import datetime as dt
import pathlib  # for current execution directory info
import sys      # for sys.exit and sys.argv to check number of arguments
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity  # for similarity calc

def construct_output_path(filename):
    if my_args.output_dir:
        # Ensure directory exists
        pathlib.Path(my_args.output_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(my_args.output_dir, filename)
    else:
        return filename

def get_sr_description(subreddit,fp):
    #Method to retrieve subreddit description from reddit and save to file
    secret = "hK8rMgJ801dpo6sMvpCqKyvwqXDMeA"
    cid = "meUVxGWvFqe9xTaDZsWvsQ"
    user = "EC-desc-scraper"

    # create read-only app instance
    reddit_ro = praw.Reddit(client_id=cid,
            client_secret=secret,
            user_agent=user)
    sr = reddit_ro.subreddit(subreddit)
    fp.write_text(sr.public_description) # write subreddit description to file

def num_users(subreddit):
    #this is used for M in M/r^2 lists
    secret = "hK8rMgJ801dpo6sMvpCqKyvwqXDMeA"
    cid = "meUVxGWvFqe9xTaDZsWvsQ"
    user = "EC-desc-scraper"
    reddit_ro = praw.Reddit(client_id=cid,client_secret=secret,user_agent=user)
    sr = reddit_ro.subreddit(subreddit)
    unique_users = set()
    #change top, new, etc.
    for post in sr.top(limit=None):
        if post.author:
            unique_users.add(post.author.name)
    return len(unique_users)

def get_top_posts(subreddit,fp):
    '''Method to make a big string of the top number_posts'''
    secret = "hK8rMgJ801dpo6sMvpCqKyvwqXDMeA"
    cid = "meUVxGWvFqe9xTaDZsWvsQ"
    user = "EC-desc-scraper"
    number_posts = 20

    # create read-only app instance
    reddit_ro = praw.Reddit(client_id=cid,client_secret=secret,user_agent=user)
    sr = reddit_ro.subreddit(subreddit)

    #fetch top posts
    top_posts = sr.top(limit=number_posts)

    #make array of the top post content
    all_text = []
    for post in top_posts:
        title = post.title.replace(" ","").lower()
        selftext = post.selftext.replace(" ","").lower() if post.selftext else ''
        post_content = f"{title}{selftext}"
        all_text.append(post_content)
    big_string = "".join(all_text)

    fp.write_text(big_string, encoding='utf-8') # write string to file

'''def get_list_squared(subreddit):
    #make list of m/r^2 values for subreddit agents
    secret = "hK8rMgJ801dpo6sMvpCqKyvwqXDMeA"
    cid = "meUVxGWvFqe9xTaDZsWvsQ"
    user = "EC-desc-scraper"
    #reddit instance
    reddit_ro = praw.Reddit(client_id=cid,client_secret=secret,user_agent=user)
    sr = reddit_ro.subreddit(subreddit)
    #put agents values into list
    M = num_users(subreddit)
    agents = []
    for diss_affinity_ratio in auth_df['diss_affinity_ratio']:
        value = M / (diss_affinity_ratio)^2
        agents.append(value)

def get_list_one(subreddit):
    #make list of m/r values for subreddit agents
    secret = "hK8rMgJ801dpo6sMvpCqKyvwqXDMeA"
    cid = "meUVxGWvFqe9xTaDZsWvsQ"
    user = "EC-desc-scraper"
    #reddit instance
    reddit_ro = praw.Reddit(client_id=cid,client_secret=secret,user_agent=user)
    sr = reddit_ro.subreddit(subreddit)
    #put agents values into list
    M = num_users(subreddit)
    agents = []
    for diss_affinity_ratio in auth_df['diss_affinity_ratio']:
        value = M / (diss_affinity_ratio)
        agents.append(value)'''

def create_std_text_col(posts_df):
# Create text column with strings from either selftext, title, or body
    # First set text column for all rows to body, as there are usually more
    ## comments than submissions
    if 'body' in posts_df.columns:
        posts_df['text']=posts_df.body
    # now set all rows from submissions to selftext column
    if 'selftext' in posts_df.columns:
        posts_df.loc[posts_df['text'].isna(), 'text'] = posts_df.selftext
    # finally, set all rows with no selftext to title column
    elif 'title' in posts_df.columns:
        posts_df.loc[(posts_df['text']=='') | (posts_df['text']=='[removed]'), 
            'text'] = posts_df.title
# Insert entry in first row containing the subreddit description as text
# Will use similarity with description to determine author affinity
    tdf = pd.DataFrame([[subr,subr,subr_desc]], columns=['author',
        'subreddit','text'])
    posts_df = pd.concat([tdf,posts_df]).reset_index(drop=True)

BATCH = 8   # batch size of 8 seems imperically most efficient for intuition nodes

# Begin processing for call from command line with optional output folder name
if __name__ == "__main__":
    if not len(sys.argv) > 1:
        sys.exit("Must call script from commandline with year, month, and " +
                "subreddit name as parameters.")
    print("\nBegan: " + dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

# Use auto python doc description
    my_parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)                     
# process zst file parameter
    my_parser.add_argument('year',
            help="specify 4-digit year of .zst files to be decompressed and " +
            "streamed",
            type=str) 
    my_parser.add_argument('month',
            help="specify 2-digit month of .zst files to be decompressed " +
            "and streamed",
            type=str) 
    my_parser.add_argument('subreddit',
            help="specify subreddit name to be analyzed",
            type=str) 
    # output directory
    my_parser.add_argument('-o', '--output_dir',
        default='F:\\',
        help="specify the output directory for the processed files",
        type=str,
        required=False)
    my_parser.add_argument('-b', '--batch',
        default=BATCH,
        help="specify batch size for BERT sentencetransformer " +
            "(default: '%(default)s')",
        type=int,
        required=False)                                                             
    my_args = my_parser.parse_args() 
    subr = my_args.subreddit
    batch_sz = my_args.batch
    posts_df = pd.DataFrame()   # create DF for storing posts
    
    # retrieve subreddit description from file or reddit, as appropriate
    desc_fp = pathlib.Path(construct_output_path(subr + '_desc.txt'))
    if not desc_fp.is_file():
        get_sr_description(subr,desc_fp)
    subr_desc = desc_fp.read_text()
    print(f'{subr}:\t{subr_desc}\n')

    #retrieve top posts
    tops_fp = pathlib.Path(construct_output_path(subr + '_top_posts.txt'))
    if not tops_fp.is_file():
        get_top_posts(subr,tops_fp)
    big_string = tops_fp.read_text()

    try:
        # Create author data json file from supplied arguments
        auths_fp = pathlib.Path(construct_output_path(subr + '_' + my_args.year + '-' + my_args.month + '_auths_agree.json.gz'))# compress file
        auths_fp.touch(exist_ok=False)   # create file to mark output  author file
    except FileExistsError:
        # if file already exists, issue warning and exit
        print(f"\t*** File {auths_fp} already exists! Move or delete file and retry. ***")
        raise SystemExit
    
    try:
        # Create found posts json file from supplied arguments
        agree_fp = pathlib.Path(construct_output_path(subr + '_' + my_args.year + '-' + my_args.month + '_agree.json.gz'))# compress file
        agree_fp.touch(exist_ok=False)   # create file to mark eventual output file
    except FileExistsError:
        # if file already exists, read it in rather than parsing zst files
        print(f"\tReading dataframe from {agree_fp}")
        with open(agree_fp, 'rb') as f:  # need 'rb' for zipped files!
            posts_df = pd.read_json(f,compression='gzip')
        print(f'\tRead {posts_df.shape} dataframe from json file {agree_fp}.')
    else:   # otherwise parse zst files for required data
        print(f"\tCreating dataframe for output to {agree_fp}")
        rs_zstfp = pathlib.Path("RS_"+my_args.year+"-"+my_args.month+".zst")
        if not rs_zstfp.exists():
            raise FileNotFoundError(f'zstfile {rs_zstfp} does not exist!')
        rc_zstfp = pathlib.Path("RC_"+my_args.year+"-"+my_args.month+".zst")
        if not rc_zstfp.exists():
            raise FileNotFoundError(f'zstfile {rc_zstfp} does not exist!')
        print(f'Decompressing data from {rs_zstfp} and {rc_zstfp}.')
        for zstfp in [rs_zstfp,rc_zstfp]:
            if zstfp == rs_zstfp:
                # columns to be saved from RS input file
                keys_to_keep = ['author', 'author_created_utc','author_fullname',
                    'created_utc','id','is_self','permalink','subreddit',
                    'subreddit_id','selftext','title']
            else:
                # columns to be save from RC input file
                keys_to_keep = ['author', 'author_created_utc','author_fullname',
                    'created_utc','id','permalink','subreddit','subreddit_id',
                    'parent_id','body']
            posts_df = pz.decompress_file_chunks(zstfp,posts_df,[subr],keys_to_keep)
        create_std_text_col(posts_df)
        # drop None text entries
        posts_df.dropna(subset=['text'],inplace=True)
        # Write found subreddit posts to json file
        print(f'\nWriting {posts_df.shape} dataframe to json file {str(agree_fp)}.')
        posts_df.to_json(agree_fp, orient='records', compression='infer')
        
    finally:
        # Process posts for input into echo chamber model
        ## i.e. run Bert on all documents to measure similarity
        print("\n\tProcess dataframe for echo chamber model: " + 
                dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
        print('jet - exiting for testing...')
        #sys.exit() 
        #^uncomment to prevent intensive bert stuff

        '''documents = posts_df.text
        model_name =  'all-mpnet-base-v2'
        print(f"\n\tmodel = SentenceTransformer({model_name}),batch={batch_sz}")
        model = SentenceTransformer(model_name)
        text_embeddings = model.encode(documents, batch_size = batch_sz, 
                show_progress_bar = True)
        similarities = cosine_similarity(text_embeddings)'''
        documents = posts_df['text'].tolist()
        model_name =  'all-mpnet-base-v2'
        print(f"\n\tmodel = SentenceTransformer({model_name}),batch={batch_sz}")
        model = SentenceTransformer(model_name)
        post_embeddings = model.encode(documents, batch_size=batch_sz, show_progress_bar = True)
        top_post_embedding = model.encode([big_string],batch_size=1)
        similarities = cosine_similarity(post_embeddings,top_post_embedding)

        # change similarity range to [0,1]
        mns = np.min(similarities)     # get bottom of range
        if mns < 0:     # if bottom of range is less than zero
            similarities = similarities - mns
        mxs = np.max(similarities)     # get top of range
        if mxs > 1:     # if top of range is greater than one
            similarities = similarities/mxs
        # Get number of items in upper triangle (minus main diagonal)
        srow_len = np.size(similarities,1)
        # Get number of items in upper triangle (minus main diagonal)
        nbr_sims = srow_len*(srow_len-1)/2
        # Find upper and lower triangles of similarities matrix (minus diag)
        sim_ut = np.triu(similarities,1)
        sim_lt = np.tril(similarities,-1)
        sult = sim_ut + sim_lt  # Create sim array with zero diagonal
        # Calculate the average overall similarity
        s_avg = sim_ut.sum()/nbr_sims
        # Calculate average similarity and dissimilarity for each document
        s_doc_avg = np.array([r.mean() for r in sult])
        d_doc_avg = 1 - s_doc_avg
        # Find all (unique) authors in database
        #authors = list(posts_df[:100].author.unique()) #testing
        authors = list(posts_df.author.unique())
        # Calculate average sim & dissimilarity for each author and store in dataframe
        auth_df = pd.DataFrame()
        auth_df['author'] = authors
        auth_df['avg_sim'] = [s_doc_avg[posts_df.author[posts_df.author==
            auth].index].mean() for auth in authors]
        auth_df['avg_diss'] = [d_doc_avg[posts_df.author[posts_df.author==
            auth].index].mean() for auth in authors]
        # Try diss to sim ratio for each author and store in DF
        auth_df['diss_sim_ratio'] = auth_df['avg_diss']/auth_df['avg_sim']
        '''# Use similarity of authors' posts to description to determine affinity
        auth_df['affinity'] = [similarities[0][posts_df.author[posts_df.
            author==auth].index].mean() for auth in authors]'''
        # similarity w big string is affinity
        top_post_embedding = model.encode([big_string], batch_size=1)
        auth_df['affinity'] = [cosine_similarity([top_post_embedding], post_embeddings[posts_df.author == auth].mean(axis=0).reshape(1, -1))[0][0] for auth in authors]

        auth_df['affinity'] = [similarities[0][posts_df.author[posts_df.author==auth].index].mean() for auth in authors]
        # Try diss to affinity ratio for each author and store in DF
        auth_df['diss_affinity_ratio'] = auth_df['avg_diss']/auth_df['affinity']

        # Write author data to json file for input to sim_ec.py
        print(f'\nWriting {auth_df.shape} dataframe to json file {str(auths_fp)}.')
        auth_df.to_json(auths_fp, orient='records', compression='infer')

        print("\nFinished: " + dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))