#!/usr/bin/env python3
"""
HPC-Optimized Parallel Bias Analysis
====================================

This script is optimized for the WSU HPC cluster architecture:
- 4x Dell Poweredge R750XA with dual Intel Gold 32-core CPUs (64-thread each)
- 256GB memory per node
- Dual NVIDIA L4 cards per node
- SLURM job scheduler
- 8 compute VMs with 120GB memory and 1 GPU each

Key HPC optimizations:
- SLURM job submission with proper resource allocation
- Multi-node MPI distribution for massive parallelization
- GPU-accelerated processing where applicable
- Optimized memory usage for 120GB+ nodes
- Batch processing with checkpointing
- Network-optimized caching across nodes

Usage:
    # Submit as SLURM job
    sbatch hpc_bias_analysis.slurm --subreddits progressive republicans mlb
    
    # Run directly on allocated nodes
    srun python hpc_parallel_bias.py --subreddits progressive republicans mlb
    
    # Interactive session
    srun --pty bash
    python hpc_parallel_bias.py --subreddits progressive republicans mlb
"""

import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
from itertools import combinations
import random
import hashlib
import time
import pickle
import gzip
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import logging
import signal
import asyncio

# MPI imports for multi-node processing
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in single-node mode.")

# Import from original bias analysis
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of CWD or invocation method
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from create_conf_bias import (
    Redditor, Comment, Subreddit, read_subreddits, ALL_SUBREDDITS,
    generate_support_prompt, generate_similarity_prompt, generate_dogmatism_prompt, generate_alignment_prompt,
    evaluate_prompt, uncluster, trim_potential_think, check_ollama_connection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hpc_bias_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for MPI
comm = None
rank = 0
size = 1

# HPC-optimized cache with compression
class HPCCache:
    """High-performance cache with compression and multi-node support"""
    
    def __init__(self, cache_dir: str = "./cache/hpc_bias_cache"):
        self.cache_dir = Path(cache_dir)
        # Note: Directory must be created manually before running
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def get_cache_key(self, function_name: str, *args) -> str:
        """Generate a cache key for function calls"""
        content = function_name + "|" + "|".join(str(arg) for arg in args)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key"""
        return self.cache_dir / f"{cache_key}.pkl.gz"
    
    def get(self, function_name: str, *args):
        """Get cached result"""
        cache_key = self.get_cache_key(function_name, *args)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_path = self.get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with gzip.open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                self.memory_cache[cache_key] = result
                self.cache_hits += 1
                return result
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        
        self.cache_misses += 1
        return None
    
    def set(self, function_name: str, *args, result=None):
        """Set cached result"""
        cache_key = self.get_cache_key(function_name, *args)
        
        # Store in memory
        self.memory_cache[cache_key] = result
        
        # Store on disk with compression
        cache_path = self.get_cache_path(cache_key)
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'memory_entries': len(self.memory_cache),
            'disk_entries': len(list(self.cache_dir.glob("*.pkl.gz")))
        }

# Global cache instance
hpc_cache = HPCCache()

def cached_api_call(function_name: str, *args, **kwargs):
    """HPC-optimized cached API call"""
    # Check cache first
    cached_result = hpc_cache.get(function_name, *args)
    if cached_result is not None:
        return cached_result
    
    # Call the original function
    if function_name == "support":
        result = hpc_support(*args, **kwargs)
    elif function_name == "similarity":
        result = hpc_similarity(*args, **kwargs)
    elif function_name == "alignment":
        result = hpc_alignment(*args, **kwargs)
    elif function_name == "dogmatism":
        result = hpc_dogmatism(*args, **kwargs)
    else:
        raise ValueError(f"Unknown function: {function_name}")
    
    # Cache the result
    hpc_cache.set(function_name, *args, result=result)
    
    return result

def hpc_support(comment_context: str, *, get_tokens: bool = False):         
    """Support function with HPC caching"""
    if get_tokens:
        print('get_tokens support')
    
    # Add retry logic for model loading delays
    for attempt in range(3):
        try:
            return float(trim_potential_think(evaluate_prompt(generate_support_prompt(comment_context))))
        except Exception as e:
            if "took too long" in str(e) or "connection" in str(e).lower():
                logger.warning(f"Attempt {attempt + 1}/3 failed due to timeout/connection: {str(e)}")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(30)  # Wait 30 seconds before retrying
                continue
            else:
                raise  # Re-raise non-timeout errors immediately
    
    # If all retries failed
    raise Exception("Model loading failed after 3 attempts - model may be too large or server overloaded")

def hpc_similarity(a_context: str, b_context: str, *, get_tokens: bool = False) -> float:
    """Similarity function with HPC caching"""
    if get_tokens:
        print('get_tokens similarity')
    return float(trim_potential_think(evaluate_prompt(generate_similarity_prompt(a_context, b_context))))

def hpc_alignment(a_context: str, b_context: str, *, get_tokens: bool = False) -> float:
    """Alignment function with HPC caching"""
    if get_tokens:
        print('get_tokens alignment')
    return float(trim_potential_think(evaluate_prompt(generate_alignment_prompt(a_context, b_context))))

def _parse_dogmatism_from_response(text: str) -> tuple[str | None, str | None]:
    """Extract dogmatism_label and dogmatism_reason from model response text."""
    # Try JSON first
    try:
        data = json.loads(text)
        author_block = data.get("author") if isinstance(data, dict) else None
        if isinstance(author_block, dict):
            return author_block.get("dogmatism_label"), author_block.get("dogmatism_reason")
        # Some models may put keys at top-level
        return data.get("dogmatism_label"), data.get("dogmatism_reason")
    except Exception:
        pass
    # Fallback to regex search
    import re as _re
    m_label = _re.search(r'"dogmatism_label"\s*:\s*"([^"]+)"', text)
    m_reason = _re.search(r'"dogmatism_reason"\s*:\s*"([^"]*)"', text)
    label = m_label.group(1) if m_label else None
    reason = m_reason.group(1) if m_reason else None
    return label, reason

def hpc_dogmatism(comment_context: str) -> dict:
    """Dogmatism evaluation with HPC caching; returns label and reason."""
    resp_text = evaluate_prompt(generate_dogmatism_prompt(comment_context))
    label, reason = _parse_dogmatism_from_response(resp_text)
    return {"label": label, "reason": reason}

class HPCRedditor(Redditor):
    """HPC-optimized Redditor class"""
    
    def bias(self, *, get_tokens: bool = False):
        '''
        Generates bias coefficient in [0.5, 2.0]
        HPC-optimized with increased pairs for better accuracy
        '''
        pairs = list(combinations(self.comments_with_parents, 2))
        pairs_og_length = len(pairs)
        
        # Use more pairs on HPC due to better resources
        max_length = 50  # Increased from 20 for better accuracy
        if pairs_og_length > max_length:
            pairs = random.Random(42).choices(pairs, k=max_length)
        
        if get_tokens:
            tokens = 0
            for old_com, new_com in pairs:
                tokens += old_com.parent.alignment(new_com.parent, get_tokens=True)
                tokens += old_com.support(get_tokens=True)
                tokens += new_com.support(get_tokens=True)
            return {'pairs': len(pairs), 'tokens': tokens}
        
        if not pairs: 
            return 1.0
            
        res = 0.0
        for old_com, new_com in pairs:
            a = old_com.parent.alignment(new_com.parent)
            q1q2 = uncluster(old_com.support() * new_com.support())
            res += uncluster(q1q2 * a)
        res /= len(pairs)
        return res*0.75+1.25

class HPCComment(Comment):
    """HPC-optimized Comment class"""
    
    def support(self, *, get_tokens: bool = False) -> float:
        assert self.parent is not None
        if get_tokens:
            return hpc_support(json.dumps(self.to_context()), get_tokens=get_tokens)
        return cached_api_call("support", json.dumps(self.to_context()))
        
    def similarity(self, other: 'Comment', *, get_tokens: bool = False) -> float:
        if get_tokens:
            return hpc_similarity(json.dumps(self.to_context()), json.dumps(other.to_context()), get_tokens=get_tokens)
        return cached_api_call("similarity", json.dumps(self.to_context()), json.dumps(other.to_context()))
    
    def alignment(self, other: 'Comment', *, get_tokens: bool = False) -> float:
        if get_tokens:
            return hpc_alignment(json.dumps(self.to_context()), json.dumps(other.to_context()), get_tokens=get_tokens)
        return cached_api_call("alignment", json.dumps(self.to_context()), json.dumps(other.to_context()))

def read_subreddits_hpc(subreddit_names: list[str]) -> list[Subreddit]:
    """HPC-optimized version of read_subreddits that uses HPCRedditor and HPCComment"""
    import json
    from pathlib import Path
    
    comment_datas = []
    
    # Debug: Print current working directory
    logger.info(f"Debug: Current working directory: {Path.cwd()}")
    
    # Only allow with_likes data sources (never use filtered data)
    possible_data_dirs = [
        Path("data") / "with_likes",           # If running from reddit_2024/
        Path("../data") / "with_likes",       # If running from hpc/
        Path("../../data") / "with_likes",    # If running from hpc/cache/
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        logger.info(f"Debug: Checking if {dir_path} exists: {dir_path.exists()}")
        if dir_path.exists():
            data_dir = dir_path
            logger.info(f"Found data directory: {data_dir}")
            break
    
    if data_dir is None:
        logger.error(f"Error: Could not find data directory. Tried: {possible_data_dirs}")
        return []
    
    for subreddit_name in subreddit_names:
        # Build filename (with_likes contains the raw agree data). Support .json and .json.gz
        base_filename = f"{subreddit_name}_2019-01_agree"
        if subreddit_name == "globeskepticism":
            base_filename = f"{subreddit_name}_2020-06_agree"
        
        json_path = data_dir / f"{base_filename}.json"
        gz_path = data_dir / f"{base_filename}.json.gz"
        
        try:
            if json_path.exists():
                with open(json_path, 'r') as f:
                    comment_datas += json.load(f)
            elif gz_path.exists():
                with gzip.open(gz_path, 'rt') as f:
                    comment_datas += json.load(f)
            else:
                logger.warning(f"Warning: Could not find data file for subreddit '{subreddit_name}' at {json_path} or {gz_path}")
                continue
        except FileNotFoundError:
            logger.warning(f"Warning: Could not find data file for subreddit '{subreddit_name}' at {json_path} or {gz_path}")
            continue
    
    # Sort by creation time
    comment_datas.sort(key=lambda data: data['created_utc'])

    redditors = {}
    for comment_data in comment_datas:
        if comment_data['author_fullname'] in redditors: continue
        redditor = HPCRedditor(
            id=comment_data['author_fullname'],
            name=comment_data['author'],
            created_utc=comment_data['author_created_utc'],
            comments=[],
            comments_with_parents=[],
        )
        # Debug: Log the first few redditors to confirm HPC classes are being used
        if len(redditors) < 5:
            logger.info(f"Debug: Created HPCRedditor: {redditor.name} (type: {type(redditor).__name__})")
        redditors[redditor.id] = redditor
    
    comments = {}
    subreddits = {}
    for comment_data in comment_datas:
        comment = HPCComment(
            id=comment_data['id'],
            text=comment_data['text'],
            score=comment_data['score'],
            author=redditors[comment_data['author_fullname']],
            created_utc=comment_data['created_utc'],
            title=comment_data.get('title', None),
            parent=None,
            ancestors=None,
        )
        # Debug: Log the first few comments to confirm HPC classes are being used
        if len(comments) < 5:
            logger.info(f"Debug: Created HPCComment: {comment.id} (type: {type(comment).__name__})")
        comments[comment.id] = comment
        comment.author.comments.append(comment)

    for comment_data in comment_datas:
        if comment_data['parent_id'] is None: 
            comments[comment_data['id']].parent = None
            continue
        parent_id = comment_data['parent_id'].split('_')[1]
        if parent_id in comments:
            comments[comment_data['id']].parent = comments[parent_id]
            comments[comment_data['id']].author.comments_with_parents.append(comments[comment_data['id']])

    for comment in comments.values():
        ancestors = []
        curr_comm = comment
        curr_parent = curr_comm.parent
        while curr_parent is not None:
            ancestors.append(curr_parent)
            curr_comm = curr_parent
            curr_parent = curr_comm.parent
        ancestors.reverse() # chronological order now
        comment.ancestors = ancestors

    for comment_data in comment_datas:
        subreddit_name = comment_data['subreddit']
        if subreddit_name not in subreddits:
            subreddits[subreddit_name] = Subreddit(
                id=comment_data['subreddit_id'],
                name=subreddit_name,
                comments=[],
                members={},
                comments_with_parents=[],
            )
        
        subreddits[subreddit_name].add_comment(comments[comment_data['id']])
    
    # Debug: Print what subreddits were found
    logger.info(f"Debug: Found subreddits in data: {list(subreddits.keys())}")
    logger.info(f"Debug: Looking for subreddits: {subreddit_names}")
    
    # Check which subreddits are missing
    missing_subreddits = [name for name in subreddit_names if name not in subreddits]
    if missing_subreddits:
        logger.warning(f"Debug: Missing subreddits: {missing_subreddits}")
    
    # Return only subreddits we successfully loaded to avoid KeyErrors
    return [subreddits[name] for name in subreddit_names if name in subreddits]

def process_redditor_hpc(redditor_data):
    """Process a single redditor for HPC execution"""
    subreddit_name, worker_id, author_name = redditor_data
    
    try:
        # Re-load the subreddit data in this worker process to avoid pickling issues
        subreddit_objects = read_subreddits_hpc([subreddit_name])
        if not subreddit_objects:
            logger.error(f"Rank {rank}, Worker {worker_id}: Could not load subreddit {subreddit_name}")
            return None
        
        subreddit = subreddit_objects[0]
        
        # Locate the requested redditor by author name
        redditor = next((r for r in subreddit.members.values() if r.name == author_name), None)
        if redditor is None:
            logger.error(f"Rank {rank}, Worker {worker_id}: Author '{author_name}' not found in subreddit {subreddit_name}")
            return None
        start_time = time.time()
        
        # Calculate bias value using the bias method
        bias_value = redditor.bias()
        bias_call_data = redditor.bias(get_tokens=True)
        
        # Skip users with 0 pairs (no meaningful bias analysis possible)
        if bias_call_data['pairs'] == 0:
            logger.warning(f"Rank {rank}, Worker {worker_id}: Skipping {redditor.name} - no comment pairs available (comments_with_parents: {len(redditor.comments_with_parents)})")
            return None
        
        processing_time = time.time() - start_time
        
        # Store in the same format as bias_values_progressive.json
        try:
            node_name = os.uname().nodename  # Unix
        except AttributeError:
            import platform
            node_name = platform.node()
        bias_entry = {
            "author": redditor.name,
            "bias_seeking": bias_value,
            "tokens": bias_call_data['tokens'],
            "pairs": bias_call_data['pairs'],
            "processing_time_seconds": processing_time,
            "worker_id": worker_id,
            "rank": rank,
            "node": node_name,
        }

        # Compute dogmatism over a small sample of this author's comments
        try:
            sample_comments = redditor.comments_with_parents[:min(3, len(redditor.comments_with_parents))]
            if sample_comments:
                label_counts: dict[str, int] = {}
                score_sum = 0.0
                samples_used = 0
                label_to_score = {
                    "Deeply Rooted": 1.0,
                    "Firm but Open": 0.66,
                    "Open to Dialogue": 0.33,
                    "Flexible": 0.0,
                }
                for comment in sample_comments:
                    ctx_json = json.dumps(comment.to_context())
                    dog = cached_api_call("dogmatism", ctx_json)
                    label = dog.get("label") if isinstance(dog, dict) else None
                    if label:
                        label_counts[label] = label_counts.get(label, 0) + 1
                        score_sum += label_to_score.get(label, 0.5)
                        samples_used += 1
                if samples_used > 0:
                    # Choose most frequent label; tie-breaker arbitrary
                    top_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
                    bias_entry["dogmatism_label"] = top_label
                    bias_entry["dogmatism_score"] = round(score_sum / samples_used, 3)
        except Exception as dog_e:
            logger.warning(f"Rank {rank}, Worker {worker_id}: Dogmatism evaluation failed for {redditor.name}: {dog_e}")
        
        logger.info(f"Rank {rank}, Worker {worker_id}: Completed {redditor.name} in {processing_time:.1f}s ({bias_call_data['pairs']} pairs)")
        return bias_entry
        
    except Exception as e:
        logger.error(f"Rank {rank}, Worker {worker_id}: Error processing author '{author_name}' in subreddit {subreddit_name}: {str(e)}")
        return None

def distribute_work_across_nodes(subreddits, sample_size=25):
    """Distribute work across MPI nodes"""
    if not MPI_AVAILABLE:
        return subreddits
    
    # Distribute subreddits across nodes
    subreddits_per_node = len(subreddits) // size
    remainder = len(subreddits) % size
    
    start_idx = rank * subreddits_per_node + min(rank, remainder)
    end_idx = start_idx + subreddits_per_node + (1 if rank < remainder else 0)
    
    node_subreddits = subreddits[start_idx:end_idx]
    
    logger.info(f"Rank {rank}: Processing subreddits {start_idx}-{end_idx-1}: {node_subreddits}")
    return node_subreddits

def create_bias_files_hpc(subreddit_subset, max_workers=None, sample_size=25):
    """HPC-optimized bias file creation with multi-node support"""
    
    # Determine optimal worker count based on HPC resources
    if max_workers is None:
        # Use more workers on HPC due to better resources
        max_workers = min(mp.cpu_count(), 16)  # Increased from 4 to 16
    
    logger.info(f"Rank {rank}: Using {max_workers} parallel workers")
    
    # Path to the bias values directory
    current_dir = Path(__file__).parent
    bias_dir = (current_dir.parent / "data" / "bias")
    bias_dir.mkdir(parents=True, exist_ok=True)
    
    # Load progress tracking
    progress_file = bias_dir / f"progress_rank_{rank}.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"completed_subreddits": [], "completed_users": {}}
    
    # Process each subreddit
    for subreddit_idx, subreddit in enumerate(subreddit_subset, 1):
        logger.info(f"Rank {rank}: Processing subreddit {subreddit_idx}/{len(subreddit_subset)}: {subreddit.name}")

        # Determine authors already present from prior runs (merged file)
        merged_path = (bias_dir / f"bias_values_{subreddit.name.lower()}.json")
        existing_authors: set[str] = set()
        if merged_path.exists():
            try:
                with open(merged_path, 'r') as f:
                    prev = json.load(f)
                for rec in prev:
                    author_name = rec.get("author")
                    if author_name:
                        existing_authors.add(author_name)
            except Exception as e:
                logger.warning(f"Rank {rank}: Failed to read existing merged file for {subreddit.name}: {e}")

        # Sync progress completed_users with existing authors
        if subreddit.name not in progress["completed_users"]:
            progress["completed_users"][subreddit.name] = []
        for a in existing_authors:
            if a not in progress["completed_users"][subreddit.name]:
                progress["completed_users"][subreddit.name].append(a)

        # If we already have at least sample_size authors, mark as completed and skip
        if len(existing_authors) >= sample_size:
            logger.info(f"Rank {rank}: Already have {len(existing_authors)} users for {subreddit.name} (>= sample_size {sample_size}); skipping")
            if subreddit.name not in progress["completed_subreddits"]:
                progress["completed_subreddits"].append(subreddit.name)
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=4)
            continue

        # List to store bias data for this subreddit
        bias_data = []

        # Get all redditors and sort by distance from median comment count (mid-range users first)
        all_redditors = list(subreddit.members.values())
        
        # Filter to eligible users with >=2 comments_with_parents
        eligible_redditors = [r for r in all_redditors if len(r.comments_with_parents) >= 2]
        if not eligible_redditors:
            logger.warning(f"Rank {rank}: No eligible users (>=2 parented comments) in {subreddit.name}")
            # Still create empty rank file for traceability
            output_file = bias_dir / f"bias_values_{subreddit.name.lower()}_rank_{rank}.json"
            with open(output_file, 'w') as f:
                json.dump(bias_data, f, indent=4)
            logger.info(f"Rank {rank}: Created bias file: {output_file}")
            continue

        # Calculate median comment count among eligible
        comment_counts = [len(r.comments_with_parents) for r in eligible_redditors]
        comment_counts.sort()
        median_count = comment_counts[len(comment_counts) // 2]

        # Sort eligible by absolute distance from median (closest to median first)
        eligible_redditors.sort(key=lambda r: abs(len(r.comments_with_parents) - median_count))

        # Determine how many additional users we need beyond existing_authors
        need_n = max(0, sample_size - len(existing_authors))
        if need_n == 0:
            logger.info(f"Rank {rank}: No additional users needed for {subreddit.name}")
            continue

        # Take up to need_n from eligible, excluding those already processed in any prior run
        eligible_pool = [r for r in eligible_redditors if r.name not in existing_authors]
        if not eligible_pool:
            logger.info(f"Rank {rank}: All eligible users already processed for {subreddit.name}")
            continue
        actual_sample_size = min(need_n, len(eligible_pool))
        # Track which authors we've already attempted in this run
        tried_authors: set[str] = set(progress["completed_users"].get(subreddit.name, []))
        initial_batch = [r for r in eligible_pool if r.name not in tried_authors][:actual_sample_size]

        logger.info(f"Rank {rank}: Sampling {actual_sample_size} eligible redditors from {len(eligible_redditors)} eligible / {len(all_redditors)} total")
        
        # Filter out already completed users (rank-progress)
        redditors_to_process = []
        for redditor in initial_batch:
            if redditor.name not in progress["completed_users"].get(subreddit.name, []):
                redditors_to_process.append(redditor)
        
        if not redditors_to_process:
            logger.info(f"Rank {rank}: All users already completed for {subreddit.name}")
            continue
        
        logger.info(f"Rank {rank}: Processing users sequentially with simple skipping logic...")

        # Simple sequential processing: skip users with too few or too many parented comments
        MIN_PARENTS = 2
        MAX_PARENTS = 200

        bias_data = []
        successful_results = 0

        # Build simple candidate list excluding already processed authors
        already_done = set(progress["completed_users"].get(subreddit.name, [])) | existing_authors
        candidates = [r for r in all_redditors if r.name not in already_done]

        # Iterate candidates; skip those outside desired comment count bounds
        for idx, redditor in enumerate(candidates):
            if successful_results >= need_n:
                break

            num_parented = len(redditor.comments_with_parents)
            if num_parented < MIN_PARENTS or num_parented > MAX_PARENTS:
                continue

            start_time_user = time.time()

            try:
                bias_value = redditor.bias()
                bias_call_data = redditor.bias(get_tokens=True)

                # Require at least one pair
                if bias_call_data.get('pairs', 0) == 0:
                    continue

                # Cross-platform node name
                try:
                    node_name = os.uname().nodename
                except AttributeError:
                    import platform
                    node_name = platform.node()

                bias_entry = {
                    "author": redditor.name,
                    "bias_seeking": bias_value,
                    "tokens": bias_call_data['tokens'],
                    "pairs": bias_call_data['pairs'],
                    "processing_time_seconds": time.time() - start_time_user,
                    "worker_id": 0,
                    "rank": rank,
                    "node": node_name,
                }

                # Optional: dogmatism on small sample
                try:
                    sample_comments = redditor.comments_with_parents[:min(3, len(redditor.comments_with_parents))]
                    if sample_comments:
                        label_counts: dict[str, int] = {}
                        score_sum = 0.0
                        samples_used = 0
                        label_to_score = {
                            "Deeply Rooted": 1.0,
                            "Firm but Open": 0.66,
                            "Open to Dialogue": 0.33,
                            "Flexible": 0.0,
                        }
                        for comment in sample_comments:
                            ctx_json = json.dumps(comment.to_context())
                            dog = cached_api_call("dogmatism", ctx_json)
                            label = dog.get("label") if isinstance(dog, dict) else None
                            if label:
                                label_counts[label] = label_counts.get(label, 0) + 1
                                score_sum += label_to_score.get(label, 0.5)
                                samples_used += 1
                        if samples_used > 0:
                            top_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
                            bias_entry["dogmatism_label"] = top_label
                            bias_entry["dogmatism_score"] = round(score_sum / samples_used, 3)
                except Exception as dog_e:
                    logger.warning(f"Rank {rank}: Dogmatism evaluation failed for {redditor.name}: {dog_e}")

                bias_data.append(bias_entry)
                successful_results += 1

                # Mark as completed for this rank
                if subreddit.name not in progress["completed_users"]:
                    progress["completed_users"][subreddit.name] = []
                progress["completed_users"][subreddit.name].append(redditor.name)

                logger.info(f"Rank {rank}: Completed {redditor.name} ({successful_results}/{need_n})")

            except Exception as e:
                logger.warning(f"Rank {rank}: Error processing {redditor.name}: {e}")
                continue

        if successful_results == 0:
            logger.warning(f"Rank {rank}: No users with comment pairs found after sequential scan")
        
        # Save progress after each subreddit
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        
        # Create output filename with subreddit name and rank
        output_file = bias_dir / f"bias_values_{subreddit.name.lower()}_rank_{rank}.json"
        
        # Write the bias data to JSON file
        with open(output_file, 'w') as f:
            json.dump(bias_data, f, indent=4)
        
        # Only mark subreddit as completed if we actually processed some users
        # If we reached target users overall, mark completed; otherwise leave for future runs
        total_after = len(existing_authors) + len(bias_data)
        if total_after >= sample_size:
            if subreddit.name not in progress["completed_subreddits"]:
                progress["completed_subreddits"].append(subreddit.name)
            with open(progress_file, 'w') as f:
                json.dump(progress, f, indent=4)
            logger.info(f"Rank {rank}: Marked {subreddit.name} as completed (have {total_after} >= {sample_size})")
        else:
            logger.info(f"Rank {rank}: {subreddit.name} not yet completed (have {total_after} < {sample_size}); will continue in future runs")
        
        logger.info(f"Rank {rank}: Created bias file: {output_file}")
        logger.info(f"Rank {rank}: Processed {len(bias_data)} redditors successfully")

def merge_results_across_nodes():
    """Merge results from all MPI nodes"""
    if not MPI_AVAILABLE:
        return
    
    # Gather all results to rank 0
    bias_dir = Path(__file__).parent.parent / "data" / "bias"
    
    if rank == 0:
        logger.info("Merging results from all nodes...")
        
        # Find all rank-specific files
        rank_files = list(bias_dir.glob("bias_values_*_rank_*.json"))
        
        # Group by subreddit
        subreddit_groups = {}
        for file_path in rank_files:
            # Extract subreddit name from filename
            filename = file_path.stem
            parts = filename.split('_rank_')
            if len(parts) == 2:
                subreddit_name = parts[0].replace('bias_values_', '')
                
                if subreddit_name not in subreddit_groups:
                    subreddit_groups[subreddit_name] = []
                subreddit_groups[subreddit_name].append(file_path)
        
        # Merge each subreddit's results
        for subreddit_name, files in subreddit_groups.items():
            merged_data = []
            
            for file_path in files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    merged_data.extend(data)
            
            # Save merged file
            merged_file = bias_dir / f"bias_values_{subreddit_name}.json"
            with open(merged_file, 'w') as f:
                json.dump(merged_data, f, indent=4)
            
            logger.info(f"Merged {len(merged_data)} entries for {subreddit_name}")
            
            # Clean up rank-specific files
            for file_path in files:
                file_path.unlink()

def estimate_hpc_runtime(subreddit_subset, sample_size=25, max_workers=16):
    """Estimate runtime for HPC architecture"""
    total_users = len(subreddit_subset) * sample_size
    
    # Estimate time per user (with 50 pairs, 3 API calls each, 44s per call)
    pairs_per_user = 50  # Increased for HPC
    api_calls_per_pair = 3
    seconds_per_call = 44
    
    # With HPC caching, assume 70% of calls are cached after first few users
    cache_efficiency = 0.3  # 30% of calls are new
    
    # Account for MPI distribution
    mpi_efficiency = size if MPI_AVAILABLE else 1
    
    time_per_user = (pairs_per_user * api_calls_per_pair * seconds_per_call * cache_efficiency) / (max_workers * mpi_efficiency)
    
    total_seconds = total_users * time_per_user
    
    hours = total_seconds / 3600
    days = hours / 24
    
    logger.info(f"HPC Runtime Estimate:")
    logger.info(f"  Total users: {total_users}")
    logger.info(f"  Workers per node: {max_workers}")
    logger.info(f"  MPI nodes: {size}")
    logger.info(f"  Estimated time per user: {time_per_user/60:.1f} minutes")
    logger.info(f"  Total estimated time: {hours:.1f} hours ({days:.1f} days)")
    
    return hours

async def verify_ollama_connection():
    """Verify Ollama server is accessible before starting analysis"""
    logger.info(f"Rank {rank}: Checking Ollama server connection...")
    
    try:
        connected, result = await check_ollama_connection()
        if connected:
            logger.info(f"Rank {rank}: Ollama server connected. Available models: {result}")
            if 'qwen2.5:7b' not in result:
                logger.warning(f"Rank {rank}: Preferred model 'qwen2.5:7b' not found. Available: {result}")
                if result:  # If any models are available
                    logger.info(f"Rank {rank}: Will use fallback model: {result[0]}")
                else:
                    logger.error(f"Rank {rank}: No models available on Ollama server!")
                    return False
            return True
        else:
            logger.error(f"Rank {rank}: Cannot connect to Ollama server: {result}")
            logger.error(f"Rank {rank}: Please ensure Ollama is running: 'ollama serve'")
            logger.error(f"Rank {rank}: And the model is installed: 'ollama pull qwen2.5:7b'")
            return False
    except Exception as e:
        logger.error(f"Rank {rank}: Error checking Ollama connection: {str(e)}")
        return False

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Rank {rank}: Received signal {signum}, shutting down gracefully...")
        if MPI_AVAILABLE:
            comm.Abort()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    global comm, rank, size
    
    # Initialize MPI if available
    if MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        logger.info(f"MPI initialized: rank {rank}/{size}")
    else:
        logger.info("Running in single-node mode (no MPI)")
    
    parser = argparse.ArgumentParser(description='HPC-optimized parallel bias analysis')
    parser.add_argument('--subreddits', nargs='+', default=[],
                        help='List of subreddit names to process')
    parser.add_argument('--all', action='store_true',
                        help='Process all subreddits')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers per node (default: 16)')
    parser.add_argument('--sample-size', type=int, default=25,
                        help='Number of users to sample per subreddit (default: 25)')
    parser.add_argument('--estimate-only', action='store_true',
                        help='Only estimate runtime, don\'t run analysis')
    parser.add_argument('--cache-dir', type=str, default="./cache/hpc_bias_cache",
                        help='Cache directory (default: ./cache/hpc_bias_cache)')
    parser.add_argument('--clear-progress', action='store_true',
                        help='Clear all progress files before starting')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Initialize HPC cache
    global hpc_cache
    hpc_cache = HPCCache(args.cache_dir)
    
    # Clear progress if requested
    if args.clear_progress:
        bias_dir = Path(__file__).parent.parent / "data" / "bias"
        bias_dir.mkdir(parents=True, exist_ok=True)
        for progress_file in bias_dir.glob("progress_rank_*.json"):
            try:
                progress_file.unlink()
                logger.info(f"Cleared progress file: {progress_file}")
            except FileNotFoundError:
                # File doesn't exist, which is fine
                pass
        logger.info("All progress files cleared")
    
    # Determine which subreddits to process
    if args.all:
        selected_subreddits = ALL_SUBREDDITS
    elif args.subreddits:
        selected_subreddits = [s for s in args.subreddits if s in ALL_SUBREDDITS]
    else:
        # Default to subreddits with existing data for comparison
        selected_subreddits = ['progressive', 'republicans', 'mlb', 'democrats', 'flatearth']
    
    # Distribute work across nodes
    node_subreddits = distribute_work_across_nodes(selected_subreddits, args.sample_size)
    
    if rank == 0:
        logger.info(f"HPC Parallel Bias Analysis")
        logger.info(f"Total subreddits: {len(selected_subreddits)}")
        logger.info(f"MPI nodes: {size}")
        logger.info(f"Workers per node: {args.workers}")
        logger.info(f"Sample size: {args.sample_size} users per subreddit")
    
    # Estimate runtime
    estimated_hours = estimate_hpc_runtime(node_subreddits, args.sample_size, args.workers)
    
    if args.estimate_only:
        if rank == 0:
            logger.info("Estimate complete. Use --estimate-only to see runtime without running analysis.")
        return
    
    # Confirm before running (only on rank 0)
    if rank == 0 and estimated_hours > 24:
        response = input(f"Estimated runtime: {estimated_hours:.1f} hours. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Analysis cancelled.")
            if MPI_AVAILABLE:
                comm.Abort()
            return
    
    # Verify Ollama connection before starting
    connection_ok = asyncio.run(verify_ollama_connection())
    if not connection_ok:
        logger.error(f"Rank {rank}: Ollama connection failed. Aborting analysis.")
        if MPI_AVAILABLE:
            comm.Abort()
        return
    
    # Synchronize all nodes
    if MPI_AVAILABLE:
        comm.Barrier()
    
    # Run the analysis
    if rank == 0:
        logger.info("Starting HPC parallel bias analysis...")
    
    start_time = time.time()
    
    subreddit_objects = read_subreddits_hpc(node_subreddits)
    create_bias_files_hpc(subreddit_objects, args.workers, args.sample_size)
    
    # Synchronize before merging
    if MPI_AVAILABLE:
        comm.Barrier()
    
    # Merge results on rank 0
    merge_results_across_nodes()
    
    total_time = time.time() - start_time
    
    if rank == 0:
        logger.info("Analysis complete!")
        logger.info(f"Total runtime: {total_time/3600:.1f} hours")
        logger.info(f"Actual vs estimated: {total_time/3600:.1f}h vs {estimated_hours:.1f}h")
        
        # Print cache statistics
        cache_stats = hpc_cache.stats()
        logger.info(f"Cache statistics: {cache_stats}")

if __name__ == "__main__":
    main()
