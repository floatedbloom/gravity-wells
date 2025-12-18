#!/usr/bin/env python3
"""
Filter JSON files to only include users with sufficient comment pairs for bias analysis
"""

import json
import gzip
from pathlib import Path
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_comment_reply(parent_id):
    """Check if parent_id is a reply to another comment (t1_) vs a post (t3_)"""
    return parent_id and parent_id.startswith('t1_')

def filter_subreddit_file(input_file, output_file, min_comment_pairs=2):
    """
    Filter a subreddit JSON file to only include users with sufficient comment pairs
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output filtered JSON file
        min_comment_pairs: Minimum number of comment pairs required (default: 2)
    """
    logger.info(f"Processing {input_file}")
    
    # Read input file
    try:
        if input_file.suffix == '.gz':
            with gzip.open(input_file, 'rt', encoding='utf-8') as f:
                comments = json.load(f)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {input_file}: {e}")
        return False
    
    logger.info(f"Loaded {len(comments)} comments from {input_file}")
    
    # Group comments by author
    author_comments = defaultdict(list)
    for comment in comments:
        author = comment.get('author')
        if author and author != '[deleted]' and author != 'AutoModerator':
            author_comments[author].append(comment)
    
    logger.info(f"Found {len(author_comments)} unique authors")
    
    # Count comment pairs per author
    analyzable_authors = set()
    author_stats = {}
    
    for author, author_comment_list in author_comments.items():
        # Count comments that are replies to other comments (t1_ prefix)
        comment_replies = [c for c in author_comment_list if is_comment_reply(c.get('parent_id'))]
        
        # Calculate number of possible pairs
        num_pairs = len(comment_replies) * (len(comment_replies) - 1) // 2
        
        author_stats[author] = {
            'total_comments': len(author_comment_list),
            'comment_replies': len(comment_replies),
            'pairs': num_pairs
        }
        
        if num_pairs >= min_comment_pairs:
            analyzable_authors.add(author)
    
    logger.info(f"Found {len(analyzable_authors)} authors with {min_comment_pairs}+ comment pairs")
    
    # Filter comments to only include analyzable authors
    filtered_comments = []
    for comment in comments:
        author = comment.get('author')
        if author in analyzable_authors:
            filtered_comments.append(comment)
    
    logger.info(f"Filtered to {len(filtered_comments)} comments from analyzable authors")
    
    # Write filtered file
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_comments, f, indent=2)
        logger.info(f"Saved filtered data to {output_file}")
        
        # Save statistics
        stats_file = output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'input_file': str(input_file),
                'output_file': str(output_file),
                'total_comments': len(comments),
                'total_authors': len(author_comments),
                'analyzable_authors': len(analyzable_authors),
                'filtered_comments': len(filtered_comments),
                'min_comment_pairs': min_comment_pairs,
                'author_stats': author_stats
            }, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write {output_file}: {e}")
        return False

def main():
    """Main function to filter all subreddit files"""
    
    # Define paths
    data_dir = Path("data")
    with_likes_dir = data_dir / "with_likes"
    filtered_dir = data_dir / "filtered_analyzable"
    
    if not with_likes_dir.exists():
        logger.error(f"Source directory {with_likes_dir} does not exist")
        return
    
    # Create output directory
    filtered_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {filtered_dir}")
    
    # Find all JSON files in with_likes directory
    json_files = list(with_likes_dir.glob("*.json")) + list(with_likes_dir.glob("*.json.gz"))
    
    if not json_files:
        logger.error(f"No JSON files found in {with_likes_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for input_file in json_files:
        # Create output filename
        if input_file.suffix == '.gz':
            output_name = input_file.stem  # Remove .json.gz
        else:
            output_name = input_file.stem  # Remove .json
        
        output_file = filtered_dir / f"{output_name}_filtered.json"
        
        if filter_subreddit_file(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Create summary file
    summary_file = filtered_dir / "filtering_summary.json"
    summary = {
        "source_directory": str(with_likes_dir),
        "output_directory": str(filtered_dir),
        "files_processed": len(json_files),
        "successful": successful,
        "failed": failed,
        "min_comment_pairs_required": 2
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
