import json
import os
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import asyncio
import re
from itertools import combinations
import ollama
import numpy as np
import random
import hashlib
# Global cache for API calls
_api_cache = {}

def get_cache_key(function_name: str, *args) -> str:
    """Generate a cache key for function calls"""
    # Create a hash of the function name and arguments
    content = function_name + "|" + "|".join(str(arg) for arg in args)
    return hashlib.md5(content.encode()).hexdigest()

def load_cache():
    """Load cache from disk if it exists"""
    global _api_cache
    cache_file = Path("api_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                _api_cache = json.load(f)
            print(f"Loaded {len(_api_cache)} cached results")
        except:
            _api_cache = {}

def save_cache():
    """Save cache to disk"""
    cache_file = Path("api_cache.json")
    with open(cache_file, 'w') as f:
        json.dump(_api_cache, f)
    print(f"Saved {len(_api_cache)} cached results")

def cached_api_call(function_name: str, *args, **kwargs):
    """Wrapper for API calls with caching"""
    cache_key = get_cache_key(function_name, *args)
    
    if cache_key in _api_cache:
        return _api_cache[cache_key]
    
    # Call the original function
    if function_name == "support":
        result = support(*args, **kwargs)
    elif function_name == "similarity":
        result = similarity(*args, **kwargs)
    elif function_name == "alignment":
        result = alignment(*args, **kwargs)
    else:
        raise ValueError(f"Unknown function: {function_name}")
    
    # Cache the result
    _api_cache[cache_key] = result
    
    # Save cache periodically (every 100 new entries)
    if len(_api_cache) % 100 == 0:
        save_cache()
    
    return result

DOGMATISM_SYSTEM_PROMPT = """
### Introduction

**Objective**: Analyze Reddit conversations to identify the stance of a specific author on sociopolitical topics and determine that author’s level of dogmatism.
**Stance Definition**: Stance is the expression of the author’s standpoint and judgement toward a given topic.
**Dogmatism Definition**: Dogmatism is an opinion strongly believed as fact, offered without question or allowance for conversation.
**Task**: Given a JSON-formatted Reddit snippet (the author’s *message*, its *parent*, and preceding *ancestors*), classify the stance shown in the **message** by assigning one of five predefined stance labels: `strongly_against`, `somewhat_against`, `somewhat_in_favor`, `strongly_in_favor`, `stance_not_inferrable`. Also assign one dogmatism label for the author using one of four predefined labels: `Deeply Rooted`, `Firm but Open`, `Open to Dialogue`, `Flexible`.

### Description of Stance Labels

1. **strongly\_against / strongly\_in\_favor**: Marks text showing strong opinions, emotional expressions, or argumentative tones.
2. **somewhat\_against / somewhat\_in\_favor**: Identifies texts with openness to discussion, less certainty, or interest in other viewpoints.
3. **stance\_not\_inferrable**: Use for texts that are neutral, support both stances, or where stance is unclear despite being on-topic.

### Description of Dogmatism Labels

1. **Deeply Rooted**: Reflects a strong, unchangeable belief—firm and unlikely to be swayed.
2. **Firm but Open**: Steadfast yet not dismissive of others; unlikely to change but allows dialogue.
3. **Open to Dialogue**: Holds an opinion while genuinely considering other viewpoints.
4. **Flexible**: Not firmly committed; readily changes opinion when presented with new information.

### Input Data Format

All elements (*ancestors*, *parent*, *message*) can be either a **post** (fields `post_title`, `post_text`) or a **comment** (field `comment_text`).

```json
{
  "id": "<message_id>",              // ID of the target comment
  "ancestors": [                     // earlier context, oldest → newest (may be empty or truncated)
    {
      "timestamp": 1547200000,
      "author_id": "t2_abc123",
      "post_title": "<optional_if_post>",
      "post_text": "<optional_if_post>",
      "comment_text": "<optional_if_comment>"
    }
  ],
  "parent": {                        // direct parent of the target message (post or comment)
    "timestamp": 1547210000,
    "author_id": "t2_def456",
    "post_title": "<optional_if_post>",
    "post_text": "<optional_if_post>",
    "comment_text": "<optional_if_comment>"
  },
  "message": {                       // latest comment by the focus author
    "timestamp": 1547220000,
    "author_id": "<focus_author_id>",
    "comment_text": "<comment_body>"
  }
}
```

*If the parent is the root post, `ancestors` will be an empty array.*

### Output Data Format

Submit annotations in JSON, grouping stance annotations under `stance_annotations`. Provide the dogmatism label and its justification under `dogmatism_label` and `dogmatism_reason`.

```json
{
  "author": {
    "name": "<focus_author_id>",
    "stance_annotations": [
      {
        "id": "<message_id>",
        "label": "<chosen_label>",
        "reason": "<Justification in ≤50 words>"
      }
    ],
    "dogmatism_label": "<chosen_dogmatism_label>",
    "dogmatism_reason": "<Justification in ≤50 words>"
  }
}
```

### Instructions for Effective Annotation

1. **Labeling Stance**: Examine only `message.comment_text`. Assign the stance label that best represents the author’s expressed viewpoint.
2. **Providing Justification**: For each stance label, supply a concise (≤ 50-word) reason highlighting tone, certainty, and argumentative cues.
3. **Dogmatism Assessment**: Evaluate the overall tenor of `message.comment_text` and select one dogmatism label. Provide a ≤ 50-word justification referencing certainty, openness, or invitational language.
4. **Use Context Wisely**: Refer to *parent* and *ancestors* solely to understand the topic being addressed; **do not** label those texts.
5. **Keep Format Exact**: Output must match the JSON schema above—no extra keys, no missing fields.
"""


def generate_dogmatism_prompt(comment_context: str):
    return [
        {
            'role': 'system',
            'content': DOGMATISM_SYSTEM_PROMPT,
        },
        {
            'role': 'user',
            'content': '' + comment_context,
        },
    ]


def generate_support_prompt(comment_context: str):
    return [
        {
            'role': 'system',
            'content':'''
                You will receive three JSON objects. "message", will
                be a comment in a Reddit thread. "parent" is the message's
                direct parent. To provide context, "ancestors" is a list of the
                previous post/comments in that thread in chronological order
                (some thread items might be missing however).

                Your task is to evaluate the extent to which "message" supports
                its "parent" comment.

                Output a confidence number in [-1, 1], with 1 meaning the comment completely
                supports the list, -1 meaning the comment completely rejects/opposes the list, and
                0 meaning completely neutral.
                
                If a comment neither supports nor rejects a parent, respond with 0.
                If a comment even slightly supports a parent, respond with on of the following:
                    0.2: very slight support, likely has some qualms but overall still supportive
                    0.4: noticeably supportive, but not enthusiastic and likely very measured instead
                    0.6: very supportive, doesn't have any qualms with the parents but is still somewhat measured in the response
                    0.8: extremely supportive, no qualms with the parents and enthusiastic about supporting them
                    1.0: completely supportive, extremely enthusiastic in supporting the parents and praises the parents an incredible amount
                If a comment even slightly rejects/opposes a parent, respond with one of the following:
                    -0.2: very slight opposition, likely has some points it agrees with the parents on but overall still rejecting
                    -0.4: noticeably opposing, but not enthusiastic in hating the parents but instead opposes them in a measured way
                    -0.6: very opposing, doesn't agree in any way with the parents but is still somewhat measured in the response
                    -0.8: extremely opposing,  doesn't agree in any way with parents and enthusiastic about opposing them
                    -1.0: completely opposing, extremely enthusiastic in opposing the parents and chastises the parents an incredible amount

                Respond with the number only. Do not include any labels, words, explanations, or punctuation. Just the number.
            ''',
        },
        {
            'role': 'user',
            'content': '' + comment_context,
        },
    ]

def generate_similarity_prompt(comment_context1: str, comment_context2: str):
    return [
        {
            'role': 'system',
            'content': '''
                You will receive two JSON objects, each representing a comment
                in a Reddit thread. The objects have three parameters:
                "message", the comment's text and metadata, "parent", the
                comment's direct parent data, and ancestors, data of further up
                ancestors in the thread. Within both objects the info is
                organized in chronological order.

                Your task is to evaluate the extent to which the two comments'
                points are *similar*. Output a confidence number in [1, 10],
                with 1 meaning the comments' points are completely similar, 10
                meaning the comments' points are completely different, and 5
                meaning sort of similar and sort of different. You might use the
                parent and ancestor thread for context.
                
                If the comments' points similarity are equal to their dissimilarity, respond with 5.
                
                If the comments' points are more similar than different, respond with
                1 = completely similar
                2 = almost completely similar
                3 = significantly more similar than different
                4 = slightly more similar than different
                5 = equally similar and different
                If the comments' points are more different than similar, respond with
                6 = slightly more different than similar
                7 = modestly more different than similar
                8 = significantly more different than similar
                9 = almost completely different
                10 = completely different
                
                Respond with the number only. Do not include any labels, words, explanations,
                or punctuation. Just the number.
            ''',
        },
        {
            'role': 'user',
            'content': '' + comment_context1,
        },
        {
            'role': 'user',
            'content': '' + comment_context2,
        },
    ]

def generate_alignment_prompt(comment_context1: str, comment_context2: str):
    return [
        {
            'role': 'system',
            'content': '''
                You will receive two JSON objects, each representing a focused
                comment in a Reddit thread. The objects have three parameters:
                "message", the comment's text and metadata, "parent", the
                comment's direct parent data, and ancestors, data of further up
                ancestors in the thread. Within both objects the info is
                organized in chronological order.

                Your task is to measure alignment between the focused comments'
                opinions, similar to a dot product in many dimensions. Given two
                strings COMMENT 1 and COMMENT 2, return a number in the set {-1,
                -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1}. You might use
                the parent and ancestor thread for context.

                Remember that we are measuring alignment between comment
                opinions, not necessarily how similar the comments are.
                Therefore, completely ignore facts and focus on opinions. The
                tone of comments is also important. However, you may consider
                slightly consider the effects of inclusions/exclusions of facts,
                but not much. Even if a comment does not contain much
                information, it could reveal the commentor's strong opinions
                beneath. We are looking to compare alignment of the commentor's
                opinions revealed through their comments. Opinions have a
                direction in many dimensions, aka alignment. If the comments
                have multiple opinions, you basically want to average.

                So the more "just the facts" either of the comments are, the
                lower the magnitude of alignment. Even if a comment contains
                strong opinions in a similar subject, if the opinions don't
                support each other or oppose each other, then the alignment is
                zero or near zero. If the comments agree, then the alignment is
                positive. The more the comments' opinions disagree, the more
                negative the alignment.

                Respond with the number only -- no labels, words, explanations.
                Nothing. Just the decimal number.
            ''',
        },
        {
            'role': 'user',
            'content': '' + comment_context1,
        },
        {
            'role': 'user',
            'content': '' + comment_context2,
        },
    ]

async def generate_deepseek_score(prompt: list[dict]):
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try to connect to Ollama server
            client = ollama.AsyncClient()
            
            # First, check if the server is responsive
            try:
                # Test connection by listing models
                models = await client.list()
                
                # Extract model names from the response (handle both dict and object formats)
                available_models = []
                for model in models['models']:
                    if hasattr(model, 'model'):
                        available_models.append(model.model)
                    elif isinstance(model, dict) and 'model' in model:
                        available_models.append(model['model'])
                    elif hasattr(model, 'name'):
                        available_models.append(model.name)
                    elif isinstance(model, dict) and 'name' in model:
                        available_models.append(model['name'])
                    else:
                        available_models.append(str(model))
                
                # Check for preferred models in order of preference
                preferred_models = ['qwen2.5:7b', 'qwen2.5', 'mistral:7b', 'llama3.1:8b', 'llama2:7b', 'deepseek-r1:32b']
                target_model = None
                
                for preferred in preferred_models:
                    if preferred in available_models or any(preferred in model for model in available_models):
                        target_model = preferred
                        print(f"Using preferred model: {target_model}")
                        break
                
                if target_model is None and available_models:
                    target_model = available_models[0]
                    print(f"Using first available model: {target_model}")
                elif target_model is None:
                    raise Exception("No models available on Ollama server")
                
            except Exception as e:
                raise Exception(f"Ollama server not accessible: {str(e)}")
            
            # Make the actual request
            response: dict | ollama.ChatResponse = await client.chat(
                model=target_model,
                messages=prompt,
                stream=False,
                options=ollama.Options(
                    top_p=0,
                    seed=54321,
                ),
            )
            return response['message']['content']
            
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt == max_retries - 1:
                # Last attempt failed, provide detailed error info
                error_msg = f"All connection attempts failed after {max_retries} retries. "
                error_msg += f"Error: {str(e)}. "
                error_msg += "Please ensure: 1) Ollama server is running ('ollama serve'), "
                error_msg += "2) The model 'qwen2.5:7b' is installed ('ollama pull qwen2.5:7b'), "
                error_msg += "3) Network connectivity is available from compute nodes."
                raise Exception(error_msg)
            
            # Wait before retrying with exponential backoff
            delay = base_delay * (2 ** attempt)
            print(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

async def check_ollama_connection():
    """Check if Ollama server is accessible and what models are available"""
    try:
        client = ollama.AsyncClient()
        models = await client.list()
        
        # Debug: print the structure to understand the response
        print(f"Debug: models response: {models}")
        
        # Extract model names from the response
        if 'models' in models:
            model_list = models['models']
            if model_list and len(model_list) > 0:
                # The models are objects with a 'model' attribute containing the name
                available_models = []
                for model in model_list:
                    if hasattr(model, 'model'):
                        available_models.append(model.model)
                    elif isinstance(model, dict) and 'model' in model:
                        available_models.append(model['model'])
                    elif hasattr(model, 'name'):
                        available_models.append(model.name)
                    elif isinstance(model, dict) and 'name' in model:
                        available_models.append(model['name'])
                    else:
                        available_models.append(str(model))
            else:
                available_models = []
        else:
            available_models = []
            
        return True, available_models
    except Exception as e:
        return False, str(e)

def evaluate_prompt(prompt):
    print('EVALUATING EVALUATING')
    return asyncio.run(generate_deepseek_score(prompt))


def uncluster(x: float | np.ndarray, epsilon: float = 1e-10):
    """
    # This function can be derived by plotting a histogram of the linear ranges'
    # product, noticing it looks like 1-lnx (the constant and log base don't
    # matter due to x and y stretches) and then integrating 1-lnx to get x*(2-lnx)
    # Due to shrinking the function horizontally and vertically by a factor of e
    # to fit the domain/range [-1, 1], the exact function is x*(1-ln|x|)
    """
    is_scalar = np.isscalar(x)
    if is_scalar:
        x = np.array(x)
    if np.any(np.abs(x) > 1):
        raise ValueError("All input values must be in the range [-1, 1].")
    safe_x = np.clip(np.abs(x), epsilon, None)
    result = np.where(x == 0, 0, x * (1 - np.log(safe_x)))
    return float(result) if is_scalar else result


def trim_potential_think(text: str) -> str:
    pattern = re.compile(r"([-+]?[0-9]*\.?[0-9]+)$", re.DOTALL)
    match_ = pattern.search(text)
    return match_.group(1).strip() if match_ else ''

def support(comment_context: str, *, get_tokens: bool = False):
    if get_tokens:
        print('get_tokesn support')
    print('sup')
    return float(trim_potential_think(evaluate_prompt(generate_support_prompt(comment_context))))

def similarity(a_context: str, b_context: str, *, get_tokens: bool = False) -> float:
    if get_tokens:
        print('get_tokesn similarity')
    print('sim')
    return float(trim_potential_think(evaluate_prompt(generate_similarity_prompt(
        a_context, b_context,
    ))))

def alignment(a_context: str, b_context: str, *, get_tokens: bool = False) -> float:
    if get_tokens:
        print('get_tokesn alignment')
    print('align')
    return float(trim_potential_think(evaluate_prompt(generate_alignment_prompt(
        a_context, b_context,
    ))))

@dataclass
class Redditor:
    id: str
    name: str
    created_utc: int
    comments: list['Comment']
    comments_with_parents: list['Comment']
    
    def __repr__(self):
        return f'<Redditor {self.name}>'
    
    def bias(self, *, get_tokens: bool = False):
        '''
        Generates bias coefficient in [0.5, 2.0]
        Very costly: O(number of user's comments^2 * avg_thread_length)
        Cached.
        '''

        pairs = list(combinations(self.comments_with_parents, 2))
        pairs_og_length = len(pairs)
        max_length = 100  # Reduced from 5000 to 100 for faster processing
        # JJ: Note that we cap at 100 pairs. This significantly reduces processing time
        # while still providing a good sample for bias calculation.
        if pairs_og_length > max_length:
            pairs = random.Random(42).choices(pairs, k=max_length)
        
        if get_tokens:
            tokens = 0
            for old_com, new_com in pairs:
                tokens += old_com.parent.alignment(new_com.parent, get_tokens=True)
                tokens += old_com.support(get_tokens=True)
                tokens += new_com.support(get_tokens=True)
            return {'pairs': len(pairs), 'tokens': tokens}
        if not pairs: return 1.0
            
        res = 0.0
        for old_com, new_com in pairs:
            # No multiplication smoothing yet
            a = old_com.parent.alignment(new_com.parent)
            q1q2 = uncluster(old_com.support() * new_com.support())
            res += uncluster(q1q2 * a)
        res /= len(pairs)
        return res*0.75+1.25


@dataclass
class Comment:
    id: str
    text: str
    score: int
    author: Redditor
    created_utc: int
    title: Optional[str]  # <- Note: reply comments don't have titles. posts aren't used yet?
    parent: Optional['Comment']
    ancestors: Optional[list['Comment']] = field(default_factory=list, repr=False)

    def support(self, *, get_tokens: bool = False) -> float:
        '''
        Generates support coefficient [-1.0, 1.0]
        Cached
        '''
        assert self.parent is not None
        if get_tokens:
            return support(json.dumps(self.to_context()), get_tokens=get_tokens)
        return cached_api_call("support", json.dumps(self.to_context()))
        
    def similarity(a: 'Comment', b: 'Comment', *, get_tokens: bool = False) -> float:
        '''
        Make sure to compare *parent* similarity
        Generates similarity coefficient [1, 10]
        Cached
        '''
        if get_tokens:
            return similarity(json.dumps(a.to_context()), json.dumps(b.to_context()), get_tokens=get_tokens)
        return cached_api_call("similarity", json.dumps(a.to_context()), json.dumps(b.to_context()))
    
    def alignment(a: 'Comment', b: 'Comment', *, get_tokens: bool = False) -> float:
        '''
        Make sure to compare *parent* alignment
        Generates alignment coefficient [-1.0, 1.0]
        Cached
        '''
        if get_tokens:
            return alignment(json.dumps(a.to_context()), json.dumps(b.to_context()), get_tokens=get_tokens)
        return cached_api_call("alignment", json.dumps(a.to_context()), json.dumps(b.to_context()))
    
    def __matmul__(a: 'Comment', b: 'Comment') -> float:
        return a.alignment(b)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id
    
    def to_dict(self):
        # timestamp, author, text and no recursive ancestors and no original comment
        return {'timestamp': self.created_utc, 'author_id': self.author.id} | (
            {'post_title': self.title, 'post_text': self.text}
            if self.title else {'comment_text': self.text}
        )
    
    def to_context(self):
        if self.parent is None:
            return {
                'id': self.id,
                'ancestors': [],
                'parent': {},
                'message': self.to_dict(),
            }
            
        return {
            'id': self.id,
            'ancestors': [
                ancestor.to_dict() for ancestor in self.parent.ancestors
            ],
            'parent': self.parent.to_dict(),
            'message': self.to_dict(),
        }
    
    def get_thread_json(self):
        return self.ancestors

@dataclass
class Subreddit:
    id: str
    name: str
    comments: list[Comment]
    members: dict[str, Redditor]
    comments_with_parents: list[Comment]

    def add_comment(self, comment: Comment):
        self.members.setdefault(comment.author.id, comment.author)
        self.comments.append(comment)
        if comment.parent is not None:
            self.comments_with_parents.append(comment)

def read_subreddits(subreddit_names: list[str]) -> list[Subreddit]:
    comment_datas = []
    
    # Debug: Print current working directory
    print(f"Debug: Current working directory: {Path.cwd()}")
    
    # Try different possible data directory paths (prefer filtered data)
    possible_data_dirs = [
        Path("data") / "filtered_analyzable",  # If running from reddit_2024/
        Path("../data") / "filtered_analyzable",  # If running from hpc/
        Path("../../data") / "filtered_analyzable",  # If running from hpc/cache/
        Path("data") / "with_likes",  # Fallback to original data
        Path("../data") / "with_likes",  # Fallback to original data
        Path("../../data") / "with_likes",  # Fallback to original data
    ]
    
    data_dir = None
    for dir_path in possible_data_dirs:
        print(f"Debug: Checking if {dir_path} exists: {dir_path.exists()}")
        if dir_path.exists():
            data_dir = dir_path
            print(f"Found data directory: {data_dir}")
            break
    
    if data_dir is None:
        print(f"Error: Could not find data directory. Tried: {possible_data_dirs}")
        return []
    
    for subreddit_name in subreddit_names:
        # Create the proper filename with extension based on subreddit name
        # Try filtered version first, then fallback to original
        base_filename = f"{subreddit_name}_2019-01_agree"
        if subreddit_name == "globeskepticism":
            base_filename = f"{subreddit_name}_2020-06_agree"
        
        # Try filtered filename first
        filtered_filename = f"{base_filename}_filtered.json"
        subreddit_filepath = data_dir / filtered_filename
        
        # If filtered file doesn't exist, try original filename
        if not subreddit_filepath.exists():
            original_filename = f"{base_filename}.json"
            subreddit_filepath = data_dir / original_filename
        
        try:
            with open(subreddit_filepath, 'r') as f:
                comment_datas += json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find data file for subreddit '{subreddit_name}' at {subreddit_filepath}")
            continue
    
    # Sort by creation time
    comment_datas.sort(key=lambda data: data['created_utc'])

    redditors = {}
    for comment_data in comment_datas:
        if comment_data['author_fullname'] in redditors: continue
        redditor = Redditor(
            id=comment_data['author_fullname'],
            name=comment_data['author'],
            created_utc=comment_data['author_created_utc'],
            comments=[],
            comments_with_parents=[],
        )
        redditors[redditor.id] = redditor
    
    comments = {}
    subreddits = {}
    for comment_data in comment_datas:
        comment = Comment(
            id=comment_data['id'],
            text=comment_data['text'],
            score=comment_data['score'],
            author=redditors[comment_data['author_fullname']],
            created_utc=comment_data['created_utc'],
            title=comment_data.get('title', None),
            parent=None,
            ancestors=None,
        )
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
    print(f"Debug: Found subreddits in data: {list(subreddits.keys())}")
    print(f"Debug: Looking for subreddits: {subreddit_names}")
    
    # Check which subreddits are missing
    missing_subreddits = [name for name in subreddit_names if name not in subreddits]
    if missing_subreddits:
        print(f"Debug: Missing subreddits: {missing_subreddits}")
    
    return [subreddits[name] for name in subreddit_names]
        
ALL_SUBREDDITS=["SandersForPresident","flatearth","trump","science","cars","Republican","SocialDemocracy","Freethought","travel","math","NeutralPolitics","PoliticalDiscussion","democrats","hiking","republicans","mlb","progressive","AmericanPolitics"]


def create_bias_files(subreddit_subset):
    global numerator, denominator
    """
    For each subreddit in subreddit_subset, create a file with bias values for all redditors
    in that subreddit, using the same format as bias_values_progressive.json
    
    Args:
        subreddit_subset: List of Subreddit objects to process
    """
    # Path to the bias values directory using relative path
    # Navigate up one directory from current script location, then to data/bias
    current_dir = Path(__file__).parent
    bias_dir = (current_dir.parent / "data" / "bias")
    
    # Ensure bias directory exists
    bias_dir.mkdir(parents=True, exist_ok=True)
    
    # Load progress tracking
    progress_file = bias_dir / "progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {"completed_subreddits": [], "completed_users": {}}
    
    # Process each subreddit
    for subreddit_idx, subreddit in enumerate(subreddit_subset, 1):
        # Skip if already completed
        if subreddit.name in progress["completed_subreddits"]:
            print(f"Skipping already completed subreddit: {subreddit.name}")
            continue
            
        print(f"Processing subreddit: {subreddit.name}")
        
        # List to store bias data for this subreddit
        bias_data = []
        
        # Get all redditors for this subreddit and take a random sample of 50
        all_redditors = list(subreddit.members.values())
        sample_size = min(25, len(all_redditors))
        redditors = random.sample(all_redditors, sample_size)
        
        print(f"Sampling {sample_size} redditors from {len(all_redditors)} total redditors")
        
        # Process each redditor
        total_redditors = len(redditors)
        for i, redditor in enumerate(redditors):
            # Skip if already completed
            if subreddit.name in progress["completed_users"] and redditor.name in progress["completed_users"][subreddit.name]:
                print(f"Skipping already completed user: {redditor.name}")
                continue
                
            # Show progress with subreddit and user numbers
            print(f"Subreddit {subreddit_idx}/{len(subreddit_subset)}: User {i+1}/{total_redditors} ({redditor.name})...")
                
            try:
                # Calculate bias value using the bias method
                # bias_value = redditor.bias()
                bias_value = redditor.bias()
                denominator += 1
                bias_call_data = redditor.bias(get_tokens=True)
                numerator += bias_call_data['tokens']
                print(f'{i+1}')
                
                # Store in the same format as bias_values_progressive.json
                bias_entry = {
                    "author": redditor.name,
                    "bias_seeking": bias_value,
                    "tokens": bias_call_data['tokens'],
                    "pairs": bias_call_data['pairs'],
                }
                
                bias_data.append(bias_entry)
                
                # Mark user as completed
                if subreddit.name not in progress["completed_users"]:
                    progress["completed_users"][subreddit.name] = []
                progress["completed_users"][subreddit.name].append(redditor.name)
                
                # Save progress after each user
                with open(progress_file, 'w') as f:
                    json.dump(progress, f, indent=4)
                    
            except Exception as e:
                print(f"Error processing redditor {redditor.name}: {str(e)}")
        
        # Create output filename with subreddit name
        output_file = bias_dir / f"bias_values_{subreddit.name.lower()}.json"
        
        # Write the bias data to JSON file
        with open(output_file, 'w') as f:
            json.dump(bias_data, f, indent=4)
        
        # Mark subreddit as completed
        progress["completed_subreddits"].append(subreddit.name)
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        
        print(f"Created bias file for {subreddit.name} at {output_file}")
        print(f"Processed {len(bias_data)} redditors successfully")

def parse_arguments():
    """Parse command line arguments for subreddit selection"""
    parser = argparse.ArgumentParser(description='Create confirmation bias files for subreddits')
    
    parser.add_argument('--subreddits', nargs='+', default=[],
                        help='List of subreddit names to process. If not provided, processes all subreddits.')
    
    parser.add_argument('--random', type=int, default=0,
                        help='Number of random subreddits to process. Ignored if --subreddits is provided.')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Load existing cache
    load_cache()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(12345)

    numerator = 0
    denominator = 0
    
    # Determine which subreddits to process
    if args.subreddits:
        print(f'Using user-specified subreddits: {args.subreddits}')
        # Use user-specified subreddits
        selected_subreddits = [s for s in args.subreddits if s in ALL_SUBREDDITS]
        if len(selected_subreddits) != len(args.subreddits):
            missing = set(args.subreddits) - set(ALL_SUBREDDITS)
            print(f"Warning: Some subreddits not found: {', '.join(missing)}")
            print(f"Available subreddits: {', '.join(ALL_SUBREDDITS)}")
    elif args.random > 0:
        # Choose random subreddits
        n_random = min(args.random, len(ALL_SUBREDDITS))
        selected_subreddits = random.sample(ALL_SUBREDDITS, n_random)
        print(f"Randomly selected {n_random} subreddits: {', '.join(selected_subreddits)}")
    else:
        # Use all subreddits
        selected_subreddits = ALL_SUBREDDITS
    
    # Read the selected subreddits
    print(f"Reading data for {len(selected_subreddits)} subreddits...")
    subreddit_objects = read_subreddits(selected_subreddits)

    # Create bias files
    create_bias_files(subreddit_objects)

    # Save final cache
    save_cache()

    print(f'Average token count per redditor: {numerator / denominator}')