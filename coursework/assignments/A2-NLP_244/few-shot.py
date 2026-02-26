"""
few-shot.py - Few-shot prompting with Ollama for Homework 2 Part 1
Complete version with proper error handling and improvements
"""
import json
import subprocess
from tqdm import tqdm
import glob
import os
import time
from collections import Counter

# Configuration
MODEL_NAME = "llama3.2:1b"  # Using the small 1B model for speed (change to "llama3.1:8b" for larger model)
UNANNOTATED_DIR = "./unannotated_mmds"
OUTPUT_FILE = "part1_predictions.json"
CHUNK_SIZE = 512  # Characters per chunk
TEST_MODE = True  # Set to False to process ALL chunks (True = only 5 chunks per file)

# Few-shot examples from the assignment
FEW_SHOT_EXAMPLES = """
Input: By S subset N having a least element, we mean that there exists an x in S...
Output: [O] [O] [O] [O] [O] [definition] [definition, name] [definition, name] [O] [O] [O] [O] [O]

Input: Corollary 7.2. Let E be a finitely generated module and E a submodule.
Output: [B-theorem, B-name] [I-theorem, I-name] [I-theorem, I-name] [O] [O] [O] [reference] [O] [reference] [O]

Input: Here is an example of a group: Z/2Z is cyclic.
Output: [O] [O] [O] [O] [example] [O] [O] [O] [O]
"""

def check_ollama():
    """Verify ollama is accessible and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(" Ollama is working!")
            print(f"   Available models:\n{result.stdout}")
            return True
        else:
            print(f" Ollama returned error: {result.stderr}")
            return False
    except FileNotFoundError:
        print(" Ollama command not found. Make sure it's installed and in PATH")
        print("   Install from: https://ollama.com/download")
        return False
    except Exception as e:
        print(f" Error checking ollama: {e}")
        return False

def ensure_model(model_name):
    """Make sure the model is pulled"""
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if model_name not in result.stdout:
            print(f" Model {model_name} not found. Pulling now (this may take a few minutes)...")
            pull_result = subprocess.run(
                ["ollama", "pull", model_name],
                capture_output=True,
                text=True
            )
            if pull_result.returncode == 0:
                print(f" Model {model_name} pulled successfully")
            else:
                print(f" Failed to pull model: {pull_result.stderr}")
                return False
        else:
            print(f" Model {model_name} already available")
        return True
    except Exception as e:
        print(f" Error ensuring model: {e}")
        return False

def query_ollama(prompt, model=MODEL_NAME):
    """Send a prompt to Ollama and get response"""
    try:
        # Create a system message to constrain the output
        full_prompt = f"""[INST] <<SYS>>
You are a BIO tagging system. You ONLY output tags in brackets. You never explain yourself.
<</SYS>>

{prompt} [/INST]"""
        
        result = subprocess.run(
            ["ollama", "run", model, full_prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=120,
            env=os.environ
        )
        
        if result.returncode != 0:
            print(f" Ollama error (code {result.returncode}): {result.stderr[:200]}")
            return ""
            
        return result.stdout.strip()
    except Exception as e:
        print(f" Error querying Ollama: {e}")
        return ""
            
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        print(" Ollama timeout after 120 seconds")
        return ""
    except Exception as e:
        print(f" Error querying Ollama: {e}")
        return ""

def create_prompt(chunk):
    """Create a few-shot prompt for a text chunk"""
    # Clean the chunk to avoid issues with special characters
    chunk = chunk.replace('\\', '\\\\').replace('"', '\\"')
    
    prompt = f"""Perform BIO tagging on the following snippet of text. Assign one or more labels for each token. You can choose from "definition", "theorem", "proof", "example", "name", or "reference".

Here are some examples:
{FEW_SHOT_EXAMPLES}

IMPORTANT INSTRUCTIONS:
- Output ONLY the tags, one per token, in order
- Do NOT include any explanations, notes, or additional text
- Do NOT label individual words in your response
- Your entire response should be just the tags in brackets

Now tag this text:
Input: {chunk}
Output:"""
    return prompt

def parse_tags_from_response(response, chunk):
    """Parse the model's response into tags with better extraction"""
    if not response:
        return []
    
    # Valid tags from the assignment
    VALID_TAGS = {'definition', 'theorem', 'proof', 'example', 'name', 'reference'}
    
    # Tag mapping for common variations
    TAG_MAPPING = {
        'exponent': 'name',
        'generator': 'name',
        'period': 'name',
        'cyclic': 'name',
        'group': 'name',
        'element': 'name',
        'integer': 'name',
        'theorem': 'theorem',
        'proof': 'proof',
        'definition': 'definition',
        'example': 'example',
        'name': 'name',
        'reference': 'reference',
        'lemma': 'theorem',
        'corollary': 'theorem',
        'proposition': 'theorem',
        'claim': 'proof',
        'show': 'proof',
        'prove': 'proof',
        'define': 'definition',
        'called': 'name',
        'denoted': 'name'
    }
    
    # First, try to find bracketed content
    import re
    bracket_matches = re.findall(r'\[(.*?)\]', response)
    
    tags = []
    
    # If we found bracketed content, parse that
    if bracket_matches:
        for match in bracket_matches:
            # Handle multiple tags in one bracket like "definition, name"
            if ',' in match:
                for t in match.split(','):
                    t = t.strip()
                    if t in TAG_MAPPING:
                        mapped = TAG_MAPPING[t]
                        if mapped not in tags:
                            tags.append(mapped)
                    elif t in VALID_TAGS and t not in tags:
                        tags.append(t)
            else:
                t = match.strip()
                if t in TAG_MAPPING:
                    mapped = TAG_MAPPING[t]
                    if mapped not in tags:
                        tags.append(mapped)
                elif t in VALID_TAGS and t not in tags:
                    tags.append(t)
    
    # If no bracketed content, scan the whole response for tag words
    if not tags:
        words = response.lower().split()
        for word in words:
            word = word.strip('.,!?;:()[]{}')
            if word in TAG_MAPPING:
                mapped = TAG_MAPPING[word]
                if mapped not in tags:
                    tags.append(mapped)
            elif word in VALID_TAGS and word not in tags:
                tags.append(word)
    
    return tags

def process_file(file_path, model=MODEL_NAME, chunk_size=CHUNK_SIZE):
    """Process a single file"""
    filename = os.path.basename(file_path)
    print(f"\nProcessing {filename}...")
    start_time = time.time()
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"   File size: {len(text):,} characters")
    except Exception as e:
        print(f"   Error reading file: {e}")
        return []
    
    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"   Split into {len(chunks)} chunks")
    
    # Determine how many chunks to process
    chunks_to_process = chunks[:5] if TEST_MODE else chunks
    if TEST_MODE:
        print(f"   TEST MODE: Processing first 5 chunks only")
    else:
        print(f"   Processing all {len(chunks)} chunks (this will take time)")
    
    all_predictions = []
    chunk_times = []
    
    # Process each chunk
    for i, chunk in enumerate(tqdm(chunks_to_process, desc="   Chunks")):
        chunk_start = time.time()
        
        # Create prompt
        prompt = create_prompt(chunk)
        
        # Get response from Ollama
        response = query_ollama(prompt, model)
        
        if not response:
            print(f"\n      No response for chunk {i+1}")
            continue
        
        # Parse tags
        tags = parse_tags_from_response(response, chunk)
        
        # Create predictions for each tag found
        for tag in tags:
            # Simple prediction - each tag gets its own annotation
            all_predictions.append({
                'fileid': filename,
                'start': i * chunk_size,
                'end': i * chunk_size + min(100, len(chunk)),
                'tag': tag,
                'text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                'chunk_index': i,
                'method': 'few-shot'
            })
        
        chunk_time = time.time() - chunk_start
        chunk_times.append(chunk_time)
    
    # Summary for this file
    elapsed = time.time() - start_time
    avg_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0
    
    # Count tags
    tag_counter = Counter([p['tag'] for p in all_predictions])
    
    print(f"\n   Completed in {elapsed:.1f} seconds")
    print(f"   Average: {avg_time:.1f} sec/chunk")
    print(f"   Predictions: {len(all_predictions)}")
    print(f"   Tags found: {dict(tag_counter)}")
    
    return all_predictions

def analyze_results(results):
    """Analyze and print summary statistics"""
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    total_predictions = 0
    all_tags = []
    file_stats = []
    
    for filename, preds in results.items():
        file_preds = len(preds)
        total_predictions += file_preds
        
        file_tags = [p['tag'] for p in preds]
        all_tags.extend(file_tags)
        
        tag_counts = Counter(file_tags)
        file_stats.append({
            'file': filename,
            'predictions': file_preds,
            'tags': dict(tag_counts)
        })
    
    # Overall statistics
    overall_tags = Counter(all_tags)
    
    print(f"\nFiles processed: {len(results)}")
    print(f"Total predictions: {total_predictions}")
    print("\nTag distribution:")
    for tag, count in overall_tags.most_common():
        percentage = (count / total_predictions) * 100
        print(f"   {tag}: {count} ({percentage:.1f}%)")
    
    print("\nPer-file breakdown:")
    for stat in file_stats:
        print(f"\n   {stat['file']}:")
        print(f"      Predictions: {stat['predictions']}")
        print(f"      Tags: {stat['tags']}")
    
    return overall_tags

def main():
    """Main function"""
    print("Starting few-shot prompting for Part 1...")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Test mode: {'ON (5 chunks per file)' if TEST_MODE else 'OFF (all chunks)'}")
    print("=" * 60)
    
    # First, check if ollama is working
    if not check_ollama():
        print("\nOllama is not accessible. Please check installation.")
        print("Try running 'ollama serve' in another terminal window.")
        return
    
    # Ensure model is available
    if not ensure_model(MODEL_NAME):
        print(f"\nCould not ensure model {MODEL_NAME} is available.")
        return
    
    # Get all unannotated files
    file_pattern = os.path.join(UNANNOTATED_DIR, "*.mmd.filtered")
    mmd_files = glob.glob(file_pattern)
    
    if not mmd_files:
        print(f"\nNo .mmd.filtered files found in {UNANNOTATED_DIR}")
        print(f"Current directory: {os.getcwd()}")
        if os.path.exists(UNANNOTATED_DIR):
            print(f"Files in {UNANNOTATED_DIR}: {os.listdir(UNANNOTATED_DIR)}")
        else:
            print(f"Directory {UNANNOTATED_DIR} not found!")
        return
    
    print(f"\nFound {len(mmd_files)} files to process:")
    for f in mmd_files:
        print(f"   - {os.path.basename(f)}")
    
    # Process each file
    all_results = {}
    total_start_time = time.time()
    
    for file_path in mmd_files:
        predictions = process_file(file_path)
        filename = os.path.basename(file_path)
        all_results[filename] = predictions
    
    total_time = time.time() - total_start_time
    
    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    
    # Analyze and display results
    tag_distribution = analyze_results(all_results)
    
    print(f"\nTotal processing time: {total_time/60:.1f} minutes")
    print("=" * 60)
    
    # Save a summary file
    summary = {
        'model': MODEL_NAME,
        'test_mode': TEST_MODE,
        'total_files': len(all_results),
        'total_predictions': sum(len(p) for p in all_results.values()),
        'tag_distribution': dict(tag_distribution),
        'processing_time_minutes': total_time/60
    }
    
    with open('part1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nSummary saved to part1_summary.json")
    print("\nPart 1 few-shot prompting complete!")

if __name__ == "__main__":
    main()