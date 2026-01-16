"""
Aggressive dataset cleaning - removes low-quality annotations.
Filters out:
- Very short annotations (< 3 chars)
- Common English words
- Email parts labeled as SKILL
- Single characters
- Over-annotated common words
"""

import json
import re
from typing import List, Tuple, Dict


# Common English words that shouldn't be labeled as SKILL
COMMON_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'make', 'made', 'get', 'got',
    'go', 'went', 'come', 'came', 'see', 'saw', 'know', 'knew', 'think',
    'thought', 'take', 'took', 'give', 'gave', 'find', 'found', 'work',
    'worked', 'use', 'used', 'try', 'tried', 'need', 'needed', 'want',
    'wanted', 'like', 'liked', 'look', 'looked', 'call', 'called', 'ask',
    'asked', 'tell', 'told', 'help', 'helped', 'show', 'showed', 'move',
    'moved', 'live', 'lived', 'believe', 'believe', 'bring', 'brought',
    'happen', 'happened', 'write', 'wrote', 'sit', 'sat', 'stand', 'stood',
    'lose', 'lost', 'pay', 'paid', 'meet', 'met', 'include', 'included',
    'continue', 'continued', 'set', 'put', 'end', 'turn', 'turned', 'start',
    'started', 'run', 'ran', 'keep', 'kept', 'let', 'allow', 'allowed',
    'provide', 'provided', 'add', 'added', 'change', 'changed', 'play',
    'played', 'spend', 'spent', 'open', 'opened', 'walk', 'walked', 'win',
    'won', 'offer', 'offered', 'remember', 'remembered', 'love', 'loved',
    'consider', 'considered', 'appear', 'appeared', 'buy', 'bought', 'wait',
    'waited', 'serve', 'served', 'die', 'died', 'send', 'sent', 'build',
    'built', 'stay', 'stayed', 'fall', 'fell', 'cut', 'cut', 'reach',
    'reached', 'kill', 'killed', 'raise', 'raised', 'pass', 'passed',
    'sell', 'sold', 'decide', 'decided', 'return', 'returned', 'explain',
    'explained', 'learn', 'learned', 'increase', 'increased', 'cover',
    'covered', 'grow', 'grew', 'thank', 'thanked', 'email', 'emails',
    'gmail', 'com', 'org', 'net', 'edu', 'www', 'http', 'https'
}

# Email-related words that shouldn't be SKILL
EMAIL_WORDS = {'email', 'emails', 'gmail', 'yahoo', 'hotmail', 'outlook',
               'com', 'org', 'net', 'edu', 'co', 'uk', 'in', 'www', 'http',
               'https', 'mail', 'contact', 'phone', 'mobile', 'tel'}

# Very common resume words that are over-annotated
RESUME_NOISE = {'professional', 'experience', 'years', 'year', 'month',
                'months', 'skills', 'skill', 'education', 'work', 'worked',
                'working', 'company', 'companies', 'project', 'projects',
                'team', 'teams', 'role', 'roles', 'responsibility',
                'responsibilities', 'developed', 'develop', 'development',
                'management', 'manager', 'managed', 'lead', 'led', 'leading'}


def is_valid_skill(text: str, start: int, end: int, full_text: str) -> bool:
    """Check if a SKILL annotation is valid."""
    skill_text = text[start:end].strip().lower()
    
    # Too short
    if len(skill_text) < 3:
        return False
    
    # Common word
    if skill_text in COMMON_WORDS:
        return False
    
    # Email-related (should be EMAIL label, not SKILL)
    if skill_text in EMAIL_WORDS:
        return False
    
    # Resume noise words
    if skill_text in RESUME_NOISE:
        return False
    
    # Check if it's part of an email address
    # Look at surrounding context
    context_start = max(0, start - 10)
    context_end = min(len(full_text), end + 10)
    context = full_text[context_start:context_end].lower()
    
    # If it looks like part of an email, reject
    if '@' in context or 'email' in context or 'mail' in context:
        if skill_text in ['gmail', 'yahoo', 'hotmail', 'outlook', 'com', 'org', 'net']:
            return False
    
    # Check if it's just punctuation or numbers
    if re.match(r'^[^a-zA-Z]+$', skill_text):
        return False
    
    # Check if it's a single letter (like 'c' or 'C')
    if len(skill_text.strip()) == 1 and skill_text.isalpha():
        return False
    
    return True


def clean_annotations(annotations: List, text: str) -> List:
    """Clean annotations by removing invalid ones."""
    cleaned = []
    
    for ann in annotations:
        if not isinstance(ann, list) or len(ann) != 3:
            continue
        
        start, end, label = ann
        
        # Validate bounds
        if start < 0 or end > len(text) or start >= end:
            continue
        
        # Get annotation text
        ann_text = text[start:end].strip()
        if not ann_text:
            continue
        
        # Special handling for SKILL labels - be more strict
        if label == 'SKILL':
            if not is_valid_skill(text, start, end, text):
                continue
        
        # Remove very short annotations for all labels
        if len(ann_text) < 3:
            continue
        
        # Remove single character annotations
        if len(ann_text.strip()) == 1:
            continue
        
        cleaned.append(ann)
    
    return cleaned


def clean_entry(entry: Dict) -> Tuple[Dict, bool]:
    """Clean a single entry aggressively."""
    if 'text' not in entry or 'annotations' not in entry:
        return None, False
    
    text = entry['text']
    annotations = entry['annotations']
    
    if not isinstance(annotations, list):
        return None, False
    
    # Clean annotations
    cleaned_annotations = clean_annotations(annotations, text)
    
    # Require at least 3 annotations to keep the entry
    if len(cleaned_annotations) < 3:
        return None, False
    
    return {
        'text': text,
        'annotations': cleaned_annotations
    }, True


def clean_dataset_aggressive(input_file: str, output_file: str):
    """Aggressively clean entire dataset."""
    print("=" * 70)
    print("AGGRESSIVE DATASET CLEANING")
    print("=" * 70)
    print("\nLoading dataset...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    cleaned_data = []
    removed_entries = 0
    total_annotations_before = 0
    total_annotations_after = 0
    
    print("\nCleaning dataset (this may take a while)...")
    for idx, entry in enumerate(data):
        if 'annotations' in entry:
            total_annotations_before += len(entry['annotations'])
        
        cleaned_entry, is_valid = clean_entry(entry)
        
        if is_valid:
            cleaned_data.append(cleaned_entry)
            total_annotations_after += len(cleaned_entry['annotations'])
        else:
            removed_entries += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} entries... "
                  f"({len(cleaned_data)} kept, {removed_entries} removed)")
    
    print(f"\n✅ Cleaning complete!")
    print("-" * 70)
    print(f"Original entries:     {len(data)}")
    print(f"Cleaned entries:      {len(cleaned_data)}")
    print(f"Removed entries:      {removed_entries} ({removed_entries/len(data)*100:.1f}%)")
    print(f"\nOriginal annotations: {total_annotations_before:,}")
    print(f"Cleaned annotations:  {total_annotations_after:,}")
    print(f"Removed annotations:  {total_annotations_before - total_annotations_after:,} "
          f"({(total_annotations_before - total_annotations_after)/total_annotations_before*100:.1f}%)")
    print("-" * 70)
    
    if len(cleaned_data) == 0:
        print("❌ ERROR: All entries were removed! Dataset is too noisy.")
        return 0, removed_entries
    
    # Save cleaned dataset
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    print(f"\n📊 Final dataset: {len(cleaned_data)} entries, "
          f"{total_annotations_after:,} annotations")
    
    return len(cleaned_data), removed_entries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggressively clean NER dataset')
    parser.add_argument('input', type=str, help='Input dataset JSON file')
    parser.add_argument('output', type=str, help='Output cleaned dataset JSON file')
    
    args = parser.parse_args()
    
    clean_dataset_aggressive(args.input, args.output)
