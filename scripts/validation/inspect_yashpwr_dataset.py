"""
Download and inspect yashpwr/resume-ner-training-data dataset.
Check data quality, format, and entity distribution before using.
"""

import json
from collections import Counter
from typing import Dict, List
import sys


def download_dataset():
    """Download the dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("📥 Downloading yashpwr/resume-ner-training-data...")
        dataset = load_dataset("yashpwr/resume-ner-training-data")
        print("✅ Dataset downloaded!")
        return dataset
    except ImportError:
        print("❌ Error: datasets library not installed")
        print("   Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None


def inspect_dataset(dataset):
    """Inspect dataset structure and quality."""
    print("\n" + "=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    
    # Get splits
    splits = list(dataset.keys())
    print(f"\n📊 Available splits: {splits}")
    
    # Inspect each split
    for split in splits:
        print(f"\n{'=' * 70}")
        print(f"SPLIT: {split.upper()}")
        print(f"{'=' * 70}")
        
        split_data = dataset[split]
        print(f"Number of examples: {len(split_data)}")
        
        # Show first example
        if len(split_data) > 0:
            print(f"\n📝 Sample Example:")
            print("-" * 70)
            example = split_data[0]
            
            # Show all keys
            print(f"Keys: {list(example.keys())}")
            
            # Show structure
            for key, value in example.items():
                if isinstance(value, str):
                    if len(value) > 200:
                        print(f"\n{key}: {value[:200]}... (truncated, total length: {len(value)})")
                    else:
                        print(f"\n{key}: {value}")
                elif isinstance(value, list):
                    print(f"\n{key}: {type(value)} with {len(value)} items")
                    if len(value) > 0:
                        print(f"  First item: {value[0]}")
                        if len(value) > 1:
                            print(f"  Second item: {value[1]}")
                else:
                    print(f"\n{key}: {type(value)} = {value}")
        
        # Analyze entity distribution
        print(f"\n📊 Entity Distribution Analysis:")
        print("-" * 70)
        
        entity_counter = Counter()
        total_entities = 0
        total_tokens = 0
        sample_entities = []
        
        for i, example in enumerate(split_data):
            # Check different possible formats
            if 'ner_tags' in example:
                tags = example['ner_tags']
                tokens = example.get('tokens', [])
                total_tokens += len(tokens)
                
                for tag in tags:
                    if isinstance(tag, (int, str)):
                        entity_counter[tag] += 1
                        total_entities += 1
                    elif isinstance(tag, list):
                        # BIO format
                        for t in tag:
                            entity_counter[t] += 1
                            total_entities += 1
                
                # Collect sample entities
                if i < 3 and 'tokens' in example:
                    sample_entities.append({
                        'tokens': example['tokens'][:20],
                        'tags': example['ner_tags'][:20] if len(example['ner_tags']) >= 20 else example['ner_tags']
                    })
            
            elif 'labels' in example:
                labels = example['labels']
                tokens = example.get('tokens', [])
                total_tokens += len(tokens)
                
                for label in labels:
                    entity_counter[label] += 1
                    total_entities += 1
                
                if i < 3 and 'tokens' in example:
                    sample_entities.append({
                        'tokens': example['tokens'][:20],
                        'labels': example['labels'][:20] if len(example['labels']) >= 20 else example['labels']
                    })
        
        print(f"Total tokens: {total_entities:,}")
        print(f"Unique entity types: {len(entity_counter)}")
        print(f"\nTop 20 Entity Types:")
        for entity, count in entity_counter.most_common(20):
            pct = (count / total_entities * 100) if total_entities > 0 else 0
            print(f"  {str(entity):30s} {count:8,} ({pct:5.2f}%)")
        
        # Show sample token-tag pairs
        if sample_entities:
            print(f"\n📋 Sample Token-Tag Pairs:")
            print("-" * 70)
            for idx, sample in enumerate(sample_entities[:2]):
                print(f"\nExample {idx + 1}:")
                tokens = sample.get('tokens', [])
                tags = sample.get('tags', sample.get('labels', []))
                
                # Show first 15 pairs
                for i in range(min(15, len(tokens), len(tags))):
                    token = tokens[i] if i < len(tokens) else "N/A"
                    tag = tags[i] if i < len(tags) else "N/A"
                    print(f"  {token:20s} -> {tag}")


def validate_format(dataset):
    """Validate if dataset format matches our training code requirements."""
    print(f"\n{'=' * 70}")
    print("FORMAT VALIDATION")
    print(f"{'=' * 70}")
    
    required_format = {
        'text': 'string',
        'annotations': 'list of [start, end, label]'
    }
    
    print(f"\nOur training code expects:")
    print(f"  - 'text': string")
    print(f"  - 'annotations': list of [start, end, label] tuples")
    
    # Check first example
    for split in dataset.keys():
        example = dataset[split][0]
        print(f"\n{split.upper()} split format:")
        print(f"  Keys: {list(example.keys())}")
        
        # Check if we need conversion
        needs_conversion = False
        
        if 'tokens' in example and 'ner_tags' in example:
            print(f"  ⚠️  Format: tokens + ner_tags (needs conversion)")
            needs_conversion = True
        elif 'text' in example and 'annotations' in example:
            print(f"  ✅ Format: text + annotations (matches our format!)")
        else:
            print(f"  ❓ Unknown format, needs inspection")
            needs_conversion = True
        
        return needs_conversion


def convert_to_our_format(dataset, output_file: str):
    """Convert dataset to our training format."""
    print(f"\n{'=' * 70}")
    print("CONVERTING TO OUR FORMAT")
    print(f"{'=' * 70}")
    
    converted_data = []
    
    for split in dataset.keys():
        print(f"\nProcessing {split} split...")
        
        for example in dataset[split]:
            # Handle different formats
            if 'tokens' in example and 'ner_tags' in example:
                # Convert from token-level to character-level
                tokens = example['tokens']
                tags = example['ner_tags']
                
                # Reconstruct text
                text = ' '.join(tokens)
                
                # Convert tags to annotations
                annotations = []
                char_pos = 0
                
                for i, (token, tag) in enumerate(zip(tokens, tags)):
                    # Skip O tags
                    if tag == 'O' or tag == 0:
                        char_pos += len(token) + 1  # +1 for space
                        continue
                    
                    # Handle BIO format
                    if isinstance(tag, str) and tag.startswith('B-'):
                        label = tag[2:]  # Remove 'B-' prefix
                        start = char_pos
                        end = char_pos + len(token)
                        annotations.append([start, end, label])
                    elif isinstance(tag, str) and tag.startswith('I-'):
                        # Continue previous entity
                        if annotations:
                            # Extend last annotation
                            annotations[-1][1] = char_pos + len(token)
                    elif tag != 'O' and tag != 0:
                        # Direct label
                        label = str(tag)
                        start = char_pos
                        end = char_pos + len(token)
                        annotations.append([start, end, label])
                    
                    char_pos += len(token) + 1
                
                if annotations:
                    converted_data.append({
                        'text': text,
                        'annotations': annotations
                    })
            
            elif 'text' in example and 'annotations' in example:
                # Already in our format
                converted_data.append(example)
    
    print(f"\n✅ Converted {len(converted_data)} examples")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Conversion complete!")
    return len(converted_data)


def main():
    print("=" * 70)
    print("YASHPWR RESUME NER DATASET INSPECTOR")
    print("=" * 70)
    
    # Download
    dataset = download_dataset()
    if dataset is None:
        sys.exit(1)
    
    # Inspect
    inspect_dataset(dataset)
    
    # Validate format
    needs_conversion = validate_format(dataset)
    
    # Ask user if they want to convert
    if needs_conversion:
        print(f"\n{'=' * 70}")
        print("CONVERSION OPTION")
        print(f"{'=' * 70}")
        print("\nDataset needs conversion to match our training format.")
        print("Would you like to convert it now? (This will create a new file)")
        
        # For script, we'll just show what would happen
        print("\nTo convert, run:")
        print("  python scripts/validation/inspect_yashpwr_dataset.py --convert output.json")
    
    print(f"\n{'=' * 70}")
    print("INSPECTION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect yashpwr resume NER dataset')
    parser.add_argument('--convert', type=str, help='Convert and save to this file')
    
    args = parser.parse_args()
    
    dataset = download_dataset()
    if dataset:
        inspect_dataset(dataset)
        needs_conversion = validate_format(dataset)
        
        if args.convert:
            convert_to_our_format(dataset, args.convert)
    else:
        sys.exit(1)
