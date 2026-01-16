"""
Convert yashpwr/resume-ner-training-data to our training format.
Run this AFTER inspecting the dataset to ensure it's good quality.
"""

from datasets import load_dataset
import json
from tqdm import tqdm

def convert_bio_to_annotations(tokens, tags, text):
    """Convert BIO tags to character-level annotations."""
    annotations = []
    
    # Build character position mapping
    char_pos = 0
    token_to_char = {}
    
    for i, token in enumerate(tokens):
        token_to_char[i] = char_pos
        char_pos += len(token)
        if i < len(tokens) - 1:  # Add space except for last token
            char_pos += 1
    
    # Convert tags to annotations
    i = 0
    while i < len(tags):
        tag = tags[i]
        
        if tag == 'O' or tag == 0:
            i += 1
            continue
        
        # Handle BIO format
        if isinstance(tag, str):
            if tag.startswith('B-'):
                label = tag[2:]  # Remove 'B-' prefix
                start = token_to_char[i]
                
                # Find end of entity (until next B- or O)
                j = i + 1
                while j < len(tags):
                    next_tag = tags[j]
                    if isinstance(next_tag, str):
                        if next_tag.startswith('B-') or next_tag == 'O':
                            break
                        elif next_tag.startswith('I-'):
                            j += 1
                        else:
                            break
                    else:
                        break
                
                # Calculate end position
                if j < len(tokens):
                    end = token_to_char[j]
                else:
                    end = token_to_char[i] + len(tokens[i])
                
                annotations.append([start, end, label])
                i = j
            elif tag.startswith('I-'):
                # Should have been handled by B- case
                i += 1
            else:
                # Direct label (not BIO)
                label = str(tag)
                start = token_to_char[i]
                end = start + len(tokens[i])
                annotations.append([start, end, label])
                i += 1
        else:
            # Numeric tag
            if tag != 0:
                label = str(tag)
                start = token_to_char[i]
                end = start + len(tokens[i])
                annotations.append([start, end, label])
            i += 1
    
    return annotations


def convert_dataset(output_file: str):
    """Download and convert yashpwr dataset."""
    print("=" * 70)
    print("CONVERTING YASHPWR DATASET")
    print("=" * 70)
    
    # Download
    print("\n📥 Downloading dataset...")
    dataset = load_dataset("yashpwr/resume-ner-training-data")
    print("✅ Downloaded!")
    
    converted_data = []
    
    # Process each split
    for split_name in dataset.keys():
        print(f"\n📊 Processing {split_name} split ({len(dataset[split_name])} examples)...")
        
        split_data = dataset[split_name]
        
        for example in tqdm(split_data, desc=f"Converting {split_name}"):
            # Handle different formats
            if 'tokens' in example and 'ner_tags' in example:
                # Reconstruct text
                tokens = example['tokens']
                tags = example['ner_tags']
                text = ' '.join(tokens)
                
                # Convert to annotations
                annotations = convert_bio_to_annotations(tokens, tags, text)
                
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
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    print(f"\n📊 Final dataset: {len(converted_data)} examples")
    
    return len(converted_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert yashpwr dataset to our format')
    parser.add_argument('output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    convert_dataset(args.output)
