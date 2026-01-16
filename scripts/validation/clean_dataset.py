"""
Clean dataset by removing overlaps and fixing invalid entries.
"""

import json
import sys
from typing import List, Tuple, Dict


def load_dataset(json_file_path: str) -> List[Dict]:
    """Load dataset."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def remove_overlaps(annotations: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Remove overlapping annotations, keeping the longest span."""
    if not annotations:
        return []
    
    # Sort by start position, then by length (longest first)
    sorted_anns = sorted(annotations, key=lambda x: (x[0], -(x[1] - x[0])))
    
    non_overlapping = []
    for start, end, label in sorted_anns:
        # Check if this annotation overlaps with any already added
        overlaps = False
        for existing_start, existing_end, existing_label in non_overlapping:
            # Check for overlap (not just adjacent)
            if not (end <= existing_start or start >= existing_end):
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append((start, end, label))
    
    return non_overlapping


def clean_entry(entry: Dict) -> Tuple[Dict, bool]:
    """Clean a single entry."""
    if 'text' not in entry or 'annotations' not in entry:
        return None, False
    
    text = entry['text']
    annotations = entry['annotations']
    
    if not isinstance(annotations, list):
        return None, False
    
    # Collect valid annotations
    valid_annotations = []
    for ann in annotations:
        if not isinstance(ann, list) or len(ann) != 3:
            continue
        
        start, end, label = ann
        
        # Validate
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if not isinstance(label, str):
            continue
        if start < 0 or end > len(text) or start >= end:
            continue
        
        # Check if annotated text is not empty
        if not text[start:end].strip():
            continue
        
        valid_annotations.append((start, end, label))
    
    # Remove overlaps
    cleaned_annotations = remove_overlaps(valid_annotations)
    
    # Convert back to list format
    cleaned_anns_list = [[start, end, label] for start, end, label in cleaned_annotations]
    
    if not cleaned_anns_list:
        return None, False
    
    return {
        'text': text,
        'annotations': cleaned_anns_list
    }, True


def clean_dataset(input_file: str, output_file: str):
    """Clean entire dataset."""
    print("Loading dataset...")
    data = load_dataset(input_file)
    print(f"Loaded {len(data)} entries")
    
    cleaned_data = []
    removed = 0
    
    print("\nCleaning dataset...")
    for idx, entry in enumerate(data):
        cleaned_entry, is_valid = clean_entry(entry)
        
        if is_valid:
            cleaned_data.append(cleaned_entry)
        else:
            removed += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(data)} entries...")
    
    print(f"\n✅ Cleaned dataset:")
    print(f"   Original: {len(data)} entries")
    print(f"   Cleaned:  {len(cleaned_data)} entries")
    print(f"   Removed:  {removed} entries ({removed/len(data)*100:.1f}%)")
    
    # Save cleaned dataset
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    return len(cleaned_data), removed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean NER dataset')
    parser.add_argument('input', type=str, help='Input dataset JSON file')
    parser.add_argument('output', type=str, help='Output cleaned dataset JSON file')
    
    args = parser.parse_args()
    
    clean_dataset(args.input, args.output)
