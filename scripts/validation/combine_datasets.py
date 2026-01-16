"""
Combine multiple datasets for more training data.
Combines DataTurks (high quality) with your cleaned dataset (larger).
"""

import json
from typing import List, Dict


def load_dataset(file_path: str) -> List[Dict]:
    """Load a dataset JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def combine_datasets(dataset_files: List[str], output_file: str):
    """Combine multiple datasets into one."""
    print("=" * 70)
    print("COMBINING DATASETS")
    print("=" * 70)
    
    combined_data = []
    total_before = 0
    
    for file_path in dataset_files:
        print(f"\nLoading {file_path}...")
        data = load_dataset(file_path)
        print(f"  Loaded {len(data)} entries")
        combined_data.extend(data)
        total_before += len(data)
    
    print(f"\n✅ Combined datasets:")
    print(f"   Total entries: {len(combined_data)}")
    
    # Remove duplicates (same text)
    print(f"\nRemoving duplicates...")
    seen_texts = set()
    unique_data = []
    duplicates = 0
    
    for entry in combined_data:
        text = entry.get('text', '').strip()
        text_hash = hash(text[:100])  # Use first 100 chars as identifier
        
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_data.append(entry)
        else:
            duplicates += 1
    
    print(f"   Unique entries: {len(unique_data)}")
    print(f"   Duplicates removed: {duplicates}")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    print(f"\n📊 Final combined dataset: {len(unique_data)} entries")
    
    return len(unique_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine multiple NER datasets')
    parser.add_argument('--datasets', nargs='+', required=True, help='Input dataset files')
    parser.add_argument('--output', type=str, required=True, help='Output combined dataset file')
    
    args = parser.parse_args()
    
    combine_datasets(args.datasets, args.output)
