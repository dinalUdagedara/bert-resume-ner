"""
Download and convert DataTurks Resume NER dataset.
Dataset: https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy
220 annotated resumes in JSON format.
"""

import json
import requests
import os
from typing import List, Dict

def download_dataturks_dataset(output_file: str):
    """Download DataTurks resume dataset from GitHub."""
    print("=" * 70)
    print("DOWNLOADING DATATURKS RESUME DATASET")
    print("=" * 70)
    
    # GitHub raw URL (this is a common location, may need to verify)
    github_urls = [
        "https://raw.githubusercontent.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/master/Entity Recognition in Resumes.json",
        "https://raw.githubusercontent.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/master/training_data.json",
        "https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy/raw/master/Entity Recognition in Resumes.json"
    ]
    
    data = None
    for url in github_urls:
        try:
            print(f"\nTrying: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Downloaded successfully!")
                break
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    if data is None:
        print("\n❌ Could not download automatically")
        print("\n📥 Manual download required:")
        print("   1. Go to: https://github.com/DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy")
        print("   2. Download the JSON file")
        print("   3. Place it in the data/ directory")
        return False
    
    # Convert format
    print(f"\nConverting format...")
    converted_data = []
    
    for entry in data:
        if 'content' in entry and 'annotation' in entry:
            text = entry['content']
            annotations = entry.get('annotation', [])
            
            converted_annotations = []
            for ann in annotations:
                if isinstance(ann, dict) and 'points' in ann:
                    points = ann['points']
                    labels = ann.get('label', [])
                    
                    if not isinstance(labels, list):
                        labels = [labels]
                    
                    for point in points:
                        if isinstance(point, dict):
                            start = point.get('start', 0)
                            end = point.get('end', 0)
                            
                            for label in labels:
                                if start < end and end <= len(text):
                                    converted_annotations.append([start, end, label])
            
            if converted_annotations:
                converted_data.append({
                    'text': text,
                    'annotations': converted_annotations
                })
    
    print(f"✅ Converted {len(converted_data)} entries")
    
    # Save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download DataTurks resume dataset')
    parser.add_argument('output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    download_dataturks_dataset(args.output)
