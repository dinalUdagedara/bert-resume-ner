#!/usr/bin/env python3
"""
Analyze PERSON labels to find remaining mislabeling issues.
"""

import json
import sys
from collections import defaultdict

def analyze_person_labels(json_file_path, sample_size=1000):
    """Analyze PERSON labels for potential mislabeling."""
    
    print(f"Loading {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries\n")
    
    person_samples = []
    suspicious_person = []
    
    # Keywords that indicate NOT a person name
    not_person_keywords = [
        'college', 'university', 'school', 'institute', 'academy',
        'institution', 'education', 'vidyalaya', 'polytechnic', 'campus',
        'ltd', 'limited', 'inc', 'corporation', 'corp', 'company',
        'technologies', 'solutions', 'services', 'systems',
        'b.tech', 'b.e', 'm.tech', 'm.e', 'bachelor', 'master', 'degree',
        'diploma', 'pg', 'ph.d', 'mba', 'mca', 'b.sc', 'm.sc'
    ]
    
    entries_to_check = data[:sample_size] if sample_size else data
    
    for entry_idx, entry in enumerate(entries_to_check):
        if 'text' not in entry or 'annotations' not in entry:
            continue
        
        text = entry['text']
        annotations = entry['annotations']
        
        for ann_idx, annotation in enumerate(annotations):
            if not isinstance(annotation, list) or len(annotation) != 3:
                continue
            
            start, end, label = annotation
            
            if label != 'PERSON':
                continue
            
            # Validate positions
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Extract annotated text
            annotated_text = text[start:end].strip()
            
            if not annotated_text:
                continue
            
            text_lower = annotated_text.lower()
            
            # Check if it looks suspicious
            is_suspicious = any(keyword in text_lower for keyword in not_person_keywords)
            
            person_samples.append({
                'entry_idx': entry_idx,
                'annotation_idx': ann_idx,
                'text': annotated_text,
                'position': [start, end],
                'suspicious': is_suspicious
            })
            
            if is_suspicious:
                suspicious_person.append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'text': annotated_text,
                    'position': [start, end]
                })
    
    return person_samples, suspicious_person


def main():
    json_file = "data/dataset-5000/train_fixed.json"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    try:
        person_samples, suspicious = analyze_person_labels(json_file, sample_size=1000)
        
        print("="*80)
        print("PERSON LABEL ANALYSIS")
        print("="*80)
        print(f"\nTotal PERSON labels found: {len(person_samples)}")
        print(f"Suspicious PERSON labels: {len(suspicious)}")
        
        if suspicious:
            print("\n" + "-"*80)
            print("SUSPICIOUS PERSON LABELS (likely mislabeled):")
            print("-"*80)
            
            for i, item in enumerate(suspicious[:50], 1):  # Show first 50
                print(f"\n{i}. Entry {item['entry_idx']}, Annotation {item['annotation_idx']}")
                print(f"   Text: \"{item['text']}\"")
                print(f"   Position: {item['position']}")
        
        print("\n" + "="*80)
        
        # Save results
        output_file = json_file.replace('.json', '_person_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_person_labels': len(person_samples),
                'suspicious_count': len(suspicious),
                'suspicious_labels': suspicious[:100]  # Save first 100
            }, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
