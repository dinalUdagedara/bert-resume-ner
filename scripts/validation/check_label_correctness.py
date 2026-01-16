#!/usr/bin/env python3
"""
Script to check if entity labels are being used correctly.
Examines what text is actually being labeled with each entity type.
"""

import json
import sys
from collections import defaultdict
from tqdm import tqdm


def analyze_label_correctness(json_file_path, sample_size=100):
    """
    Analyze if labels are being used correctly by examining sample annotations.
    """
    print(f"Loading {json_file_path}...")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries\n")
    
    # Collect samples for each label type
    label_samples = defaultdict(list)
    label_counts = defaultdict(int)
    
    # Analyze first N entries (or all if sample_size is None)
    entries_to_check = data[:sample_size] if sample_size else data
    
    print(f"Analyzing {len(entries_to_check)} entries...")
    
    for entry_idx, entry in enumerate(tqdm(entries_to_check, desc="Processing")):
        if 'text' not in entry or 'annotations' not in entry:
            continue
        
        text = entry['text']
        annotations = entry['annotations']
        
        for ann_idx, annotation in enumerate(annotations):
            if not isinstance(annotation, list) or len(annotation) != 3:
                continue
            
            start, end, label = annotation
            
            # Validate positions
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Extract annotated text
            annotated_text = text[start:end].strip()
            
            if annotated_text:  # Only add non-empty annotations
                label_counts[label] += 1
                
                # Store sample (limit to 10 per label)
                if len(label_samples[label]) < 10:
                    label_samples[label].append({
                        'text': annotated_text,
                        'entry_idx': entry_idx,
                        'position': [start, end]
                    })
    
    return label_samples, label_counts


def print_label_analysis(label_samples, label_counts):
    """Print analysis of label usage."""
    print("\n" + "="*80)
    print("LABEL CORRECTNESS ANALYSIS")
    print("="*80)
    
    # Expected label meanings based on README
    expected_meanings = {
        'SKILL': 'Technical skills, tools, and competencies',
        'DESIGNATION': 'Job titles and positions',
        'LOCATION': 'Cities, states, and geographical locations',
        'EXPERIENCE': 'Work experience and duration',
        'PERSON': 'Names and personal information',
        'EDUCATION': 'Degrees, colleges, and educational background',
        'EXPERTISE': 'Areas of professional expertise',
        'EMAIL': 'Contact email addresses',
        'COMPANY': 'Company and organization names',
        'COLLABORATION': 'Teamwork and collaboration skills',
        'LANGUAGE': 'Language proficiencies',
        'ACTION': 'Professional actions and responsibilities',
        'CERTIFICATION': 'Professional certifications',
        'OTHER': 'Miscellaneous entities'
    }
    
    print("\nLabel Usage Samples (first 10 examples per label):\n")
    
    for label in sorted(label_samples.keys()):
        print("-" * 80)
        print(f"\n{label} ({label_counts[label]:,} instances)")
        if label in expected_meanings:
            print(f"Expected: {expected_meanings[label]}")
        print("\nSample annotations:")
        
        for i, sample in enumerate(label_samples[label], 1):
            # Truncate long text
            text_preview = sample['text']
            if len(text_preview) > 100:
                text_preview = text_preview[:100] + "..."
            
            print(f"  {i}. \"{text_preview}\"")
            print(f"     Entry: {sample['entry_idx']}, Position: {sample['position']}")
    
    print("\n" + "="*80)
    
    # Check for potential issues
    print("\nPOTENTIAL ISSUES TO REVIEW:\n")
    
    issues_found = False
    
    # Check PERSON label
    if 'PERSON' in label_samples:
        person_samples = [s['text'] for s in label_samples['PERSON']]
        # Check if PERSON labels look like names
        non_name_like = [s for s in person_samples if not any(c.isupper() for c in s.split()[:2])]
        if non_name_like:
            print("⚠️  PERSON label may include non-name entities:")
            for sample in non_name_like[:3]:
                print(f"   - \"{sample}\"")
            issues_found = True
    
    # Check EMAIL label
    if 'EMAIL' in label_samples:
        email_samples = [s['text'] for s in label_samples['EMAIL']]
        non_email_like = [s for s in email_samples if '@' not in s and 'email' not in s.lower()]
        if non_email_like:
            print("⚠️  EMAIL label may include non-email entities:")
            for sample in non_email_like[:3]:
                print(f"   - \"{sample}\"")
            issues_found = True
    
    # Check LOCATION label
    if 'LOCATION' in label_samples:
        location_samples = [s['text'] for s in label_samples['LOCATION']]
        # Locations should typically be capitalized or be place names
        suspicious = [s for s in location_samples if len(s) < 3 or s.isdigit()]
        if suspicious:
            print("⚠️  LOCATION label may include suspicious entities:")
            for sample in suspicious[:3]:
                print(f"   - \"{sample}\"")
            issues_found = True
    
    # Check SKILL vs OTHER distinction
    if 'SKILL' in label_samples and 'OTHER' in label_samples:
        skill_samples = [s['text'].lower() for s in label_samples['SKILL']]
        other_samples = [s['text'].lower() for s in label_samples['OTHER']]
        
        # Check if there's overlap that might indicate mislabeling
        common_terms = set(skill_samples) & set(other_samples)
        if common_terms:
            print("⚠️  SKILL and OTHER labels have overlapping terms:")
            for term in list(common_terms)[:5]:
                print(f"   - \"{term}\"")
            issues_found = True
    
    if not issues_found:
        print("✅ No obvious issues detected in label usage.")
        print("   (This is a basic check - manual review recommended for critical applications)")
    
    print("\n" + "="*80)


def main():
    json_file = "data/dataset-5000/train.json"
    sample_size = 500  # Check first 500 entries for detailed analysis
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    if len(sys.argv) > 2:
        sample_size = int(sys.argv[2])
    
    try:
        label_samples, label_counts = analyze_label_correctness(json_file, sample_size)
        print_label_analysis(label_samples, label_counts)
        
        # Save detailed analysis
        output_file = json_file.replace('.json', '_label_analysis.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'label_samples': {k: v for k, v in label_samples.items()},
                'label_counts': dict(label_counts)
            }, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed analysis saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
