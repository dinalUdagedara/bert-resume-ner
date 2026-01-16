"""
Comprehensive dataset validation script.
Checks data quality, label correctness, and alignment issues.
"""

import json
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict


def load_dataset(json_file_path: str) -> List[Dict]:
    """Load dataset in the expected format."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return []


def validate_annotation(text: str, start: int, end: int, label: str) -> Tuple[bool, str]:
    """Validate a single annotation."""
    issues = []
    
    # Check bounds
    if start < 0:
        issues.append(f"start < 0: {start}")
    if end > len(text):
        issues.append(f"end > text length: {end} > {len(text)}")
    if start >= end:
        issues.append(f"start >= end: {start} >= {end}")
    
    # Check if text matches
    if 0 <= start < end <= len(text):
        annotated_text = text[start:end]
        if not annotated_text.strip():
            issues.append(f"annotated text is empty/whitespace: '{annotated_text}'")
    
    return len(issues) == 0, "; ".join(issues) if issues else "OK"


def check_overlaps(annotations: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str, int, int, str]]:
    """Check for overlapping annotations."""
    overlaps = []
    sorted_anns = sorted(annotations, key=lambda x: (x[0], x[1]))
    
    for i in range(len(sorted_anns) - 1):
        start1, end1, label1 = sorted_anns[i]
        start2, end2, label2 = sorted_anns[i + 1]
        
        # Check if they overlap (not just adjacent)
        if start2 < end1:
            overlaps.append((start1, end1, label1, start2, end2, label2))
    
    return overlaps


def validate_dataset(data: List[Dict], verbose: bool = True) -> Dict:
    """Comprehensive dataset validation."""
    stats = {
        'total_entries': len(data),
        'valid_entries': 0,
        'invalid_entries': 0,
        'total_annotations': 0,
        'valid_annotations': 0,
        'invalid_annotations': 0,
        'overlapping_annotations': 0,
        'label_distribution': Counter(),
        'issues': [],
        'sample_issues': []
    }
    
    if verbose:
        print("=" * 70)
        print("DATASET VALIDATION REPORT")
        print("=" * 70)
        print(f"\n📊 Total entries: {len(data)}\n")
    
    for idx, entry in enumerate(data):
        entry_issues = []
        
        # Check required fields
        if 'text' not in entry:
            entry_issues.append("Missing 'text' field")
            stats['invalid_entries'] += 1
            continue
        
        if 'annotations' not in entry:
            entry_issues.append("Missing 'annotations' field")
            stats['invalid_entries'] += 1
            continue
        
        text = entry['text']
        annotations = entry['annotations']
        
        if not isinstance(annotations, list):
            entry_issues.append(f"'annotations' is not a list: {type(annotations)}")
            stats['invalid_entries'] += 1
            continue
        
        # Validate each annotation
        valid_annotations = []
        for ann_idx, ann in enumerate(annotations):
            stats['total_annotations'] += 1
            
            # Check format
            if not isinstance(ann, list) or len(ann) != 3:
                entry_issues.append(f"Annotation {ann_idx}: Invalid format (expected [start, end, label], got {ann})")
                stats['invalid_annotations'] += 1
                continue
            
            start, end, label = ann
            
            # Type checks
            if not isinstance(start, int) or not isinstance(end, int):
                entry_issues.append(f"Annotation {ann_idx}: start/end must be integers (got {type(start)}, {type(end)})")
                stats['invalid_annotations'] += 1
                continue
            
            if not isinstance(label, str):
                entry_issues.append(f"Annotation {ann_idx}: label must be string (got {type(label)})")
                stats['invalid_annotations'] += 1
                continue
            
            # Validate annotation
            is_valid, issue_msg = validate_annotation(text, start, end, label)
            
            if is_valid:
                valid_annotations.append((start, end, label))
                stats['valid_annotations'] += 1
                stats['label_distribution'][label] += 1
            else:
                entry_issues.append(f"Annotation {ann_idx} ({label}): {issue_msg}")
                stats['invalid_annotations'] += 1
        
        # Check for overlaps
        if len(valid_annotations) > 1:
            overlaps = check_overlaps(valid_annotations)
            if overlaps:
                stats['overlapping_annotations'] += len(overlaps)
                for ov in overlaps:
                    entry_issues.append(f"Overlap: {ov[2]} [{ov[0]}:{ov[1]}] overlaps with {ov[5]} [{ov[3]}:{ov[4]}]")
        
        # Entry summary
        if entry_issues:
            stats['invalid_entries'] += 1
            if len(stats['sample_issues']) < 5:  # Keep first 5 examples
                stats['sample_issues'].append({
                    'entry_idx': idx,
                    'text_preview': text[:100] + "..." if len(text) > 100 else text,
                    'issues': entry_issues
                })
        else:
            stats['valid_entries'] += 1
    
    # Calculate percentages
    if stats['total_entries'] > 0:
        stats['valid_entry_pct'] = (stats['valid_entries'] / stats['total_entries']) * 100
    else:
        stats['valid_entry_pct'] = 0
    
    if stats['total_annotations'] > 0:
        stats['valid_annotation_pct'] = (stats['valid_annotations'] / stats['total_annotations']) * 100
    else:
        stats['valid_annotation_pct'] = 0
    
    # Print report
    if verbose:
        print("✅ VALIDATION RESULTS")
        print("-" * 70)
        print(f"Valid entries:     {stats['valid_entries']}/{stats['total_entries']} ({stats['valid_entry_pct']:.2f}%)")
        print(f"Invalid entries:  {stats['invalid_entries']}/{stats['total_entries']} ({100-stats['valid_entry_pct']:.2f}%)")
        print(f"\nValid annotations: {stats['valid_annotations']}/{stats['total_annotations']} ({stats['valid_annotation_pct']:.2f}%)")
        print(f"Invalid annotations: {stats['invalid_annotations']}")
        print(f"Overlapping annotations: {stats['overlapping_annotations']}")
        
        print(f"\n📋 LABEL DISTRIBUTION (Top 15):")
        print("-" * 70)
        for label, count in stats['label_distribution'].most_common(15):
            pct = (count / stats['valid_annotations'] * 100) if stats['valid_annotations'] > 0 else 0
            print(f"  {label:30s} {count:6d} ({pct:5.2f}%)")
        
        if stats['sample_issues']:
            print(f"\n⚠️  SAMPLE ISSUES (showing first {len(stats['sample_issues'])}):")
            print("-" * 70)
            for sample in stats['sample_issues']:
                print(f"\nEntry {sample['entry_idx']}:")
                print(f"  Text preview: {sample['text_preview']}")
                print(f"  Issues:")
                for issue in sample['issues']:
                    print(f"    - {issue}")
        
        print("\n" + "=" * 70)
        
        # Overall assessment
        if stats['valid_entry_pct'] >= 95 and stats['valid_annotation_pct'] >= 95:
            print("✅ Dataset quality: EXCELLENT")
        elif stats['valid_entry_pct'] >= 90 and stats['valid_annotation_pct'] >= 90:
            print("✅ Dataset quality: GOOD")
        elif stats['valid_entry_pct'] >= 80 and stats['valid_annotation_pct'] >= 80:
            print("⚠️  Dataset quality: ACCEPTABLE (some issues found)")
        else:
            print("❌ Dataset quality: POOR (many issues found)")
        
        print("=" * 70)
    
    return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate NER dataset')
    parser.add_argument('dataset', type=str, help='Path to dataset JSON file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_dataset(args.dataset)
    
    if not data:
        print(f"❌ Failed to load dataset from {args.dataset}")
        sys.exit(1)
    
    # Validate
    stats = validate_dataset(data, verbose=not args.quiet)
    
    # Exit code based on quality
    if stats['valid_entry_pct'] < 80 or stats['valid_annotation_pct'] < 80:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
