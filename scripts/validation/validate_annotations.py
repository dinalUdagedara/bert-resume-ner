#!/usr/bin/env python3
"""
Validation script for checking annotation positions in train.json
Validates that all annotation start/end positions are correct and within text bounds.
"""

import json
import sys
from collections import defaultdict
from tqdm import tqdm


def validate_annotations(json_file_path):
    """
    Validate annotation positions against text in the dataset.
    
    Returns:
        dict: Validation results with statistics and errors
    """
    print(f"Loading {json_file_path}...")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries\n")
    
    # Statistics
    stats = {
        'total_entries': len(data),
        'total_annotations': 0,
        'valid_annotations': 0,
        'invalid_annotations': 0,
        'errors': [],
        'error_types': defaultdict(int),
        'entity_label_counts': defaultdict(int)
    }
    
    # Validation checks
    print("Validating annotations...")
    for entry_idx, entry in enumerate(tqdm(data, desc="Processing")):
        if 'text' not in entry or 'annotations' not in entry:
            stats['errors'].append({
                'entry_idx': entry_idx,
                'error': 'Missing text or annotations field',
                'entry': entry
            })
            stats['error_types']['missing_fields'] += 1
            continue
        
        text = entry['text']
        text_length = len(text)
        annotations = entry['annotations']
        
        if not isinstance(annotations, list):
            stats['errors'].append({
                'entry_idx': entry_idx,
                'error': 'Annotations is not a list',
                'entry': entry
            })
            stats['error_types']['invalid_annotations_type'] += 1
            continue
        
        stats['total_annotations'] += len(annotations)
        
        for ann_idx, annotation in enumerate(annotations):
            # Check annotation format
            if not isinstance(annotation, list) or len(annotation) != 3:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Invalid annotation format: {annotation}',
                    'expected_format': '[start, end, label]'
                })
                stats['error_types']['invalid_format'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            try:
                start, end, label = annotation
            except ValueError:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Cannot unpack annotation: {annotation}',
                })
                stats['error_types']['unpack_error'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            # Check types
            if not isinstance(start, int) or not isinstance(end, int):
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Start/end positions must be integers: start={start} (type: {type(start)}), end={end} (type: {type(end)})',
                    'annotation': annotation
                })
                stats['error_types']['non_integer_positions'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            # Check label type
            if not isinstance(label, str):
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Label must be string: {label} (type: {type(label)})',
                    'annotation': annotation
                })
                stats['error_types']['non_string_label'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            # Check position validity
            if start < 0:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Start position is negative: {start}',
                    'annotation': annotation
                })
                stats['error_types']['negative_start'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            if end < 0:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'End position is negative: {end}',
                    'annotation': annotation
                })
                stats['error_types']['negative_end'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            if start >= end:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Start position >= end position: start={start}, end={end}',
                    'annotation': annotation
                })
                stats['error_types']['start_ge_end'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            if start > text_length:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Start position exceeds text length: start={start}, text_length={text_length}',
                    'annotation': annotation,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                })
                stats['error_types']['start_out_of_bounds'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            if end > text_length:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'End position exceeds text length: end={end}, text_length={text_length}',
                    'annotation': annotation,
                    'text_preview': text[:100] + '...' if len(text) > 100 else text
                })
                stats['error_types']['end_out_of_bounds'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            # Extract annotated text for verification
            try:
                annotated_text = text[start:end]
            except Exception as e:
                stats['errors'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'error': f'Error extracting text: {str(e)}',
                    'annotation': annotation,
                    'start': start,
                    'end': end,
                    'text_length': text_length
                })
                stats['error_types']['extraction_error'] += 1
                stats['invalid_annotations'] += 1
                continue
            
            # If we get here, annotation is valid
            stats['valid_annotations'] += 1
            stats['entity_label_counts'][label] += 1
    
    return stats


def print_report(stats):
    """Print validation report."""
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    
    print(f"\nTotal Entries: {stats['total_entries']:,}")
    print(f"Total Annotations: {stats['total_annotations']:,}")
    print(f"Valid Annotations: {stats['valid_annotations']:,}")
    print(f"Invalid Annotations: {stats['invalid_annotations']:,}")
    
    if stats['total_annotations'] > 0:
        validity_rate = (stats['valid_annotations'] / stats['total_annotations']) * 100
        print(f"Validity Rate: {validity_rate:.2f}%")
    
    print("\n" + "-"*80)
    print("ERROR SUMMARY")
    print("-"*80)
    
    if stats['error_types']:
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")
    else:
        print("  No errors found!")
    
    print("\n" + "-"*80)
    print("ENTITY LABEL DISTRIBUTION")
    print("-"*80)
    
    for label, count in sorted(stats['entity_label_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count:,}")
    
    # Print detailed errors (limit to first 20)
    if stats['errors']:
        print("\n" + "-"*80)
        print(f"DETAILED ERRORS (showing first 20 of {len(stats['errors'])}):")
        print("-"*80)
        
        for i, error in enumerate(stats['errors'][:20]):
            print(f"\nError #{i+1}:")
            print(f"  Entry Index: {error.get('entry_idx', 'N/A')}")
            print(f"  Annotation Index: {error.get('annotation_idx', 'N/A')}")
            print(f"  Error: {error.get('error', 'N/A')}")
            if 'annotation' in error:
                print(f"  Annotation: {error['annotation']}")
            if 'text_preview' in error:
                print(f"  Text Preview: {error['text_preview']}")
    
    print("\n" + "="*80)
    
    # Return validation status
    return stats['invalid_annotations'] == 0


def main():
    json_file = "data/dataset-5000/train.json"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    try:
        stats = validate_annotations(json_file)
        is_valid = print_report(stats)
        
        # Save detailed report to file
        report_file = json_file.replace('.json', '_validation_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {report_file}")
        
        if is_valid:
            print("\n✅ All annotations are valid!")
            sys.exit(0)
        else:
            print(f"\n❌ Found {stats['invalid_annotations']} invalid annotations")
            sys.exit(1)
            
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
