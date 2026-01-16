#!/usr/bin/env python3
"""
Script to fix label correctness issues in the dataset.
Identifies and corrects mislabeled annotations.
"""

import json
import re
import sys
from collections import defaultdict
from tqdm import tqdm


# Keywords that indicate an entity should be EDUCATION, not PERSON
EDUCATION_KEYWORDS = [
    'college', 'university', 'school', 'institute', 'academy', 
    'institution', 'education', 'vidyalaya', 'polytechnic', 'campus',
    'cdac', 'acts', 'training', 'centre', 'center'
]

# Keywords that indicate an entity should be COMPANY, not PERSON
COMPANY_KEYWORDS = [
    'ltd', 'limited', 'inc', 'corporation', 'corp', 'company', 
    'technologies', 'solutions', 'services', 'systems'
]

# Patterns for URLs (not emails)
URL_PATTERNS = [
    r'https?://',
    r'www\.',
    r'\.com/',
    r'\.org/',
    r'\.net/',
    r'indeed\.com',
    r'linkedin\.com',
    r'github\.com'
]

# Degree patterns that should be EDUCATION, not DESIGNATION
DEGREE_PATTERNS = [
    r'\bB\.?E\.?\b',
    r'\bB\.?Tech\.?\b',
    r'\bB\.?Sc\.?\b',
    r'\bB\.?A\.?\b',
    r'\bB\.?Com\.?\b',
    r'\bM\.?E\.?\b',
    r'\bM\.?Tech\.?\b',
    r'\bM\.?Sc\.?\b',
    r'\bM\.?A\.?\b',
    r'\bM\.?B\.?A\.?\b',
    r'\bM\.?C\.?A\.?\b',
    r'\bPh\.?D\.?\b',
    r'\bDiploma\b',
    r'\bPG\b',
    r'\bBachelor\b',
    r'\bMaster\b',
    r'\bDegree\b'
]


def is_url(text):
    """Check if text is a URL."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in URL_PATTERNS)


def is_email(text):
    """Check if text is actually an email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text.strip()))


def is_education_entity(text):
    """Check if text should be labeled as EDUCATION."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in EDUCATION_KEYWORDS)


def is_company_entity(text):
    """Check if text should be labeled as COMPANY."""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in COMPANY_KEYWORDS)


def is_degree(text):
    """Check if text is a degree (should be EDUCATION, not DESIGNATION)."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in DEGREE_PATTERNS)


def is_location(text):
    """Check if text looks like a location."""
    # Simple heuristic: locations are usually short, may contain commas, 
    # and often have state/country names
    location_indicators = [
        'karnataka', 'maharashtra', 'delhi', 'bangalore', 'mumbai', 
        'hyderabad', 'chennai', 'pune', 'kolkata', 'india', 'state',
        'city', 'district', 'taluk'
    ]
    text_lower = text.lower()
    return any(indicator in text_lower for indicator in location_indicators) and len(text) < 100


def clean_unicode_text(text):
    """Remove invalid Unicode surrogates from text."""
    if not isinstance(text, str):
        return text
    # Remove surrogates by encoding and decoding with error handling
    return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')


def clean_entry(entry):
    """Clean Unicode issues in an entry."""
    if isinstance(entry, dict):
        cleaned = {}
        for key, value in entry.items():
            if key == 'text' and isinstance(value, str):
                cleaned[key] = clean_unicode_text(value)
            else:
                cleaned[key] = value
        return cleaned
    return entry


def fix_annotations(data, dry_run=True):
    """
    Fix label issues in the dataset.
    
    Args:
        data: List of entries with text and annotations
        dry_run: If True, only report changes without modifying data
    
    Returns:
        dict: Statistics about fixes made
    """
    stats = {
        'total_entries': len(data),
        'total_annotations': 0,
        'fixes': defaultdict(int),
        'fixed_annotations': []
    }
    
    print(f"Analyzing {len(data)} entries...")
    
    for entry_idx, entry in enumerate(tqdm(data, desc="Processing")):
        if 'text' not in entry or 'annotations' not in entry:
            continue
        
        text = entry['text']
        annotations = entry['annotations']
        
        if not isinstance(annotations, list):
            continue
        
        stats['total_annotations'] += len(annotations)
        
        fixed_annotations = []
        
        for ann_idx, annotation in enumerate(annotations):
            if not isinstance(annotation, list) or len(annotation) != 3:
                fixed_annotations.append(annotation)
                continue
            
            start, end, label = annotation
            
            # Validate positions
            if start < 0 or end > len(text) or start >= end:
                fixed_annotations.append(annotation)
                continue
            
            # Extract annotated text
            annotated_text = text[start:end].strip()
            
            if not annotated_text:
                fixed_annotations.append(annotation)
                continue
            
            original_label = label
            new_label = label
            fix_reason = None
            
            # Fix 1: EMAIL label used for URLs
            if label == 'EMAIL' and is_url(annotated_text) and not is_email(annotated_text):
                new_label = 'OTHER'
                fix_reason = 'URL labeled as EMAIL'
                stats['fixes']['email_to_other'] += 1
            
            # Fix 2: PERSON label used for schools/colleges (should be EDUCATION)
            elif label == 'PERSON' and is_education_entity(annotated_text):
                new_label = 'EDUCATION'
                fix_reason = 'School/College labeled as PERSON'
                stats['fixes']['person_to_education'] += 1
            
            # Fix 2b: PERSON label used for degrees (should be EDUCATION)
            elif label == 'PERSON' and is_degree(annotated_text):
                new_label = 'EDUCATION'
                fix_reason = 'Degree labeled as PERSON'
                stats['fixes']['person_to_education'] += 1
            
            # Fix 3: PERSON label used for companies (should be COMPANY)
            elif label == 'PERSON' and is_company_entity(annotated_text):
                new_label = 'COMPANY'
                fix_reason = 'Company labeled as PERSON'
                stats['fixes']['person_to_company'] += 1
            
            # Fix 4: DESIGNATION label used for degrees (should be EDUCATION)
            elif label == 'DESIGNATION' and is_degree(annotated_text):
                new_label = 'EDUCATION'
                fix_reason = 'Degree labeled as DESIGNATION'
                stats['fixes']['designation_to_education'] += 1
            
            # Fix 5: LOCATION label used for URLs (should be OTHER)
            elif label == 'LOCATION' and is_url(annotated_text):
                new_label = 'OTHER'
                fix_reason = 'URL labeled as LOCATION'
                stats['fixes']['location_to_other'] += 1
            
            # Apply fix if label changed
            if new_label != original_label:
                fixed_annotations.append([start, end, new_label])
                stats['fixed_annotations'].append({
                    'entry_idx': entry_idx,
                    'annotation_idx': ann_idx,
                    'original_label': original_label,
                    'new_label': new_label,
                    'text': annotated_text[:100],  # Truncate for display
                    'reason': fix_reason
                })
            else:
                fixed_annotations.append(annotation)
        
        # Update annotations if not dry run
        if not dry_run:
            entry['annotations'] = fixed_annotations
    
    return stats


def print_fix_report(stats, dry_run=True):
    """Print report of fixes."""
    print("\n" + "="*80)
    print("LABEL FIX REPORT")
    print("="*80)
    
    mode = "DRY RUN (no changes made)" if dry_run else "FIXES APPLIED"
    print(f"\nMode: {mode}")
    
    print(f"\nTotal Entries: {stats['total_entries']:,}")
    print(f"Total Annotations: {stats['total_annotations']:,}")
    print(f"Total Fixes: {sum(stats['fixes'].values()):,}")
    
    print("\n" + "-"*80)
    print("FIXES BY TYPE:")
    print("-"*80)
    
    if stats['fixes']:
        for fix_type, count in sorted(stats['fixes'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {fix_type}: {count:,}")
    else:
        print("  No fixes needed!")
    
    # Show sample fixes
    if stats['fixed_annotations']:
        print("\n" + "-"*80)
        print(f"SAMPLE FIXES (showing first 20 of {len(stats['fixed_annotations'])}):")
        print("-"*80)
        
        for i, fix in enumerate(stats['fixed_annotations'][:20], 1):
            print(f"\n{i}. Entry {fix['entry_idx']}, Annotation {fix['annotation_idx']}")
            print(f"   Original: {fix['original_label']} → New: {fix['new_label']}")
            print(f"   Reason: {fix['reason']}")
            print(f"   Text: \"{fix['text']}\"")
    
    print("\n" + "="*80)
    
    return sum(stats['fixes'].values()) > 0


def main():
    input_file = "data/dataset-5000/train.json"
    output_file = "data/dataset-5000/train_fixed.json"
    dry_run = True
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3].lower() == 'apply':
        dry_run = False
    
    try:
        print(f"Loading {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} entries\n")
        
        # Fix annotations
        stats = fix_annotations(data, dry_run=dry_run)
        
        # Print report
        has_fixes = print_fix_report(stats, dry_run=dry_run)
        
        # Save fixed data if not dry run
        if not dry_run and has_fixes:
            print(f"\nSaving fixed data to {output_file}...")
            # Clean Unicode issues before saving
            cleaned_data = [clean_entry(entry) for entry in data]
            with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=None, separators=(',', ':'))
            print(f"✅ Fixed dataset saved to: {output_file}")
            
            # Also save detailed fix report
            report_file = output_file.replace('.json', '_fix_report.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"📊 Detailed fix report saved to: {report_file}")
        elif dry_run:
            print("\n" + "="*80)
            print("⚠️  This was a DRY RUN - no changes were made.")
            print("To apply fixes, run:")
            print(f"  python fix_label_issues.py {input_file} {output_file} apply")
            print("="*80)
        
        if not has_fixes:
            print("\n✅ No fixes needed - all labels appear correct!")
        
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
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
