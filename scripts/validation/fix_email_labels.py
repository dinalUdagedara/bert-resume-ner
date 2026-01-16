"""
Fix email mislabeling in dataset.
Removes annotations labeled as "Email Address" that are actually URLs or invalid.
"""

import json
import re
from typing import List, Dict


def is_valid_email(text: str) -> bool:
    """Check if text is a valid email address."""
    text = text.strip()
    
    # Must contain @
    if '@' not in text:
        return False
    
    # Basic email pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if it matches email pattern
    if re.match(email_pattern, text):
        return True
    
    # Check for common email domains
    email_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 
                     'icloud.com', 'aol.com', 'mail.com', 'protonmail.com',
                     'company.com', 'corp.com', 'edu', 'org', 'net']
    
    text_lower = text.lower()
    if '@' in text_lower:
        domain = text_lower.split('@')[-1] if '@' in text_lower else ''
        # Check if domain contains any valid email domain
        for valid_domain in email_domains:
            if valid_domain in domain:
                return True
    
    return False


def is_url(text: str) -> bool:
    """Check if text is a URL."""
    text = text.strip().lower()
    
    # Common URL patterns
    url_patterns = [
        r'^https?://',
        r'^www\.',
        r'\.com/',
        r'\.org/',
        r'\.net/',
        r'\.edu/',
        r'indeed\.com',
        r'linkedin\.com',
        r'github\.com',
        r'facebook\.com',
        r'twitter\.com',
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def fix_email_labels(data: List[Dict]) -> List[Dict]:
    """Fix email mislabeling in dataset."""
    fixed_data = []
    removed_count = 0
    fixed_count = 0
    
    for entry in data:
        text = entry['text']
        annotations = entry['annotations']
        
        fixed_annotations = []
        
        for ann in annotations:
            if len(ann) == 3:
                start, end, label = ann
                
                # Check if it's labeled as Email Address
                if label == 'Email Address' or 'email' in label.lower():
                    annotated_text = text[start:end].strip()
                    
                    # Check if it's actually a valid email
                    if is_valid_email(annotated_text):
                        # Keep it
                        fixed_annotations.append(ann)
                    elif is_url(annotated_text):
                        # Remove - it's a URL, not an email
                        removed_count += 1
                    elif '@' not in annotated_text:
                        # Remove - no @ symbol
                        removed_count += 1
                    else:
                        # Has @ but doesn't look like valid email - remove
                        removed_count += 1
                else:
                    # Not an email label, keep it
                    fixed_annotations.append(ann)
        
        if fixed_annotations:
            fixed_data.append({
                'text': text,
                'annotations': fixed_annotations
            })
    
    print(f"✅ Fixed email labels:")
    print(f"   Removed invalid email annotations: {removed_count}")
    print(f"   Kept valid entries: {len(fixed_data)}")
    
    return fixed_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix email mislabeling in dataset')
    parser.add_argument('input', type=str, help='Input dataset JSON file')
    parser.add_argument('output', type=str, help='Output fixed dataset JSON file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIXING EMAIL MISLABELING")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} entries")
    
    # Fix email labels
    print(f"\nFixing email mislabeling...")
    fixed_data = fix_email_labels(data)
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, ensure_ascii=False, indent=2)
    
    print("✅ Done!")
    
    # Validate
    print(f"\nValidating fixed dataset...")
    from collections import Counter
    entity_counter = Counter()
    for entry in fixed_data:
        for ann in entry['annotations']:
            if len(ann) == 3:
                entity_counter[ann[2]] += 1
    
    email_count = entity_counter.get('Email Address', 0)
    print(f"  Email Address annotations remaining: {email_count}")
    print(f"  (Should be only valid emails now)")


if __name__ == "__main__":
    main()
