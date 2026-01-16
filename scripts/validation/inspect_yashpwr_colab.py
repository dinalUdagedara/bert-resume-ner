"""
Colab-friendly script to inspect yashpwr/resume-ner-training-data dataset.
Run this in a Colab cell to check data quality before using.
"""

# Install if needed (uncomment if needed)
# !pip install datasets

from datasets import load_dataset
import json
from collections import Counter

print("=" * 70)
print("YASHPWR RESUME NER DATASET INSPECTION")
print("=" * 70)

# Download dataset
print("\n📥 Downloading yashpwr/resume-ner-training-data...")
try:
    dataset = load_dataset("yashpwr/resume-ner-training-data")
    print("✅ Dataset downloaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    raise

# Show splits
print(f"\n📊 Available splits: {list(dataset.keys())}")

# Inspect each split
for split_name in dataset.keys():
    print(f"\n{'=' * 70}")
    print(f"SPLIT: {split_name.upper()}")
    print(f"{'=' * 70}")
    
    split_data = dataset[split_name]
    print(f"Number of examples: {len(split_data):,}")
    
    # Show first example structure
    if len(split_data) > 0:
        example = split_data[0]
        print(f"\n📝 Example Structure:")
        print(f"  Keys: {list(example.keys())}")
        
        # Show sample data
        print(f"\n📋 Sample Data (first example):")
        for key, value in example.items():
            if isinstance(value, str):
                if len(value) > 300:
                    print(f"  {key}: {value[:300]}... (length: {len(value)})")
                else:
                    print(f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"    First 5 items: {value[:5]}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
        
        # Analyze entity distribution
        print(f"\n📊 Entity Distribution:")
        print("-" * 70)
        
        entity_counter = Counter()
        total_entities = 0
        sample_pairs = []
        
        # Check format
        if 'tokens' in example and 'ner_tags' in example:
            print("  Format: tokens + ner_tags (BIO format)")
            
            for i, ex in enumerate(split_data):
                tokens = ex.get('tokens', [])
                tags = ex.get('ner_tags', [])
                
                for token, tag in zip(tokens, tags):
                    if tag != 'O' and tag != 0:
                        entity_counter[tag] += 1
                        total_entities += 1
                
                # Collect sample
                if i < 2 and len(tokens) > 0:
                    sample_pairs.append(list(zip(tokens[:15], tags[:15])))
        
        elif 'labels' in example:
            print("  Format: labels")
            for ex in split_data:
                labels = ex.get('labels', [])
                for label in labels:
                    entity_counter[label] += 1
                    total_entities += 1
        
        if total_entities > 0:
            print(f"\n  Total entity tokens: {total_entities:,}")
            print(f"  Unique entity types: {len(entity_counter)}")
            print(f"\n  Top 15 Entity Types:")
            for entity, count in entity_counter.most_common(15):
                pct = (count / total_entities * 100)
                print(f"    {str(entity):30s} {count:8,} ({pct:5.2f}%)")
        
        # Show sample token-tag pairs
        if sample_pairs:
            print(f"\n  📋 Sample Token-Tag Pairs:")
            for idx, pairs in enumerate(sample_pairs):
                print(f"\n    Example {idx + 1}:")
                for token, tag in pairs[:10]:
                    print(f"      {token:20s} -> {tag}")

# Check if format matches our training code
print(f"\n{'=' * 70}")
print("FORMAT COMPATIBILITY CHECK")
print(f"{'=' * 70}")

example = dataset[list(dataset.keys())[0]][0]

if 'text' in example and 'annotations' in example:
    print("✅ Format matches our training code!")
    print("   - Has 'text' field")
    print("   - Has 'annotations' field")
    print("\n   Ready to use directly!")
else:
    print("⚠️  Format needs conversion")
    print(f"   Current keys: {list(example.keys())}")
    print("   Expected: 'text' and 'annotations'")
    print("\n   Will need to convert from tokens+tags to text+annotations")

print(f"\n{'=' * 70}")
print("INSPECTION COMPLETE")
print(f"{'=' * 70}")
print("\n💡 Next steps:")
print("   1. Review the entity distribution above")
print("   2. Check if format matches (or needs conversion)")
print("   3. If good, proceed with conversion/usage")
