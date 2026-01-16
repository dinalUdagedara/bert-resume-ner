"""
Improved test script with better entity extraction.
Handles subword tokenization and label alignment properly.
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np


def extract_entities_improved(model, tokenizer, idx2tag, device, text, max_len=500):
    """
    Improved entity extraction that properly handles subword tokenization.
    """
    # Tokenize with offset mapping
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        padding=True,
        add_special_tokens=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    
    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # Extract entities with proper alignment
    entities = []
    current_entity = None
    
    for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
        token_id = input_ids[0][i].item()
        
        # Skip special tokens and padding
        if token_id in [cls_token_id, sep_token_id, pad_token_id]:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Skip tokens with no offset (shouldn't happen, but safety check)
        if offset[0] == 0 and offset[1] == 0 and token_id != cls_token_id:
            continue
        
        label = idx2tag[pred_id]
        start, end = offset
        
        # Skip 'O' (outside) and 'UNKNOWN' labels
        if label in ['O', 'UNKNOWN']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Handle BIO format
        is_bio = False
        actual_label = label
        is_beginning = False
        
        if label.startswith('B-'):
            is_bio = True
            actual_label = label[2:]
            is_beginning = True
        elif label.startswith('I-'):
            is_bio = True
            actual_label = label[2:]
            is_beginning = False
        
        # Start new entity if:
        # 1. No current entity
        # 2. B- tag (beginning)
        # 3. Different label
        # 4. Not adjacent (gap > 1 character)
        if current_entity is None:
            if is_bio and is_beginning:
                current_entity = {
                    'label': actual_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end] if start < len(text) and end <= len(text) else ''
                }
            elif not is_bio:
                current_entity = {
                    'label': actual_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end] if start < len(text) and end <= len(text) else ''
                }
        else:
            # Continue entity if same label and adjacent
            if current_entity['label'] == actual_label:
                gap = start - current_entity['end']
                # Allow small gaps (up to 1 char, usually spaces)
                if gap <= 1:
                    current_entity['end'] = end
                    if start < len(text) and end <= len(text):
                        current_entity['text'] = text[current_entity['start']:end]
                else:
                    # Gap too large, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'label': actual_label,
                        'start': start,
                        'end': end,
                        'text': text[start:end] if start < len(text) and end <= len(text) else ''
                    }
            else:
                # Different label, start new entity
                entities.append(current_entity)
                current_entity = {
                    'label': actual_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end] if start < len(text) and end <= len(text) else ''
                }
    
    # Add last entity
    if current_entity:
        entities.append(current_entity)
    
    # Clean and filter entities
    cleaned_entities = []
    seen = set()
    
    for ent in entities:
        # Filter very short entities (likely noise)
        text_clean = ent['text'].strip()
        if len(text_clean) < 2:
            continue
        
        # Filter common false positives
        if text_clean.lower() in ['the', 'a', 'an', 'is', 'are', 'was', 'were', 'at', 'in', 'on', 'to', 'for', 'of', 'and', 'or']:
            continue
        
        # Remove duplicates
        key = (ent['label'], text_clean.lower())
        if key not in seen:
            seen.add(key)
            cleaned_entities.append({
                'entity': ent['label'],
                'text': text_clean,
                'start': ent['start'],
                'end': ent['end']
            })
    
    return cleaned_entities


def test_model_improved(model_path='model-state.bin', vocab_path='vocab.txt', test_text=None):
    """Test the model with improved extraction."""
    
    if test_text is None:
        test_text = """
John Smith
Software Engineer at Google
Email: john.smith@email.com
Location: San Francisco, California

WORK EXPERIENCE
Senior Software Engineer
Google Inc.
5 years of experience in Python, Machine Learning, and Cloud Computing

EDUCATION
Bachelor of Science in Computer Science
Stanford University
2015-2019

SKILLS
Python, Java, TensorFlow, PyTorch, AWS, Docker, Kubernetes
"""
    
    print("="*80)
    print("TESTING TRAINED MODEL (IMPROVED EXTRACTION)")
    print("="*80)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 Device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device)
    tag2idx = checkpoint['tag2idx']
    idx2tag = checkpoint['idx2tag']
    
    print(f"✅ Loaded {len(tag2idx)} labels")
    print(f"Labels: {list(tag2idx.keys())}")
    
    tokenizer = BertTokenizerFast(vocab_path, lowercase=True)
    
    model = BertForTokenClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=len(tag2idx)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!\n")
    
    # Display test text
    print("-"*80)
    print("TEST TEXT:")
    print("-"*80)
    print(test_text)
    print()
    
    # Extract entities
    entities = extract_entities_improved(model, tokenizer, idx2tag, device, test_text)
    
    # Group by type
    entities_by_type = {}
    for ent in entities:
        label = ent['entity']
        if label not in entities_by_type:
            entities_by_type[label] = []
        entities_by_type[label].append(ent['text'])
    
    # Display results
    print("="*80)
    print("EXTRACTED ENTITIES:")
    print("="*80)
    
    if not entities_by_type:
        print("⚠️  No entities extracted. This might indicate:")
        print("   - Model needs more training")
        print("   - Text format doesn't match training data")
        print("   - Try testing on actual resume text from your dataset")
    else:
        for label in sorted(entities_by_type.keys()):
            print(f"\n📌 {label}:")
            unique_texts = list(set(entities_by_type[label]))  # Remove duplicates
            for text in unique_texts:
                print(f"   • {text}")
    
    print(f"\n✅ Total unique entities: {len(entities)}")
    print(f"✅ Entity types found: {len(entities_by_type)}")
    
    # Show confidence (optional - can add softmax probabilities)
    print("\n" + "-"*80)
    print("NOTE:")
    print("-"*80)
    print("Based on your training metrics:")
    print("  • SKILL, PERSON, EDUCATION should work reasonably well")
    print("  • EMAIL, LOCATION may have lower accuracy")
    print("  • Some entities might be misclassified due to overfitting")
    print("\n💡 Tip: Test on actual resume text from your dataset for better results")
    
    return entities, entities_by_type


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model-state.bin'
    vocab_path = sys.argv[2] if len(sys.argv) > 2 else 'vocab.txt'
    
    # Optional: provide custom test text
    custom_text = None
    if len(sys.argv) > 3:
        custom_text = sys.argv[3]
    
    try:
        test_model_improved(model_path, vocab_path, custom_text)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure model-state.bin and vocab.txt are in the current directory")
        print("Or provide paths: python test_model_improved.py <model_path> <vocab_path>")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
