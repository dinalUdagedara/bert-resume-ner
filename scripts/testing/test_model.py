"""
Test script for the trained BERT NER model.
Tests the model on sample resume text and displays extracted entities.
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import json


def load_model(model_path='model-state.bin', vocab_path='vocab.txt'):
    """Load the trained model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")
    
    # Load saved model state
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get label mappings from saved model
    tag2idx = checkpoint.get('tag2idx')
    idx2tag = checkpoint.get('idx2tag')
    model_name = checkpoint.get('model_name', 'bert-base-uncased')
    
    if tag2idx is None or idx2tag is None:
        raise ValueError("Model file doesn't contain tag mappings. Make sure you saved the model with tag2idx and idx2tag.")
    
    print(f"✅ Loaded {len(tag2idx)} labels")
    print(f"Labels: {list(tag2idx.keys())[:10]}...")  # Show first 10
    
    # Load tokenizer
    try:
        tokenizer = BertTokenizerFast(vocab_path, lowercase=True)
    except:
        # Fallback: use from_pretrained if vocab.txt not found
        print(f"⚠️ vocab.txt not found, using tokenizer from {model_name}")
        tokenizer = BertTokenizerFast.from_pretrained(model_name, lowercase=True)
    
    # Initialize model
    model = BertForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(tag2idx)
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, tokenizer, idx2tag, device


def extract_entities(model, tokenizer, idx2tag, device, text, max_len=500):
    """Extract entities from resume text."""
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        return_offsets_mapping=True,
        padding=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    offset_mapping = encoding['offset_mapping'][0].cpu().numpy()
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    
    # Extract entities
    entities = []
    current_entity = None
    
    # Get special tokens
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    
    for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
        # Skip special tokens
        token_id = input_ids[0][i].item()
        if token_id in [cls_token_id, sep_token_id, pad_token_id]:
            continue
        
        label = idx2tag[pred_id]
        start, end = offset
        
        # Skip 'O' and 'UNKNOWN' labels
        if label in ['O', 'UNKNOWN']:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        # Handle BIO format if present, or simple labels
        if label.startswith('B-') or label.startswith('I-'):
            # BIO format
            actual_label = label[2:]  # Remove B- or I-
            if label.startswith('B-') or (current_entity and current_entity['label'] != actual_label):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'label': actual_label,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                }
            else:
                # Continue current entity
                if current_entity:
                    current_entity['end'] = end
                    current_entity['text'] = text[current_entity['start']:end]
        else:
            # Simple label format
            if current_entity and current_entity['label'] == label and start - current_entity['end'] <= 1:
                # Continue entity if adjacent
                current_entity['end'] = end
                current_entity['text'] = text[current_entity['start']:end]
            else:
                # New entity
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'label': label,
                    'start': start,
                    'end': end,
                    'text': text[start:end]
                }
    
    # Add last entity
    if current_entity:
        entities.append(current_entity)
    
    # Clean up entities (remove duplicates, filter short ones)
    cleaned_entities = []
    seen = set()
    for ent in entities:
        # Skip very short entities (likely noise)
        if len(ent['text'].strip()) < 2:
            continue
        
        # Skip duplicates
        key = (ent['label'], ent['text'].strip().lower())
        if key not in seen:
            seen.add(key)
            cleaned_entities.append({
                'entity': ent['label'],
                'text': ent['text'].strip(),
                'start': ent['start'],
                'end': ent['end']
            })
    
    return cleaned_entities


def test_model(model_path='model-state.bin', vocab_path='vocab.txt'):
    """Test the model on sample resume text."""
    
    # Sample resume text for testing
    sample_resume = """
    John Smith
    Software Engineer
    Email: john.smith@email.com
    Location: San Francisco, California
    
    EXPERIENCE
    Senior Software Engineer at Google
    5 years of experience in Python, Machine Learning, and Cloud Computing
    
    EDUCATION
    Bachelor of Science in Computer Science
    Stanford University
    2015-2019
    
    SKILLS
    Python, Java, TensorFlow, PyTorch, AWS, Docker, Kubernetes
    """
    
    print("="*80)
    print("TESTING TRAINED MODEL")
    print("="*80)
    
    # Load model
    model, tokenizer, idx2tag, device = load_model(model_path, vocab_path)
    
    print("\n" + "-"*80)
    print("SAMPLE RESUME TEXT:")
    print("-"*80)
    print(sample_resume)
    
    # Extract entities
    print("\n" + "-"*80)
    print("EXTRACTED ENTITIES:")
    print("-"*80)
    
    entities = extract_entities(model, tokenizer, idx2tag, device, sample_resume)
    
    # Group by entity type
    entities_by_type = {}
    for ent in entities:
        label = ent['entity']
        if label not in entities_by_type:
            entities_by_type[label] = []
        entities_by_type[label].append(ent['text'])
    
    # Display results
    for label in sorted(entities_by_type.keys()):
        print(f"\n{label}:")
        for text in entities_by_type[label]:
            print(f"  - {text}")
    
    print(f"\n✅ Total entities extracted: {len(entities)}")
    print(f"✅ Entity types found: {len(entities_by_type)}")
    
    # Return as JSON format
    return {
        'entities': entities,
        'entities_by_type': {k: v for k, v in entities_by_type.items()}
    }


def test_on_custom_text(model_path, vocab_path, custom_text):
    """Test model on custom text."""
    model, tokenizer, idx2tag, device = load_model(model_path, vocab_path)
    entities = extract_entities(model, tokenizer, idx2tag, device, custom_text)
    
    print("\n" + "="*80)
    print("CUSTOM TEXT TEST")
    print("="*80)
    print(f"\nInput Text:\n{custom_text}\n")
    print("Extracted Entities:")
    print("-"*80)
    
    for ent in entities:
        print(f"{ent['entity']}: \"{ent['text']}\" (position: {ent['start']}-{ent['end']})")
    
    return entities


if __name__ == "__main__":
    import sys
    
    # Default paths (update if needed)
    model_path = 'model-state.bin'
    vocab_path = 'vocab.txt'
    
    # Check if paths provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        vocab_path = sys.argv[2]
    
    try:
        # Test on sample resume
        results = test_model(model_path, vocab_path)
        
        # Option to test on custom text
        print("\n" + "="*80)
        print("Want to test on your own text?")
        print("="*80)
        print("You can modify the script or call:")
        print("  test_on_custom_text(model_path, vocab_path, 'Your resume text here')")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  1. model-state.bin is in the current directory")
        print("  2. vocab.txt is in the current directory")
        print("  3. Or provide paths as arguments:")
        print("     python test_model.py /path/to/model-state.bin /path/to/vocab.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
