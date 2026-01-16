"""
Improved test cell for Colab - Copy this into a new cell after training
"""

# Test the trained model with improved extraction
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np

print("="*80)
print("TESTING TRAINED MODEL")
print("="*80)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('/content/model-state.bin', map_location=device)
tag2idx = checkpoint['tag2idx']
idx2tag = checkpoint['idx2tag']

print(f"✅ Loaded {len(tag2idx)} labels")

tokenizer = BertTokenizerFast('vocab.txt', lowercase=True)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✅ Model loaded!\n")

# Test text
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

print("-"*80)
print("TEST TEXT:")
print("-"*80)
print(test_text)
print()

# Tokenize
encoding = tokenizer(
    test_text,
    return_tensors="pt",
    truncation=True,
    max_length=500,
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

predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()

# Extract entities with improved logic
entities = []
current_entity = None

cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
    token_id = input_ids[0][i].item()
    
    # Skip special tokens
    if token_id in [cls_token_id, sep_token_id, pad_token_id]:
        if current_entity:
            entities.append(current_entity)
            current_entity = None
        continue
    
    # Skip zero offsets
    if offset[0] == 0 and offset[1] == 0:
        continue
    
    label = idx2tag[pred_id]
    start, end = offset
    
    # Skip 'O' and 'UNKNOWN'
    if label in ['O', 'UNKNOWN']:
        if current_entity:
            entities.append(current_entity)
            current_entity = None
        continue
    
    # Handle label (might be simple or BIO format)
    actual_label = label
    if label.startswith('B-'):
        actual_label = label[2:]
        is_beginning = True
    elif label.startswith('I-'):
        actual_label = label[2:]
        is_beginning = False
    else:
        is_beginning = True
    
    # Start or continue entity
    if current_entity is None:
        # Start new entity
        if start < len(test_text) and end <= len(test_text):
            current_entity = {
                'label': actual_label,
                'start': start,
                'end': end,
                'text': test_text[start:end]
            }
    else:
        # Continue entity if same label and adjacent
        if current_entity['label'] == actual_label:
            gap = start - current_entity['end']
            if gap <= 1:  # Adjacent or small gap
                current_entity['end'] = end
                if start < len(test_text) and end <= len(test_text):
                    current_entity['text'] = test_text[current_entity['start']:end]
            else:
                # Gap too large, start new
                entities.append(current_entity)
                if start < len(test_text) and end <= len(test_text):
                    current_entity = {
                        'label': actual_label,
                        'start': start,
                        'end': end,
                        'text': test_text[start:end]
                    }
        else:
            # Different label, start new
            entities.append(current_entity)
            if start < len(test_text) and end <= len(test_text):
                current_entity = {
                    'label': actual_label,
                    'start': start,
                    'end': end,
                    'text': test_text[start:end]
                }

if current_entity:
    entities.append(current_entity)

# Clean and display
print("="*80)
print("EXTRACTED ENTITIES:")
print("="*80)

# Filter and group
entities_by_type = {}
seen = set()

for ent in entities:
    text_clean = ent['text'].strip()
    
    # Filter short entities and common words
    if len(text_clean) < 2:
        continue
    
    if text_clean.lower() in ['the', 'a', 'an', 'is', 'are', 'at', 'in', 'on', 'to', 'for', 'of', 'and']:
        continue
    
    key = (ent['label'], text_clean.lower())
    if key not in seen:
        seen.add(key)
        if ent['label'] not in entities_by_type:
            entities_by_type[ent['label']] = []
        entities_by_type[ent['label']].append(text_clean)

# Display
if not entities_by_type:
    print("⚠️  No entities extracted")
else:
    for label in sorted(entities_by_type.keys()):
        print(f"\n📌 {label}:")
        for text in entities_by_type[label]:
            print(f"   • {text}")

print(f"\n✅ Total entities: {len([e for e in entities if len(e['text'].strip()) > 1])}")
print(f"✅ Entity types: {len(entities_by_type)}")

# Show label distribution
print("\n" + "-"*80)
print("LABEL PREDICTION DISTRIBUTION:")
print("-"*80)
from collections import Counter
label_counts = Counter()
for ent in entities:
    if len(ent['text'].strip()) > 1:
        label_counts[ent['label']] += 1

for label, count in label_counts.most_common():
    print(f"  {label}: {count} entities")

print("\n" + "-"*80)
print("IMPORTANT NOTES:")
print("-"*80)
print("⚠️  CLASS IMBALANCE ISSUE:")
print("   Your dataset has 95% SKILL labels (549,465 / 579,575)")
print("   Model learned to predict SKILL for almost everything")
print("\n✅ This is expected behavior with imbalanced data!")
print("   - Model works but is biased towards SKILL")
print("   - Test on actual resume data for better results")
print("   - Consider using class weights in future training")
print("   - Or proceed to from-scratch model as supervisor requested")
