"""
Test script for Colab - Test your trained BERT model.
Run this in Colab after training completes.
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Configuration
MODEL_PATH = '/content/model-state.bin'  # Update if saved elsewhere
VOCAB_PATH = 'vocab.txt'
MAX_LEN = 500

# Load model
print("="*60)
print("Loading Trained Model")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
tag2idx = checkpoint['tag2idx']
idx2tag = checkpoint['idx2tag']

print(f"✅ Loaded {len(tag2idx)} labels")
print(f"Sample labels: {list(tag2idx.keys())[:10]}")

# Load tokenizer
tokenizer = BertTokenizerFast(VOCAB_PATH, lowercase=True)

# Initialize and load model
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=len(tag2idx)
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✅ Model loaded and ready for testing!")

# Sample resume for testing
sample_resume = """
Abhishek Jha
Application Development Associate at Accenture
Bengaluru, Karnataka
Email: abhishek.jha@email.com

WORK EXPERIENCE
Application Development Associate
Accenture - November 2017 to Present
Working on Chat-bot development using Oracle PeopleSoft.

EDUCATION
B.E in Information Science and Engineering
B.V.B College of Engineering and Technology
Hubli, Karnataka
August 2013 to June 2017

SKILLS
C, C++, Java, Oracle PeopleSoft, Machine Learning, Database Management
"""

print("\n" + "="*60)
print("Testing on Sample Resume")
print("="*60)
print(f"\nResume Text:\n{sample_resume}\n")

# Tokenize
encoding = tokenizer(
    sample_resume,
    return_tensors="pt",
    truncation=True,
    max_length=MAX_LEN,
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

# Extract entities
entities = []
current_entity = None

cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
    token_id = input_ids[0][i].item()
    if token_id in [cls_token_id, sep_token_id, pad_token_id]:
        continue
    
    label = idx2tag[pred_id]
    start, end = offset
    
    if label in ['O', 'UNKNOWN']:
        if current_entity:
            entities.append(current_entity)
            current_entity = None
        continue
    
    # Handle label format
    if label.startswith('B-') or label.startswith('I-'):
        actual_label = label[2:]
        if label.startswith('B-') or (current_entity and current_entity['label'] != actual_label):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'label': actual_label,
                'start': start,
                'end': end,
                'text': sample_resume[start:end]
            }
        else:
            if current_entity:
                current_entity['end'] = end
                current_entity['text'] = sample_resume[current_entity['start']:end]
    else:
        if current_entity and current_entity['label'] == label and start - current_entity['end'] <= 1:
            current_entity['end'] = end
            current_entity['text'] = sample_resume[current_entity['start']:end]
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'label': label,
                'start': start,
                'end': end,
                'text': sample_resume[start:end]
            }

if current_entity:
    entities.append(current_entity)

# Clean and display
print("="*60)
print("EXTRACTED ENTITIES")
print("="*60)

entities_by_type = {}
for ent in entities:
    if len(ent['text'].strip()) < 2:
        continue
    label = ent['label']
    if label not in entities_by_type:
        entities_by_type[label] = []
    entities_by_type[label].append(ent['text'].strip())

for label in sorted(entities_by_type.keys()):
    print(f"\n{label}:")
    for text in set(entities_by_type[label]):  # Remove duplicates
        print(f"  • {text}")

print(f"\n✅ Total entities: {len(entities)}")
print(f"✅ Entity types: {len(entities_by_type)}")

# Test on your own text
print("\n" + "="*60)
print("Test on Your Own Text")
print("="*60)
print("To test on custom text, modify 'sample_resume' variable above")
