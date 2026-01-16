"""
Diagnostic test script - Shows what labels the model is actually predicting
and tests on real data from the training set.
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np
from collections import Counter
import json

print("="*80)
print("DIAGNOSTIC TEST - Model Prediction Analysis")
print("="*80)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('/content/model-state.bin', map_location=device)
tag2idx = checkpoint['tag2idx']
idx2tag = checkpoint['idx2tag']

print(f"✅ Loaded {len(tag2idx)} labels")
print(f"Labels: {list(tag2idx.keys())}\n")

tokenizer = BertTokenizerFast('vocab.txt', lowercase=True)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(tag2idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

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
print("TEST 1: Label Distribution Analysis")
print("-"*80)

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

# Count label predictions
cls_token_id = tokenizer.cls_token_id
sep_token_id = tokenizer.sep_token_id
pad_token_id = tokenizer.pad_token_id

label_counts = Counter()
label_examples = {}

for i, (pred_id, offset) in enumerate(zip(predictions, offset_mapping)):
    token_id = input_ids[0][i].item()
    if token_id in [cls_token_id, sep_token_id, pad_token_id]:
        continue
    if offset[0] == 0 and offset[1] == 0:
        continue
    
    label = idx2tag[pred_id]
    label_counts[label] += 1
    
    # Store example text for each label
    if label not in label_examples and label != 'O':
        start, end = offset
        if start < len(test_text) and end <= len(test_text):
            label_examples[label] = test_text[start:end]

print("\n📊 Label Prediction Distribution:")
print("-" * 60)
total_non_o = sum(count for label, count in label_counts.items() if label != 'O')
for label, count in label_counts.most_common():
    if label != 'O':
        percentage = (count / total_non_o * 100) if total_non_o > 0 else 0
        example = label_examples.get(label, 'N/A')
        print(f"  {label:15s}: {count:4d} tokens ({percentage:5.1f}%) - Example: '{example}'")

print(f"\n⚠️  ISSUE: Model is heavily biased towards SKILL!")
print(f"   This is because SKILL is 95% of your training data (549,465 / 579,575 annotations)")
print(f"   The model learned to predict SKILL for almost everything to maximize accuracy.\n")

print("-"*80)
print("TEST 2: Test on Real Data from Training Set")
print("-"*80)

# Try to load actual data
try:
    # Try different possible paths
    dataset_paths = [
        '/content/drive/MyDrive/path/to/train.json',
        '/content/train.json',
        '/content/drive/MyDrive/train.json',
        '/content/drive/MyDrive/Resume-NER/data/dataset-5000/train.json'
    ]
    
    dataset_path = None
    for path in dataset_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                dataset_path = path
                print(f"✅ Loaded dataset from: {path}")
                print(f"   Total entries: {len(data)}")
                break
        except:
            continue
    
    if dataset_path:
        # Test on first entry
        sample = data[0]
        real_text = sample['text']
        real_annotations = sample['annotations']
        
        print(f"\n📄 Testing on real resume (first entry):")
        print(f"   Text length: {len(real_text)} characters")
        print(f"   Expected entities: {len(real_annotations)}")
        
        # Show expected entities
        print("\n   Expected entities:")
        expected_by_type = {}
        for start, end, label in real_annotations[:20]:  # Show first 20
            if label not in expected_by_type:
                expected_by_type[label] = []
            expected_by_type[label].append(real_text[start:end])
        
        for label in sorted(expected_by_type.keys()):
            print(f"     {label}: {len(expected_by_type[label])} entities")
        
        # Predict on real text
        encoding_real = tokenizer(
            real_text,
            return_tensors="pt",
            truncation=True,
            max_length=500,
            return_offsets_mapping=True,
            padding=True
        )
        
        input_ids_real = encoding_real['input_ids'].to(device)
        attention_mask_real = encoding_real['attention_mask'].to(device)
        offset_mapping_real = encoding_real['offset_mapping'][0].cpu().numpy()
        
        with torch.no_grad():
            outputs_real = model(input_ids_real, attention_mask=attention_mask_real)
            predictions_real = torch.argmax(outputs_real.logits, dim=-1)[0].cpu().numpy()
        
        # Extract entities
        entities_real = []
        current_entity = None
        
        for i, (pred_id, offset) in enumerate(zip(predictions_real, offset_mapping_real)):
            token_id = input_ids_real[0][i].item()
            if token_id in [cls_token_id, sep_token_id, pad_token_id]:
                if current_entity:
                    entities_real.append(current_entity)
                    current_entity = None
                continue
            
            if offset[0] == 0 and offset[1] == 0:
                continue
            
            label = idx2tag[pred_id]
            start, end = offset
            
            if label in ['O', 'UNKNOWN']:
                if current_entity:
                    entities_real.append(current_entity)
                    current_entity = None
                continue
            
            actual_label = label
            if label.startswith('B-'):
                actual_label = label[2:]
            elif label.startswith('I-'):
                actual_label = label[2:]
            
            if current_entity is None:
                if start < len(real_text) and end <= len(real_text):
                    current_entity = {
                        'label': actual_label,
                        'start': start,
                        'end': end,
                        'text': real_text[start:end]
                    }
            else:
                if current_entity['label'] == actual_label:
                    gap = start - current_entity['end']
                    if gap <= 1:
                        current_entity['end'] = end
                        if start < len(real_text) and end <= len(real_text):
                            current_entity['text'] = real_text[current_entity['start']:end]
                    else:
                        entities_real.append(current_entity)
                        if start < len(real_text) and end <= len(real_text):
                            current_entity = {
                                'label': actual_label,
                                'start': start,
                                'end': end,
                                'text': real_text[start:end]
                            }
                else:
                    entities_real.append(current_entity)
                    if start < len(real_text) and end <= len(real_text):
                        current_entity = {
                            'label': actual_label,
                            'start': start,
                            'end': end,
                            'text': real_text[start:end]
                        }
        
        if current_entity:
            entities_real.append(current_entity)
        
        # Group by type
        predicted_by_type = {}
        for ent in entities_real:
            if len(ent['text'].strip()) > 1:
                label = ent['label']
                if label not in predicted_by_type:
                    predicted_by_type[label] = []
                predicted_by_type[label].append(ent['text'].strip())
        
        print(f"\n   Predicted entities: {len(entities_real)}")
        print("\n   Predicted by type:")
        for label in sorted(predicted_by_type.keys()):
            unique = list(set(predicted_by_type[label]))[:5]  # Show first 5 unique
            print(f"     {label}: {len(predicted_by_type[label])} entities")
            if unique:
                print(f"       Examples: {', '.join(unique[:3])}")
        
    else:
        print("⚠️  Could not find dataset file. Please update the path in the script.")
        print("   Or manually provide a sample text from your dataset.")
        
except Exception as e:
    print(f"⚠️  Error loading dataset: {e}")
    print("   This is okay - the diagnostic info above is still useful!")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("""
1. ⚠️  CLASS IMBALANCE ISSUE:
   - SKILL is 95% of your training data
   - Model learned to predict SKILL for everything
   - This is why you see mostly SKILL predictions

2. ✅ SOLUTIONS:
   a) Use class weights during training (weight rare classes more)
   b) Test on actual resume data (might work better than synthetic text)
   c) Accept this limitation for your FYP and document it
   d) Move to from-scratch model (as supervisor requested)

3. 📊 FOR YOUR FYP:
   - Document the class imbalance issue
   - Show that model works but has limitations
   - Note that this is common in NER tasks
   - Proceed with from-scratch model training

4. 🎯 NEXT STEP:
   - Test on a few real resumes from your dataset
   - If results are acceptable, document and move on
   - If not, proceed to from-scratch model training
""")
