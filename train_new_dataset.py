"""
Updated training script for the new fixed dataset.
Uses the new dataset format and updated entity labels.
"""

import argparse
import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from utils import trim_entity_spans, get_hyperparameters, train_and_val_model, collate_fn
from load_new_dataset import load_new_dataset
from torch.utils.data import Dataset
import torch


# Updated tag system based on the new dataset labels
# Map new dataset labels to training tags
NEW_LABELS = [
    "SKILL", "DESIGNATION", "LOCATION", "EXPERIENCE", "PERSON", 
    "EDUCATION", "EXPERTISE", "EMAIL", "COMPANY", "COLLABORATION",
    "LANGUAGE", "ACTION", "CERTIFICATION", "OTHER"
]

# Create tag mappings
tags_vals = ["UNKNOWN", "O"] + NEW_LABELS
tag2idx = {t: i for i, t in enumerate(tags_vals)}
idx2tag = {i: t for i, t in enumerate(tags_vals)}


def process_resume(data, tokenizer, tag2idx, max_len, is_test=False):
    """Process resume data for training (updated from utils.py)."""
    tok = tokenizer.encode_plus(
        data[0], max_length=max_len, return_offsets_mapping=True, truncation=True)
    curr_sent = {'orig_labels': [], 'labels': []}

    padding_length = max_len - len(tok['input_ids'])

    if not is_test:
        labels = data[1]['entities']
        labels.reverse()
        for off in tok['offset_mapping']:
            label = get_label(off, labels)
            curr_sent['orig_labels'].append(label)
            curr_sent['labels'].append(tag2idx[label])
        curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)
    return curr_sent


def get_label(offset, labels):
    """Get label for a token offset."""
    if offset[0] == 0 and offset[1] == 0:
        return 'O'
    for label in labels:
        if offset[1] > label[0] and offset[0] < label[1]:
            return label[2]
    return 'O'


parser = argparse.ArgumentParser(description='Train Bert-NER on New Dataset')
parser.add_argument('-e', type=int, default=5, help='number of epochs')
parser.add_argument('-o', type=str, default='.',
                    help='output path to save model state')
parser.add_argument('-d', type=str, default='data/dataset-5000/train.json',
                    help='path to training dataset')
parser.add_argument('--train-split', type=float, default=0.9,
                    help='train/val split ratio (default: 0.9)')
parser.add_argument('--batch-size', type=int, default=8,
                    help='batch size for training')
parser.add_argument('--val-batch-size', type=int, default=4,
                    help='batch size for validation')

args = parser.parse_args().__dict__

output_path = args['o']
dataset_path = args['d']
train_split = args['train_split']
batch_size = args['batch_size']
val_batch_size = args['val_batch_size']

MAX_LEN = 500
EPOCHS = args['e']
MAX_GRAD_NORM = 1.0
MODEL_NAME = 'bert-base-uncased'
TOKENIZER = BertTokenizerFast('./vocab/vocab.txt', lowercase=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*80)
print("Loading Dataset")
print("="*80)
print(f"Dataset: {dataset_path}")
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Epochs: {EPOCHS}")
print(f"Train/Val Split: {train_split}")
print("="*80)

# Load new dataset format
data = load_new_dataset(dataset_path)
print(f"\nLoaded {len(data)} entries")

# Clean entity spans
data = trim_entity_spans(data)
print(f"After cleaning: {len(data)} entries")

# Split into train/val
total = len(data)
split_idx = int(total * train_split)
train_data, val_data = data[:split_idx], data[split_idx:]

print(f"\nTrain: {len(train_data)} entries")
print(f"Validation: {len(val_data)} entries")

# Create custom dataset class for new labels
class ResumeDatasetNew(Dataset):
    def __init__(self, resume, tokenizer, tag2idx, max_len, is_test=False):
        self.resume = resume
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, idx):
        data = process_resume(
            self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)
        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long),
            'orig_label': data['orig_labels']
        }

# Create datasets
train_d = ResumeDatasetNew(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDatasetNew(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)
val_dl = DataLoader(val_d, batch_size=val_batch_size, collate_fn=collate_fn)

print(f"\nNumber of labels: {len(tag2idx)}")
print(f"Labels: {NEW_LABELS}")

# Initialize model
print("\n" + "="*80)
print("Initializing Model")
print("="*80)
model = BertForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(tag2idx))
model.to(DEVICE)

optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

# Train model
print("\n" + "="*80)
print("Starting Training")
print("="*80)
train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl
)

# Save model
print("\n" + "="*80)
print("Saving Model")
print("="*80)
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "tag2idx": tag2idx,
        "idx2tag": idx2tag,
        "model_name": MODEL_NAME
    },
    f'{output_path}/model-state.bin',
)
print(f"✅ Model saved to: {output_path}/model-state.bin")
