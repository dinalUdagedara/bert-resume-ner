# Fix: Training Appears Stuck - Add Progress Indicators

## Problem
Training shows "Starting training loop" but no progress updates, making it seem stuck.

## Solution: Add Progress Indicators

Add this code **before** the training cell to improve visibility:

```python
# Add progress tracking to training loop
from tqdm import tqdm

# Modified training function with progress bars
def train_and_val_model_with_progress(model, tokenizer, optimizer, epochs, idx2tag, tag2idx, max_grad_norm, device, train_dataloader, valid_dataloader):
    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)

    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*60}")

        # Training loop with progress bar
        print("Starting training loop...")
        model.train()
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        # Add tqdm progress bar for batches
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch}")
        
        for step, batch in enumerate(train_progress):
            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss, tr_logits = outputs[:2]

            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            preds_mask = ((b_input_ids != cls_tok) & (b_input_ids != pad_tok) & (b_input_ids != sep_tok))

            tr_logits = tr_logits.cpu().detach().numpy()
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask = preds_mask.cpu().detach().numpy()
            tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze()], axis=1)
            tr_batch_labels = tr_label_ids.to("cpu").numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)

            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy

            # Update progress bar with loss
            train_progress.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{tmp_tr_accuracy:.4f}'})

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            model.zero_grad()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps
        print(f"\n✅ Train loss: {tr_loss:.4f}")
        print(f"✅ Train accuracy: {tr_accuracy:.4f}")

        # Validation loop with progress bar
        print("\nStarting validation loop...")
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        pred_tags_sequences, valid_tags_sequences = [], []

        val_progress = tqdm(valid_dataloader, desc=f"Validation Epoch {epoch}")
        
        for batch in val_progress:
            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                tmp_eval_loss, logits = outputs[:2]

            batch_size = b_input_ids.size(0)
            logits = logits.cpu().detach().numpy()
            b_labels_np = b_labels.cpu().numpy()
            b_input_ids_np = b_input_ids.cpu().numpy()
            
            batch_preds = []
            batch_labels = []
            
            for i in range(batch_size):
                sample_mask = ((b_input_ids_np[i] != cls_tok) & (b_input_ids_np[i] != pad_tok) & (b_input_ids_np[i] != sep_tok))
                sample_preds = np.argmax(logits[i][sample_mask], axis=1)
                sample_labels = b_labels_np[i][sample_mask]
                
                sample_pred_tags = [idx2tag[pred] for pred in sample_preds]
                sample_valid_tags = [idx2tag[label] for label in sample_labels]
                
                pred_tags_sequences.append(sample_pred_tags)
                valid_tags_sequences.append(sample_valid_tags)
                
                batch_preds.extend(sample_preds)
                batch_labels.extend(sample_labels)
                predictions.extend(sample_preds)
                true_labels.extend(sample_labels)

            tmp_eval_accuracy = flat_accuracy(batch_labels, batch_preds)
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            val_progress.set_postfix({'loss': f'{tmp_eval_loss.mean().item():.4f}', 'acc': f'{tmp_eval_accuracy:.4f}'})

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        pred_tags = [idx2tag[i] for i in predictions]
        valid_tags = [idx2tag[i] for i in true_labels]
        cl_report = classification_report(valid_tags_sequences, pred_tags_sequences)
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps

        print(f"\n✅ Validation loss: {eval_loss:.4f}")
        print(f"✅ Validation Accuracy: {eval_accuracy:.4f}")
        print(f"\nClassification Report:\n{cl_report}")
```

## Quick Check: Is It Actually Working?

Add this cell **before training** to verify GPU usage:

```python
# Check GPU status
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ Warning: Using CPU - training will be very slow!")

# Check dataset size
print(f"\nTraining batches: {len(train_dl)}")
print(f"Validation batches: {len(val_dl)}")
print(f"Estimated time per epoch (GPU): ~10-15 minutes")
print(f"Estimated time per epoch (CPU): ~1-2 hours")
```

## Alternative: Quick Test with Small Subset

Test with a smaller dataset first:

```python
# Test with first 100 samples only
print("Testing with 100 samples...")
test_train_data = train_data[:100]
test_val_data = val_data[:20]

test_train_d = ResumeDataset(test_train_data, tokenizer, tag2idx, MAX_LEN)
test_val_d = ResumeDataset(test_val_data, tokenizer, tag2idx, MAX_LEN)

test_train_dl = DataLoader(test_train_d, batch_size=8, collate_fn=collate_fn)
test_val_dl = DataLoader(test_val_d, batch_size=4, collate_fn=collate_fn)

# Train for 1 epoch only
print("Training for 1 epoch (test)...")
train_and_val_model_with_progress(
    model, tokenizer, optimizer, 1, idx2tag, tag2idx, 
    MAX_GRAD_NORM, device, test_train_dl, test_val_dl
)
```

If this works, then use full dataset. If it's still stuck, there's a deeper issue.
