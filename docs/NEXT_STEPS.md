# Next Steps - Training Your Resume NER Model

## ✅ What We've Completed

1. **Dataset Validation**: All 579,575 annotations validated and correct
2. **Label Fixes**: Fixed 2,112 mislabeled annotations
3. **Dataset Cleanup**: Cleaned up intermediate files, kept only the fixed dataset
4. **Training Script**: Created updated training script for the new dataset

## 📋 Current Status

- **Fixed Dataset**: `data/dataset-5000/train.json` (5,960 entries, 579,575 annotations)
- **Validation Report**: `data/dataset-5000/train_validation_report.json`
- **Original Backup**: `data/dataset-5000/train_original_backup.json`

## 🚀 Next Steps

### Step 1: Verify Your Environment

```bash
# Check Python version (recommended: 3.8-3.10)
python3 --version

# Install/update dependencies
pip3 install -r requirements.txt
```

### Step 2: Test Dataset Loading

```bash
# Test that the dataset loads correctly
python3 load_new_dataset.py
```

Expected output:
```
Loaded 5960 entries

First entry:
  Text length: <number>
  Entities: <number>
  First entity: (<start>, <end>, '<label>')
```

### Step 3: Start Training

**Basic training (5 epochs, default settings):**
```bash
python3 train_new_dataset.py
```

**Custom training:**
```bash
# 10 epochs, custom output directory
python3 train_new_dataset.py -e 10 -o ./models

# Custom train/val split (default is 90/10)
python3 train_new_dataset.py -e 5 --train-split 0.85

# Adjust batch sizes if you have memory issues
python3 train_new_dataset.py -e 5 --batch-size 4 --val-batch-size 2
```

### Step 4: Monitor Training

**What to expect:**
- Training loss should decrease over epochs
- Training accuracy should increase
- Validation metrics will be printed after each epoch
- Model will be saved as `model-state.bin`

**Training time estimates:**
- **CPU**: ~4-8 hours for 5 epochs (5,364 training samples)
- **GPU**: ~30-60 minutes for 5 epochs

### Step 5: Evaluate Results

After training completes, check:
- ✅ `model-state.bin` file exists (should be ~400-500 MB)
- ✅ Training loss decreased
- ✅ Validation accuracy improved
- ✅ Classification report shows good metrics

## 📊 Dataset Statistics

- **Total Entries**: 5,960
- **Training Set**: ~5,364 (90% split)
- **Validation Set**: ~596 (10% split)
- **Entity Labels**: 14 categories
  - SKILL, DESIGNATION, LOCATION, EXPERIENCE, PERSON
  - EDUCATION, EXPERTISE, EMAIL, COMPANY, COLLABORATION
  - LANGUAGE, ACTION, CERTIFICATION, OTHER

## 🔧 Troubleshooting

### Memory Issues
```bash
# Reduce batch size
python3 train_new_dataset.py --batch-size 4 --val-batch-size 2

# Or reduce MAX_LEN in train_new_dataset.py (line 45)
MAX_LEN = 300  # instead of 500
```

### Slow Training
- Use GPU if available (automatically detected)
- Reduce number of epochs for testing: `-e 2`
- Use smaller train split for faster iteration: `--train-split 0.8`

### Import Errors
```bash
# Reinstall dependencies
pip3 install --upgrade -r requirements.txt
```

## 📝 Notes

- The new training script uses updated entity labels that match your fixed dataset
- Model will be saved with tag mappings included
- You can resume training or fine-tune later using the saved model state

## 🎯 After Training

Once training is complete, you can:
1. Test the model using `app.py` (may need updates for new labels)
2. Evaluate on test data
3. Deploy the model for inference
4. Fine-tune further if needed

Good luck with your training! 🚀
