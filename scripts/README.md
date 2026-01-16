# Scripts Directory

This directory contains all Python scripts organized by purpose.

## 📁 Structure

```
scripts/
├── training/      # Model training scripts
├── validation/    # Data validation and cleaning
└── testing/       # Model testing and evaluation
```

## 🎯 Usage

### Training Scripts
Located in `scripts/training/`:
- **train.py**: Original BERT training
- **train_new_dataset.py**: Training with cleaned dataset
- **train_from_scratch.py**: From-scratch BiLSTM-CRF training

### Validation Scripts
Located in `scripts/validation/`:
- **validate_annotations.py**: Validate annotation positions
- **check_label_correctness.py**: Check label semantics
- **fix_label_issues.py**: Auto-fix label errors
- **load_new_dataset.py**: Dataset format converter

### Testing Scripts
Located in `scripts/testing/`:
- Various test scripts for model evaluation
- Diagnostic tools for prediction analysis

## 💡 Running Scripts

From project root:
```bash
# Training
python scripts/training/train_new_dataset.py

# Validation
python scripts/validation/validate_annotations.py data/dataset-5000/train.json

# Testing
python scripts/testing/test_model.py
```

Or from within scripts directory:
```bash
cd scripts/training
python train_new_dataset.py
```

## ⚠️ Import Notes

Some scripts use relative imports. Make sure to:
- Run from project root, OR
- Update sys.path in scripts if needed
