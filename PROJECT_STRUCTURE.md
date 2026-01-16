# Project Structure

This document explains the organization of the Resume NER project.

## 📁 Directory Structure

```
Resume-NER/
├── README.md                    # Main project README
├── LICENSE                      # License file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── api/                        # API and server code
│   ├── app.py                  # Flask REST API
│   └── server/
│       └── utils.py            # API utility functions
│
├── data/                       # Datasets and data files
│   ├── dataset-5000/           # Main training dataset
│   │   ├── train.json         # Training data
│   │   ├── sample.json        # Sample data
│   │   └── ...
│   └── Resumes.json            # Additional resume data
│
├── demo/                       # Demo files and examples
│   ├── response.json           # Example API response
│   └── Resume - Ayush Srivastava.pdf
│
├── docs/                       # Documentation files
│   ├── TRAINING_GUIDE.md       # How to train models
│   ├── TESTING_GUIDE.md        # How to test models
│   ├── FROM_SCRATCH_GUIDE.md   # From-scratch training guide
│   ├── COLAB_SETUP.md          # Google Colab setup
│   ├── AFTER_TRAINING.md       # Post-training steps
│   └── ...                     # Other documentation
│
├── notebooks/                  # Jupyter notebooks
│   ├── Resume_NER_Training_Colab.ipynb      # BERT fine-tuning notebook
│   └── Resume_NER_From_Scratch_Colab.ipynb # From-scratch training notebook
│
├── scripts/                    # Python scripts
│   ├── training/               # Training scripts
│   │   ├── train.py            # Original training script
│   │   ├── train_new_dataset.py # Training with new dataset
│   │   ├── train_from_scratch.py # From-scratch training
│   │   └── utils.py            # Training utilities
│   │
│   ├── validation/             # Data validation scripts
│   │   ├── validate_annotations.py    # Validate annotation positions
│   │   ├── check_label_correctness.py # Check label semantics
│   │   ├── fix_label_issues.py        # Fix label errors
│   │   ├── analyze_person_labels.py    # Analyze PERSON labels
│   │   └── load_new_dataset.py        # Load dataset converter
│   │
│   └── testing/                # Testing scripts
│       ├── test_model.py              # Basic model testing
│       ├── test_model_improved.py     # Improved testing
│       ├── test_model_colab.py        # Colab testing
│       ├── test_cell_colab.py         # Colab test cell
│       ├── test_with_diagnostics.py   # Diagnostic testing
│       ├── test_diagnostics_colab_cell.py # Diagnostic Colab cell
│       └── test_api.py                # API testing
│
└── vocab/                      # Vocabulary files
    └── vocab.txt               # BERT vocabulary
```

## 📂 Folder Descriptions

### `api/`
Contains the Flask REST API for serving NER predictions.
- **app.py**: Main Flask application
- **server/utils.py**: Helper functions for API (preprocessing, prediction)

### `data/`
All datasets and data files.
- **dataset-5000/**: Main training dataset with 5,960 entries
- Contains train.json, validation reports, backups

### `demo/`
Example files and demo outputs.

### `docs/`
All documentation and guides.
- Training guides
- Setup instructions
- Testing guides
- Troubleshooting docs

### `notebooks/`
Jupyter notebooks for Google Colab.
- **Resume_NER_Training_Colab.ipynb**: BERT fine-tuning
- **Resume_NER_From_Scratch_Colab.ipynb**: From-scratch training

### `scripts/`
All Python scripts organized by purpose:

#### `scripts/training/`
Scripts for training models:
- **train.py**: Original training script
- **train_new_dataset.py**: Training with cleaned dataset
- **train_from_scratch.py**: BiLSTM-CRF from-scratch training
- **utils.py**: Training utilities and helper functions

#### `scripts/validation/`
Scripts for data validation and cleaning:
- **validate_annotations.py**: Check annotation positions
- **check_label_correctness.py**: Validate label semantics
- **fix_label_issues.py**: Auto-fix label errors
- **load_new_dataset.py**: Dataset format converter

#### `scripts/testing/`
Scripts for testing trained models:
- Various test scripts for different scenarios
- Diagnostic tools for analyzing model predictions

### `vocab/`
BERT vocabulary file for tokenization.

## 🚀 Quick Start

### Training a Model
```bash
# BERT fine-tuning
cd scripts/training
python train_new_dataset.py

# From-scratch training
python train_from_scratch.py
```

### Running the API
```bash
cd api
python app.py
```

### Testing
```bash
cd scripts/testing
python test_model.py
```

## 📝 Notes

- **Import paths**: Some scripts may need path adjustments when run from different directories
- **Notebooks**: Designed for Google Colab - update paths for local use
- **Data paths**: Update dataset paths in scripts/notebooks to match your setup

## 🔄 Migration Notes

If you're updating from the old structure:
- Scripts moved to `scripts/` subdirectories
- Documentation moved to `docs/`
- Notebooks moved to `notebooks/`
- API files moved to `api/`

Update any custom scripts that import from these locations.
