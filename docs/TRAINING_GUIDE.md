# Training Guide - Resume NER

## Quick Start

### Step 1: Install Dependencies

```bash
pip3 install -r requirements.txt
```

**Note:** If you encounter compatibility issues with Python 3.12, you may need to:
- Use Python 3.8-3.10, or
- Update the package versions in requirements.txt

### Step 2: Run Training

**Basic training (default: 5 epochs):**
```bash
python3 train.py
```

**Custom number of epochs:**
```bash
python3 train.py -e 10
```

**Specify output directory:**
```bash
python3 train.py -e 5 -o ./models
```

### Step 3: Verify Training is Working

#### What to Look For:

1. **Initial Setup Output:**
   - Model loading messages
   - Data loading confirmation
   - Device detection (CPU/GPU)

2. **Training Progress (per epoch):**
   ```
   Starting training loop.
   Train loss: <decreasing number>
   Train accuracy: <increasing number between 0-1>
   ```

3. **Validation Progress (per epoch):**
   ```
   Starting validation loop.
   Validation loss: <number>
   Validation Accuracy: <number between 0-1>
   Classification Report:
   <detailed metrics>
   Confusion Matrix:
   <matrix data>
   ```

4. **Model File Created:**
   - After training completes, check for `model-state.bin` in the output directory
   - File size should be ~400-500 MB (BERT model size)

### Step 4: Check Training Results

**Successful training indicators:**
- ✅ Training loss decreases over epochs
- ✅ Training accuracy increases over epochs
- ✅ Validation loss decreases (or stabilizes)
- ✅ `model-state.bin` file is created
- ✅ Classification report shows non-zero precision/recall for entity classes

**Warning signs:**
- ❌ Loss is NaN or extremely high
- ❌ Accuracy stays at 0 or doesn't improve
- ❌ All predictions are the same class
- ❌ No model file created

### Expected Training Time

- **CPU:** ~2-4 hours for 5 epochs (depending on hardware)
- **GPU:** ~15-30 minutes for 5 epochs

### Troubleshooting

1. **Memory Error:**
   - Reduce batch size in `train.py` (line 35: `batch_size=8`)
   - Reduce MAX_LEN (line 19: `MAX_LEN = 500`)

2. **Import Errors:**
   - Make sure all dependencies are installed
   - Check Python version compatibility

3. **Data Loading Issues:**
   - Verify `data/Resumes.json` exists and is valid JSON
   - Check file permissions

4. **Model Download Issues:**
   - First run will download BERT model (~500MB) - requires internet
   - Check network connection
