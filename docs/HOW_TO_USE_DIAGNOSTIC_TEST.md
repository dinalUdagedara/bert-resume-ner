# How to Use the Diagnostic Test

## Quick Start in Colab

### Option 1: Copy the entire file (Easiest)

1. **Open** `test_diagnostics_colab_cell.py` in your editor
2. **Copy the entire content** (everything inside the file)
3. **Paste into a new Colab cell** (after your training completes)
4. **Update the dataset path** if needed (line with `'/content/drive/MyDrive/path/to/train.json'`)
5. **Run the cell**

### Option 2: Use the original file

1. **Upload** `test_with_diagnostics.py` to Colab:
   ```python
   # In a Colab cell
   from google.colab import files
   files.upload()  # Select test_with_diagnostics.py
   ```

2. **Run it**:
   ```python
   exec(open('test_with_diagnostics.py').read())
   ```

## What the Test Shows

### Test 1: Label Distribution
- Shows which labels the model is predicting
- Shows percentage distribution
- Reveals the SKILL bias issue

### Test 2: Real Data Test
- Tests on actual resume from your dataset
- Compares expected vs predicted entities
- Shows how well model works on real data

## Expected Output

You'll see something like:

```
📊 Label Prediction Distribution:
  SKILL          :  150 tokens ( 95.0%) - Example: 'Python'
  PERSON         :    5 tokens (  3.0%) - Example: 'John'
  EDUCATION      :    2 tokens (  1.0%) - Example: 'Bachelor'
  ...

⚠️  ISSUE: Model is heavily biased towards SKILL!
   This is because SKILL is 95% of your training data
```

## Troubleshooting

### "Could not find dataset file"
- Update the `dataset_paths` list in the script
- Make sure Google Drive is mounted
- Check the exact path where you saved `train.json`

### "Model not found"
- Make sure training completed and model was saved
- Check that `/content/model-state.bin` exists

### "vocab.txt not found"
- Make sure you downloaded vocab.txt in Step 4 of training
- Or it will use the tokenizer from HuggingFace (slower)

## What to Do With Results

1. **Document the findings**:
   - Class imbalance issue
   - Label distribution
   - Model limitations

2. **For your FYP**:
   - Show that model works but has limitations
   - Note this is expected with imbalanced data
   - Proceed to from-scratch model as requested

3. **Next steps**:
   - Test on a few more real resumes
   - Document results
   - Move to from-scratch training

## Quick Test Without Dataset

If you just want to see label distribution (without testing on real data), you can use the simpler `test_cell_colab.py` instead.
