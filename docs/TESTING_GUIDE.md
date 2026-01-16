# Testing Your Trained Model - Complete Guide

## Quick Test Options

### Option 1: Test in Colab (Easiest) - RECOMMENDED

After training completes in Colab, add a new cell and copy the code from `test_cell_colab.py`:

1. **Create new cell** after the training cell
2. **Copy the entire content** from `test_cell_colab.py`
3. **Run the cell**

This will:
- Load your trained model
- Test on sample resume text
- Show extracted entities with improved extraction logic
- Filter out noise and common false positives

**Or** use the simpler version:
```python
# Test the trained model
exec(open('test_model_colab.py').read())
```

### Option 2: Test Locally

1. **Download the model from Colab:**
   ```python
   from google.colab import files
   files.download('/content/model-state.bin')
   ```

2. **Download vocab.txt:**
   ```python
   files.download('/content/vocab.txt')
   ```

3. **Run test script:**
   ```bash
   python test_model.py model-state.bin vocab.txt
   ```

### Option 3: Test via Flask API

Update `app.py` to use your new model (see below), then:
```bash
python app.py
# Test with curl
curl -X POST http://localhost:5001/predict \
  -F "resume=@path/to/resume.pdf"
```

---

## What the Test Shows

The test will extract:
- **PERSON**: Names
- **SKILL**: Technical skills
- **EDUCATION**: Degrees and institutions
- **DESIGNATION**: Job titles
- **COMPANY**: Company names
- **LOCATION**: Cities, states
- **EMAIL**: Email addresses
- And other entity types

---

## Expected Results

Based on your training:
- **SKILL**: Should work well (F1: 0.30)
- **PERSON**: Should work well (F1: 0.27)
- **EDUCATION**: Should work reasonably (F1: 0.25)
- **EMAIL**: May not work (F1: 0.00 in training)
- **LOCATION**: Should work moderately (F1: 0.09)

---

## Testing Different Resumes

### Test 1: Simple Resume
```python
text = "John Doe is a Software Engineer at Google with 5 years of Python experience."
```

### Test 2: Full Resume
```python
text = """
John Smith
Software Engineer
Email: john@email.com
Location: San Francisco

Experience: 5 years at Google
Education: B.Tech from MIT
Skills: Python, Machine Learning, AWS
"""
```

### Test 3: Your Own Resume
Paste your actual resume text and test!

---

## Troubleshooting

**Model not found?**
- Check path: `/content/model-state.bin` in Colab
- Or download and use local path

**Wrong labels?**
- Make sure you're using the model with correct `tag2idx` and `idx2tag`
- The saved model includes these mappings

**No entities extracted?**
- Check if text is too short
- Try longer resume text
- Some entities (EMAIL, ACTION) had 0.00 F1 in training

---

## Next Steps After Testing

1. ✅ **If results are good**: Use the model for your project
2. ✅ **If results need improvement**: 
   - Train more epochs
   - Adjust hyperparameters
   - Try from-scratch model
3. ✅ **For your FYP**: Document the testing process and results

Good luck with testing! 🚀
