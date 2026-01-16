# What to Do After Training Completes

## Step 1: Review Training Results

After training finishes, you'll see:
- ✅ **Training metrics**: Loss and accuracy for each epoch
- ✅ **Validation metrics**: How well the model generalizes
- ✅ **Classification Report**: Precision, recall, F1-score for each entity type
- ✅ **Confusion Matrix**: Shows which entities are confused with others

**What to look for:**
- Training loss should decrease over epochs
- Validation accuracy should improve
- F1-scores > 0.7 for major entities (SKILL, PERSON, EDUCATION) are good
- If validation loss increases while training loss decreases → overfitting

## Step 2: Save the Model

The notebook Step 7 will save your model. The saved file contains:
- Model weights (`model_state_dict`)
- Label mappings (`tag2idx`, `idx2tag`)
- Model name

**File size**: ~400-500 MB (BERT model size)

## Step 3: Download the Model

### Option A: Download to Your Computer
```python
from google.colab import files
files.download('/content/model-state.bin')
```

### Option B: Save to Google Drive (Recommended)
```python
# Save to Drive (persists after Colab session)
drive_model_path = '/content/drive/MyDrive/Resume-NER/models/model-state.bin'
torch.save({
    "model_state_dict": model.state_dict(),
    "tag2idx": tag2idx,
    "idx2tag": idx2tag,
    "model_name": MODEL_NAME
}, drive_model_path)
print(f"✅ Model saved to Drive: {drive_model_path}")
```

## Step 4: Update Your Flask API

Your `app.py` needs updates to use the new model:

### Changes Needed:

1. **Update NUM_LABELS**: 12 → 16 (14 entity types + UNKNOWN + O)

2. **Load tag mappings from saved model**:
```python
STATE_DICT = torch.load("model-state.bin", map_location=DEVICE)
tag2idx = STATE_DICT['tag2idx']
idx2tag = STATE_DICT['idx2tag']
NUM_LABELS = len(tag2idx)
```

3. **Update model initialization**:
```python
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=NUM_LABELS)
model.load_state_dict(STATE_DICT['model_state_dict'])
```

4. **Update server/utils.py** to use new label system

## Step 5: Test Your Model

### Quick Test in Colab:
```python
# Test on a sample resume text
test_text = "John Doe is a Software Engineer at Google with 5 years of experience in Python and Machine Learning. He has a B.Tech from MIT."

# Tokenize and predict
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=500)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Convert to labels
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = [idx2tag[pred.item()] for pred in predictions[0]]

# Display results
for token, label in zip(tokens, labels):
    if label != 'O':
        print(f"{token}: {label}")
```

### Test with Flask API:
```bash
# Start the API
python app.py

# Test with a resume PDF
curl --location --request POST 'http://localhost:5001/predict' \
  --form 'resume=@/path/to/resume.pdf'
```

## Step 6: Evaluate Model Performance

### Check Entity-Specific Performance:
- **SKILL**: Should have high recall (finds most skills)
- **PERSON**: Should be very accurate (names are usually clear)
- **EDUCATION**: Should identify degrees and institutions
- **LOCATION**: Should find cities and states

### Common Issues:
- **Low SKILL recall**: Model might miss some technical terms
- **PERSON false positives**: Might label company names as persons
- **EDUCATION confusion**: Might mix up degrees and designations

## Step 7: Fine-Tuning (Optional)

If results aren't good enough:

1. **Train for more epochs**: Increase `EPOCHS = 10`
2. **Adjust learning rate**: Try `lr=2e-5` or `lr=5e-5`
3. **Increase batch size**: If you have GPU memory
4. **Add more data**: Collect more resume samples

## Step 8: Deploy Your Model

### Local Deployment:
- Use Flask API (`app.py`)
- Run on your local machine
- Access via `http://localhost:5001`

### Cloud Deployment Options:
- **Heroku**: Free tier available
- **AWS EC2**: More control, pay-per-use
- **Google Cloud Run**: Serverless, auto-scaling
- **Docker**: Containerize for easy deployment

## Step 9: Monitor and Improve

- **Log predictions**: Track what the model predicts
- **Collect errors**: Note cases where model fails
- **Retrain periodically**: Add new data and retrain
- **A/B testing**: Compare model versions

## Quick Checklist

- [ ] Training completed successfully
- [ ] Model saved (`model-state.bin`)
- [ ] Model downloaded to computer or Drive
- [ ] `app.py` updated with new labels
- [ ] Model tested on sample resumes
- [ ] Flask API working
- [ ] Performance metrics reviewed
- [ ] Ready for deployment

## Expected Model Performance

With your dataset (5,960 samples, 14 entity types):
- **Overall Accuracy**: 85-90% is good
- **SKILL F1-score**: 0.80-0.90 (most common entity)
- **PERSON F1-score**: 0.90-0.95 (usually very accurate)
- **EDUCATION F1-score**: 0.75-0.85
- **Other entities**: 0.70-0.85 depending on frequency

## Troubleshooting

**Model not loading?**
- Check file path is correct
- Verify `tag2idx` and `idx2tag` are loaded
- Ensure `NUM_LABELS` matches saved model

**Poor predictions?**
- Check if input text is preprocessed correctly
- Verify tokenizer matches training tokenizer
- Ensure model is in `eval()` mode

**API errors?**
- Check model is loaded before API starts
- Verify device (CPU/GPU) matches training
- Check input format matches expected format

Good luck with your deployment! 🚀
