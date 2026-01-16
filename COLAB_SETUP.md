# Training on Google Colab - Quick Guide

## Why Use Colab?

✅ **Free GPU access** - Training will be **10-20x faster** (30-60 min instead of 4-8 hours)  
✅ **No local resources needed** - Run everything in the cloud  
✅ **Easy to share** - Share notebook with others  
✅ **Automatic setup** - No local environment configuration needed  

## Setup Steps

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2: Upload the Notebookz
1. File → Upload notebook
2. Upload `Resume_NER_Training_Colab.ipynb`
3. Or create a new notebook and copy the cells

### Step 3: Enable GPU
1. Runtime → Change runtime type
2. Select **GPU** (T4 is free)
3. Click Save

### Step 4: Upload Your Dataset
**Option A: Direct Upload (Easiest)**
1. Click the folder icon (📁) on the left sidebar
2. Click the upload icon (📤)
3. Upload `data/dataset-5000/train.json`
4. Wait for upload to complete (~37 MB)

**Option B: From Google Drive**
1. Mount Google Drive in the notebook
2. Copy file from Drive to `/content/`

### Step 5: Run the Notebook
1. Run cells from top to bottom
2. Or click: Runtime → Run all

## Expected Training Time

- **With GPU (T4)**: ~30-60 minutes for 5 epochs
- **Without GPU (CPU)**: ~4-8 hours for 5 epochs

## Important Notes

⚠️ **Colab Session Limits**:
- Free tier: ~12 hours maximum per session
- If disconnected, training progress is lost
- Save model checkpoints periodically if training long epochs

💡 **Tips**:
- Keep the browser tab open during training
- Use fewer epochs (2-3) for testing first
- Model will be saved in `/content/model-state.bin`

## Downloading Your Model

After training completes, run the download cell:
```python
from google.colab import files
files.download('/content/model-state.bin')
```

This will download the model (~400-500 MB) to your computer.

## Troubleshooting

**GPU not available?**
- Free tier may be limited during high usage
- Try again later or use CPU (slower)

**Out of memory?**
- Reduce `BATCH_SIZE` to 4
- Reduce `MAX_LEN` to 300

**Disconnected during training?**
- Colab disconnects after inactivity
- Keep the tab active or upgrade to Pro

**File not found?**
- Check that `train.json` uploaded successfully
- Verify path: `/content/train.json`

## Next Steps After Training

1. Download the model (`model-state.bin`)
2. Use it with your Flask API (`app.py`)
3. Update `app.py` to use the new tag mappings if needed

Good luck with training! 🚀
