# Using Google Drive for Dataset in Colab

## Why Use Google Drive?

✅ **Persistent Storage** - Dataset stays between Colab sessions  
✅ **No Re-upload** - Upload once, use many times  
✅ **Large Files** - Better for files >100MB  
✅ **Shareable** - Share with team members easily  

## Setup Steps

### Step 1: Upload Dataset to Google Drive

1. Open [Google Drive](https://drive.google.com)
2. Create a folder (e.g., `Resume-NER-Dataset`)
3. Upload `data/dataset-5000/train.json` to that folder
4. **Note the full path** - Example: `MyDrive/Resume-NER-Dataset/train.json`

### Step 2: In Colab Notebook

#### Option A: Use Directly from Drive (Recommended)

1. In **Step 2** of the notebook, mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Update the path** to match your file location:
   ```python
   DRIVE_DATASET_PATH = '/content/drive/MyDrive/Resume-NER-Dataset/train.json'
   ```

3. In **Step 5** (dataset loading), update:
   ```python
   dataset_path = '/content/drive/MyDrive/Resume-NER-Dataset/train.json'
   ```

#### Option B: Copy to Local First (Faster Loading)

This copies the file to Colab's local storage for faster access:

1. Mount Drive (Step 2):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Copy to local (uncomment the copy line):
   ```python
   !cp "/content/drive/MyDrive/Resume-NER-Dataset/train.json" /content/train.json
   ```

3. Use local path (Step 5):
   ```python
   dataset_path = '/content/train.json'
   ```

## Finding Your File Path in Google Drive

After mounting Drive, you can list files:

```python
# List contents of MyDrive
!ls "/content/drive/MyDrive/"

# Or navigate to find your file
!ls "/content/drive/MyDrive/Resume-NER-Dataset/"
```

## Common Path Examples

```python
# If file is in root of MyDrive:
'/content/drive/MyDrive/train.json'

# If file is in a folder:
'/content/drive/MyDrive/Resume-NER-Dataset/train.json'

# If file is in a subfolder:
'/content/drive/MyDrive/Projects/FYP/Resume-NER/data/dataset-5000/train.json'
```

## Troubleshooting

**"File not found" error?**
- Check the path matches exactly (case-sensitive)
- Use quotes around paths with spaces: `"/content/drive/MyDrive/My Folder/train.json"`
- List directory to verify: `!ls "/content/drive/MyDrive/"`

**Permission denied?**
- Make sure you clicked "Allow" when mounting Drive
- Check file sharing settings in Google Drive

**Slow loading?**
- Copy to local storage first (Option B) for faster repeated access
- Local copy is faster but is lost when Colab session ends

## Saving Model to Drive (Bonus)

After training, save model to Drive for persistence:

```python
# Save to Drive
drive_model_path = '/content/drive/MyDrive/Resume-NER/models/model-state.bin'
torch.save({
    "model_state_dict": model.state_dict(),
    "tag2idx": tag2idx,
    "idx2tag": idx2tag,
    "model_name": MODEL_NAME
}, drive_model_path)
print(f"✅ Model saved to Drive: {drive_model_path}")
```

This way your model persists even after Colab session ends!
