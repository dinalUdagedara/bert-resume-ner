# Virtual Environment Setup Guide

## Why Use a Virtual Environment?

✅ **Isolate dependencies** - Won't conflict with other projects  
✅ **Clean environment** - Start fresh for this project  
✅ **Easy cleanup** - Just delete the venv folder when done  
✅ **Best practice** - Standard for Python projects  

## Step-by-Step Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
```

This creates a `venv` folder in your project directory.

### 2. Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when activated.

### 3. Upgrade pip (recommended)

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Deactivate (when you're done)

```bash
deactivate
```

## Important Note About Python Version

⚠️ **Compatibility Warning:** This project uses older packages:
- `torch==1.5.1` (from 2020)
- `transformers==3.0.2` (from 2020)

These may not be compatible with Python 3.12.3. 

**Options if you encounter errors:**

1. **Use Python 3.8 or 3.9** (recommended for compatibility):
   ```bash
   # If you have pyenv or similar
   pyenv install 3.9.18
   pyenv local 3.9.18
   python3 -m venv venv
   ```

2. **Update package versions** (may require code changes):
   - Update to newer versions of torch, transformers, etc.
   - Test if the code still works

## Quick Reference

```bash
# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Run training (after activation)
python3 train.py

# Deactivate when done
deactivate
```
