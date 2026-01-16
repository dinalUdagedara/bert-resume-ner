"""
Converter function to load the new dataset format (train.json) 
and convert it to the format expected by the training pipeline.
"""

import json


def load_new_dataset(json_file_path):
    """
    Load the new dataset format and convert to training format.
    
    New format:
    {
        "text": "...",
        "annotations": [[start, end, "LABEL"], ...]
    }
    
    Expected format (spaCy style):
    [
        (text, {"entities": [(start, end, "LABEL"), ...]}),
        ...
    ]
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = []
    
    for entry in data:
        if 'text' not in entry or 'annotations' not in entry:
            continue
        
        text = entry['text']
        annotations = entry['annotations']
        
        entities = []
        for ann in annotations:
            if not isinstance(ann, list) or len(ann) != 3:
                continue
            
            start, end, label = ann
            
            # Validate positions
            if start < 0 or end > len(text) or start >= end:
                continue
            
            # Convert to (start, end+1, label) format expected by training
            # Note: end+1 because the training expects exclusive end
            entities.append((start, end, label))
        
        if entities:  # Only add entries with valid entities
            training_data.append((text, {"entities": entities}))
    
    return training_data


if __name__ == "__main__":
    # Test the loader
    data = load_new_dataset('data/dataset-5000/train.json')
    print(f"Loaded {len(data)} entries")
    if data:
        print(f"\nFirst entry:")
        print(f"  Text length: {len(data[0][0])}")
        print(f"  Entities: {len(data[0][1]['entities'])}")
        print(f"  First entity: {data[0][1]['entities'][0]}")
