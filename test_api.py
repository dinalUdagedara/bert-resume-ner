#!/usr/bin/env python3
"""
Simple test script for the Resume NER API
"""
import requests
import json

# Test the API endpoint
url = 'http://localhost:5001/predict'
file_path = 'demo/Resume - Ayush Srivastava.pdf'

try:
    print(f"Testing API with file: {file_path}")
    print("Sending request...")
    
    with open(file_path, 'rb') as f:
        files = {'resume': (file_path.split('/')[-1], f, 'application/pdf')}
        response = requests.post(url, files=files)
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Success!")
        data = response.json()
        print(f"\nFound {len(data.get('entities', []))} entities:")
        print("\n" + "="*60)
        
        # Group entities by type
        entities_by_type = {}
        for entity in data.get('entities', []):
            entity_type = entity.get('entity', 'Unknown')
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity.get('text', ''))
        
        # Print grouped entities
        for entity_type, texts in entities_by_type.items():
            print(f"\n{entity_type}:")
            for text in texts[:5]:  # Show first 5 of each type
                print(f"  - {text}")
            if len(texts) > 5:
                print(f"  ... and {len(texts) - 5} more")
        
        print("\n" + "="*60)
        print("\nFull JSON response:")
        print(json.dumps(data, indent=2))
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response text: {response.text}")
        try:
            error_data = response.json()
            print(f"Error JSON: {json.dumps(error_data, indent=2)}")
        except:
            pass
        
except FileNotFoundError:
    print(f"❌ Error: File not found: {file_path}")
except requests.exceptions.ConnectionError:
    print("❌ Error: Could not connect to server. Make sure Flask app is running on http://localhost:5000")
except Exception as e:
    print(f"❌ Error: {type(e).__name__}: {e}")
