import io
import argparse
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
from flask import Flask, jsonify, request
from server.utils import preprocess_data, predict, idx2tag

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

MAX_LEN = 500
NUM_LABELS = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'bert-base-uncased'
STATE_DICT = torch.load("model-state.bin", map_location=DEVICE)
TOKENIZER = BertTokenizerFast("./vocab/vocab.txt", lowercase=True)

# Load model first, then load the trained state dict
model = BertForTokenClassification.from_pretrained(
    'bert-base-uncased', num_labels=NUM_LABELS)
model.load_state_dict(STATE_DICT['model_state_dict'])
model.to(DEVICE)
model.eval()  # Set to evaluation mode


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Resume NER API is running'})

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        if 'resume' not in request.files:
            return jsonify({'error': 'No file provided. Please send a file with key "resume"'}), 400
        
        file = request.files.get('resume')
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        data = io.BytesIO(file.read())
        resume_text = preprocess_data(data)
        entities = predict(model, TOKENIZER, idx2tag,
                           DEVICE, resume_text, MAX_LEN)
        return jsonify({'entities': entities})
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e), 
            'type': type(e).__name__,
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    app.run(port=5001)
