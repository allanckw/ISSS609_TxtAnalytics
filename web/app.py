import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from tensorflow import keras
tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json['text']
    saved_model = AutoModelForSequenceClassification.from_pretrained('../electra_ys')
    unseen_encodings = tokenizer(text, truncation=True, padding=True,return_tensors="pt")
    saved_model.eval()
    with torch.no_grad():
        logits = saved_model(**unseen_encodings).logits
    probabilities = torch.sigmoid(logits)
    print("testtt")
    print(float(probabilities[0][0]))
    predicted_class_id = probabilities.argmax().item()
    output = saved_model.config.id2label[predicted_class_id]
    return jsonify({
        "results": output,
        "prob1":float(probabilities[0][0]),
     "prob2":float(probabilities[0][1]),
    }), 200

if __name__ == '__main__':
    app.run(port=3000, debug=True)