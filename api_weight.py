# -*- coding: utf-8 -*-
import flask
import tensorflow as tf
import io
import json
import string
import time
import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request

#Load model

model = tf.keras.models.load_model('./model_3classes.h5')

#Load labels json file\
with open("./train_labels.json", "r") as read_file:
    label = json.load(read_file)

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img,0)  #np.reshape(img,[1, 224, 224, 3])
    return img

def predict_result(img):
  predictions = model.predict(img)
  pred_labels = np.argmax(predictions, axis = 1)  #o:nofeet, 1:feet, 2:others
  labels = dict((v,k) for k,v in label.items())
  for k in pred_labels:
    predictions = labels[k] #[0,0,1]: nofeet, [0,1,0]: feet, [1,0,0]:others
    return predictions


#Set the name on Flask objects
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    img_bytes = file.read()
    img = prepare_image(img_bytes)

    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')