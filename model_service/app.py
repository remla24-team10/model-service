from flask import Flask, request, jsonify

import numpy as np
import pickle
import os
from keras._tf_keras.keras import Model
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

app = Flask(__name__)


#TODO: swagger documentation
@app.route('/')
def test():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    data = req.get('url')
    
    #TODO: Combine preprocessing and postprocessing and move function to lib-ml
    processed = preprocess(np.array([data]), model=model, tokenizer=tokenizer, encoder=encoder)

    prediction = model.predict(processed)[0]
    
    label = encoder.transform(prediction)
    
    res = {
        "result": label,
        "probability": prediction.tolist(),
        "url": data 
    }
    print(res)
    return jsonify(res)




#Move this to lib-ml?
def load() -> tuple[Model, Tokenizer, LabelEncoder]:
    
    model = tf.keras.models.load_model('../trained_model.keras')
    with open('../tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('../encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    print(model.summary())

    return model, tokenizer, encoder

#Move this to lib-ml?
def preprocess(raw_X_test: np.ndarray, model: Model, tokenizer: Tokenizer, encoder: LabelEncoder):
    X_test = pad_sequences(tokenizer.texts_to_sequences(raw_X_test), maxlen=200)
    #char_index = tokenizer.word_index
    
    
    return X_test

if __name__ == '__main__':
    app.run(debug=True)
    
model, tokenizer, encoder = load()