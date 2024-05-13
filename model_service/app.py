from flask import Flask, request, jsonify

import pickle
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from flasgger import Swagger
#import lib_ml_remla as libml


app = Flask(__name__)
swagger = Swagger(app)


# @app.route('/')
# def test():
#     return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the label and probability for a given URL.

    ---
    parameters:
      - in: body
        name: request_body
        description: JSON object containing the URL
        required: true
        schema:
          type: object
          properties:
            url:
              type: string
              description: URL for prediction
    responses:
      200:
        description: Successful prediction
        content:
          application/json:
            schema:
              type: object
              properties:
                result:
                  type: string
                  description: Predicted label
                probability:
                  type: array
                  items:
                    type: number
                  description: Predicted probabilities for each class
                url:
                  type: string
                  description: Input URL
      400:
        description: Bad request, invalid input format
    """
    req = request.get_json()
    data = req.get('url')
    
    #TODO: Combine preprocessing and postprocessing and move function to lib-ml
    processed = preprocess(np.array([data]), model=model, tokenizer=tokenizer, encoder=encoder)

    prediction = model.predict(processed)[0]
    
    y_pred_binary = (np.array(prediction) > 0.5).astype(int)
    
    label = encoder.inverse_transform([y_pred_binary])

    res = {
        "result": label.tolist(),
        "probability": prediction.tolist(),
        "url": data 
    }
    print(res)
    return jsonify(res)

#TODO: either download from public link or mount files
def load() -> tuple[Model, Tokenizer, LabelEncoder]:
    model = tf.keras.models.load_model('trained_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder

#TODO: Move this to lib-ml
def preprocess(raw_X_test: np.ndarray, model: Model, tokenizer: Tokenizer, encoder: LabelEncoder):
    
    X_test = pad_sequences(tokenizer.texts_to_sequences(raw_X_test), maxlen=200)
    
    return X_test

if __name__ == '__main__':
    model, tokenizer, encoder = load()
    app.run(debug=True, host='0.0.0.0', port=8080)
    
