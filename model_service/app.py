
import gdown
import pickle
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from flasgger import Swagger
from flask import Flask, request, jsonify
from lib_ml_remla.preprocess import prepare
from lib_ml_remla.utils import predict_classes



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
    
    processed = prepare(np.array([data]), tokenizer=tokenizer)

    labels, probabilities = predict_classes(model=model, encoder=encoder, X_test=processed, threshold=0.5)
    
    res = {
        "result": labels[0].tolist(),
        "probability": probabilities[0].tolist(),
        "url": data 
    }
    print(res)
    return jsonify(res)

#TODO: either download from public link or mount files
def load() -> tuple[Model, Tokenizer, LabelEncoder]:
    gdown.download(id="1Ob0dzKS5mu_t8zMvgiyGfv5mSXVYzNE-", output='trained_model.keras', quiet=False)
    gdown.download(id="1IKIE_OV90T82VILOUaq3uWJIBJoQ8xCI", output='tokenizer.pkl', quiet=False)
    gdown.download(id="1iL1FYHyhKeES59pQVJbK_CzVk5im2lpj", output='encoder.pkl', quiet=False)
    model = tf.keras.models.load_model('trained_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
          
    return model, tokenizer, encoder


if __name__ == '__main__':
    model, tokenizer, encoder = load()
    app.run(debug=True, host='0.0.0.0', port=8080)
    
