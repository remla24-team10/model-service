
import numpy as np

from keras._tf_keras.keras import Model
from sklearn.preprocessing import LabelEncoder
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

def load() -> Model:
    
    model = Model # load pretrained from url
    # load encoder
    #load tokenizer


    return model

def preprocess(raw_X_test: np.ndarray, model: Model, tokenizer: Tokenizer, encoder: LabelEncoder):
    X_test = pad_sequences(tokenizer.texts_to_sequences(raw_X_test), maxlen=200)
    #char_index = tokenizer.word_index
    
    
    return X_test

def predict_classes(model: Model, x_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Predict class labels for samples in x_test.

    Args:
        model: Trained model to use for prediction.
        x_test: Test data.
        threshold: Threshold for converting probabilities to binary labels.

    Returns:
        Predicted binary labels for the samples in x_test.
    """
    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > threshold).astype(int)

    return y_pred_binary
