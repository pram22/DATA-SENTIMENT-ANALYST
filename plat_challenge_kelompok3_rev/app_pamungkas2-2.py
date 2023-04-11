#Import Flask
from flask import Flask, jsonify, make_response, render_template
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from


import json
import pandas as pd
import pickle5 as pickle
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask API =================================================================
app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda: 'API Sentiment Analysis Kelompok 3'),
    'version': LazyString(lambda: '1.0.0'),
    'description': LazyString(lambda: 'API Sentiment Analysis - Kelompok 3 (Pram, Dico, Hisyam)')
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
# Initialize Swagger from swagger_template & swagger_config============================================
swagger = Swagger(app, template=swagger_template, 
                  config=swagger_config)

#initialize LSTM feature
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative','neutral','positive']

# Fungsi buat cleaning=================================================================================

    # Function for text cleansing
def cleaning(sent):
    string = sent.lower()

    string = re.sub(r'[^a-zA-Z0-9]',' ',string)
    return string

#load pickle feature lstm
file = open("LSTM/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()
#load model lstm
model_file_from_lstm = load_model('LSTM/model_lstm.h5')

#load pickle feature cnn
file = open("CNN/x_pad_sequences-3.pickle", 'rb')
feature_file_from_cnn = pickle.load(file)
file.close()
#load model cnn
model_file_from_cnn = load_model('CNN/model-2.h5')


#load pickle feature rnn
file = open("RNN/x_pad_sequences-4.pickle", 'rb')
feature_file_from_rnn = pickle.load(file)
file.close()
#load model rnn
model_file_from_rnn = load_model('RNN/model-3.h5')

#LSTM
# Input Text
@swag_from('docs/lstm_text.yml', methods=['POST'])
@app.route('/lstm_text', methods=['POST'])
def lstm_text():

    original_text = request.form.get('text')
    text = [cleaning(original_text)]

    predicted = tokenizer.texts_to_sequences(text)
    guess = pad_sequences(predicted, maxlen=feature_file_from_lstm.shape[1])

    prediction = model_file_from_lstm.predict(guess)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Input FILE
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm_file', methods=['POST'])

def lstm_file():
    
    file = request.files['file']

    df = pd.read_csv(file, encoding='utf-8')

    kolom_text = df.iloc[:, 0]

    text_clean = []
    get_sentiment = []

    for text in kolom_text:
        text_clean_test = [cleaning(text)]

        feature = tokenizer.texts_to_sequences(text_clean_test)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])

        prediction = model_file_from_lstm.predict(feature)
        get_sentiment_predict = sentiment[np.argmax(prediction[0])]

        text_clean.append(text_clean_test)
        get_sentiment.append(get_sentiment_predict)

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using LSTM",
        'data' : {
            'text': text_clean,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data
    
#CNN    
# Input TEXT
@swag_from('docs/cnn_text.yml', methods=['POST'])
@app.route('/cnn_text', methods=['POST'])
def nn_text():

    original_text = request.form.get('text')
    text = [cleaning(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

    prediction = model_file_from_cnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using CNN",
        'data' : {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Input FILE
@swag_from("docs/cnn_file.yml", methods=['POST'])
@app.route('/cnn_file', methods=['POST'])

def cnn_file():
    
    file = request.files['file']

    df = pd.read_csv(file, encoding='utf-8')

    kolom_text = df.iloc[:, 0]

    text_clean = []
    get_sentiment = []

    for text in kolom_text:
        text_clean_test = [cleaning(text)]

        feature = tokenizer.texts_to_sequences(text_clean_test)
        feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])

        prediction = model_file_from_cnn.predict(feature)
        get_sentiment_predict = sentiment[np.argmax(prediction[0])]

        text_clean.append(text_clean_test)
        get_sentiment.append(get_sentiment_predict)

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using CNN",
        'data' : {
            'text': text_clean,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data

#RNN    
# Input TEXT
@swag_from("docs/rnn_text.yml", methods=['POST'])
@app.route('/rnn_text', methods=['POST'])
def rnn_text():

    original_text = request.form.get('text')
    text = [cleaning(original_text)]

    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

    prediction = model_file_from_rnn.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code' : 200,
        'description' : "Result of Sentiment Analysis using RNN",
        'data' : {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Input FILE
@swag_from("docs/rnn_file.yml", methods=['POST'])
@app.route('/rnn_file', methods=['POST'])

def rnn_file():
    
    file = request.files['file']

    df = pd.read_csv(file, encoding='utf-8')

    kolom_text = df.iloc[:, 0]

    text_clean = []
    get_sentiment = []

    for text in kolom_text:
        text_clean_test = [cleaning(text)]

        feature = tokenizer.texts_to_sequences(text_clean_test)
        feature = pad_sequences(feature, maxlen=feature_file_from_rnn.shape[1])

        prediction = model_file_from_rnn.predict(feature)
        get_sentiment_predict = sentiment[np.argmax(prediction[0])]

        text_clean.append(text_clean_test)
        get_sentiment.append(get_sentiment_predict)

    json_response = {
        'status_code' : 200,
        'description' : "Result of Analysis Using RNN",
        'data' : {
            'text': text_clean,
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
    app.run()