import pandas as pd
import json
from pprint import pprint
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Flask!'

@app.route('/predictions')
def predictions():
    model = None
    model_filename = 'model.pickle'
    model_file = Path(model_filename)

    if model_file.is_file():
        read_model = open(model_file, 'rb')
        model = pickle.load(read_model)
        read_model.close()
    else:
        data = pd.read_csv('data/products.csv', delimiter=';')
        stemmer = RSLPStemmer()
        words = stopwords.words('portuguese')

        data['cleared'] = data['title'].apply(lambda x: ' '.join([stemmer.stem(i) for i in re.sub("^a-zA-Z", " ", x).split() if i not in words]))

        X_train, X_test, y_train, y_test = train_test_split(data['cleared'], data['category'], test_size=0.2)

        clf = LinearSVC(C=1.0, dual=False)

        pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                            ('clf', clf)])

        model = pipeline.fit(X_train, y_train)

        print('Accuracy score: {}'.format(model.score(X_test, y_test)))
        save_model = open(model_file, 'wb')
        pickle.dump(model, save_model)
        save_model.close()

    print(request.json)
    productName = request.json['name']

    prediction = model.predict([productName])[0]

    print('Prediction: {}'.format(prediction))
    return {'category': prediction}
