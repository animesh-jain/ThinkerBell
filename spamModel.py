from pyexpat import features

import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Reading testData from txt file
df = pd.read_table('SMSSpamCollection.txt')
# mapping = {'ham': float(0), 'spam': float(1)}
# df['Label'] = df['Label'].map(mapping)
X = df['Message']
y = df['Label']
cv = CountVectorizer()
X = cv.fit_transform(X)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# SVM Model
# model = svm.SVC()
# model.fit(features, y_train)
# features_test = cv.transform(X_test)
# print(model.score(features_test, y_test))

# test Multiple Messages through testCollection file
df2 = pd.read_table('testCollection')
testSet = df2['Message']
strings = cv.transform(testSet).toarray()
prediction = clf.predict(strings)
print(df2['Message'])
print(prediction)

# Creating Flask Web Application to predict a message via UI
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        if my_prediction == 'spam':
            my_prediction = 1
        else:
            my_prediction = 0
        print(my_prediction)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
