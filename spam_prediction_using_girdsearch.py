#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
from pprint import pprint

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np

spam_file = 'data/sms.csv'

if not os.path.isfile(spam_file):
    print(spam_file, ' is missing.')
    exit()

# 1. Loading dataset
sms_df = pd.read_csv(spam_file, sep='\t')
# converting label to a numerical
sms_df['label_num'] = sms_df['label'].map({'ham': 0, 'spam': 1})

# 2. Feature matrix (X), response vector (y) and train_test_split
X = sms_df['message']
y = sms_df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 2. Converting text and numbers (matrix)
# 3. training in ml (all can be written in one line)

models = {'Multinomial NaiveBayes': {'clf': MultinomialNB(),
                                     'clf_params': {
                                         'clf__alpha': (0.001, 1.0),
                                         'clf__fit_prior': (True, False),

                                     }},
          'SGDClassifier (SVM)': {'clf': SGDClassifier(loss='hinge', penalty='l2',
                                                       alpha=1e-3, random_state=1,
                                                       max_iter=5, tol=None),
                                  'clf_params': {
                                      'clf__alpha': (0.001, 1.0),
                                  }
                                  }}

for model in models.keys():
    print('\nRunning the model - {}'.format(model))
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', models[model]['clf'])])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    accuracy = np.mean(predicted == y_test)
    print('\nFirst run - Accuracy of {} - {}'.format(model, accuracy * 100))

    print('\nTuning training parameters')
    # 4. Auto-tuning the training parameters using Grid Search for both feature extraction and classifier
    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'vect__stop_words': ['english', None],
                  'vect__max_df': (0.5, 1.0),
                  'vect__min_df': (1, 2),
                  'tfidf__use_idf': (True, False),
                  'tfidf__smooth_idf': (True, False),
                  'tfidf__sublinear_tf': (True, False),
                  }

    parameters.update(models[model]['clf_params'])

    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf.fit(X_train, y_train)

    gs_predicted = gs_clf.predict(X_test)
    accuracy = np.mean(gs_predicted == y_test)
    print('\nAfter tuning - Accuracy (after tuning) of MultinomialNB (naive Bayes) - {}'.format(accuracy * 100))

    print('\nGrid Search best score -')
    print(gs_clf.best_score_)

    print('\nGrid Search best parameters -')
    pprint(gs_clf.best_params_)

    print('\nMetrics classification report ')
    print(metrics.classification_report(y_test, predicted))

    print('\nMetric Confusion matrix')
    print(metrics.confusion_matrix(y_test, predicted))


'''
Running the model - SGDClassifier (SVM)

First run - Accuracy of SGDClassifier (SVM) - 97.77618364418939

Tuning training parameters

After tuning - Accuracy (after tuning) of MultinomialNB (naive Bayes) - 98.20659971305595

Grid Search best score -
0.977990430622

Grid Search best parameters -
{'clf__alpha': 0.001,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': True,
 'tfidf__use_idf': True,
 'vect__max_df': 0.5,
 'vect__min_df': 2,
 'vect__ngram_range': (1, 1),
 'vect__stop_words': None}

Metrics classification report 
             precision    recall  f1-score   support

          0       0.98      1.00      0.99      1212
          1       0.99      0.84      0.91       182

avg / total       0.98      0.98      0.98      1394


Metric Confusion matrix
[[1210    2]
 [  29  153]]

Running the model - Multinomial NaiveBayes

First run - Accuracy of Multinomial NaiveBayes - 96.26972740315638

Tuning training parameters

After tuning - Accuracy (after tuning) of MultinomialNB (naive Bayes) - 99.0674318507891

Grid Search best score -
0.983253588517

Grid Search best parameters -
{'clf__alpha': 0.001,
 'clf__fit_prior': True,
 'tfidf__smooth_idf': True,
 'tfidf__sublinear_tf': True,
 'tfidf__use_idf': False,
 'vect__max_df': 0.5,
 'vect__min_df': 1,
 'vect__ngram_range': (1, 2),
 'vect__stop_words': None}

Metrics classification report 
             precision    recall  f1-score   support

          0       0.96      1.00      0.98      1212
          1       1.00      0.71      0.83       182

avg / total       0.96      0.96      0.96      1394


Metric Confusion matrix
[[1212    0]
 [  52  130]]
 '''