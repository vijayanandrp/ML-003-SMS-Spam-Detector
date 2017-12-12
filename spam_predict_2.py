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
    print('\n', 'Running the model - {}'.format(model))
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', models[model]['clf'])])

    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)

    accuracy = np.mean(predicted == y_test)
    print('\n', 'First run - Accuracy of {} - {}'.format(model, accuracy * 100))

    print('\n', 'Tuning training parameters')
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
    print('\n', 'After tuning - Accuracy (after tuning) of {} - {}'.format(model, accuracy * 100))

    print('\n', 'Grid Search best score -')
    print(gs_clf.best_score_)

    print('\n', 'Grid Search best parameters -')
    pprint(gs_clf.best_params_)

    print('\n', 'Metrics classification report ')
    print(metrics.classification_report(y_test, predicted))

    print('\n', 'Metric Confusion matrix')
    print(metrics.confusion_matrix(y_test, predicted))
