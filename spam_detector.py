#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)

# 3. Vectorize dataset
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)


# 4. Building and evaluating model
def build_evaluate_model(model_obj):
    print(model_obj.__class__.__name__)
    print('-'*20)

    model_obj.fit(X_train_dtm, y_train)
    y_predict = model_obj.predict(X_test_dtm)
    print('Accuracy - ', metrics.accuracy_score(y_test, y_predict))
    print('Confusion matrix \n', metrics.confusion_matrix(y_test, y_predict), '\n')

    y_predict_prob = model_obj.predict_proba(X_test_dtm)[:, 1]
    print('Area Under Curve - ', metrics.roc_auc_score(y_test, y_predict_prob), '\n\n')

nb = MultinomialNB()
build_evaluate_model(nb)

log_reg = LogisticRegression()
build_evaluate_model(log_reg)

# 5. Model insights
print('Finding top spam and ham words')
X_train_tokens = vect.get_feature_names()

print('Total Features: ', len(X_train_tokens))

ham_token_count = nb.feature_count_[0, :]
spam_token_count = nb.feature_count_[1, :]
token_count = pd.DataFrame({'token':X_train_tokens, 'ham':ham_token_count,
                            'spam': spam_token_count}).set_index('token')
token_count['ham'] += 1
token_count['spam'] += 1

print('Total Observations in each class ', nb.class_count_)
token_count['ham_freq'] = token_count['ham'] / nb.class_count_[0]
token_count['spam_freq'] = token_count['spam'] / nb.class_count_[1]

token_count['spam_ratio'] = token_count['spam_freq'] / token_count['ham_freq']
token_count.sort_values('spam_ratio', ascending=False, inplace=True)

print('-'*50)
print(token_count.head(5))
print('-'*50)
print(token_count.sample(5, random_state=42))





