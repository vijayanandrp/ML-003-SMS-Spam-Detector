# SMS Spam detection - prediction using machine learning


#### spam_detector.py

**Result Score**

```text
MultinomialNB
--------------------
Accuracy -  0.989956958393
Confusion matrix
 [[1193    2]
 [  12  187]]

Area Under Curve -  0.981829650344


LogisticRegression
--------------------
Accuracy -  0.982065997131
Confusion matrix
 [[1192    3]
 [  22  177]]

Area Under Curve -  0.986076827653


Finding top spam and ham words
Total Features:  7465
Total Observations in each class  [ 3632.   548.]
--------------------------------------------------
       ham  spam  ham_freq  spam_freq  spam_ratio
token
claim    1    82  0.000275   0.149635  543.474453
prize    1    71  0.000275   0.129562  470.569343
uk       1    61  0.000275   0.111314  404.291971
150p     1    53  0.000275   0.096715  351.270073
tone     1    45  0.000275   0.082117  298.248175
--------------------------------------------------
        ham  spam  ham_freq  spam_freq  spam_ratio
token
kudi      2     1  0.000551   0.001825    3.313869
advice    4     1  0.001101   0.001825    1.656934
panic     2     1  0.000551   0.001825    3.313869
vday      3     1  0.000826   0.001825    2.209246
ones      5     1  0.001377   0.001825    1.325547
```


#### spam_prediction_using_girdsearch.py

**Result Score**

```text
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


```