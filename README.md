# SMS Spam detection - prediction using machine learning

# Visit (informationcorners.com/text-sms-spam-classifier/)[https://informationcorners.com/text-sms-spam-classifier/]

#### spam_predict_1.py

**Result Score**

```text
--------------------------------------------------

 Accuracy of  MultinomialNB  -  98.9956958393

 Confusion matrix 
 [[1193    2]
 [  12  187]] 


 Area Under Curve of  MultinomialNB  -  98.1829650344 


--------------------------------------------------
--------------------------------------------------

 Accuracy of  LogisticRegression  -  98.2065997131

 Confusion matrix 
 [[1192    3]
 [  22  177]] 


 Area Under Curve of  LogisticRegression  -  98.6076827653 


--------------------------------------------------
Finding top spam and ham words
Total Features:  7465
Total Observations in each class  [ 3632.   548.]

 -------------------- Top 20 spam words --------------------
            ham  spam  spam_ratio
token                            
claim         1    82  543.474453
prize         1    71  470.569343
uk            1    61  404.291971
150p          1    53  351.270073
tone          1    45  298.248175
16            1    40  265.109489
18            1    37  245.226277
guaranteed    1    37  245.226277
1000          1    34  225.343066
500           1    32  212.087591
100           1    29  192.204380
cs            1    27  178.948905
ringtone      1    26  172.321168
10p           1    24  159.065693
awarded       1    24  159.065693
www           3    69  152.437956
000           1    22  145.810219
5000          1    21  139.182482
weekly        1    21  139.182482
mob           1    21  139.182482

 -------------------- Top 20 non-spam words --------------------
          ham  spam  spam_ratio
token                          
gt        240     1    0.027616
lt        237     1    0.027965
he        176     1    0.037658
lor       123     1    0.053884
she       118     1    0.056167
later     116     1    0.057136
da        111     1    0.059709
ask        67     1    0.098921
but       332     5    0.099815
amp        66     1    0.100420
said       65     1    0.101965
doing      65     1    0.101965
home      129     2    0.102756
really     63     1    0.105202
morning    60     1    0.110462
come      175     3    0.113618
lol        57     1    0.116276
its       170     3    0.116960
anything   55     1    0.120504
cos        55     1    0.120504

```


#### spam_predict_2.py

**Result Score**

```text
 Running the model - SGDClassifier (SVM)

 First run - Accuracy of SGDClassifier (SVM) - 97.77618364418939

 Tuning training parameters

 After tuning - Accuracy (after tuning) of SGDClassifier (SVM) - 98.20659971305595

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

 After tuning - Accuracy (after tuning) of Multinomial NaiveBayes - 99.0674318507891

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
