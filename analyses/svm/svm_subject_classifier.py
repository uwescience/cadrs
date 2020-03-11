#!/usr/bin/env python
import csv
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

sys.path.append(project_root)
import text_preprocess as tp

path_root = os.path.join(project_root, "data") + '/'
path_to_metadata = os.path.join(project_root, "metadata") + '/'
path_to_cadrs = path_root + 'cadrs/'

crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'training_data_updated.csv'), delimiter = ',')

# look at class sizes 
crs_cat["subject_class"].value_counts()
# use the subject class as a "classifier"
text =  crs_cat['Name']
labels = crs_cat['subject_class']

num_words = [len(words.split()) for words in text]
max(num_words)

text = text.apply(tp.clean_text)

crs_abb = tp.get_metadata_dict(os.path.join(path_to_metadata, 'course_abb.json'))
d = tp.update_data(crs_cat, json_abb=crs_abb) #fix the weird procedure

text = text.replace(to_replace = d, regex=True)

# we might want to get rid of duplication after standardization
dedup_fl = pd.concat([text,labels], axis = 1).drop_duplicates()
dedup_fl['subject_class'].value_counts()

text = dedup_fl['Name']
labels = dedup_fl['subject_class']
# beggin algorithm prep
# use statify parameter to ensure balance between classes when data is split 
x_train, x_test, y_train, y_test = train_test_split(text, labels, stratify = labels ,test_size=0.2, random_state = 42)

#look at class sizes for training and test sets
y_train.value_counts()
y_test.value_counts()

###### Pipeline
sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2', random_state=42, max_iter=5, tol=None)),
               ])

parameters = {
    'vect__analyzer': ['word','char'],
    'vect__ngram_range': [(1,1), (1,2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3,1e-1),
}

gs_clf = GridSearchCV(sgd, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf.fit(x_train, y_train)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

gs_clf.cv_results_
# check the test set for results 

test_pred = gs_clf.predict(x_test)
print('accuracy %s' % accuracy_score(test_pred, y_test))
print(classification_report(y_test, test_pred))
print(confusion_matrix(y_test, test_pred))

### SVM
### with hyper parameters

# clf__alpha: 0.001
# tfidf__use_idf: True
# vect__analyzer: 'word'
# vect__ngram_range: (1, 2)

sgd = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), analyzer='word')),
                ('tfidf', TfidfTransformer(use_idf= True)),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#####
# We need to have more class support for the elective and 
# other categpries 
# we can over sample or we can do manual labeling 
