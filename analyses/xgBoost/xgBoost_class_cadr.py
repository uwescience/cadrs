import csv
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from gensim.models.word2vec import Word2Vec

import os
import re
import random
import operator
import nltk 
#nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
from collections import Counter
from bs4 import BeautifulSoup

from xgboost import XGBClassifier

path_root = '/home/joseh/data/'
path_to_cadrs = path_root + 'cadrs/'
path_to_pretrained_wv = path_root
path_to_plot = path_root
path_to_save = path_root


crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training.csv'), delimiter = ',')
print('The shape: %d x %d' % crs_cat.shape)
crs_cat.columns

crs_cat.shape
crs_cat.head


# Create lists of texts and labels 
text =  crs_cat['Name']
len(text)

labels = crs_cat['cadrs']

num_words = [len(words.split()) for words in text]
min(num_words)

# Clean up text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
# STOPWORDS = set(stopwords.words('english'))
REM_GRADE = re.compile(r'\b[0-9]\w+')
REPLACE_NUM_RMN = re.compile(r'([0-9]+)|(^IX|IV|V?I{0,3}$)')
# re.sub(r'\b[0-9]\w+|([0-9]+)|([IVXLCDM]+)', '', test_2)
#test = 'English Language Arts IV 10th grade Vietnam'
#re.sub(r'(^IX|IV|V?I{0,3}$)', '', test)

def clean_text(text):
    text = REM_GRADE.sub('', text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    # text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

text = text.apply(clean_text)

text.apply(lambda x: len(x.split(' '))).sum()
text
#####
# x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state = 42)


sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', XGBClassifier(n_estimators=300, learning_rate=0.1)),
               ])

scores = cross_val_score(sgd, text, labels, cv=4)
scores.mean()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#### GRD SEARCH for hyperparameter tunning 
parameters = {
    'vect__analyzer': ['word','char'],
    'vect__ngram_range': [(1,1), (1,2),(2,3)],
    'tfidf__use_idf': (True, False),
    'clf__max_depth': (3, 6),
}

gs_clf = GridSearchCV(sgd, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf.fit(text, labels)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

gs_clf.cv_results_
#########

len(text)

student_pred = gs_clf.predict(text)

len(student_pred)

pred_cols = pd.DataFrame(student_pred, columns = ['p_CADRS'])
pred_cols.head

combined_pred = crs_student.merge(pred_cols, left_index=True, right_index=True)
combined_pred.head
combined_pred.to_csv('/home/joseh/data/cnn_cadr_student_predictions_xgBoost_CV.csv', encoding='utf-8', index=False)