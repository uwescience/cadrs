import warnings
warnings.filterwarnings('ignore')

import csv
import json
import numpy as np
import pandas as pd


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

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


path_root = '/Users/josehernandez/Documents/eScience/'
path_to_cadrs = path_root + 'data/'


crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training_rsd.csv'), delimiter = ',')
print('The shape: %d x %d' % crs_cat.shape)
crs_cat.columns

crs_cat.shape
crs_cat.head

train_crs = crs_cat.sample(frac=0.8,random_state=200)
test_crs = crs_cat.drop(train_crs.index)

train_crs.shape
test_crs.shape

# Create lists of texts and labels 
text =  train_crs['Name']
len(text)

labels = train_crs['cadr']

num_words = [len(words.split()) for words in text]
min(num_words)

# Clean up text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
REM_GRADE = re.compile(r'\b[0-9]\w+')
REPLACE_NUM_RMN = re.compile(r"([0-9]+)|(i[xv]|v?i{0,3})$")


def clean_text(text):
    text = text.lower() # lowercase text
    text = REM_GRADE.sub('', text)
    text = REPLACE_NUM_RMN.sub('', text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

text = text.apply(clean_text)
text

text.apply(lambda x: len(x.split(' '))).sum()
#####
x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state = 42)

####

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import EntityExtractor

from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('text_union', FeatureUnion(
        transformer_list = [
            ('keyphrase_feature', Pipeline([
                ('keyphrase_extractor', KeyphraseExtractor()),
                ('keyphrase_vect', TfidfVectorizer()),
            ])),
        ],
        transformer_weights= {
            'entity_feature': 0.6,
            'keyphrase_feature': 0.2,
        }
    )),
    ('clf', LogisticRegression()),
])

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=["not CADR", "CADR"]))

# accuracy 0.7647058823529411

#              precision    recall  f1-score   support

#    not CADR       0.66      0.81      0.72        26
#        CADR       0.86      0.74      0.79        42

#    accuracy                           0.76        68
#   macro avg       0.76      0.77      0.76        68
#weighted avg       0.78      0.76      0.77        68
