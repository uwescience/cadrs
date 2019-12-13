import warnings
warnings.filterwarnings('ignore')

import csv
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

train_crs = crs_cat.sample(frac=0.8,random_state=200)
test_crs = crs_cat.drop(train_crs.index)

train_crs.shape
test_crs.shape

# Create lists of texts and labels 
text =  train_crs['Description']
len(text)

labels = train_crs['cadrs']

num_words = [len(words.split()) for words in text]
min(num_words)

# Clean up text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

text = text.apply(clean_text)

text.apply(lambda x: len(x.split(' '))).sum()
#####
x_train, x_test, y_train, y_test = train_test_split(text, labels, test_size=0.2, random_state = 42)
x_train.todense()
####
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', MultinomialNB()),
              ])


from sklearn.metrinb.fit(x_train, y_train)
cs import classification_report
y_pred = nb.predict(x_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

######          precision    recall  f1-score   support
# 
#           0       0.75      0.69      0.72        26
#           1       0.82      0.86      0.84        42

#    accuracy                           0.79        68
#   macro avg       0.78      0.77      0.78        68
#weighted avg       0.79      0.79      0.79        68