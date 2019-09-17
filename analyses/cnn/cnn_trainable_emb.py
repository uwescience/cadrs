import csv
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant

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

use_pretrained = True
name_save = 'cadrs_cnn_titles_cadrs.hdf5'

validation_split = 0.2

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
text =  crs_cat['Name']

num_words = [len(words.split()) for words in text]
max_seq_len = max(num_words) + 1

## prep outcome labes 1=cadrs, 0=not cadrs 
labels = crs_cat['cadr']
labels = list(labels)
labels

# Clean up text
# Conditions might not use all in short text sequences 
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
REM_GRADE = re.compile(r'\b[0-9]\w+')
REPLACE_NUM_RMN = re.compile(r'([0-9]+)|(^IX|IV|V?I{0,3}$)')


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text) 
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

text = text.apply(clean_text)
text.apply(lambda x: len(x.split(' '))).sum()
text = text.astype(str).values.tolist()
text[1:50]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = 650 # total words of vocabulary we will consider

text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape
np.mean(text_tok > 0)

from keras.utils import to_categorical

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', text_tok.shape)
print('Shape of label tensor:', labels.shape)

# split training data into test, validation
x_train, x_val, y_train, y_val = train_test_split(text_tok, labels, test_size=0.2, random_state = 42)

# Prepare embedding matrix
word_vector_dim=100
vocabulary_size= max_words+1
embedding_matrix = np.zeros((vocabulary_size, word_vector_dim))

def create_model(nb_filters, filter_size_a, drop_rate, my_optimizer = 'adam'):
    my_input = Input(shape=(None,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
        input_length=max_seq_len,
        output_dim=word_vector_dim,
        trainable=True,)(my_input)
   # embedding_dropped = Dropout(drop_rate)(embedding)
    x = Conv1D(filters = nb_filters,
        kernel_size = filter_size_a,
        activation = 'relu',)(embedding)
    x = MaxPooling1D(pool_size=5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    prob = Dense(2, activation = 'sigmoid',)(x)
    model = Model(my_input, prob)
    
    model.compile(loss='binary_crossentropy',
        optimizer = my_optimizer,
        metrics = ['accuracy']) 
    return model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

drop_rate = 0.3
batch_size = 20
nb_epoch = 20

param_grid = dict(nb_filters=[32, 64, 128, 150],
    filter_size_a=[1, 2, 3, 4],
    drop_rate=[drop_rate])

model = KerasClassifier(build_fn=create_model,
    epochs=nb_epoch, batch_size=batch_size,
    verbose=False)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
    cv=4, verbose=1, n_iter=5)

grid_result = grid.fit(x_train, y_train)

test_accuracy = grid.score(x_val, y_val)

s = ('Best Accuracy : ''{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')

output_string = s.format(
            grid_result.best_score_,
            grid_result.best_params_,
            test_accuracy)

print(output_string)

# Run without a test split, only CV ? Is this the appropriate way?
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

drop_rate = 0.3
batch_size = 20
nb_epoch = 20

param_grid = dict(nb_filters=[32, 64, 128, 150],
    filter_size_a=[1, 2, 3, 4],
    drop_rate=[drop_rate])

model = KerasClassifier(build_fn=create_model,
    epochs=nb_epoch, batch_size=batch_size,
    verbose=False)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
    cv=4, verbose=1, n_iter=5)

grid_result_cv = grid.fit(text_tok, labels)


s = ('Best Accuracy : ''{:.4f}\n{}\n')

output_string = s.format(
            grid_result_cv.best_score_,
            grid_result_cv.best_params_)

print(output_string)

