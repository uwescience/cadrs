import csv
import json
import numpy as np
import pandas as pd

import matplotlib as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, StratifiedKFold

from gensim.models.word2vec import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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

crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training_rsd.csv'), delimiter = ',')
print('The shape: %d x %d' % crs_cat.shape)
crs_cat.columns

crs_cat.shape
crs_cat.head


# Create lists of texts and labels 
text =  crs_cat['Name']
text[3]
num_words = [len(words.split()) for words in text]
max_seq_len = max(num_words) + 1

## prep outcome labes 1=cadrs, 0=not cadrs 
labels = crs_cat['cadr']
labels = list(labels)
labels

# Clean up text
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
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
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = ' '.join(word for word in text.split() if len(word)>1)
    return text

text = text.apply(clean_text)
text[1]

text.apply(lambda x: len(x.split(' '))).sum()

text = text.astype(str).values.tolist() # list of text samples
text
##### Tokenize
#maxlen = max(num_words) + 1  # max number of words in a description to consider

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = 500 # total words of vocabulary we will consider

text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape
np.mean(text_tok > 0)


# split training data into test, validation
def data_kfold(k):
    X_train = text_tok       
    y_train = labels
    folds = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=1).split(X_train, y_train))
    return folds, X_train, y_train

k = 7
folds, X_train, y_train = data_kfold(k)

from keras.utils import to_categorical

y_train = to_categorical(np.asarray(y_train))
# Load google's pretrained model
word_vectors = KeyedVectors.load_word2vec_format(path_root + 'GoogleNews-vectors-negative300.bin', binary=True)

word_vector_dim=300
vocabulary_size= max_words+1
embedding_matrix = np.zeros((vocabulary_size, word_vector_dim))

for word, i in word_index.items():
    if i>=max_words:
        continue
    try:
        embedding_vector = word_vectors[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),word_vector_dim)


len(embedding_matrix)
embedding_matrix.shape
type(embedding_matrix)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / max_words



def create_model(nb_filters, filter_size_a, drop_rate, my_optimizer = 'adam'):
    my_input = Input(shape=(None,))
    embedding = Embedding(input_dim=embedding_matrix.shape[0],
        output_dim=word_vector_dim,
        weights=[embedding_matrix],
        input_length=max_seq_len,
        trainable=False,)(my_input)
    embedding_dropped = Dropout(drop_rate)(embedding)
    conv_a = Conv1D(filters = nb_filters,
        kernel_size = filter_size_a,
        activation = 'relu',)(embedding_dropped)
    pooled_conv_a = GlobalMaxPooling1D()(conv_a)
    prob = Dense(2, activation = 'sigmoid',)(pooled_conv_a)
    model = Model(my_input, prob)
    
    model.compile(loss='binary_crossentropy',
        optimizer = my_optimizer,
        metrics = ['accuracy']) 
    return model

def get_callbacks(path_to_save, name_save, my_patience):
    early_stopping = EarlyStopping(monitor='val_acc',
        patience=my_patience, mode='max')
    checkpointer = ModelCheckpoint(filepath=path_to_save + name_save,
        monitor='val_acc', save_best_only=True, verbose=0)
    return [early_stopping,checkpointer]

## k-fold cross validation
for j, (train_idx, val_idx) in enumerate(folds):
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv= y_train[val_idx]
    
    batch_size = 10
    name_save = "final_model_fold" + str(j) + "_title.h5"
    callbacks = get_callbacks(path_to_save=path_to_save, name_save=name_save, my_patience=2)
    
    model = create_model(nb_filters=64, filter_size_a=2, drop_rate=.3)
    
    model.fit(X_train_cv,y_train_cv, batch_size=batch_size, epochs=20, 
            validation_data = (X_valid_cv, y_valid_cv), callbacks = callbacks)
 
print(model.metrics_names)
print(model.evaluate(X_valid_cv, y_valid_cv, verbose=1))

###
Xnew = [['japanese'],['algebra'],['geometry'],['algebra', '1'],['english', 'language', 'literature'], [
    'computer', 'science', 'courses', 'prepare', 'students',
    'take', 'the', 'international', 'baccalaureate', 'computer',
    'science', 'exam'], 
    ['biology'],
    ['word history'],
    ['spanish'],
     ['mathematics', 'test', 'preparation', 'courses', 'provide', 'students', 'activities', 'analytical', 'thinking']
    ]

new_sequences2 = tokenizer.texts_to_sequences(Xnew)
new_sequences2
new_data = pad_sequences(new_sequences2, maxlen=max_seq_len)
new_data
predictions = model.predict(new_data)
predictions


