import warnings

import csv
import json
import numpy as np
import pandas as pd

import matplotlib as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

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
text =  train_crs['Name']
text[3]
num_words = [len(words.split()) for words in text]
max_seq_len = max(num_words) + 1

## prep outcome labes 1=cadrs, 0=not cadrs 
labels = train_crs['cadr']
labels = list(labels)
labels

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

text = text.astype(str).values.tolist() # list of text samples
text
##### Tokenize
#maxlen = max(num_words) + 1  # max number of words in a description to consider

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = 600 # total words of vocabulary we will consider

text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape
np.mean(text_tok > 0)

from keras.utils import to_categorical

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', text_tok.shape)
print('Shape of label tensor:', labels.shape)

# split training data into test, validation
x_train, x_val, y_train, y_val = train_test_split(text_tok, labels, test_size=0.2, random_state = 42)

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
#####

model = create_model(nb_filters=64, filter_size_a=2, drop_rate=.3)

model.summary()

model.fit(x_train, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_data = (x_val, y_val))

####
from numpy import argmax

test_crs.head


test_crs['Name']=test_crs['Name'].fillna("")

text_out =  test_crs['Name']
num_words_2 = [len(words.split()) for words in text_out]
max(num_words_2)

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

text = text_out.apply(clean_text)

text.apply(lambda x: len(x.split(' '))).sum()

text = text.astype(str).values.tolist() # list of text samples

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)


text_tok = pad_sequences(sequences, maxlen=max_seq_len+1)
text_tok.shape


predictions_new = model.predict(text_tok)

pred_cols = pd.DataFrame(predictions_new, columns = ['p_notCADRS', 'p_CADRS'])
pred_cols.head
pred_cols.shape

class_y = np.argmax(predictions_new,axis=1)
len(class_y)

pred_cols['pred_class'] = class_y
pred_cols.head()
pred_cols.shape
test_crs.shape
test_crs = test_crs.reset_index(drop=True)
pred_cols = pred_cols.reset_index(drop=True)

combined_pred = test_crs.merge(pred_cols, left_index=True, right_index=True)

combined_pred.shape
combined_pred.to_csv('/home/joseh/data/cnn_cadr_student_testset_title.csv', encoding='utf-8', index=False)


## Try k-fold cross validation

for j, (train_idx, val_idx) in enumerate(folds):
    
    print('\nFold ',j)
    X_train_cv = X_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    y_valid_cv= y_train[val_idx]
    
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)
    generator = gen.flow(X_train_cv, y_train_cv, batch_size = batch_size)
    model = get_model()
    model.fit_generator(
                generator,
                steps_per_epoch=len(X_train_cv)/batch_size,
                epochs=15,
                shuffle=True,
                verbose=1,
                validation_data = (X_valid_cv, y_valid_cv),
                callbacks = callbacks)
    
    print(model.evaluate(X_valid_cv, y_valid_cv))