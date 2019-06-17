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


def extract_regions(tokens, filter_size):
    regions = []
    regions.append(' '.join(tokens[:filter_size]))
    for i in range(filter_size, len(tokens)):
        regions.append(' '.join(tokens[(i-filter_size+1):(i+1)]))
    return regions


path_root = '/home/joseh/data/'
path_to_cadrs = path_root + 'cadrs/'
path_to_pretrained_wv = path_root
path_to_plot = path_root
path_to_save = path_root

use_pretrained = True
name_save = 'cadrs_cnn_two_branches.hdf5'

max_features = int(2e4)
stpwd_thd = 1 
max_size = 200
word_vector_dim = int(3e2)
do_static = False
nb_filters = 150
filter_size_a = 3
filter_size_b = 4
drop_rate = 0.3
batch_size = 64
nb_epoch = 6
my_optimizer = 'adam' 
my_patience = 2
validation_split = 0.2

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

num_words = [len(words.split()) for words in text]
max(num_words)

## prep outcome labes 1=cadrs, 0=not cadrs 
labels = train_crs['cadrs']
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

##### Tokenize
maxlen = max(num_words) + 1  # max number of words in a title to consider

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = len(word_index) + 1 # total words of vocabulary we will consider

text_tok = pad_sequences(sequences, maxlen=maxlen)
text_tok.shape
np.mean(text_tok > 0)

from keras.utils import to_categorical

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', text_tok.shape)
print('Shape of label tensor:', labels.shape)

x_train, x_val, y_train, y_val = train_test_split(text_tok, labels, test_size=0.2, random_state = 42)

####
glove_dir = '/home/joseh/data/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

#let's take a look
first_items = {k: embeddings_index[k] for k in sorted(embeddings_index.keys())[200000:200010]}

for k,v in first_items.items():
    print (k, len(list(filter(None, v))))

# type(embeddings_index)

embedding_dim = 100 # from the GLOVE embeddings 
max_words_em = len(word_index) + 1 # USE TO REDUCE THE size to only what is present 

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

len(embedding_matrix)
type(embedding_matrix)

nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
nonzero_elements / max_words

####

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

###########

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.initializers import Constant
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

embedding_layer = Embedding(max_words,
                            embedding_dim,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)


sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 2, activation='relu')(embedded_sequences)
x = MaxPooling1D()(x)

x = Conv1D(128, 2, activation='relu')(embedded_sequences)
x = MaxPooling1D()(x)

x = Conv1D(128, 4, activation='relu')(x)
x = GlobalMaxPooling1D()(x)

x = Dense(128, activation='relu')(x)

preds = Dense(2, activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))