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
name_save = 'cadrs_cnn_two_branches.hdf5'


word_vector_dim = int(3e2)
nb_filters = 150
filter_size_a = 3
filter_size_b = 4
drop_rate = 0.3
batch_size = 64
nb_epoch = 20
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
maxlen = max(num_words) + 1  # max number of words in a description to consider

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

# split training data into test, validation
x_train, x_val, y_train, y_val = train_test_split(text_tok, labels, test_size=0.2, random_state = 42)

# Load google's pretrained model
word_vectors = KeyedVectors.load_word2vec_format(path_root + 'GoogleNews-vectors-negative300.bin', binary=True)

word_vector_dim=300
vocabulary_size=min(len(word_index)+1,max_words)
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

####
word_vector_dim = int(3e2)
nb_filters = 150
filter_size_a = 3
filter_size_b = 4
drop_rate = 0.3
batch_size = 64
nb_epoch = 6
my_optimizer = 'adam' 
my_patience = 2
## Model 
my_input = Input(shape=(maxlen,)) # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument

embedding = Embedding(input_dim=embedding_matrix.shape[0], # vocab size, including the 0-th word used for padding
                        output_dim=word_vector_dim,
                        weights=[embedding_matrix], # we pass our pre-trained embeddings
                        input_length=maxlen,
                        trainable=False,
                        )(my_input)

embedding_dropped = Dropout(drop_rate)(embedding)

# feature map size should be equal to max_size-filter_size+1
# tensor shape after conv layer should be (feature map size, nb_filters)
print('branch A:',nb_filters,'feature maps of size',maxlen-filter_size_a+1)
print('branch B:',nb_filters,'feature maps of size',maxlen-filter_size_b+1)

# A branch
conv_a = Conv1D(filters = nb_filters,
              kernel_size = filter_size_a,
              activation = 'relu',
              )(embedding_dropped)

pooled_conv_a = GlobalMaxPooling1D()(conv_a)

pooled_conv_dropped_a = Dropout(drop_rate)(pooled_conv_a)

# B branch
conv_b = Conv1D(filters = nb_filters,
              kernel_size = filter_size_b,
              activation = 'relu',
              )(embedding_dropped)

pooled_conv_b = GlobalMaxPooling1D()(conv_b)

pooled_conv_dropped_b = Dropout(drop_rate)(pooled_conv_b)

concat = Concatenate()([pooled_conv_dropped_a,pooled_conv_dropped_b])

concat_dropped = Dropout(drop_rate)(concat)

# we finally project onto a single unit output layer with sigmoid activation
prob = Dense(2,
             activation = 'sigmoid',
             )(concat_dropped)

model = Model(my_input, prob)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

model.summary()

early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                               patience=my_patience,
                               mode='max')

# make sure that the model corresponding to the best epoch is saved
checkpointer = ModelCheckpoint(filepath=path_to_save + name_save,
                               monitor='val_acc',
                               save_best_only=True,
                               verbose=0)

model.fit(x_train, 
          y_train,
          batch_size = batch_size,
          epochs = nb_epoch,
          validation_data = (x_val, y_val),
          callbacks = [early_stopping,checkpointer])

# plot
get_doc_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[9].output])

n_plot = 50

doc_emb = get_doc_embedding([np.array(x_val[:n_plot]),0])[0]

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
doc_emb_pca = my_pca.fit_transform(doc_emb) 
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

labels_plt = y_val[:n_plot]
my_colors = ['blue','red']

fig, ax = plt.subplots()


labels_plt = labels_plt[:,0].astype(int)
labels_plt

for label in list(set(labels_plt)):
    idxs = [idx for idx,elt in enumerate(labels_plt) if elt==label]
    ax.scatter(doc_emb_tsne[idxs,0], 
               doc_emb_tsne[idxs,1], 
               c = my_colors[label],
               label=str(label),
               alpha=0.7,
               s=10)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based course embeddings \n (first 50 courses from test set)',fontsize=10)
fig.set_size_inches(6,4)
# fig.savefig(path_to_plot + 'doc_embeddings_init.pdf',bbox_inches='tight')