import warnings
warnings.filterwarnings('ignore')

import csv
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from gensim.models.word2vec import Word2Vec

from keras.models import Model, load_model
from keras import backend as K
from keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Dense, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import re
import random
import operator
from nltk import pos_tag
from collections import Counter
from bs4 import BeautifulSoup

print('packages loaded')

def extract_regions(tokens, filter_size):
    regions = []
    regions.append(' '.join(tokens[:filter_size]))
    for i in range(filter_size, len(tokens)):
        regions.append(' '.join(tokens[(i-filter_size+1):(i+1)]))
    return regions

print('functions defined')

path_root = '/home/joseh/data/'
path_to_cadrs = path_root + 'cadrs/'
path_to_pretrained_wv = path_root
path_to_plot = path_root
path_to_save = path_root

use_pretrained = True
name_save = 'cadrs_cnn_two_branches.hdf5'
print('best model will be saved with name:',name_save)

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

if not use_pretrained:
    # if the embeddings are initialized randomly, using static mode doesn't make sense
    do_static = False
    print("not using pre-trained embeddings, overwriting 'do_static' argument")

crs_cat =  pd.read_csv(os.path.join(path_to_cadrs,'cadrs_training.csv'), delimiter = ',')
print('The shape: %d x %d' % crs_names.shape)
crs_cat.columns

crs_cat.shape
crs_cat.head

train_crs = crs_cat.sample(frac=0.8,random_state=200)
test_crs = crs_cat.drop(train_crs.index)

train_crs.shape
test_crs.shape

# Create lists of texts and labels 
text =  train_crs['Description']
text = text.astype(str).values.tolist()
len(text)

labels = train_crs['cadrs']
labels = labels.astype(str).values.tolist()

num_words = [len(words.split()) for words in text]
max(num_words)

# Clean up text
regex_ap = re.compile(r"(\b[']\b)|[\W_]")

cleaned_text = []
counter = 0

for stuff in text:
    # remove HTML formatting
    temp = BeautifulSoup(stuff)
    text = temp.get_text()
    text = text.lower()
    # note that we exclude apostrophes from our list of punctuation marks: we want to keep don't, shouldn't etc.
    text = re.sub(r'[()\[\]{}.,;:!?\<=>?@^_`~#$%"&*-]', ' ', text)
    # remove apostrophes that are not intra-word
    text = regex_ap.sub(lambda x: (x.group(1) if x.group(1) else ' '), text)
    # strip extra white space
    text = re.sub(' +',' ',text)
    # strip leading and trailing white space
    text = text.strip()
    # tokenize
    tokens = text.split()
    # remove single letter tokens (we don't remove stopwords as some of them might be useful in determining polarity, like not, but...)
    tokens = [tok for tok in tokens if len(tok)>1]
    # POS tag
    #tagged_tokens = pos_tag(tokens)
    # convert to lower case words that are not identified as proper nouns
    #tokens = [token.lower() if tagged_tokens[idx][1]!='NNP' else token for idx,token in enumerate(tokens)]
    # save
    cleaned_text.append(tokens)

# get list of tokens from all descriptions
all_tokens = [token for sublist in cleaned_text for token in sublist]

counts = dict(Counter(all_tokens))

sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)


# assign to each word an index based on its frequency in the corpus
# the most frequent word will get index equal to 1

word_to_index = dict([(tuple[0],idx+1) for idx, tuple in enumerate(sorted_counts)])
####
# save dictionary
with open(path_to_cadrs + 'word_to_index.json', 'w') as my_file:
    json.dump(word_to_index, my_file, sort_keys=True, indent=4)

#####
cleaned_text_integers = []
counter = 0

for txt in cleaned_text:
    sublist = []
    for token in txt:
        sublist.append(word_to_index[token])
    cleaned_text_integers.append(sublist)
    if counter % 1e4 == 0:
        print(counter, '/', len(text), 'reviews cleaned')
    counter += 1

# Split for validation CHECK WITH PREVIOUS IMPLEMENTATION 
len(cleaned_text_integers)

idx_split = 240
training_text = cleaned_text_integers[:idx_split]
training_labels = labels[:idx_split]

test_text = cleaned_text_integers[idx_split:]
test_labels = labels[idx_split:]

random.seed(3272017)
shuffle_train = random.sample(range(len(training_text)), len(training_text))
shuffle_test = random.sample(range(len(test_text)), len(test_text))

x_train = [training_text[shuffle_train[elt]] for elt in shuffle_train]
y_train = [training_labels[shuffle_train[elt]] for elt in shuffle_train]

x_test = [test_text[shuffle_test[elt]] for elt in shuffle_test]
y_test = [test_labels[shuffle_test[elt]] for elt in shuffle_test]

x_train = [[int(elt) for elt in sublist] for sublist in x_train]
x_test = [[int(elt) for elt in sublist] for sublist in x_test] 

y_train = [int(elt) for elt in y_train]
y_test = [int(elt) for elt in y_test]
### look at the text 
index_to_word = dict((v,k) for k, v in word_to_index.items())
print (' '.join([index_to_word[elt] for elt in x_train[4]]))

stpwds = [index_to_word[idx] for idx in range(1,stpwd_thd)]

x_train = [[elt for elt in rev if elt<=max_features and elt>=stpwd_thd] for rev in x_train]
x_test =  [[elt for elt in rev if elt<=max_features and elt>=stpwd_thd] for rev in x_test]

x_train = [rev[:max_size] for rev in x_train]
x_test = [rev[:max_size] for rev in x_test]


x_train = [rev+[0]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_train]

# sanity check: all reviews should now be of size 'max_size'
assert(max_size == list(set([len(rev) for rev in x_train]))[0]),"1st sanity check failed!"

print('padding',len([elt for elt in x_test if len(elt)<max_size]),'reviews from the test set')

x_test = [rev+[0]*(max_size-len(rev)) if len(rev)<max_size else rev for rev in x_test]

# sanity check
assert(max_size == list(set([len(rev) for rev in x_test]))[0]),"2nd sanity check failed!"
###
# convert integer reviews into word reviews
x_full = x_train + x_test
x_full_words = [[index_to_word[idx] for idx in rev if idx!=0] for rev in x_full]
all_words = [word for rev in x_full_words for word in rev]

print(len(all_words),'words')
print(len(list(set(all_words))),'unique words')

if use_pretrained:

    # initialize word vectors
    word_vectors = Word2Vec(size=word_vector_dim, min_count=1)

    # create entries for the words in our vocabulary
    word_vectors.build_vocab(x_full_words)

    # sanity check
    assert(len(list(set(all_words))) == len(word_vectors.wv.vocab)),"3rd sanity check failed!"

    # fill entries with the pre-trained word vectors
    word_vectors.intersect_word2vec_format(path_to_pretrained_wv + 'GoogleNews-vectors-negative300.bin', binary=True)

    print('pre-trained word vectors loaded')

    norms = [np.linalg.norm(word_vectors[word]) for word in list(word_vectors.wv.vocab)] # in Python 2.7: word_vectors.wv.vocab.keys()
    idxs_zero_norms = [idx for idx, norm in enumerate(norms) if norm<0.05]
    no_entry_words = [list(word_vectors.wv.vocab)[idx] for idx in idxs_zero_norms]
    print('# of vocab words w/o a Google News entry:',len(no_entry_words))

    # create numpy array of embeddings  
    embeddings = np.zeros((max_features + 1,word_vector_dim))
    
    for word in list(word_vectors.wv.vocab):
        idx = word_to_index[word]
        # word_to_index is 1-based! the 0-th row, used for padding, stays at zero
        embeddings[idx,] = word_vectors[word]
        
    print('Found %s word vectors.' % len(embeddings))

else:
    print('not using pre-trained embeddings')

from keras.utils import to_categorical

labels = to_categorical(np.asarray(labels))

print('Shape of label tensor:', labels.shape)

## Model 
my_input = Input(shape=(max_size,)) # we leave the 2nd argument of shape blank because the Embedding layer cannot accept an input_shape argument

if use_pretrained:
    embedding = Embedding(input_dim=embeddings.shape[0], # vocab size, including the 0-th word used for padding
                          output_dim=word_vector_dim,
                          weights=[embeddings], # we pass our pre-trained embeddings
                          input_length=max_size,
                          trainable=not do_static,
                          ) (my_input)
else:
    embedding = Embedding(input_dim=max_features + 1,
                          output_dim=word_vector_dim,
                          trainable=not do_static,
                          ) (my_input)

embedding_dropped = Dropout(drop_rate)(embedding)

# feature map size should be equal to max_size-filter_size+1
# tensor shape after conv layer should be (feature map size, nb_filters)
print('branch A:',nb_filters,'feature maps of size',max_size-filter_size_a+1)
print('branch B:',nb_filters,'feature maps of size',max_size-filter_size_b+1)

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
prob = Dense(units = 1, # dimensionality of the output space
             activation = 'sigmoid',
             )(concat_dropped)

model = Model(my_input, prob)

model.compile(loss='binary_crossentropy',
              optimizer = my_optimizer,
              metrics = ['accuracy'])

####
# in test mode, we should set the 'learning_phase' flag to 0 (we don't want to use dropout)
get_doc_embedding = K.function([model.layers[0].input,K.learning_phase()],
                               [model.layers[9].output])

n_plot = 100

doc_emb = get_doc_embedding([np.array(x_test[:n_plot]),0])[0]

my_pca = PCA(n_components=10)
my_tsne = TSNE(n_components=2,perplexity=10) #https://lvdmaaten.github.io/tsne/
doc_emb_pca = my_pca.fit_transform(doc_emb) 
doc_emb_tsne = my_tsne.fit_transform(doc_emb_pca)

labels_plt = y_test[:n_plot]
my_colors = ['blue','red']

fig, ax = plt.subplots()

for label in list(set(labels_plt)):
    idxs = [idx for idx,elt in enumerate(labels_plt) if elt==label]
    ax.scatter(doc_emb_tsne[idxs,0], 
               doc_emb_tsne[idxs,1], 
               c = my_colors[label],
               label=str(label),
               alpha=0.7,
               s=10)

ax.legend(scatterpoints=1)
fig.suptitle('t-SNE visualization of CNN-based doc embeddings \n (first 100 courses from test set)',fontsize=10)
fig.set_size_inches(6,4)
#########
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
          validation_data = (x_test, y_test),
          callbacks = [early_stopping,checkpointer])

y_train.shape