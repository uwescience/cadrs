import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = '/home/joseh/data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'courses_ospi_b.csv')

crs_names =  pd.read_csv(TEXT_DATA_DIR, delimiter = ',')
print('The shape: %d x %d' % crs_names.shape)
crs_names.columns

crs_names.shape
crs_names.head

train_crs = crs_names.sample(frac=0.8,random_state=200)
test_crs = crs_names.drop(train_crs.index)

train_crs.shape
test_crs.shape
# second, prepare text samples and their labels MY DATA
titles = train_crs['crs_copy']
titles = titles.str.replace('-', ' ')
titles = titles.str.replace('/', ' ')
titles = titles.str.replace('_', ' ')
titles = titles.str.replace('.', '')
titles = titles.str.replace('^\d+\s+', ' ')
# get rid of "intro" and "intro to"
titles = titles.str.replace('intro to ', '')
titles = titles.str.replace('intro ', '')
texts = titles.astype(str).values.tolist() # list of text samples

len(texts)
print(*texts, sep=",")

num_words = [len(words.split()) for words in texts]

max(num_words)

subject = train_crs['ospi_sub'].str.replace(',', '')
subject = subject.str.replace(' ', '.')
subject = subject.str.replace('-', ' ')
subject = subject.str.replace('/', '.')
subject = subject.str.lower()
subject = subject.factorize()
type(subject)

label_names = list(subject[1])
label_values = list(range(0,31))
len(label_values)
len(label_names)

labels_index = dict(zip(label_names, label_values))
	
labels_index


labels = list(subject[0])  # list of label ids
len(labels)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

VALIDATION_SPLIT = 0.2

maxlen = max(num_words) + 1  # max number of words in a title to consider

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index # word and their token # ordered by most frequent
print('Found %s unique tokens.' % len(word_index))

max_words = len(word_index) + 1 # total words of vocabulary we will consider

data = pad_sequences(sequences, maxlen=maxlen)
data.shape
np.mean(data > 0)

from keras.utils import to_categorical

labels = to_categorical(np.asarray(labels))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# Split the data into a training set and a validation set
# But first, shuffle the data, since we started from data
# where sample are ordered (all negative first, then all positive).
# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

##############################################
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

## first remove what i don't have 

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

preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))

from numpy import argmax

y_pred = model.predict(x_val, verbose=1)
y_pred

class_y = np.argmax(y_pred,axis=1)
class_y

y_val_test = argmax(y_val, axis=1) 
y_val_test.shape


pred_out = pd.DataFrame(y_pred, columns=[*labels_index])

pred_out.head()
pred_out.shape

pred_out['Y'] = y_val_test
pred_out['pred_class'] = class_y

#word_index = tokenizer.word_index 

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

# sequences = tokenizer.texts_to_sequences(texts)
x_val.shape
x_val2 = [np.trim_zeros(a, 'f') for a in x_val]
x_val2
# Creating texts 
my_texts = list(map(sequence_to_text, x_val2))
len(my_texts)

my_texts_str = pd.DataFrame(my_texts, columns=['a', 'b', 'c','d','e','f','g','h'])

test = pd.concat([pred_out, my_texts_str], axis=1)
test.head
#pred_out.to_csv('/home/joseh/data/cnn_result_b.csv', encoding='utf-8', index=False)

test.to_csv('/home/joseh/data/cnn_result_names4.csv', encoding='utf-8', index=False)

labels_index