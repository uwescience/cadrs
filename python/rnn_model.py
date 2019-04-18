import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = '/home/joseh/data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'courses_cadrs_text.csv') #name changed

crs_names =  pd.read_csv(TEXT_DATA_DIR)
print('The shape: %d x %d' % crs_names.shape)
crs_names.columns

train_crs = crs_names.sample(frac=0.8,random_state=200)
test_crs = crs_names.drop(train_crs.index)

# second, prepare text samples and their labels MY DATA

titles = train_crs['crs_copy'].str.replace(',', '')
titles = titles.str.replace('-', ' ')
titles = titles.str.replace('/', ' ')
titles = titles.str.replace('_', ' ')
titles = titles.str.lower()
texts = list(titles)  # list of text samples
len(texts)

# num_words = [len(sentence.split()) for sentence in texts]
# max(num_words)

subject = train_crs['cadr_sub'].str.replace(',', '')
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


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

#MAX_SEQUENCE_LENGTH = 10
#MAX_NUM_WORDS = 20
#EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

maxlen = 10 # We will cut reviews after 100 words
#training_samples = 200 # We will be training on 200 samples
#validation_samples = 10000 # We will be validating on 10000 samples
max_words = 2000 # We will only consider the top 10,000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

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

glove_dir = '/home/joseh/data/glove.6B'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100 # from the GLOVE embeddings 

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='softmax'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

# model.save_weights('pre_trained_glove_model.h5')

# WITHOUT pRE-TRAINED EMBEDINGS 
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(11, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['acc'])
history = model.fit(x_train, y_train,
                epochs=10,
                batch_size=32,
                validation_data=(x_val, y_val))
####################

from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential

model = Sequential()
model.add(layers.Embedding(max_words, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(11, activation='sigmoid'))

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['acc'])

history = model.fit(x_train, y_train,
epochs=10,
batch_size=128,
validation_split=0.2)

###########

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.initializers import Constant

num_words = min(max_words, len(word_index)) + 1 # how many words total 

# embedding_layer = Embedding(num_words,
#                            embedding_dim,
#                            embeddings_initializer=Constant(embedding_matrix),
#                            input_length=maxlen,
#                            trainable=False)

# train a 1D convnet with global maxpooling
# sequence_input = Input(shape=(maxlen,), dtype='int32')
# embedded_sequences = embedding_layer(sequence_input)

model = Sequential()

model.add(layers.Embedding(num_words,
                            embedding_dim,
                            input_length=maxlen,
                            trainable=False))

model.add(layers.Conv1D(100, 5, activation='relu'))
model.add(layers.MaxPooling1D(5))

#model.add(layers.Conv1D(100, 5, activation='relu'))
#model.add(layers.GlobalMaxPooling1D())

# model.add(layers.Dense(128, activation = 'relu'))

model.add(layers.Dense(len(labels_index), activation='sigmoid')) #kkff
model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
loss='categorical_crossentropy',
metrics=['acc'])

history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
## CNN 
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

model.add(layers.Conv1D(activation="relu", input_shape=(10, 100), filters=1200, kernel_size=5, padding="valid"))

                        

model.add(layers.MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(layers.Dense(len(labels_index), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(x_train, y_train,
epochs=10,
batch_size=32,
validation_split=0.2)


Xnew = [['amer', 'government']]

new_sequences2 = tokenizer.texts_to_sequences(Xnew)
new_sequences2
new_data = pad_sequences(new_sequences2, maxlen=10)
new_data
predictions = model.predict(new_data)
predictions

scoredict = {labels_index: predictions[0][idx] for idx, labels_index in enumerate(labels_index)}
scoredict

sorted(scoredict, key=scoredict.get, reverse=True)[:3]