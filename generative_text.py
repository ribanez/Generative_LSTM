'''Script to generate text from Model.'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import os

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

writter = 'nicanor'
path_weight = "./Models/weights-"+writter+"-50-0.6235.hdf5"

## Read file
path = './Documents/' + writter + '_clear.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# Build the model: a single LSTM
n_block = 256
drop_out = 0.2
maxlen = 40
print('Build model...')
model = Sequential()
model.add(LSTM(n_block, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(drop_out))
model.add(LSTM(n_block))
model.add(Dropout(drop_out))
model.add(Dense(len(chars)))

model.load_weights(path_weight)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Generate Text
batch_size = 64
diversity = 1.0

while(True):
    print('-'*50)
    generated = ''
    sentence = input('Ingrese texto: ')
    if sentence == 'exit' or sentence == 'exit()': break

    generated += sentence.lower()
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    for i in range(batch_size):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

print("-"*50)
