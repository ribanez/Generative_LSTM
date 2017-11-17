'''Example script to generate text from different writters.'''

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

def clear_file(path_in, path_out):
    ## Delete empty lines in original file
    try:
        os.path.isfile(path_out)
        print("File clear exist")
    except :
        print("File clear don't exist")
    else:
        print("Creating File Clear")
        with open(path_in) as infile, open(path_out, "w") as outfile:
            for line in infile:
                if not line.strip(): continue  # skip the empty line
                outfile.write(line)  # non-empty line. Write it to output

writter = 'nicanor'
path_in = './Documents/' + writter + '.txt'
path_out = './Documents/' + writter + '_clear.txt'

## Delete empty lines in original file
clear_file(path_in, path_out)

## Read file
path = path_out
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Build the model: a single LSTM
n_block = 256
drop_out = 0.2
print('Build model...')
model = Sequential()
model.add(LSTM(n_block, input_shape=(maxlen, len(chars)), return_sequences=True))
model.add(Dropout(drop_out))
model.add(LSTM(n_block, return_sequences=True))
model.add(Dropout(drop_out))
model.add(LSTM(n_block))
model.add(Dropout(drop_out))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

#optimizer = RMSprop(lr=0.01)
#model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model
batch_size = 64
n_epochs = 50

print()
print('-' * 50)
filepath="./Models/weights_3layers-"+writter+"-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

historial = model.fit(x, y,
          batch_size=batch_size,
          epochs=n_epochs,
          callbacks=callbacks_list)

plt.plot(historial.history["loss"])
plt.savefig("./Figures/"+writter+"_loss.png")

start_index = random.randint(0, len(text) - maxlen - 1)

## Generate text with different diversity
for diversity in [0.2, 0.5, 1.0, 1.2]:
    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
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


