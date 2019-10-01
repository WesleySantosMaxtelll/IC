import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


file = open("/home/maxtelll/Documents/arquivos/frank/frank.txt").read()

def tokenize_words(input):
    # lowercase everything to standardize it
    input = input.lower()

    # instantiate the tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

    # if the created token isn't in the stop words, make it part of "filtered"
    filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    # return " ".join(filtered)
    return list(filtered)
# a = 'the Case of house member tOnight'


processed_inputs = tokenize_words(file)
# print(processed_inputs)

chars = sorted(list(set(processed_inputs)))
print(chars)
# chars = sorted(processed_inputs)
# print(chars)
char_to_num = dict((c, i) for i, c in enumerate(chars))
print(char_to_num)

input_len = len(processed_inputs)
vocab_len = len(chars)
print ("Total number of words:", input_len)
print ("Total vocab:", vocab_len)

seq_length = 100
x_data = []
y_data = []



# loop through inputs, start at the beginning and go until we hit
# the final character we can create a sequence out of
for i in range(0, input_len - seq_length, 1):
    # Define input and output sequences
    # Input is the current character plus desired sequence length
    in_seq = processed_inputs[i:i + seq_length]

    # Out sequence is the initial character plus total sequence length
    out_seq = processed_inputs[i + seq_length]

    # We now convert list of characters to integers based on
    # previously and add the values to our lists
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns = len(x_data)
print("Total Patterns:", n_patterns)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y_data)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model_weights_words_saved.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
#

model.fit(X, y, epochs=4, batch_size=256, callbacks=desired_callbacks)


filename = "model_weights_words_saved.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))
import random
numpy.random.seed(random.randint(12414, 42352))
# print(pattern)
for i in range(100):
    texto = input('Digita ai: ')
    # start = numpy.random.randint(0, len(x_data) - 1)
    # pattern = x_data[start]
    # x = numpy.reshape(pattern, (1, len(pattern), 1))
    # x = x / float(vocab_len)
    x = [char_to_num[i] for i in texto.lower().split()]
    x+=[0 for _ in range(seq_length-len(x))]
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]
    seq_in = [num_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# start = numpy.random.randint(0, len(x_data) - 1)
# pattern = x_data[start]
# print("Random Seed:")
# print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
#


