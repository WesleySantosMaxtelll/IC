import pandas as pd
import numpy as np
import string
from string import digits
import matplotlib.pyplot as plt
# %matplotlib inline
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
import data_getter
from math import ceil
from rouge import Rouge 

batch_size = 200
epochs = 100
num_textos = 10000


raw_data = data_getter.get_files(num_textos)
input_texts, target_texts = [], []
for i in raw_data:
    input_texts.append(i[0])
    target_texts.append(i[1])

# lines= pd.read_table('./conversa/mar.txt', names=['eng', 'mar'])
# print(lines.mar)
input_texts, target_texts = (list(map(lambda x: x.lower(), input_texts)), 
                             list(map(lambda x: x.lower(), target_texts)))
input_texts, target_texts = (list(map(lambda x: re.sub("'", '', x), input_texts)),
                             list(map(lambda x: re.sub("'", '', x), target_texts)))
input_texts, target_texts = (list(map(lambda x: re.sub('"', '', x), input_texts)),
                             list(map(lambda x: re.sub('"', '', x), target_texts)))

exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
input_texts=list(map(lambda x: ''.join(ch for ch in x if ch not in exclude), input_texts))
target_texts=list(map(lambda x: ''.join(ch for ch in x if ch not in exclude), target_texts))
remove_digits = str.maketrans('', '', digits)
input_texts=list(map(lambda x: x.translate(remove_digits), input_texts))
target_texts=list(map(lambda x: x.translate(remove_digits), target_texts))

# Remove extra spaces
input_texts=list(map(lambda x: x.strip(), input_texts))
target_texts=list(map(lambda x: x.strip(), target_texts))
input_texts=list(map(lambda x: re.sub(" +", " ", x), input_texts))
target_texts=list(map(lambda x: re.sub(" +", " ", x), target_texts))

# Add start and end tokens to target sequences
target_texts=list(map(lambda x: 'START_ '+ x + ' _END', target_texts))

# for i, j in zip(input_texts, target_texts):
#     print('{}\t{}'.format(j, i))



# Vocabulary of Input
all_input_words=set()
for eng in input_texts:
    for word in eng.split():
        if word not in all_input_words:
            all_input_words.add(word)

# Vocabulary of Target
all_target_words=set()
for mar in target_texts:
    for word in mar.split():
        if word not in all_target_words:
            all_target_words.add(word)

# Max Length of source sequence
lenght_list=[]
for l in input_texts:
    lenght_list.append(len(l.split(' ')))
max_length_src = np.max(lenght_list)
print('Maior texto de entrada: {}'.format(max_length_src))

# Max Length of target sequence
lenght_list=[]
for l in target_texts:
    lenght_list.append(len(l.split(' ')))
max_length_tar = np.max(lenght_list)
print('Maior texto de saida: {}'.format(max_length_tar))

input_words = sorted(list(all_input_words))
target_words = sorted(list(all_target_words))
num_encoder_tokens = len(all_input_words)
num_decoder_tokens = len(all_target_words)
print('Encoder tokes: {}\nDecoder tokes: {}'.format(num_encoder_tokens, num_decoder_tokens))
num_decoder_tokens += 1
num_encoder_tokens+=1

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())

# Train - Test Split
X, y = input_texts, target_texts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
# print('{} {} {} {}'.format(len(X_train), len(y_train), len(X_test), len(y_test)))
# X_train.to_pickle('Weights_Mar/X_train.pkl')
# X_test.to_pickle('Weights_Mar/X_test.pkl')

def generate_batch(X = X_train, y = y_train, batch_size = 128):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros((batch_size, max_length_src),dtype='float32')
            decoder_input_data = np.zeros((batch_size, max_length_tar),dtype='float32')
            decoder_target_data = np.zeros((batch_size, max_length_tar, num_decoder_tokens),dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = input_token_index[word] # encoder input seq
                for t, word in enumerate(target_text.split()):
                    if t<len(target_text.split())-1:
                        decoder_input_data[i, t] = target_token_index[word] # decoder input seq
                    if t>0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1, target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


def rouge_metric(y_true, y_pred):
    rouge = Rouge()
    scores = rouge.get_scores(y_pred, y_true, avg=True)
    return scores

latent_dim = 50
latent_dims = [512, 256, 128, 64, 32]

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
# encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)

outputs = enc_emb
encoder_states = []
for j in range(len(latent_dims))[::-1]:
    outputs, h, c = LSTM(latent_dims[j], return_state=True, return_sequences=bool(j), dropout=0.2)(outputs)
    encoder_states += [h, c]


# Set up the decoder, setting the initial state of each layer to the state of the layer in the encoder
# which is it's mirror (so for encoder: a->b->c, you'd have decoder initial states: c->b->a).

decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

# decoder_inputs = Input(shape=(None, num_decoder_tokens))

outputs = dec_emb
output_layers = []
for j in range(len(latent_dims)):
    output_layers.append(
        LSTM(latent_dims[len(latent_dims) - j - 1], return_sequences=True, return_state=True, dropout=0.2)
    )
    outputs, dh, dc = output_layers[-1](outputs, initial_state=encoder_states[2*j:2*(j+1)])


decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(outputs)
#
# # Set up the decoder, using `encoder_states` as initial state.
# decoder_inputs = Input(shape=(None,))
# dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
# dec_emb = dec_emb_layer(decoder_inputs)
# # We set up our decoder to return full output sequences,
# # and to return internal states as well. We don't use the
#
# out_layer1 = LSTM(latent_dim, return_sequences=True, return_state=True)
# d_outputs, dh1, dc1 = out_layer1(dec_emb,initial_state= [state_h, state_c])
# out_layer2 = LSTM(latent_dim, return_sequences=True, return_state=True)
# final, dh2, dc2 = out_layer2(d_outputs, initial_state= [h2, c2])
# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(final)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# from IPython.display import Image
# Image(retina=True, filename='train_model.png')

train_samples = len(X_train)
val_samples = len(X_test)

model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                    steps_per_epoch = ceil(train_samples/batch_size),
                    epochs=epochs,
                    validation_data = generate_batch(X_test, y_test, batch_size = batch_size),
                    validation_steps = ceil(val_samples/batch_size))

model.save_weights('modelos/{}-{}-nmt_weights.h5'.format(num_textos,epochs))
model.load_weights('modelos/{}-{}-nmt_weights.h5'.format(num_textos,epochs))


# Define sampling models (modified for n-layer deep network)
encoder_model = Model(encoder_inputs, encoder_states)


d_outputs = dec_emb
decoder_states_inputs = []
decoder_states = []
for j in range(len(latent_dims))[::-1]:
    current_state_inputs = [Input(shape=(latent_dims[j],)) for _ in range(2)]

    temp = output_layers[len(latent_dims)-j-1](d_outputs, initial_state=current_state_inputs)

    d_outputs, cur_states = temp[0], temp[1:]

    decoder_states += cur_states
    decoder_states_inputs += current_state_inputs

decoder_outputs = decoder_dense(d_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        # print(target_seq)
        # print('----------------------------')
        # print([target_seq] + states_value)
        upt = decoder_model.predict([target_seq] + states_value)
        output_tokens = upt[0]
        # print(upt)
        # output_tokens, h, c, h1, c1 = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += ' '+sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
           len(decoded_sentence) > 80):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = upt[1:]

    return decoded_sentence

train_gen = generate_batch(X_train, y_train, batch_size = 1)
# print(list(train_gen))
k=-1
# print(k)

f= open("saidas/gerador_words-{}-{}.txt".format(num_textos,epochs),"w+")
pred_text = []
original_text = []
saida = "["
for _ in range(len(X_test)):
    # print(next(train_gen))
    k += 1
    (input_seq, actual_output), _ = next(train_gen)
    saida +="{"
    decoded_sentence = decode_sequence(input_seq)
    saida += "'reportagem' : '"+str(X_test[k:k+1][0])+"',"
    saida+= "'original': '"+str(y_test[k:k+1][0])[6:-4]+"',"
    original_text.append(str(y_test[k:k+1][0])[6:-4])
    saida+= "'gerado': '"+str(decoded_sentence)[:-4]+"'}"
    pred_text.append(decoded_sentence)
    saida+= ', '
    # print(saida)
    # f.write(saida)

rouge_value = rouge_metric(original_text, pred_text)
saida+=str(rouge_value)+"]"
f.write(saida)
print(rouge_value)
f.close()