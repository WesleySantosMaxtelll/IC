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

batch_size = 150
epochs = 500
num_textos = 20000


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




latent_dim = 50

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
_, h2, c2 = LSTM(latent_dim, return_state=True)(encoder_outputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c, h2, c2]






# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
# decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs = decoder_dense(decoder_outputs)



out_layer1 = LSTM(latent_dim, return_sequences=True, return_state=True)
d_outputs, dh1, dc1 = out_layer1(dec_emb,initial_state= [state_h, state_c])
out_layer2 = LSTM(latent_dim, return_sequences=True, return_state=True)
final, dh2, dc2 = out_layer2(d_outputs, initial_state= [h2, c2])
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(final)

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

model.save_weights('{}-{}-nmt_weights.h5'.format(num_textos,epochs))
model.load_weights('{}-{}-nmt_weights.h5'.format(num_textos,epochs))
# Encode the input sequence to get the "thought vectors"
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input_h1 = Input(shape=(latent_dim,))
decoder_state_input_c1 = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,
                            decoder_state_input_h1, decoder_state_input_c1]

dec_emb2= dec_emb_layer(decoder_inputs) # Get the embeddings of the decoder sequence

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs0, state_h1, state_c1 = out_layer1(dec_emb2, initial_state=decoder_states_inputs[:2])
decoder_outputs2, state_h2, state_c2 = out_layer2(decoder_outputs0, initial_state=decoder_states_inputs[-2:])
decoder_states2 = [state_h1, state_c1, state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2) # A dense softmax layer to generate prob dist. over the target vocabulary

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


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
        output_tokens, h, c, h1, c1 = decoder_model.predict([target_seq] + states_value)

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
        states_value = [h, c, h1, c1]

    return decoded_sentence

train_gen = generate_batch(X_train, y_train, batch_size = 1)
# print(list(train_gen))
k=-1
# print(k)

f= open("gerador_words-{}-{}.txt".format(num_textos,epochs),"w+")

for _ in range(len(X_train)):
    # print(next(train_gen))
    k += 1
    (input_seq, actual_output), _ = next(train_gen)

    decoded_sentence = decode_sequence(input_seq)
    saida = 'Reportagem: '+str(X_train[k:k+1][0])+'\n'
    saida+= 'Titulo atual: '+str(y_train[k:k+1][0])[6:-4]+'\n'
    saida+= 'Titulo gerado: '+str(decoded_sentence)[:-4]+'\n'
    saida+= '\n------------------------------------------\n'
    print(saida)
    f.write(saida)


f.close()