'''
#Restore a character-level sequence to sequence model from to generate predictions.
This script loads the ```s2s.h5``` model saved by [lstm_seq2seq.py
](/examples/lstm_seq2seq/) and generates sequences from it. It assumes
that no changes have been made (for example: ```latent_dim``` is unchanged,
and the input data and model architecture are unchanged).
See [lstm_seq2seq.py](/examples/lstm_seq2seq/) for more details on the
model architecture and how it is trained.
'''
from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import data_getter
from math import ceil
import re
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

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

# Restore the model and construct the encoder and decoder.
model = load_model('{}-{}-nmt_weights.h5'.format(num_textos,epochs))

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# Decodes an input sequence.  Future work should support beam search.
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)