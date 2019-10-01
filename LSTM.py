
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from representations import Representations
from data_getter import Getter


def detectar_posicionamento(X, Y):
    # print(Y)
    Y = [0 if i == '0' else 1 for i in Y]
    # print(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


def favoravel_contrario(X, Y):
    x, y = [], []
    for text, tag in zip(X, Y):
        if tag != '0':
            x.append(text)
            y.append(int(tag))
    # print(x)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test


def RNN2():
    model = Sequential()
    model.add(LSTM(30, input_shape=(15000, ), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(len(labels), activation='softmax'))
    return model

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(20)(layer)
    layer = Dense(5,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


labels = ['maconha', 'cotas', 'pena', 'aborto', 'maioridade']
for l in labels:
    print('\n\n Classificando {}\n\n\n'.format(l))
    X, Y = Getter().get_label(l)

    X_train, Y_train, X_test, Y_test = favoravel_contrario(X, Y)
    X_train, Y_train, X_test, Y_test = Representations().get_representation(
        rep='selecao', train_x=X_train, train_y=Y_train, test_x=X_test, test_y=Y_test)
    # max_words = 10000
    # max_len = 150
    # tok = Tokenizer(num_words=max_words)
    # tok.fit_on_texts(X_train)
    # sequences = tok.texts_to_sequences(X_train)
    # sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)

    model = RNN2()

    model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # print(Y_train)

    model.fit(X_train, Y_train, batch_size=1000, epochs=200,
              validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
    # test_sequences = tok.texts_to_sequences(X_test)
    # test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)

    accr = model.evaluate(X_test, Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
