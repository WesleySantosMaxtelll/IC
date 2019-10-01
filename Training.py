import sys

from data_getter import Getter
from representations import Representations
from Models import Models
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pickle
from collections import Counter

from joblib import dump, load
# import nltk
# nltk.download('stopwords')
#

def train_model(train_x, train_y, test_x, test_y, label, classifier_name, representation_name, corrector, k):
    output = 'Classificando {} com representacao: {} e classificacao: {} e data base {}\n'.format(label, representation_name, classifier_name, corrector)
    output+= 'k equals to {}\n\n'.format(k)
    train_x, train_y, test_x, test_y = Representations().get_representation(
                    rep=representation_name, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, k=k, cat=label)

    if corrector == 'undersampled':
        rus = RandomUnderSampler(random_state=2)
        # rus.fit(X, Y)
        # print(len(train_y))
        # print(len(train_x))
        train_x, train_y = rus.fit_resample(train_x, train_y)
    if corrector == 'oversampled':
        # sm = SMOTE(random_state=42)
        # train_x, train_y = sm.fit_resample(train_x, train_y)
        rus = RandomOverSampler(random_state=2)
        # rus.fit(X, Y)
        # print(len(train_y))
        # print(len(train_x))
        train_x, train_y = rus.fit_resample(train_x, train_y)

    # print(Counter(train_y))

    cls = Models().get_classifier(classifier_name, len(train_y))

    # print(len(train_x))
    # print(test_x)
    cls.fit(train_x, train_y)

    guardar = '/home/maxtelll/Downloads/get-started-python/modelos/posicionamento/'
    dump(cls, guardar+'cotas_classifier.joblib')

    # classifier = load(guardar+'pena_classifier.joblib')
    # print(output)
    # print('Numero de layers {}'.format(classifier.n_layers_))
    # return
    pred = cls.predict(test_x)
    # print(pred)
    # print(len(train_y))
    # print(type(pred))
    # print(type(test_y))
    # print(Counter(test_y))
    output+= classification_report(test_y, pred, target_names=['Contra', 'A favor'])
    print(output)
    # joblib.dump(to_persist, filename)
    # output_file(output, classifier_name, representation_name, label, corrector)


def output_file(output, classifier_name, representation_name, category, corrector):
    path = '/home/maxtelll/Desktop/ic_novo_backup/posicionamento/resultados/ternario/{}/{}/{}'.format(
        classifier_name, representation_name, category)
    file = category+'-'+representation_name+'-'+classifier_name+'-'+corrector+'.txt'
    if not os.path.exists(path):
        os.mkdir(path)
    f = open(path+'/'+file, 'w+')
    f.write(output)
    f.close()

def polaridade_BRmoral(cat):
    path = '/home/maxtelll/Documents/arquivos/ic_novo_backup/posicionamento/BRmoral/'
    x_train = path + cat + '/' + cat + '-X-train.pkl'
    x_test = path + cat + '/' + cat + '-X-test.pkl'
    y_train = path + cat + '/' + cat + '-Y-train.pkl'
    y_test = path + cat + '/' + cat + '-Y-test.pkl'

    with open(x_train, 'rb') as f:
        x_train = pickle.load(f)

    with open(y_train, 'rb') as f:
        y_train = pickle.load(f)

    with open(x_test, 'rb') as f:
        x_test = pickle.load(f)

    with open(y_test, 'rb') as f:
        y_test = pickle.load(f)
    # print(y_train)

    x_treino, y_treino = [], []
    for x, y in zip(x_train, y_train):
        if y== 'for':
            x_treino.append(x)
            y_treino.append(2)
        elif y == 'against':
            x_treino.append(x)
            y_treino.append(1)

    x_val, y_val = [], []
    for x, y in zip(x_test, y_test):
        if y == 'for':
            x_val.append(x)
            y_val.append(2)
        elif y == 'against':
            x_val.append(x)
            y_val.append(1)

    t = y_treino + y_val
    # print(t)
    from collections import Counter
    print(Counter(t))

    return x_treino, y_treino, x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(x_treino, y_treino, test_size=0.2, random_state=42)

    # from collections import Counter
    # print(Counter(y_teste))
    # print(y_test)
    # print(y_teste)
    # print(len(y_train))
    return x_train, y_train, x_test, y_test, x_val, y_val


def detectar_posicionamento(X, Y):
    # print(Y)
    Y = [0 if i == '0' else 1 for i in Y]
    # print(Counter(Y))

    X_train, x_val, Y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, Y_train, [], [], x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test, x_val, y_val


def favoravel_contrario(X, Y):
    x, y = [], []
    for text, tag in zip(X, Y):
        if tag != '0':
            x.append(text)
            y.append(int(tag))
    # print(x)
    print(Counter(y))

    X_train, x_val, Y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # return X_train,Y_train, [], [], x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # print(len(x_train))
    # print(len(y_train))
    # print(len(x_test))
    # print(len(y_test))
    # print(len(x_val))
    # print(len(y_val))

    return x_train, y_train, x_test, y_test, x_val, y_val

def ternario(X, Y):
    x, y = [], []
    for text, tag in zip(X, Y):
        x.append(text)
        y.append(int(tag))
    # print(x)
    print(Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return X_train, y_train, X_test, y_test

# classifiers = ['MLP', 'NB', 'KNN', 'LogReg']
represetations = ['CountVec']
classifiers = ['MLP']

# labels = ['aborto', 'cotas', 'maconha', 'maioridade', 'pena']
# labels = ['pena']
# labels = ['quotas', 'age', 'abortion', 'death', 'drugs']
labels = ['cotas']
correctors = ['oversampled']
import time
tempo = time.time()
nk = {
'quotas':14000,
'age':28000,
'abortion':20000,
'death':9000,
'drugs':44000
}

# k = 10

for label in labels:
    for corr in correctors:
        for k in range(37000, 38000, 1000):
            for c in classifiers:
                for r in represetations:
                    if r == 'word2vec' and c == 'NB':
                        continue
                    # print(r)
                    print(label)
                    # X_train, y_train, x_val, y_val = polaridade_BRmoral(label)
                    X, Y = Getter().get_label(label)
                    # print(label)
                    # X_train, x_val, Y_train, y_val = favoravel_contrario(X, Y)

                    X_train, y_train, X_test, y_test, x_val, y_val = detectar_posicionamento(X, Y)
                    # continue
                    # exit(0)
                    # print(c)
                    # print(r)
                    train_model(X_train, y_train, x_val, y_val, label, c, r, corr, k)

print('Demorou {} segundos'.format(tempo-time.time()))
# X, Y = Getter().get_label('maconha')
# print(X)