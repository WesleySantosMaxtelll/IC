import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
import nltk
from sklearn.model_selection import train_test_split
# from string import punctuation
from joblib import dump, load
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from selecao_de_atributos import Selecao
import os
import pickle

# guardar = '/home/maxtelll/Documents/arquivos/site/modelos/posicionamento'
# a = os.listdir(guardar)
# print(a)
#
# for i in a:
#     cls = load(guardar+i)
#     # filehandler = open(guardar + i.split('.')[0]+'.p', "wb")
#     # pickle.dump(cls, filehandler)
#     # filehandler.close()
#     dump(cls, guardar + i)



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
    # from collections import Counter
    # print(Counter(t))

    return x_treino, y_treino, x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(x_treino, y_treino, test_size=0.2, random_state=42)

    # from collections import Counter
    # print(Counter(y_teste))
    # print(y_test)
    # print(y_teste)
    # print(len(y_train))
    return x_train, y_train, x_test, y_test, x_val, y_val
#
#
#
# stopwords = nltk.corpus.stopwords.words('portuguese')
# pont = list(punctuation)
# pont.append(' ')
# resp = ' |\n|!|"|#|$|%|&|\'|\(|\)|\*|\+|\,|\-|\.|/|:|;|<|=|>|\?|@|[|\|]|^|_|`|{|||}|~'
#
# # print('resp')
# # print(resp)
# proibidos = stopwords + pont
#
# def myTokenizer(self, texto):
#     words = []
#     global proibidos
#     for w in re.split(self.resp, texto):
#         if len(w) > 0 and w not in proibidos:
#             words.append(w.lower())
#     return words
#


def detectar_posicionamento(X, Y):
    # print(Y)
    Y = [0 if i == '0' else 1 for i in Y]
    # print(Counter(Y))

    X_train, x_val, Y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, Y_train, [], [], x_val, y_val
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test, x_val, y_val
from data_getter import Getter
def teste():
    X, Y = Getter().get_label('cotas')
    # print(label)
    # X_train, x_val, Y_train, y_val = favoravel_contrario(X, Y)

    X_train, y_train, X_test, y_test, x_val, y_val = detectar_posicionamento(X, Y)
    # X_train, y_train, x_val, y_val = polaridade_BRmoral('quotas')
    # print(X_train)
    # rus = RandomOverSampler(random_state=2)
    # rus.fit(X, Y)
    # print(len(train_y))
    # print(len(train_x))
    # train_x, train_y = rus.fit_resample(X_train, y_train)
    # vec = TfidfVectorizer(min_df=2, stop_words=nltk.corpus.stopwords.words('portuguese'))
    vec = CountVectorizer(analyzer="word", stop_words=nltk.corpus.stopwords.words('portuguese'),
                                   max_features=5000)
    # vec = Selecao(42000, 'a')
    # vec = CountVectorizer(analyzer="char", ngram_range=([3, 16]), tokenizer=None, preprocessor=None,
    #                                max_features=3000)
    vec.fit_transform(X_train, y_train)

    # guardar = '/home/maxtelll/Documents/arquivos/site/modelos/'
    guardar = '/home/maxtelll/Downloads/get-started-python/modelos/posicionamento/'
    dump(vec, guardar + 'cotas_r.joblib')

teste()