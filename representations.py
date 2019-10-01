import nltk as nltk
from string import punctuation
import numpy as np
import nltk
from w2vClass import w2v
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re
from selecao_de_atributos import Selecao
from word2vec_ponderado import w2v_ponderado
from joblib import dump, load

class Representations:


    def __init__(self):

        stopwords = nltk.corpus.stopwords.words('portuguese')
        pont = list(punctuation)
        pont.append(' ')
        self.resp = ' |\n|!|"|#|$|%|&|\'|\(|\)|\*|\+|\,|\-|\.|/|:|;|<|=|>|\?|@|[|\|]|^|_|`|{|||}|~'

        # print('resp')
        # print(resp)
        self.proibidos = stopwords + pont


    def myTokenizer(self, s):
        words = []
        for w in re.split(self.resp, s):
            if len(w) > 0 and w not in self.proibidos:
                words.append(w.lower())
        return words

    def Create_Vectorizer(self, name, k, cat):

        if (name == 'CountVec'):
            return CountVectorizer(analyzer="word", stop_words=nltk.corpus.stopwords.words('portuguese'),
                                   max_features=5000)
        elif (name == 'NGram'):
            return CountVectorizer(analyzer="char", ngram_range=([3, 16]), tokenizer=None, preprocessor=None,
                                   max_features=3000)
        elif (name == 'TFidf'):
            return TfidfVectorizer(min_df=2, stop_words=nltk.corpus.stopwords.words('portuguese'))

        elif (name == 'word2vec'):
            return w2v.instance()

        elif (name == 'word2vec_ponderado'):
            return w2v_ponderado.instance()

        elif (name == 'selecao'):
            return Selecao(k, cat)
        else:
            raise NameError('Vectorizer not found')

    def get_representation(self, rep, train_x, train_y, test_x, test_y, k, cat):
        vec = self.Create_Vectorizer(rep, k, cat)

        X_train = vec.fit_transform(train_x, train_y)
        Y_train = np.array(train_y)
        # vec.mostre_melhores()
        X_test = vec.transform(test_x)
        Y_test = np.array(test_y)
        # print(vec)
        guardar = '/home/maxtelll/Documents/arquivos/site/modelos/'
        # dump(vec, guardar + 'pena_representation.joblib')

        # classifier = load(guardar + 'aborto_classifier.joblib')

        return X_train, Y_train, X_test, Y_test
