from Singleton import Singleton
from gensim.models import KeyedVectors
import gensim
import re
import numpy as np
import nltk
from string import punctuation

@Singleton
class w2v:
    def __init__(self):
        path = '/home/maxtelll/Documents/arquivos/ic_novo_backup/posicionamento/embedding/cbow_s50.txt'
        self.model = KeyedVectors.load_word2vec_format(path, unicode_errors="ignore")
        self.model = self.model.wv
        self.model.init_sims(replace=True)
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

    ### Defining Word averaging
    def word_averaging(self, words):
        all_words, mean = set(), []

        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            elif word in self.model.vocab:
                mean.append(self.model.syn0norm[self.model.vocab[word].index])
                all_words.add(self.model.vocab[word].index)

        if not mean:
            return np.zeros(50)

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def fit_transform(self, train_x, y):
        return np.vstack([self.word_averaging(self.myTokenizer(i)) for i in train_x])

    def transform(self, test_x):
        return np.vstack([self.word_averaging(self.myTokenizer(i)) for i in test_x])
