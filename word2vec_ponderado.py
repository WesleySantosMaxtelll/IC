from Singleton import Singleton
from gensim.models import KeyedVectors
import gensim
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import SelectKBest
import nltk
from string import punctuation

@Singleton
class w2v_ponderado:
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
                mean.append([e*self.final_dict.get(word, 0) for e in self.model.syn0norm[self.model.vocab[word].index]])
                all_words.add(self.model.vocab[word].index)

        if not mean:
            return np.zeros(50)

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

    def tfidf_peso(self, docs):
        tfidf_model = TfidfVectorizer(min_df=2, tokenizer=self.myTokenizer, preprocessor=None)
        tfidf_model.fit(docs)
        palavras = tfidf_model.get_feature_names()
        pesos = tfidf_model.idf_
        self.final_dict = {}
        for i in range(len(palavras)):
            self.final_dict[palavras[i]] = pesos[i]

    def fit_transform(self, train_x, train_y):
        self.tfidf_peso(train_x)
        X = np.vstack([self.word_averaging(self.myTokenizer(i)) for i in train_x])
        return X

    def transform(self, test_x):
        X = np.vstack([self.word_averaging(self.myTokenizer(i)) for i in test_x])
        return X
