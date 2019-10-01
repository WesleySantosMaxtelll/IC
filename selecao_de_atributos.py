import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegressionCV
import sklearn
from string import punctuation
import re

class Selecao:

    def __init__(self, k, cat):
        pont = list(punctuation)
        pont.append(' ')
        self.cat = cat
        self.resp = ' |\n|!|"|#|$|%|&|\'|\(|\)|\*|\+|\,|\-|\.|/|:|;|<|=|>|\?|@|[|\|]|^|_|`|{|||}|~'

        # print('resp')
        # print(resp)
        stopwords = nltk.corpus.stopwords.words('portuguese')
        self.proibidos = stopwords + pont
        self.vect = TfidfVectorizer(analyzer="char", ngram_range=([1, 25]), tokenizer=None, preprocessor=None, max_features=80000)
            # TfidfVectorizer(ngram_range=[1, 6])
        # x_tfidf = vect.fit_transform(texts.data)


        kvalue = k
        self.tfidf_kbest_selector = SelectKBest(k=kvalue)
        # tfidf_kbest_fit = tfidf_kbest_selector.fit(x_tfidf, texts.target)
        # x_tfidf_kbest = tfidf_kbest_fit.transform(x_tfidf)

    def myTokenizer(self, s):
        words = []
        for w in re.split(self.resp, s):
            if len(w) > 0 and w not in self.proibidos:
                words.append(w.lower())
        return words
    def mostre_melhores(self):
        print('\n\n\nLista de palavras selecionadas')
        feat = self.vect.get_feature_names()
        # self.tfidf_kbest_selector
        lista = self.tfidf_kbest_selector.get_support()
        saida = open('/home/maxtelll/Desktop/ic_novo_backup/posicionamento/'+self.cat+'.txt', 'w+')
        for l, f in zip(lista, feat):
            if l:
                saida.write(f+'\n')

    def fit_transform(self, X, Y):
        X = self.vect.fit_transform(X)
        # print(X)
        self.tfidf_kbest_selector.fit(X, Y)
        # self.mostre_melhores()
        return self.tfidf_kbest_selector.transform(X).toarray()

    def transform(self, X):
        X = self.vect.transform(X)
        return self.tfidf_kbest_selector.transform(X).toarray()

