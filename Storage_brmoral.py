from brmoral_getter import brmoral_data
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
import nltk
from imblearn.over_sampling import SMOTE

tag = 'pena'
X, Y = brmoral_data().polaridade(tag)
caminho = '/home/maxtelll/Downloads/get-started-python/modelos/polaridade/'

# vec = CountVectorizer(analyzer="char", ngram_range=([3, 16]), tokenizer=None, preprocessor=None, max_features=3000)
vec = TfidfVectorizer(min_df=2, stop_words=nltk.corpus.stopwords.words('portuguese'))
# vec = CountVectorizer(analyzer="word", stop_words=nltk.corpus.stopwords.words('portuguese'), max_features=5000)

train = vec.fit_transform(X, Y)
dump(vec, caminho + tag+'_r.joblib')

sm = SMOTE(sampling_strategy='auto', random_state=None)
train, Y = sm.fit_sample(train, Y)

clf = MultinomialNB(alpha=0.1)
# clf =linear_model.LogisticRegression(n_jobs=1, C=100)
# clf = MLPClassifier(solver='lbfgs',hidden_layer_sizes=(150,150,150), alpha=1e-5, max_iter=300, learning_rate_init=0.05, power_t=0.1, learning_rate='constant',  random_state=1)
clf.fit(train, Y)
dump(clf, caminho + tag+'_classifier.joblib')
