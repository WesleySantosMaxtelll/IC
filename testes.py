# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_selection import SelectKBest
# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import classification_report
# from sklearn.feature_selection import SelectKBest
# from sklearn.linear_model import LogisticRegressionCV
# import sklearn
#
#
#


import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)  # in bytes



# s = SelectKBest(k=7)
# # print(a.get_params())
# #
# x = ['oi tudo bem', 'ta tudo bem sim', 'blah blah', 'nossa ta tudo bem', 'bem no rabo', 'sim maria, foda se', 'ta tudo otimo']
# y = [1,1,2,1,2,2,1]
# a = TfidfVectorizer(analyzer="char", ngram_range=([5, 8]), tokenizer=None, preprocessor=None, stop_words='english')
# x1 = a.fit_transform(x).toarray()
# feat = a.get_feature_names()
# print(feat)
# print(x1)
#
# s.fit(x1, y)
# lista = s.get_support()
#
# for l, f in zip(lista, feat):
#      if l:
#           print(f)
# print(s.transform(x1))
# #
# # palavras = a.get_feature_names()
# # pesos = a.idf_
# # final_dict = {}
# # for i in range(len(palavras)):
# #     final_dict[palavras[i]] = pesos[i]
# # print(final_dict)
# #
# # a[0]
# import os
# import numpy as np
#
# from mpl_toolkits.mplot3d import Axes3D
# # import somoclu
#
# # n_rows, n_columns = 100, 160
# # som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
# #
# # # os.mkdir('/home/maxtelll/Desktop/ic_novo_backup/posicionamento/resultados/KNN/word2vec/maconha', 55)
# # # from imblearn.over_sampling import RandomOverSampler
# # import numpy as np
# # from imblearn.under_sampling import RandomUnderSampler
# #
# #
# # c1 = np.random.rand(50, 3)/5
# # c2 = (0.6, 0.1, 0.05) + np.random.rand(50, 3)/5
# # c3 = (0.4, 0.1, 0.7) + np.random.rand(50, 3)/5
# # data = np.float32(np.concatenate((c1, c2, c3)))
# # colors = ["red"] * 50
# # colors.extend(["green"] * 50)
# # colors.extend(["blue"] * 50)
# #
# # print(data)
#
#
# # y_true = [0, 1, 0, 1, 1]
# # y_pred = [0, 0, 1, 1, 1]
# # target_names = ['class 0', 'class 1']
# # print(classification_report(y_true, y_pred, target_names=target_names))
# #
# # p = [1,0,1,1,0]
# # y = [0,0,0,1,1]
# # print(classification_report(y, p, target_names=['0','1']))
#
# #
# X = [[1,2,3, 0],
#      [10,20,30, 0],
#      [11,21,31, 1],
#      [12,22,32, 0],
#      [13,23,33, 1],
#      [14,24,34, 0],
#      [15,25,35, 0],
#      [16,26,36, 0],
#      [17,27,37, 0],
#      [18,28,38, 0]]
#
# Y = [1,0,1,0,1,0,0,1,0,0]
#
#
# X1 = [[1,2,3, 0],
#      [10,20,30, 0],
#      [11,21,31, 1],
#      [12,22,32, 0],
#      [16,26,36, 0],
#      [17,27,37, 0],
#      [18,28,38, 0]]
#
# Y2 = [1,0,1,0,1,0,0]
#
#
#
#
# m = DummyClassifier(strategy="most_frequent", random_state=None, constant=None)
# m.fit(X, Y)
# pred = m.predict(X1)
# print(classification_report(Y2, pred, target_names=['Contra', 'A favor']))
#
#
# # classificador
# # rus = RandomOverSampler(random_state=2)
# # # rus.fit(X, Y)
# # X_resampled, y_resampled = rus.fit_resample(X, Y)
# # print(X_resampled)
# # print(y_resampled)
# #
# #
# #
# #
# # for f, b in zip(X, Y):
# #     print(f, b)
# # train_x, train_y, test_x, test_y = train_test_split(X, Y,test_size=0.4, random_state=42)
# #
# # print(train_x)
# # print(train_y)
# # print(test_x)
# # print(test_y)