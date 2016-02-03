# coding=utf-8
import numpy

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

task_num = 7

# Загрузите объекты из новостного датасета 20 newsgroups, относящиеся к категориям "космос" и "атеизм"
#  (инструкция приведена выше).
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

texts = newsgroups.data
labels = newsgroups.target

# Вычислите TF-IDF-признаки для всех текстов.
vec = TfidfVectorizer()
features = vec.fit_transform(texts)

# Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
# и параметром random_state=241 при помощи кросс-валидации по 5 блокам.
c_values = numpy.apply_along_axis(lambda x: pow(10.0, x), 0, numpy.arange(-5, 5))
tuned_parameters = [{'C': c_values}]

clf_svc = SVC(kernel='linear', random_state=241)
clf = GridSearchCV(clf_svc, tuned_parameters, cv=5)
# clf.fit(features, labels)

# Обучите SVM по всей выборке с лучшим параметром C, найденным на предыдущем шаге.
best_c = 10# clf.best_params_['C']
best_clf = SVC(kernel='linear', C=best_c, random_state=241)
clf.fit(features, labels)

print best_clf.coef_
