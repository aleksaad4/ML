# coding=utf-8
import numpy

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from ad4.utils.utils import save_res

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
c_values = numpy.apply_along_axis(lambda x: pow(10.0, x), 0, numpy.arange(-5, 6))
tuned_parameters = [{'C': c_values}]

clf_svc = SVC(kernel='linear', random_state=241)
clf = GridSearchCV(clf_svc, tuned_parameters, cv=5)
# clf.fit(features, labels)

# Обучите SVM по всей выборке с лучшим параметром C, найденным на предыдущем шаге.
best_c = 10 # clf.best_params_['C']
best_clf = SVC(kernel='linear', C=best_c, random_state=241)
best_clf.fit(features, labels)

# Найдите 10 слов с наибольшим по модулю весом. Они являются ответом на это задание.
# Укажите их через запятую или пробел, в нижнем регистре, в лексикографическом порядке.
best_coef_idx = numpy.absolute(numpy.asarray(best_clf.coef_.todense())).argsort().reshape(-1)[-10:][::-1]

best_words = [vec.get_feature_names()[i] for i in best_coef_idx]
best_words.sort()
res = ",".join(best_words)

print res
save_res(task_num, 1, res)
