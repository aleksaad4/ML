# coding=utf-8
import numpy
import pandas

from sklearn.svm import SVC

from ad4.utils.utils import save_res

task_num = 6

# Загрузите выборку из файла svm-data.csv.
# В нем записана двумерная выборка (целевая переменная указана в первом столбце, признаки — во втором и третьем).
data = pandas.read_csv('../../../data/task6/svm-data.csv', header=None)

# features and labels
data_features = data[[1, 2]]
data_labels = numpy.ravel(data[[0]])

# Обучите классификатор с линейным ядром, параметром C = 100000 и random_state=241.
# Такое значение параметра нужно использовать, чтобы убедиться, что SVM работает с выборкой как с линейно разделимой.
# При более низких значениях параметра алгоритм будет настраиваться с учетом слагаемого в функционале,
# штрафующего за маленькие отступы, из-за чего результат может не совпасть с решением классической задачи SVM
# для линейно разделимой выборки.
clf = SVC(kernel='linear', C=100000, random_state=241)
clf.fit(data_features, data_labels)

# Найдите номера объектов, которые являются опорными (нумерация с единицы).
# Они будут являться ответом на задание.
# Обратите внимание, что в качестве ответа нужно привести номера объектов
# в возрастающем порядке через запятую или пробел. Нумерация начинается с 1.
sv = " ".join(map(lambda i: str(i + 1), clf.support_.tolist()))
print sv
save_res(task_num, 1, sv)
