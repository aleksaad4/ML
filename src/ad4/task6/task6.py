# coding=utf-8
import numpy
import pandas

from sklearn.svm import SVC

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
clf = SVC(kernel='linear')
clf.fit(data_features, data_labels)
