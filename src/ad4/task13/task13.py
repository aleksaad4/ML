# coding=utf-8
import matplotlib.pyplot as plt
import numpy
import pandas
from numpy import argmin

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

from ad4.utils.utils import save_res

task_num = 13

# Загрузите выборку из файла gbm-data.csv с помощью pandas и преобразуйте ее в массив numpy
# (параметр values у датафрейма). В первой колонке файла с данными записано, была или нет реакция.
# Все остальные колонки (d1 - d1776) содержат различные характеристики молекулы, такие как размер, форма и т.д.

data = pandas.read_csv('../../../data/task13/gbm-data.csv').values

labels = data[:, 0]
features = data[:, 1:]

# Разбейте выборку на обучающую и тестовую, используя функцию train_test_split с параметрами
# test_size = 0.8 и random_state = 241.
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=241)


def get_loss(clf, X, y):
    loss = []
    # Используйте метод staged_decision_function для предсказания качества
    # на обучающей и тестовой выборке на каждой итерации.
    for y_pred in clf.staged_decision_function(X):
        # Вычислите и постройте график значений log-loss (которую можно посчитать с помощью функции
        # sklearn.metrics.log_loss) на обучающей и тестовой выборках, а также найдите минимальное значение метрики
        #  и номер итерации, на которой оно достигается.
        loss.append(log_loss(y, 1.0 / (1.0 + numpy.exp(-y_pred))))

    min_iter = argmin(loss)
    min_loss = loss[min_iter]
    return loss, min_iter, min_loss


# Обучите GradientBoostingClassifier с параметрами n_estimators=250, verbose=True, random_state=241
# и для каждого значения learning_rate из списка [1, 0.5, 0.3, 0.2, 0.1] проделайте следующее:
# for lr in [1, 0.5, 0.3, 0.2, 0.1]:
for lr in [0.2]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=lr)
    clf.fit(X_train, y_train)

    train_loss, train_min_iter, train_min_loss = get_loss(clf, X_train, y_train)
    test_loss, test_min_iter, test_min_loss = get_loss(clf, X_test, y_test)

    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

    # 4. Приведите минимальное значение log-loss и номер итерации, на котором оно достигается, при learning_rate = 0.2.
    if lr == 0.2:
        res_min_loss = test_min_loss
        res_min_iter = test_min_iter

# 3. Как можно охарактеризовать график качества на тестовой выборке, начиная с некоторой итерации:
# переобучение (overfitting) или недообучение (underfitting)?
# В ответе укажите одно из слов overfitting либо underfitting.
res = "overfitting"
print res
save_res(task_num, 1, res)

res = str(round(res_min_loss, 2)) + " " + str(res_min_iter)
print res
save_res(task_num, 2, res)

# 5. На этих же данных обучите RandomForestClassifier с количеством деревьев, равным количеству итераций,
# на котором достигается наилучшее качество у градиентного бустинга из предыдущего пункта, c random_state=241 и
#  остальными параметрами по умолчанию. Какое значение log-loss на тесте получается у этого случайного леса?
# (Не забывайте, что предсказания нужно получать с помощью функции predict_proba)
clf = RandomForestClassifier(n_estimators=res_min_iter, random_state=241)
clf.fit(X_train, y_train)
y_proba = clf.predict_proba(X_test)[:, 1]
rf_loss = log_loss(y_test, y_proba)

res = str(round(rf_loss, 2))
print res
save_res(task_num, 3, res)
