# coding=utf-8
import numpy
import pandas
from sklearn.metrics import roc_auc_score

from ad4.task8.gd import gd, proba
from ad4.utils.utils import save_res

task_num = 8

# Загрузите данные из файла data-logistic.csv.
# Это двумерная выборка, целевая переменная на которой принимает значения -1 или 1.
data = pandas.read_csv('../../../data/task8/data-logistic.csv', header=None)

# features and labels
X = data[[1, 2]]
y = numpy.ravel(data[[0]])

# Убедитесь, что выше выписаны правильные формулы для градиентного спуска.
# Обратите внимание, что мы используем полноценный градиентный спуск, а не его стохастический вариант!

# Реализуйте градиентный спуск для обычной и L2-регуляризованной (с коэффициентом регуляризации 10)
# логистической регрессии. Используйте длину шага k=0.1. В качестве начального приближения используйте вектор (0, 0).
# Запустите градиентный спуск и доведите до сходимости (евклидово расстояние между векторами весов
#  на соседних итерациях должно быть не больше 1e-5). Рекомендуется ограничить сверху число итераций десятью тысячами.

w = gd(X, y)
w_with_reg = gd(X, y, C=10)

print "W: " + str(w)
print "W with L2 reg: " + str(w_with_reg)

# Какое значение принимает AUC-ROC на обучении без регуляризации и при ее использовании?
# Эти величины будут ответом на задание.
#  Обратите внимание, что на вход функции roc_auc_score нужно подавать оценки вероятностей,
# подсчитанные обученным алгоритмом. Для этого воспользуйтесь сигмоидной функцией: a(x) = 1 / (1 + exp(-w1 x1 - w2 x2)).
p = proba(X, w)
p_with_reg = proba(X, w_with_reg)

roc = roc_auc_score(y, p)
roc_with_reg = roc_auc_score(y, p_with_reg)

res = str(round(roc, 3)) + " " + str(round(roc_with_reg, 3))
print res
save_res(task_num, 1, res)

# Попробуйте поменять длину шага. Будет ли сходиться алгоритм, если делать более длинные шаги?
# Как меняется число итераций при уменьшении длины шага?
w_with_reg = gd(X, y, C=10, k=0.01)

# Попробуйте менять начальное приближение. Влияет ли оно на что-нибудь?
w_dif_init = gd(X, y, C=10, k=0.01, init_w=numpy.array([1, 2]))
print "W with L2 reg in different init point: " + str(w_with_reg)

p_dif_init = proba(X, w_dif_init)

roc_dif_init = roc_auc_score(y, p_dif_init)
print roc_dif_init
