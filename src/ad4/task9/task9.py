# coding=utf-8
import numpy
import pandas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, \
    precision_recall_curve

from ad4.utils.utils import save_res

task_num = 9

# Загрузите файл classification.csv.
# В нем записаны истинные классы объектов выборки (колонка true) и ответы некоторого классификатора (колонка predicted).
data = pandas.read_csv('../../../data/task9/classification.csv')

true = data['true']
pred = data['pred']

# Заполните таблицу ошибок классификации:
# Actual Positive	Actual Negative
# Predicted Positive	TP	FP
# Predicted Negative	FN	TN

# Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных алгоритмом к классу 1.
# Ответ в данном вопросе — четыре числа через пробел.
TP = (numpy.logical_and(true, pred) == True).value_counts()[True]
FP = (numpy.logical_and(numpy.logical_not(true), pred) == True).value_counts()[True]
FN = (numpy.logical_and(true, numpy.logical_not(pred)) == True).value_counts()[True]
TN = (numpy.logical_and(numpy.logical_not(true), numpy.logical_not(pred)) == True).value_counts()[True]

res_1 = str(TP) + " " + str(FP) + " " + str(FN) + " " + str(TN)
print res_1
save_res(task_num, 1, res_1)

# Посчитайте основные метрики качества классификатора:
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score

a = round(accuracy_score(true, pred), 2)
p = round(precision_score(true, pred), 2)
r = round(recall_score(true, pred), 2)
f = round(f1_score(true, pred), 2)

res_2 = str(a) + " " + str(p) + " " + str(r) + " " + str(f)
print res_2
save_res(task_num, 2, res_2)

# Имеется четыре обученных классификатора. В файле scores.csv
# записаны истинные классы и значения степени принадлежности положительному классу для каждого классификатора на
# некоторой выборке:
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

data = pandas.read_csv('../../../data/task9/scores.csv')
true = data['true']

#  Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC
# (укажите название столбца с ответами этого классификатора)? Воспользуйтесь функцией sklearn.metrics.roc_auc_score.
clfs = ["score_logreg", "score_svm", "score_knn", "score_tree"]
roc_scores = [(c_name, roc_auc_score(true, data[c_name])) for c_name in clfs]

res_3 = max(roc_scores, key=lambda i: i[1])[0]
print res_3
save_res(task_num, 3, res_3)

# Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Какое значение точности при этом получается?

# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
#  sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
# В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds.
# Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

# Если ответом является нецелое число, то целую и дробную часть необходимо разграничивать точкой,
# например, 0.42. При необходимости округляйте дробную часть до двух знаков.
best_prec = []
for c_name in clfs:
    p, r, th = precision_recall_curve(true, data[c_name])
    best_prec.append((c_name, max(p[r >= 0.7])))

res_4 = max(best_prec, key=lambda i: i[1])[0]
print res_4
save_res(task_num, 4, res_4)
