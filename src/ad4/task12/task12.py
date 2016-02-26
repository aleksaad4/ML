# coding=utf-8
import pandas

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

from ad4.utils.utils import save_res

task_num = 12

# Загрузите данные из файла abalone.csv.
# Это датасет, в котором требуется предсказать возраст ракушки (число колец) по физическим измерениям.
# Загрузите данные close_prices.csv.
# В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
data = pandas.read_csv('../../../data/task12/abalone.csv')

# Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
# Если вы используете Pandas, то подойдет следующий код:
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# Разделите содержимое файлов на признаки и целевую переменную.
# В последнем столбце записана целевая переменная, в остальных — признаки.
labels = data['Rings']
features = data.ix[:, :-1]

# Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев:
#  от 1 до 50 (не забудьте выставить "random_state=1" в конструкторе).
# Для каждого из вариантов оцените качество работы полученного леса на кросс-валидации по 5 блокам.
# Используйте параметры "random_state=1" и
# "shuffle=True" при создании генератора кросс-валидации sklearn.cross_validation.KFold.
# В качестве меры качества воспользуйтесь долей правильных ответов (sklearn.metrics.r2_score).
best_n_est = None
best_score = None
for n_est in range(1, 50):
    clf = RandomForestRegressor(n_estimators=n_est, random_state=1)
    clf.fit(features, labels)

    kf = KFold(len(labels), n_folds=5, random_state=1, shuffle=True)
    scores = cross_val_score(clf, features, labels, cv=kf, scoring='r2')

    # Определите, при каком минимальном количестве деревьев случайный лес
    # показывает качество на кросс-валидации выше 0.52. Это количество и будет ответом на задание.
    if (best_n_est is None or n_est < best_n_est) and scores.mean() > 0.52:
        best_n_est = n_est
        best_score = scores.mean()

    print str(n_est) + ":" + str(scores.mean())

# Обратите внимание на изменение качества по мере роста числа деревьев. Ухудшается ли оно?
# Да
print(best_n_est)
save_res(task_num, 1, best_n_est)
