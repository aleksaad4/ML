# coding=utf-8
from numpy import linspace
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale

from ad4.utils.utils import save_res

task_num = 4

# Загрузите выборку Boston с помощью функции sklearn.datasets.load_boston().
# Результатом вызова данной функции является объект,
# у которого признаки записаны в поле data, а целевой вектор — в поле target.
data = load_boston()

# признаки
features = data['data']

# классы
prices = data['target']

# Отнормируйте выборку с помощью функции sklearn.preprocessing.scale.
# производим нормализацию признаков
features = scale(features)

# Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом,
#  чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace).
# Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр добавляет в алгоритм веса,
#  зависящие от расстояния до ближайших соседей.
# В качестве метрики качества используйте среднеквадратичную ошибку
#  (параметр scoring='mean_squared_error' у cross_val_score).
# Качество оценивайте, как и в предыдущем задании, с помощью кросс-валидации по 5 блокам с random_state = 42.

best_p = None
best_score = None
for p in linspace(1, 10, 200):
    clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
    clf.fit(features, prices)

    kf = KFold(len(prices), n_folds=5, random_state=42)
    scores = cross_val_score(clf, features, prices, cv=kf, scoring='mean_squared_error')

    if best_score is None or best_score < scores.mean():
        best_p = p
        best_score = scores.mean()

# Определите, при каком p качество на кросс-валидации оказалось оптимальным
# (обратите внимание, что показатели качества, которые подсчитывает cross_val_score, необходимо максимизировать).
#  Это значение параметра и будет ответом на задачу.
print best_p
save_res(4, 1, round(best_p, 1))
