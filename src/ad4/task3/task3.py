# coding=utf-8

import pandas
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

from ad4.utils.utils import save_res

task_num = 3

# Извлеките из данных признаки и классы.
# Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний.
# в файле нет строки с заголовками -> header=None
data = pandas.read_csv('../../../data/task3/wine.data', header=None)


def find_best_param(labels, features):
    """
    Функция находящая лучшее значение k для KNN и соответствующее значение качества (доля верных ответов)
    :param labels: labels
    :param features: features
    :return: tuple of k and best score
    """
    # Найдите точность классификации на кросс-валидации
    # для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50.
    # Для проведения кросс-валидации воспользуйтесь генератором sklearn.cross_validation.KFold
    #  с 5 блоками и методом cross_val_score.
    # Для воспроизводимости результата создавайте генератор KFold с параметром random_state=42.
    # Мера качества — доля верных ответов.

    result_scores = []
    for k in range(1, 50):
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(features, labels)

        kf = KFold(len(labels), n_folds=5, random_state=42)
        scores = cross_val_score(clf, features, labels, cv=kf)

        result_scores.append((k, scores.mean()))
        print str(k) + ":" + str(scores.mean())

    # сортируем по убыванию score
    result_scores.sort(key=lambda tup: -tup[1])

    # возвращаем лучшее значение
    return result_scores[0]


# классы
labels = data.ix[:, 0]

# признаки
features = data.ix[:, 1:]

# При каком k получилось оптимальное качество? Чему оно равно (число в интервале от 0 до 1)?
# Данные результаты и будут ответами для вопросов 1 и 2
best_k, best_score = find_best_param(labels, features)
print best_k
save_res(3, 1, best_k)
print best_score
save_res(3, 2, round(best_score, 2))

# Произведите нормировку признаков с помощью функции sklearn.preprocessing.scale.
#  Снова найдите оптимальное k на кросс-валидации.
# Это k, а также значение качества при нем (число в интервале от 0 до 1), будут ответами для вопросов 3 и 4.

# производим нормализацию признаков
features = scale(features)

best_k, best_score = find_best_param(labels, features)
print best_k
save_res(3, 3, best_k)
print best_score
save_res(3, 4, round(best_score, 2))
