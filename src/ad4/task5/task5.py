# coding=utf-8
import numpy
import pandas

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from ad4.utils.utils import save_res

task_num = 5

# Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
data_train = pandas.read_csv('../../../data/task5/perceptron-train.csv', header=None)
data_test = pandas.read_csv('../../../data/task5/perceptron-test.csv', header=None)

# features and labels
data_train_features = data_train[[1, 2]]
data_train_labels = numpy.ravel(data_train[[0]])

data_test_features = data_test[[1, 2]]
data_test_labels = numpy.ravel(data_test[[0]])


def get_accuracy(_data_train_features, _data_train_labels, _data_test_features, _data_test_labels):
    # Обучите персептрон со стандартными параметрами и random_state=241.
    clf = Perceptron(random_state=241, shuffle=True)
    clf.fit(_data_train_features, numpy.ravel(_data_train_labels))

    # Подсчитайте качество (долю правильно классифицированных объектов, accuracy)
    # полученного классификатора на тестовой выборке.
    predictions = clf.predict(_data_test_features)
    score = accuracy_score(_data_test_labels, predictions)
    return score


accuracy = get_accuracy(data_train_features, data_train_labels, data_test_features, data_test_labels)
print accuracy

# Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
scaler = StandardScaler()
data_train_features_scaled = scaler.fit_transform(data_train_features)
data_test_features_scaled = scaler.transform(data_test_features)

# Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
accuracy_stand = get_accuracy(data_train_features_scaled, data_train_labels, data_test_features_scaled,
                              data_test_labels)
print accuracy_stand

# Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.
result = round(accuracy_stand - accuracy, 3)
print result
save_res(task_num, 1, result)
