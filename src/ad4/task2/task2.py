# coding=utf-8
import pandas
from sklearn.tree import DecisionTreeClassifier

from ad4.utils.utils import save_res

task_num = 2

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
data = pandas.read_csv('../../../data/task2/titanic.csv', index_col='PassengerId')


# Обратите внимание, что признак Sex имеет строковые значения.
def convert_sex(sex):
    if sex == 'male':
        return 1
    else:
        return 0


data['Sex'] = data['Sex'].apply(lambda sex: convert_sex(sex))

# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст.
# Такие записи при чтении их в pandas принимают значение nan.
# Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
data = data[['Survived', 'Pclass', 'Fare', 'Age', 'Sex']].dropna()

# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare),
# возраст пассажира (Age) и его пол (Sex)
feature_names = ['Pclass', 'Fare', 'Age', 'Sex']
features = data[feature_names]
print features

# Выделите целевую переменную — она записана в столбце Survived.
labels = data[['Survived']]

# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию.
clf = DecisionTreeClassifier(random_state=241)
clf.fit(features, labels)
feature_importances = clf.feature_importances_
print feature_importances

imp = zip(feature_names, feature_importances)

imp.sort(key=lambda tup: -tup[1])
res1 = str(imp[0][0]) + "," + str(imp[1][0])
print res1
save_res(task_num, 1, res1)
