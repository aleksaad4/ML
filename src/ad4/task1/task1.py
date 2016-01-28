# coding=utf-8
import re

import pandas

from ad4.utils.utils import save_res

task_num = 1

# index_col - колонка, отвечающая за нумерацию строк датафрейма
data = pandas.read_csv('../../../data/task1/titanic.csv', index_col='PassengerId')

# первые 10 строк данных
print data[:10]

# первые 5 строк данных
print data.head()

# доступ к определенному столбцу данных
print data['Pclass']

# group by and count по столбцу
print data['Pclass'].value_counts()

# Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
sex_counts = data['Sex'].value_counts()
res1 = str(sex_counts["male"]) + " " + str(sex_counts["female"])
print res1
save_res(task_num, 1, res1)

# Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (знак процента не нужен).
survived_counts = data['Survived'].value_counts()
res2 = round(100. * survived_counts[1] / len(data), 2)
print res2
save_res(task_num, 2, res2)

# Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (знак процента не нужен).
pclass_counts = data['Pclass'].value_counts()
res3 = round(100. * pclass_counts[1] / len(data), 2)
print res3
save_res(task_num, 3, res3)

# Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
ages = data['Age']
res4 = str(round(ages.mean(), 2)) + " " + str(round(ages.median(), 2))
print res4
save_res(task_num, 4, res4)

# Коррелируют ли число братьев/сестер с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
sibsp = data['SibSp']
parch = data['Parch']
res5 = round(sibsp.corr(parch), 2)
print res5
save_res(task_num, 5, res5)

# Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
female_names = data[data['Sex'] == 'female']['Name']


def extract_first_name(name):
    """
    Функция извлечения first name from name
    :param name: name
    :return: first name
    """
    # первое слово в скобках
    m = re.search(".*\\((.*)\\).*", name)
    if m is not None:
        return m.group(1).split(" ")[0]
    # первое слово после Mrs. or Miss. or else
    m1 = re.search(".*\\. ([A-Za-z]*)", name)
    return m1.group(1)


# получаем имя с максимальной частотой
res6 = female_names.map(lambda full_name: extract_first_name(full_name)).value_counts().idxmax()
print res6
save_res(task_num, 6, res6)
