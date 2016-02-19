# coding=utf-8
import pandas
from numpy import corrcoef

from sklearn.decomposition import PCA

from ad4.utils.utils import save_res

task_num = 11

# Загрузите данные close_prices.csv.
# В этом файле приведены цены акций 30 компаний на закрытии торгов за каждый день периода.
prices = pandas.read_csv('../../../data/task11/close_prices.csv').ix[:, 1:]

# На загруженных данных обучите преобразование PCA с числом компоненты равным 10.
pca = PCA(n_components=10)
pca.fit(prices)

# Скольких компонент хватит, чтобы объяснить 90% дисперсии?
percent_90 = 0
sum_var = 0
component_count = 0
for v in pca.explained_variance_ratio_:
    sum_var += v
    component_count += 1
    if sum_var >= 0.9:
        percent_90 = component_count
        break

print percent_90
save_res(task_num, 1, percent_90)

# Примените построенное преобразование к исходным данным и возьмите значения первой компоненты.
proj = pca.transform(prices)

# Загрузите информацию об индексе Доу-Джонса из файла djia_index.csv.
jia = pandas.read_csv('../../../data/task11/jia_index.csv')['^DJI']

# Чему равна корреляция Пирсона между первой компонентой и индексом Доу-Джонса?
corr = round(corrcoef(proj[:, 0], jia)[1, 0], 2)
print(corr)
save_res(task_num, 2, corr)

# Какая компания имеет наибольший вес в первой компоненте? Укажите ее название с большой буквы.
c_name = prices.columns.values[pca.components_[0].argmax()]
print(c_name)
save_res(task_num, 3, c_name)
