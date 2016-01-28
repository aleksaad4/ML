# coding=utf-8
import numpy as np

# генерация матрицы с выборкой из нормального распределения
# loc - mu
# scale - sigma
# size - размерность матрицы
X = np.random.normal(loc=1, scale=10, size=(1000, 50))
print X

# вычисление среднего значения в каждом столбце
m = np.mean(X, axis=0)
print m

# вычисление стандратного отклонения в каждом столбце
std = np.std(X, axis=0)
print std

# нормированная матрица
X_norm = ((X - m) / std)
print X_norm

Z = np.array([[4, 5, 0],
              [1, 9, 3],
              [5, 1, 1],
              [3, 3, 3],
              [9, 9, 9],
              [4, 7, 1]])
# суммирование по строкам
r = np.sum(Z, axis=1)
# индексы строк, в которых сумма элементов не превосходит 10
print np.nonzero(r > 10)

# генерируем единичные матрицы
A = np.eye(3)
B = np.eye(3)
print A
print B

# вертикальная стыковка матриц
AB = np.vstack((A, B))
print AB

# вертикальная стыковка матриц
A_B = np.hstack((A, B))
print A_B
