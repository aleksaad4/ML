# coding=utf-8
import pandas
from scipy.sparse import hstack

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from ad4.utils.utils import save_res

task_num = 10

# Загрузите данные об описаниях вакансий и соответствующих годовых зарплатах из файла salary-train.csv
data_train = pandas.read_csv('../../../data/task10/salary-train.csv')
data_test = pandas.read_csv('../../../data/task10/salary-test-mini.csv')


# Проведите предобработку:
def pre_handle(data):
    # Приведите тексты к нижнему регистру (text.lower()).
    data['FullDescription'] = data['FullDescription'].str.lower()
    # data['LocationNormalized'] = data['LocationNormalized'].str.lower()

    # Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение текста на слова.
    # Для такой замены в строке text подходит следующий вызов: re.sub('[^a-zA-Z0-9]', ' ', text).
    # Также можно воспользоваться методом replace у DataFrame, чтобы сразу преобразовать все тексты:
    data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)

    # Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'.
    # Код для этого был приведен выше.
    data_train['LocationNormalized'].fillna('nan', inplace=True)
    data_train['ContractTime'].fillna('nan', inplace=True)


pre_handle(data_train)
pre_handle(data_test)

vec_tfidf = TfidfVectorizer(min_df=5)
train_features_tfidf = vec_tfidf.fit_transform(data_train['FullDescription'])
test_features_tfidf = vec_tfidf.transform(data_test['FullDescription'])

# Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
vec_dict = DictVectorizer()
train_features_categ = vec_dict.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
test_features_categ = vec_dict.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# Объедините все полученные признаки в одну матрицу "объекты-признаки".
# Обратите внимание, что матрицы для текстов и категориальных признаков являются разреженными.
# Для объединения их столбцов нужно воспользоваться функцией scipy.sparse.hstack.
train_features = hstack([train_features_tfidf, train_features_categ])
test_features = hstack([test_features_tfidf, test_features_categ])

# 3. Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце SalaryNormalized.
clf = Ridge(alpha=1.0)
clf.fit(train_features, data_train['SalaryNormalized'])

# Постройте прогнозы для двух примеров из файла salary-test-mini.csv.
# Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
res = " ".join([str(round(v, 2)) for v in clf.predict(test_features)])
print res
save_res(task_num, 1, res)
