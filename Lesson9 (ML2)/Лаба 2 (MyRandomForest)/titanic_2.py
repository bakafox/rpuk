import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


"""
Реализация алгоритма случайного леса
"""
class MyRandomForest:
    def __init__(self, n_estimators, max_depth):
        self.n_estimators = n_estimators # количество деревьев
        self.max_depth = max_depth # максимальная глубина
        self.forest = [] # список деревьев

    def fit(self, data_train, label_train):
        # берём случайный поднабор данных и строим по нему дерево
        data_bootstrap = np.array_split(data_train, self.n_estimators)
        label_bootstrap = np.array_split(label_train, self.n_estimators)

        for i in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(data_bootstrap[i], label_bootstrap[i])
            self.forest.append(tree)

    def predict(self, data_test):
        predicted = np.zeros(len(data_test))
        for tree in self.forest:
            predicted += tree.predict(data_test)
        # возвращаем округлённые средние значения предсказаний
        return np.round(predicted / self.n_estimators)


"""
Функция выполнения оценки точности
(взята из предыдущей лабы)
"""
def get_accuracy(predicted, control):
    return np.round((np.count_nonzero(predicted == control)
                       / len(control) * 100), 2)


"""
Выполним оценку точности для алгоритма
случайного леса и алгоритма решающего дерева
"""
df = pd.read_csv('rpuk/Lesson9 (ML2)/Лаба 1 (Титаник ML)/titanic_prepared.csv')
df_train, df_test = train_test_split(df, test_size=0.1, random_state=1)
#df_train = df_train.drop(['morning', 'day', 'evening'], axis=1)
#df_test = df_test.drop(['morning', 'day', 'evening'], axis=1)

df_train_main = df_train[['label']]
df_train = df_train.drop(['label'], axis=1)
df_test_main = df_test[['label']]
df_test = df_test.drop(['label'], axis=1)

model_DT = DecisionTreeClassifier(max_depth=3)
model_DT.fit(df_train, df_train_main)
predict_DT = model_DT.predict(df_test)

model_MRF = MyRandomForest(n_estimators=10, max_depth=3)
model_MRF.fit(df_train, df_train_main)
predict_MRF = model_MRF.predict(df_test)

print('\nТочность DecisionTree:',
      get_accuracy(predict_DT, df_test_main['label']), '%')
print('\nТочность MyRandomForest:',
      get_accuracy(predict_MRF, df_test_main['label']), '%')
