import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('rpuk/Lesson9 (ML2)/Лаба 1 (Титаник ML)/titanic_prepared.csv')
df_train, df_test = train_test_split(df, test_size=0.1, random_state=1)
#df_train = df_train.drop(['morning', 'day', 'evening'], axis=1)
#df_test = df_test.drop(['morning', 'day', 'evening'], axis=1)

df_train_main = df_train[['label']]
df_train = df_train.drop(['label'], axis=1)
df_test_main = df_test[['label']]
df_test = df_test.drop(['label'], axis=1)


"""
DecisionTree Classifier
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model_DT = DecisionTreeClassifier(max_depth=6, criterion='entropy')
model_DT.fit(df_train, df_train_main)
predict_DT = model_DT.predict(df_test)

"""
XGBoost Classifier
"""
from xgboost import XGBClassifier
model_XGB = XGBClassifier(n_estimators=10, max_depth=4)
model_XGB.fit(df_train, df_train_main)
predict_XGB = model_XGB.predict(df_test)

"""
Logistic Regression
"""
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression(C=0.01, solver='lbfgs')
model_LR.fit(df_train_scaled, df_train_main)
predict_LR = model_LR.predict(df_test_scaled)


"""
Выполняем оценку точности
"""
def get_accuracy(predicted, control):
    return np.round((np.count_nonzero(predicted == control)
                       / len(control) * 100), 2)

print('\nТочность DecisionTree:',
      get_accuracy(predict_DT, df_test_main['label']), '%')
print('Точность XGBoost:',
      get_accuracy(predict_XGB, df_test_main['label']), '%')
print('Точность LogisticRegression:',
      get_accuracy(predict_LR, df_test_main['label']), '%')


"""
Выбираем 2 самых важных признака по DecisionTree
и проверяем точность модели, обученной только по ним
"""
importances = model_DT.feature_importances_
features = df_train.columns[np.argsort(importances)][::-1][:2]
df_train_2 = df_train[features]
df_test_2 = df_test[features]

model_DT_2 = DecisionTreeClassifier(max_depth=6, criterion='entropy')
model_DT_2.fit(df_train_2, df_train_main)
predict_DT_2 = model_DT_2.predict(df_test_2)

print('\nТоп-2 признака DecisionTree:',
      np.array(features))
print('Точность DecisionTree (топ-2 признака):',
      get_accuracy(predict_DT_2, df_test_main['label']), '%')
