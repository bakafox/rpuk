import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

"""
Данные при обучении разделяются на 3 части:
тренировочная (train), валидационная (validation) и часть для тестирования (test)
 
* train — часть набора данных на основании которого будет строиться модель
* validation — часть набора данных для подбора параметров модели (опционально)
* test — часть набора данных для проверки модели
"""

df = pd.read_csv('wells_info_with_prod.csv')


# 1. извлекаем нужные нам признаки для анализа
df_meta = df[['API', 'operatorNameIHS', 'formation']]

df_date = df[['FirstProductionDate', 'CompletionDate']]
df_date['FirstProductionDate'] = pd.to_datetime(df['FirstProductionDate'])
df_date['FirstProductionDate'] = df_date['FirstProductionDate'].fillna(df_date['FirstProductionDate'].median())
df_date['CompletionDate'] = pd.to_datetime(df['CompletionDate'])
df_date['CompletionDate'] = df_date['CompletionDate'].fillna(df_date['CompletionDate'].median())

df_main = df[['Prod1Year']]
df_main = df_main.fillna(df_main.median())

#print( df_meta, df_date, df_main )


# 2. разделяем данные на train и test
df_meta_train, df_meta_test = train_test_split(df_meta, test_size=0.2, random_state=1)
df_date_train, df_date_test = train_test_split(df_date, test_size=0.2, random_state=1)
df_main_train, df_main_test = train_test_split(df_main, test_size=0.2, random_state=1)


# 3. масштабируем train и test, т.е. приводим значения к диапазону 0-1
from sklearn.preprocessing import MinMaxScaler
scaler_date = MinMaxScaler()
scaler_main = MinMaxScaler()
df_date_train_scaled = scaler_date.fit_transform(df_date_train)
df_main_train_scaled = scaler_main.fit_transform(df_main_train)

df_date_test_scaled = scaler_date.transform(df_date_test)
df_main_test_scaled = scaler_main.transform(df_main_test)


print(pd.concat(
    [df_meta_train.reset_index(drop=True),
    pd.DataFrame(df_date_train_scaled),
    pd.DataFrame(df_main_train_scaled)],
    axis=1
))