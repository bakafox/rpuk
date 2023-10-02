import pandas as pd

df = pd.read_csv('rpuk/Lesson5 (pandas)/Задания/2: wells_info.csv')
df['FirstProductionDate'] = pd.to_datetime(df['FirstProductionDate'])
df['CompletionDate'] = pd.to_datetime(df['CompletionDate'])

"""
print(pd.concat([
        df['FirstProductionDate'], df['CompletionDate'],
        df['CompletionDate'] - df['FirstProductionDate']
    ], axis=1))
"""
print(((df['CompletionDate'] - df['FirstProductionDate'])
       .dt.days // 30).to_list())
