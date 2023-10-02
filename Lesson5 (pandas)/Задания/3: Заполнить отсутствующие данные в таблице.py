import pandas as pd

df = pd.read_csv('rpuk/Lesson5 (pandas)/Задания/3: wells_info_na.csv')

# его идеи... ***** что это
df_fields_numeric = (df.dtypes[df.dtypes == 'int64'].index.to_list() +
                     df.dtypes[df.dtypes == 'float64'].index.to_list())
df_fields_notnumeric = df.dtypes.index.difference(df_fields_numeric).to_list()

df[df_fields_numeric] = df[df_fields_numeric].fillna(
    df[df_fields_numeric].median())
df[df_fields_notnumeric] = df[df_fields_notnumeric].fillna(
    df[df_fields_notnumeric].sort_values(df_fields_notnumeric).loc[0])

df.to_csv('rpuk/Lesson5 (pandas)/Задания/3: result.csv', index=False)
