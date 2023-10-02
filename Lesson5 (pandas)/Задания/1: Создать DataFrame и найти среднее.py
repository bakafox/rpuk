import numpy as np
import pandas as pd

my_data = np.random.random((10, 5))
my_dataframe = pd.DataFrame(my_data)
# print(my_dataframe)

my_dataframe_analysis = my_dataframe[my_dataframe > 0.3]
print(my_dataframe_analysis.mean(axis=1))