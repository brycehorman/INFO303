import pandas as pd
import numpy as np

my_series = pd.Series(np.random.randn(6))
print(my_series)

print(my_series.axes)

print(my_series.empty)
my_series2 = pd.Series()
print(my_series2.empty)

print(my_series.ndim)

print(my_series.size)

print(my_series.values)
print(type(my_series.values))

print(my_series.head(2))
print(my_series.tail(3))

data = {'emps':pd.Series(['Joe','Sally','Jim','Micky','Tim','Tom']),
        'exp':pd.Series([1,3,5,7,5,4]),
        'satrating':pd.Series([4.7,5,5.3,7.5,8.1,9])
        }
my_df = pd.DataFrame(data)
print(my_df)

print(my_df.T)
print(my_df.transpose())
my_df2 = my_df.T
print(my_df2)

print(my_df.axes)

print(my_df.dtypes)

print(my_df.empty)
my_df2 = pd.DataFrame()
print(my_df2.empty)

print(my_df.ndim)

print(my_df.shape)

print(my_df)
print(my_df.size)

print(my_df.values)
print(type(my_df.values))

