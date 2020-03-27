import pandas as pd
import numpy as np

my_df1 = pd.DataFrame({
    'Name': ['Joe','Sue','Sally','Tim','Joe'],
    'Course': ['math','econ','mgmt','art','hist'],
    'Grades': [75,95,98,97,88]
}, index=[1,2,3,4,5])
print(my_df1)

my_df2 = pd.DataFrame({
    'Name': ['Joe','Sue','Sally','Time','Joe'],
    'Course': ['econ','fin','mgmt','music','lead'],
    'Grades': [85,99,100,74,66]
}, index=[1,2,3,4,5])
print(my_df1)

my_df3 = pd.concat([my_df1,my_df2])
print(my_df3)
print(my_df3.loc[1])
print(my_df3.iloc[0])
print(my_df3.iloc[5])

my_df3 = pd.concat([my_df1,my_df2], keys=['a','b'])
print(my_df3)
print(my_df3.loc['a'])
print(my_df3.iloc[0])

my_df3 = pd.concat([my_df1,my_df2], keys=['a','b'], ignore_index=True)
print(my_df3)

my_df3 = pd.concat([my_df2,my_df1], axis=1)
print(my_df3)

my_df3 = my_df2.append([my_df1,my_df2,my_df1])
print(my_df3)