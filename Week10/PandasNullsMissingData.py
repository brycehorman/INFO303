import numpy as np
import pandas as pd

my_df = pd.DataFrame(np.arange(0,30,2).reshape(5,3), index=['a','c','e','f','h'],
                     columns=['c1','c2','c3'])
print(my_df)
my_df = my_df.reindex(['a','b','c','d','e','f','g','h'])
print(my_df)
print(my_df.isnull())
print(my_df.notnull())
print(my_df['c1'].isnull())

print(my_df['c1'].sum())
print(my_df['c1'].mean())

my_df['c1']=my_df['c1'].fillna(0)
print(my_df)

my_df['c2']=my_df['c2'].fillna(method='pad')
print(my_df)

my_df['c3']=my_df['c3'].fillna(method='backfill')
my_df['c3'].fillna(method='backfill', inplace=True)
print(my_df)

my_df = pd.DataFrame(np.arange(0,30,2).reshape(5,3), index=['a','c','e','f','h'],
                     columns=['c1','c2','c3'])
my_df = my_df.reindex(['a','b','c','d','e','f','g','h'])
print(my_df)

my_df = my_df.dropna()
my_df.dropna(inplace=True, axis=1)
print(my_df)

my_df = my_df.replace({np.nan:2222})
print(my_df)
my_df = my_df.replace({2222:1111})
print(my_df)

my_avg = my_df['c1'].mean()
print(my_avg)
my_df['c1'].fillna(my_avg, inplace=True)
print(my_df)
my_df['c2'].fillna(my_df['c2'].median(), inplace=True)
print(my_df)
my_df['c3'].fillna(my_df['c3'].sum(), inplace=True)
print(my_df)

