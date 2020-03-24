import numpy as np
import pandas as pd

my_df = pd.DataFrame({
    'dts': pd.date_range(start='2021-01-01', periods=15, freq='D'),
    'dsc': np.linspace(0,stop=14,num=15),
    'dval': np.random.rand(15),
    'risk': np.random.choice(['High','Medium','Low'],15).tolist(),
    'scre': np.random.normal(100,10,size=15).tolist()
})

print(my_df)
print(type(my_df))
my_df2 = my_df.reindex(index=[0,3,6,9], columns=['dts','dval','scre','risk'])
print(my_df2)
print(my_df2.shape)
print(my_df2.size)
print(my_df.shape)

my_df1 = pd.DataFrame(np.random.randn(10,3), columns=['c1','c2','c3'])
print(my_df1)
my_df2 = pd.DataFrame(np.random.randn(7,3), columns=['c1','c2','c3'])
print(my_df1)
my_df1 = my_df1.reindex_like(my_df2)
print(my_df1)
print(my_df2.reindex_like(my_df1))
my_df2 = my_df2.reindex_like(my_df1, method='ffill')
print(my_df2)

my_df2 = pd.DataFrame(np.arange(1,24,2).reshape(4,3), index=[6,7,8,9], columns=['c1','c2','c3'])
print(my_df2)
my_df2 = my_df2.reindex_like(my_df1, method='bfill', limit=1)
print(my_df2)

my_df2 = pd.DataFrame(np.arange(1,24,2).reshape(4,3), index=[1,4,5,6], columns=['c1','c2','c3'])
print(my_df2)
my_df2 = my_df2.reindex_like(my_df1, method='nearest')
print(my_df2)

my_df2 = pd.DataFrame(np.arange(1,24,2).reshape(4,3), index=[6,7,8,9], columns=['c1','c2','c3'])
print(my_df2)
my_df2 = my_df2.rename(columns={'c1':'reve', 'c2':'revw','c3':'revc'},
                       index={6:'Bill',7:'Sally',8:'Sue',9:'Jane'})
print(my_df2)

my_df = pd.DataFrame({
    'dts': pd.date_range(start='2021-01-01', periods=15, freq='D'),
    'dsc': np.linspace(0,stop=14,num=15),
    'dval': np.random.rand(15),
    'risk': np.random.choice(['High','Medium','Low'],15).tolist(),
    'scre': np.random.normal(100,10,size=15).tolist()
})

print(my_df)

for col in my_df:
    print(col)

for key,value in my_df.iteritems():
    print(key)
    print(value)

for key,value in my_df['dsc'].iteritems():
    print(f'Key is {key} and the value is {value}.')

for key,value in my_df.iterrows():
    print(key)
    print(value)

for key,value in my_df.iterrows():
    value['dsc'] = 240
print(my_df)

for row in my_df.itertuples():
    print(row[2])

my_df = pd.DataFrame({
    'dts': pd.date_range(start='2021-01-01', periods=15, freq='D'),
    'dsc': np.linspace(0,stop=14,num=15),
    'dval': np.random.rand(15),
    'risk': np.random.choice(['High','Medium','Low'],15).tolist(),
    'scre': np.random.normal(100,10,size=15).tolist()
}, index=[0,1,2,13,14,5,6,7,8,9,10,11,12,3,4])
print(my_df)
my_df_sorted = my_df.sort_index(ascending=False)
print(my_df_sorted)

my_df_sorted = my_df.sort_values(by='scre', ascending=False)
print(my_df_sorted)

my_df_sorted = my_df.sort_values(by=['risk','scre'])
print(my_df_sorted)
