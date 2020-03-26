import pandas as pd
import numpy as np

my_series = pd.Series([15,20,30,60,90,120])
print(my_series)
print(my_series.pct_change())

my_df = pd.DataFrame(np.arange(1,11).reshape(5,2))
print(my_df)
print(my_df.pct_change())
print(my_df.pct_change(axis=0))
print(my_df.pct_change(axis=1))

my_series1 = pd.Series([11,10,18,22,30,40,80,10,23,13])
print(my_series1)
print(my_series1.mean())
print(my_series1.std())
my_series2 = pd.Series([9,77,14,69,99,6,70,58,75,88])
print(my_series2)
print(my_series2.mean())
print(my_series2.std())
print(my_series1.cov(my_series2))
print(my_series2.cov(my_series1))
print(my_series1.corr(my_series2))
print(my_series1.cov(my_series2)/(my_series1.std()*my_series2.std()))

frame = {'s1': my_series1, 's2': my_series2}
my_df = pd.DataFrame(frame)
print(my_df)
print(my_df['s1'].cov(my_df['s2']))
print(my_df.cov())
print(my_df.corr())
print(my_df['s1'].corr(my_df['s2']))

print(my_df['s1'].cov(my_df['s2'])/(my_df['s2'].std()*my_df['s1'].std()))

print(my_series1)
print(my_series1.rank())
print(my_series1.rank(method='average'))
print(my_series1.rank(method='min'))
print(my_series1.rank(method='max'))
print(my_series1.rank(method='average', ascending=False))

frame = {'s1': my_series1, 's2': my_series2}
my_df = pd.DataFrame(frame)
print(my_df['s1'].rank())
print(my_df.rank())

my_df = pd.DataFrame(np.arange(0,40).reshape(10,4),
                     index=pd.date_range('1/1/2021', periods=10, freq='D'),
                     columns=['bill','joe','sally','sue'])
print(my_df)

print(my_df.rolling(window=2).mean())
print(my_df.rolling(window=2).std())
print(my_df.rolling(window=2).sum())

print(my_df[['bill','sue']].rolling(window=3).mean())

print(my_df)
print(my_df.expanding(min_periods=3).mean())
print(my_df.expanding(min_periods=3).sum())
print(my_df.expanding(min_periods=3).std())

my_o = my_df.rolling(window=3)
print(my_o)
print(type(my_o))

print(my_o.aggregate(np.sum))
print(my_o.aggregate(np.mean))

my_o = my_df.expanding(min_periods=3)
print(my_o)
print(type(my_o))

print(my_o.aggregate(np.median))
print(my_o['joe'].aggregate(np.sum))
print(my_o['joe'].aggregate([np.sum,np.mean]))
print(my_o[['joe','sue']].aggregate({'joe': [np.sum, np.mean], 'sue': np.mean}))


