import numpy as np
import pandas as pd

data = {'emps':pd.Series(['Joe','Sally','Jim','Micky','Tim','Tom','Jane','Ginger']),
        'exp': pd.Series([1,3,5,7,5,4,3,1]),
        'satrating':pd.Series([4.7,5,5.3,7.5,8.1,9.4,4.7,8.3]),
        'revenue':pd.Series([2000,2300,2100,2500,1900,1850,2400,2475]),
        'region':pd.Series(['d','f','d','d','f','f','f','d'])
        }

my_df = pd.DataFrame(data)
print(my_df)
print(my_df.shape)

print(my_df.sum(axis=0))
print(my_df.sum(axis=1))

print(my_df.mean())
print(my_df.mean(axis=1))

print(my_df.std(axis=0))

print(my_df.cumsum(axis=0))
print(my_df.cumproc(axis=0))

print(my_df.describe())
print(my_df.describe(include=['number']))
print(my_df.describe(include=['object','number']))
print(my_df.describe(include='all'))

def addfunc(e1,e2):
    return e1+e2

def multfunct(e1,e2):
    return e1*e2

my_df = pd.DataFrame(np.random.randn(5,4), columns=['c1','c2','c3','c4'])
print(my_df)

print(my_df.pipe(addfunc,2))
print(my_df)
my_df = my_df.pipe(addfunc,2)
print(my_df.pipe(multfunct,100))

print(my_df)
print(my_df.apply(np.mean))
print(my_df.mean())
print(my_df.apply(np.mean, axis=1))

print(my_df)
print(my_df.shape)
my_df2 = my_df.apply(lambda a: a.max()-a.min(), axis=1)
print(my_df2)
print(my_df2.shape)

print(my_df)
print(my_df['c1'].map(lambda a:a*100))
my_df2 = my_df.applymap(lambda a:a*100)
print(my_df2)