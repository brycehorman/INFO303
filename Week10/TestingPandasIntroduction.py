import pandas as pd
import numpy as np

s1 = pd.Series(dtype=object)
print(s1)

data = np.array(['Ally','Bill','Joe','Dan','Mitch','Edward'])
print(data)
print(type(data))
s1 = pd.Series(data)
print(s1)

data = np.arange(5,25,2)
data = data.reshape(2,5)
print(data)
s1 = pd.Series(data)
print(s1)
print(s1[5])

data = np.array(['Ally','Bill','Joe','Dan','Mitch','Edward'])
s2 = pd.Series(data, index=[100,101,102,103,104,105])
print(s2)
print(s2[100])

data = {'Ally':0, 'Bill':1, 'Charlie':2}
print(data['Bill'])
s1 = pd.Series(data)
s1 = pd.Series(data, index=['Ally','William','Chuck', 'Ally'])
print(s1)
print(s1['Bill'])

s1 = pd.Series(7, index=[0,1,2,3,4,5])
print(s1)

s1 = pd.Series([1,2,30,40,90], index=['a','b','cat','d','edward'])
print(s1)
print(s1[-3:])
print(s1[3:])

my_df = pd.DataFrame()
print(my_df)
my_list = [100,101,102,103,104,105]
my_df = pd.DataFrame(my_list)
print(my_df)

data = [['Ally',1000],['Robert',5000],['Sally', 7500]]
my_df = pd.DataFrame(data, columns=['Emps', 'Sales'], dtype=float)
print(my_df)

data = {'Emps':['Ally','Robert','Sally'], 'Sales':[1000,5000,7500]}
my_df = pd.DataFrame(data, index=['Emp1','Emp2','Emp3'])
print(my_df)

data = [{'Ally':1000,'Tim':44,'Robert':5000},{'Ally':7500,'Robert':9904,'Tim':3300}]
my_df = pd.DataFrame(data, index=['r1','r2'], columns=['Ally','Robert','Tim','Bill'])
print(my_df)

data = {'c1':pd.Series([10,20,30], index=['Ally','Tim','Robert']),
        'c2':pd.Series([15,25,35,45], index=['Ally','Tim','Robert','Jane'])}
my_df = pd.DataFrame(data)
print(my_df)

print(my_df['c2'])

my_df['c3'] = pd.Series([150,175,225,335], index=['Ally','Tim','Jane','Robert'])
print(my_df)

my_df['c4'] = my_df['c2'] + my_df['c3']
print(my_df)
del my_df['c4']
print(my_df)
my_df.pop('c3')
print(my_df)

print(my_df.loc['Jane'])
print(my_df.iloc[3])
print(my_df[2:])

print(my_df)
data = pd.DataFrame([[900,999]],columns=['c1','c2'], index=['Jane'])
my_df = my_df.append(data)
print(my_df)
my_df = my_df.drop('Ally')
print(my_df)