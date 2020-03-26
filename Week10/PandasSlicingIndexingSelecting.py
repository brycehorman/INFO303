import pandas as pd
import numpy as np

my_df = pd.DataFrame(np.random.randn(8,4),
                     index=range(100,108),
                     columns=['bill','joe','sally','sue'])
print(my_df)
print(my_df.shape)
print(my_df.loc[:,'sally'])
print(my_df.loc[101:104,'sally'])
print(my_df.loc[103:106,['sally','bill']])

my_df = pd.DataFrame(np.random.randn(8,4),
                     index=['a','b','c','d','e','f','g','h'],
                     columns=['bill','joe','sally','sue'])
print(my_df)
print(my_df.loc[:,['sally','bill','joe']])
print(my_df.loc['a':'d',['sally','bill','joe']])
print(my_df.loc[['a','f','h'],['sally','bill','joe']])

print(my_df.loc['a']>0)
print(my_df.loc['a':'b',['joe','sue']]>0)

print(my_df)
print(my_df.iloc[:4])
print(my_df.iloc[1:5])

print(my_df.iloc[1:3,:2])
print(my_df.iloc[[0,3,7],:])
print(my_df.iloc[[0,3,7]])
print(my_df.iloc[[0,3,7],[0,3]])
print(my_df.iloc[:,[0,3]])
print(my_df.iloc['a']) #will not work

print(my_df['sue'])
print(my_df[['sue','bill']])
print(my_df[0:2])
print(my_df[0:0])
print(my_df[0:2]['sue'])

print(my_df.bill)
print(my_df.sue[0:4])