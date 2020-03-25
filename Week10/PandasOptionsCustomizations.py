import pandas as pd
import numpy as np

print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_columns'))
print(pd.get_option('display.max_colwidth'))
print(pd.get_option('display.precision'))

print(pd.describe_option('display.max_rows'))
print(pd.describe_option('display.precision'))

pd.set_option('display.max_columns', 5)
print(pd.get_option('display.max_columns'))

my_df = pd.DataFrame(np.arange(1, 1001).reshape(10,100))
print(my_df)

pd.set_option('display.max_rows', 10)
print(pd.get_option('display.max_columns'))

my_df = pd.DataFrame(np.arange(1, 1001).reshape(100,10))
print(my_df)

pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')
print(pd.get_option('display.max_rows'))
print(pd.get_option('display.max_columns'))

pd.set_option('display.precision', 5)
my_df = pd.DataFrame(np.random.randn(10))
print(my_df)

with pd.option_context('display.precision',2):
    print(my_df)
print(my_df)
