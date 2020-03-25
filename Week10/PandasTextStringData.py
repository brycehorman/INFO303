import pandas as pd
import numpy as np

my_series = pd.Series(['Tom','Mattson    ', '   Bill  ',np.nan, 'Al@gmail.com','detroit','matty','123'])
print(my_series)

print(my_series.str.lower())
print(my_series.str.upper())
print(my_series.str.swapcase())
print(my_series.str.isupper())
print(my_series.str.islower())
print(my_series.str.isnumeric())
print(my_series.str.startswith('m'))
print(my_series.str.strip().str.endswith('n'))
print(my_series.str.findall(' '))

print(my_series.str.find('o'))
print(my_series.str.findall('t'))

print(my_series.str.len())

print(my_series.str.strip())
print(my_series.str.cat(sep='_'))
print(my_series.str.strip().str.cat(sep='_'))

print(my_series.str.get_dummies())

print(my_series.str.contains('t'))

print(my_series.str.replace('@','|'))
print(my_series.str.replace('m','@'))

print(my_series.str.repeat(3))

print(my_series.str.count('t'))

