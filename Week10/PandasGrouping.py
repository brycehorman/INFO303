import pandas as pd
import numpy as np

my_data = {'office': ['dc','dc','nyc','nyc','boston','boston',
                      'boston','boston','dc','richmond','richmond','dc'],
           'salesvol':[15,12,13,19,5,6,11,8,9,10,11,12],
           'month':['Jan','Feb','Jan','Feb','Jan','Feb','Mar','Apr','Mar','Jan','Feb','Apr'],
           'satscore':[800,700,875,675,750,800,775,790,700,685,800,500]}

my_df = pd.DataFrame(my_data)
print(my_df)
print(my_df.groupby('office'))
print(my_df.groupby('office').groups)
print(my_df.groupby(['office','month']).groups)

grouped = my_df.groupby('office')
for name,group in grouped:
    print(name)
    print(group)

print(grouped.get_group('boston'))
print(grouped.get_group('nyc'))

print(grouped.agg(np.sum))
print(grouped.agg([np.size,np.sum,np.mean,np.std]))

print(grouped['salesvol'].agg(np.sum))
print(grouped['salesvol'].agg([np.mean,np.std]))

grouped = my_df.groupby(['office','month'])
for name,group in grouped:
    print(name)
    print(group)
print(grouped.get_group(('boston','Jan')))

grouped = my_df.groupby('office')
print(grouped)
tval = lambda a: (a-a.mean())/a.std()*10
ngrp = grouped.transform(tval)
print(ngrp)
print(ngrp.median())
print(type(ngrp))

grouped = my_df.groupby('office')
print(grouped)
print(grouped.agg(np.size))
print(grouped.filter(lambda a: len(a)==2))