#Name: Bryce Horman
#Running each block of code sequentially will give the desired plots, one by one.
import seaborn as sb
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import importlib
from scipy import stats
import json

#Problem 1
handle = open('Projects.txt', 'r')
reader = csv.reader(handle, delimiter=';')
rows = list(reader)
print(rows)
my_df = pd.DataFrame(rows[1:], columns=rows[0], dtype=np.float64)
my_df = my_df.rename(columns={'ManagerExperienceRating': 'Manager Experience Rating', 'ActualCosts':'Actual Costs'})
handle.close()
sb.set()
sb.set_style('white')
sb.set_palette('hls')
plot = sb.distplot(my_df[['ActualRevenue']], kde=False)
plot.set_title('Project Revenue Distribution All Projects')
plot.set_ylabel('Frequency')
plot.set_xlabel('Project Revenue')
plt.show()


#Problem 2
importlib.reload(sb)
sb.set()
sb.set_context('notebook')
sb.set_palette('Paired')
sb.set_style('whitegrid')
plot = sb.violinplot(x='ManagementStyle', y='ActualRevenue', data=my_df, kde=False)
plot.set_title('Management Style by Actual Revenue')
plot.set_ylabel('Actual Revenue')
plot.set_xlabel('Management Style')
plt.show()

#Problem 3
importlib.reload(sb)
sb.set()
sb.set_style('ticks')
sb.set_palette('husl')
plot = sb.jointplot(x='Manager Experience Rating', y='Actual Costs', data=my_df, kind='reg')
x = stats.linregress(my_df['Manager Experience Rating'], my_df['Actual Costs'])
plt.legend([f'intercept is {x[1]:.2f} and slope is {x[0]:.2f}'], loc='upper left')
plt.show()

#Problem 4
my_df['Profit'] = my_df['ActualRevenue'] - my_df['Actual Costs']
my_df['StartYear'] = pd.DatetimeIndex(my_df['StartDate']).year
my_df['Complexity']= my_df['Complexity'].apply(lambda a: 'Low' if a == 1.0 else ('Medium' if a == 2.0 else 'High'))
importlib.reload(sb)
sb.set()
sb.set_palette('Paired')
plot = sb.FacetGrid(my_df, col='Complexity', hue='StartYear', hue_order=[2018,2019], col_order=['Medium', 'Low', 'High'])
plot.map(plt.hist,'Profit')
plot.add_legend()
plt.show()

#Problem 5
my_df['Project Length'] = pd.to_datetime(my_df['EndDate']) - pd.to_datetime(my_df['StartDate'])
my_df['Project Length'] = my_df['Project Length'].apply(lambda a: a.days)
my_df['Percent Return'] = ((my_df['ActualRevenue'] - my_df['Actual Costs'])/my_df['Actual Costs'])*100
importlib.reload(sb)
sb.set()
sb.set_context('notebook')
sb.set_style('white')
sb.set_palette('Paired')
plot = sb.FacetGrid(my_df, col = 'Complexity', hue='StartYear')
plot.map(plt.scatter, 'Project Length', 'Percent Return')
plot.add_legend()
plt.show()

#Problem 6
frame = {'Actual Revenue': my_df['ActualRevenue'], 'Actual Costs': my_df['Actual Costs'], 'Manager Experience Rating': my_df['Manager Experience Rating'], 'ManagementStyle': my_df['ManagementStyle']}
my_df2 = pd.DataFrame(frame)
importlib.reload(sb)
sb.set()
sb.set_style('white')
sb.set_palette('husl') #color is not exactly how i want it
plot = sb.pairplot(my_df2, hue='ManagementStyle')
plt.show()


#Problem 7
handle = open('Managers.json', 'r')
json_loaded = json.load(handle)
handle.close()
def getGender(id):
    for dict in json_loaded['managers']:
        if float(dict['mgrid']) == id:
            return [dict['Gender'], dict['YearsOfEducation'], dict['HireDate']]
my_df['Manager Gender'] = my_df['ManagerID'].apply(lambda a: getGender(a)[0])
my_df['Manager Years of Education'] = my_df['ManagerID'].apply(lambda a: getGender(a)[1])
my_df['HireDate'] = my_df['ManagerID'].apply(lambda a: getGender(a)[2])
importlib.reload(sb)
sb.set()
sb.set_style('white')
sb.set_palette('Paired')
plot = sb.FacetGrid(data=my_df, hue='Complexity', col='Manager Gender')
plot.map(plt.scatter, 'Actual Costs', 'Manager Years of Education')
plot.add_legend()
plt.show()

#Problem 8
importlib.reload(sb)
sb.set()
sb.set_style('white')
fig, ax = plt.subplots(2, 2, sharex='col')
plot1 = sb.stripplot(x='Manager Years of Education', y='Actual Costs', data=my_df, ax=ax[0,0], color='Orange')
plot1.set_title('Years of Education by Costs')
plot2 = sb.stripplot(x='Manager Years of Education', y='ActualRevenue', data=my_df, ax=ax[0,1], color='Blue')
plot2.set_title('Years of Education by Revenue')
plot3 = sb.stripplot(x='Manager Years of Education', y='Profit', data=my_df, ax=ax[1,0], color='Green')
plot3.set_title('Years of Education by Profit')
plot4 = sb.stripplot(x='Manager Years of Education', y='Percent Return', data=my_df, ax=ax[1,1], color='Red')
plot4.set_title('Years of Education by Percent Return')
ax[0,0].set_xlabel('')
ax[0,0].set_ylabel('')
ax[1,1].set_xlabel('')
ax[1,1].set_ylabel('')
ax[0,1].set_xlabel('')
ax[0,1].set_ylabel('')
ax[1,0].set_xlabel('')
ax[1,0].set_ylabel('')
plt.show()
