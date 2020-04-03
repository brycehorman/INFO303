import seaborn as sb, numpy as np, pandas as pd, matplotlib.pyplot as plt

# Seaborn will have to be installed
# python -m pip install seaborn
# On the lab computers, you may have to add the --user flag
# you may also need to install bs4, which contains the web scraping utility (BeautifulSoup)
# python -m pip install bs4
# # On the lab computers, you may have to add the --user flag
# There are several dependencies associated with the seaborn library such as numpy, scipy, pandas, matplotlib.

# Matplotlib tries to make easy things easy and difficult things possible
# Seaborn tries to make a well-defined set of hard things easy to do
# Seaborn is built on top of matplotlib.  It is designed to complement (not replace) matplotlib

# seaborn has built in themes for styling matplotlib visuals,
# can visualize univariate and bivariate data,
# linear regression plotting/visualization,
# plotting time series data,
# and works well with both numpy and pandas data structures.

# seaborn comes with a few built-in datasets that we can use to experiment with
print(sb.get_dataset_names())

# if we want to use one of these existing datasets, we can load them directly!
my_df = sb.load_dataset('tips')
print(type(my_df)) #notice that it is a DataFrame
print(my_df.head(10))
print(my_df.shape) # 244 rows by 7 columns
print(my_df.describe())
print(my_df.info())

# Visualizing data generally involves two steps:
# 1) creating the plot/visual and 2) making the visual more aesthetically pleasing
# Visualization is an art of representing data in an easy and effective way.

# Unlike Matplotlib, seaborn comes with customized themes and a high-level interface for
# customizing the look and feel of matplotlib graphics.

# Let's build five sin wave plots
# First let's use the standard matplotlib library
x = np.linspace(1,15,75)
for i in range(1,6):
    plt.plot(x, np.sin(x+i*0.75)*(8-i))
plt.show()

# Now let's use the seaborn defaults...sb.set() makes seaborn the default charting algorithm
sb.set() # establishes the seaborn defaults (plot styles and scales)
sb.set_style('ticks') # a few of the style options include darkgrid, whitegrid, dark, white, ticks
x = np.linspace(1,15,75)
for i in range(1,6):
    plt.plot(x, np.sin(x+i*0.75)*(8-i))
#plt.show()
#now remove the right and top axis spines
sb.despine() # we must call this function after we make the plot
plt.show()

# We can override the styles using a dictionary object
print(sb.axes_style())

# Let's add in the gridlines
sb.set_style('ticks', {'axes.grid': True, 'grid.color':'red', 'grid.linestyle':'--'})
x = np.linspace(1,15,75)
for i in range(1,6):
    plt.plot(x, np.sin(x+i*0.75)*(8-i))
plt.show()

# We can also control the context (plot elements and scale of the plot) using the set_context() function
# There are four built-in preset contexts (paper, notebook (default), talk, poster)
sb.set_style('white')
sb.set_context('poster')
x = np.linspace(1,15,75)
for i in range(1,6):
    plt.plot(x, np.sin(x+i*0.75)*(8-i))
plt.show()

# We can set the color palette in our visual
# There are several palettes that are readily available (deep, muted, bright, pastel, dark, and colorblind)
# To determine the current color palette, we can use the color_palette() function
print(sb.color_palette())  # Not terribly useful, it might be easier to visualize the palette.
sb.palplot(sb.color_palette())
# Let's view a few sequential color palette's
sb.palplot(sb.color_palette('Reds'))
sb.palplot(sb.color_palette('Greens'))
sb.palplot(sb.color_palette('Greys'))
sb.palplot(sb.color_palette('Blues', n_colors=8))
# Diverging palette's -1 to 0 one color and 0 to +1 another color
sb.palplot(sb.color_palette('BrBG', n_colors=10))
sb.palplot(sb.color_palette('hls', n_colors=10))
sb.palplot(sb.color_palette('husl', n_colors=10))
sb.palplot(sb.color_palette('Paired', n_colors=10))
sb.palplot(sb.color_palette('Set2', n_colors=10))

# Now, let's use a color scheme in one of our plots
sb.set_style('white')
sb.set_context('paper')
sb.set_palette('Paired',n_colors=4)  # Notice how we are setting the color palette here!
x = np.linspace(1,15,75)
for i in range(1,6):
    plt.plot(x, np.sin(x+i*0.75)*(8-i))
plt.show()

# One powerful feature of seaborn is the ability to plot distributions
# Let's experiment with the distplot() function, which provides a convenient way to visualize univariate distributions
# let's build a histogram, which displays the data distribution by forming bins along the range of the data and
# then drawing bars to show the number of observations that fall in each bin!

# For this set of examples, let's use the built-in iris dataset
my_df = sb.load_dataset('iris')
print(type(my_df))
print(my_df.head(10))
print(my_df.shape)
sb.set()
sb.set_style('white')
sb.distplot(my_df['petal_length'], bins=14)  # Plot with and without the bins argument
sb.distplot(my_df['petal_length'], hist=False) #Just the kde (kernel density estimation)
sb.distplot(my_df['petal_length'], hist=True, kde=True) #both the histogram and the kde

# Seaborn allows us to create scatter plots and to visualize the relationship (correlation) between two variables
# How one variable is behaving with respect to another variable.
# We can use the jointplot function to do this for us!
sb.set()
sb.jointplot(x=my_df['petal_length'], y=my_df['petal_width'])
sb.jointplot(x='petal_length', y='petal_width', data=my_df)
plt.show()

import scipy.stats as stats

sb.jointplot(x='petal_length', y='petal_width', data=my_df, kind='reg')
print(stats.pearsonr(my_df['petal_length'],my_df['petal_width']))
rsquared = round(stats.pearsonr(my_df['petal_length'],my_df['petal_width'])[0] ** 2,2)
pvalue = round(stats.pearsonr(my_df['petal_length'],my_df['petal_width'])[1],5)
plt.legend(labels=[f'r2 is {rsquared} with a pvalue of {pvalue}'], loc ='upper left') # Try the best option for loc of the legend
plt.show()

print(stats.linregress(my_df['petal_length'],my_df['petal_width']))
#Notice how this returns a tuple!
slope, intercept, r_value, p_value, std_err = stats.linregress(my_df['petal_length'],my_df['petal_width'])
print(slope)
print(intercept)
sb.jointplot(x='petal_length', y='petal_width', data=my_df, kind='reg')
plt.legend(labels=[f'intercept is {round(intercept,2)} and slope is {round(slope,2)}'], loc ='upper left')
plt.show()

# let's build a hexbin plot, which are used in bivariate analyses when the data are sparse, which sometimes
# makes it difficult to visualize through scatter plots
sb.set()
sb.jointplot(x='petal_length', y='petal_width', data=my_df, kind='hex')
plt.show()

#We can also build a kernel density estimation plot using jointplot and kde
sb.set()
sb.jointplot(x='petal_length', y='petal_width', data=my_df, kind='kde')
plt.show()

# Let's say we have to analyze multiple pairwise bivariate distributions (n,2) combinations
# In seaborn, we can do this using the pairplot() function
# hue is the variable in the dataset to map plot aspects to different colors
# diag_kind is the type of chart for the diagonals
sb.set_style('ticks')
sb.pairplot(my_df, hue='species', diag_kind='kde', kind='scatter', palette='husl')
#sb.pairplot(my_df, hue='species', diag_kind='hist', kind='scatter', palette='husl')
plt.show()

# We can also plot categorical data
my_df.info() #looks like the species column is a categorical column
my_df['species'].describe()
my_df.head(5)
# We can use the stripplot function to plot one categorical and one continuous variable
sb.set()
# This should show the difference in petal length by each species of this iris plant
sb.stripplot(x='species', y='petal_length', data=my_df)
plt.show()
# Notice how the points are overlapping
sb.stripplot(x='species', y='petal_length', data=my_df, jitter=True)
sb.stripplot(x='species', y='petal_length', data=my_df, jitter=0.20)
plt.show()

sb.set()
# The swarmplot is an alternative to the jitter with the stripplt().
# This will position each point on the scatter plot on the categorical x axis to
# avoid the overlapping problem.
sb.swarmplot(x='species', y='petal_length', data=my_df)
plt.show()

# box plot and violin plots to view spread or distribution of data
sb.set()
sb.boxplot(x='species',y='petal_length', data=my_df)
sb.violinplot(x='species',y='petal_length', data=my_df)
plt.show()

plt.savefig('testingseaborn.png')