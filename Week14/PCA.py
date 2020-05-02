# One approach to simplify a large, multi-dimensional data set is to reduce the number of
# dimensions that must be processed to mke the algorithms converge to solutions more efficiently
# and to develop simpler models. In some cases, dimensions can be removed from analysis
# based on business logic. More generally, however, we can employ machine learning to seek
# out relationships between the original dimensions/features (or columns of a DataFrame) to
# identify new dimensions that better capture the inherent relationships within the data.

# Here, we explore how to effectively use dimension reduction to reduce the number of features
# in a given data set. Dimension reduction is generally an unsupervised learning technique,
# and the reduction in features (or dimensions) can be important for a number of reasons.
# First, fewer features can increase the performance of machine learning algorithms
# (since there is less data to process). Second, the results may be more robust
# since dimensions that contain less information can be excluded
# (thus an algorithm will be forced to focus on the most important dimensions or features).
#
# Here, we will focus on identifying those dimensions (or features) that contain the most
# signal or information so that algorithms can focus on just those dimensions.

# While there are a number of algorithms that have been developed to perform dimension reduction,
# the most popular technique is principal component analysis or PCA. PCA is an important tool
# for exploratory data analysis, since PCA can quickly identify the most relevant features.

# In mathematical terms, PCA is used to identify the eigenvalues and eigenvectors of a system
# of linear equations that specify a transformation into a new dimensional space.
# These eigenvalues contain the expected variance, or amount of the underlying signal,
# in the data and thus provide insight into the new dimensions (specified by the eigenvectors)
# that contain the most information. Thus, we can select a subset of the new dimensions to
# reduce the dimensionality of our data set.

import warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp
warnings.filterwarnings('ignore')

my_data = pd.read_csv('EmpData.txt', delimiter='|', index_col=0)

print(my_data.head(5))
print(my_data.info())
print(my_data.describe())
print(my_data.shape)

# Check for missing data
print(my_data.isnull().sum())

# Gender is entered as strings so we will have to convert them
my_data['gender'] = my_data['gender'].map({'Female': 1, 'Male': 0})
print(my_data['gender'])

corrMatrix = my_data.corr()
print (corrMatrix)
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Let's make a few plots and grids to visualize our results
pd.set_option('display.max_columns',10)
rp.summary_cont(my_data['currentsalary'].groupby(my_data['flightrisk']))
rp.summary_cont(my_data['joblevel'].groupby(my_data['flightrisk']))
rp.summary_cont(my_data['yearseducation'].groupby(my_data['flightrisk']))
rp.summary_cont(my_data['travelamt'].groupby(my_data['flightrisk']))

rp.summary_cont(my_data['currentsalary'].groupby(my_data['gender']))
rp.summary_cont(my_data['joblevel'].groupby(my_data['gender']))
rp.summary_cont(my_data['yearseducation'].groupby(my_data['gender']))
rp.summary_cont(my_data['travelamt'].groupby(my_data['gender']))

# Now, let's visualize the relationships between the different features (columns/attributes/variables)
sns.pairplot(my_data, hue='flightrisk', diag_kind='hist', kind='scatter', palette='husl')
sns.pairplot(my_data, hue='gender', diag_kind='hist', kind='scatter', palette='husl')

# Mathematically, we can derive PCA by using linear algebra to solve a set of linear equations.
# This process effectively rotates the data into a new set of dimensions, and also provides
# a ranking of each new feature's importance. Thus, by employing PCA, we can often identify
# a reduced set of features that we can use to perform machine learning, while retaining the
# majority of the signal we wish to model. This can result in faster computations and
# reliable models since we have reduced the amount of noise used to train the model.

# The following code sample and associated figure demonstrate principal component analysis.
# The first block of code generates a distribution of data that are spread out in the
# original space. This is done by generating 2500 points that lie along the y=x
# line with added noise.
# Next, we display the expected principal components by drawing a new coordinate
# system over the data (this corresponds to transforming the original coordinate
# system in the same manner we transformed the original, random data).

# Define random state
rng = np.random.RandomState(19)

# Number of points to plot
num_pts = 2500

# Plot mask as image
fig, axs = plt.subplots(figsize=(8, 8))

# Generate points
x = rng.normal(0, 2.5, num_pts)
y = x + rng.normal(0, 0.75, num_pts)

# Define new cordinate axes, rotated 45 degrees
theta1 = np.deg2rad(45.0)
theta2 = np.deg2rad(135.0)

# Rotate points and plot
xp = x*np.cos(theta1)
yp = y*np.sin(theta1)
axs.scatter(xp, yp, s=25, c='b', alpha=0.1)

# Draw new coordinate system
axs.arrow(0, 0, 4 * np.cos(theta1), 4 * np.sin(theta1),
          lw=3, fc="k", ec="k", head_width=0.25, head_length=0.25)
axs.arrow(0, 0, 2 * np.cos(theta2), 2 * np.sin(theta2),
          lw=3, fc="k", ec="k", head_width=0.25, head_length=0.25)

# Decorate plot
axs.set(title='PCA Demonstration (Original)', xlabel='X', xlim=(-7, 7), ylabel='Y', ylim=(-7, 7))
sns.despine(offset=2, trim=True)

# To demonstrate how PCA affects a data distribution, we can use the PCA transformer
# in the sklearn library. The PCA transformer requires one tunable hyperparameter
# that specifies the target number of dimensions. This value can be arbitrarily selected,
# perhaps based on prior information, or it can be iteratively determined.
# After the transformer is created, we fit the model to the data to determine the
# principal components. Finally, we display the original data in these new, transformed coordinates.

from sklearn.decomposition import PCA

# Compute Principal Components
d = PCA().fit(np.stack((x, y)))

# Plot mask as image
fig, axs = plt.subplots(figsize=(8, 8))

# display data in new components
axs.scatter(d.components_[0], d.components_[1], s=50, c='b', alpha=0.25)

# Decorate plot
axs.set(title='PCA Demonstration (Transformed)', xlabel='$X^{\prime}$', xlim=(-0.1,.1), ylabel='$Y^{\prime}$', ylim=(-0.1,.1))
sns.despine(offset=2, trim=True)

# Now, let's work with the employee data instead of the 'made-up' data in the previous example!
# Extract features from the pandas my_data and convert them to a numpy ndarray

features = my_data[['currentsalary', 'joblevel',
                 'yearseducation', 'travelamt', 'gender']].values
print(features)

# Principal Component Analysis
pca = PCA()

# Fit model to the data
pca.fit(features)

# We can print out rotation matrix and variance associated with each projected dimension

vars = pca.explained_variance_ratio_
c_names = ['currentsalary','joblevel', 'yearseducation','travelamt','gender']

print('Variance:  Projected dimension')
print('------------------------------')
for idx, row in enumerate(pca.components_):
    output = '{0:4.1f}%:    '.format(100.0 * vars[idx])
    output += " + ".join("{0:5.2f} * {1:s}".format(val, name) \
                      for val, name in zip(row, c_names))
    print(output)

# This output indicates a single feature PC1 explains all of the variance!
# Why is this the case?
# PCA works based on variance and the salary feature contains more variance than
# the other features.  To fix, let's standardize the scales of our four features.
from sklearn.preprocessing import StandardScaler

# Create and fit the StandardScaler
sc = StandardScaler().fit(features) #StandardScaler object
features_sc = sc.transform(features)
print(features_sc)
#Let's convert our males and females back to zeros and ones!
for x in np.nditer(features_sc[:,4], op_flags = ['readwrite']):
    if np.round(x,2) == 0.67:
       x[...] = 1 # Females
    else:
       x[...] = 0 # Males
print(features_sc)
print(np.mean(features_sc))

# Fit model to the data
pca.fit(features_sc)

# We can print out rotation matrix and variance associated with each projected dimension

vars = pca.explained_variance_ratio_
c_names = ['currentsalary','joblevel', 'yearseducation','travelamt','gender']

print('Variance:  Projected dimension')
print('------------------------------')
for idx, row in enumerate(pca.components_):
    output = '{0:4.1f}%:    '.format(100.0 * vars[idx])
    output += " + ".join("{0:5.2f} * {1:s}".format(val, name) \
                      for val, name in zip(row, c_names))
    print(output)

# In this case, we see that the first three new dimensions capture
# the majority of the total variance in the data.
# Why is this the case?  Because not all of our features are not highly correlated.
# Is this typical?  This is difficult to say because it depends on the business problem, the features, and the data!
# Will we lose some predictive power by eliminating 10% of the variance if we only use 3 components?
# By reducing the number of dimensions, we will inevitably lose some
# signal (predictive power) by removing dimensions, but often the loss is small to insignificant, and,
# for big data sets in the billions of rows this dimension reduction
# makes it computationally feasible.

# To demonstrate the power of dimension reduction, let's explore the impact on
# classification of this dataset by using a regression decision tree on the full five-dimensional data set
# and on the projected features.

# We analyze the full data by using a Regression Decision Tree

rp.summary_cont(my_data['currentsalary'].groupby(my_data['flightrisk']))
my_data['flightrisk2'] = my_data['flightrisk'].map({'low': 0, 'medium': 1, 'high': 2})
rp.summary_cont(my_data['currentsalary'].groupby(my_data['flightrisk2']))
print(my_data.head(5))

labels = my_data['flightrisk2'].values.reshape(my_data.shape[0],1)
print(labels)

from sklearn.model_selection import train_test_split

f_train, f_test, l_train, l_test = train_test_split(features_sc, labels, test_size=0.5, random_state=23)

from sklearn.tree import DecisionTreeRegressor

# Lets build our model and train it all at once
projects_model = DecisionTreeRegressor(random_state=23, min_samples_leaf=150)

# Fit estimator and display score
trained_model = projects_model.fit(f_train, l_train)
print('Score = {:.1%}'.format(projects_model.score(f_test, l_test)))

# Now, let's use the PCA...reducing the number of features from 5 to 3.
# The model is fine but this model is worse, why is it?
# Why reduce the number of dimensions if our goal is to maximize predictive power?
# For speed of processing with larger data sets!  Going from 1000 features to, say 5 features, reduces the
# number of calculations required to converge to a solution by several orders of magnitude!
pca = PCA(n_components=2, random_state=23)
features_reduced = pca.fit_transform(features_sc)

f_train, f_test, l_train, l_test = train_test_split(features_reduced, labels, test_size=0.5, random_state=23)

projects_model = DecisionTreeRegressor(random_state=23, min_samples_leaf=150)

# Fit estimator and display score
trained_model = projects_model.fit(f_train, l_train)
print('Score = {:.1%}'.format(projects_model.score(f_test, l_test)))