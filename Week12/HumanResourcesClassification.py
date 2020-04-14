import warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp
from matplotlib.colors import ListedColormap

# Turn off warnings just in case our KNeighborsClassifier throws a warning that will cause our program to pause
warnings.filterwarnings('ignore')

# Let's load the data into a Pandas DataFrame using the read_csv
my_df = pd.read_csv('EmpData.txt',
                    delimiter='|', index_col=0)

print(my_df.head(5))
print(my_df.info())
print(my_df.describe())
print(my_df.shape)

# It looks like we have 5497 observations (examples) and 6 features (dimensions) but one of those dimensions
# is the unique identifier.

# Notice how we are missing data in the currentsalary data column
# We have to decide what to do with the missing data.
# We could delete the examples/observations or we could impute (estimate)
# the missing values.  This decision is not made in isolation by a single individual.
# Let's use the average for the low, medium and high groups to impute the missing values.
print(my_df.isnull().sum())

pd.set_option('display.max_columns',10)
rp.summary_cont(my_df['currentsalary'].groupby(my_df['flightrisk']))

my_df1 = my_df.query('flightrisk=="high"')
my_avg1 = my_df1['currentsalary'].mean()
#my_df1['currentsalary'] = my_df1['currentsalary'].fillna(my_avg1)
my_df1['currentsalary'].fillna(my_avg1, inplace=True)
print(my_df1)

my_df2 = my_df.query('flightrisk=="medium"')
my_avg2 = my_df2['currentsalary'].mean()
my_df2['currentsalary'].fillna(my_avg2, inplace=True)
print(my_df2)

my_df3 = my_df.query('flightrisk=="low"')
my_avg3 = my_df3['currentsalary'].mean()
my_df3['currentsalary'].fillna(my_avg3, inplace=True)

my_data = pd.concat([my_df1, my_df2, my_df3])
# At this point our data should be organized/sorted by flightrisk (high, medium and then low)
# It might be easier to export your data in its current form to a csv file so you can look at it in excel
# with the assumption that it is not too big to fit into Excel.
my_data.to_csv('testingHRExample.csv')
print(my_data.isnull().sum())

rp.summary_cont(my_data['currentsalary'].groupby(my_data['flightrisk']))
rp.summary_cont(my_data['joblevel'].groupby(my_df['flightrisk']))
rp.summary_cont(my_data['yearseducation'].groupby(my_df['flightrisk']))
rp.summary_cont(my_data['travelamt'].groupby(my_df['flightrisk']))

rp.summary_cont(my_data['currentsalary'].groupby(my_data['gender']))
rp.summary_cont(my_data['joblevel'].groupby(my_df['gender']))
rp.summary_cont(my_data['yearseducation'].groupby(my_df['gender']))
rp.summary_cont(my_data['travelamt'].groupby(my_df['gender']))

# Now, let's visualize the relationships between the different features (columns/attributes/variables)
sns.pairplot(my_data, hue='flightrisk', diag_kind='hist', kind='scatter', palette='husl')
sns.pairplot(my_data, hue='gender', diag_kind='hist', kind='scatter', palette='husl')

# Now, we are ready to use sklearn or scikit-learn to run our knn.
# Recall that sklearn works with a numpy ndarray and not a pandas DataFrame so we will have to do a conversion
# Note: that it must be two dimensional so this often involves reshaping

#Let's first create the 2-dimensional ndarray by selecting the data from the my_df DataFrame by using the values attribute.
print(my_data.head(5))
# Let's code the Females as 1s and Males as 2s
my_data['gender2'] = my_data['gender'].map({'Female': 1, 'Male': 0})
print(my_data.head(5))
my_ndarray = my_data[['currentsalary', 'joblevel', 'yearseducation', 'travelamt', 'gender2']].values
print(my_ndarray)
#Now, let's create a numerical array for the data types, where I will map 0,1,and 2 into high, medium, and low

# Notice how the first 1879 are for high, next 1799 are for medium, and last 1819 are for low
# We will use this order of our instances to create the labels (0, 1, and 2)
highs = np.zeros(1879)
mediums = np.ones(1799)
lows = np.full(1819,2)
labels = np.append(highs,mediums)
labels = np.append(labels, lows)
print(labels)
print(labels.shape)

#Now, let's create a list of label names so we can reference back to the label names from the 0,1,2 aliases later
label_names = ['high', 'medium', 'low']

from sklearn.model_selection import train_test_split
# test_size is a parameter that we set.  0.3 means that 30% of our data will be reserved for testing and
# 70% will be used to generate the model.
# What is a good rule of thumb for how to do this split?
# There are two competing concerns: with less training data, your parameter
# estimates have greater variance. However, with less testing data, your performance
# statistic will have greater variance. Broadly speaking you should be concerned
# with dividing data such that neither variance is too high.

# RandomState initiates a repeatable random sequence so our results can be reproducible.
# For this example, let's split our data 50/50 between training and testing/validating.
d_train, d_test, l_train, l_test = train_test_split(my_ndarray, labels, test_size=0.5, random_state=15)

# Let's rescale our data because this distance classification algorithm
# can be sensitive to variables with larger variances,
# which could lead to sub-optimal results.
# The scaling should be applied to both the training and testing data sets.

from sklearn.preprocessing import StandardScaler

# Create and fit the StandardScaler
sc = StandardScaler().fit(d_train) #StandardScaler object
d_train_sc = sc.transform(d_train)
d_test_sc = sc.transform(d_test)

# Note: My males are now coded as -1.498 and my females are coded as 0.667, which isn't exactly correct
# because we don't have a linear relationship between males and females along this standardized sacle.
# Let's go back to our zeros and ones
print(d_train_sc)
# Now I will loop over and change my males and females back to the 0s (males) and to the 1s (females)
for x in np.nditer(d_train_sc[:,4], op_flags = ['readwrite']):
    #x = 2 * x  # Note how this will not work because is the iterable variable and not part of the a ndarray!
    #print(x[...])
    if np.round(x,2) == 0.68:
       x[...] = 1 # Females
    else:
       x[...] = 0 # Males
print(d_train_sc)

# Repeat the process for the test data
for x in np.nditer(d_test_sc[:,4], op_flags = ['readwrite']):
    #x = 2 * x  # Note how this will not work because is the iterable variable and not part of the a ndarray!
    #print(x[...])
    if np.round(x,2) == 0.68:
       x[...] = 1 # Females
    else:
       x[...] = 0 # Males
print(d_test_sc)

# Let's take a quick look at the first few rows of data to view the transformation
print(d_train[:5])
for val, val_sc in zip(d_train[:5], d_train_sc[:5]):
    print(val, val_sc, sc.inverse_transform(val_sc))
    #Note how we can back transform the data using the inverse_transform functon (will not work for
    # gender because I already changed them back to 0s and 1s)

# Create the appropriate scikit-learn estimator.
# k-Nearest Neighbors - Classify a test instance to a group (cluster) based on how close the data point is to
# a specified number of neighbors.  The 'k' is specified by the analyst.
# first fit the model to the training data set and then subsequently apply this model to predict values
# for the testing data.  We will determine accuracy using the score method to compare predicted and known
# labels for the testing data.

from sklearn import neighbors

#specify the k
nbrs = 3

# Let's now construct the model
# The default metric is minkowski

#euclidean: supports the standard concept of spatial distance
#manhattan: restricts distance measurements to follow grid lines.This metric is sometimes referred
#           to as the Taxi cab distance, since taxis must follow streets, which
#haversine: calculates the distance travelled over the surface of a sphere, such as the Earth.
#chebyshev: assumes the distance is equal to the greatest distance along the individual dimensions.
#minkowski: a generalization of the Manhattan and Euclidean distances to arbitrary powers.

#Weights is another important hyperparameter that we can set. The two most common options are:
#‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
#and
#‘distance’ : weight points by the inverse of their distance, which means that closer
# neighbors of a query point will have a greater influence than neighbors which are further away.

knn = neighbors.KNeighborsClassifier(n_neighbors=nbrs, metric='manhattan', weights='uniform')

#Now, train the model by passing in the training data and the training labels (classifier)
knn.fit(d_train_sc, l_train)

# Compute the score & display the model accuracy with the testing/validation data set!
score_test = 100 * knn.score(d_test_sc, l_test)
print(f'KNN ({nbrs} neighbors) prediction accuracy with test data = {score_test:.1f}%')

# Score with training data to see if we have an overfitting problem (which would result in a
# large difference in accuracy between training and validation/testing)
score_train = 100 * knn.score(d_train_sc, l_train)
print(f'KNN ({nbrs} neighbors) prediction accuracy with training data = {score_train:.1f}%')

# Let's look at some performance metrics beyond just the score, which is excellent in this case!
# One of the common ways to understand our performance is to create and display a
# confusion matrix. A confusion matrix has rows that correspond to the true labels
# and columns that correspond to the predicted labels. The elements of the confusion
# matrix contain the number of instances with true label given by the row index and
# the predicted label by the column index.
# A perfect classification, therefore, would have a confusion matrix populated entirely along the diagonal!

from sklearn.metrics import confusion_matrix

# Generate predictions
l_pred = knn.predict(d_test_sc)
# Create and display confusion matrix
print(confusion_matrix(l_test, l_pred))

# This is a litte hard to interpret because we don't have any labels indicating which axes
# are the predicted and which are the actual

def confusion(test, predict, title, labels):
    """Plot the confusion matrix to make it easier to interpret.
       This function produces a colored heatmap that displays the relationship
        between predicted and actual types from a machine learning method."""

    # Make a 2D histogram from the test and result arrays
    # pts is essentially the output of the scikit-learn confusion_matrix method
    pts, xe, ye = np.histogram2d(test, predict, bins=3)

    # For simplicity we create a new DataFrame for the confusion matrix
    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)

    # Display heatmap and add decorations
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('True Label', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)

    return None

# Call confusion matrix plotting routine
confusion(l_test, l_pred, f'KNN-({nbrs}) Model', label_names)
# When I am right nobody remembers but when I am wrong nobody forgets!!!
# That is how we remember and evaluate a machine learning algorithm!

# We can also print off a report of other fit metrics
from sklearn.metrics import classification_report

# Compute and display classification report
# support is the sample size
print(classification_report(l_test, l_pred, target_names = label_names))

#We can save (persist) the model if we want to use it later (without re-training the data) using joblib
from sklearn.externals import joblib
filename = 'knn-model.pkl'
with open(filename, 'wb') as fout:
    joblib.dump(knn, fout)

# Now let's show the decision surface area for our solution.
# A decision surface is a visualization that shows a particular space occupied by the training data.
# This has the effect of showing how new test data points would be classified as they move around the plot region.
# Let's demonstrate this with current salary, which is column index position 0
# and
# travelamt, which is column index position 3
# Notice how we build the decision surface with the training (not the testing data)
sns.set()
my_sdata = np.zeros((d_train_sc.shape[0], 3))
my_sdata[:, 0] = d_train_sc[:, 0]
my_sdata[:, 1] = d_train_sc[:, 3]
my_sdata[:, 2] = l_train[:]

# We will have to create a mesh grid containing all pairs of points in our visual
# Conceptualize a mesh grid as a piece of graph paper where I put in a point
# for every x and y coordinate (based on a specified grid size)
def get_mdata(data, grid_size=500):
    # We grab the min and max of the points, and make the space a bit bigger.
    x_min, x_max = data[:, 0].min() - .25, data[:, 0].max() + .25
    y_min, y_max = data[:, 1].min() - .25, data[:, 1].max() + .25

    # Meshgrid gives two 2D arrays of the points
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))

    # We want to return these points as an array of two-dimensional points
    # Stack 1-D arrays as columns into a 2-D array.
    return np.c_[xx.ravel(), yy.ravel()]


# Construct mesh grid data
mdata = get_mdata(my_sdata)

# I will create a function to help me build my plots
# This will essentially plot out the mesh grid.
# Then, I plot the data points on top of the mesh grid (which is our decision surface area)
def splot_data(ax, data, mdata, z, label1, label2, sz, grid_size=500):
    """This function constructs our plot"""
    cmap_back = ListedColormap(sns.hls_palette(3, l=.4, s=.1))
    cmap_pts = ListedColormap(sns.hls_palette(3, l=.9, s=.5))

    ax.set_aspect('equal') # one unit on the x axis is equal to one unit on the y axis

    # Set the x and y axis labels on our the plot
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    # We need grid points and values to make the colormesh plot
    xx = mdata[:, 0].reshape((grid_size, grid_size))
    yy = mdata[:, 1].reshape((grid_size, grid_size))
    zz = z.reshape((grid_size, grid_size))

    ax.pcolormesh(xx, yy, zz, cmap=cmap_back, alpha=0.9)

    # Now draw the points, with bolder colors.
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=sz, cmap=cmap_pts)


#5 uniform manhattan....compare with 3 distance euclidean
sns.set()
nbrs = 3
wgts='distance'
metr='euclidean'
knn = neighbors.KNeighborsClassifier(n_neighbors=nbrs, metric=metr, weights=wgts)
# Now, train the model by passing in the training data and the training labels (classifier)
# for just currentsalary and salary amount
knn.fit(my_sdata[:, :2], my_sdata[:, 2])
# Predict for mesh grid
z = knn.predict(mdata) #predict for each point along the mesh grid (which is each x,y coordinate combination)
# Plot training data and mesh grid
fig, axs = plt.subplots(figsize=(10, 20), nrows=1, ncols=1)
splot_data(axs, my_sdata, mdata, z, 'CurrentSalary', 'TravelAmount', 50)
axs.set_title(f'{nbrs}-Nearest Neighbors for {wgts} weights & {metr} distance calculation')
plt.text(-1.5,0,'Low', color='w')
plt.text(0,0,'Medium', color='w')
plt.text(1.5,0,'High', color='w')
plt.show()