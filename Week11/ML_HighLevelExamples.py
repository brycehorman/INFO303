import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp

# A few of these algorithms may throw warning messages, which will interupt its processing
# Therefore, let's turn them off!
import warnings
warnings.filterwarnings('ignore')

# Let's use the iris sample dataset for this example
my_df = sns.load_dataset('iris')
# With this data set, we have rows, which correspond to different instances (e.g., different flowers)
# , and columns, which correspond to different features of the instances (e.g., different attributes or measurements
# of the flowers).

# Whenever, we get a new dataset, we want to describe our data so we understand what features we have available
# and the descriptive statistics of our instances.
sns.set_style('white')
my_df.head(5)
# we can also use the sample function to grab a random sample of instances
my_df.sample(5) #Notice how we get a different sample of five each time
print(my_df.describe())
# As we can see, these data consist of three types, each with 50 instances, and every row has four measured features
# the four features are sepal_length, sepal width, petal length, and petal width!
# From my understanding, petals are the showy, colorful part of the flower,
# and the sepals provide protection and support for the petals.

# let's group the data by species of the iris plant
pd.set_option('display.max_columns',10)
rp.summary_cont(my_df['sepal_width'].groupby(my_df['species']))
rp.summary_cont(my_df['sepal_length'].groupby(my_df['species']))
rp.summary_cont(my_df['petal_width'].groupby(my_df['species']))
rp.summary_cont(my_df['petal_length'].groupby(my_df['species']))

# Note, we are also looking for missing data here.  We don't have any missing data so we are good.
# If, we had missing data, then we could dropna's or impute (estimate) the missing data points using a fillna function.

#Now, let's visualize the relationships between the different features (columns/attributes/variables)
sns.pairplot(my_df, hue='species', diag_kind='hist', kind='scatter', palette='husl')

#From these pairplots, we can see that the three iris species appear to cluster naturally in these dimensions!

# Now, we are ready to use sklearn or scikit-learn.
# You will have to install this package if you have not already done so
# python -m pip install sklearn
# or
# python -m pip install scikit-learn
# Note: The above statements might vary depending on your machine.
# sklearn works with a numpy ndarray and not a pandas DataFrame (argh!!!!!)
# Therefore, our next step in this process is to build a two dimensional ndarray.
# Note: that it must be two dimensional so this often involves reshaping
print(type(my_df))
pd.set_option('display.max_rows',200)
print(my_df)
#Notice how the first 50 are for setosa, next 50 are versicolor, and last 50 are for virginica
#We will use this order of our instances to create the labels (0,1,and 2)

#Let's first create the 2-dimensional ndarray by selecting the data from the my_df DataFrame by using the values attribute.
my_ndarray = my_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
print(my_ndarray)
#Now, let's create a numerical array for the data types, where I will map 0,1,and 2 into setosa, versicolor and virginica species

labels = np.array([i//50 for i in range(my_df.shape[0])])
print(labels)
#Now, let's create a list of label names so we can reference back to the label names from the 0,1,2 aliases later
label_names = ['Setosa', 'Versicolor', 'Virginica']

# Now we are ready to run a few different machine learning algorithms against these data!
# The general steps to follow in sklearn are as follows:
# 1. Perform data pre-processing by randomly splitting the dataset into a training and testing dataset. The training
#    dataset is used to generate the model while the testing data are used to quantify the quality of the
#    generated model.
# 2. Rescale your data (this is not technically mandatory but it is certainly a best practice for most algorithms).
#    Options include: standardization, normalization, range, and binarization.
# 3.  Given the application type (classification, regression, dimensional reduction, or clustering),
#    create the appropriate scikit-learn estimator.
# 4. Determine the best hyper-parameters for the machine learning algorithm (number of neighbors for the k-nearest neighbors algorithm)
# 5. Extract the feature matrix (our ndarray data array) and, if appropriate, the target vector (our labels array)
# 6. Apply the fit method for the selected estimator to generate a best fit model.
# 7. Apply the generated model to new data by calling:
#    a. predict: method for classification and rgeression applications
#    b. transform: method for dimension reduction applications
#    c. either predict or transform for clustering applications, depending o whether data are being assigned
#       to clusters (predict) or the data are being converted so that the distances from each data point
#       to each cluster center are computed (transform)
# 8. Compute the efficacy of the machine learning algorithm for test data by calling an appropriate score method
#    and/or other available/appropriate performance metrics.

# sklearn will have to be installed!

#CLASSIFICATION example of unsupervised using k-nearest neighbors
# Let's split the data into training and testing data sets.
from sklearn.model_selection import train_test_split
# test_size is a parameter that we set.  0.4 means that 40% of our data will be reserved for testing and
# 60% will be used to generate the model.  We can tweak this number to see how different algorithms perform with
# more or less training data.
# RandomState initiates a repeatable random sequence.  If we don't set this keyword argument, then our results
# will not be reproducible.
d_train, d_test, l_train, l_test = train_test_split(my_ndarray, labels, test_size=0.4, random_state=23)

# Rescale our data because this distance classification algorithm is sensitive to variables with larger variances,
# which could lead to sub-optimal results.
# The scaling should be applied to both the training and testing data sets.  Said differently, the testing
# data should always match the training data

from sklearn.preprocessing import StandardScaler

# Create and fit scaler
sc = StandardScaler().fit(d_train) #StandardScaler object
d_train_sc = sc.transform(d_train)
d_test_sc = sc.transform(d_test)

# Let's take a quick look at the first 10 rows to see the transformation
print(d_train[:10])
for val, val_sc in zip(d_train[:10], d_train_sc[:10]):
    print(val, val_sc)
# print out the min and max
# MIN ignores the logical values passed as arguments, and MINA takes them into account in the search process.
print(np.min(d_train_sc, axis=0))
print(np.max(d_train_sc, axis=0))

# Create the appropriate scikit-learn estimator.
# k-Nearest Neighbors - Classify a test instance to a group (cluster) based on how close the data point is to
# a specified number of neighbors.  The 'k' is specified by the analyst.
# first fit the model to the training data set and then subsequently apply this model to predict values
# for the testing data.  We will determine accuracy using the score method to compare predicted and known
# labels for the testing data.

from sklearn import neighbors

#specify the k
nbrs = 5

#Let's now construct the model
knn = neighbors.KNeighborsClassifier(n_neighbors=nbrs)

#Now, train the model by passing in the training data and the training labels (classifier)
knn.fit(d_train_sc, l_train)

#Compute the score & display the model accuracy
score = 100 * knn.score(d_test_sc, l_test)
print(f'KNN ({nbrs} neighbors) prediction accuracy = {score:.1f}%')

#We can save (persist) the model if we want to use it later (without re-training the data) using joblib
from sklearn.externals import joblib
filename = 'knn-model.pkl'
with open(filename, 'wb') as fout:
    joblib.dump(knn, fout)

# Open model file and load model
with open(filename, 'rb') as fin:
    new_knn = joblib.load(fin)

# Compute and display accuracy score
score = 100.0 * new_knn.score(d_test_sc, l_test)
print(f"New KNN model ({nbrs} neighbors) prediction accuracy = {score:5.1f}%\n")

# Now, let's demonstrate a regression classifier (supervised learning example) using a decision tree
# A decision tree simply asks a set of questions of the data, and based on the answers, constructs a model representation.
# The tree (or model) is constructed by recursively splitting a data set into new groupings based on the
# statistical measure of the data along each dimension.

# In a decision tree, the terminal nodes are referred to as leaf nodes and they provide the final predictions.
# In its simplest form, the leaf node simply provides the final prediction.

# Before generating the regression model, we must pre-process our data to identify our independent variables (features)
# and our dependent variable (feature).

# For this analysis, let's make the last column the dependent variable and the first three columns the
# independent variables.  This means we will attempt to predict petal_width.
# For a regression, having standardized scales is not that important so we will just use the non-transformed data for this example.
ind_data = my_ndarray[:,0:3]
dep_data = my_ndarray[:,-1]
from sklearn.model_selection import train_test_split

d_train, d_test, r_train, r_test = train_test_split(ind_data, dep_data, test_size=0.4, random_state=23)

# Now let's create the decision tree regressor, fit the estimator to the training data, and apply the
# model to make predictions.  We can then calculate a performace score for the resulting model by comparing
# the predictions to the actual values for the testing data.

from sklearn import tree
#construct the tree regressor
dtr = tree.DecisionTreeRegressor()

#fit the regression to the training data
dtr.fit(d_train, r_train)

#compute and display score from the test data
score = 100 * dtr.score(d_test, r_test)
print(f'DT regression accuracy = {score:.1f}%')

# sklearn calculates the importance of each feature using
# "gini importance" or "mean decrease impurity", which is defined as the
# total decrease in node impurity (weighted by the probability of reaching
# that node (which is approximated by the proportion of samples reaching
# that node)) averaged over all trees of the ensemble.
print(dtr.feature_importances_)
# 'sepal_length', 'sepal_width', 'petal_length', 'petal_width' is the target (dv)


# Dimension Reduction Example
# This is the idea that with a complex problem involving many features, we might have to reduce the
# number of features (dimensions) that must be processed.  We can certainly do this subjectively based
# on our expert business knowledge but we can also employ a dimension reduction machine learning algorithm.
# These algorithms can quantify the relationships between the original dimensions (features/attributes/columns)
# to better capture the inherent relationships in the data.

# The main approach here is principal component analysis (PCA).  Through  linear algebra, PCA rotates the data
# into a new set of dimensions and ranks the importance of the new dimensions.  Then, we can mathematically select
# fewer dimensions to use in other machine learning algorithms.

# PCA is built-in to scikit-learn!  We only have to set a single hyper-parameter for the target number of dimensions.
# This value can be arbitrary or it can be done iteratively.  Once we create the model, we fit the model to the
# data and create a new (rotated) data set with the fewer columns.

from sklearn.decomposition import PCA

# First, we will create the PCA model. Let's use two features as our hyper-parameter in this example
pca = PCA(n_components=2)

# Fit the model to the data
pca.fit(my_ndarray)

# Compute the transformed data (rotation to the PCA space)
data_reduced = pca.transform(my_ndarray)

# Let's now construct a DataFrame to hold the results
cols = ['PCA1', 'PCA2', 'Species']
# Start with a temporary ndarray
tmp_d = np.concatenate((data_reduced, my_df['species'].values.reshape((150,1))), axis=1)
print(tmp_d)
pca_df = pd.DataFrame(tmp_d, columns=cols)
print(pca_df)

for idx, evr in enumerate(pca.explained_variance_ratio_):
    print(f'Component {idx} explains {100*evr:.2f} of the original variance')
# Note how these two new dimensions capture almost 98% of the variance in the original data.
# Finally, what is the equation that the algorithm used to create these two dimensions (from the original four dimensions?
col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
for row in pca.components_:
    print(r' + '.join('{0:6.3f} * {1:s}'.format(val, name) for val, name in zip(row, col_names)))

# Clustering Example
# Let's use the k-means algorithm.  Here, we start with a guess for the number of clusters (hyper-parameter that is
# determined based on an educated guess or iteratively).  We then randomly place cluster centers in the data
# and determine how well the data cluster to these cluster centers.  This information is used to pick cluster centers
# and this process continues until a solution converges (or we reach a pre0defined number of iterations).

# In scikit-learn, we can use the KMeans estimator in the cluster module.
# Steps to follow:
# 1. We pass in the number of clusters (k)
# 2. We pass the unscaled data because we really want to cluster the original cluster centers.
# 3. After the model is created, we fit the model to the data to obtain our predictions.  We can also
#    specify a value for the random_state hyper-parameter to ensure reproducibility.
# NOTE: This process is unsupervised so we do not use any label array in this process.

# After we find our clusters, we plot the original data and the new cluster centers to visually quantify how well
# the algorithm performed.
from sklearn.cluster import KMeans

# Let's build the model with 3 clusters
k_means = KMeans(n_clusters=3, random_state=23)
# Fit our data to the model
k_means.fit(my_ndarray)
print(k_means.cluster_centers_)

#Now let's build a scatter plot with the cluster centers
from matplotlib import cm

sns.set()
sns.set_style('white')
#Cluster centers
xcc = k_means.cluster_centers_[:, 1]
ycc = k_means.cluster_centers_[:, 3]
#Original data
x = my_ndarray[:, 1]
y = my_ndarray[:, 3]

# Now we create our figure and axes for the plot we will make.
fig, ax = plt.subplots(figsize=(10, 10))

for idx in np.unique(labels):
    # Convert index into an int
    i = int(idx)
    ax.scatter(x[labels == i], y[labels == i], label=f'{label_names[i]}',
               s=200, alpha=.5, cmap=cm.coolwarm)

# Plot cluster centers
ax.scatter(xcc, ycc, marker='*', label='Cluster Centers',
           s=500, alpha=0.75, cmap=cm.coolwarm)

# Decorate and clean plot
ax.set_xlabel('Sepal Width', fontsize=16)
ax.set_ylabel('Petal Width', fontsize=16)
ax.legend(loc=7, labelspacing=2)
ax.set_title('K-Means Cluster Demonstration', fontsize=18)
sns.despine(offset=0, trim=True)