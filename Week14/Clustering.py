# Clustering is generally an unsupervised learning technique,
# because we are using the data features to determine some concept
# of closeness among our observations.

# In this python script, I will provide an example of k-means, k-medians, k-medoid, and dbscan.

# The ultimate goal is to find similar and dissimilar others (in our case customers)

import warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp
warnings.filterwarnings('ignore')

# Note that this example is slightly different from previous examples where we randomly split our data
# into training and testing data.
# In this example, we have a training data set of customers but we don't know whether they bought a product
# based on a specific promotion (hence unsupervised!!!).  Therefore, we want to group them together based on
# common features without knowing their label/target (outcome).
# Then, with the testing data set, we want to figure out how well those clusters
# predict whether or not a customer will make a purchase when the company offers a similar promotion.
# This type of analysis is fairly common when we are segmenting our customers.  We might have a customer list
# with certain attributes but not their purchase histories, which makes it impossible to run a supervised
# machine learning algorithm.
# After we perform the cluster analysis, we obtain 'unseen' data related to their purchases (or some other label/target).
# Then, we can see if our customer segmentation was valuable or not in terms of predicting purchases based
# on a similar promotion.

# If we have a large enough sample of purchase data (related back to our customers data),
# we could run a supervised classification machine learning algorithm using a logistic regression.
# Normally, that type of supervised machine learning algorithm will have
# better predictive power but we don't always have that data available.  Yet, we still have a business need to
# segment our customers to determine which customers have similar attributes or previous purchase histories.

# Might also have a business need to cluster products.  Netflix might want to cluster different movies based on
# a variety of attributes (beyond the simple genre grouping) or a grocery store clustering customers based on
# their shopping baskets (along with other demographics)

# Another business use case for our clusters could be to run experiments or quasi-esperiments with our clusters.
# For instance, randomly send 1/2 the customers in Cluster #1 a promotion to see if they make more purchases relative
# to the other half of the customers in Cluster #1 (control group).  We can use these clusters for targeted
# experimentation.  Your online retailers (Amazon, Facebook, Microsoft, etc.) do this quite frequently.

my_df = pd.read_csv('CustomerList.txt', delimiter='|')
print(my_df.head(5))
print(my_df.info())
print(my_df.describe())
print(my_df.shape)

pd.set_option('display.max_columns',7)
rp.summary_cont(my_df['PriorPurchases'].groupby(my_df['AccountClass']))
# Normally, we display our label grouped by a set of features when describing our data
# for a supervised machine learning problem but we
# don't have a label with unsupervised machine learning problems so these reports are not that meaningful.

# Instead, let's construct a correlation matrix to see how correlated our features are
corrMatrix = my_df.corr()
print (corrMatrix)
sns.heatmap(corrMatrix, annot=True)
plt.show()

# Let's set a few global figure properties, which should limit the amount of properties we have to set later
import matplotlib as mpl
mpl.rcParams.update({'axes.titlesize' : 20,'axes.labelsize' : 18, 'legend.fontsize': 16})

# Set default Seaborn plotting style
sns.set_style('white')
sns.set_context('paper')
sns.pairplot(my_df, hue='NumberAssignedSalesStaff', diag_kind='hist', kind='scatter', palette='husl')

# In the k-means algorithm we start with a guess for k, the number of clusters
# (this guess can be based on prior information or iteratively quantified).
# We then randomly place cluster centers in the data and determine how well the
# data cluster to these cluster centers. This information is used to pick
# new cluster centers, via a weighting process (for the k-means algorithm we take
# the mean (or median if k-median instead of k-means) of the points assigned to
# each cluster), and the process continues until a solution converges (or we
# reach a predefined number of iterations).

# In general, the k-means algorithm can be quite fast. However, as N
# becomes larger and the number of features increase, this task can
# become computationally difficult since the distances
# between each cluster center and each data point must be repeatedly calculated.
# In addition, the k-means algorithm is iterative where the cluster centers are
# repeatedly updated, and new distances calculated.

# Finally, the k-means algorithm can become trapped in a local minima. To avoid this, the
# k-means algorithm is typically run multiple times, each with a different set of random
# cluster locations, and the best result is selected in the end.

# Given a set of random, initial cluster locations, the k-means algorithm has two steps,
# over which the algorithm iterates until the process is completed:

#    Assignment Step: Each data point is assigned to the closest cluster
#                     (k-means uses Euclidean distance).
#    Update Step: The new cluster centers are computed by taking the mean of all data points
#                 that have been assigned to each cluster.

# This repetitive process terminates either when there is no change to the data during the assignment step
# or when the pre-defined number of iterations has completed.

# We perform k-means clustering by using the KMeans estimator within the cluster module of the
# sklearn library. This algorithm accepts a number of hyperparameters that control its
# performance, some of the most commonly changed include:

#    n_clusters: the number of clusters, k, the algorithm will find, the default value is eight.
#    n_init: the number of times the algorithm is run with different initial cluster centers, the default is ten.
#    max_iter: maximum number of iterations for the algorithm in any given run, the default is 300.
#    random_state: random seed used by the random number generator, enables reproducibility.

my_ndarray = np.column_stack((my_df.AccountClass, my_df.NumberofStores,
                              my_df.PriorPurchases, my_df.NumberAssignedSalesStaff))
print(my_ndarray)
# There is not concept of a label in a true unsupervised machine learning algorithm.
# If we knew the label, we would probably be better off running a supervised classification algorithm!
# Often, I might put the primary key of the 'things' that I am clustering into a y ndarray but it
# is really not necessary!  As you will see, I don't use y unless I want to link the clusters back
# to the individual customers (via the common CustomerID) field.
y = my_df['CustomerID'].to_numpy().reshape(my_df.shape[0],1)
print(y)

# We will first use the kmeans library from sklearn to perform this analysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sklearn.metrics as skm

# We build our model assuming three clusters
# Let's also try a few different options for the number of clusters!
nclus = 3
k_means = KMeans(n_clusters=nclus, n_init=100, random_state=23)

# Kmeans is a distance algorithm (centroid to observations) so it can be
# sensitive to features with different scales.  We see this often when one feature contains salaries
# and another feature is a percentage (such as percent of work time spent on email or out of the office).
# Therefore, I am going to standardize the scales (mean 0 and standard deviation of 1).

# The next line of code should standardize the variables to have a mean of 0 and a standard deviation of 1.
x = StandardScaler().fit_transform(my_ndarray)

# We can use a Principal Component Analysis (PCA) to reduce our feature set from four features to two features
# This will use all features but convert those four features to two features (hyperparameter that we set).
# This will make our kmeans run faster and it will be significantly easier to interpret the results.

# NOTE: If you are going to run a PCA to reduce the number of features and you are going to re-scale the
#       features, then you should re-scale first and then perform the PCA.

pca = PCA(n_components=2, random_state=23)
pca.fit(x)
xt = pca.fit_transform(my_ndarray)
print(xt)

# We can print out the equation that was used to reduce the number of features from four to two
col_names = ['AccountClass', 'NumberOfStores', 'PriorPurchases', 'NumberAssignedSalesStaff']
for row in pca.components_:
    print(r' + '.join('{0:6.3f} * {1:s}'.format(val, name) for val, name in zip(row, col_names)))

# We fit our data using the k-means algorithm to assign the clusters
# Notice how I am using the xt numpy ndarray here because this contains the reduced feature set.
k_means.fit(xt)

# Compute cluster centers, and transform
# to principal component space
cc = k_means.cluster_centers_
cc_pca = pca.fit_transform(cc)

# Display cluster centers
for idx, xy_c in enumerate(cc_pca):
    print(f'Cluster {idx} center located at ({xy_c[0]:4.2f}, {xy_c[1]:4.2f})')

#Let's visualize the results of our cluster analysis
cols = ['PCA1', 'PCA2', 'Cluster']
# Transform the data to make plotting easier!
# this transformed data is a numpy ndarray with three columns
data = np.concatenate((xt, k_means.predict(xt).reshape(xt.shape[0],1)), axis=1)
print(data)
# Now, convert it to a pandas data frame
dt = pd.DataFrame(data, columns=cols)
print(dt)

#Set up for a handful of colors for the different clusters
clr = [sns.xkcd_rgb['pale red'],sns.xkcd_rgb['denim blue'],sns.xkcd_rgb['medium green'],sns.xkcd_rgb['dark green'],
       sns.xkcd_rgb['dark blue'],sns.xkcd_rgb['cyan'],sns.xkcd_rgb['aqua'],sns.xkcd_rgb['tan'],sns.xkcd_rgb['orange'],
       sns.xkcd_rgb['light green']
       ]
# Now make the plot
fig, ax = plt.subplots(figsize=(12, 10))
#nclus is the variable that contains the number of clusters
for x in range(nclus):
    my_tmpdf = dt.query(f'Cluster == {x}')
    ax.scatter(my_tmpdf['PCA1'], my_tmpdf['PCA2'], label=f'Class {x}', alpha=1, s=25, c=clr[x])

# Plot cluster centers
ax.scatter(cc_pca[:, 0], cc_pca[:, 1],s=150, c='k', marker='X', label='Cluster Center')

# Clean up the plot
ax.set(title=f'K-Means for {nclus} clusters', xlabel='PCA 1', ylabel='PCA 2')
ax.legend(bbox_to_anchor=(1.0, 1), loc='best')
sns.despine(offset=5, trim=True)

# So how do we know how well our k-means algorithm performed?  Is this a good or a bad cluster analysis?
# We can get a general idea of how well our model performed by calculating the model inertia,
# which is the sum total distance of every point to its cluster center.
# In general, lower is better than higher.
print(k_means.inertia_)

# The problem with this approach is that the number is difficult to contextualize (short of smaller is better!!!).
# Another common metric to evaluate the performance of un-labeled data is the Silhouette score.

# Silhouette: takes values in the range [-1, 1] and does not require ground
#             truth labels, which is good if we don't know the true labels.
#             This score is based on the intra-cluster distance and the mean
#             nearest-cluster distance (this is the distance to the closest
#             cluster to which the data were not assigned).
#             A value of one means highly dense clustering, and a value of
#             minus one indicates incorrect clustering (since the data are often
#             closer to a different cluster than the one to which they are assigned).
#             A value of zero indicates overlapping clusters

# Cluster labels
lbls = k_means.labels_
print(np.unique(lbls))

ss = skm.silhouette_score(xt.reshape(xt.shape[0], 2), lbls.reshape(xt.shape[0], 1),metric='euclidean')
print(f'Silhouette (l2) score      = {ss:5.3f}')

# We could write a loop to try clustering our data with different k's to see
# which value of k results in the best Silhouette score
# Another approach is to use the 'elbow method', which provides a heuristic
# estimate that can provide a reasonable value.

# The elbow method has several different variations, all of which rely on some measure of the
# quality with which a given number of clusters best fits the data.
# Here, I will use the cluster inertia, which is the sum total distance of every
# point to its cluster center. When the inertia is plotted against the number
# of clusters, the value starts high for small numbers of clusters, and as
# the number of clusters increases, this value should quickly decrease.
# This decrease arises since more clusters will, on average, reduce the distance
# between any given point and its cluster center.
# Eventually, however, the inertia levels off as existing clusters are broken
# into sub-groups. The best value for k is selected as the elbow point within this plot.

n_clusters = np.arange(1, 11)
distances = np.zeros(n_clusters.shape[0])

# Perform k-means clustering for different numbers of clusters
# Use the inertia (or sum of total distances between points
# and cluster centers) as the performance metric.
for idx, nc in enumerate(n_clusters):
    # We build our model for nc clusters
    model = KMeans(n_clusters=nc, n_init=100, random_state=23)
    model.fit(xt)
    distances[idx] = model.inertia_

# Plot elbow method
fig, ax = plt.subplots(figsize=(12, 10))

# Draw points and connect them
ax.scatter(n_clusters, distances, s=150,
           c=sns.xkcd_rgb['pale red'], marker='X', alpha=0.5)
ax.plot(n_clusters, distances, lw=3, linestyle='-',
        color=sns.xkcd_rgb['denim blue'])

# Define elbow at three clusters
elb = (n_clusters[2], distances[2])

# Draw an arrow showing the elbow
ax.annotate('Elbow', xytext=[6, 200000], xy=elb,
                arrowprops=dict(facecolor=sns.xkcd_rgb['dark green'],
                                alpha=0.25, shrink=0.05))

# Clean up our visual
ax.set(title='Elbow Plot', xlabel='Number of Clusters', ylabel='Total distance')
sns.despine(offset=5, trim=True)

# Let's now run a k-medians (instead of mean distance, let's calculate median distance)
# We can do this efficiently with the pyclustering library.
# If you don't have it installed, install it!
#python -m pip install pyclustering
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster import cluster_visualizer

# Create instance of K-Medians algorithm.
initial_medians = [[0.0, 0.1], [2.5, 0.7], [3.5, 1.5]]
kmedians_instance = kmedians(xt, initial_medians)
# Run cluster analysis and obtain results.
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = kmedians_instance.get_medians()
# How well did it do?
# Sum of metric errors is calculated using distance between point and its center
print(kmedians_instance.get_total_wce())

# Visualize clustering results.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, xt)
#visualizer.append_cluster(initial_medians, marker='*', markersize=10)
visualizer.append_cluster(medians, marker='*', markersize=10, color='k')
visualizer.show()

# DB-SCAN
# DB-SCAN automatically determines the number of clusters within a data set.
# Since the DBSCAN algorithm is a density-based clustering algorithm, the discovered
# clusters can have arbitrary shapes. On the other hand, since the clusters and
# their membership are defined by the density, the hyperparameters used to specify
# the target density can dramatically affect the cluster determination.
# Thus, hyperparameter tuning may be required to achieve optimal results.

# DB-SCAN is a density-based clustering algorithm, as opposed to spatial clustering
# algorithms such as k-means. Fundamentally, the DB-SCAN algorithm operates by
# classifying points. A point is a core point if a minimum number of points are
# within a given distance.
# Therefore, we have a density defined by ratio of the number of points to the
# volume enclosed within the specified distance. The DB-SCAN algorithm is deterministic,
# meaning that for a given ordering of a data set and algorithmic hyperparameters, the
# same set of clusters will always be found.
# Changing the ordering of the input data, however, can change the identified clusters.
#
# As a result, this algorithm takes two critical hyperparameters: eps (or ϵ) and min_samples.
# The eps hyperparameter defines the maximum distance between two points for them to
# still be considered in the same density neighborhood. The min_samples hyperparameter is
# the number of points that must lie within the neighborhood of the current point in order
# for it to be considered a core point.
#
# Clusters are defined by connecting points, which involves determining if there is a path
# between the core points. For this process, we have that a point is considered reachable
# from another point if there is a path consisting of core points between the starting and ending
# point. Any point that is not reachable is considered an outlier, or in the sklearn
# implementation, noise.

# In the sklearn library, we use the DBSCAN estimator in the cluster module to perform
# DBSCAN clustering. This estimator takes a number of hyperparameters that
# impact its performance, including:

#    eps: the maximum distance between two instances for them to be considered
#         in the same neighborhood, the default is 0.5.
#    min_samples: the number of instances in a neighborhood for a point to be
#         considered a core point, the default is five.
#    metric: the function used to compute distances between points, default is euclidean.

# Once the estimator has been created, we can use the fit method to determine the
# clusters in an array containing the instances to cluster. By default, this function creates
# a model attribute called labels_ that contain the cluster label for each instance in the data set.
# Alternatively, the fit_predict method can be called, which also returns the label array. Two other
# model attributes computed during the fitting process are core_sample_indices_, which contains
# the indices of the core points within the instance array that was provided to the fit method,
# and components_, which is an array consisting of the core points.

from sklearn.cluster import DBSCAN

# Apply DBSCAN
db = DBSCAN(eps=1, metric='euclidean', min_samples=7)
mdl = db.fit(xt)

print(f'Number of core points = {mdl.components_.shape[0]}')
print(f'Cluster labels: {np.unique(mdl.labels_)}')

# We can now determine the number of observations in each cluster.
# This will help us to fine-tune our hyperparameters
from collections import Counter
cnt =  Counter(np.sort(mdl.labels_))

# Display some basic results of the clustering
print('DBSCAN Cluster membership.')
print(30*'-')
for itm in cnt:
    if itm < 0:
        print(f'Noise Cluster : {cnt[itm]:>4d} members')
    else:
        print(f'Cluster {itm}     : {cnt[itm]:>4d} members')

# How did the DBSCAN cluster perform?
lbls = mdl.labels_
print(lbls)

ss = skm.silhouette_score(xt.reshape(xt.shape[0], 2), lbls.reshape(xt.shape[0], 1),metric='euclidean')
print(f'Silhouette (l2) score      = {ss:5.3f}')

# Let's visualize the results
# Transform core points to PCA space
cp = pca.transform(mdl.components_)

# Label data
cols = ['PCA1', 'PCA2', 'Cluster']
print(mdl.labels_) #Assigned labels
data = np.concatenate((xt, mdl.labels_.reshape(xt.shape[0],1)), axis=1)
dt = pd.DataFrame(data, columns=cols)
print(dt)

# Make plot
fig, ax = plt.subplots(figsize=(12, 10))

# Get cluster labels and assign plotting colors/labels.
dblbls = set(db.labels_)
dbclrs = sns.hls_palette(len(dblbls))
dbcls = ['Class {0}'.format(idx) if idx >= 0 else 'Noise' for idx in dblbls]

nolabel = np.unique(mdl.labels_).shape[0]
for x in range(nolabel):
    my_tmpdf = dt.query(f'Cluster == {x}')
    ax.scatter(my_tmpdf['PCA1'], my_tmpdf['PCA2'], label=dbcls[x], alpha=1, s=25, c=dbclrs[x])

# Plot core points
ax.scatter(cp[:, 0], cp[:, 1], label='Core Point', color=sns.xkcd_rgb['dusty purple'], marker='X', alpha=1, s=25)

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_title('Sample DBSCAN')
ax.legend(bbox_to_anchor=(1, 1), loc=2)
sns.despine(offset=5, trim=True)

# Now, let's say we want to use these clusters to predict whether the customers will make a purchase?
# To do so, we were able to obtain some purchase data.
# Note (repeated from above), if we had the purchase data ahead of time, we could
# have simply run a supervised machine learning classification algorithm.
# However, it is not uncommon to have a list of customers (or products or services) first.
# Then, we get the sales data a month or two later.  Yet, we want to act on those clusters of customers
# (or potential customers) before obtaining any purchase data.

# We can then supplement our unsupervised cluster analysis with a supervised learning algorithm later.
my_df_purchases = pd.read_csv('CustomerPurchases.txt', delimiter='|')
print(my_df_purchases.head(5))
print(my_df_purchases.info())
print(my_df_purchases.describe())
print(my_df_purchases.shape)

# These data points probably represent data where the customer made a purchase.
# Let's try to merge these with the original my_df so we can have those customers
# who made purchases and those that did not!
print(my_df.head())
my_df.set_index('CustomerID', inplace=True)
my_df_purchases.set_index('CustomerID', inplace=True)
my_combined = pd.merge(my_df, my_df_purchases, on='CustomerID', how='left')
my_combined['Purchased'] = my_combined['Purchased'].fillna(0)
print(my_combined)

# Now, grab the cluster from the original k-means result
assigned = k_means.predict(xt).reshape(xt.shape[0],1)
print(np.unique(assigned))
my_combined['Cluster'] = assigned
print(np.unique(my_combined['Cluster']))

# Let's run a logistic regression to predict purchases based on whether the customer was assigned to
# Cluster A, B, or C.
from sklearn.linear_model import LogisticRegression
#reduce the penalty for idiosyncratic models smaller means smoother curve.
model = LogisticRegression(C=1)

from sklearn.model_selection import train_test_split

my_features = my_combined['Cluster'].values.reshape(my_combined.shape[0],1)
print(np.unique(my_features))
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
rgrps = [0, 1, 2]
le = LabelEncoder()
le.fit(rgrps)

ohe = OneHotEncoder(sparse=False)
le_data = le.transform(my_features).reshape(my_combined.shape[0], 1)
ohecluster = ohe.fit_transform(le_data)
print(ohecluster)
enc=[0,0,1]
print(le.inverse_transform(np.argmax(enc).reshape(1,1)))

labels = my_combined['Purchased'].values.reshape(my_combined.shape[0],1)
print(labels)

# Evaluate the model by splitting into train and test sets
# Notice the stratify keyword argument.
# Roughly 40% of our data are lost contracts and 60% are won contracts.
# We want our random testing and training data sets to have close to this same ratio.
# Otherwise, we might be training or testing based on a biased sample.
x_train, x_test, y_train, y_test = train_test_split(ohecluster, labels, test_size=0.4,
                                                    stratify = labels,
                                                    random_state=23)
lr_model = model.fit(x_train, y_train)
predicted = lr_model.predict(x_test)
from sklearn.metrics import accuracy_score, classification_report
score = 100.0 * accuracy_score(y_test, predicted)
print(f'Logistic Regression [Making a Purchase] Score = {score:4.1f}%\n')
print(classification_report(y_test, predicted))