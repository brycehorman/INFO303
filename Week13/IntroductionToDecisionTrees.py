import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp, warnings
warnings.filterwarnings('ignore') #Let's turn off all warning messages

my_df = pd.read_csv('ProjectReturnData.txt', delimiter=';')

print(my_df.head(5))
print(my_df.info())
print(my_df.describe())
print(my_df.shape)

# It looks like we have 5181 observations (examples) and 4 features (dimensions) in this dataset.
# It does not appear that we are missing any data
print(my_df.isnull().sum())

pd.set_option('display.max_columns',10)
rp.summary_cont(my_df['Percent Return'].groupby(my_df['Leader Gender']))
rp.summary_cont(my_df['Percent Return'].groupby(my_df['Team Size']))
rp.summary_cont(my_df['Percent Return'].groupby(my_df['Aggregate Sales Experience']))

# Let's build a series of scatter plots to visualize our data
sns.pairplot(my_df, hue='Leader Gender', diag_kind='hist', kind='scatter', palette='husl')

# Let's map the genders to 0s and 1s....let's have the females be 1's and males be 0s
my_df['Leader Gender'] = my_df['Leader Gender'].map({'Male': 0, 'Female': 1})
print(my_df['Leader Gender'].head(5))

from sklearn.model_selection import train_test_split

ind_data = np.column_stack((my_df['Team Size'], my_df['Leader Gender'], my_df['Aggregate Sales Experience']))
dep_data = my_df['Percent Return'].to_numpy().reshape(my_df.shape[0],1)

ind_train, ind_test, dep_train, dep_test = train_test_split(ind_data, dep_data, test_size=0.4, random_state=23)

# let's try to fit a linear regression model first.
# We will use this model to see if our DecisionTree algorithm does any better.
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(ind_train, dep_train)

# Compute model predictions for test data
results = model.predict(ind_test)

# Compute score and display result (Coefficient of Determination)
score = 100.0 * model.score(ind_test, dep_test)
print(f'Multivariate LR Model score = {score:5.2f}%')

# Let's now apply the decision tree algorithm to classification tasks. To do this we will use the
# DecisionTreeClassifier estimator from the scikit-learn tree module. This estimator will construct,
# by default, a tree from a training data set. This estimator accepts a number of hyperparameters, including:


#    max_depth : the maximum depth of the tree. By default this is None, which means the tree is
#                constructed until either all leaf nodes are pure, or all leaf nodes contain fewer
#                instances than the min_samples_split hyperparameter value.
#    min_samples_split : the minimum number of instances required to split a node into two
#                 child nodes, by default this is two.
#    min_samples_leaf: the minimum number of instances required to make a node terminal
#                (i.e., a leaf node). By default this value is one.
#    max_features: the number of features to examine when choosing the best split feature and value.
#                By default this is None, which means all features will be explored.
#    random_state: the seed for the random number generator used by this estimator.
#                Setting this value ensures reproducibility.
#    class_weight: values that can improve classification performance on
#                unbalanced data sets. By default this value is None.

from sklearn.tree import DecisionTreeRegressor

# Create Regressor with default properties
# ccp_alpha is the complexity parameter used for Minimal Cost-Complexity Pruning.
# The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen.
# By default, no pruning is performed.
projects_model = DecisionTreeRegressor(random_state=23, min_samples_leaf=20, ccp_alpha=50)

# Fit estimator and display score
trained_model = projects_model.fit(ind_train, dep_train)
print('Score = {:.1%}'.format(projects_model.score(ind_test, dep_test)))
# Notice the difference in model fit with the unseen test data!!!!!

# let's print some other performance metrics
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

# Regress on test data
pred = trained_model.predict(ind_test)
# Compute performance metrics
mae = mean_absolute_error(dep_test, pred)
mse = mean_squared_error(dep_test, pred)
mbe = median_absolute_error(dep_test, pred)
mr2 = r2_score(dep_test, pred)

ev_score = explained_variance_score(dep_test, pred)

# Display metrics
print(f'Mean Absolute Error   = {mae:4.2f}')
print(f'Mean Squared Error    = {mse:4.2f}')
print(f'Median Absolute Error = {mbe:4.2f}')
print(f'R^2 Score             = {mr2:5.3f}')
print(f'Explained Variance    = {ev_score:5.3f}')


#Let's now display the relative importance of each feature on the Decision Tree
# Feature names
feature_names = ['Team Size', 'Leader Gender', 'Aggregate Sales Experience']

# Display name and importance
for name, val in zip(feature_names, projects_model.feature_importances_):
    print(f'{name} importance = {100.0*val:5.2f}%')

# Let's try to visualize the decision tree: Visualizing the Tree

# The sklearn learn library includes an export_graphviz method that actually generates a visual tree
# representation of a constructed decision tree classifier. This representation is in the
# dot format recognized by the standard, open source graphviz library.

# let's attempt to export a dot format of the Iris decision tree we just constructed,
# convert it to an SVG image, and subsequently display this image.

# If we have graphivz installed, one can simply
# perform the following steps to create and visualize a tree.
#
from sklearn.tree import export_graphviz

# Write the dot representation of the tree to a file
with open('tree.dot', 'w') as fdot:
    export_graphviz(projects_model, fdot, feature_names=feature_names)

# I don't have anything installed on my machine to render this .dot file as an image
# so let's go to the http://www.webgraphviz.com/ to display it online!