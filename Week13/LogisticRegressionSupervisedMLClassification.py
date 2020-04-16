import warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp

# Turn off warnings just in case our KNeighborsClassifier throws a warning that will cause our program to pause
warnings.filterwarnings('ignore')

# Now, let's load the json file into a DataFrame
my_df = pd.read_json('Contracts.json', orient='columns')
print(my_df.head(10))

#Let's split each column in the dictionary into a separate series in the DataFrame as opposed to a single column/series
my_list = []
for my_dict in my_df['contracts']:
    my_list.append(my_dict)

print(my_list)
my_contracts_df = pd.DataFrame(my_list)
print(my_contracts_df.head(5))
#I want to use the ContractPK column as the index instead of the 0,1,2,3,4,5, etc. that gets entered by default
my_contracts_df.set_index('ContractPK', inplace=True)
print(my_contracts_df.head(5))

pd.set_option('display.max_columns',10)
#To describe the data, let's construct a correlation matrix to see how correlated our data are
corrMatrix = my_contracts_df.corr()
print (corrMatrix)
#For as many features as we have in these data, they don't seem to be excessively correlated.
sns.heatmap(corrMatrix, annot=True)
plt.show()

print(my_contracts_df.info())

rp.summary_cont(my_contracts_df['QuotedPrice'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['NumberofSocialMediaConnections'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['SizeOfSalesTeam'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['SalesTeamExperience'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['NumberPriorPurchases'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['CreditPercentage'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['InterestRate'].groupby(my_contracts_df['Status']))
rp.summary_cont(my_contracts_df['FinanceTermMonths'].groupby(my_contracts_df['Status']))

# I do notice that the scales are significantly different, especially for the salary so I will probably re-scale them.
# This should not impact the predictive power of the model but it should make the coefficients easier to interpret.

# In a binary classification process, we have two possible outcomes,
# which for the sake of generality, we can label as Won or Lost (for the contract).
# Denoting the probability of these two outcomes as P(W) and P(L) respectively,
# we can write the probability of success as P(S)=p, and the probability of loss as
# P(L)=1−p
# Therefore, the odds of a successful outcome (winning the contract), which is the ratio of the
# probability of winning the contract to the probability of losing the contract.

# We can extend the framework of linear regression to the task of binary classification
# by employing a mapping between the continuous value predicted by a linear regressor
# and the probability of an event occurring, which is bounded by the range [0,1].
# To do this, we need a function that maps the real numbers into this range,
# which enables a regression onto a set of discrete values (0 or 1) that provides
# us the binary classification.

# The logit function is defined as the logarithm of the odds (i.e, p/(1−p)),
# which is also known as the log-odds. Therefore, the logit function
# can be written for a probability of winning the contract p

# logit(p)=log(p / (1−p))
# where 0 ≤ p ≤ 1

# We can invert this relationship to obtain the logistic function, which for a parameter α
# is defined by the following expression:
# logistic(α) = 1 / (1+exp(−α))
# this logistic function is sometimes referred to as a sigmoid function.

# The next block of code graphically shows the shape of the logistic function
# Notice how it is bounded between 0 and 1

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 5))

# Compute and plot logistic function with a few random values
x = np.linspace(-10, 10, 100)
# The logistic function for our y axis, which will be computed for each value of x in the x numpy ndarray
y = 1 / (1 + np.exp(-x))
ax.plot(x, y, alpha=0.75)

# Draw probability barrier (50%...above greater likelihood, below decreased likelihood)
ax.hlines(0.5, -10, 10, linestyles='--')

# Clean up or dress up the plot to make it more professional
ax.set_xlabel(r'$\alpha$', fontsize=16)
ax.set_ylabel(r'$p$', fontsize=16)
ax.set_title('Logistic Function', fontsize=18)
sns.despine(offset = 2, trim=True)

# The generally used cost (or loss for the individual examples) function for logistic regression is the
# sum of the squared errors between the actual classes and the predicted classes (L2 losses).
#
# One of the most popular techniques for finding the minimum of this cost function is to
# use stochastic gradient descent. Gradient descent computes the derivative of (or finds
# the slope of the tangent line to) the cost function at a particular point. This can be
# used to modify the parameters of our model to move in a direction that is expected to
# reach the minimum of the cost function.
# Standard gradient descent computes these corrections by summing up all the contributions
# from each training data point.
#
# In stochastic gradient descent (or SGD), however, the corrections are computed for a random
# training point. As a result, SGD often generates a path towards the minimum that
# is somewhat rambling, but this has the benefit of avoiding local minima and being more robust.

# The following block of code generates a figure to help explain gradient descent.
# Here, I use a fictitious cost function, along with the tangent (or derivative) at a
# particular point. The arrows specify the direction that the derivative indicates we must move
# to reach the minimum. The repeated use of arrows signifies how an incremental approach to
# gradient descent, such as that employed by stochastic gradient descent, might
# converge to the true minimum of the cost function.

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 5))
#
# # Compute and plot logistic function
# x = np.linspace(-4, 4, 100)
# # Just building a curve because the curve will clearly show the minimum point
# y = x**2
# ax.plot(x, y, alpha=0.75) # alpha is just used for color blending
#
# # Draw probability barrier
# ax.plot([1, 2, 3], [0, 4, 8], ls='--')
# ax.scatter(0, 0, marker='x', c='r', s=100)
#
# # Add text and arrow annotations
ax.annotate('Descent', xy=(1.75, 4), xytext=(2.0, 6), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))

ax.annotate('Gradient', xy=(2, 4), xytext=(2.5, 1), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))

ax.annotate('Minimum', xy=(0, 0), xytext=(-0.5, 8), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))

# Add extra arrows
ax.annotate('', xy=(1.45, 3), xytext=(1.75, 4), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))
ax.annotate('', xy=(1.15, 2), xytext=(1.45, 3), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))
ax.annotate('', xy=(0.65, 1), xytext=(1.15, 2), fontsize=14,
    arrowprops=dict(arrowstyle="->", linewidth=2))

# Limit plot boundaries
ax.set_xlim(-1, 3)
ax.set_ylim(-0.5, 10)

# Decorate plot to make it more professional
ax.set_xlabel(r'$x$', fontsize=16)
ax.set_ylabel(r'$y$', fontsize=16)
ax.set_title('Gradient Descent Example', fontsize=18)
sns.despine(offset = 2, trim=True)

# The scikit-learn (sklearn) library has a standard LogisticRegression estimator and an SGDRegression estimator.
# In this example, I will demonstrate both estimators; however, the latter is a general technique
# that uses SGD to minimize the cost function. By specifying the log value to the loss
# hyperparameter, we can use the SGDRegression estimator to perform logistic regression.
# This can be very useful, since, by default, the LogisticRegression estimator performs
# regularization. Regularization is a technique to minimize the likelihood of over-fitting
# and works by penalizing complex models,
# which, in effect, alters the coefficients of our logistic regression model.

# let's convert the Lost contracts to 0 and Won contracts to 1
# Our model will predict whether a contract is won (the 1s)
# It might make sense to build a separate column instead of overriding the existing the
# column just in case you 'mess' up the statement.
my_contracts_df['Status'] = my_contracts_df['Status'].map({'Lost': 0, 'Won': 1})
print(my_contracts_df['Status'])

# In this first code snippet, let's use the LogisticRegression estimator.
# This estimator accepts a number of hyperparameters (values set by the analyst),
# of which the most important for our purposes include:

#   C: inverse of regularization strength...smaller values specify stronger regularization.
#    Regularization is the degree of idiosyncratic structure of the curve.  In gerenal, we want
#    a smooth (non-idiosyncratic curve) that will fit both the training and the testing (validation)
#    data set.
#    class_weight: weights to be applied to classes when performing regression, default is uniform
#    penalty: type of regularization to be applied, can be l1 ( mean absolute deviation)or l2 (least square errors),
#    fit_intercept: specifies if a constant term should be included in the regression, the default is True
#    random_state: the seed used to initialize the random number generator, a constant value ensures reproducibility.
#    solver: ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
#     Algorithm to use in the optimization problem.
#     Limited-memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm

from sklearn.linear_model import LogisticRegression
#C=1E6 # reduce the penalty for idiosyncratic models...smaller means smoother curve.
model = LogisticRegression(C=1)

from sklearn.model_selection import train_test_split

my_features = my_contracts_df[['QuotedPrice','NumberofSocialMediaConnections','SizeOfSalesTeam',
                               'SalesTeamExperience','NumberPriorPurchases','CreditPercentage',
                               'InterestRate','FinanceTermMonths']].values
print(my_features)

labels = my_contracts_df['Status'].values.reshape(my_contracts_df.shape[0],1)
print(labels)

# Evaluate the model by splitting into train and test (validation) sets
# Notice the stratify keyword argument.
# Roughly 40% of our data are lost contracts and 60% are won contracts.
# We want our random testing and training data sets to have close to this same ratio.
# Otherwise, we might be training or testing based on a biased sample.
x_train, x_test, y_train, y_test = train_test_split(my_features, labels, test_size=0.4,
                                                    stratify = labels,
                                                    random_state=23)

# Now, let's standardize our features so that they are all on the same scales.
# This should help with the interpretation of our coefficients in our model.
from sklearn.preprocessing import StandardScaler

# Create and fit the StandardScaler
sc = StandardScaler().fit(x_train) #StandardScaler object
x_train_sc = sc.transform(x_train)
x_test_sc = sc.transform(x_test)

# Fit a new model and predict on test data
lr_model = model.fit(x_train_sc, y_train)
predicted = lr_model.predict(x_test_sc)

# Display the model values
print(f'LR Model Fit: {lr_model.intercept_[0]:4.2f} + {lr_model.coef_[0][0]:4.2f} * QuotedPrice + '
      f'{lr_model.coef_[0][1]:4.2f} * NumberofSocialMediaConnections + {lr_model.coef_[0][2]:4.2f} * SizeOfSalesTeam +'
      f'{lr_model.coef_[0][3]:4.2f} * SalesTeamExperience + {lr_model.coef_[0][4]:4.2f} * NumberPriorPurchases +'
      f'{lr_model.coef_[0][5]:4.2f} * CreditPercentage + {lr_model.coef_[0][6]:4.2f} * InterestRate +'
      f'{lr_model.coef_[0][7]:4.2f} * FinanceTermMonths')

# Display coefficients for different features using a loop instead of the previous statement
print(f'{lr_model.intercept_[0]:4.2f}')
for c, f in zip(lr_model.coef_[0], ['QuotedPrice','NumberofSocialMediaConnections','SizeOfSalesTeam',
                               'SalesTeamExperience','NumberPriorPurchases','CreditPercentage',
                               'InterestRate','FinanceTermMonths']):
    print(f'{c:5.2f} * {f}')

# Display data and predicted labels
for data, label in zip(x_test_sc, predicted):
    print(data, label)

from sklearn.metrics import accuracy_score, classification_report

# Generate and display different evaluation metrics
score = 100.0 * accuracy_score(y_test, predicted)
print(f'Logistic Regression [Winning a contract] Score = {score:4.1f}%\n')

print(classification_report(y_test, predicted))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test.reshape(y_test.shape[0]), predicted))

tn, fp, fn, tp = confusion_matrix(y_test.reshape(y_test.shape[0]), predicted).ravel()

print('Computed metrics from the confusion matrix')
print(42*'-')
print(f'Precision   = {100.0 * tp/(tp + fp):5.2f}%')
print(f'Accuracy    = {100.0 * (tp + tn)/(tp + tn + fp + fn):5.2f}%')
print(f'Recall      = {100.0 * tp/(tp + fn):5.2f}%')
print(f'F1-score    = {100.0 * 2 * tp/(2 * tp + fp + fn):5.2f}%')

# Instead of manually calculating these values, we can calculate them via the sklearn built-in functions
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print('Computed metrics from the scikit-learn module')
print(45*'-')
print(f'Precision   = {100.0 * precision_score(y_test, predicted):5.2f}%')
print(f'Accuracy    = {100.0 * accuracy_score(y_test, predicted):5.2f}%')
print(f'Recall      = {100.0 * recall_score(y_test, predicted):5.2f}%')
print(f'F1-score    = {100.0 * f1_score(y_test, predicted):5.2f}%')

def confusion(test, predict, title, labels):
    """Plot the confusion matrix to make it easier to interpret.
       This function produces a colored heatmap that displays the relationship
        between predicted and actual types from a machine learning method."""

    # Make a 2D histogram from the test and result arrays
    # pts is essentially the output of the scikit-learn confusion_matrix method
    pts, xe, ye = np.histogram2d(test, predict, bins=2)

    # For simplicity we create a new DataFrame for the confusion matrix
    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)

    # Display heatmap and add decorations
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('True Label', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)

    return None

# Call confusion matrix plotting routine
confusion(y_test.reshape(y_test.shape[0]), predicted, f'Predicting Winning a Contract', ['Won', 'Lost'])

# Another way to construct the model is to use the SGDClassifier
# The main difference is the algorithm that each classifier will use is slightly different.
# SGD stands for stochastic gradient descent.
# SGD is typically used for large-scale problems where it's very efficient.
# Compared to the others, it might be very dependent on chosen hyperparameters
# (learning-rate, decay, ...). Bad hyperparameters not only result in slow performance,
# but also bad results (global-min not reached).
from sklearn.linear_model import SGDClassifier

# Create SGD estimator with log loss function
# The ‘log’ loss gives logistic regression, a probabilistic classifier.
# One of the benefits of this classifier is that you can specify other types of loss functions
sgd_model = SGDClassifier(loss='log')

# Fit training data and predict for test data
sgd_model = sgd_model.fit(x_train_sc, y_train)
# Use that model to fit the testing data
predicted = sgd_model.predict(x_test_sc)

# Display performance metrics
score = 100.0 * accuracy_score(y_test, predicted)
print(f'Logistic Regression [Winning a contract] Score = {score:4.1f}%\n')
print('Classification Report:\n {0}\n'.format(classification_report(y_test, predicted)))

# Compute ROC (receiver operating characteristic), which is a graphical representation
# of the true positive rate (TPR) on the y-axis against the false positive rate (FPR) on the x-axis.
# Can be a good way to visualize how much better than simply flipping a coin, a particular model is.
from sklearn.metrics import roc_curve, auc

fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = model.fit(x_train_sc, y_train).decision_function(x_test_sc) # These are the decision estimates
# Use the roc_curve function to determine false positives and false negatives
fpr[0], tpr[0], _ = roc_curve(y_test[:, 0], y_score)
# AUC - ROC curve is a performance measurement for classification problem at various
# thresholds settings. ROC is a probability curve and AUC represents degree or measure
# of separability. It tells how much model is capable of distinguishing between classes.
roc_auc[0] = auc(fpr[0], tpr[0])
print(roc_auc[0])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Plot for a single feature (in this case let's just use NumberofSocialMediaConnections (standardized)
# (assuming zero (mean) for all other features).
x = np.linspace(-3, 3)
lgy = lr_model.intercept_[0] + lr_model.coef_[0][1] * x

# We subtract fit from one since in the data won = 1.
# A normal logit model is won = 0
y = 1 - 1.0 / (1.0 + np.exp(lgy))

# Make the plots
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data and model
print(y_test[:,0])
plt.scatter(x_test_sc[:,1], y_test[:,0],
            c='b', alpha=0.5, label=f'Measured Data')
ax.plot(x, y, c='g', alpha=0.5, lw=2, linestyle='--', label='Model')
print(y)

# Decorate plot appropriately
ax.set_title('Logistic Fit', fontsize=18)
ax.set_xlabel('Number of Social Media Connections', fontsize=16)
ax.set_ylabel('Probability of Winning', fontsize=16)
ax.set_xlim(-3, 3)
ax.set_ylim(-0.05, 1.05)
ax.legend(loc=7, fontsize=16)
sns.despine(offset=5, trim=True)