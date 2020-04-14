import warnings, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns, researchpy as rp

# Turn off warnings just in case our KNeighborsClassifier throws a warning that will cause our program to pause
warnings.filterwarnings('ignore')

# In this example, we will use regression to fit a model to data.
# The model generated in this fashion can be explored in greater detail
# to either understand why the provided data follow the generated model
# (i.e., gain insight into the data), or the model can be used to
# generate new dependent values from future or unseen data
# (i.e., make predictions from the model).

# Let's load the data into a Pandas DataFrame using the read_csv
my_df = pd.read_csv('EmpData.txt', delimiter='|', index_col=0)

print(my_df.head(5))
print(my_df.info())
print(my_df.describe())
print(my_df.shape)

# It looks like we have 5497 observations (examples) and 6 features (dimensions) in this dataset.
# Also notice how we are missing data in the currentsalary data column
print(my_df.isnull().sum())

pd.set_option('display.max_columns',10)
rp.summary_cont(my_df['currentsalary'].groupby(my_df['flightrisk']))

# For this analysis, let's fill the missing values with the average values based on each flightrisk group
my_df1 = my_df.query('flightrisk=="high"')
my_avg1 = my_df1['currentsalary'].mean()
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
print(my_data.isnull().sum())
rp.summary_cont(my_data['currentsalary'].groupby(my_data['flightrisk']))

# Now, let's perform simple linear regression using scipy and statsmodels
# Conceptualize this as a traditional regression analysis.
# Is there a correlation between x and y?
# We will do this for a single independent variable and multiple independent variables!
from scipy import stats as sts

slope, intercept, r_value, p_value, slope_std_error = sts.linregress(my_data['yearseducation'], my_data['currentsalary'])

# Display the results of our one predictor linear regression model
print(f'Best fit line: y = {slope:4.2f}*x + {intercept:5.3f}')
print(f'Pearsonr correlation = {r_value:5.3f}\n')
print(f'R squared = {r_value**2:5.3f}\n')

from statsmodels.formula.api import ols
# This syntax may look a little cryptic but it is actually derived from R, which is also a little cryptic!
results = ols('currentsalary ~ yearseducation', data=my_data).fit()
print(results.summary()) # This will give us an output similar to SAS, SPSS or R

# Now, let's construct a regression model using multiple independent variables
results = ols('currentsalary ~ yearseducation + joblevel + travelamt + C(gender)', data=my_data).fit()
print(results.summary())
print(results.rsquared)
print(results.rsquared_adj)
print(results.params)

#This model finds our regression parameters that minimizes the value of the error term.
print(results.resid)
print(results.mse_resid)

# So we have explained the historical correlation between our predictor variables and our outcome variable.
# However, we have not explicitly constructed a predictive model.  Instead, we have constructed an explanatory model.

# Here, we will minimize our cost function in our randomly selected training data set.
# Before we do so, let's investigate our cost function in a little more detail.

# A fundamental concept in machine learning is the minimization of a cost (or loss) function,
# which quantifies how well a model represents a data set. For a given data set, the cost function
# is completely specified by the model parameters, thus a more complex model has a more complex
# cost function, which can become difficult to minimize.

def display_costfunc():
    """Darker section of the plot is where the cost function is minimized.
       This function should probably accept a few of these variables as parameters to
       make it more user friendly."""

    # The beta variables will represent the range of values on the x axis, which are the possible
    # beta coefficients for the independent variable (years of education for this example)
    beta_low = 0
    beta_high = 6000
    # The alpha variables will represent the possible values for the intercept.
    alpha_low = 13000
    alpha_high = 15000
    # NOTE: I set the above values via a process of trial and error and by observing the coefficients from
    #       the previously executed regression models.

    # Define our sampling grid for slopes and intercepts
    # My betas and alphas variables will be two dimensionsal ndarrays containing the points on the x (betas)
    # and y (alphas) axis.
    # 100 was an arbitrary value that I inserted just to have enough variation in the visual
    betas, alphas = np.meshgrid(np.linspace(beta_low, beta_high, 100),
                                np.linspace(alpha_low, alpha_high, 100))

    # My cost function will work with a numpy ndarray, so I am converting my dependent variable and my
    # independent variable from the Series within the DataFrame to a numpy ndarray.
    # I should be fine with the default shape that results from the to_numpy() function.
    # If not, I would have had to reshape it (to go down instead of going across).
    x = my_data['yearseducation'].to_numpy()
    y = my_data['currentsalary'].to_numpy()

    # Our cost function: Here, I compute the natural logarithm of the standard
    # l2-norm of the model residuals (MSE) to enhance the scale.
    # l2-norm is the mean square error (MSE).
    # In this equation, y is the true value
    # m is the beta coefficient for our independent variable
    # x is an observed value for the independent variable
    # b is the intercept.
    def cost(m, b):
        return np.log(np.sum((y - m * x - b)**2))

    # Now vectorize our function, which turns our cost function into a vectorized function
    # that can be broadcast to a NumPy array. This simplifies our code as we remove the need for an explicit loop.
    # Said differently, the purpose of np.vectorize is to transform functions that are not
    # numpy-aware (e.g. take floats as input and return floats as output) into functions
    # that can operate on (and return) numpy arrays.
    v_cost = np.vectorize(cost)

    # Our vectorized cost function sampled at every grid point
    epsilons = v_cost(betas, alphas)

    # Now, let's attempt to plot/graph/visualize the result
    fig, ax = plt.subplots()

    # First we draw the OLS result
    ax.plot(slope, intercept, marker='o', color='r', markersize=5)

    # Now plot the sampled grid as an image
    # https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.axes.Axes.imshow.html
    # If we want to blur the image to reduce image noise and reduce detail add the following:
    #interpolation="gaussian"
    ax.imshow(epsilons, origin='lower', extent=[beta_low, beta_high, alpha_low, alpha_high],
              aspect='auto', interpolation='gaussian')

    # clean up the final plot
    ax.set_xlabel(r"$\beta$", fontsize=14)
    ax.set_ylabel(r"$\alpha$", fontsize=14)
    ax.set_title(r"Cost function for $y = \beta*x + \alpha$", fontsize=18)
    sns.despine(offset=0, trim=True)

display_costfunc()

# As we move to higher dimensional data sets or more complex cost functions, the challenge
# of finding the global minimum becomes increasingly difficult.
# As a result, many mathematical techniques have been developed to find the global
# minimum of a (potentially) complex function. The standard approach is gradient descent,
# where we use the fact that the first derivative (or gradient) measures the slope of a
# function at a given point. We can then use the slope to infer which direction is
# downhill and thus travel (hopefully) towards the minimum.

# For a mental picture of this process, imagine hiking in the mountains and flip the challenge to finding the highest peak,
# so we will use gradient ascent. Gradient ascent is similar to finding the local mountain peak and climbing it.
# This local peak might look like it is the largest, but a random jump away from the local peak might enable one to
# view much larger peaks beyond, which can subsequently be climbed with a new gradient ascent.

# Whenever you perform machine learning, you should keep in mind that the model
# that you generate for a given data set has generally resulted from the minimization of a cost function.
# Thus, there remains the possibility that with more effort, more data, or a better cost minimization
# strategy, a new, and better model may potentially exist.

# We have to split our data into training and testing
from sklearn.model_selection import train_test_split

# For this example, let's pick a single independent variable and the single dependent vairable
# We must reshape the arrays to ensure they are the same size (two dimensional ndarray - observations 5497 by features 1)
# For this example, let's predict currentsalary based on yearseducation
ind_data = my_data['yearseducation'].to_numpy().reshape(my_data.shape[0],1)
dep_data = my_data['currentsalary'].to_numpy().reshape(my_data.shape[0],1)

# Create test/train splits for independent and dependent data
# Explicitly set our random seed to enable reproducibility.
# Test size is the amount of data that we hold out for testing.
# Leaving too much data for testing might result in inadequate training.
# However, the smaller the test set, the more inaccurate the estimation
# of the generalization error.  There are no great rules of thumb.  Larger data sets
# might need to be trained more, so you might see a 90 (training) to 10 (testing) split but smaller data
# sets are typically in the 60 (training) to 40 (testing) range.  However, this is a pretty broad generalization
ind_train, ind_test, dep_train, dep_test = train_test_split(ind_data, dep_data, test_size=0.4, random_state=47)

# Let's use the scikit-learn LinearRegression algorithm for this example
# This algorithm most closely aligns with an OLS regression
from sklearn.linear_model import LinearRegression

# When this estimator is created, the following parameters can be specified (they are all optional):

    #fit_intercept: If True, the default an intercept is fit for this model.
    #normalize: If True all the features supplied in the fit method will be normalized, the default is False.
    #copy_X: If True, the default, the feature matrix will be copied, otherwise the data
    #        may be overwritten during the regression.

# This regressor has two commonly used attributes, which can be accessed after the model has
# been fit to the data (note that model attributes in scikit learn are suffixed by an underscore):

    # coef_: An array of the estimated coefficients for the regressed linear model, in typical
    #        usage this is a single dimensional array.
    # intercept_: The constant term in the regressed linear model, only computed if fit_intercept is True.

# Once created, this estimator is fit to the training data, after which it can be used to predict
# new values from the test data. These two actions, along with a measure of the performance of the
# regressor, are encapsulated in the following three functions that can be called on a LinearRegression estimator:

    #fit: Fits a linear model to the supplied features.
    #predict: Predicts new results from the given model for the new supplied features
    #score: Computes a regression score, specifically the coefficient of determination R2 of the prediction.

# Create and fit our linear regression model to training data
# Run this with and without the intercept to observe the differences.
# Why would we drop the intercept from our model?
# Dropping the intercept in a regression model forces the regression
# line to go through the origin–the y intercept must be 0.
#
# The problem with dropping the intercept is if the slope is steeper
# just because you are forcing the line through the origin, not because
# it fits the data better.  If the intercept really should be something else,
# you are creating that steepness artificially.
# Does it fit your business scenario?
model = LinearRegression(fit_intercept=False)
model.fit(ind_train, dep_train)

# Display model fit parameters for training data
print(f"y = {model.coef_[0][0]:4.2f}*x")
# Display model fit parameters for training data when we include the intercept
#print(f"y = {model.intercept_[0]:4.2f} + {model.coef_[0][0]:4.2f}*x")

# Compute model predictions for test data
results = model.predict(ind_test)

# Compute score and display result (Coefficient of Determination)
score = 100.0 * model.score(ind_test, dep_test)
print(f'LR Model score = {score:5.1f}%')

# Using a machine learning algorithm for a single variable model is not typically done very frequently.
# Let's now perform an analysis with multiple independent variables.
# Often, adding more features will result in more accurate models, since finer details can be captured
# in our training data set.

# Let's select several independent variables and a single dependent variable.
print(my_df.head(5))
# Notice how the flightrisk is categorical and the gender is also categorical.
# The gender only has two options in these data (Male or Female)
# We refer to this as a nominal feature because there is no relationship or ordering
# between the different values.
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(my_data['gender'].to_numpy().reshape(my_data.shape[0],1))
my_data['gender2']=lb.transform(my_data['gender'].to_numpy().reshape(my_data.shape[0],1))

from sklearn.preprocessing import LabelEncoder
# Define allowed flightriskgroups
rgrps = ['low', 'medium', 'high']
le = LabelEncoder()
le.fit(rgrps)

# Transform sample data, and reshape vector
tst = my_data['flightrisk'].to_numpy().reshape(my_data.shape[0],1)
le_data = le.transform(tst).reshape(my_data.shape[0], 1)
print(le_data)
# Display encode label and color
for frg, idx in zip(tst, le_data):
    print(idx, frg)

#Notice that the order is alphabetical, which is not what we would want in this case
my_data['ord_frg'] = my_data['flightrisk'].apply(lambda x: ['low', 'medium', 'high'].index(x))
print(my_data.head(5))
# This custom lambda function will code everything as 0s, 1s, or 2s going from low to medium to high.
# If we plugged this into a regression, this would imply an order

# If we don't want this implied order, we can use a one hot encoder to generate a matrix of 0s and 1s to
# refer to each of the three flightrisk groups as nominal groups without any implied order.
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
nd_array_frg = my_data['flightrisk'].to_numpy().reshape(my_data.shape[0],1)
le_data = le.transform(nd_array_frg).reshape(my_data.shape[0], 1)
ohefrg = ohe.fit_transform(le_data)
print(ohefrg)
enc=[0,1,0]
print(le.inverse_transform(np.argmax(enc).reshape(1,1)))

# We only need to reshape the dependent variable array, since the feature matrix should be properly sized
# by passing in a tuple or a list of independent variables.
ind_data = np.column_stack((my_data.joblevel, my_data.travelamt, my_data.gender2, ohefrg))
dep_data = my_data['currentsalary'].to_numpy().reshape(my_data.shape[0],1)

# Create test/train splits for data and labels
# Explicitly set our random seed to enable reproduceability
ind_train, ind_test, dep_train, dep_test = train_test_split(ind_data, dep_data, test_size=0.4, random_state=23)

# Fit our linear regression model to training data
model = LinearRegression(fit_intercept=True)
model.fit(ind_train, dep_train)

# Display model fit parameters for training data
print(f"y = {model.intercept_[0]:5.2f} + {model.coef_[0][0]:5.2f}*joblevel ",
           f"+ {model.coef_[0][1]:5.2f}*travelamt + {model.coef_[0][2]:5.2f}*gender ",
           f"+ {model.coef_[0][3]:5.2f}*frg=='High' + {model.coef_[0][4]:5.2f}*frg=='Low' + {model.coef_[0][5]:5.2f}*frg=='Medium'")

# Compute model predictions for test data
results = model.predict(ind_test)

# Compute score and display result (Coefficient of Determination)
score = 100.0 * model.score(ind_test, dep_test)
print(f'Multivariate LR Model score = {score:5.2f}%')