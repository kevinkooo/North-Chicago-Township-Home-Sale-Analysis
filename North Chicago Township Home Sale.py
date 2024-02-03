#!/usr/bin/env python
# coding: utf-8

# ### Assignment 4

# import document

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
import seaborn as sns

df = pd.read_csv('/Users/kevinko/Desktop/Uchicago/Statistical Analysis/Assignment 4/NorthChicagoTownshipHomeSale.csv')
df.head()


# Q1) We will explore the distribution of Sale Price as it is.  Please draw a histogram and a horizontal boxplot for the Sale Price.  We will put the histogram on top of the boxplot in the same chart frame.  The bin width is 100 for the histogram.  You must label the axes with appropriate tick marks to receive full credits.  Does the distribution of Sale Price resemble that of a normal distribution?

# In[43]:


# Create a figure with subplots
plt.figure(figsize=(10, 8))

# Histogram
plt.subplot(2, 1, 1)
sns.histplot(df['Sale Price'], bins=range(0, int(max(df['Sale Price'])) + 100, 100), kde=False, color='lightblue')
plt.ylabel('Frequency')
plt.xlabel('')  # Clear x-label for the histogram

# Boxplot
plt.subplot(2, 1, 2)
sns.boxplot(x=df['Sale Price'])
plt.ylabel('')  # Clear y-label for the boxplot
plt.xlabel('Sale Price (in thousands of dollars)')

plt.tight_layout()

plt.show()


# Q2. In my opinion, the distribution of Sale Price is heavily skewed to the right.  It looks more like a Gamma distribution.  Since a linear regression model assumes a normal distribution for the error, we will transform the Sale Price to produce a seemingly normal distribution.  Many transformations serve this purpose.  For us, we will apply the natural logarithm transformation to Sale Price.  Let us name the transformed variable as Log Sale Price.
# 
# (a)	(15 points) Please check whether the Log Sale Price is normally distributed.  Therefore, you will use the Normal Q-Q plot, the Shapiro-Wilks test, and the Anderson-Darling test to help you draw your conclusions.
# 
# (b)	(5 points) Log Sale Price will unlikely follow a perfectly normal distribution.  Despite the non-normality of the Log Sale Price, are you comfortable using it for your linear regression model? Note: it is okay to disagree with me.
# 

# In[8]:


import statsmodels.api as sm
import scipy.stats as stats

# Apply natural logarithm transformation to Sale Price
df['Log_Sale_Price'] = np.log(df['Sale Price'])

# Create a Q-Q plot
sm.qqplot(df['Log_Sale_Price'], line='s')
plt.title('Normal Q-Q Plot for Log Sale Price')
plt.show()

# Shaprio - Wilk Test
stat, p_value_sw = stats.shapiro(df['Log_Sale_Price'])
print(f'Shapiro-Wilks test p-value: {p_value_sw}')

# Anderson - Darling test
result_ad = stats.anderson(df['Log_Sale_Price'], dist='norm')
print(f'Anderson-Darling test statistic: {result_ad.statistic}')
print(f'Anderson-Darling test critical values: {result_ad.critical_values}')


# Q3. We will train a multiple linear regression model using all eight predictors.  The target variable is the Log Sale Price.  The model must include the Intercept term.
# 
# (a)	(10 points) What is the Coefficient of Determination of your final model?
# 
# (b)	(10 points) What are the regression coefficients, their standard errors, and their 95% confidence interval?  Please show your answers in a table with proper labels.
# 

# In[9]:


df = df.rename(columns={'Building Square Feet': 'Building_Square_Feet',
                        'Full Baths': 'Full_Baths',
                        'Garage Size': 'Garage_Size',
                        'Half Baths': 'Half_Baths',
                        'Land Acre': 'Land_Acre',
                        'Tract Median Income': 'Tract_Median_Income',
                        })


# In[12]:


import statsmodels.formula.api as smf

model = smf.ols(formula='Log_Sale_Price ~ Age + Bedrooms + Building_Square_Feet + Full_Baths + Garage_Size + Half_Baths + Land_Acre + Tract_Median_Income', data=df).fit()

r_squared = model.rsquared
print(f"Coefficient of Determination (R-squared): {r_squared}")

model_summary = model .summary()
print(model_summary)


# In[13]:


X = df[['Age', 'Bedrooms', 'Building_Square_Feet', 'Full_Baths', 'Garage_Size', 'Half_Baths', 'Land_Acre', 'Tract_Median_Income']]
X


# Q4.We will predict the Sale Price of a single-family home whose features are at the median of all predictors (e.g., Age is 24).  We will also provide the 95% confidence interval for our prediction. Hint: Suppose v is a p×1 vector of the median values, the predicted value is v^t b ̂ and the standard error is σ ̂√(v^t (X^t X)^(-1) v).  After you have calculated the confidence interval for the predicted Log Sale Price, and exponentiate the boundaries to produce the confidence interval for Sale Price.
# 
# (10 points) What are the medians of the eight predictors?

# In[14]:


medians = X.median()
medians


# (10 points) What is the predicted Sale Price and its 95% confidence interval?
# 

# In[15]:


predicted_sale_price = model.predict(medians)
print('The predicted sale price is :', np.exp(predicted_sale_price)[0])
print(predicted_sale_price)


# In[16]:


# design matrix X
X.insert(0, 'Intercept', 1)
design_matrix_X = X
design_matrix_X


# In[17]:


from sklearn.metrics import mean_squared_error


# Add intercept to medians
medians_with_intercept = medians.values
medians_with_intercept = np.insert(medians_with_intercept, 0, 1).T

# Standard error
#se = np.sqrt(np.diag(model.cov_params()))


inv_X = np.linalg.inv((design_matrix_X.T) @ (design_matrix_X))

MSE = mean_squared_error(df['Log_Sale_Price'], model.predict(X))

se = (((medians_with_intercept.T) @ inv_X @ medians_with_intercept) * MSE)**0.5


# Confidence interval for Log Sale Price
lower_bound = predicted_sale_price - 1.96 * se
upper_bound = predicted_sale_price + 1.96 * se

# Exponentiate to get Sale Price and its confidence interval
ci_lower = np.exp(lower_bound)
ci_upper = np.exp(upper_bound)

# Display medians and predicted Sale Price with confidence interval
print('\nMedians of Predictors:')
print(medians)
print('\nPredicted Sale Price and 95% Confidence Interval:')
result_df = pd.DataFrame({'Predicted Sale Price': predicted_sale_price, 'CI Lower': ci_lower, 'CI Upper': ci_upper})
print(result_df)


# Q5) To conclude this mini project, we calculate the Shapley values for the predictors included in our regression model.  According to the Shapley values, which predictor(s) in your final model are most influential in determining the sale price of a single-family home in the North Chicago township?

# In[19]:


def pearson_correlation_test (x, y):
    ''' Calculate the Pearson correlation coefficient and perform test of zero correlation

    Argument
    --------
    x: an input vector
    y: another input vector of the same length as x

    Return
    ------
    n_valid: number of valid values
    n_missing: number of missing values
    pearson_corr: the Pearson correlation coefficient
    se_corr: the standard error of the Pearson correlation coefficient
    t_stat: the Student's t statistic for testing if the population correlation is zero
    t_df: the degree of freedom of the Student's t test
    t_sig: the significance value
    '''

    n_missing = 0
    n_valid = 0
    mean_x = 0.0
    mean_y = 0.0
    for u, v in zip(x, y):
        if (np.isnan(u) or np.isnan(v)):
            n_missing = n_missing + 1
        else:
            n_valid = n_valid + 1
            mean_x = mean_x + u
            mean_y = mean_y + v

    if (n_valid > 0):
        mean_x = mean_x / n_valid
        mean_y = mean_y / n_valid

        ssx = 0.0
        ssy = 0.0
        ssxy = 0.0
        for u, v in zip(x, y):
            if (not np.isnan(u) and not np.isnan(v)):
                devx = (u - mean_x)
                devy = (v - mean_y)
                ssx = ssx + devx * devx
                ssy = ssy + devy * devy
                ssxy = ssxy + devx * devy

        if (ssx > 0.0 and ssy > 0.0):
            pearson_corr = (ssxy / ssx) * (ssxy / ssy)
            pearson_corr = np.sign(ssxy) * np.sqrt(pearson_corr)

            if (n_valid > 2):
                t_df = (n_valid - 2)
                se_corr = (1.0 - pearson_corr * pearson_corr) / t_df
                se_corr = np.sqrt(se_corr)
                t_stat = pearson_corr / se_corr
                t_sig = 2.0 * t.sf(np.abs(t_stat), t_df)

    return ([n_valid, n_missing, pearson_corr, se_corr, t_stat, t_df, t_sig])

def SWEEPOperator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    ''' Implement the SWEEP operator

    Argument
    --------
    pDim: dimension of matrix inputM, integer greater than one
    inputM: a square and symmetric matrix, numpy array
    origDiag: the original diagonal elements before any SWEEPing
    sweepCol: a list of columns numbers to SWEEP
    tol: singularity tolerance, positive real

    Return
    ------
    A: negative of a generalized inverse of input matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = np.copy(inputM)
    ANext = np.zeros((pDim,pDim))

    for k in sweepCol:
        Akk = A[k,k]
        pivot = tol * abs(origDiag[k])
        if (not np.isinf(Akk) and abs(Akk) >= pivot):
            nonAliasParam.append(k)
            ANext = A - np.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / abs(Akk)
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = np.zeros(pDim)
            ANext[k, :] = np.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)

def LinearRegressionModel (X, y, tolSweep = 1e-7):
    ''' Train a linear regression model

    Argument
    --------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    A list of model output:
    (0) parameter_table: a Pandas DataFrame of regression coefficients and statistics
    (1) cov_matrix: a Pandas DataFrame of covariance matrix for regression coefficient
    (2) residual_variance: residual variance
    (3) residual_df: residual degree of freedom
    (3) aliasParam: a list of aliased rows/columns in input matrix
    (4) nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    # X: A Pandas DataFrame, rows are observations, columns are regressors
    # y: A Pandas Series, rows are observations of the response variable

    Z = X.join(y)
    n_sample = X.shape[0]
    n_param = X.shape[1]

    ZtZ = Z.transpose().dot(Z)
    diag_ZtZ = np.diagonal(ZtZ)
    eps_double = np.finfo(np.float64).eps
    tol = np.sqrt(eps_double)

    ZtZ_transf, aliasParam, nonAliasParam = SWEEPOperator ((n_param+1), ZtZ, diag_ZtZ, sweepCol = range(n_param), tol = tol)

    residual_df = n_sample - len(nonAliasParam)
    residual_variance = ZtZ_transf[n_param, n_param] / residual_df

    b = ZtZ_transf[0:n_param, n_param]
    b[aliasParam] = 0.0

    parameter_name = X.columns

    XtX_Ginv = - residual_variance * ZtZ_transf[0:n_param, 0:n_param]
    XtX_Ginv[:, aliasParam] = 0.0
    XtX_Ginv[aliasParam, :] = 0.0
    cov_matrix = pd.DataFrame(XtX_Ginv, index = parameter_name, columns = parameter_name)

    parameter_table = pd.DataFrame(index = parameter_name,
                                       columns = ['Estimate','Standard Error', 't', 'Significance', 'Lower 95 CI', 'Upper 95 CI'])
    parameter_table['Estimate'] = b
    parameter_table['Standard Error'] = np.sqrt(np.diag(cov_matrix))
    parameter_table['t'] = np.divide(parameter_table['Estimate'], parameter_table['Standard Error'])
    parameter_table['Significance'] = 2.0 * t.sf(abs(parameter_table['t']), residual_df)

    t_critical = t.ppf(0.975, residual_df)
    parameter_table['Lower 95 CI'] =  parameter_table['Estimate'] - t_critical * parameter_table['Standard Error']
    parameter_table['Upper 95 CI'] =  parameter_table['Estimate'] + t_critical * parameter_table['Standard Error']

    return ([parameter_table, cov_matrix, residual_variance, residual_df, aliasParam, nonAliasParam])


# In[20]:


import sys

from scipy.special import comb
from itertools import (chain, combinations)


# In[21]:


# Set some options for printing all the columns
np.set_printoptions(precision = 10, threshold = sys.maxsize)
np.set_printoptions(linewidth = np.inf)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

pd.options.display.float_format = '{:,.10f}'.format


# In[22]:


continuous_feature = ['Age', 'Bedrooms', 'Building_Square_Feet', 'Full_Baths', 'Garage_Size', 'Half_Baths', 'Land_Acre', 'Tract_Median_Income']
target_name = 'Log_Sale_Price'

train_data = df[[target_name] + continuous_feature]


# In[23]:


# Generate a dataframe that contains the all-possible model specifications

candidate = continuous_feature
n_candidate = len(candidate)
feature_position = []
for i in range(n_candidate):
   feature_position.append('POS_' + str(i))

all_model_spec = pd.DataFrame(chain(*map(lambda x: combinations(candidate, x), range(0, n_candidate+1))),
                                  columns = feature_position)

n_feature_list = []
model_df_list = []
sse_list = []


# In[24]:


# Generate the full model matrix and remember each candidate's columns in the full model matrix

X_all = train_data[[]].copy()
X_all.insert(0, 'Intercept', 1.0)

start_column = 0
last_column = 0

component_column = {}
for pred in candidate:
   X_term = train_data[[pred]]
   X_all = X_all.join(X_term)
   start_column = last_column + 1
   last_column = start_column + X_term.shape[1] - 1
   component_column[pred] = [j for j in range(start_column, last_column+1)]


# In[25]:


# Train all the model specifications
y = train_data[target_name]

for idx, row in all_model_spec.iterrows():
   model_column = [0]
   n_feature = 0
   for pos in feature_position:
      pred = row[pos]
      if (pred is not None):
         n_feature = n_feature + 1
         model_column = model_column + component_column[pred]
      else:
         break
   X = X_all.iloc[:, model_column]
   result_list = LinearRegressionModel (X, y)
   model_df = len(result_list[5])
   SSE =  result_list[2] * result_list[3]

   if (n_feature == 0):
      SST = SSE

   n_feature_list.append(n_feature)
   model_df_list.append(model_df)
   sse_list.append(SSE)

all_model_spec['N_Feature'] = n_feature_list
all_model_spec['Model DF'] = model_df_list
all_model_spec['RSquare'] = 1.0 - sse_list / SST


# In[26]:


# Make the model specifications as a Python set

model_k_spec = {}
for k in range(0, n_candidate+1):
   subset = all_model_spec[all_model_spec['N_Feature'] == k]
   out_list = []
   for idx, row in subset.iterrows():
      cur_rsq = row['RSquare']
      cur_set = set(list(row[feature_position].dropna()))
      out_list.append([cur_set, cur_rsq])
   model_k_spec[k] = pd.DataFrame(out_list, columns = ['FeatureSet','RSquare'])


# In[ ]:


# Find the nested models and calculate the R-Square changes

result_list = []
for k in range(0,n_candidate,1):
   spec_0 = model_k_spec[k]
   spec_1 = model_k_spec[k+1]
   for idx_0, row_0 in spec_0.iterrows():
      set_0 = row_0['FeatureSet']
      rsq_0 = row_0['RSquare']
      for idx_1, row_1 in spec_1.iterrows():
         set_1 = row_1['FeatureSet']
         rsq_1 = row_1['RSquare']
         set_diff = set_1.difference(set_0)
         print(set_0, set_1, set_diff, len(set_diff))
         if (len(set_diff) == 1):
            rsq_diff = rsq_1 - rsq_0
            wgt = (n_candidate - 1) / comb((n_candidate-1), k)
            result_list.append([k, list(set_diff)[0], rsq_diff, wgt])

result_df = pd.DataFrame(result_list, columns = ['k', 'Feature', 'RSqChange', 'Wgt'])


# In[28]:


# Calculate the Shapley values

def weighted_average(df, values, weights):
    return sum(df[weights] * df[values]) / df[weights].sum()

shapley = result_df.groupby('Feature').apply(weighted_average, 'RSqChange', 'Wgt')
total_shapley = np.sum(shapley)
percent_shapley = 100.0 * (shapley / total_shapley)
print(' Sum of Shapley Values = ', total_shapley)


# In[29]:


# Check if the sum of Shapley values is equal to the Full Model's R-Square

subset = all_model_spec[all_model_spec['N_Feature'] == n_candidate]
print('R-Square of Full Model = ', subset['RSquare'].values[0])


# In[30]:


shapley


# In[ ]:




