#!/usr/bin/env python
# coding: utf-8

# # <center> <u> Feature Selection </u> </center>

# <h2>Purpose of Feature Selection</h2>
# <p>Many learning algorithms perform poorly on high-dimensional data. This is known as the <b>curse of dimensionality</b>
#     <p>There are other reasons we may wish to reduce the number of features including:
#         <p>1. Reducing computational cost
#             <p>2. Reducing the cost associated with data collection
#                 <p>3. Improving Interpretability

# **Problem Statement:**
# 
# Build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes

# In[1]:


#import numpy, pandas 

import numpy as np
import pandas as pd

# import SelectFromModel, train_test_split, DecisionTreeClassifier, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# import RandomForestClassifier, roc_auc_score, StandardScaler, make_pipeline, KFold, KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

# import mean_squared_error, r2_score, cross_val_predict, LinearRegression, sqrt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

# Note: just have a look what all libraries you have imported


# ## 1.Filter Methods:
# 
# 
# 
# Filter method applies a statistical measure to assign a scoring to each feature.Then we can decide to keep or remove those features based on those scores. The methods are often univariate and consider the feature independently, or with regard to the dependent variable.
# 
# In this section we will cover below approaches:
# 
# 1. Missing Value Ratio Threshold
# 2. Variance Threshold
# 3. $Chi^2$ Test
# 4. Anova Test

# Dataset = https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/diabetes.csv
# 
# Dataset used - diabetes.csv
# 
# Download instruction: go to the given link--->click raw button on top right corner---->Press Ctrl+S -->save it as .csv file.

# ## (a) Missing Value Ratio Threshold
# 
# 
# 
# Data Dict:
# ---
# 
# **Pregnancies:** Number of times pregnant <br>
# **Glucose:** Plasma glucose concentration a 2 hours in an oral glucose tolerance test.<br>
# **BloodPressure:** Diastolic blood pressure (mm Hg).<br>
# **SkinThickness:** Triceps skin fold thickness (mm).<br>
# **Insulin:** 2-Hour serum insulin (mu U/ml).<br>
# **BMI:** Body mass index (weight in kg/(height in m)^2). <br>
# **DiabetesPedigreeFunction:** A function which scores likelihood of diabetes based on family history<br>
# **Age:** Age (years)<br>
# **Outcome:** Class variable (0 or 1)
# 
# 
# 

# In[2]:


# create a data frame named diabetes and load the csv file

diabetes = pd.read_csv("diabetes.csv")
#print the head 
diabetes.head(5)


# We know that some features can not be zero(e.g. a person's blood pressure can not be 0) hence we will impute zeros with nan value in these features.
# 
# Reference to impute: https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.replace.html

# In[3]:


#Glucose BloodPressure, SkinThickness, Insulin, and BMI features cannot be zero ,we will impute zeros with nan value in these features.

diabetes['Glucose'].replace(0, np.nan, inplace = True)
diabetes['SkinThickness'].replace(0, np.nan, inplace = True)
diabetes['Insulin'].replace(0, np.nan, inplace = True)
diabetes['BMI'].replace(0, np.nan, inplace = True)
diabetes['BloodPressure'].replace(0, np.nan, inplace = True)


# In[4]:


#display the no of null values in each feature

diabetes.isnull().sum()


# Now let's see for each feature what is the percentage of having missing values.

# In[5]:


#percentage of missing values for Glucose(sum null values , divide by length and multiply by 100)

diabetes["Glucose"].isnull().sum()/len(diabetes)*100


# In[6]:


# calculate the percentage for Bloodpressure

diabetes["SkinThickness"].isnull().sum()/len(diabetes)*100


# In[7]:


# calculate the percentage for SkinThickness

diabetes["Insulin"].isnull().sum()/len(diabetes)*100


# In[8]:


# calculate the percentage for Insulin

diabetes["BloodPressure"].isnull().sum()/len(diabetes)*100


# In[9]:


# calculate the percentage for BMI

diabetes["BMI"].isnull().sum()/len(diabetes)*100


# In[10]:


#we are keep only those features which are having missing data less than 10% 

diabetes_missing_value_threshold = diabetes.dropna(thresh = int(diabetes.shape[0] * .9), axis = 1)


# print diabetes_missing_value_threshold 

diabetes_missing_value_threshold


# In[11]:



diabetes_missing_value_threshold_features = diabetes_missing_value_threshold.drop('Outcome',axis=1)

diabetes_missing_value_threshold_label= diabetes_missing_value_threshold['Outcome']


# In[12]:


#print diabetes_missing_value_threshold_features

diabetes_missing_value_threshold_features


# In[13]:


#print diabetes_missing_value_threshold_label

diabetes_missing_value_threshold_label


# ## (b) Variance Threshold
# 
# 
# If the variance is low or close to zero, then a feature is approximately constant and will not improve the performance of the model. In that case, it should be removed.
# 
# Variance will also be very low for a feature if only a handful of observations of that feature differ from a constant value.
# 

# datsset - https://github.com/arupbhunia/Data-Pre-processing/blob/master/datasets/diabetes_cleaned.csv
# 
# Dataset used - diabetes_cleaned.csv

# In[14]:


# load the csv to dataframe name "diabetes" and print the head values
diabetes = pd.read_csv("diabetes_cleaned.csv")

# print head
diabetes.head(5)


# In[15]:


# seperate the features and the target as x and y 
X = diabetes.drop('Outcome', axis = 1)
Y = diabetes['Outcome']


# In[16]:


# Return  the variance for X along the specified axis=0.

X.var(axis = 0)


# In[17]:


# import minmax_scale

from sklearn.preprocessing import minmax_scale

# use minmax scale with feature_range=(0,10) and columns=X.columns,to scale the features of dataframe and store them into X_scaled_df 

X_scaled_df = pd.DataFrame(minmax_scale(X, feature_range = (0,10)),columns = X.columns)


# In[18]:


# return X_scaled_df

X_scaled_df


# In[19]:


# Again return  the variance for X along the specified axis=0 to check the scales after using minmax scaler.

X_scaled_df.var()


# In[20]:


# import variancethreshold

from sklearn.feature_selection import VarianceThreshold

# set threshold=1 and define it to variable select_features
select_features = VarianceThreshold(threshold = 1.0)


#    Impliment fit_transform on select_features passing X_scaled_df into it and save this result in variable X_variance_threshold_df
# 

# In[21]:


X_variance_threshold_df = select_features.fit_transform(X_scaled_df)


# In[22]:


#print X_variance_threshold_df

X_variance_threshold_df


# In[23]:


#Convert X_variance_threshold_df into dataframe
X_variance_threshold_df = pd.DataFrame(X_variance_threshold_df)


# In[24]:


# print of head values of X_variance_threshold_df 

X_variance_threshold_df


# Below mentioned is the function get_selected_features for returning selected_features to be used further 

# In[59]:


def get_selected_features(raw_df,processed_df):
    selected_features=[]
    for i in range(len(processed_df.columns)):
        for j in range(len(raw_df.columns)):
            if (processed_df.iloc[:,i].equals(raw_df.iloc[:,j])):
                selected_features.append(raw_df.columns[j])
    return selected_features


# In[26]:


# pass the X_scaled_df as raw_df and X_variance_threshold_df as processed_df inside get_selected_features function

selected_features = get_selected_features( X_scaled_df, X_variance_threshold_df)

# print selected_features

selected_features


#  SkinThickness feature is not selected as its variance is less.
# 
# Lets give column names to our X_variance_threshold_df

# In[27]:


# define selected_features as columns and save it in variabe named X_variance_threshold_df

X_variance_threshold_df.columns = selected_features

#print X_variance_threshold_df


# ## (c) Chi-Squared statistical test (SelectKBest)
# 
# 
# 
# Chi2 is a measure of dependency between two variables. It gives us a goodness of fit measure because it measures how well an observed distribution of a particular feature fits with the distribution that is expected if two features are independent.
# 
# Scikit-Learn offers a feature selection estimator named SelectKBest which select K numbers of features based on the statistical analysis.
# 
# 

# Reference link: https://chrisalbon.com/machine_learning/feature_selection/chi-squared_for_feature_selection/
# 
# The below mentioned function generate_feature_scores_df is used to get feature score for using it in  Chi-Squared statistical test explained below

# In[47]:


def generate_feature_scores_df(X,Score):
    feature_score=pd.DataFrame()
    for i in range(X.shape[1]):
        new =pd.DataFrame({"Features":X.columns[i],"Score":Score[i]},index=[i])
        feature_score=pd.concat([feature_score,new])
    return feature_score


# In[48]:


# create a data frame named diabetes and load the csv file again
diabetes = pd.read_csv('diabetes.csv')


# In[49]:


# assign features to X variable and 'outcome' to y variable from the dataframe diabetes
X = diabetes.drop('Outcome', axis = 1)
Y = diabetes['Outcome']


# Reference Doc: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

# In[50]:


#import chi2 and SelectKBest

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[51]:


# converting data cast to a float type.
X = X.astype(np.float64)


# In[66]:


X


# Lets use SelectKBest to calculate the best feature score. Use Chi2 as Score Function and no.of feature i.e. k as 4
# 

# In[52]:


# Initialise SelectKBest with above parameters 
chi2_test = SelectKBest(score_func = chi2, k=4)

# and fit it with X and Y
chi2_model = chi2_test.fit(X, Y)


# In[53]:


#print the scores of chi2_model

chi2_model.scores_


# In[54]:


# use generate_feature_scores_df function to get features and their respective scores passing X and chi2_model.scores_ as paramter
feature_score_df = generate_feature_scores_df(X, chi2_model.scores_)

# return feature_score_df

feature_score_df


# Did you see the features and corresponding chi square scores? This is so easy right, higher the score better the feature. Just like higher the marks in assignment better the student of ours. 

# In[81]:


##Lets get X to the selected features of chi2_model using tranform function so we will have X_new

X_new = chi2_model.transform(X)


# In[82]:


X_new


# In[83]:


# create a dataframe using X_new
X_new = pd.DataFrame(X_new)


# In[84]:


X_new


# In[85]:


#repeat the previous steps of calling get_selected_features function( pass X and X_new as score in the function)
selected_features = get_selected_features(X, X_new)

# return selected_features


# In[86]:


selected_features


# Let have X with all features given in list selected_features and save this dataframe in variable chi2_best_features

# In[95]:


# X with all features given in list selected_features 
chi2_best_features = X[selected_features]

# print chi2_best_features.head()

chi2_best_features.head()


# chi-squared test helps us to select  important independent features out of the original features that have the strongest relationship with the target feature.

# ## (d) Anova-F Test
# 
# 
# The F-value scores examine the varaiance by grouping the numerical feature by the target vector, the means for each group are significantly different.

# In[96]:


#import libraries f_classif,SelectPercentile from sklearn
#import libraries f_classif,SelectPercentile from sklearn
#import libraries f_classif,SelectPercentile from sklearn
from sklearn.feature_selection import f_classif, SelectPercentile


# Initialise SelectPercentile function with parameters f_classif and percentile as 80
Anova_test = SelectPercentile(f_classif, percentile = 80)

#Fit the above object to the features and target i.e X and Y
Anova_model = Anova_test.fit(X,Y)


# here you have used f_classif for Anova-F test. To know more about this test you can check this artical.
# 
# https://towardsdatascience.com/anova-for-feature-selection-in-machine-learning-d9305e228476
# 
# 

# In[97]:


# return scores of anova model

Anova_model.scores_


# In[99]:


# use generate_feature_scores_df function to get features and their respective scores by passing X and Anova_model.scores_ as score in function 
feature_scores_df = generate_feature_scores_df(X, Anova_model.scores_)

feature_scores_df


# In[104]:


# Get all supported columns values in Anova_model with indices=True
cols = Anova_model.get_support(indices = True)

# Reduce X to the selected features of anova model using tranform 
X_new = X.iloc[:, cols]


# In[105]:


#print X_new.head()

X_new.head()


#  Anova F-test method for feature selection has selected 6 best features as you can see in above output

# # 2. Wrapper Methods
# 
# 

# Wrapper methods are used to select a set of features by preparing where different combinations of features, then each combination is evaluated and compared to other combinations.Next a predictive model is used to assign a score based on model accuracy and to evaluate the combinations of these features.

# In[106]:


# load and read the csv using pandas and print the head values
diabetes = pd.read_csv("diabetes.csv")
# print head
diabetes.head()


# In[110]:


# assign features to X and target 'outcome' to Y(Think why the 'outcome' column is taken as the target)
X = diabetes.drop('Outcome', axis = 1)
Y = diabetes['Outcome']

#return X,Y
X, Y


# ## (a) Recursive Feature Elimination
# 
# 
# 
# Recursive Feature Elimination selects features by recursively considering smaller subsets of features by pruning the least important feature at each step. Here models are created iteartively and in each iteration it determines the best and worst performing features and this process continues until all the features are explored.Next ranking is given on eah feature based on their elimination orde. In the worst case, if a dataset contains N number of features RFE will do a greedy search for $N^2$ combinations of features.

# In[111]:


# import required libraries RFE, LogisticRegression and dependencies

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[113]:


# Initialise model variable with LogisticRegression function with solver = 'liblinear'
model = LogisticRegression(solver = 'liblinear')

# rfe variable has RFE instance with should have model and n_features_to_select=4 as parameters
rfe = RFE(model, n_features_to_select=4)


# In[114]:


# fit rfe with X and Y
fit = rfe.fit(X, Y)


# In[116]:


# print fit.n_features_, fit.support_, fit.ranking_

print("Number of selected features -",fit.n_features_)
print("Selected Features -",fit.support_)
print("Rank -",fit.ranking_)


# In[118]:


# use below function to get ranks of all the features
def feature_ranks(X,Rank,Support):
    feature_rank=pd.DataFrame()
    for i in range(X.shape[1]):
        new =pd.DataFrame({"Features":X.columns[i],"Rank":Rank[i],'Selected':Support[i]},index=[i])
        feature_rank=pd.concat([feature_rank,new])
    return feature_rank


# In[119]:


#Get all feature's ranks using feature_ranks function with suitable parameters in variable called feature_rank_df
feature_rank_df = feature_ranks(X, fit.ranking_, fit.support_)

# print feature_rank_df

feature_rank_df


# We can see there are four features with rank 1 ,RFE states that these are the most significant features.

# In[120]:


# filter feature_rank_df  with selected column values as True and save it variable called recursive_feature_names 
recursive_feature_names =  feature_rank_df.loc[feature_rank_df["Selected"] == True]

# print recursive_feature_names

recursive_feature_names


# In[121]:


# finally get all the features selected by RFE in dataframe X and this result in variable called RFE_selected_features
RFE_selected_features = X[recursive_feature_names['Features'].values]

# print RFE head()
RFE_selected_features.head()


# # 3. Embedded Method using random forest
# 
# 

# Feature selection using Random forest comes under the category of Embedded methods. Embedded methods combine the qualities of filter and wrapper methods. They are implemented by algorithms that have their own built-in feature selection methods. Some of the benefits of embedded methods are :
# 1. They are highly accurate.
# 2. They generalize better.
# 3. They are interpretable

# In[129]:


#Importing libraries pd, RandomForestClassifier, SelectFromModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


# In[130]:


# load the csv file using pandas and print the head values
diabetes = pd.read_csv("diabetes.csv")

# print head

diabetes.head()


# In all feature selection procedures, it is a good practice to select the features by examining only the training set. This is to avoid overfitting.
# So considering we have a train and a test dataset. We select the features from the train set and then transfer the changes to the test set later

# In[131]:


# assign features to X and target 'outcome' to Y(Think why the 'outcome' column is taken as the target)
X = diabetes.drop('Outcome', axis = 1)
Y = diabetes['Outcome']


# In[132]:


# import test_train_split module


# splitting of dataset(test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 24)


# model fitting and feature selection together in one line of code.
# 
# Specify the random forest instance, indicating the number of trees.
# 
# Then using selectFromModel object from sklearn to automatically select the features. 
# 
# Reference link to use selectFromModel: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html

# In[133]:


#create an instance of Select from Model. Pass an object of Random Forest Classifier with n_estimators=100 as argument. 
sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))


# fit sel on X and y 

sel.fit(X, Y)


# SelectFromModel will select those features which importance is greater than the mean importance of all the features by default, but we can alter this threshold if we want.
# 
#  To see which features are important we can use get_support method on the fitted model.

# In[134]:


# Using sel.get_support() print the boolean values for the features selected. 

sel.get_support()


# In[135]:


#make a list named selected_feat with all columns which are True
selected_feat = X_train.columns[sel.get_support()]

# print length of selected_feat
len(selected_feat)


# In[136]:


# Print selected_feat
print(selected_feat)


# ## Feature selection using SelectFromModel
# 
# 
# 
# SelectFromModel is a meta-transformer that can be used along with any estimator that has a coef_ or featureimportances attribute after fitting. The features are considered unimportant and removed, if the corresponding coef_ or featureimportances values are below the provided threshold parameter. Apart from specifying the threshold numerically, there are built-in heuristics for finding a threshold using a string argument. Available heuristics are “mean”, “median” and float multiples of these like “0.1*mean”.
# 
# Lets use selectfrommodel again with LinearSVC

# In[138]:


# import libraries LinearSVC, SelectFromModel , and dependencies

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


# In[139]:


#Use SelectFromModel with LinearSVC() as its parameter and save it in variable 'm'

m = SelectFromModel(LinearSVC())

# fit on X, y

m.fit(X,Y)


# In[140]:


#make a list named selected_feat with all columns which are supported and count the selected features.

selected_feat = X_train.columns[(m.get_support())]

# print selected_feat

selected_feat


#  # 4. Handling Multicollinearity with VIF
# 

# Multicollinearity refers to a situation in which more than two explanatory variables in a multiple regression model are highly linearly related. We have perfect multicollinearity if, for example as in the equation above, the correlation between two independent variables is equal to 1 or −1.

# Variance inflation factor measures how much the behavior (variance) of an independent variable is influenced, or inflated, by its interaction/correlation with the other independent variables.
# 
# VIF has big defination but for now understand that:-
# Variance inflation factor (VIF) is a measure of the amount of multicollinearity in a set of multiple regression variables

# In[141]:


#load and read the diabetes_cleaned.csv file using pandas and print the head values
dia_df = pd.read_csv("diabetes.csv")

# print dia_df

dia_df.head()


# In[142]:


# describe the dataframe using .describe()

dia_df.describe()


# These features are very different that means they all are in different scales so lets standardize the features using sklearn's scale function.
# 
# reference doc: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.scale.html

# In[144]:


# import preprocessing
from sklearn import preprocessing

#iterate over all features in dia_df and scale

for i in dia_df:
    dia_df[[i]]= preprocessing.scale(dia_df[[i]].astype('float64'))


# In[145]:


# describe dataframe using .describe()

dia_df.describe()


# In[146]:


# import train_test_split

from sklearn.model_selection import train_test_split


# In[149]:


#import variance inflation factor 

from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[152]:


# assign features to X and target to Y by analyzing which columns to be dropped and which is to be considered as target
X = dia_df.drop('Outcome', axis = 1)

Y =dia_df['Outcome']



# split the data to test and train with test_size=0.2
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2)


# In[154]:


#assign an empty dataframe to variable vif
vif = pd.DataFrame()

# make a new column 'VIF Factor' in vif dataframe and calculate the variance_inflation_factor for each X 
vif['VIF Factor'] = [variance_inflation_factor(X.values, i)for i in range(X.shape[1])]


# In[155]:


# define vif['Features'] with columns names in X

vif['Features'] = X.columns


# In[156]:


#  round off all the decimal values in the dataframe to 2 decimal places for VIF dataframe
vif.round(2)


# * VIF = 1: Not correlated
# * VIF =1-5: Moderately correlated
# * VIF >5: Highly correlated
# 
# Glucose, Insulin, and Age are having large VIF scores, so lets drop it.
# 
# 

# In[157]:


# according to above observation , drop  'Glucose', 'Insulin' and 'Age' from X
X = X.drop('Glucose', axis =1)


# Now again we calculate the VIF for the rest of the features
# 
# repeating the previous steps to assign an empty dataframe() to vif and make a new column 'VIF Factor' and calculate the variance_inflation_factorfor each X 
# 

# In[158]:


# create an empty df
vif = pd.DataFrame()

# make a new column 'VIF Factor' and calculate the variance_inflation_factorfor each X
vif['VIF Factor'] =  [variance_inflation_factor(X.values, i)for i in range(X.shape[1])]



# In[159]:


#define vif['Features'] as columns of X and return vif with round off to 2 decimal places
vif['Features'] = X.columns
vif['Features'] = X.columns

# round up to 2 
vif.round(2)


# So now colinearity of features has been reduced using VIF.

# 
# The need to fix multicollinearity depends primarily on the below reasons:
# ---
# 
# 1. If focusing on how much each individual feature rather than a group of features affects the target variable, then removing multicollinearity may be a good option
# 
# 2. If multicollinearity is not present in the features you are interested in, then multicollinearity may not be a problem.

# ## **`Conclusion:`**
# Feature selection is a very important step in the construction of Machine Learning models. <br>
# 
# It can speed up training time, make our models simpler, easier to debug, and reduce the time to market of Machine Learning products. 

# ------------------------------
