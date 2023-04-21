#!/usr/bin/env python
# coding: utf-8

# ### Data Preprocessing Session by Engr. Raj Kumar 

# #### Importing libraries 

# In[4]:


import numpy as np 
# numpy is a fundamental library used to do scientific calculations

import pandas as pd
# pandas libaray is used to manage the datasets


# #### Reading the dataset 

# In[9]:


dataset = pd.read_csv('Data.csv')
# used to read the dataset that is uploaded on Jupyter 


# In[10]:


dataset
# to view your dataset over here


# #### Defining or separating the independent variables from our dataset

# In[14]:


x = dataset[['Country', 'Age', 'Salary']]
# defining the features of country, age, and salary as an indepent variables


# In[16]:


x
# to view the values of x or the data which we defined by x


# In[17]:


x = dataset[['Country', 'Age', 'Salary']].values
# to set these independent variables in an array form

x
# to view the array form of independent variables


# In[20]:


y = dataset[['Purchased']]
# defining the depended variable as a y 

y
# to view the dependent variable values in our dataset


# In[22]:


y = dataset[['Purchased']].values
# to set these dependent variables in an array form

y
# to view the array form of dependent variables


# #### To find out the unknown values in our dataset

# In[25]:


from sklearn.impute import SimpleImputer

# importing the SimpleImputer class from sklearn library


# In[30]:


imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')

# using the function of finding missing values 
# it needs two input values one is unknown number that is nan
# the output that we want to give for this nan number is the mean of dataset


# In[32]:


imputer.fit(x[:,1:3])

# now this take all the rows and columns to find out the missing values


# In[33]:


x[:,1:3] = imputer.transform(x[:,1:3])

# it will replace all the missing values with the mean 


# In[35]:


x
# to check the output
# you can see that unknown values are replaced with mean


# #### Encoding Catagorial Data
# ##### Here we will convert the names of countries in the numbering values like 0 or 1

# In[37]:


from sklearn.preprocessing import LabelEncoder 

# importing the label encoder class from sklearn library


# In[38]:


label_encoder_x = LabelEncoder()

# used for label encoder for x variable


# In[41]:


label_encoder_x.fit_transform(x[:,0])


# In[43]:


x[:,0] = label_encoder_x.fit_transform(x[:,0])
# values of label encoder is stored in variable x[:0]

x
# to view the values
# you can see now, we don't have name of countries
# we have the numerical data for name of countries
# 1 for France and 2 for Germany


# #### Using another dummy example that will give clear number coding for countries name

# In[48]:


from sklearn.preprocessing import OneHotEncoder

# importing the dummy encoder class of OneHotEncoder from sklearn library


# In[49]:


onehotencoder = OneHotEncoder()
# function OneHotEncoder is stored in onehotencoder


# In[51]:


onehotencoder.fit_transform(dataset.Country.values.reshape(-1,1)).toarray()

# performing or using this encoder for our dataset
# reshape is used for the single sample otherwise it will give error


# #### Now again move towards our example of LabelEncoder 

# In[52]:


labelencoder_y = LabelEncoder()


# In[55]:


y = labelencoder_y.fit_transform(y)
y
# we are using our previous label encoder b/c we've already values in yes/no


# #### Training and Testing of Dataset

# ##### Now here training and testing will be on some part of dataset. If we will use the whole dataset for training purposes then it might be difficult for model to findout the correlation. OR If we will use the whole dataset for training purposes then we will not have the testing dataset for testing purposes. If we are using the whole dataset for training and from that dataset we are also doing testing then it is useless to findout the predictions because we have already trained machine on that datsset. 

# ##### So it is necessary to train the some part of dataset and use reamining part for testing purposes to check our performance or results

# In[59]:


from sklearn.model_selection import train_test_split

# importing a class from sklearn for the spliting of data in train and test


# In[62]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# now here we gave four arguments to our data as shown
# now in arguments x and y are the array
# test_size 0.2 shows that we use 20% dataset for testing 
# random state=0 will show that it will not give us the random results
# random is zero means it will give us same results


# In[63]:


x_train
# checking the x_train values 


# In[64]:


x_test
# checking the x_test values


# In[65]:


y_train
# checking the y_train values 


# In[66]:


y_test
# checking the y_test values


# #### Feature Scaling 

# ##### We use feature scaling to make the features in our dataset of same scaling. Possibly there can be some high varying features in terms of range/magnitude/values in our dataset so we will perform the feature scaling to make all the features of same scale. 

# In[68]:


from sklearn.preprocessing import StandardScaler

# importing class of StandardScaler from the sklearn library


# In[69]:


sc_x = StandardScaler()

# store the value of StandardScaler in sc_x


# In[70]:


x_train = sc_x.fit_transform(x_train)

# applying the standard scaler on x_train


# In[71]:


x_test = sc_x.transform(x_test)

# applyig the standard scaler on x_test as well


# In[73]:


x_train
# showing the results at standard scale


# In[75]:


x_test
# showing the results at standard scale


# ##### All these are the data pre-processing steps. Now our data is ready for any machine learning algorithm. 

# ##### For any comment of query you can approach me at engr.rajkumar898@gmail.com

# #### Thank You

# In[ ]:




