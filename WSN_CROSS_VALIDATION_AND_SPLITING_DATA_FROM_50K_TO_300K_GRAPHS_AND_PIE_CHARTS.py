
# coding: utf-8

# In[1]:



# coding: utf-8

# In[1]:


#all header files
#import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz

import seaborn as sns


import pandas as pd
from pandas import Series
import numpy as np
import csv
import matplotlib.pyplot as plt


# In[4]:


#loading dataset
df = pd.read_csv('WSN-DS.csv')
df.head()


# In[5]:




# In[6]:


df


# In[7]:


#changing collumn name

df = df.rename(columns={"Attack type": "Attack"})
df = df.rename(columns={"Consumed Energy": "Energy"})
df = df.rename(columns={"who CH": "who_CH"})
df = df.rename(columns={"Is_CH": "CH"})



# In[8]:


# 0 normal and greater than 0 is attacking 
#type = {'Normal': 0,'TDMA': 1,'Flooding':2,'Grayhole':3,'Blackhole':4} 
   
#df.Attack = [type[item] for item in df.Attack] 
#pd.set_option('display.max_rows', 50)

df.head()


# In[9]:


# create design matrix X and target vector y
X = np.array(df.ix[:, 1:]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
x_try=X_test[0:19,0:19]


# In[10]:


index = df.index
columns = df.columns
 


# In[11]:


index


# In[20]:


from matplotlib.gridspec import GridSpec

import seaborn as sns

sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df
x=['50k','100k','150k','200k','250k','300k','Total']
y=[0.9709,0.97225,0.9778666666666667,0.981375,0.98364,0.9845166666666667,0.9862010062322342]
plt.figure(300, figsize=(50,25))
the_grid = GridSpec(4, 4)

plt.subplot(the_grid[0,1],  title=' Comparison of Data size, Accuracy OF KNN ')
sns.barplot(x,y, data=source_data, palette='Spectral')



# In[4]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df
x=['50k','100k','150k','200k','250k','300k','Total']
y=[0.7673,0.76375,0.7772666666666667,0.77855,0.7714,0.7788166666666667,0.8687761066552787]
plt.figure(2, figsize=(35,35))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' Comparison of Data size, Accuracy OF Naive Bayes')
sns.barplot(x,y, data=source_data, palette='Spectral')


# In[5]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df
x=['50k','100k','150k','200k','250k','300k','Total']
y=[0.9554,0.9606,0.9626666666666667,0.95835,0.961,0.9623166666666667,0.9671840177224987]
plt.figure(2, figsize=(35,55))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' Comparison of Data size, Accuracy  OF Logistic Regression')
sns.barplot(x,y, data=source_data, palette='Spectral')


# In[7]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df
x=['50k','100k','150k','200k','250k','300k','Total']
y=[0.9042,0.90595,0.9111333333333334,0.905775,0.91452,0.9174833333333333,0.923918700705964]
plt.figure(2, figsize=(35,55))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' Comparison of Data size, Accuracy  OF SVM')
sns.barplot(x,y, data=source_data, palette='Spectral')

