
# coding: utf-8

# In[1]:


#all header files

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

import pandas as pd
from pandas import Series
import numpy as np
import csv
import matplotlib.pyplot as plt


# In[2]:


#loading dataset
df = pd.read_csv('WSN-DS.csv')
df.head()


# In[3]:


#changing collumn name

df = df.rename(columns={"Attack type": "Attack"})
df = df.rename(columns={"Consumed Energy": "Energy"})
df = df.rename(columns={"who CH": "who_CH"})


# In[4]:


# 0 normal and greater than 0 is attacking 
type = {'Normal': 0,'TDMA': 1,'Flooding':1,'Grayhole':1,'Blackhole':1} 
   
df.Attack = [type[item] for item in df.Attack] 
pd.set_option('display.max_rows', 50)

df.head()


# In[5]:


# create design matrix X and target vector y
X = np.array(df.ix[:, 1:]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
x_try=X_test[0:19,0:19]


# In[6]:


index = df.index
columns = df.columns
 


# In[7]:


index


# In[8]:


columns


# In[9]:


from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors=3)  


# In[10]:


from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5) 


# In[11]:


print(all_accuracies)  


# In[12]:


print(all_accuracies.mean())  


# In[13]:


NB = MultinomialNB(alpha=1.5)
from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator=NB, X=X_train, y=y_train, cv=5) 


# In[14]:


print(all_accuracies)  


# In[15]:


print(all_accuracies.mean())  


# In[16]:


LR = LogisticRegression()
from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator=LR, X=X_train, y=y_train, cv=5) 


# In[17]:


print(all_accuracies)  


# In[18]:


print(all_accuracies.mean())  


# In[19]:


S = LinearSVC()
from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator=S, X=X_train, y=y_train, cv=5) 
print(all_accuracies)  


# In[20]:


print(all_accuracies.mean())  


# In[21]:


DT = DecisionTreeClassifier(max_depth=10)
from sklearn.model_selection import cross_val_score  
all_accuracies = cross_val_score(estimator=DT, X=X_train, y=y_train, cv=5) 
print(all_accuracies)  


# In[29]:


from pprint import pprint
# Look at parameters used by our current forest
print('Parameters of discition tree currently in use:\n')
pprint(DT.get_params())


# In[31]:


print('Parameters of Naive Bayes currently in use:\n')
pprint(NB.get_params())


# In[33]:


print('Parameters of LR currently in use:\n')
pprint(LR.get_params())


# In[35]:


RF = RandomForestClassifier(max_depth=50)

print('Parameters of RF currently in use:\n')
pprint(RF.get_params())


# In[23]:


from sklearn import svm, grid_search
from sklearn.model_selection import GridSearchCV


# In[24]:


from sklearn.ensemble import RandomForestClassifier  
classifie = RandomForestClassifier(n_estimators=300, random_state=0) 


# In[25]:


grid_param = {  
    'n_estimators': [10000, 30000, 50000, 80000, 100000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}


# In[26]:


grid_param


# In[27]:


gd_sr = GridSearchCV(estimator=classifie,  
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)


# In[28]:


print(gd_sr)

