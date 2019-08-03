
# coding: utf-8

# In[132]:


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
from sklearn import tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz


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


# In[133]:


df['Attack'].describe()


# In[134]:


df.groupby('Attack').size()


# In[135]:


df.describe().transpose()


# In[136]:


df.std()


# In[137]:


# In[8]:


# 0 normal and greater than 0 is attacking 
type = {'Normal': 0,'TDMA': 1,'Flooding':2,'Grayhole':3,'Blackhole':4} 
   
df.Attack = [type[item] for item in df.Attack] 
pd.set_option('display.max_rows', 50)

df.head()


# In[9]:


# create design matrix X and target vector y
X = np.array(df.ix[:, :19]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
x_try=X_test[0:19,0:19]


# In[10]:


index = df.index
columns = df.columns
 


# In[11]:


index


# In[127]:


columns


# In[128]:


df.describe().transpose()


# In[94]:


df.describe()


# In[95]:


df.tail()


# In[64]:


#df=df.iloc[:,[2,17,18]]


# In[96]:


df.columns


# In[97]:


df.std()


# In[98]:


df['Attack'].describe()


# a=df.Attack
# bins=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
# plt.hist(a,bins,histtype='bar',rwidth=0.8)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# In[146]:




from apyori import apriori
association_rules=apriori(df,min_support=0.0056,min_confidence=0.2,min_lift=3,min_lenghth=3)
association_results=list(association_rules)


# In[147]:


print(association_results)


# In[148]:


association_results[0]


# In[149]:


association_results[1]


# In[150]:


association_results


# In[151]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.datasets import load_wine
from IPython.display import SVG
from graphviz import Source
from IPython.display import display


# In[152]:


# class labels
labels = df.Attack
# print dataset description
#print(df.DESCR)
estimator = DecisionTreeClassifier(criterion='gini')
estimator.fit(X_train, y_train)

graph = Source(tree.export_graphviz(estimator, out_file=None
   , feature_names=labels,
    class_names=df.columns
   , filled = True))
display(SVG(graph.pipe(format='svg')))


# In[ ]:


dt=DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=30)
fit_decision=dt.fit(X_train,y_train)
from graphviz import Source
from sklearn import tree
Source( tree.export_graphviz(fit_decision, out_file=None, feature_names=df.columns))

