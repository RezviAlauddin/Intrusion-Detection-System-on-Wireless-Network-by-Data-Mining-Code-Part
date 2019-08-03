
# coding: utf-8

# In[ ]:



# coding: utf-8

# In[1]:


#all header files
import tensorflow as tf

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


# In[4]:


# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt(df, delimiter=",")
# split into input (X) and output (Y) variables


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, y, epochs=150, batch_size=10,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# In[4]:


# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset


# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))

# evaluate each model in turn
X = np.array(df.ix[:, 1:]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# In[14]:


df.plot.area()


# In[26]:


df.groupby('Attack').size()


# In[12]:


df=df.iloc[:,[5,6,7,8,11]]
ax = df.plot(kind='area', stacked=True, title='100 % stacked area chart')

ax.set_ylabel('Percent (%)')
ax.margins(0, 0) # Set margins to avoid "whitespace"
fig = plt.gcf()
fig.set_size_inches(20,20)
plt.show()


# In[27]:


df.iloc[:,[2,5,6,7,8,17,18]]


# In[28]:


df[df.Attack != 'Normal']


# In[29]:


df.groupby('Energy').size()


# In[33]:


df.groupby('Attack').size()


# In[102]:


# Data to plot
labels = 'Blackhole', 'Flooding', 'Grayhole', 'Normal','TDMA'
sizes = [10049, 3312, 14596,340066,6638]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','hotpink']
explode = (0,0,0,0.1,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
fig = plt.gcf()
fig.set_size_inches(10,10) # or (4,4) or (5,5) or whatever
plt.suptitle('Pie chart Of Total Dataset', fontsize=16)

plt.show()


# In[42]:


from matplotlib.gridspec import GridSpec

import seaborn as sns

sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ENERGY vs ATTACK')
sns.barplot(x='Attack',y='Energy', data=source_data, palette='Spectral')



# In[32]:


df.groupby('Attack').size()


# In[47]:


df1=df.iloc[:,2]


# In[51]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK vs Is_CH')
sns.barplot(x='Attack',y=df1, data=source_data, palette='Spectral')


# In[57]:


df2=df.iloc[:,3]


# In[59]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='who_CH vs ATTACK')
sns.barplot(x='Attack',y=df2, data=source_data, palette='Spectral')


# In[60]:


df.head()


# In[63]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' ATTACK vs Dist_to_CH')
sns.barplot(x='Attack',y=df.iloc[:,4], data=source_data, palette='Spectral')


# In[80]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' ATTACK vs ADV_S')
sns.barplot(x='Attack',y=df.iloc[:,5], data=source_data, palette='Spectral')


# In[81]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK vs ADV_R')
sns.barplot(x='Attack',y=df.iloc[:,6], data=source_data, palette='Spectral')


# In[82]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK VS JOIN_S')
sns.barplot(x='Attack',y=df.iloc[:,7], data=source_data, palette='Spectral')


# In[83]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK vs JOIN_R')
sns.barplot(x='Attack',y=df.iloc[:,8], data=source_data, palette='Spectral')


# In[84]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK VS SCH_S')
sns.barplot(x='Attack',y=df.iloc[:,9], data=source_data, palette='Spectral')


# In[85]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK VS SCH_R')
sns.barplot(x='Attack',y=df.iloc[:,10], data=source_data, palette='Spectral')


# In[86]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK VS Rank')
sns.barplot(x='Attack',y=df.iloc[:,11], data=source_data, palette='Spectral')


# In[87]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' ATTACK vs DATA_S')
sns.barplot(x='Attack',y=df.iloc[:,12], data=source_data, palette='Spectral')


# In[88]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title=' ATTACK VS DATA_R')
sns.barplot(x='Attack',y=df.iloc[:,13], data=source_data, palette='Spectral')


# In[89]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK Data_Send_To_BS')
sns.barplot(x='Attack',y=df.iloc[:,14], data=source_data, palette='Spectral')


# In[91]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK VS DIST_CH_To_BS ')
sns.barplot(x='Attack',y=df.iloc[:,15], data=source_data, palette='Spectral')


# In[92]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK vs send_code')
sns.barplot(x='Attack',y=df.iloc[:,16], data=source_data, palette='Spectral')


# In[94]:


sns.set(style="whitegrid")
#sns.set_color_codes("Spectral")

source_data = df


plt.figure(2, figsize=(20,15))
the_grid = GridSpec(2, 2)

plt.subplot(the_grid[0, 1],  title='ATTACK vs Time')
sns.barplot(x='Attack',y=df.iloc[:,1], data=source_data, palette='Spectral')

