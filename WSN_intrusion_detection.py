
# coding: utf-8

# In[1]:


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


df = pd.read_csv('WSN-DS.csv')
df.head()


# In[3]:


df = df.rename(columns={"Attack type": "Attack"})
df = df.rename(columns={"Consumed Energy": "Energy"})
df = df.rename(columns={"who CH": "who_CH"})


# In[4]:


# 0 normal and greater than 0 is attacking 
type = {'Normal': 0,'TDMA': 1,'Flooding':2,'Grayhole':3,'Blackhole':4} 
   
df.Attack = [type[item] for item in df.Attack] 
pd.set_option('display.max_rows', 50)

df.head()


# In[5]:


# create design matrix X and target vector y
X = np.array(df.ix[:, 1:]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
x_try=X_test[0:19,0:19]


# In[6]:



# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)


# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test, pred))
cm=confusion_matrix(y_test,pred)
print("Normal|  TDMA|  Flooding|  Grayhole| Blackhole ")

print(cm)

NBC = knn.score(X_test,y_test)

print("Confusion Matrix For Naive Bayes(Multinominal) Classifier: ")
print("......Normal|  TDMA|  Flooding|  Grayhole| Blackhole ")

print("Normal   : ",cm[0],"\nTDMA    : ",cm[1],"\nFlooding: ",cm[2],"\nGrayhole: ",cm[3],"\nBlackhole: ",cm[4])

print("\nAccuracy of Naive Bayes Classifier: ",NBC*100,"%\n")
print("Error of Naive Bayes Classifier: ",(1-NBC)*100,"%\n")


# In[8]:


# 0 normal and 1 attacking 
type = {0: 0,1: 1,2:1,3:1,4:1} 
   
df.Attack = [type[item] for item in df.Attack] 
pd.set_option('display.max_rows', 50)

df.head()


# In[9]:


# create design matrix X and target vector y
X = np.array(df.ix[:, 1:]) 	# end index is exclusive
y = np.array(df['Attack']) 	# another way of indexing a pandas df

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
x_try=X_test[0:19,0:19]


# In[10]:


X_train


# In[11]:


X_test


# In[12]:


x_try


# In[13]:


y_test


# In[14]:


y_train


# In[15]:



# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)


# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test, pred))
cm=confusion_matrix(y_test,pred)
print(cm)


tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
cm = [tn,fp,fn,tp]
NBC = knn.score(X_test,y_test)

print("Confusion Matrix For Naive Bayes(Multinominal) Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Naive Bayes Classifier: ",NBC*100,"%\n")
print("Error of Naive Bayes Classifier: ",(1-NBC)*100,"%\n")


NBP = tp/(tp+fp)
NBR = tp/(tp+fn)


# In[16]:


NB = MultinomialNB(alpha=1.5)
NB.fit(X_train,y_train)
NB_predicted = NB.predict(X_test)
NBC = NB.score(X_test,y_test)


pred = NB.predict(X_test)

print (accuracy_score(y_test, pred))
cm=confusion_matrix(y_test,pred)
print(cm)

tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Naive Bayes(Multinominal) Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Naive Bayes Classifier: ",NBC*100,"%\n")
print("Error of Naive Bayes Classifier: ",(1-NBC)*100,"%\n")


NBP = tp/(tp+fp)
NBR = tp/(tp+fn)


# In[17]:


LR = LogisticRegression()
LR.fit(X_train,y_train)
LR_predicted = LR.predict(X_test)
LRC = LR.score(X_test,y_test)

pred = LR.predict(X_test)
print (accuracy_score(y_test, pred))
cm=confusion_matrix(y_test,pred)
print(cm)

tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Logistic Regression Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Logistic Regression Classifier: ",LRC*100,"%\n")
print("Error of Logistic Regression Classifier: ",(1-LRC)*100,"%\n")


LRP = tp/(tp+fp)
LRR = tp/(tp+fn)


# In[18]:


S = LinearSVC()
S.fit(X_train,y_train)
S_predicted = S.predict(X_test)
SC = S.score(X_test,y_test)


pred = S.predict(X_test)
print (accuracy_score(y_test, pred))
cm=confusion_matrix(y_test,pred)
print(cm)


tn,fp,fn,tp=confusion_matrix(y_test,pred).ravel()
cm = [tn,fp,fn,tp]
print("Confusion Matrix For Linear Support Vector  Classifier: ")
print("TN: ",cm[0],"  FP: ",cm[1],"\nFN: ",cm[2],"    TP: ",cm[3])

print("\nAccuracy of Linear Support Vector Classifier: ",SC*100,"%\n")
print("Error of Linear Support Vector Classifier: ",(1-SC)*100,"%\n")

SCP = tp/(tp+fp)
SCR = tp/(tp+fn)


# In[47]:


plt.hist([[NBC],[LRC],[SC],[RFC]], bins=2, color=['red','blue','yellow','orange'])
plt.xlabel=('"Classifiers')
plt.ylabel=("Accuracy")
plt.title=('Accuracy of different classifiers')
plt.show()




X = [NBC,LRC,SC,RFC]
plt.hist(X)
plt.ylabel=("Accuracy")
plt.xlabel=("NBC","LRC","SC","RFC")
plt.show()




X = [1-NBC,1-LRC,1-SC,1-RFC]
plt.hist(X)
plt.ylabel=("Error")
plt.xlabel=("NBC","LRC","SC","RFC")
plt.show()




X = [NBP,LRP,SCP,RFP]
plt.hist(X)
plt.ylabel=("Precision")
plt.xlabel=("NBC","LRC","RFC")
plt.show()




X = [NBR,LRR,SCR,RFR]
plt.hist(X)
plt.ylabel=("Recall")
plt.xlabel=("NBC","LRC","SC","RFC")
plt.show()

