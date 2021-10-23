#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('train.csv')


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[6]:


df.isna().sum()


# In[8]:


q1 = df.quantile(0.10)
q3 = df.quantile(0.90)
iqr = q3 - q1


# In[9]:


outlier = ((df < (q1 - 1.5*iqr)) | (df > (q3 + 1.5*iqr))).values.sum()
print("number of outliers are ", outlier)


# In[10]:


df.describe()


# In[11]:


plt.figure(figsize=(16,8))
sns.countplot(x='subject', hue='Activity', data=df)
plt.title("User data", fontsize=20)
plt.show()


# In[12]:


df_PCA = df.drop(['Activity', 'subject'], axis=1)


# In[13]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_PCA)


# In[14]:


humanActivity = pd.DataFrame({'x':principalComponents[:,0], 'y':principalComponents[:,1] ,'label':df['Activity']})
humanActivity


# In[15]:


sns.lmplot(data=humanActivity, x='x', y='y', hue='label', fit_reg=False, height=8, markers=['^','v','s','o', '1','2'])
plt.show()


# In[16]:


pca.explained_variance_ratio_


# In[19]:


sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in df.columns[20:30]:
    index = index + 1
    fig = sns.kdeplot(df[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)


# In[20]:


sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='Activity', y= df.loc[df['subject']==15].iloc[:,7], data= df.loc[df['subject']==15], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)


# In[22]:


X_train = df.iloc[:,0:len(df.columns)-1]
Y_train = df.iloc[:,-1]


# In[23]:


X_test = df.iloc[:,0:len(df.columns)-1]
Y_test = df.iloc[:,-1]


# In[26]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# In[27]:


le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

le = LabelEncoder()
Y_test = le.fit_transform(Y_test)


# In[28]:


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)


# In[29]:


pca = PCA(0.95)

pca.fit(X_train)
pca.fit(X_test)

train_x_pca = pca.transform(X_train)
test_x_pca = pca.transform(X_test)

print(pca.n_components_)
print(pca.explained_variance_)


# In[30]:


ex_variance = np.var(train_x_pca,axis=0)
print(ex_variance)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)


# In[31]:


ex_variance = np.var(test_x_pca,axis=0)
print(ex_variance)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)


# In[32]:


from sklearn.linear_model import LogisticRegression   
  
classifier = LogisticRegression(penalty='l2',solver='lbfgs',class_weight='balanced', max_iter=10000,random_state = 0) 
classifier.fit(train_x_pca, Y_train)
print(Y_train)


# In[33]:


y_pred = classifier.predict(test_x_pca)
print(test_x_pca)


# In[34]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred) 
print (cm)


# In[36]:


plt.figure(figsize=(10,10))
plt.pie(np.array(df['Activity'].value_counts()), labels = sorted(df['Activity'].unique()), autopct = '%0.6f')


# In[37]:


from sklearn.metrics import classification_report,accuracy_score
print(classification_report(Y_test,y_pred))
print("Accuracy:",accuracy_score(Y_test, y_pred)*100)

print(y_pred)


# In[39]:


from IPython.display import display,HTML
c1,c2,f1,f2,fs1,fs2='#eb3434','#eb3446','Akronim','Smokum',30,15

def dhtml(string,fontcolor=c1,font=f1,fontsize=fs1):
 display(HTML("""<style>
 @import 'https://fonts.googleapis.com/css?family="""\
 +font+"""&effect=3d-float';</style>
 <h1 class='font-effect-3d-float' style='font-family:"""+\
 font+"""; color:"""+fontcolor+"""; font-size:"""+\
 str(fontsize)+"""px;'>%s</h1>"""%string))
 
 
dhtml('created by AVANISH')


# In[ ]:




