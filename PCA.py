#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[2]:


## Dataset
mean_01 = np.array([0.0, 0.0])
cov_01 = np.array([[1.0, 0.8], [0.7, 1.6]])

data = np.random.multivariate_normal(mean_01, cov_01, 800)
print data.shape


# In[3]:


plt.scatter(data[:, 0], data[:, 1])
plt.show()


# In[32]:


def PCA(X, m=None):
    if m is None:
        m = X.shape[1]
        
    ## Find Covariance Matrix
    A = np.cov(X, rowvar=False)
    
    ## Eigenvalues and Eigenvectors
    eig_val, eig_vec = np.linalg.eig(A)
    
    ## Project points to new space
    Z = X.dot(eig_vec)
    
    ## Choose m dimensions
    red_Z = Z[:, -m-1:]
    
    return eig_val, eig_vec, Z, red_Z


# In[33]:


print 'Original Data Shape: ', data.shape 
eig_val, eig_vec, Z, red_Z = PCA(data, m=1)

print 'Eig_Vals shape: ', eig_val.shape
print 'Eig_Vecs shape: ', eig_vec.shape
print 'Z shape: ', Z.shape
print 'Red_Z shape: ', red_Z.shape


# In[34]:


print eig_val
print eig_vec


# In[35]:


indexes = np.array(range(red_Z.shape[0]))
print indexes.shape
for ix in range(red_Z.shape[0]):
    plt.scatter(indexes[ix], red_Z[ix,0],c='b')

plt.show()


# In[36]:


info = []
for ix in range(eig_val.shape[0]):
    dx = [eig_val[ix], eig_vec[:, ix]]
    info.append(dx)
print info


# In[37]:


for k in info:
    print k


# In[13]:


info = sorted(info, key=lambda z:z[0], reverse=True)


# In[14]:


sorted_vals = []
sorted_vecs = []
for ix in range(len(info)):
    sorted_vals.append(info[ix][0])
    sorted_vecs.append(info[ix][1])
print sorted_vals


# In[16]:


sorted_vals = np.array(sorted_vals)
explained_variances_ratio = sorted_vals/sum(sorted_vals)
print explained_variances_ratio


# In[17]:


ds = pd.read_csv('/Users/pawan/Downloads/fashionmnist/fashion-mnist_test.csv')


# In[18]:


ds.head()


# In[21]:


input_data = ds.values[:, 1:]
print input_data.shape


# In[ ]:




