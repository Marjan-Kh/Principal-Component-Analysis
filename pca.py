# The goal of this program:

# 1) how principal component analysis works.
# 2) On what type of problems can be applied.

#============================================
# Author: Marjan Khamesian
# Date: June 2020
#============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# ==========================
# Generate some random data:
# 6 random features, 3 of them have some form of correlation.

# Random Features : a dictionary of random numbers
data_raw = {'x{}'.format(i):np.random.random(100) for i in range(3)}

# Turn the dictionary into a Pandas dataframe
data = pd.DataFrame(data_raw)
print(data)

# =========================================================
# functions as a dictionary to generate the correlated data
funcs = {'x6': lambda x: 12*x**2, 'x7': lambda x:23*x**3, 'x8': lambda x:27*x**3 + 32*x**2,}

# Correlated Data 
correlated_data = {col:funcs[col](np.random.random(100)) for col in funcs.keys()}

# Turn the dictionary into a Pandas dataframe
data_c = pd.DataFrame(correlated_data)
print(data_c)    # 100 rows × 3 columns

# Join the two dataframes
data = data.join(pd.DataFrame(correlated_data))
data.describe()

# Generate a regression target
data['y'] = data.iloc[:,:3].sum(axis=1) + \
            0.15*data.iloc[:,3] + \
            0.07*data.iloc[:,4] + \
            0.02*data.iloc[:,5]

data.describe()

# ============================================
# Convert the target to classification problem

# Generate a classification target
def label_func(x):
    if x < 3.0:
        return 0
    elif ((x>=3.0) and (x<4.1)):
        return 1
    else:
        return 2

data['target'] = data['y'].apply(label_func)
data.describe()

# ============================
# Correlation between Features

import seaborn as sns

# Calculate and plot the correlation between all features and target
sns.heatmap(data.corr())

# Construct the Covariance matrix
cov_mtx = data.drop(['y', 'target'], axis=1).cov()

print('Covariance matrix:')
print(cov_mtx)
print("\n================================\n")

# ==================================
# Singular Value Decomposition (SVD)
# ==================================

# Find singular values by applying SVD on the constructed correlation matrix

U, S, V = np.linalg.svd(cov_mtx, full_matrices=False)

# ===  Compare with PCA ============

from sklearn.decomposition import PCA

# Apply PCA to features of data
pca = PCA(n_components=6)
X_t = pca.fit_transform(data.drop(['y','target'], axis=1))

# =======================================================================
# Print covariance matrix from PCA and compare it from previous approach.
# Check if both approach lead to the same result.

cov_pca = pca.get_covariance() # covariance matrix from PCA 

np.abs(cov_pca - cov_mtx) < 0.00001*np.ones(cov_mtx.shape) # Comparison

#========================================================================
# Print singular values from PCA and compare them from previous approach.

S_pca = pca.singular_values_  # singular values from PCA

np.abs(S - S_pca) < 0.00001*np.ones(S.shape)  # Comparison
