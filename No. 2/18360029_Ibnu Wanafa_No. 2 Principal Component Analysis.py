from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pizzaDataset = pd.read_csv('Pizza.csv',names = ['Brand', 'mois', 'prot', 'fat', 'ash','sodium','carb','cal'], header=0, sep=",")
X = pizzaDataset.drop(["Brand"], axis=1)
# standardize data
X_std = StandardScaler().fit_transform(X)
# create covariance matrix
cov_mat = np.cov(X_std.T)
print('Covariance matrix \n%s' % cov_mat)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' % eig_vecs)
print('\nEigenvalues \n%s' % eig_vals)
# sort eigenvalues in decreasing order
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
# Selecting the number of Principal Components
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Cummulative Variance Explained", cum_var_exp)
# Plotting the graphics
plt.figure(figsize=(8, 7))
plt.bar(range(7), var_exp, alpha=0.5, align='center',label='Individual explained variance')
plt.step(range(7), cum_var_exp, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
