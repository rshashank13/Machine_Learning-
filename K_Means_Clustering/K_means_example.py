# -*- coding: utf-8 -*-
from K_means_model import K_Means
from sklearn import datasets

dat = datasets.load_iris()
X = dat.data

my_model = K_Means()
clusters = my_model.run_K_means(X,3)
print('Cluster_centroids:\n',clusters)
my_model.plot_cost()

'''
Expected_output: According to https://www.kaggle.com/aschakra/k-means-clustering-for-iris-dataset
1      5.006000     3.418000      1.464000     0.244000
2      5.901613     2.748387      4.393548     1.433871
3      6.850000     3.073684      5.742105     2.071053
'''