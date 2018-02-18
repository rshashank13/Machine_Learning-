# -*- coding: utf-8 -*-
from scipy.spatial import distance
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class K_Means():
    
    def set_K(self,K):
        self.K = K
        
    def set_X(self,X):
        self.X = X
    
    def initialize_centroids(self):
        #Randomly initialize K items from dataset to be initial cluster-centroids
        indicies = np.array(list(rd.sample(range(0,self.X.shape[0]),self.K)))
        self.centroids = self.X[indicies,:].copy()
        self.old_centroids = np.zeros(self.centroids.shape)
        
    def assign_cluster(self):
        '''
        assigns clusters to each of the examples
        1. find out distance of a point to all the centroids
        2. assign point to the closest centroid's cluster by index of the centroid
        '''
        assigned_cluster = list() # holds index assigned to each of the example 
        indicies = np.array([i for i in range(self.centroids.shape[0])])
        for i in range(self.X.shape[0]):
            distance_vec = list()
            for j in range(self.centroids.shape[0]):
                dist = distance.euclidean(self.X[i],self.centroids[j])
                distance_vec.append(dist)
            distance_vec = np.array(distance_vec)
            min_dist = list(indicies[distance_vec==np.min(distance_vec)])
            assigned_cluster.append(min_dist[0])
        self.assigned_clusters = np.array(assigned_cluster).reshape((len(assigned_cluster),1))
        
    def move_centroid(self):
        '''
        Move the cluster centroids to the mean of the assigned points
        1. compute sum of all points assigned to a particular cluster centroid
        2. compute mean of each
        3. new centroid = mean of the points assinged to each clusters
        '''
        sums = np.zeros(self.centroids.shape)
        counts = np.zeros((self.centroids.shape[0],1))
        for i in range(self.X.shape[0]):
            ind = self.assigned_clusters[i,0]
            sums[ind] += self.X[i]
            counts[ind]+= 1
        for i in range(self.centroids.shape[0]):
            if counts[i]==0:
                sums[i] = 0
            else:
                sums[i] = sums[i]/counts[i]
        self.old_centroids = self.centroids.copy()
        self.centroids = sums
        
    def change_greater_than(self,epsilon):
        t_flag = 0
        f_flag = 0
        for i in range(self.centroids.shape[0]):
            if distance.euclidean(self.centroids[i],self.old_centroids[i])>= epsilon:
                t_flag += 1
            else:
                f_flag += 1
        if t_flag >= f_flag:
            return True
        else:
            return False
        
    def compute_cost(self):
        '''
        cost = average of distances of points to their centroids
        '''
        cost = 0
        for i in range(self.X.shape[0]):
            ind = self.assigned_clusters[i]
            cost += distance.euclidean(self.X[i,:],self.centroids[ind,:])
        cost = float(cost / self.X.shape[0])
        return cost
    
    def plot_cost(self):
        plt.figure(1)
        plt.title('Cost v/s Iteration')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        y = self.cost
        x = self.iters
        plt.plot(x,y)
        plt.show()
        
    def get_clusters(self,X,K,iterations,epsilon):
        '''
        K Means clustering:
            1. randomly initialize cluster centroids
            2. Iteratively do until convergence:
                -assign clusters to each datapoint
                -compute mean of datapoints per cluster
                -move the cluster centroids to the means
        '''
        self.set_K(K)
        self.set_X(X)
        self.num_of_iteration = iterations
        self.iters = []
        self.initialize_centroids()
        self.cost = []
        
        for i in range(iterations):
            self.assign_cluster()
            self.move_centroid()
            self.cost.append(self.compute_cost())
            self.iters.append(i+1)
            if self.change_greater_than(epsilon):
                pass
            else:
                break
        return self.centroids,self.iters,self.cost
    
    def run_K_means(self,X,K,iterations = 1500, epsilon = 1e-07,best_of=10):
        self.models = dict()
        costs = list()
        print('Running K means for ',best_of,' iterations:')
        for i in range(best_of):
            cen,iters,coi = self.get_clusters(X,K,iterations,epsilon)
            cost = self.compute_cost()
            self.models[i] = (cen,iters,coi)
            costs.append(cost)
            print('iteration:',i+1,', cost: ',cost)
        index = np.array([i for i in range(len(costs))])
        costs = np.array(costs)
        ind = index[costs==np.min(costs)]
        ind = ind[0]
        self.centroids = self.models[ind][0]
        self.iters = self.models[ind][1]
        self.cost = self.models[ind][2]
        return self.centroids
            
            