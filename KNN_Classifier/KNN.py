# -*- coding: utf-8 -*-
from scipy.spatial import distance
import numpy as np
import math as mat

class K_NN():
    
    def __init__(self):
        self.inf = mat.inf
        
    def set_K(self,K):
        self.K = K
    
    def get_classes(self):
        distinct_classes = set()
        for item in self.Y_train:
            distinct_classes.add(item)
        return list(distinct_classes)
    
    def compute_distance(self, X1, X2):
        return distance.euclidean(X1, X2)
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.compute_class(row)
            predictions.append(label)
        return predictions
    
    def compute_class(self, test_sample):
        for k in range(2,len(self.X_train)):
            self.set_K(k)
            list_of_neighbours = self.compute_K_nearest_neighbours(test_sample)
            class_label = self.vote(list_of_neighbours)
            if class_label == None:
                pass
            else:
                return class_label
            
    def compute_K_nearest_neighbours(self,sample):
        distances = []
        indicies = []
        for i in range(len(self.X_train)):
            dist = self.compute_distance(sample,self.X_train[i])
            distances.append(dist)
            indicies.append(i)
        k_neighbours = []
        d = np.array(distances)
        ind = np.array(indicies)
        while len(k_neighbours) < self.K:
            list_ind = list(ind[d==np.min(d)])
            for i in list_ind:
                d[i] = self.inf
                k_neighbours.append(i)
        return k_neighbours #contains indicies of K_nearest_neighbours
    
    def vote(self,k_neighbours):
        classes = self.get_classes()
        votes = [i*0 for i in range(len(classes))]
        class_wise_neighbours = [self.Y_train[i] for i in k_neighbours]
        c_w_n = np.array(class_wise_neighbours)
        
        for i in range(len(classes)):
            count = np.sum(np.int32(c_w_n == classes[i]))
            votes[i] = count
        v = np.array(votes)
        if np.sum(np.int32(v==np.max(v)))>1:
            return None
        else:
            return classes[votes.index(max(votes))]

    def get_accuracy(self,pred,actu):
        num_correct = 0
        
        for i in range(len(pred)):
            if(pred[i] == actu[i]):
                num_correct += 1
        return (num_correct/len(actu))*100
