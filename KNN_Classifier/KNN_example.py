# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split 
from sklearn import datasets
from KNN import K_NN
Datset = datasets.load_iris()

X = Datset.data
y = Datset.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)


my_classifier = K_NN()
my_classifier.fit(X_train,y_train)
predictions = my_classifier.predict(X_test)
print('accuracy: '+str(my_classifier.get_accuracy(predictions,y_test)))
