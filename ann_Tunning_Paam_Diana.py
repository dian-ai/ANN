#1- is data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
x[:,1] = labelencoder_X_1.fit_transform(x[:,1])

labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2])

ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

x = x[:,1:]

#splitting our datas into 2 groups


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



#Feature Scaling:

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)


# Parameter Tunning
#we have parameters that are learned from model due to training like weights
#and we have some othere that are fixed called hyperparameter:
#epochs, batch size, optimizer, #neurons in layers, so with Tunning these fixed
#parameter we can get a better accuracy with k-fold cross validation
#with Gridsearch we find best values leading to best accuracy
#we use KerasClassifier to wrap ANN in a classfier that can be used for ScikitLearn
#and we use GridSearch cv class (instead of cross_val_score function), then we
#creat an object of this class, that will apply parameter tunning on our
#KerasClassifier that is our traditional NN.

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

#we are ready to start implementing K-fold cross validation inside keras
# to use KerasClassifier we should define a function (its an argument)
#this function build the architecture of ANN that we have built on part 2 and 3

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim= 11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

#we wrapp the architecture of ANN in the global classifier var. witth KerasClassifier
#but we wont put the arguments that we want to tune in the object KerasClassifier,
#we put them in GridSearch object
    
classifier = KerasClassifier(build_fn= build_classifier)

#creating GridSearch object: we create a dictionary that contains hyperparameters
#that we want to optimize, the values of the  keys are the values we want to try
#the GridSearch will test all the combinations of the values we put into key values
#we can also tune some of hyperparamters in the architecture of ANN, like optimizer etc. (rmsprop is optimizer based on stochastic gradient descnet for regular NN)


parameters = {'batch_size':[25,32], 'epochs':[500,100], 'optimizer': ['rmsprop', 'adam']}

#now implement Gridsearch, by creatng an oject of this class, that will contain parameter dictionary, and it contains our classifier(our estimator), and it holds info related to Kfold cross validation so we need to have scoring metric which is going to be our accuracy and number of folds

grid_search =GridSearchCV(estimator = classifier, param_grid= parameters, scoring= 'accuracy', cv=10)

#this Gridsearch object is not yet fitted to the training set.

grid_search= grid_search.fit(x_train, y_train)

#we are interested in best selection of parameters and the best accuracy, the attributes to GridSearchCV object is best_params_ , best_score_

best_parameters= grid_search.best_params_
best_accuracy=grid_search.best_score_
