# we are trying to figure out which costumers are leaving the bank
# is a classification problem and ANN can do a perfect job
#install keras library
#Theano : s fast open source numerical omputation library, very efficient
#Tensoflow also runs fast computations
#Keras wraps ensorflow and theano, used to build efficient deep-L models
#we use scikit-learn to build efficient ML models

# There will be two parts to build this model 
#1- is data preprocessing
#2- is creating ANN model

#Classification Template_Diana
# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data

dataset = pd.read_csv('Churn_Modelling.csv')

#we have figured out which independent variables have impact
#ANN will figure out which one has higher impact via weights

x = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#we need to encode our categoical features before splitting our data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#we have to create 2 encoder objects because of Country and Gender


labelencoder_X_1 = LabelEncoder()
x[:,1] = labelencoder_X_1.fit_transform(x[:,1])


#to apply lableencoder object to our column

#2nd object

labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2])

#Country has 3 categories and Gender has 2, then we have to remove one column to avoide
#dummy variable trap we prepare only for Country
#if you have m numbe of categories use m-1 in the model

#x[:,1] = labelencoder_X_1.fit_transform(x[:,1])


ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
x = ct.fit_transform(x)

# now we have to remove one dummy variable here to avoide falling into dummy Var. trap
# we take all the columns except the 1st column(index 0) and leave them in x matrix


x = x[:,1:]


#splitting our datas into 2 groups


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)



#Feature Scaling:

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)

#Part 2 - Make ANN
#Importing Keras Libraries and packages


import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#we need to import sequential module to initialize our neural network
#also Dense module to build layers of our ANN

#Initializing the ANN, defining it as a sequence of layers
#two ways to initialize: either by defining a sequence of layers or defining a grapgh
# we definie it by sequence, so we create an object of a class
#this object is the model itself,  neural network that will have a role of classifiers here
#so we call it classifier


classifier = Sequential()


#we dont need any input in this class, since we are going to define the layers step by step
#1st we add input layer, then a hidden layers & then more hidden layers & then output layer

#step 1: Randomly initialize the weights is done by Dense
#step2: 1st observation row goes as input into NN, each feature goes into 1 inpute node
#number of inpute nodes = 11 #independdent variables hat  we have in matrix of features
#step 3: forward propagation - activation function, defines the impact of each neuron
#to define a hiden layer we need to choose the activation function - we choose rectifier
#Q(x) = max(x,0)
# for the output layer we pickthe sigmoid function Q(x)=1/(1+exp(-x)) 
#it gives probability of  class = 1 for each observation, probability if the customer 
#leaves the bank o stays in the bank segmentation model, make a ranking of the customers,
#ranked by their probability to leave the bank
#step4: algorithm compares the predicted resut to actual results>> produce an error
#step5: error will be back propagated in NN >> all the weights get udated according to
#how much they are responsible for the error
#seveeral ways of updating these weights, by learning rate parameter
#step6: we repeat, either by each observation or batch of observation
#step7: when the whole training set passed thorugh ANN, it makes an epoch, redo many epochs

#with add method we add the different layers in our neural network
#right now we add input layer and 1st hidden layer with Dense,
#arguments in Dense: how weights are updated, what is activation func., 
#number of nodes in layer, number of input nodes
#by adding hidden layer we specify the number of nodes in input layer8previous layer)

##tip: choose  #nodes in hidden layer as average of input nodes and output nodes (11+1)/2=6
#also you can use a technique called parameter tunning
#init argument: step1 of stocastic gradient descent, randomly initialize he weights
#methods: glorot_uniform, simple_uniform
#activation function for hidden layer
# we need to specify input_node for 1st hidden layer
#when we are adding more hidden layers they know the input

classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim= 11))

#adding the 2nd hidden layer

classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

#adding the outplayer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

#if you are dealing with a dependent Var.(y) that has more than 2 categories, like 3
# you have to change 2 things, output (set as number of classes = 3)
# actvation function should be changed from sigmoid into softmax ( for a dependent var.
#that has more than two categories)

#compiling ANN: applying stochastic gradient decent on the whole ANN
#Argument, optimier= algorithm you want to use to find the optimal set of weights
#optimizer: stochastic gradient descent = Adam
#loss: loss function in gradien descent algorithm which is adam
# loss of logistic regression is sum of squared errors (linear regression)
##loss of gradient descent is logarithmic loss
#binary out com: loss function = binary_crossentropy
#more that 2 out comes: loss function = categorical_crossentropy
#metrics: when the weights are updating after each observation, the algorithm uses
#accuracy criterion to improve the models performance, it increases little by little till
#it reaches top accuracy

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

#Fitting ANN into training set
#argument3 = when do you want to modify the weighs? after a batch o observation?
#argument4 : epochs
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)




##Part3 - Making the predictions and evaluating the model
# predicting the test set results 
y_pred = classifier.predict(x_test)


#we want to figure out if the accuracy obtained on the test set is as good as
#training set, we do that with confusion matrix
#Note: the predict method returns the probabilities, in order to use the confusion-
#matrix we dont need the probability, we need True or False
#we need to choose a threshold to predict if the result is 1 or 0
#threshold = 0.5

y_pred = (y_pred > 0.5)

#making confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#we get accuracy of 86%, maybe if we do tunning we get better accuracies

#predicting a single new observation
# =============================================================================
#  Use our ANN model to predict if the customer with the following informations will leave the bank: 
# 
#     Geography: France
#     Credit Score: 600
#     Gender: Male
#     Age: 40 years old
#     Tenure: 3 years
#     Balance: $60000
#     Number of Products: 2
#     Does this customer have a credit card ? Yes
#     Is this customer an Active Member: Yes
#     Estimated Salary: $50000
# 
# So should we say goodbye to that customer ?
# 
# =============================================================================

# we need to use y_pred: our inputs should be in 2-D shape array np.array([[]])
# put the information as the same order of feature vectors, and it should be scaled
#same scale as training set, we take the same sc object because iit was fitted into
#training set, we need to pply iit to the whole array


new_prediction = classifier.predict(sc_x.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction>0.5)



#part 4 - Evaluating ANN k-fold cross validation: fixes variance of accuracy that we obtained 
#in previous parts, it will fix it by splitting the training set into 10 folds(k=10)
# we train our model on 9 folds and test on last fold>> gives a better 
#evaluation of models.
# for that we need to just do the data preprocessing and then this part:
# so far we built a model with keras and k-fold validation is from scikit-learn
# we have to combine them together: perfect module is keras wrapper, that will wrap 
# k-fold into keras model, used by keros classifier


## evaluating ANN:
#keras wrapper

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#we are ready to start implementing K-fold cross validation inside keras
# to use KerasClassifier we should define a function (its an argument)
#this function build the architecture of ANN that we have built on part 2 and 3

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim= 11))
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier

#now we wrapp the  whole thing, we should build a global classifier now
# this classifier wont be trained on the whole training set, but on 
# 10 different training fold through k-fold cross validation

classifier = KerasClassifier(build_fn= build_classifier, batch_size=10 , nb_epoch=100)

#now we use k-fold cross validation
# returning the relevent measure of the accuracy of ANN on test set
# 10 accuracies of the 10 test folds that occur
#X is the training set from which we buil our 10 train test folds
#cv is the number of folds

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv=10, n_jobs=-1)


    


    
    
