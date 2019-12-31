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
#to apply lableencoder object to our column

#2nd object
labelencoder_X_2 = LabelEncoder()
x[:,2] = labelencoder_X_2.fit_transform(x[:,2])
#Country has 3 categories and Gender has 2, then we have to remove one column to avoide
#dummy variable trap we prepare only for Country
#if you have m numbe of categories use m-1 in the model

x[:,1] = labelencoder_X_1.fit_transform(x[:,1])
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
from keras.models import Sequential
from keras.layers import Dense
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








#Creat our classifier right here
#fitting logistic regression to the training set



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

