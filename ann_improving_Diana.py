#part6- Improving the ANN
#overfitting is when your model was trained too much on the training set, that it
#has much less performance on the test set, we solve it by drop out regularization

#where do we need to add drop out: at each iteration of the training, some neurons
#of ANN are randomly disabled, to prevent them from being too dependent on eachother
#that prevents neurons from learnng too much, because each time there is not the 
#same configuration. >> importing dropout class in the archituctury of ANN 


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


#Part 2 - Make ANN

#dropout is applied to neurons so it means on Layers. it can be on 1 layer or several
#when you have overfitting, its better to apply dropout to all layers

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN, defining it as a sequence of layers
classifier = Sequential()

#adding input layer & 1st hidden layer with dropout
#Arguments- p: fraction of inputs you want to drop out

classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim= 11))
classifier.add(Dropout(P=0.1))

#adding the 2nd hidden layer

classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(P=0.1))

#adding the outplayer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


#compiling ANN: applying stochastic gradient decent on the whole ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

#Fitting ANN into training set
classifier.fit(x_train, y_train, batch_size=10, nb_epoch=100)

##Part3 - Making the predictions and evaluating the model
# predicting the test set results 
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

#making confusion matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
