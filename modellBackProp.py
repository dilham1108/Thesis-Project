import numpy as np 
import pandas as pd

import random
import tensorflow as tf
seed = 1 #(angka dapat berupa 0-9)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

np.set_printoptions(suppress=True)
df = pd.read_excel('dataset/train.xlsx', sheet_name='Sheet1')
print(df)

from sklearn.preprocessing import LabelEncoder
Predictors = ['IR740nm', 'IR770nm', 'IR800nm', 'IR830nm', 'IR880nm']

df = pd.DataFrame(df, columns=['IR740nm', 'IR770nm', 'IR800nm', 'IR830nm', 'IR880nm', 'class'])
x = df.iloc[:, 0:5].values
y = df.iloc[:, -1].values

# print(x, y)


df.info()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('X_train  ', x_train.shape)
print('y_train  ', y_train.shape)
print('X_test   ', x_test.shape)
print('y_test   ', y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# scPredict = StandardScaler()
# scTarget = StandardScaler()

# PredSF = sc.fit(x)
# TarVarSF = sc.fit(y)

# x = PredSF.transform(x)
# y = TarVarSF.transform(y)
# scPredictFit = scPredict.fit(x)
# scTargetFit = scTarget.fit(y)

# xx = scPredictFit.transform(x)
# yy = scTargetFit.transform(y)

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# y_train = sc.fit_transform(y_train)
# y_test = sc.fit_transform(y_test)

# y_test = sc.fit_transform(y_test)
# y_train = sc.fit_transform(y)
# y_test = sc.fit_transform(y)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
import tensorflow as tf 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=5, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
# model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=800, batch_size=128)





Predictions= model.predict(x_test)
print('Predictions\n',Predictions)

# y_test = sc.fit_transform(x)

for i in range(len(Predictions)):
	print(np.argmax(Predictions[i]))
# scaling the predicted kadar/alb data back to original scale
# Predictions= x_train.inverse_transform(Predictions)
# print('Predictions\n',Predictions)

# y_test_orig=y_test.inverse_transform(y_test)
# print('y_test_orig',y_test_orig)

# # # scaling the y_test kadar/alb data to back original scale
# # x_test= x_test.inverse_transform(x_test)
# # print('x_test_orig\n',x_test_orig)

# # # scaling the test data back to original scale
# Test_Data= x_test.inverse_transform(x_test)
# print('Test_Data\n',Test_Data)

# # y_pred = model.predict(x_test)
# # print('Predictions\n', y_pred) 

# TestingData= pd.DataFrame(data=Test_Data, columns=Predictors)
# print('TestingData\n',TestingData)
# TestingData['kelas']= y_test_orig
# TestingData['Predicted kelas']= Predictions
# TestingData.head()




testing = model.evaluate(x_test, y_test, batch_size=10)

history.history.keys()
# testing.keys()

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()


plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()




######################################################
# ANN for prediction Kadar, ALB #panjang gelombang NIR

# import pandas as pd 
# import numpy as np 

# # untuk menghindari perubahan nilai hasil setiap running
# import random
# import tensorflow as tf

# seed = 1 #(angka dapat berupa 0-9)
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)


# # to remove the scientific notation from numpy arrays
# np.set_printoptions(suppress=True)

# # read the data
# Data = pd.read_excel('dataset/train.xlsx', sheet_name='Sheet1')
# Data.head()

# # separate the target and predictor variables
# # TargetVariable    = ['Kadar']
# TargetVariable = ['class']

# # Predictors = ['IR770nm', 'IR800nm', 'IR830nm', 'IR880nm']
# Predictors = ['IR740nm', 'IR800nm', 'IR880nm']

# X = Data[Predictors].values
# y = Data[TargetVariable].values

# Data.info()

# ## standarization of data ##
# from sklearn.preprocessing import StandardScaler
# PredictorsScaler  = StandardScaler()
# TargetVariScaler = StandardScaler()

# # storing the fit object for later reference
# PredictorsScalerFit	= PredictorsScaler.fit(X)
# TargetVariScalerFit = TargetVariScaler.fit(y) 

# # generating the standarized values of X and y
# X = PredictorsScalerFit.transform(X)
# y = TargetVariScalerFit.transform(y.reshape(-1))[:,0]


# # split the data into training and testing set
# from sklearn.model_selection import train_test_split
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42) # karena depan belakang, 0.2 = 42 data
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state= 42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)
# # quick sanity check with the shapes of training and testing datasets
# print('X_train  ', X_train.shape)
# print('y_train  ', y_train.shape)
# print('X_test   ', X_test.shape)
# print('y_test   ', y_test.shape)



# # importing the libraries
# from keras.models import Sequential
# from keras.layers import Dense

# # Create ANN model
# model = Sequential()
# print(model)

# # defining the input layer and first hidden layer, both are same!
# model.add(Dense(units=3, activation='relu'))

# # defining the second layer of the model
# # after the fist layer we don't have to specify input_dim as keras configure it automatically
# model.add(Dense(units=10, activation='relu'))
# model.add(Dense(units=10, activation='relu'))

# # model.add(Dense(units=5, kernel_initializer='normal', activation='tanh'))

# # the output neuron is a single fully connected node
# # since we will be predicting a single number
# model.add(Dense(units=3, activation='softmax'))

# # compiling the model
# model.compile(optimizer='rmsprop', loss='kullback_leibler_divergence', metrics=['accuracy'])

# # fitting the ANN to the training set
# # model.fit(X_train, y_train, batch_size=5, epochs=50, verbose=1)

# # a = model.fit(X_train, y_train, batch_size=2, epochs=50, verbose=1)
# # print(a)

# ####################################################################################################
# ###=================== defining a function to find the best parameters for ANN===================###
# # def FunctionFindBestParams(X_train, y_train, X_test, y_test):

# # 	# defining the list of hyper parameters to try
# # 	batch_size_list = [5,10,15,20, 25, 30, 50]
# # 	epochs_list		= [5,10,50,100, 200, 400, 1000]

# # 	import pandas as pd 
# # 	SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])

# # 	# initializing the trials
# # 	TrialNumber=0
# # 	for batch_size_trial in batch_size_list:
# # 		for epochs_trial in epochs_list:
# # 			TrialNumber+=1

# # 			# create ANN model
# # 			model = Sequential()

# # 			# defining the first layer of the model
# # 			model.add(Dense(units=5, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

# # 			# defining the second layer of the model
# # 			model.add(Dense(units=5, kernel_initializer='normal', activation='relu'))

# # 			# the output neuron is a single fully connected node
# # 			# since we will be predciting a single number
# # 			model.add(Dense(units=1, kernel_initializer='normal'))

# # 			# compiling the model
# # 			model.compile(loss='mean_squared_error', optimizer='adam')

# # 			# fitting the ANN to the training set
# # 			model.fit(X_train, y_train, batch_size = batch_size_trial, verbose=0)

# # 			MAPE = np.mean(100*(np.abs(y_test - model.predict(X_test)) / y_test))

# # 			# printing the results of the current iteration
# # 			print(TrialNumber, 'Parameters:', 'batch_size:', batch_size_trial, '-', 'epochs:', epochs_trial, '-', 'Accuracy:', 100-MAPE)

# # 			SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), 100-MAPE]],
# # 																	columns=['TrialNumber', 'Parameters', 'Accuracy']))

# # 	return(SearchResultsData)



# # ############################################
# # # calling the function
# # ResultsData = FunctionFindBestParams(X_train, y_train, X_test, y_test)



# # # %matplotlib inline
# # import matplotlib.pyplot as plt 
# # ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')

# # plt.show()



# #################################################
# # didapatkan parameter terbaik batch_size = 10, epochd=1000
# # Using the best set of parameters found, training the model again and predicting the kadar/alb on testing data.

# # fitting the ANN to the training data set
# history = model.fit(X_train, y_train, batch_size=128, epochs=100)
# print('history\n',history)

# # generating predictions on testing data
# Predictions= model.predict(X_test)
# print('Predictions\n',Predictions)

# # scaling the predicted kadar/alb data back to original scale
# Predictions= TargetVariScalerFit.inverse_transform(Predictions)
# print('Predictions\n',Predictions)

# # scaling the y_test kadar/alb data to back original scale
# y_test_orig=TargetVariScalerFit.inverse_transform(y_test)
# print('y_test_orig\n',y_test_orig)

# # scaling the test data back to original scale
# Test_Data= PredictorsScalerFit.inverse_transform(X_test)
# print('Test_Data\n',Test_Data)

# TestingData= pd.DataFrame(data=Test_Data, columns=Predictors)
# print('TestingData\n',TestingData)
# TestingData['kelas']= y_test_orig
# TestingData['Predicted kelas']= Predictions
# TestingData.head()


# # TestingData= pd.DataFrame(data=Test_Data, columns=Predictors)
# # TestingData['ALB']= y_test_orig
# # TestingData['Predicted ALB']= Predictions
# # TestingData.head()


# print(TestingData)


# ######
# """ Using the final trained model, now we are generating the prediction error 
# for each row in testing data as the Absolute Percentage Error.Taking the average 
# for all the rows is known as Mean Absolute Percentage Error(MAPE) """


# # computing the absolute percent error
# APE= 100*(abs(TestingData['kelas'] - TestingData['Predicted kelas']) / TestingData['kelas'])
# TestingData['APE']= APE 

# print('The Accuracy of ANN model is:', 100 - np.mean(APE))
# TestingData.head()

# # """ 
# # 	Saat pencarian parameter terbaik, hasil yang didapatkan akan selalu berubah, begitu juga
# # 	dengan hasil prediksi akan selalu berubah jika dilakukan running ulang program py nya, ini
# # 	terjadi karena :

# # 			Why the accuracy comes different every time I train ANN?

# # 	Even when you use the same hyperparameters, the result will be slightly 
# # 	different for each run of ANN. This happens because the initial step for 
# # 	ANN is the random initialization of weights. So every time you run the code, 
# # 	there are different values that get assigned to each neuron as weights and bias,
# # 	 hence the final outcome also differs slightly.

# # untuk mengatasi hal tsb kita harus menambahkan : (diawal program)
					
# # seed = 1 (angka dapat berupa 0-9)
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # """

# history.history.keys()

# import matplotlib.pyplot as plt

# plt.plot(history.history['loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()


# plt.plot(history.history['accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.show()