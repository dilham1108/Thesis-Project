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

df.info()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print('X_train  ', x_train.shape)
print('y_train  ', y_train.shape)
print('X_test   ', x_test.shape)
print('y_test   ', y_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

import tensorflow as tf 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=5, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
model.add(tf.keras.layers.Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=800, batch_size=128)

Predictions= model.predict(x_test)
print('Predictions\n',Predictions)


for i in range(len(Predictions)):
	print(np.argmax(Predictions[i]))

testing = model.evaluate(x_test, y_test, batch_size=10)

history.history.keys()

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
