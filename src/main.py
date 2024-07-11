import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from model_utils import create_model  # Import your function from a separate module if needed

# Setting seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load data from Excel file
df = pd.read_excel('dataset/train.xlsx', sheet_name='Sheet1')

# Assuming 'class' column needs encoding if not numeric
# encoder = LabelEncoder()
# df['class'] = encoder.fit_transform(df['class'])

# Select predictors and target
Predictors = ['IR740nm', 'IR770nm', 'IR800nm', 'IR830nm', 'IR880nm']
x = df[Predictors].values
y = df['class'].values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

# Standard scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Define the learning rate scheduler function
def scheduler(epoch, lr):
    if epoch < 100:
        return lr
    else:
        return lr * tf.math.exp(-0.01)

# Create the model
input_shape = (len(Predictors),)
num_classes = 3
model = create_model(input_shape, num_classes)

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(scheduler)

# Train the model with validation split
history = model.fit(x_train, y_train, epochs=800, batch_size=128, validation_split=0.2,
                    callbacks=[early_stopping, lr_scheduler], verbose=0)

# Evaluate the model
testing = model.evaluate(x_test, y_test, batch_size=10)
print(f"Test Loss: {testing[0]} - Test Accuracy: {testing[1]}")

# Predictions
predictions = model.predict(x_test)
print('Predictions:\n', predictions)

# Plot training history
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.show()
