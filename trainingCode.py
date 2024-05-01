import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define a neural network model class inheriting from tf.keras.Model
class LegMovementModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        super(LegMovementModel, self).__init__()
        # Define the layers of the neural network
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')  # Dense layer with 64 neurons and ReLU activation
        self.dense2 = tf.keras.layers.Dense(output_size)  # Output layer

    def call(self, inputs):
        # Forward pass of the neural network
        x = self.dense1(inputs)  # Pass input through the first dense layer
        return self.dense2(x)  # Output prediction of the model


# Load data from CSV files
hip_data = pd.read_csv("C://Users//harki//OneDrive//Documents//SP24//CS356//Project//HipJointData.csv")  # Load HipJointData.csv
knee_data = pd.read_csv("C://Users//harki//OneDrive//Documents//SP24//CS356//Project//KneeJointData.csv")  # Load KneeJointData.csv

# Clean data: Select rows with vertex value of 1
hip_data = hip_data[hip_data['Vertex'] == 1]
knee_data = knee_data[knee_data['Vertex'] == 1]

# Ensure both files have the same number of rows
min_length = min(len(hip_data), len(knee_data))
hip_data = hip_data[:min_length]
knee_data = knee_data[:min_length]

# Prepare input and output data
input_data = hip_data[['X', 'Y', 'Z']].values  # Extract 'x', 'y', 'z' coordinates from HipJointData.csv
output_data = knee_data[['X', 'Y', 'Z']].values  # Extract 'x', 'y', 'z' coordinates from KneeJointData.csv

# Split data into training and testing sets (75% training, 25% testing)
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.25, random_state=42)

# Define parameters
input_size = input_data.shape[1]  # Number of input features (3 for x, y, z coordinates)
output_size = output_data.shape[1]  # Number of output actions (3 for x, y, z coordinates of knee joint)
learning_rate = 0.001  # Learning rate for optimizer
epochs = 100 # Number of training epochs

# Initialize the neural network model, optimizer, and loss function
model = LegMovementModel(input_size, output_size)  # Initialize the model
optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam optimizer for training
loss_function = tf.keras.losses.MeanSquaredError()  # Mean Squared Error loss function

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(input_train, training=True)  # Forward pass: generate predictions
        loss = loss_function(output_train, predictions)  # Calculate loss
    
    gradients = tape.gradient(loss, model.trainable_variables)  # Compute gradients
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Update model weights
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss: {loss.numpy()}")  # Print loss every 100 epochs

#===========================================================================================================
# Testing the model
predictions_test = model(input_test, training=False)  # Forward pass: generate predictions on testing data

# Evaluate the model
test_loss = loss_function(output_test, predictions_test)  # Calculate loss on testing data
print(f"Testing Loss: {test_loss.numpy()}")

# Optionally, you can analyze individual predictions as well
# For example, you can print the predicted and actual values for the first sample in the testing data
print("Sample Prediction:")
print("Predicted Values:", predictions_test[0].numpy())
print("Actual Values:", output_test[0])
