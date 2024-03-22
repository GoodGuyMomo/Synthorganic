import numpy as np
import tensorflow as tf

class LegMovementModel(tf.keras.Model):
    def __init__(self, input_size, output_size):
        """
        Initialize the LegMovementModel.

        Parameters:
        - input_size (int): Number of input features.
        - output_size (int): Number of output actions.

        Returns:
        None
        """
        super(LegMovementModel, self).__init__()
        
        # Define the layers of the neural network
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_size)

    def call(self, inputs):
        """
        Forward pass of the neural network.

        Parameters:
        - inputs (tf.Tensor): Input data to the model.

        Returns:
        tf.Tensor: Output prediction of the model.
        """
        # Pass the input through the first dense layer with ReLU activation function
        x = self.dense1(inputs)
        
        # Pass the output of the first dense layer through the second dense layer
        return self.dense2(x)


# Define parameters
input_size = 2  # Adjust according to the number of input features
output_size = 2  # Adjust according to the number of output actions
learning_rate = 0.001
epochs = 1000

# Initialize the model
model = LegMovementModel(input_size, output_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()

# Dummy input and output data for demonstration
'''Write not this is random numbers but in theory we would use blender and take in the force of 
gravity as well as the angle of the knee joint as inputs and we would give it the output of the
force measured on the thigh and calf part of the leg.'''
input_data = np.random.rand(100, input_size) #this would be gravity and the angle of the knee joint
output_data = np.random.rand(100, output_size) #this would be the friction the calf and thigh would need to exert

# Training loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(input_data, training=True)
        loss = loss_function(output_data, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss: {loss.numpy()}")

# Example of using the trained model
example_input = np.random.rand(1, input_size)
predicted_output = model.predict(example_input)
print("Example Input:", example_input)
print("Predicted Output:", predicted_output)

# Compare predicted output with expected output
expected_output = model(example_input).numpy()
print("Expected Output:", expected_output)
