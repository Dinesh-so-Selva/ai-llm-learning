import numpy as np

# Training data
# House sizes in thousands of square feet
X = np.array([1.0, 1.5, 2.0, 2.5])

# House prices in lakhs
y = np.array([30.0, 45.0, 60.0, 75.0])

# Initial parameters
weight = 10.0
bias = 0.0

# Training settings
learning_rate = 0.01
epochs = 10000

# Train the model
for epoch in range(epochs):

    # Forward propagation: make predictions
    predictions = weight * X + bias

    # Calculate prediction errors
    errors = predictions - y

    # Calculate gradients
    dw = (2 / len(X)) * np.sum(errors * X)
    db = (2 / len(X)) * np.sum(errors)

    # Update weight and bias
    weight = weight - learning_rate * dw
    bias = bias - learning_rate * db

# Display results
print("Learned weight:", round(weight, 4))
print("Learned bias:", round(bias, 4))

print("\nActual prices:", y)
print("Predicted prices:", np.round(weight * X + bias, 2))

# Predict for a new house
new_house = 3.0  # 3,000 square feet
predicted_price = weight * new_house + bias

print("\nHouse size: 3000 sq. ft.")
print("Predicted price:", round(predicted_price, 2), "lakhs")