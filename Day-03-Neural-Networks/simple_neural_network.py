from sklearn.neural_network import MLPRegressor
import numpy as np

# Training data: house size in thousands of sq. ft.
X = np.array([[1.0], [1.5], [2.0], [2.5]])

# House prices in lakhs
y = np.array([30.0, 45.0, 60.0, 75.0])

# Create a neural network
model = MLPRegressor(
    hidden_layer_sizes=(5,),
    activation="relu",
    solver="lbfgs",
    max_iter=10000,
    random_state=42
)

# Train the neural network
model.fit(X, y)

# Predict the price of a 3,000 sq. ft. house
prediction = model.predict(np.array([[3.0]]))

print("Predicted price:", round(prediction[0], 2), "lakhs")