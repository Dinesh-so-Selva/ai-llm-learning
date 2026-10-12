import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
# House sizes in square feet
X = np.array([[1000], [1500], [2000], [2500]])

# Actual prices in lakhs
y = np.array([30, 45, 60, 75])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict house prices
predictions = model.predict(X)

# Display results
print("Learned Weight:", model.coef_[0])
print("Learned Bias:", model.intercept_)

print("\nActual Prices:", y)
print("Predicted Prices:", predictions)

# Predict price for a new house
new_house = np.array([[3000]])
predicted_price = model.predict(new_house)

print("\nHouse Size:", new_house[0][0], "sq. ft.")
print("Predicted Price: ₹", round(predicted_price[0], 2), "lakhs")