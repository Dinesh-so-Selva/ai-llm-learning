from sklearn.linear_model import LinearRegression

# Training data
# X = Hours studied
# y = Marks obtained

X = [[1], [2], [3], [4], [5]]
y = [20, 35, 50, 65, 80]

# Create the machine learning model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict marks for a student who studies for 6 hours
hours = [[6]]
prediction = model.predict(hours)

print("Hours studied:", hours[0][0])
print("Predicted marks:", round(prediction[0], 2))