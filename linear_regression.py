import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Example dataset: Predicting house price from area (sq ft)
# Area (sq ft)
X = np.array([500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]).reshape(-1, 1)
# Price ($1000s)
y = np.array([150, 200, 250, 275, 300, 325, 400, 425, 450])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Results
print("Test Data (Area):", X_test.flatten())
print("Actual Prices:", y_test)
print("Predicted Prices:", y_pred)

# Visualize results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.show()