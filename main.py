
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv('data.csv')

print("Dataset:")
print(data)

# Step 2: Define features and target
X = data[['area', 'bedrooms', 'bathrooms', 'location']]
y = data['price']

# Step 3: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create model
model = LinearRegression()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Evaluate model
print("\nModel Evaluation:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 8: Predict new house price
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
location = int(input("Enter location (1-3): "))

new_house = np.array([[area, bedrooms, bathrooms, location]])  # Example input
predicted_price = model.predict(new_house)

print("\nPredicted Price for new house:", predicted_price[0])

# Step 9: Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()