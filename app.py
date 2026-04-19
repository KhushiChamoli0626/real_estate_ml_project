
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load dataset
data = pd.read_csv('data.csv')

# Train model
X = data[['area', 'bedrooms', 'bathrooms', 'location']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    location = int(request.form['location'])

    prediction = model.predict([[area, bedrooms, bathrooms, location]])

    return render_template('index.html', prediction_text=f"Predicted Price: ₹ {int(prediction[0])}")

if __name__ == "__main__":
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=8000)
    
