from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__) 

# Load the trained model
model = joblib.load('customer_segkmeans_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('cseg.html')

# Define a route to handle form submissions and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    Income = request.form['Income']
    SpendingScore = request.form['SpendingScore']
    
    # Create a DataFrame from the input
    data = pd.DataFrame([[Income, SpendingScore]], columns=['Annual Income (k$)', 'Spending Score (1-100)'])
    
    # Predict the cluster
    cluster = model.predict(data)[0]
    
    return render_template('cseg.html', cluster=cluster)

if __name__ == '__main__':
    app.run(debug=True)


