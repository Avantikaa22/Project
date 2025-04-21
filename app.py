import logging
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask app
app = Flask(__name__)

# Preprocess data (ensure you have already trained the model)
# This is a simplified version; adjust according to your full model training process
le_room_type = LabelEncoder()
le_neighbourhood_group = LabelEncoder()
le_neighbourhood = LabelEncoder()

# Example: your model and encoders should be already trained and loaded
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Dummy model training (replace with actual model training or loading process)
# model.fit(X_train, y_train)  # Use the actual trained model

# Function to predict price based on user input
def predict_price(room_type, neighbourhood_group, neighbourhood, latitude, longitude, 
                  number_of_reviews, reviews_per_month, availability_365):
    try:
        # Label encoding for categorical data
        room_type_encoded = le_room_type.transform([room_type])[0]
        neighbourhood_group_encoded = le_neighbourhood_group.transform([neighbourhood_group])[0]
        neighbourhood_encoded = le_neighbourhood.transform([neighbourhood])[0]

        # Prepare the input features for prediction
        input_features = np.array([[room_type_encoded, neighbourhood_group_encoded, neighbourhood_encoded,
                                    latitude, longitude, number_of_reviews, reviews_per_month, availability_365]])
        
        # Predict the price
        predicted_price = model.predict(input_features)
        return predicted_price[0]
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    error_message = None
    if request.method == 'POST':
        try:
            # Log the form data for debugging purposes
            logging.debug("Form data: %s", request.form)  # Log the form data
            
            # Get input values from the form
            room_type_input = request.form['room_type']
            neighbourhood_group_input = request.form['neighbourhood_group']
            neighbourhood_input = request.form['neighbourhood']
            latitude_input = float(request.form['latitude'])
            longitude_input = float(request.form['longitude'])
            number_of_reviews_input = int(request.form['number_of_reviews'])
            reviews_per_month_input = float(request.form['reviews_per_month'])
            availability_365_input = int(request.form['availability_365'])

            # Call the prediction function
            predicted_price = predict_price(room_type_input, neighbourhood_group_input, neighbourhood_input, 
                                            latitude_input, longitude_input, number_of_reviews_input, 
                                            reviews_per_month_input, availability_365_input)

        except Exception as e:
            error_message = f"Error: {e}"

    return render_template('index.html', predicted_price=predicted_price, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
