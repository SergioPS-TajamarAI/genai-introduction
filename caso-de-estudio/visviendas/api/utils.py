import json
import numpy as np
from tensorflow.keras.models import load_model

def predict_price(input_json):
    """
    Predicts the price of a house based on the input JSON data.

    Parameters:
        input_json (dict): A dictionary containing the features of the house.

    Returns:
        float: The predicted price of the house.
    """
    # Load the pre-trained model
    model = load_model('../final_model.keras')

    # Extract features from the input JSON
    try:
        # Exclude the "price" key as it's the target variable
        features = np.array([[value for key, value in input_json.items() if key != "price"]])
        print(features)
    except Exception as e:
        raise ValueError(f"Error processing input JSON: {e}")

    # Make a prediction
    prediction = model.predict(features)

    # Return the predicted price
    return float(prediction[0][0])