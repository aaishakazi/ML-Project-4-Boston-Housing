import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('housing_model.pkl', 'rb'))
features = pickle.load(open('model_features.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:

        data = request.get_json(force=True)

        raw_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        
        try:
            input_data = {f: [data[f]] for f in raw_features}
        except KeyError as e:
            return jsonify({'error': f'Missing input field: {str(e)}'}), 400
        
        input_df = pd.DataFrame(input_data)
        
        # Feature engineering for the new data 
        input_df['IncPerOcc'] = input_df['MedInc'] / input_df['AveOccup']
        input_df['RoomsPerHousehold'] = input_df['AveRooms'] / input_df['AveOccup']
        input_df['BedroomsPerRoom'] = input_df['AveBedrms'] / input_df['AveRooms']

        hub_lat, hub_lon = 37.7749, -122.4194
        input_df['Distance_to_Nearest_Hub'] = np.sqrt(
            (input_df['Latitude'] - hub_lat)**2 + (input_df['Longitude'] - hub_lon)**2
        )
        
        input_df['Wealth_Location_Score'] = input_df['MedInc'] / (input_df['Distance_to_Nearest_Hub'] + 1)
        
        # Align Columns
        input_df = input_df[features]
        
        prediction = model.predict(input_df)[0]

        return jsonify({
                'status': 'success',
                'predicted_price_usd': round(float(prediction * 100000), 2),
                'currency': 'USD'
            })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400 

@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data into a dictionary
    form_data = {f: [float(request.form[f])] for f in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']}
    
    input_df = pd.DataFrame(form_data)
    
    # Apply your feature engineering (ensure these match your predict_api logic)
    input_df['IncPerOcc'] = input_df['MedInc'] / input_df['AveOccup']
    input_df['RoomsPerHousehold'] = input_df['AveRooms'] / input_df['AveOccup']
    input_df['BedroomsPerRoom'] = input_df['AveBedrms'] / input_df['AveRooms']
    
    hub_lat, hub_lon = 37.7749, -122.4194
    input_df['Distance_to_Nearest_Hub'] = np.sqrt(
        (input_df['Latitude'] - hub_lat)**2 + (input_df['Longitude'] - hub_lon)**2
    )
    input_df['Wealth_Location_Score'] = input_df['MedInc'] / (input_df['Distance_to_Nearest_Hub'] + 1)
    
    # Align with model features
    input_df = input_df[features]
    
    prediction = model.predict(input_df)[0]
    
    return render_template('home.html', 
                           prediction_text=f'Estimated Market Value: ${prediction * 100000:,.2f}')


if __name__ == '__main__':
    app.run(debug=True)