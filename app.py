import pickle
from flask import Flask, request, jsonify, app, url_for, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

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
        
        # Aligning Columns
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

    try:
    # Capture form data into a dictionary
        form_data = {f: [float(request.form[f])] for f in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']}
        
        input_df = pd.DataFrame(form_data)
        
        input_df['IncPerOcc'] = input_df['MedInc'] / input_df['AveOccup']
        input_df['RoomsPerHousehold'] = input_df['AveRooms'] / input_df['AveOccup']
        input_df['BedroomsPerRoom'] = input_df['AveBedrms'] / input_df['AveRooms']
        
        hub_lat, hub_lon = 37.7749, -122.4194
        input_df['Distance_to_Nearest_Hub'] = np.sqrt(
            (input_df['Latitude'] - hub_lat)**2 + (input_df['Longitude'] - hub_lon)**2
        )
        input_df['Wealth_Location_Score'] = input_df['MedInc'] / (input_df['Distance_to_Nearest_Hub'] + 1)
        
        # Aligning with model features
        input_df = input_df[features]
        prediction = model.predict(input_df)[0]
        predicted_val = prediction * 100000

        # GRAPH 
        plt.figure(figsize=(8, 4))
        # We'll use a representative sample of house prices (approximate distribution)
        # In a real app, you'd load a small array of actual prices from your training set
        sample_prices = np.random.normal(200000, 100000, 1000) 
        
        sns.kdeplot(sample_prices, fill=True, color="#2c3e50", label='Market Distribution')
        plt.axvline(predicted_val, color='red', linestyle='--', label=f'Your Estimate: ${predicted_val:,.0f}')
        
        plt.title('Property Value Relative to Market', fontsize=12, pad=15)
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.legend()
        
        #Saving plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close() # Closing the plot to free memory

        return render_template('home.html', 
                               prediction_text=f'Estimated Value: ${predicted_val:,.2f}',
                               plot_url=plot_url)
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)