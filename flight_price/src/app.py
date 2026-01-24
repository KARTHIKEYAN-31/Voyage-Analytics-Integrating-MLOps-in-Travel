from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
MODEL_PATH = 'models/flight_price_model.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def preprocess_input(data):
    # Convert dict to DataFrame
    df = pd.DataFrame([data])
    
    # Date processing
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['weekday'] = df['date'].dt.weekday
        df = df.drop(columns=['date'])
    
    # Ensure all expected columns are present (the pipeline handles OHE, but we need the raw columns)
    expected_cols = ['from', 'to', 'flightType', 'agency', 'time', 'distance', 'month', 'day', 'weekday']
    # Add missing cols with default if necessary (though for prediction we usually expect full input)
    # Here we assume user provides all necessary fields.
    
    return df[expected_cols]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        
        return jsonify({
            'predicted_price': float(prediction)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
