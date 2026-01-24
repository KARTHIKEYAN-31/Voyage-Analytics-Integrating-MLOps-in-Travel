from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model with absolute path resolution for robustness
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir) # Voyage/flight_price
MODEL_PATH = os.path.join(project_dir, 'models', 'flight_price_model.pkl')

if not os.path.exists(MODEL_PATH):
    # Fallback to relative if running from root
    MODEL_PATH = 'flight_price/models/flight_price_model.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
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
    
    # Ensure numeric types
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'])
    if 'distance' in df.columns:
        df['distance'] = pd.to_numeric(df['distance'])
    
    # Ensure all expected columns are present
    expected_cols = ['from', 'to', 'flightType', 'agency', 'time', 'distance', 'month', 'day', 'weekday']
    
    # Reorder columns to match training
    return df[expected_cols]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        print(f"Received prediction request: {data}")
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        
        return jsonify({
            'predicted_price': float(prediction)
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Registered Routes:")
    print(app.url_map)
    app.run(port=5000, debug=True)
