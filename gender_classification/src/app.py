from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Hardcoded robust path for this environment
MODEL_DIR = r"d:\project\woolf\Voyage\gender_classification\models"

# Load model
model_path = os.path.join(MODEL_DIR, 'gender_model_enhanced.pkl')

print(f"Loading model from: {model_path}")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    # Mock model for UI testing if load fails
    class MockModel:
        def predict(self, X): return ["Unknown"]
    model = MockModel()

def feature_engineering(df):
    # Extract name features matching train.py logic
    df['name_len'] = df['name'].apply(lambda x: len(str(x)))
    df['name_start'] = df['name'].apply(lambda x: str(x)[0].lower())
    df['name_end'] = df['name'].apply(lambda x: str(x)[-1].lower())
    df['name_last2'] = df['name'].apply(lambda x: str(x)[-2:].lower())
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        age = data.get('age')
        company = data.get('company')
        name = data.get('name')
        
        if age is None or company is None or name is None:
            return jsonify({'error': 'Missing age, company, or name'}), 400
            
        # Create initial dataframe
        input_df = pd.DataFrame([[age, company, name]], columns=['age', 'company', 'name'])
        
        # Apply feature engineering
        processed_df = feature_engineering(input_df)
        
        # Select features expected by the pipeline
        # X = df[['age', 'company', 'name_len', 'name_start', 'name_end', 'name_last2']]
        final_input = processed_df[['age', 'company', 'name_len', 'name_start', 'name_end', 'name_last2']]
        
        prediction = model.predict(final_input)[0]
        
        # Convert prediction (0/1 or string) to readable string
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
            
        return jsonify({
            'gender': str(prediction)
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    print("Registered Routes:")
    print(app.url_map)
    app.run(host='0.0.0.0', port=5001, debug=True)
