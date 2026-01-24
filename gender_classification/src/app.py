from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and mapping
MODEL_DIR = 'gender_classification/models'
model = joblib.load(os.path.join(MODEL_DIR, 'gender_model.pkl'))
company_mapping = joblib.load(os.path.join(MODEL_DIR, 'company_mapping.pkl'))
inverse_company_mapping = {v: k for k, v in company_mapping.items()}

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        age = data.get('age')
        company = data.get('company')
        
        if age is None or company is None:
            return jsonify({'error': 'Missing age or company'}), 400
            
        company_code = inverse_company_mapping.get(company, -1) # Default to -1 if unknown
        
        # Prepare input
        input_data = pd.DataFrame([[age, company_code]], columns=['age', 'company_cat'])
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'gender': prediction
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
