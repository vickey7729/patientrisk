from flask import Flask, render_template, request, jsonify
import pickle
import os
import json
import numpy as np
from model3 import PatientRiskModel2  # Import the PatientRiskModel2 class

# Get the absolute path to the current project directory
base_dir = os.path.abspath(os.path.dirname(__file__))

# Create a Flask app with template folder explicitly defined
app = Flask(__name__, 
            template_folder=os.path.join(base_dir, 'templates'),
            static_folder=os.path.join(base_dir, 'static'))

# Print debugging information on startup
print(f"Current working directory: {os.getcwd()}")
print(f"Base directory: {base_dir}")
print(f"Templates directory: {os.path.join(base_dir, 'templates')}")

# Load the trained model - use relative path with base_dir
model_path = os.path.join(base_dir, 'models', 'risk_model3.pkl')  # Updated model path
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model successfully loaded from {model_path}")
    print(f"Model type: {type(model)}")
    if not isinstance(model, PatientRiskModel2):
        print(f"Warning: Loaded model is not PatientRiskModel2. Found: {type(model)}")
except FileNotFoundError:
    model = None
    print(f"Warning: Model file not found at {model_path}. Please run train.py first.")
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@app.route('/')
def index():
    try:
        return render_template('index2.html')
    except Exception as e:
        # Provide more detailed error for template issues
        return f"""
        <h1>Template Error</h1>
        <p>Could not render template: {str(e)}</p>
        <p>Looking for template in: {os.path.join(base_dir, 'templates')}</p>
        <p>Available templates: {os.listdir(os.path.join(base_dir, 'templates')) if os.path.exists(os.path.join(base_dir, 'templates')) else 'templates directory not found'}</p>
        """

# Helper function to convert numpy types to native Python types
def convert_to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(i) for i in obj)
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please run train.py first to train and save the model."
        }), 500
    
    try:
        # Get patient data from the form
        patient_data = {
            "demographics": {
                "age": int(request.form.get('age', 0)),
                "gender": request.form.get('gender', 'Unknown'),
                "ethnicity": request.form.get('ethnicity', 'Unknown'),
            },
            "conditions": {
                "chronic_conditions": request.form.getlist('chronic_conditions')
            },
            "vitals": {
                "bmi": float(request.form.get('bmi', 0)) if request.form.get('bmi') else 0,
                # Added blood pressure and temperature fields
                "systolic_bp": float(request.form.get('systolic_bp', 120)) if request.form.get('systolic_bp') else 120,
                "diastolic_bp": float(request.form.get('diastolic_bp', 80)) if request.form.get('diastolic_bp') else 80,
                "temperature": float(request.form.get('temperature', 98.6)) if request.form.get('temperature') else 98.6
            },
            "medications": request.form.getlist('medications'),
            "smoking_status": request.form.get('smoking_status', ''),
            "depression_score": int(request.form.get('depression_score', 0)) if request.form.get('depression_score') else 0,
            "admission_count": int(request.form.get('admission_count', 0)) if request.form.get('admission_count') else 0
        }
        
        # Add lab values if they exist
        if any(key.startswith('lab_') for key in request.form):
            patient_data['lab_values'] = {}
            for key in request.form:
                if key.startswith('lab_') and request.form.get(key):
                    lab_name = key[4:]  # Remove 'lab_' prefix
                    patient_data['lab_values'][lab_name] = float(request.form.get(key))
        
        print(f"Processing patient data: {json.dumps(patient_data, indent=2)}")
        
        # Make prediction using the model
        prediction = model.predict(patient_data)
        
        # Convert any NumPy types to native Python types
        prediction = convert_to_json_serializable(prediction)
        
        # Print prediction for debugging
        print(f"Prediction result: {json.dumps(prediction, indent=2, cls=NumpyEncoder)}")
        
        # Use custom JSON encoder
        return app.response_class(
            response=json.dumps(prediction, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Prediction error: {str(e)}")
        print(f"Traceback: {traceback_str}")
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Return information about the loaded model for diagnostic purposes"""
    if model is None:
        return jsonify({
            "status": "Model not loaded",
            "message": "Please run train.py first to train and save the model."
        })
    
    try:
        info = {
            "model_type": str(type(model).__name__),
            "feature_columns": model.feature_columns,
            "weights": {
                "random_forest": model.model_weights[0],
                "xgboost": model.model_weights[1],
                "clinical_rules": model.model_weights[2]
            },
            "high_risk_medications_count": len(model.high_risk_meds),
            "clinical_rules_count": len(model.clinical_rules)
        }
        return jsonify(convert_to_json_serializable(info))
    except Exception as e:
        return jsonify({
            "status": "Error",
            "message": f"Error getting model info: {str(e)}"
        }), 500

@app.route('/clinical_rules', methods=['GET'])
def clinical_rules():
    """Return clinical rules for reference"""
    if model is None:
        return jsonify({
            "error": "Model not loaded"
        }), 500
    
    try:
        # Convert the tuples (which are keys in the clinical_rules dict) to strings for JSON
        rules = {f"{rule[0]} + {rule[1]}": value for rule, value in model.clinical_rules.items()}
        return jsonify(convert_to_json_serializable(rules))
    except Exception as e:
        return jsonify({
            "error": f"Could not retrieve clinical rules: {str(e)}"
        }), 500

@app.route('/high_risk_meds', methods=['GET'])
def high_risk_meds():
    """Return high risk medications for reference"""
    if model is None:
        return jsonify({
            "error": "Model not loaded"
        }), 500
    
    try:
        return jsonify(convert_to_json_serializable(model.high_risk_meds))
    except Exception as e:
        return jsonify({
            "error": f"Could not retrieve high risk medications: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.join(base_dir, 'templates'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'static'), exist_ok=True)
    
    # Check if index.html exists
    template_path = os.path.join(base_dir, 'templates', 'index2.html')
    if not os.path.exists(template_path):
        print(f"Warning: index2.html not found at {template_path}")
        print(f"Please create an index2.html file in the templates directory: {os.path.join(base_dir, 'templates')}")
    
    app.run(debug=True)