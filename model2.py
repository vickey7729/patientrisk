import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pickle
from typing import Dict, List, Tuple, Union


class PatientRiskModel2:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42
        )

        self.xgb_model = xgb.XGBRegressor(
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

        self.label_encoders = {
            'gender': LabelEncoder(),
            'ethnicity': LabelEncoder()
        }
        
        self.feature_columns = []
        self.model_weights = (0.4, 0.4, 0.2)  # Default weights for RF, XGB, rules

        # High risk conditions/combinations based on doctor inputs
        self.clinical_rules = {
            ('age>80', 'falls'): 20,
            ('heart_failure', 'copd'): 15,
            ('diabetes', 'kidney_disease'): 15,
            ('dementia', 'falls'): 15,
            ('respiratory_failure', 'pneumonia'): 15,
            ('afib', 'heart_failure'): 10,
            ('age>70', 'multiple_admissions'): 10
        }

        # High risk medications
        self.high_risk_meds = [
            'clonazePAM', 'traMADol', 'morphine',
            'oxycodone', 'insulin', 'warfarin', 'amitriptyline',
            'haloperidol', 'chlorpromazine', 'quetiapine', 'risperidone',
            'olanzapine', 'temazepam', 'triazolam', 'diazepam', 'lorazepam',
            'digoxin', 'doxepin', 'metoclopramide', 'nitrofurantoin', 'hydroxyzine'
        ]

    def preprocess_features(self, data: Dict) -> pd.DataFrame:
        """
        Preprocess patient data into a standardized feature set for prediction.
        This method is used for both training and prediction to ensure consistency.
        """
        # Start with basic features
        features = {
            'age': data.get('demographics', {}).get('age', 0),
            'gender': data.get('demographics', {}).get('gender', 'Unknown'),
            'age>70': int(data.get('demographics', {}).get('age', 0) > 70),
            'age>80': int(data.get('demographics', {}).get('age', 0) > 80),
            'num_medications': len(data.get('medications', [])),
        }
        
        # Handle gender encoding
        if hasattr(self.label_encoders['gender'], 'classes_'):
            # If the encoder is already fitted, transform the value
            if features['gender'] in self.label_encoders['gender'].classes_:
                features['gender'] = self.label_encoders['gender'].transform([features['gender']])[0]
            else:
                features['gender'] = self.label_encoders['gender'].transform(['Unknown'])[0]

        # Process conditions
        conditions = data.get('conditions', {}).get('chronic_conditions', [])
        conditions_lower = [c.lower() for c in conditions]
        features['num_chronic_conditions'] = len(conditions)
        
        # Check for specific conditions
        features['has_copd'] = int(any('copd' in c.lower() or 'respiratory failure' in c.lower() for c in conditions))
        features['has_heart_failure'] = int(any('heart failure' in c.lower() or 'chf' in c.lower() for c in conditions))
        features['has_diabetes'] = int(any('diabetes' in c.lower() or 'dm' in c.lower() for c in conditions))
        features['has_kidney_disease'] = int(any('kidney' in c.lower() for c in conditions))
        features['has_dementia'] = int(any('dementia' in c.lower() for c in conditions))
        features['has_afib'] = int(any('afib' in c.lower() or 'atrial fibrillation' in c.lower() for c in conditions))
        features['has_pneumonia'] = int(any('pneumonia' in c.lower() for c in conditions))
        features['has_falls'] = int(any('fall' in c.lower() for c in conditions))
        
        # Process medications
        medications = data.get('medications', [])
        medications_lower = [m.lower() for m in medications]
        
        for med in self.high_risk_meds:
            med_lower = med.lower()
            features[f'takes_{med_lower}'] = int(any(med_lower in m.lower() for m in medications))
            
        # Handle vitals
        vitals = data.get('vitals', {})
        features['bmi'] = vitals.get('bmi', 0)
        features['bmi>30'] = int(features['bmi'] > 30) if features['bmi'] > 0 else 0
        
        # Additional features that might be in the training set
        features['is_current_smoker'] = int(data.get('smoking_status', '').lower() == 'current')
        features['depression_score'] = data.get('depression_score', 0)
        features['high_depression'] = int(features['depression_score'] > 15) if features['depression_score'] > 0 else 0
        
        # Multiple admissions feature (if available)
        features['multiple_admissions'] = int(data.get('admission_count', 0) > 1)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all expected feature columns exist
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Only keep columns used in training
            df = df[self.feature_columns]
        
        return df

    def _prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from training DataFrame."""
        features = pd.DataFrame()

        # Fix typo - check both spellings to handle inconsistent data
        chronic_conditions_col = 'Chronic Conditions' if 'Chronic Conditions' in df.columns else 'Chronice Conditions'

        # Demographic features
        features['age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(0)
        features['age>70'] = (features['age'] > 70).astype(int)
        features['age>80'] = (features['age'] > 80).astype(int)
        
        # Encode gender
        df['Gender'] = df['Gender'].fillna('Unknown')
        self.label_encoders['gender'].fit(df['Gender'])
        features['gender'] = self.label_encoders['gender'].transform(df['Gender'])

        # Problem features
        for problem in ['Problem1', 'Problem2', 'Problem3']:
            if problem in df.columns:
                problem_data = df[problem].fillna('')
                features[f'has_heart_failure_{problem}'] = problem_data.str.contains('heart failure', case=False).fillna(False).astype(int)
                features[f'has_copd_{problem}'] = problem_data.str.contains('COPD|respiratory failure', case=False).fillna(False).astype(int)
                features[f'has_diabetes_{problem}'] = problem_data.str.contains('diabetes', case=False).fillna(False).astype(int)
                features[f'has_kidney_{problem}'] = problem_data.str.contains('kidney', case=False).fillna(False).astype(int)
                features[f'has_pneumonia_{problem}'] = problem_data.str.contains('pneumonia', case=False).fillna(False).astype(int)
                features[f'has_falls_{problem}'] = problem_data.str.contains('fall', case=False).fillna(False).astype(int)

        # Chronic conditions
        if chronic_conditions_col in df.columns:
            features['num_chronic_conditions'] = df[chronic_conditions_col].fillna('').str.count(',').fillna(0) + 1
            features['has_copd'] = df[chronic_conditions_col].fillna('').str.contains('COPD', case=False).fillna(False).astype(int)
            features['has_heart_failure'] = df[chronic_conditions_col].fillna('').str.contains('heart failure|CHF', case=False).fillna(False).astype(int)
            features['has_diabetes'] = df[chronic_conditions_col].fillna('').str.contains('diabetes|DM', case=False).fillna(False).astype(int)
            features['has_dementia'] = df[chronic_conditions_col].fillna('').str.contains('dementia', case=False).fillna(False).astype(int)
            features['has_afib'] = df[chronic_conditions_col].fillna('').str.contains('afib|atrial fibrillation', case=False).fillna(False).astype(int)
        else:
            # Add empty columns if not found
            features['num_chronic_conditions'] = 0
            features['has_copd'] = 0
            features['has_heart_failure'] = 0
            features['has_diabetes'] = 0
            features['has_dementia'] = 0
            features['has_afib'] = 0

        # Medication features
        if 'Medications' in df.columns:
            features['num_medications'] = df['Medications'].fillna('').str.count(';').fillna(0) + 1
            for med in self.high_risk_meds:
                med_lower = med.lower()
                features[f'takes_{med_lower}'] = df['Medications'].fillna('').str.contains(med, case=False).fillna(False).astype(int)
        else:
            features['num_medications'] = 0
            for med in self.high_risk_meds:
                features[f'takes_{med.lower()}'] = 0

        # Clinical measurements
        if 'BMI' in df.columns:
            features['bmi'] = pd.to_numeric(df['BMI'], errors='coerce').fillna(0)
            features['bmi>30'] = (features['bmi'] > 30).fillna(False).astype(int)
        else:
            features['bmi'] = 0
            features['bmi>30'] = 0
            
        if 'Depression screening score' in df.columns:
            features['depression_score'] = pd.to_numeric(df['Depression screening score'], errors='coerce').fillna(0)
            features['high_depression'] = (features['depression_score'] > 15).fillna(False).astype(int)
        else:
            features['depression_score'] = 0
            features['high_depression'] = 0

        # Smoking status
        if 'Smoking Status' in df.columns:
            features['is_current_smoker'] = df['Smoking Status'].fillna('').str.contains('current', case=False).fillna(False).astype(int)
        else:
            features['is_current_smoker'] = 0
            
        # Multiple admissions (if available)
        if 'Previous Admissions' in df.columns:
            features['multiple_admissions'] = (pd.to_numeric(df['Previous Admissions'], errors='coerce').fillna(0) > 1).astype(int)
        else:
            features['multiple_admissions'] = 0

        return features

    def _apply_clinical_rules(self, patient_data: Dict) -> float:
        """Apply clinical rules to calculate a risk score."""
        score = 0

        # Extract conditions and demographics
        conditions = [c.lower() for c in patient_data.get('conditions', {}).get('chronic_conditions', [])]
        demographics = patient_data.get('demographics', {})
        age = demographics.get('age', 0)
        
        # Create a dictionary of condition presence for easier checking
        condition_dict = {
            'copd': any('copd' in c or 'respiratory failure' in c for c in conditions),
            'heart_failure': any('heart failure' in c or 'chf' in c for c in conditions),
            'diabetes': any('diabetes' in c or 'dm' in c for c in conditions),
            'kidney_disease': any('kidney' in c for c in conditions),
            'dementia': any('dementia' in c for c in conditions),
            'afib': any('afib' in c or 'atrial fibrillation' in c for c in conditions),
            'pneumonia': any('pneumonia' in c for c in conditions),
            'falls': any('fall' in c for c in conditions),
            'respiratory_failure': any('respiratory failure' in c for c in conditions),
            'multiple_admissions': patient_data.get('admission_count', 0) > 1,
            'age>70': age > 70,
            'age>80': age > 80
        }

        # Check each clinical rule
        for (condition1, condition2), value in self.clinical_rules.items():
            if condition_dict.get(condition1, False) and condition_dict.get(condition2, False):
                score += value

        # Additional rules based on doctors' input
        if age > 90:
            score += 25  # Highest risk age group
        if len(conditions) > 5:
            score += 15  # Multiple conditions

        return min(score, 100)  # Cap at 100

    def _get_contributing_factors(self, patient_features: pd.DataFrame) -> List[Dict]:
        """
        Calculate contributing factors specific to this patient.
        This version ensures consistent handling of feature columns.
        """
        contributions = []
        
        # Ensure features are in the right order
        patient_features = patient_features[self.feature_columns]
        
        # Get feature importances
        rf_importances = self.rf_model.feature_importances_
        xgb_importances = self.xgb_model.feature_importances_
        
        # Combine model importances (weighted average)
        combined_importances = (0.5 * rf_importances + 0.5 * xgb_importances)
        
        # Get actual feature values
        feature_values = patient_features.iloc[0].values
        
        # For each feature, calculate if it applies to this patient
        for i, (feat, imp) in enumerate(zip(self.feature_columns, combined_importances)):
            value = feature_values[i]
            
            # Only include factors that actually apply to this patient
            # Check if the feature has a positive value (applies to the patient)
            # and has a meaningful importance (above threshold)
            if value > 0 and imp > 0.01:  # Minimum threshold for importance
                factor_desc = self._get_factor_description(feat)
                # Convert numpy types to Python native types
                contribution = float(imp * 100)  # Convert to percentage and to native Python float
                
                contributions.append({
                    "factor": factor_desc,
                    "contribution": round(contribution, 2),
                    "description": self._get_detailed_description(feat, imp)
                })
        
        # Sort by contribution and return top factors
        return sorted(contributions, key=lambda x: x['contribution'], reverse=True)[:5]  # Return top 5

    def _calculate_confidence(self, patient_data: Dict) -> float:
        """Calculate confidence score for the prediction."""
        confidence = 1.0

        # Critical fields that affect confidence
        critical_fields = [
            ('demographics', ['age', 'gender']),
            ('conditions', ['chronic_conditions']),
            ('vitals', ['bmi']),
            ('medications', [])
        ]

        # Check for missing critical data
        for section, fields in critical_fields:
            if section not in patient_data:
                confidence -= 0.2
                continue

            for field in fields:
                if field not in patient_data[section]:
                    confidence -= 0.1

        # Additional confidence modifiers
        if 'lab_values' not in patient_data:
            confidence -= 0.1

        # Consider data quality
        if patient_data.get('conditions', {}).get('chronic_conditions', []):
            conditions = patient_data['conditions']['chronic_conditions']
            if len(conditions) < 1:  # Suspiciously few conditions
                confidence -= 0.1

        return max(0.5, min(1.0, confidence))  # Keep between 0.5 and 1.0

    def _get_factor_description(self, feature_name: str) -> str:
        """Convert feature names to human-readable descriptions."""
        if feature_name.startswith('has_'):
            return feature_name.replace('has_', '').replace('_', ' ').title()
        elif feature_name.startswith('takes_'):
            return f"Medication: {feature_name.replace('takes_', '').title()}"
        elif feature_name == 'multiple_admissions':
            return "Multiple Hospital Admissions"
        elif feature_name == 'age>70':
            return "Age Over 70"
        elif feature_name == 'age>80':
            return "Age Over 80"
        elif feature_name == 'bmi>30':
            return "BMI Over 30 (Obese)"
        elif feature_name == 'high_depression':
            return "High Depression Score"
        return feature_name.replace('_', ' ').title()

    def _get_detailed_description(self, feature: str, importance: float) -> str:
        """Generate detailed description of why a feature is important."""
        descriptions = {
            'age': "Age is a critical factor in patient risk assessment",
            'age>70': "Being over 70 years old increases risk of complications",
            'age>80': "Being over 80 years old significantly increases risk",
            'has_heart_failure': "Presence of heart failure significantly increases risk",
            'has_copd': "COPD indicates increased respiratory risk",
            'has_diabetes': "Diabetes increases risk for complications",
            'has_kidney_disease': "Kidney disease affects medication processing and overall health",
            'has_dementia': "Dementia can complicate treatment adherence",
            'has_afib': "Atrial fibrillation increases risk of stroke and other complications",
            'has_pneumonia': "Pneumonia increases risk of respiratory complications",
            'has_falls': "History of falls indicates frailty and risk of injury",
            'multiple_admissions': "Multiple previous admissions suggest higher risk of readmission",
            'bmi>30': "Obesity is linked to numerous health complications",
            'high_depression': "Depression can impact treatment adherence and outcomes"
        }
        
        # For medications
        if feature.startswith('takes_'):
            med = feature.replace('takes_', '').title()
            return f"Taking {med} indicates underlying condition requiring monitoring"
            
        # Return the specific description if available, otherwise a general one
        # Convert numpy float to native Python float for JSON serialization
        return descriptions.get(feature, f"This factor contributes {round(float(importance * 100), 1)}% to the risk score")

    def train(self, df: pd.DataFrame):
        """Train the model using the provided DataFrame."""
        # Handle missing values first
        df = df.copy()
        if 'DOD' not in df.columns:
            raise ValueError("Training data must contain 'DOD' column for target variable")
            
        # Prepare target (1 if patient has DOD - was admitted)
        y = (df['DOD'].notna()).astype(int)

        # Prepare features
        X = self._prepare_training_features(df)
        self.feature_columns = X.columns.tolist()

        # Train models
        self.rf_model.fit(X, y)
        self.xgb_model.fit(X, y)
        
        print(f"Model trained with {len(X.columns)} features: {', '.join(X.columns)}")

        return self
    
    def validate(self, validation_df: pd.DataFrame) -> Dict:
        """
        Validate model performance and optimize ensemble weights
        """
        # Prepare features and target
        X_val = self._prepare_training_features(validation_df)
        y_val = (validation_df['DOD'].notna()).astype(int)
        
        # Get predictions from each model
        rf_probs = self.rf_model.predict_proba(X_val)[:, 1]
        xgb_probs = self.xgb_model.predict(X_val)
        
        # Calculate rule-based scores
        rule_scores = []
        for _, row in validation_df.iterrows():
            # Convert row to patient_data format
            patient_data = {
                'demographics': {'age': row['Age'], 'gender': row['Gender']},
                'conditions': {'chronic_conditions': []}
            }
            
            # Handle chronic conditions
            chronic_conditions_col = 'Chronic Conditions' if 'Chronic Conditions' in validation_df.columns else 'Chronice Conditions'
            if chronic_conditions_col in validation_df.columns:
                conditions = str(row.get(chronic_conditions_col, '')).split(',')
                patient_data['conditions']['chronic_conditions'] = [c.strip() for c in conditions if c.strip()]
            
            # Handle medications
            if 'Medications' in validation_df.columns:
                medications = str(row.get('Medications', '')).split(';')
                patient_data['medications'] = [m.strip() for m in medications if m.strip()]
            else:
                patient_data['medications'] = []
                
            rule_scores.append(self._apply_clinical_rules(patient_data) / 100)  # Scale to 0-1
        
        rule_scores = np.array(rule_scores)
        
        # Grid search for optimal weights
        best_auc = 0
        best_weights = (0.4, 0.4, 0.2)
        
        for w1 in np.arange(0.1, 0.7, 0.1):
            for w2 in np.arange(0.1, 0.7, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                    
                # Combine predictions
                combined_probs = w1 * rf_probs + w2 * xgb_probs + w3 * rule_scores
                
                # Calculate AUC
                auc = roc_auc_score(y_val, combined_probs)
                
                if auc > best_auc:
                    best_auc = auc
                    best_weights = (w1, w2, w3)
        
        # Update weights
        self.model_weights = best_weights
        
        # Convert numpy types to Python native types for JSON serialization
        return {
            'auc': float(best_auc),
            'weights': tuple(float(w) for w in best_weights),
            'rf_auc': float(roc_auc_score(y_val, rf_probs)),
            'xgb_auc': float(roc_auc_score(y_val, xgb_probs)),
            'rule_auc': float(roc_auc_score(y_val, rule_scores))
        }

    def predict(self, patient_data: Dict) -> Dict:
        """
        Predict patient risk using consistent preprocessing pipeline.
        """
        # First create a standardized feature set using the same method used in training
        features = self.preprocess_features(patient_data)
        
        # Make predictions with each model
        rf_prob = float(self.rf_model.predict_proba(features)[0][1] * 100)  # Convert to native Python float
        xgb_prob = float(self.xgb_model.predict(features)[0] * 100)  # Convert to native Python float
        rule_score = float(self._apply_clinical_rules(patient_data))  # Convert to native Python float
        
        # Use optimized weights if available
        w1, w2, w3 = self.model_weights
        final_score = float(w1 * rf_prob + w2 * xgb_prob + w3 * rule_score)  # Convert to native Python float
        
        # Ensure all numeric values are Python native types (not numpy types)
        return {
            "risk_score": round(final_score, 2),
            "risk_category": self._get_risk_category(final_score),
            "contributing_factors": self._get_contributing_factors(features),
            "confidence": float(self._calculate_confidence(patient_data)),  # Convert to native Python float
            "model_breakdown": {
                "random_forest_score": round(rf_prob, 2),
                "xgboost_score": round(xgb_prob, 2),
                "clinical_rules_score": round(rule_score, 2),
                "weights": {"rf": float(w1), "xgb": float(w2), "rules": float(w3)}  # Convert to native Python floats
            }
        }

    def _get_risk_category(self, score: float) -> str:
        """Determine risk category based on score."""
        if score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        return "LOW"
        
    