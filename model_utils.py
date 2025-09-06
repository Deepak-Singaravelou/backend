import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

def create_sample_data(csv_path):
    """Create sample hospital readmission data if file doesn't exist"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic hospital data
    ages = np.random.normal(65, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 95)  # Clip to realistic age range
    
    data = {
        'age': ages,
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'primary_diagnosis': np.random.choice(['Heart Disease', 'Diabetes', 'Pneumonia', 'Surgery', 'Other'], n_samples),
        'length_of_stay': np.random.randint(1, 15, n_samples),
        'num_medications_prescribed': np.random.randint(1, 20, n_samples),
        'procedures_count': np.random.randint(0, 5, n_samples),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
        'discharge_location': np.random.choice(['Home', 'Transfer', 'SNF', 'Home Health'], n_samples),
    }
    
    # Create readmission target with realistic logic
    readmission_prob = np.zeros(n_samples)
    
    # Add risk factors
    readmission_prob += (ages > 65) * 0.2  # Age > 65 increases risk
    readmission_prob += (data['length_of_stay'] > 7) * 0.15  # Long stay
    readmission_prob += (data['num_medications_prescribed'] > 10) * 0.15  # Many medications
    readmission_prob += (data['admission_type'] == 'Emergency') * 0.2  # Emergency admission
    readmission_prob += np.random.random(n_samples) * 0.3  # Random factor
    
    # Convert to binary (0 or 1)
    data['readmitted_30_days'] = (readmission_prob > 0.4).astype(int)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Sample data created at {csv_path} with {len(df)} records")
    print(f"Readmission rate: {df['readmitted_30_days'].mean():.2%}")
    
    return df

def train_model(csv_path):
    """Train ensemble model on hospital data"""
    
    try:
        # Check if file exists, if not create sample data
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}. Creating sample data...")
            data = create_sample_data(csv_path)
        else:
            data = pd.read_csv(csv_path)
            print(f"Loaded existing data from {csv_path}")
        
        # Check for target column - use the actual column name from your data
        target_column = None
        if 'readmitted_30_days' in data.columns:
            target_column = 'readmitted_30_days'
        elif 'readmission' in data.columns:
            target_column = 'readmission'
        else:
            print("Error: Neither 'readmitted_30_days' nor 'readmission' column found.")
            print("Available columns:", data.columns.tolist())
            raise KeyError("Target column not found in the dataset")
        
        print(f"Using target column: '{target_column}'")
        
        # Select relevant features for training
        feature_columns = [
            'age', 'gender', 'primary_diagnosis', 'length_of_stay', 
            'num_medications_prescribed', 'procedures_count', 'admission_type', 
            'discharge_location', 'num_prior_admissions', 'chronic_conditions_count',
            'icu_stay_flag', 'high_risk_medications_flag'
        ]
        
        # Filter to only include columns that exist in the data
        available_features = [col for col in feature_columns if col in data.columns]
        print(f"Available features: {available_features}")
        
        # Prepare features and target
        X = data[available_features].copy()
        y = data[target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('Unknown')
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Identify categorical and numerical columns BEFORE encoding
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"Categorical columns: {categorical_columns}")
        print(f"Numerical columns: {numerical_columns}")
        
        # Encode categorical variables
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column].astype(str))
            label_encoders[column] = le
            print(f"Encoded column '{column}': {le.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale ONLY numerical features
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        if numerical_columns:
            X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
            X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])
            print(f"Scaled numerical columns: {numerical_columns}")
        else:
            print("No numerical columns to scale")
        
        # Create ensemble model
        clf1 = LogisticRegression(max_iter=1000, random_state=42)
        clf2 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf3 = GradientBoostingClassifier(random_state=42)
        
        ensemble = VotingClassifier(estimators=[
            ('lr', clf1), 
            ('rf', clf2), 
            ('gb', clf3)
        ], voting='soft')
        
        # Train model
        print("Training ensemble model...")
        ensemble.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return ensemble, accuracy, label_encoders, scaler, available_features, numerical_columns
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        import traceback
        traceback.print_exc()
        raise

def predict_patient(model, label_encoders, scaler, feature_columns, numerical_columns, patient_data):
    """Predict readmission for a patient"""
    
    try:
        # Create DataFrame from patient data
        df = pd.DataFrame([patient_data])
        
        # Map form field names to actual column names
        field_mapping = {
            'diagnosis': 'primary_diagnosis',
            'num_medications': 'num_medications_prescribed',
            'num_procedures': 'procedures_count',
            'emergency_admission': 'admission_type',
            'discharge_disposition': 'discharge_location'
        }
        
        # Apply field mapping
        for form_field, actual_field in field_mapping.items():
            if form_field in df.columns and actual_field in feature_columns:
                df[actual_field] = df[form_field]
                df.drop(form_field, axis=1, inplace=True)
        
        # Handle emergency admission mapping
        if 'emergency_admission' in patient_data:
            df['admission_type'] = 'Emergency' if patient_data['emergency_admission'] == '1' else 'Elective'
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in df.columns:
                # Set default values based on column type
                if col in ['age', 'length_of_stay', 'num_medications_prescribed', 'procedures_count', 'num_prior_admissions', 'chronic_conditions_count']:
                    df[col] = 0
                elif col in ['icu_stay_flag', 'high_risk_medications_flag']:
                    df[col] = 0
                elif col == 'gender':
                    df[col] = 'Male'
                elif col == 'primary_diagnosis':
                    df[col] = 'Other'
                elif col == 'admission_type':
                    df[col] = 'Elective'
                elif col == 'discharge_location':
                    df[col] = 'Home'
                else:
                    df[col] = 0
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(0)
        
        # Encode categorical variables
        for column, encoder in label_encoders.items():
            if column in df.columns:
                try:
                    df[column] = encoder.transform(df[column].astype(str))
                except ValueError as e:
                    print(f"Unknown category for {column}: {df[column].iloc[0]}")
                    df[column] = 0  # Default for unknown categories
        
        # Reorder columns to match training data
        df = df[feature_columns]
        
        # Scale ONLY numerical features
        if numerical_columns:
            df[numerical_columns] = scaler.transform(df[numerical_columns])
        
        # Make prediction
        pred_proba = model.predict_proba(df)[0][1]  # Probability of readmission
        prediction = model.predict(df)[0]
        
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        result = f"{risk_level} - Readmission {'Likely' if prediction == 1 else 'Unlikely'}"
        
        print(f"Prediction: {result}, Probability: {pred_proba:.3f}")
        
        return result, round(pred_proba, 3)
        
    except Exception as e:
        print(f"Error in predict_patient: {e}")
        import traceback
        traceback.print_exc()
        return "Error in prediction", 0.0
