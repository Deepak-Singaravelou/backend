import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, 
                            AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys
import warnings
import threading
import time
import joblib
from datetime import datetime, timedelta
import schedule
from pathlib import Path
import glob
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Dynamic path detection
def get_dynamic_paths():
    """Dynamically detect and set up all necessary paths from OS"""
    paths = {}
    
    # Get current working directory
    current_dir = os.getcwd()
    paths['current_dir'] = current_dir
    
    # Get user home directory
    home_dir = os.path.expanduser("~")
    paths['home_dir'] = home_dir
    
    # Get script directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths['script_dir'] = script_dir
    
    # Common potential locations for Firebase credentials
    firebase_search_paths = [
        # Current directory
        os.path.join(current_dir, "*.json"),
        os.path.join(current_dir, "credentials", "*.json"),
        os.path.join(current_dir, "config", "*.json"),
        os.path.join(current_dir, "firebase", "*.json"),
        
        # Script directory
        os.path.join(script_dir, "*.json"),
        os.path.join(script_dir, "credentials", "*.json"),
        os.path.join(script_dir, "config", "*.json"),
        os.path.join(script_dir, "firebase", "*.json"),
        
        # Home directory
        os.path.join(home_dir, ".firebase", "*.json"),
        os.path.join(home_dir, ".credentials", "*.json"),
        os.path.join(home_dir, "Documents", "firebase", "*.json"),
        os.path.join(home_dir, "Desktop", "firebase", "*.json"),
        
        # System-wide locations (Unix/Linux)
        "/etc/firebase/*.json",
        "/opt/firebase/*.json",
        "/usr/local/etc/firebase/*.json",
        
        # Common project directories
        os.path.join(current_dir, "..", "credentials", "*.json"),
        os.path.join(current_dir, "..", "config", "*.json"),
    ]
    
    # Search for Firebase credential files
    firebase_credentials = []
    for search_path in firebase_search_paths:
        try:
            found_files = glob.glob(search_path)
            firebase_credentials.extend(found_files)
        except:
            continue
    
    # Filter for likely Firebase service account files
    likely_firebase_files = []
    for file_path in firebase_credentials:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path).lower()
            # Check if filename suggests it's a Firebase service account key
            if any(keyword in filename for keyword in [
                'firebase', 'service-account', 'serviceaccount', 
                'credentials', 'key', 'admin', 'sdk'
            ]):
                likely_firebase_files.append(file_path)
    
    paths['firebase_credentials'] = likely_firebase_files
    
    # Model storage paths
    model_storage_paths = [
        os.path.join(current_dir, "models"),
        os.path.join(script_dir, "models"),
        os.path.join(home_dir, ".hospital_ai", "models"),
        current_dir,  # fallback to current directory
        script_dir    # fallback to script directory
    ]
    
    # Create model directory if it doesn't exist
    for model_path in model_storage_paths[:3]:  # Try the first 3 preferred locations
        try:
            os.makedirs(model_path, exist_ok=True)
            if os.access(model_path, os.W_OK):  # Check if writable
                paths['model_storage'] = model_path
                break
        except:
            continue
    
    # If none of the preferred locations work, use current directory
    if 'model_storage' not in paths:
        paths['model_storage'] = current_dir
    
    # Data storage paths
    data_storage_paths = [
        os.path.join(current_dir, "data"),
        os.path.join(script_dir, "data"), 
        os.path.join(home_dir, ".hospital_ai", "data"),
        current_dir,  # fallback
        script_dir    # fallback
    ]
    
    for data_path in data_storage_paths[:3]:
        try:
            os.makedirs(data_path, exist_ok=True)
            if os.access(data_path, os.W_OK):
                paths['data_storage'] = data_path
                break
        except:
            continue
    
    if 'data_storage' not in paths:
        paths['data_storage'] = current_dir
    
    # Log storage paths
    log_storage_paths = [
        os.path.join(current_dir, "logs"),
        os.path.join(script_dir, "logs"),
        os.path.join(home_dir, ".hospital_ai", "logs"),
        current_dir,  # fallback
        script_dir    # fallback
    ]
    
    for log_path in log_storage_paths[:3]:
        try:
            os.makedirs(log_path, exist_ok=True)
            if os.access(log_path, os.W_OK):
                paths['log_storage'] = log_path
                break
        except:
            continue
    
    if 'log_storage' not in paths:
        paths['log_storage'] = current_dir
    
    # Environment-specific paths
    if sys.platform.startswith('win'):
        # Windows-specific paths
        paths['temp_dir'] = os.environ.get('TEMP', os.path.join(home_dir, 'AppData', 'Local', 'Temp'))
        paths['app_data'] = os.environ.get('APPDATA', os.path.join(home_dir, 'AppData', 'Roaming'))
    else:
        # Unix/Linux/Mac paths
        paths['temp_dir'] = os.environ.get('TMPDIR', '/tmp')
        paths['app_data'] = os.path.join(home_dir, '.local', 'share')
    
    # Python environment paths
    paths['python_executable'] = sys.executable
    paths['python_path'] = sys.path
    
    return paths

# Initialize dynamic paths
DYNAMIC_PATHS = get_dynamic_paths()

# Global variables
model = None
label_encoders = {}
scaler = None
imputer = None
feature_selector = None
feature_columns = []
numerical_columns = []
original_feature_names = []
model_accuracy = 0
last_training_time = None
db = None
training_lock = threading.Lock()

def print_detected_paths():
    """Print all detected paths for debugging"""
    print("=== DETECTED SYSTEM PATHS ===")
    print(f"Current Directory: {DYNAMIC_PATHS['current_dir']}")
    print(f"Script Directory: {DYNAMIC_PATHS['script_dir']}")
    print(f"Home Directory: {DYNAMIC_PATHS['home_dir']}")
    print(f"Model Storage: {DYNAMIC_PATHS['model_storage']}")
    print(f"Data Storage: {DYNAMIC_PATHS['data_storage']}")
    print(f"Log Storage: {DYNAMIC_PATHS['log_storage']}")
    print(f"Temp Directory: {DYNAMIC_PATHS['temp_dir']}")
    print(f"App Data: {DYNAMIC_PATHS['app_data']}")
    print(f"Python Executable: {DYNAMIC_PATHS['python_executable']}")
    
    if DYNAMIC_PATHS['firebase_credentials']:
        print(f"Found Firebase Credentials:")
        for i, cred_path in enumerate(DYNAMIC_PATHS['firebase_credentials'], 1):
            print(f"  {i}. {cred_path}")
    else:
        print("No Firebase credential files found in common locations")
    
    print("=" * 50)

# Firebase initialization with dynamic path detection
def initialize_firebase():
    """Initialize Firebase connection using detected paths"""
    global db
    try:
        if not firebase_admin._apps:
            firebase_cred_path = None
            
            # Try to find Firebase credentials
            if DYNAMIC_PATHS['firebase_credentials']:
                # Use the first found credential file
                firebase_cred_path = DYNAMIC_PATHS['firebase_credentials'][0]
                print(f"Using Firebase credentials: {firebase_cred_path}")
            else:
                # Check environment variables
                if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                    firebase_cred_path = os.environ['GOOGLE_APPLICATION_CREDENTIALS']
                    print(f"Using Firebase credentials from environment: {firebase_cred_path}")
            
            if firebase_cred_path and os.path.exists(firebase_cred_path):
                cred = credentials.Certificate(firebase_cred_path)
                firebase_admin.initialize_app(cred)
                print(f"Firebase initialized successfully with: {firebase_cred_path}")
            else:
                print("No Firebase credentials found, using synthetic data fallback")
                return False
        
        db = firestore.client()
        print("Firebase Firestore client initialized successfully")
        return True
    except Exception as e:
        print(f"Firebase initialization failed: {e}")
        print("Using synthetic data fallback")
        return False

def fetch_firebase_data():
    """Fetch patient data from Firebase Firestore"""
    try:
        if db is None:
            print("Firebase not initialized, using synthetic data")
            return create_enhanced_training_data()
        
        print("Fetching data from Firebase...")
        
        # Fetch data from all hospitals
        hospitals_ref = db.collection('Hospital')
        hospitals = hospitals_ref.stream()
        
        all_patient_data = []
        
        for hospital in hospitals:
            hospital_id = hospital.id
            print(f"Fetching data from hospital: {hospital_id}")
            
            # Get csv_data subcollection
            csv_data_ref = hospitals_ref.document(hospital_id).collection('csv_data')
            patients = csv_data_ref.stream()
            
            for patient in patients:
                patient_data = patient.to_dict()
                patient_data['hospital_id'] = hospital_id
                all_patient_data.append(patient_data)
        
        if len(all_patient_data) == 0:
            print("No data found in Firebase, using synthetic data")
            return create_enhanced_training_data()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_patient_data)
        print(f"Fetched {len(df)} records from Firebase")
        
        # Clean and prepare the data
        df = clean_firebase_data(df)
        
        # If we don't have enough data, supplement with synthetic data
        if len(df) < 100:
            print("Insufficient Firebase data, supplementing with synthetic data")
            synthetic_df = create_enhanced_training_data()
            df = pd.concat([df, synthetic_df], ignore_index=True)
        
        return df
        
    except Exception as e:
        print(f"Error fetching Firebase data: {e}")
        print("Falling back to synthetic data")
        return create_enhanced_training_data()

def clean_firebase_data(df):
    """Clean and standardize Firebase data with robust NaN handling"""
    try:
        print(f"Original data shape: {df.shape}")
        print(f"Columns with NaN values: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")
        
        # Map common field variations
        field_mapping = {
            'readmission_risk_score': 'readmitted_30_days',
            'readmission_probability': 'readmitted_30_days',
            'riskLevel': 'readmitted_30_days',
            'risk_level': 'readmitted_30_days',
            'name': 'patient_name',
            'diagnosis': 'primary_diagnosis'
        }
        
        # Apply field mapping
        for old_field, new_field in field_mapping.items():
            if old_field in df.columns and new_field not in df.columns:
                df[new_field] = df[old_field]
        
        # Handle readmission target with robust conversion
        if 'readmitted_30_days' not in df.columns:
            if 'readmission_probability' in df.columns:
                # Convert probability to binary, handle NaN
                prob_series = pd.to_numeric(df['readmission_probability'], errors='coerce')
                df['readmitted_30_days'] = (prob_series > 0.5).astype(int)
            elif 'risk_level' in df.columns:
                # Map risk levels to binary outcome
                risk_mapping = {
                    'Low Risk': 0, 'Low': 0, 'low': 0,
                    'Medium Risk': 0, 'Medium': 0, 'medium': 0,
                    'High Risk': 1, 'High': 1, 'high': 1,
                    'Very High Risk': 1, 'very high': 1,
                    'Critical Risk': 1, 'critical': 1
                }
                df['readmitted_30_days'] = df['risk_level'].map(risk_mapping).fillna(0).astype(int)
            else:
                # Create synthetic target based on available features
                df['readmitted_30_days'] = create_synthetic_target(df)
        else:
            # Clean existing readmitted_30_days column
            df['readmitted_30_days'] = pd.to_numeric(df['readmitted_30_days'], errors='coerce').fillna(0).astype(int)
        
        # Ensure required columns exist with proper handling
        required_columns = {
            'age': 65,
            'gender': 'Male',
            'primary_diagnosis': 'Other',
            'length_of_stay': 3,
            'num_medications_prescribed': 5,
            'procedures_count': 1,
            'admission_type': 'Elective',
            'discharge_location': 'Home'
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
            else:
                # Handle NaN values in existing columns
                if col in ['age', 'length_of_stay', 'num_medications_prescribed', 'procedures_count']:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)
                else:
                    df[col] = df[col].fillna(default_val)
        
        # Clean numerical columns
        numerical_cols = ['age', 'length_of_stay', 'num_medications_prescribed', 'procedures_count']
        for col in numerical_cols:
            if col in df.columns:
                # Convert to numeric and handle invalid values
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else required_columns.get(col, 0))
                
                # Apply reasonable bounds
                if col == 'age':
                    df[col] = df[col].clip(0, 120)
                elif col == 'length_of_stay':
                    df[col] = df[col].clip(0, 365)
                elif col == 'num_medications_prescribed':
                    df[col] = df[col].clip(0, 50)
                elif col == 'procedures_count':
                    df[col] = df[col].clip(0, 20)
        
        # Clean categorical columns
        categorical_cols = ['gender', 'primary_diagnosis', 'admission_type', 'discharge_location']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
        
        # Handle disease-specific parameters
        disease_params = ['chest_pain_type', 'resting_bp', 'cholesterol', 'max_heart_rate', 
                         'exercise_angina', 'st_depression', 'blood_glucose', 'hba1c', 'bmi']
        
        for param in disease_params:
            if param in df.columns:
                if param in ['chest_pain_type', 'exercise_angina']:
                    df[param] = df[param].fillna('Unknown').astype(str)
                else:
                    df[param] = pd.to_numeric(df[param], errors='coerce').fillna(0)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove invalid records
        df = df[(df['age'] >= 0) & (df['age'] <= 120)]
        df = df[df['length_of_stay'] >= 0]
        
        # Final check for any remaining NaN values
        print(f"Remaining NaN values after cleaning: {df.isnull().sum().sum()}")
        
        if df.isnull().sum().sum() > 0:
            print("Warning: Some NaN values remain, will be handled by imputer")
            print(f"Columns with remaining NaN: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")
        
        print(f"Cleaned data shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error cleaning Firebase data: {e}")
        import traceback
        traceback.print_exc()
        return df

def create_synthetic_target(df):
    """Create synthetic readmission target based on available features"""
    risk_score = np.zeros(len(df))
    
    try:
        if 'age' in df.columns:
            age_series = pd.to_numeric(df['age'], errors='coerce').fillna(65)
            risk_score += (age_series > 65) * 0.3
        
        if 'length_of_stay' in df.columns:
            los_series = pd.to_numeric(df['length_of_stay'], errors='coerce').fillna(3)
            risk_score += (los_series > 7) * 0.3
        
        if 'num_medications_prescribed' in df.columns:
            med_series = pd.to_numeric(df['num_medications_prescribed'], errors='coerce').fillna(5)
            risk_score += (med_series > 10) * 0.2
        
        if 'admission_type' in df.columns:
            risk_score += (df['admission_type'].fillna('Elective') == 'Emergency') * 0.2
        
        # Add random noise
        risk_score += np.random.uniform(0, 0.3, len(df))
        
        return (risk_score > 0.5).astype(int)
    except Exception as e:
        print(f"Error creating synthetic target: {e}")
        return np.zeros(len(df), dtype=int)

def create_enhanced_training_data():
    """Create high-quality synthetic training data as fallback"""
    np.random.seed(42)
    n_samples = 2000
    
    # Generate realistic patient data
    ages = np.random.gamma(2, 30, n_samples).astype(int)
    ages = np.clip(ages, 18, 95)
    
    data = {
        'age': ages,
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'primary_diagnosis': np.random.choice([
            'Heart Disease', 'Diabetes', 'Pneumonia', 'Surgery', 'Stroke', 
            'Kidney Disease', 'COPD', 'Cancer', 'Orthopedic', 'Mental Health'
        ], n_samples, p=[0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05])
    }
    
    # Generate correlated features with strong predictive signals
    risk_scores = np.zeros(n_samples)
    
    # Age risk
    risk_scores += np.where(ages > 75, 0.4, np.where(ages > 65, 0.25, 0.1))
    
    # Length of stay
    los_base = np.where(ages > 70, 6, 4)
    los_noise = np.random.exponential(2, n_samples)
    length_of_stay = np.clip(los_base + los_noise, 1, 30).astype(int)
    risk_scores += np.where(length_of_stay > 10, 0.3, np.where(length_of_stay > 5, 0.15, 0))
    
    # Medications
    med_base = np.where(ages > 65, 8, 5)
    medications = np.clip(med_base + np.random.poisson(3, n_samples), 1, 25)
    risk_scores += np.where(medications > 15, 0.25, np.where(medications > 8, 0.1, 0))
    
    # Procedures
    procedures = np.random.poisson(1.5, n_samples)
    procedures = np.clip(procedures, 0, 8)
    risk_scores += np.where(procedures > 3, 0.2, 0)
    
    # Admission type
    admission_types = []
    for i in range(n_samples):
        if risk_scores[i] > 0.4:
            admission_types.append(np.random.choice(['Emergency', 'Elective', 'Urgent'], p=[0.6, 0.3, 0.1]))
        else:
            admission_types.append(np.random.choice(['Emergency', 'Elective', 'Urgent'], p=[0.3, 0.5, 0.2]))
    
    risk_scores += np.where(np.array(admission_types) == 'Emergency', 0.25, 0)
    
    # Discharge location
    discharge_locations = []
    for i in range(n_samples):
        if risk_scores[i] > 0.5:
            discharge_locations.append(np.random.choice(['Home', 'Home Health', 'SNF', 'Transfer'], p=[0.3, 0.25, 0.25, 0.2]))
        else:
            discharge_locations.append(np.random.choice(['Home', 'Home Health', 'SNF', 'Transfer'], p=[0.7, 0.15, 0.1, 0.05]))
    
    risk_scores += np.where(np.array(discharge_locations) != 'Home', 0.15, 0)
    
    # Add all features
    data.update({
        'length_of_stay': length_of_stay,
        'num_medications_prescribed': medications,
        'procedures_count': procedures,
        'admission_type': admission_types,
        'discharge_location': discharge_locations
    })
    
    # Add disease-specific parameters
    disease_params = ['chest_pain_type', 'resting_bp', 'cholesterol', 'max_heart_rate', 
                     'exercise_angina', 'st_depression', 'blood_glucose', 'hba1c', 'bmi']
    
    for param in disease_params:
        if param in ['chest_pain_type', 'exercise_angina']:
            data[param] = ['Unknown'] * n_samples
        else:
            data[param] = [0] * n_samples
    
    # Create target
    noise = np.random.uniform(-0.1, 0.1, n_samples)
    final_risk = risk_scores + noise
    data['readmitted_30_days'] = (final_risk > 0.55).astype(int)
    
    df = pd.DataFrame(data)
    print(f"Created synthetic data: {len(df)} records, readmission rate: {df['readmitted_30_days'].mean():.2%}")
    
    return df

def create_advanced_ensemble():
    """Create sophisticated ensemble with multiple algorithms"""
    
    classifiers = {
        'logistic': LogisticRegression(
            max_iter=2000, 
            random_state=42, 
            C=0.1, 
            class_weight='balanced',
            solver='liblinear'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'gradient_boost': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            subsample=0.8
        ),
        'extra_trees': ExtraTreesClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    }
    
    ensemble = VotingClassifier(
        estimators=list(classifiers.items()),
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble

def train_ensemble_model(use_firebase=True):
    """Train advanced ensemble model with comprehensive data cleaning"""
    global model, label_encoders, scaler, imputer, feature_selector, feature_columns
    global numerical_columns, original_feature_names, model_accuracy, last_training_time
    
    with training_lock:
        print("Starting model training...")
        
        # Fetch data
        if use_firebase:
            df = fetch_firebase_data()
        else:
            df = create_enhanced_training_data()
        
        # Prepare features and target
        target_col = 'readmitted_30_days'
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available columns: {list(df.columns)}")
            return None, 0
        
        # Remove non-feature columns
        columns_to_drop = ['patient_name', 'patient_id', 'hospital_id', 'created_at', 
                          'last_updated', 'admission_date', target_col]
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        y = df[target_col]
        
        original_feature_names = X.columns.tolist()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Features: {original_feature_names[:10]}...")  # Show first 10 features
        
        # Handle categorical encoding BEFORE imputation
        label_encoders = {}
        categorical_columns = X.select_dtypes(include=['object']).columns
        
        print(f"Categorical columns: {list(categorical_columns)}")
        
        for col in categorical_columns:
            le = LabelEncoder()
            # Fill NaN with 'Unknown' before encoding
            X[col] = X[col].fillna('Unknown').astype(str)
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
        
        # Identify numerical columns AFTER encoding
        numerical_columns = [col for col in X.columns if col not in label_encoders]
        print(f"Numerical columns: {numerical_columns[:10]}...")  # Show first 10
        
        # Handle any remaining NaN values with imputation
        print(f"NaN values before imputation: {X.isnull().sum().sum()}")
        
        if X.isnull().sum().sum() > 0:
            # Separate imputation for numerical and categorical
            if len(numerical_columns) > 0:
                num_imputer = SimpleImputer(strategy='median')
                X[numerical_columns] = num_imputer.fit_transform(X[numerical_columns])
            
            # Any remaining categorical NaN (shouldn't happen but just in case)
            cat_cols = [col for col in X.columns if col in label_encoders]
            if len(cat_cols) > 0:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
        
        print(f"NaN values after imputation: {X.isnull().sum().sum()}")
        
        # Verify no NaN values remain
        if X.isnull().sum().sum() > 0:
            print("ERROR: NaN values still present after imputation!")
            print(f"Columns with NaN: {X.isnull().sum()[X.isnull().sum() > 0]}")
            # Force fill any remaining NaN
            X = X.fillna(0)
        
        # Feature selection (now should work without NaN errors)
        try:
            feature_selector = SelectKBest(score_func=f_classif, k='all')
            X_selected = feature_selector.fit_transform(X, y)
            print("Feature selection completed successfully")
        except Exception as e:
            print(f"Feature selection failed: {e}")
            print("Proceeding without feature selection")
            feature_selector = None
        
        # Scaling for numerical features
        if len(numerical_columns) > 0:
            scaler = RobustScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
            print(f"Scaled {len(numerical_columns)} numerical features")
        
        feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train ensemble
        ensemble = create_advanced_ensemble()
        
        print("Training advanced ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        
        model_accuracy = accuracy
        last_training_time = datetime.now()
        
        print(f"Model training completed!")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"AUC Score: {auc_score:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Training completed at: {last_training_time}")
        
        # Save model
        save_model(ensemble)
        
        model = ensemble
        return ensemble, accuracy

def save_model(model_obj):
    """Save trained model and preprocessors using dynamic paths"""
    try:
        model_file_path = os.path.join(DYNAMIC_PATHS['model_storage'], 'hospital_readmission_model.pkl')
        
        model_data = {
            'model': model_obj,
            'label_encoders': label_encoders,
            'scaler': scaler,
            'imputer': imputer,
            'feature_selector': feature_selector,
            'feature_columns': feature_columns,
            'numerical_columns': numerical_columns,
            'original_feature_names': original_feature_names,
            'accuracy': model_accuracy,
            'training_time': last_training_time,
            'dynamic_paths': DYNAMIC_PATHS
        }
        
        joblib.dump(model_data, model_file_path)
        print(f"Model saved successfully to: {model_file_path}")
        
        # Also save backup in script directory
        backup_path = os.path.join(DYNAMIC_PATHS['script_dir'], 'hospital_readmission_model_backup.pkl')
        try:
            joblib.dump(model_data, backup_path)
            print(f"Model backup saved to: {backup_path}")
        except Exception as backup_error:
            print(f"Warning: Could not save backup model: {backup_error}")
            
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model():
    """Load saved model and preprocessors using dynamic paths"""
    global model, label_encoders, scaler, imputer, feature_selector, feature_columns
    global numerical_columns, original_feature_names, model_accuracy, last_training_time
    
    try:
        # Try primary model location
        model_file_path = os.path.join(DYNAMIC_PATHS['model_storage'], 'hospital_readmission_model.pkl')
        
        if not os.path.exists(model_file_path):
            # Try backup location
            model_file_path = os.path.join(DYNAMIC_PATHS['script_dir'], 'hospital_readmission_model_backup.pkl')
        
        if not os.path.exists(model_file_path):
            # Try current directory as last resort
            model_file_path = 'hospital_readmission_model.pkl'
        
        if os.path.exists(model_file_path):
            model_data = joblib.load(model_file_path)
            
            model = model_data['model']
            label_encoders = model_data['label_encoders']
            scaler = model_data['scaler']
            imputer = model_data.get('imputer')
            feature_selector = model_data.get('feature_selector')
            feature_columns = model_data['feature_columns']
            numerical_columns = model_data['numerical_columns']
            original_feature_names = model_data['original_feature_names']
            model_accuracy = model_data['accuracy']
            last_training_time = model_data.get('training_time')
            
            print(f"Model loaded successfully from: {model_file_path}")
            print(f"Model Accuracy: {model_accuracy:.2%}")
            return True
    except Exception as e:
        print(f"Error loading model: {e}")
    
    return False

def save_logs(message, log_type="INFO"):
    """Save logs to dynamic log directory"""
    try:
        log_file_path = os.path.join(DYNAMIC_PATHS['log_storage'], 'hospital_ai.log')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{log_type}] {message}\n"
        
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not save log: {e}")

def retrain_model_periodically():
    """Retrain model with new Firebase data"""
    print("Checking for model retraining...")
    save_logs("Checking for model retraining", "INFO")
    
    if db is None:
        print("Firebase not available, skipping retraining")
        save_logs("Firebase not available, skipping retraining", "WARNING")
        return
    
    if last_training_time:
        time_since_training = datetime.now() - last_training_time
        if time_since_training < timedelta(hours=1):
            print("Model was recently trained, skipping")
            save_logs("Model was recently trained, skipping", "INFO")
            return
    
    print("Starting scheduled model retraining...")
    save_logs("Starting scheduled model retraining", "INFO")
    try:
        train_ensemble_model(use_firebase=True)
        print("Scheduled retraining completed successfully")
        save_logs("Scheduled retraining completed successfully", "INFO")
    except Exception as e:
        print(f"Error during scheduled retraining: {e}")
        save_logs(f"Error during scheduled retraining: {e}", "ERROR")

# Schedule periodic retraining
schedule.every(6).hours.do(retrain_model_periodically)

def run_scheduler():
    """Run the scheduler in a separate thread"""
    while True:
        schedule.run_pending()
        time.sleep(60)

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction with robust preprocessing"""
    try:
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
        
        input_data = request.json
        print(f"Received prediction request: {input_data}")
        save_logs(f"Prediction request received: {input_data}", "INFO")
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Add missing columns with defaults
        for col in original_feature_names:
            if col not in df.columns:
                if col in label_encoders:
                    df[col] = 'Unknown'
                else:
                    df[col] = 0
        
        # Reorder columns
        df = df[original_feature_names]
        
        # Handle categorical encoding
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].fillna('Unknown').astype(str))
                except ValueError:
                    df[col] = 0
        
        # Convert numerical columns
        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Scale numerical features
        if scaler and len(numerical_columns) > 0:
            df[numerical_columns] = scaler.transform(df[numerical_columns])
        
        # Make prediction
        probability = model.predict_proba(df)[0][1]
        prediction = model.predict(df)[0]
        
        # Risk assessment
        if probability >= 0.85:
            risk_level = "Critical Risk"
            status = "Critical Risk - Immediate Intervention Required"
        elif probability >= 0.7:
            risk_level = "Very High Risk"
            status = "Very High Risk - Close Monitoring Required"
        elif probability >= 0.5:
            risk_level = "High Risk"
            status = "High Risk - Enhanced Care Plan Needed"
        elif probability >= 0.3:
            risk_level = "Medium Risk"
            status = "Medium Risk - Standard Follow-up"
        else:
            risk_level = "Low Risk"
            status = "Low Risk - Routine Care"
        
        result = {
            'status': status,
            'score': round(probability, 3),
            'risk': risk_level,
            'confidence': f"{probability * 100:.1f}%",
            'prediction': int(prediction),
            'disease_type': input_data.get('primary_diagnosis', 'Unknown'),
            'model_accuracy': f"{model_accuracy:.1%}",
            'last_training': last_training_time.isoformat() if last_training_time else None
        }
        
        print(f"Prediction result: {result}")
        save_logs(f"Prediction completed: {result}", "INFO")
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        save_logs(f"Error in prediction: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/retrain', methods=['POST'])
def retrain_model():
    """Manual model retraining endpoint"""
    try:
        print("Manual retraining triggered")
        save_logs("Manual retraining triggered", "INFO")
        model_obj, accuracy = train_ensemble_model(use_firebase=True)
        
        result = {
            'status': 'success',
            'message': 'Model retrained successfully',
            'new_accuracy': f"{accuracy:.2%}",
            'training_time': last_training_time.isoformat() if last_training_time else None
        }
        save_logs(f"Manual retraining completed: {result}", "INFO")
        return jsonify(result)
    except Exception as e:
        error_msg = f'Retraining failed: {str(e)}'
        save_logs(error_msg, "ERROR")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get comprehensive model information"""
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    return jsonify({
        'model_type': 'Advanced Multi-Algorithm Ensemble',
        'algorithms': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees'],
        'features': original_feature_names,
        'categorical_features': list(label_encoders.keys()),
        'numerical_features': numerical_columns,
        'accuracy': f"{model_accuracy:.2%}",
        'last_training': last_training_time.isoformat() if last_training_time else None,
        'data_source': 'Firebase + Synthetic',
        'auto_retrain': True,
        'retrain_interval': '6 hours',
        'model_storage_path': DYNAMIC_PATHS['model_storage'],
        'data_storage_path': DYNAMIC_PATHS['data_storage'],
        'log_storage_path': DYNAMIC_PATHS['log_storage']
    })

@app.route('/system-info', methods=['GET'])
def system_info():
    """Get system and path information"""
    return jsonify({
        'detected_paths': DYNAMIC_PATHS,
        'firebase_credentials_found': len(DYNAMIC_PATHS['firebase_credentials']),
        'firebase_connected': db is not None,
        'platform': sys.platform,
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'script_location': os.path.abspath(__file__)
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'firebase_connected': db is not None,
        'accuracy': f"{model_accuracy:.2%}" if model else "N/A",
        'last_training': last_training_time.isoformat() if last_training_time else None,
        'data_records': 'Firebase integrated',
        'paths_configured': True,
        'model_storage': DYNAMIC_PATHS['model_storage'],
        'firebase_credentials': len(DYNAMIC_PATHS['firebase_credentials'])
    })

@app.route('/data-stats', methods=['GET'])
def data_stats():
    """Get statistics about the training data"""
    try:
        # Check if Firebase is available
        if db is not None:
            df = fetch_firebase_data()
        else:
            df = create_enhanced_training_data()
        
        target_col = 'readmitted_30_days'
        if target_col in df.columns:
            stats = {
                'total_records': len(df),
                'readmission_rate': f"{df[target_col].mean():.2%}",
                'age_distribution': {
                    'mean': f"{df['age'].mean():.1f}" if 'age' in df.columns else "N/A",
                    'median': f"{df['age'].median():.1f}" if 'age' in df.columns else "N/A"
                },
                'diagnosis_distribution': df['primary_diagnosis'].value_counts().to_dict() if 'primary_diagnosis' in df.columns else {},
                'data_quality': {
                    'missing_values': df.isnull().sum().sum(),
                    'complete_records': len(df) - df.isnull().any(axis=1).sum()
                },
                'data_source': 'Firebase' if db is not None else 'Synthetic'
            }
        else:
            stats = {'error': 'Target column not found'}
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Failed to get data stats: {str(e)}'}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """Get recent logs"""
    try:
        log_file_path = os.path.join(DYNAMIC_PATHS['log_storage'], 'hospital_ai.log')
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as log_file:
                logs = log_file.readlines()
                # Return last 50 lines
                recent_logs = logs[-50:] if len(logs) > 50 else logs
                return jsonify({
                    'logs': [log.strip() for log in recent_logs],
                    'log_file_path': log_file_path,
                    'total_lines': len(logs)
                })
        else:
            return jsonify({
                'logs': [],
                'message': 'No log file found',
                'log_file_path': log_file_path
            })
    except Exception as e:
        return jsonify({'error': f'Failed to read logs: {str(e)}'}), 500

if __name__ == "__main__":
    print("=" * 80)
    print("STARTING ADVANCED HOSPITAL READMISSION PREDICTION API")
    print("WITH DYNAMIC PATH DETECTION AND FIREBASE INTEGRATION")
    print("=" * 80)
    
    # Print all detected paths for debugging
    print_detected_paths()
    
    # Initialize logging
    save_logs("Application starting up", "INFO")
    save_logs(f"Detected paths: {DYNAMIC_PATHS}", "INFO")
    
    # Initialize Firebase
    firebase_available = initialize_firebase()
    save_logs(f"Firebase initialization: {'Success' if firebase_available else 'Failed'}", 
              "INFO" if firebase_available else "WARNING")
    
    # Try to load existing model
    if not load_model():
        print("Training new model...")
        save_logs("Training new model", "INFO")
        try:
            train_ensemble_model(use_firebase=firebase_available)
        except Exception as e:
            print(f"Training failed: {e}")
            save_logs(f"Training failed: {e}", "ERROR")
            print("Creating fallback model with synthetic data...")
            save_logs("Creating fallback model with synthetic data", "WARNING")
            train_ensemble_model(use_firebase=False)
    
    # Start scheduler thread for periodic retraining
    if firebase_available:
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("Auto-retraining scheduler started")
        save_logs("Auto-retraining scheduler started", "INFO")
    
    print(f"\n🎯 Model initialization complete! Accuracy: {model_accuracy:.2%}")
    print(f"📁 Model stored in: {DYNAMIC_PATHS['model_storage']}")
    print(f"📊 Data stored in: {DYNAMIC_PATHS['data_storage']}")
    print(f"📝 Logs stored in: {DYNAMIC_PATHS['log_storage']}")
    print(f"🔥 Firebase: {'Connected' if firebase_available else 'Not Connected'}")
    
    if DYNAMIC_PATHS['firebase_credentials']:
        print(f"🔑 Firebase credentials found: {len(DYNAMIC_PATHS['firebase_credentials'])} files")
    else:
        print("⚠️  No Firebase credentials found in common locations")
        print("   Please ensure your firebase-service-account-key.json is in one of these locations:")
        print("   - Current directory")
        print("   - ./credentials/")
        print("   - ./config/")
        print("   - ./firebase/")
        print("   Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
    
    print("\n" + "=" * 80)
    print("🚀 STARTING FLASK SERVER...")
    print("=" * 80)
    
    save_logs("Flask server starting", "INFO")
    app.run(debug=True, host='0.0.0.0', port=8000)
