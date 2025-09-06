import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import sys
from pathlib import Path
import platform
import tempfile
import joblib
from datetime import datetime
import glob

def get_dynamic_paths():
    """Dynamically detect and set up all necessary paths from OS"""
    paths = {}
    
    # Get basic system paths
    paths['current_dir'] = os.getcwd()
    paths['script_dir'] = os.path.dirname(os.path.abspath(__file__))
    paths['home_dir'] = os.path.expanduser("~")
    paths['temp_dir'] = tempfile.gettempdir()
    
    # Get user documents directory (platform-specific)
    if platform.system() == 'Windows':
        paths['documents_dir'] = os.path.join(paths['home_dir'], 'Documents')
        paths['desktop_dir'] = os.path.join(paths['home_dir'], 'Desktop')
        paths['app_data'] = os.environ.get('APPDATA', os.path.join(paths['home_dir'], 'AppData', 'Roaming'))
        paths['local_app_data'] = os.environ.get('LOCALAPPDATA', os.path.join(paths['home_dir'], 'AppData', 'Local'))
    elif platform.system() == 'Darwin':  # macOS
        paths['documents_dir'] = os.path.join(paths['home_dir'], 'Documents')
        paths['desktop_dir'] = os.path.join(paths['home_dir'], 'Desktop')
        paths['app_data'] = os.path.join(paths['home_dir'], 'Library', 'Application Support')
        paths['local_app_data'] = paths['app_data']
    else:  # Linux and other Unix-like systems
        paths['documents_dir'] = os.path.join(paths['home_dir'], 'Documents')
        paths['desktop_dir'] = os.path.join(paths['home_dir'], 'Desktop')
        paths['app_data'] = os.path.join(paths['home_dir'], '.local', 'share')
        paths['local_app_data'] = os.path.join(paths['home_dir'], '.local')
    
    # Data storage paths (prioritized list)
    data_search_paths = [
        # Project-specific data directories
        os.path.join(paths['current_dir'], 'data'),
        os.path.join(paths['script_dir'], 'data'),
        os.path.join(paths['current_dir'], 'datasets'),
        os.path.join(paths['script_dir'], 'datasets'),
        os.path.join(paths['current_dir'], 'csv'),
        os.path.join(paths['script_dir'], 'csv'),
        
        # User directories
        os.path.join(paths['documents_dir'], 'hospital_data'),
        os.path.join(paths['documents_dir'], 'data'),
        os.path.join(paths['desktop_dir'], 'hospital_data'),
        
        # Application data directories
        os.path.join(paths['app_data'], 'hospital_ai', 'data'),
        os.path.join(paths['local_app_data'], 'hospital_ai', 'data'),
        
        # Fallback directories
        paths['current_dir'],
        paths['script_dir']
    ]
    
    # Find or create data directory
    paths['data_storage'] = None
    for data_path in data_search_paths:
        try:
            os.makedirs(data_path, exist_ok=True)
            if os.access(data_path, os.W_OK):
                paths['data_storage'] = data_path
                break
        except (OSError, PermissionError):
            continue
    
    if not paths['data_storage']:
        paths['data_storage'] = paths['current_dir']
    
    # Model storage paths
    model_search_paths = [
        os.path.join(paths['current_dir'], 'models'),
        os.path.join(paths['script_dir'], 'models'),
        os.path.join(paths['app_data'], 'hospital_ai', 'models'),
        os.path.join(paths['local_app_data'], 'hospital_ai', 'models'),
        os.path.join(paths['documents_dir'], 'hospital_models'),
        paths['current_dir'],
        paths['script_dir']
    ]
    
    # Find or create model directory
    paths['model_storage'] = None
    for model_path in model_search_paths:
        try:
            os.makedirs(model_path, exist_ok=True)
            if os.access(model_path, os.W_OK):
                paths['model_storage'] = model_path
                break
        except (OSError, PermissionError):
            continue
    
    if not paths['model_storage']:
        paths['model_storage'] = paths['current_dir']
    
    # Log storage paths
    log_search_paths = [
        os.path.join(paths['current_dir'], 'logs'),
        os.path.join(paths['script_dir'], 'logs'),
        os.path.join(paths['app_data'], 'hospital_ai', 'logs'),
        os.path.join(paths['local_app_data'], 'hospital_ai', 'logs'),
        paths['temp_dir'],
        paths['current_dir']
    ]
    
    # Find or create log directory
    paths['log_storage'] = None
    for log_path in log_search_paths:
        try:
            os.makedirs(log_path, exist_ok=True)
            if os.access(log_path, os.W_OK):
                paths['log_storage'] = log_path
                break
        except (OSError, PermissionError):
            continue
    
    if not paths['log_storage']:
        paths['log_storage'] = paths['temp_dir']
    
    # Search for existing CSV files in common locations
    csv_search_patterns = [
        # Current and script directories
        os.path.join(paths['current_dir'], '*.csv'),
        os.path.join(paths['script_dir'], '*.csv'),
        os.path.join(paths['current_dir'], 'data', '*.csv'),
        os.path.join(paths['script_dir'], 'data', '*.csv'),
        
        # User directories
        os.path.join(paths['documents_dir'], '*.csv'),
        os.path.join(paths['desktop_dir'], '*.csv'),
        os.path.join(paths['documents_dir'], 'hospital_data', '*.csv'),
        os.path.join(paths['documents_dir'], 'data', '*.csv'),
        
        # Downloads directory (common location for datasets)
        os.path.join(paths['home_dir'], 'Downloads', '*.csv'),
        os.path.join(paths['home_dir'], 'downloads', '*.csv'),  # lowercase for Linux
    ]
    
    # Find existing CSV files
    existing_csv_files = []
    for pattern in csv_search_patterns:
        try:
            found_files = glob.glob(pattern)
            existing_csv_files.extend(found_files)
        except:
            continue
    
    # Filter for likely hospital/medical data files
    likely_hospital_files = []
    hospital_keywords = [
        'hospital', 'patient', 'readmission', 'medical', 'health', 
        'admission', 'discharge', 'clinical', 'healthcare', 'diagnosis'
    ]
    
    for file_path in existing_csv_files:
        filename = os.path.basename(file_path).lower()
        if any(keyword in filename for keyword in hospital_keywords):
            likely_hospital_files.append(file_path)
    
    paths['existing_csv_files'] = existing_csv_files
    paths['likely_hospital_files'] = likely_hospital_files
    
    # Set default CSV path
    paths['default_csv_path'] = os.path.join(paths['data_storage'], 'hospital_readmission_data.csv')
    
    # Additional system information
    paths['platform'] = platform.system()
    paths['python_executable'] = sys.executable
    paths['python_version'] = sys.version
    paths['username'] = os.getenv('USERNAME') or os.getenv('USER') or 'unknown'
    
    return paths

def print_detected_paths():
    """Print all detected paths for debugging"""
    print("=" * 80)
    print("DETECTED SYSTEM PATHS AND INFORMATION")
    print("=" * 80)
    
    for key, value in DYNAMIC_PATHS.items():
        if isinstance(value, list):
            print(f"{key.upper()}:")
            if value:
                for i, item in enumerate(value, 1):
                    print(f"  {i}. {item}")
            else:
                print("  (None found)")
        else:
            print(f"{key.upper()}: {value}")
    
    print("=" * 80)

def log_activity(message, log_type="INFO"):
    """Log activities to file in dynamic log directory"""
    try:
        log_file_path = os.path.join(DYNAMIC_PATHS['log_storage'], 'hospital_model.log')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{log_type}] {message}\n"
        
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry)
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

# Initialize dynamic paths
DYNAMIC_PATHS = get_dynamic_paths()

def create_sample_data(csv_path):
    """Create sample hospital readmission data if file doesn't exist"""
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    log_activity(f"Creating sample data at {csv_path}")
    
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
    readmission_prob += (np.array(data['admission_type']) == 'Emergency') * 0.2  # Emergency admission
    readmission_prob += np.random.random(n_samples) * 0.3  # Random factor
    
    # Convert to binary (0 or 1)
    data['readmitted_30_days'] = (readmission_prob > 0.4).astype(int)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    print(f"Sample data created at {csv_path} with {len(df)} records")
    print(f"Readmission rate: {df['readmitted_30_days'].mean():.2%}")
    log_activity(f"Sample data created: {len(df)} records, readmission rate: {df['readmitted_30_days'].mean():.2%}")
    
    return df

def find_best_csv_file():
    """Find the best CSV file to use for training"""
    
    # Check if there are any likely hospital files
    if DYNAMIC_PATHS['likely_hospital_files']:
        print(f"Found {len(DYNAMIC_PATHS['likely_hospital_files'])} potential hospital data files:")
        for i, file_path in enumerate(DYNAMIC_PATHS['likely_hospital_files'], 1):
            file_size = os.path.getsize(file_path)
            print(f"  {i}. {file_path} ({file_size} bytes)")
        
        # Use the first (presumably best) match
        selected_file = DYNAMIC_PATHS['likely_hospital_files'][0]
        print(f"Using: {selected_file}")
        log_activity(f"Using existing CSV file: {selected_file}")
        return selected_file
    
    # Check for any CSV files
    if DYNAMIC_PATHS['existing_csv_files']:
        print(f"Found {len(DYNAMIC_PATHS['existing_csv_files'])} CSV files:")
        for i, file_path in enumerate(DYNAMIC_PATHS['existing_csv_files'][:5], 1):  # Show first 5
            file_size = os.path.getsize(file_path)
            print(f"  {i}. {file_path} ({file_size} bytes)")
        
        if len(DYNAMIC_PATHS['existing_csv_files']) > 5:
            print(f"  ... and {len(DYNAMIC_PATHS['existing_csv_files']) - 5} more files")
        
        # Use the first CSV file found
        selected_file = DYNAMIC_PATHS['existing_csv_files'][0]
        print(f"Using: {selected_file}")
        log_activity(f"Using existing CSV file: {selected_file}")
        return selected_file
    
    # No existing files, use default path
    print("No existing CSV files found. Will create sample data.")
    log_activity("No existing CSV files found. Will create sample data.")
    return DYNAMIC_PATHS['default_csv_path']

def train_model(csv_path=None):
    """Train ensemble model on hospital data"""
    
    try:
        # If no path provided, find the best one
        if csv_path is None:
            csv_path = find_best_csv_file()
        
        log_activity(f"Starting model training with data from: {csv_path}")
        
        # Check if file exists, if not create sample data
        if not os.path.exists(csv_path):
            print(f"CSV file not found at {csv_path}. Creating sample data...")
            data = create_sample_data(csv_path)
        else:
            data = pd.read_csv(csv_path)
            print(f"Loaded existing data from {csv_path}")
            print(f"Data shape: {data.shape}")
            log_activity(f"Loaded data: {data.shape[0]} records, {data.shape[1]} columns")
        
        # Check for target column - use the actual column name from your data
        target_column = None
        if 'readmitted_30_days' in data.columns:
            target_column = 'readmitted_30_days'
        elif 'readmission' in data.columns:
            target_column = 'readmission'
        else:
            print("Error: Neither 'readmitted_30_days' nor 'readmission' column found.")
            print("Available columns:", data.columns.tolist())
            log_activity(f"Error: Target column not found. Available columns: {data.columns.tolist()}")
            raise KeyError("Target column not found in the dataset")
        
        print(f"Using target column: '{target_column}'")
        log_activity(f"Using target column: {target_column}")
        
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
        log_activity(f"Available features: {len(available_features)} out of {len(feature_columns)}")
        
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
        log_activity("Training ensemble model started")
        ensemble.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = ensemble.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained successfully!")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Model accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        log_activity(f"Model training completed. Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Save model to dynamic model storage path
        save_model_components(ensemble, label_encoders, scaler, available_features, numerical_columns, accuracy, csv_path)
        
        return ensemble, accuracy, label_encoders, scaler, available_features, numerical_columns
        
    except Exception as e:
        print(f"Error in train_model: {e}")
        log_activity(f"Error in train_model: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        raise

def save_model_components(model, label_encoders, scaler, feature_columns, numerical_columns, accuracy, data_source):
    """Save all model components to the dynamic model storage path"""
    try:
        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'numerical_columns': numerical_columns,
            'accuracy': accuracy,
            'data_source': data_source,
            'training_time': datetime.now(),
            'system_info': {
                'platform': DYNAMIC_PATHS['platform'],
                'python_version': DYNAMIC_PATHS['python_version'],
                'username': DYNAMIC_PATHS['username']
            }
        }
        
        # Primary save location
        model_file_path = os.path.join(DYNAMIC_PATHS['model_storage'], 'hospital_readmission_ensemble_model.pkl')
        joblib.dump(model_data, model_file_path)
        print(f"Model saved to: {model_file_path}")
        
        # Backup save location
        backup_path = os.path.join(DYNAMIC_PATHS['script_dir'], 'hospital_model_backup.pkl')
        try:
            joblib.dump(model_data, backup_path)
            print(f"Model backup saved to: {backup_path}")
        except Exception as backup_error:
            print(f"Warning: Could not save backup model: {backup_error}")
        
        log_activity(f"Model saved successfully to {model_file_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        log_activity(f"Error saving model: {e}", "ERROR")

def load_model_components():
    """Load saved model components from dynamic paths"""
    try:
        # Try primary location
        model_file_path = os.path.join(DYNAMIC_PATHS['model_storage'], 'hospital_readmission_ensemble_model.pkl')
        
        if not os.path.exists(model_file_path):
            # Try backup location
            model_file_path = os.path.join(DYNAMIC_PATHS['script_dir'], 'hospital_model_backup.pkl')
        
        if not os.path.exists(model_file_path):
            print("No saved model found.")
            return None
        
        model_data = joblib.load(model_file_path)
        print(f"Model loaded from: {model_file_path}")
        print(f"Model accuracy: {model_data['accuracy']:.4f} ({model_data['accuracy']*100:.2f}%)")
        print(f"Training time: {model_data.get('training_time', 'Unknown')}")
        
        log_activity(f"Model loaded from {model_file_path}")
        
        return (
            model_data['model'],
            model_data['label_encoders'],
            model_data['scaler'],
            model_data['feature_columns'],
            model_data['numerical_columns']
        )
        
    except Exception as e:
        print(f"Error loading model: {e}")
        log_activity(f"Error loading model: {e}", "ERROR")
        return None

def predict_patient(model, label_encoders, scaler, feature_columns, numerical_columns, patient_data):
    """Predict readmission for a patient"""
    
    try:
        log_activity(f"Prediction request: {patient_data}")
        
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
        log_activity(f"Prediction result: {result}, Probability: {pred_proba:.3f}")
        
        return result, round(pred_proba, 3)
        
    except Exception as e:
        print(f"Error in predict_patient: {e}")
        log_activity(f"Error in predict_patient: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return "Error in prediction", 0.0

def main():
    """Main function to demonstrate the system"""
    print("=" * 80)
    print("HOSPITAL READMISSION PREDICTION SYSTEM")
    print("WITH DYNAMIC OS PATH DETECTION")
    print("=" * 80)
    
    # Print detected paths
    print_detected_paths()
    
    # Initialize logging
    log_activity("System startup", "INFO")
    
    # Try to load existing model first
    loaded_components = load_model_components()
    
    if loaded_components:
        model, label_encoders, scaler, feature_columns, numerical_columns = loaded_components
        print("Using loaded model.")
    else:
        print("No saved model found. Training new model...")
        model, accuracy, label_encoders, scaler, feature_columns, numerical_columns = train_model()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    sample_patient = {
        'age': 75,
        'gender': 'Male',
        'primary_diagnosis': 'Heart Disease',
        'length_of_stay': 8,
        'num_medications_prescribed': 12,
        'procedures_count': 2,
        'admission_type': 'Emergency',
        'discharge_location': 'Home'
    }
    
    result, probability = predict_patient(
        model, label_encoders, scaler, feature_columns, numerical_columns, sample_patient
    )
    
    print(f"Sample Patient Data: {sample_patient}")
    print(f"Prediction Result: {result}")
    print(f"Readmission Probability: {probability}")
    
    print(f"\nSystem information logged to: {DYNAMIC_PATHS['log_storage']}")
    print(f"Model saved to: {DYNAMIC_PATHS['model_storage']}")
    print(f"Data stored in: {DYNAMIC_PATHS['data_storage']}")

if __name__ == "__main__":
    main()
