import os
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Resolve paths automatically ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_KEY = os.path.join(BASE_DIR, "serviceAccountKey.json")
CSV_FILE = os.path.join(BASE_DIR, "apollo.csv")

print("🔍 Using service key:", SERVICE_KEY)
print("🔍 Using CSV file:", CSV_FILE)

# --- Initialize Firebase ---
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_KEY)
    firebase_admin.initialize_app(cred)

# --- Connect to Firestore ---
db = firestore.client()

def clean_data_for_firestore(data):
    """Clean data to ensure compatibility with Firestore"""
    cleaned = {}
    for key, value in data.items():
        # Handle NaN values
        if pd.isna(value):
            cleaned[key] = None
        # Handle numpy types
        elif hasattr(value, 'item'):
            cleaned[key] = value.item()
        # Handle numpy.bool_
        elif isinstance(value, np.bool_):
            cleaned[key] = bool(value)
        # Handle numpy integers/floats
        elif isinstance(value, (np.integer, np.floating)):
            cleaned[key] = value.item()
        # Handle strings that are too long (Firestore has 1MB document limit)
        elif isinstance(value, str) and len(value) > 1000000:
            cleaned[key] = value[:1000000]
        # Handle regular values
        else:
            cleaned[key] = value
    return cleaned

def upload_hospital_metadata(hospital_id, hospital_name, location=""):
    """Upload hospital metadata to Firestore"""
    try:
        hospital_ref = db.collection("Hospital").document(hospital_id)
        hospital_data = {
            "name": hospital_name,
            "location": location,
            "created_at": firestore.SERVER_TIMESTAMP,
            "total_records": 0
        }
        hospital_ref.set(hospital_data, merge=True)
        print(f"✅ Hospital metadata uploaded: {hospital_name}")
        return True
    except Exception as e:
        print(f"❌ Error uploading hospital metadata: {e}")
        return False

def upload_batch_with_retry(batch_data, collection_ref, max_retries=3):
    """Upload a batch of data with retry logic and smaller batch sizes"""
    
    for retry in range(max_retries):
        try:
            batch = db.batch()
            
            for doc_id, row_data in batch_data.items():
                doc_ref = collection_ref.document(doc_id)
                batch.set(doc_ref, row_data)
            
            # Commit with timeout handling
            batch.commit()
            return len(batch_data)
            
        except Exception as e:
            print(f"⚠️ Batch upload attempt {retry + 1} failed: {e}")
            if retry < max_retries - 1:
                # Wait before retry with exponential backoff
                wait_time = (retry + 1) * 2
                print(f"⏳ Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"❌ Batch upload failed after {max_retries} attempts")
                return 0

def upload_csv_data_optimized(hospital_id, csv_file_path):
    """Optimized CSV upload with smaller batches and retry logic"""
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file_path)
        print(f"📊 CSV loaded successfully with {len(df)} rows")
        print("📑 CSV Columns:", df.columns.tolist())
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Firestore collection reference
    collection_ref = db.collection("Hospital").document(hospital_id).collection("csv_data")
    
    # Use smaller batch size to avoid timeout
    BATCH_SIZE = 100  # Reduced from 500 to 100
    total_uploaded = 0
    
    # Process data in smaller chunks
    for start_idx in range(0, len(df), BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        # Prepare batch data
        batch_data = {}
        
        for index, row in chunk.iterrows():
            # Create document ID
            if "patient_id" in df.columns and not pd.isna(row["patient_id"]):
                doc_id = f"patient_{str(row['patient_id'])}"
            elif "id" in df.columns and not pd.isna(row["id"]):
                doc_id = f"patient_{str(row['id'])}"
            else:
                doc_id = f"patient_{index + 1}"
            
            # Clean data for Firestore
            row_data = clean_data_for_firestore(row.to_dict())
            batch_data[doc_id] = row_data
        
        # Upload batch with retry logic
        uploaded_count = upload_batch_with_retry(batch_data, collection_ref)
        total_uploaded += uploaded_count
        
        if uploaded_count > 0:
            print(f"📤 Uploaded batch: {total_uploaded}/{len(df)} records ({(total_uploaded/len(df)*100):.1f}%)")
        
        # Small delay between batches to avoid rate limiting
        time.sleep(0.1)
    
    # Update hospital record count
    try:
        hospital_ref = db.collection("Hospital").document(hospital_id)
        hospital_ref.update({"total_records": total_uploaded})
        print(f"✅ Updated hospital record count: {total_uploaded}")
    except Exception as e:
        print(f"⚠️ Could not update hospital record count: {e}")
    
    print(f"✅ CSV upload completed under /Hospital/{hospital_id}/csv_data")
    print(f"📊 Total records uploaded: {total_uploaded}/{len(df)}")

def upload_single_document(doc_data, collection_ref, max_retries=3):
    """Upload a single document with retry logic"""
    doc_id, row_data = doc_data
    
    for retry in range(max_retries):
        try:
            doc_ref = collection_ref.document(doc_id)
            doc_ref.set(row_data)
            return 1
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep((retry + 1) * 0.5)
            else:
                print(f"❌ Failed to upload document {doc_id}: {e}")
                return 0
    return 0

def upload_csv_data_parallel(hospital_id, csv_file_path, max_workers=5):
    """Upload CSV data using parallel processing for better performance"""
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file_path)
        print(f"📊 CSV loaded successfully with {len(df)} rows")
        print("📑 CSV Columns:", df.columns.tolist())
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    # Firestore collection reference
    collection_ref = db.collection("Hospital").document(hospital_id).collection("csv_data")
    
    # Prepare all documents
    documents_to_upload = []
    
    for index, row in df.iterrows():
        # Create document ID
        if "patient_id" in df.columns and not pd.isna(row["patient_id"]):
            doc_id = f"patient_{str(row['patient_id'])}"
        else:
            doc_id = f"patient_{index + 1}"
        
        # Clean data for Firestore
        row_data = clean_data_for_firestore(row.to_dict())
        documents_to_upload.append((doc_id, row_data))
    
    # Upload using thread pool
    total_uploaded = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all upload tasks
        future_to_doc = {
            executor.submit(upload_single_document, doc_data, collection_ref): doc_data[0]
            for doc_data in documents_to_upload
        }
        
        # Process completed uploads
        for future in as_completed(future_to_doc):
            doc_id = future_to_doc[future]
            try:
                result = future.result()
                total_uploaded += result
                
                # Progress indicator
                if total_uploaded % 100 == 0:
                    progress = (total_uploaded / len(documents_to_upload)) * 100
                    print(f"📤 Uploaded: {total_uploaded}/{len(documents_to_upload)} ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error uploading document {doc_id}: {e}")
    
    # Update hospital record count
    try:
        hospital_ref = db.collection("Hospital").document(hospital_id)
        hospital_ref.update({"total_records": total_uploaded})
    except Exception as e:
        print(f"⚠️ Could not update hospital record count: {e}")
    
    print(f"✅ CSV upload completed under /Hospital/{hospital_id}/csv_data")
    print(f"📊 Total records uploaded: {total_uploaded}/{len(df)}")

def main():
    """Main function with improved error handling"""
    
    # Hospital configuration
    hospitals_config = [
        {
            "id": "smvch",
            "name": "SMVCH Hospital", 
            "location": "Puducherry",
            "csv_file": os.path.join(BASE_DIR, "smvch.csv")
        },
        {
            "id": "apollo",
            "name": "Apollo Hospital",
            "location": "Chennai", 
            "csv_file": os.path.join(BASE_DIR, "apollo.csv")
        },
        {
            "id": "jipmer",
            "name": "JIPMER",
            "location": "Puducherry",
            "csv_file": os.path.join(BASE_DIR, "jipmer.csv")
        }
    ]
    
    for hospital in hospitals_config:
        if os.path.exists(hospital["csv_file"]):
            print(f"\n🏥 Processing {hospital['name']}...")
            
            # Upload hospital metadata
            metadata_success = upload_hospital_metadata(
                hospital["id"], 
                hospital["name"], 
                hospital["location"]
            )
            
            if metadata_success:
                # Use optimized upload method
                print("📤 Starting optimized CSV upload...")
                upload_csv_data_optimized(hospital["id"], hospital["csv_file"])
            else:
                print(f"❌ Skipping CSV upload for {hospital['name']} due to metadata upload failure")
                
        else:
            print(f"⚠️ CSV file not found for {hospital['name']}: {hospital['csv_file']}")
    
    print("\n🎉 Upload process completed!")

if __name__ == "__main__":
    main()
