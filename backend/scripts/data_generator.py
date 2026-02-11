import pandas as pd
import numpy as np
from ctgan import CTGAN
import random
from sqlalchemy import create_engine, text
import sys
import os

# Add backend to path to import data.db_config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(BACKEND_DIR)
from data.db_config import get_db_connection_string, DB_NAME

def generate_base_data(n_samples=1000):
    """
    Generates a base synthetic dataset with logical medical correlations
    to serve as the 'real' distribution for the GAN to learn from.
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        age = np.random.randint(18, 90)
        gender = np.random.choice(['Male', 'Female', 'Transgender'])
        bmi = np.random.normal(25, 5) # Mean 25, std 5
        
        # Correlate BP and Glucose with Age and BMI
        systolic_bp = np.random.normal(120, 15) + (age - 30)/2 + (bmi - 25)
        glucose = np.random.normal(100, 20) + (bmi - 25)*1.5
        
        # Fever/Infection indicators
        body_temp = np.random.normal(98.6, 1)
        if random.random() < 0.1: # 10% chance of fever
            body_temp += np.random.uniform(1.5, 4.0)
            
        # Risk Calculation (Simplified Logic for Ground Truth)
        risk_score = 0
        if systolic_bp > 140: risk_score += 2
        if glucose > 140: risk_score += 2
        if bmi > 30: risk_score += 1
        if body_temp > 100.4: risk_score += 3
        if age > 65: risk_score += 1
        
        risk_label = 'High Risk' if risk_score >= 3 else 'Low Risk'
        
        data.append({
            'Age': int(age),
            'Gender': gender,
            'BMI': round(bmi, 1),
            'Systolic_BP': int(systolic_bp),
            'Glucose': int(glucose),
            'Body_Temp': round(body_temp, 1),
            'Risk_Label': risk_label
        })
        
    return pd.DataFrame(data)

def augment_data(df, samples_to_generate=500):
    """
    Uses CTGAN to generate synthetic data based on the input dataframe.
    """
    print(f"Training CTGAN on {len(df)} samples...")
    discrete_columns = ['Gender', 'Risk_Label']
    
    ctgan = CTGAN(epochs=50) # Reduced epochs for faster execution in this demo
    ctgan.fit(df, discrete_columns)
    
    print(f"Generating {samples_to_generate} synthetic samples...")
    synthetic_data = ctgan.sample(samples_to_generate)
    
    return synthetic_data

def save_to_mysql(df):
    """
    Saves the dataframe to MySQL database.
    Creates database and table if they don't exist.
    """
    connection_string = get_db_connection_string()
    
    # We need to create the database first if it doesn't exist
    # This requires connecting to the server without a database selected first
    # Or assuming the user created it. 
    # Let's try to connect to the specific database, if fail, try to create it.
    
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            print("Successfully connected to database.")
            pass
    except Exception:
        print(f"Database '{DB_NAME}' might not exist. Attempting to create it...")
        # Construct connection string without DB name
        base_conn_str = connection_string.rsplit('/', 1)[0]
        try:
            temp_engine = create_engine(base_conn_str)
            with temp_engine.connect() as conn:
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
                print(f"Database '{DB_NAME}' created/verified.")
        except Exception as e:
            print(f"Error creating database: {e}")
            print("Please ensure your MySQL server is running and credentials in src/data/db_config.py are correct.")
            return

    # Now connect with the database and save table
    engine = create_engine(connection_string)
    try:
        df.to_sql('patient_records', con=engine, if_exists='replace', index=False)
        print(f"Data saved to MySQL table 'patient_records'. Total records: {len(df)}")
    except Exception as e:
         print(f"Error saving to MySQL: {e}")

if __name__ == "__main__":
    print("Generating base dataset...")
    base_df = generate_base_data(1000)
    
    print("Starting Generative AI Augmentation...")
    synthetic_df = augment_data(base_df, samples_to_generate=500)
    
    # Combine and shuffle
    final_df = pd.concat([base_df, synthetic_df], ignore_index=True)
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    print("Saving to MySQL...")
    save_to_mysql(final_df)
    
    # Also save CSV as backup/reference
    csv_path = os.path.join(BACKEND_DIR, 'data', 'augmented_medical_data.csv')
    final_df.to_csv(csv_path, index=False)
    print("Process Complete.")
