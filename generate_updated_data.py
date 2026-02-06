import pandas as pd
import numpy as np
import random

def generate_medical_data_updated_ranges(n_samples=1500):
    """
    Generates medical data using the UPDATED feature ranges from DATASET_INFO.md:
    Age:           Min=8,    Max=85
    BMI:           Min=17.8, Max=200
    Systolic_BP:   Min=84,   Max=200
    Glucose:       Min=81,   Max=200
    Body_Temp:     Min=95.99, Max=200
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Generate Age (unchanged: 8-85)
        age = np.random.randint(8, 86)
        
        # Generate Gender
        gender = np.random.choice(['Male', 'Female', 'Transgender'])
        
        # Generate BMI with updated range (17.8-200)
        bmi = np.random.uniform(17.8, 200)
        
        # Systolic BP with updated range (84-200), correlated with age and BMI
        base_bp = 84 + (age - 8) * 0.8 + (bmi - 17.8) * 0.25
        systolic_bp = np.clip(base_bp + np.random.normal(0, 15), 84, 200)
        
        # Glucose with updated range (81-200), correlated with age and BMI
        base_glucose = 81 + (age - 8) * 0.4 + (bmi - 17.8) * 0.35
        glucose = np.clip(base_glucose + np.random.normal(0, 18), 81, 200)
        
        # Body_Temp with updated range (95.99-200), mostly normal with some high values
        if random.random() < 0.85:  # 85% normal range
            body_temp = np.random.normal(98.4, 0.7)
            body_temp = np.clip(body_temp, 95.99, 102.0)
        else:  # 15% elevated/extreme values
            body_temp = np.random.uniform(102.0, 200)
        
        # Calculate risk based on medical logic
        risk_score = 0
        
        # Age contribution
        if age > 65:
            risk_score += 2
        elif age > 50:
            risk_score += 1
        
        # BMI contribution
        if bmi > 100:
            risk_score += 3
        elif bmi > 50:
            risk_score += 2
        elif bmi > 30:
            risk_score += 1
        elif bmi < 20:
            risk_score += 0.5
        
        # Systolic BP contribution
        if systolic_bp > 160:
            risk_score += 3
        elif systolic_bp > 140:
            risk_score += 2
        elif systolic_bp > 120:
            risk_score += 1
        
        # Glucose contribution
        if glucose > 150:
            risk_score += 3
        elif glucose > 130:
            risk_score += 2
        elif glucose > 110:
            risk_score += 1
        
        # Body Temp contribution
        if body_temp > 100:
            risk_score += 2
        elif body_temp < 96.5:
            risk_score += 1
        
        # Determine Risk Label
        risk_label = 'High Risk' if risk_score >= 4 else 'Low Risk'
        
        data.append({
            'Age': int(age),
            'Gender': gender,
            'BMI': round(bmi, 2),
            'Systolic_BP': round(systolic_bp, 1),
            'Glucose': round(glucose, 1),
            'Body_Temp': round(body_temp, 2),
            'Risk_Label': risk_label
        })
    
    return pd.DataFrame(data)

def main():
    print("=" * 60)
    print("GENERATING MEDICAL DATA WITH UPDATED FEATURE RANGES")
    print("=" * 60)
    print("\nFeature Ranges (Updated):")
    print("  Age:           Min=8,     Max=85")
    print("  BMI:           Min=17.8,  Max=200")
    print("  Systolic_BP:   Min=84,    Max=200")
    print("  Glucose:       Min=81,    Max=200")
    print("  Body_Temp:     Min=95.99, Max=200")
    print("\n" + "=" * 60)
    
    # Generate data
    df = generate_medical_data_updated_ranges(n_samples=1500)
    
    # Display statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk Label Distribution:")
    print(df['Risk_Label'].value_counts())
    print(f"\nClass proportions:")
    print(df['Risk_Label'].value_counts(normalize=True))
    
    print(f"\nFeature Statistics:")
    print(df[['Age', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].describe())
    
    # Save to CSV
    output_path = 'augmented_medical_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Low Risk: {len(df[df['Risk_Label'] == 'Low Risk'])} ({len(df[df['Risk_Label'] == 'Low Risk'])/len(df)*100:.1f}%)")
    print(f"  High Risk: {len(df[df['Risk_Label'] == 'High Risk'])} ({len(df[df['Risk_Label'] == 'High Risk'])/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
