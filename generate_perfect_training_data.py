import pandas as pd
import numpy as np
import random

def generate_perfect_medical_data(n_samples=2500):
    """
    Generates medical data with STRONG medical logic and clear risk patterns.
    Every sample follows real medical reasoning for its classification.
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        is_high_risk = i < n_samples // 2
        
        if is_high_risk:
            # HIGH RISK: Patients with conditions that increase illness risk
            # Choose dominant risk factor
            risk_type = random.choice(['hypertension', 'diabetes', 'obesity', 'fever', 'age'])
            
            if risk_type == 'hypertension':
                age = np.random.randint(40, 86)
                bmi = np.random.uniform(18, 50)
                systolic_bp = np.random.uniform(150, 200)  # High BP
                glucose = np.random.uniform(81, 140)  # Usually normal
                body_temp = np.random.normal(98.6, 0.8)
                
            elif risk_type == 'diabetes':
                age = np.random.randint(35, 86)
                bmi = np.random.uniform(25, 80)  # Higher BMI
                systolic_bp = np.random.uniform(110, 160)
                glucose = np.random.uniform(140, 200)  # HIGH glucose
                body_temp = np.random.normal(98.6, 0.8)
                
            elif risk_type == 'obesity':
                age = np.random.randint(30, 86)
                bmi = np.random.uniform(80, 150)  # OBESE
                systolic_bp = np.random.uniform(120, 180)
                glucose = np.random.uniform(100, 160)
                body_temp = np.random.normal(98.6, 0.8)
                
            elif risk_type == 'fever':
                age = np.random.randint(8, 86)
                bmi = np.random.uniform(17.8, 40)
                systolic_bp = np.random.uniform(100, 150)
                glucose = np.random.uniform(85, 130)
                body_temp = np.random.uniform(101, 105)  # HIGH FEVER
                
            else:  # age
                age = np.random.randint(65, 86)  # ELDERLY
                bmi = np.random.uniform(20, 50)
                systolic_bp = np.random.uniform(130, 190)
                glucose = np.random.uniform(95, 160)
                body_temp = np.random.normal(98.6, 1.0)
            
            # Clip to valid ranges
            age = int(np.clip(age, 8, 85))
            bmi = np.clip(bmi, 17.8, 200)
            systolic_bp = np.clip(systolic_bp, 84, 200)
            glucose = np.clip(glucose, 81, 200)
            body_temp = np.clip(body_temp, 95.99, 102)
            
            risk_label = 'High Risk'
            
        else:
            # LOW RISK: Healthy patients
            age = np.random.randint(8, 60)  # Younger on average
            bmi = np.random.uniform(18, 28)  # Healthy BMI
            systolic_bp = np.random.uniform(100, 130)  # Normal BP
            glucose = np.random.uniform(81, 110)  # Normal glucose
            body_temp = np.random.normal(98.6, 0.5)  # Normal temp
            
            # Clip to valid ranges
            age = int(np.clip(age, 8, 85))
            bmi = np.clip(bmi, 17.8, 200)
            systolic_bp = np.clip(systolic_bp, 84, 200)
            glucose = np.clip(glucose, 81, 200)
            body_temp = np.clip(body_temp, 95.99, 102)
            
            risk_label = 'Low Risk'
        
        gender = np.random.choice(['Male', 'Female', 'Transgender'])
        
        data.append({
            'Age': age,
            'Gender': gender,
            'BMI': round(bmi, 2),
            'Systolic_BP': round(systolic_bp, 1),
            'Glucose': round(glucose, 1),
            'Body_Temp': round(body_temp, 2),
            'Risk_Label': risk_label
        })
    
    return pd.DataFrame(data)

def validate_data(df):
    """Validate that data has clear separation between classes"""
    print("\n" + "="*70)
    print("DATA VALIDATION")
    print("="*70)
    
    low_risk = df[df['Risk_Label'] == 'Low Risk']
    high_risk = df[df['Risk_Label'] == 'High Risk']
    
    print(f"\nLow Risk - Average values:")
    print(f"  Age:         {low_risk['Age'].mean():.1f}")
    print(f"  BMI:         {low_risk['BMI'].mean():.1f}")
    print(f"  Systolic_BP: {low_risk['Systolic_BP'].mean():.1f}")
    print(f"  Glucose:     {low_risk['Glucose'].mean():.1f}")
    print(f"  Body_Temp:   {low_risk['Body_Temp'].mean():.2f}")
    
    print(f"\nHigh Risk - Average values:")
    print(f"  Age:         {high_risk['Age'].mean():.1f}")
    print(f"  BMI:         {high_risk['BMI'].mean():.1f}")
    print(f"  Systolic_BP: {high_risk['Systolic_BP'].mean():.1f}")
    print(f"  Glucose:     {high_risk['Glucose'].mean():.1f}")
    print(f"  Body_Temp:   {high_risk['Body_Temp'].mean():.2f}")
    
    print(f"\nDifference (High - Low):")
    print(f"  Age:         {high_risk['Age'].mean() - low_risk['Age'].mean():.1f}")
    print(f"  BMI:         {high_risk['BMI'].mean() - low_risk['BMI'].mean():.1f}")
    print(f"  Systolic_BP: {high_risk['Systolic_BP'].mean() - low_risk['Systolic_BP'].mean():.1f}")
    print(f"  Glucose:     {high_risk['Glucose'].mean() - low_risk['Glucose'].mean():.1f}")
    print(f"  Body_Temp:   {high_risk['Body_Temp'].mean() - low_risk['Body_Temp'].mean():.2f}")

def main():
    print("=" * 70)
    print("GENERATING PERFECT MEDICAL DATA WITH CLEAR SEPARATION")
    print("=" * 70)
    print("\nFeature Ranges (Updated):")
    print("  Age:           Min=8,     Max=85")
    print("  BMI:           Min=17.8,  Max=200")
    print("  Systolic_BP:   Min=84,    Max=200")
    print("  Glucose:       Min=81,    Max=200")
    print("  Body_Temp:     Min=95.99, Max=102")
    print("\nStrategy: Clear medical logic with strong class separation")
    print("="*70)
    
    # Generate data
    df = generate_perfect_medical_data(n_samples=2500)
    
    # Ensure balance
    low_risk = df[df['Risk_Label'] == 'Low Risk']
    high_risk = df[df['Risk_Label'] == 'High Risk']
    min_samples = min(len(low_risk), len(high_risk))
    df = pd.concat([
        low_risk.sample(min_samples, random_state=42),
        high_risk.sample(min_samples, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk Label Distribution:")
    print(df['Risk_Label'].value_counts())
    
    # Validate data
    validate_data(df)
    
    # Feature ranges check
    print(f"\nActual Feature Ranges in Dataset:")
    print(f"  Age:         {df['Age'].min()}-{df['Age'].max()}")
    print(f"  BMI:         {df['BMI'].min():.1f}-{df['BMI'].max():.1f}")
    print(f"  Systolic_BP: {df['Systolic_BP'].min():.1f}-{df['Systolic_BP'].max():.1f}")
    print(f"  Glucose:     {df['Glucose'].min():.1f}-{df['Glucose'].max():.1f}")
    print(f"  Body_Temp:   {df['Body_Temp'].min():.2f}-{df['Body_Temp'].max():.2f}")
    
    # Save to CSV
    output_path = 'augmented_medical_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    
    print("\n" + "=" * 70)
    print("Data generation complete! Data is ready for perfect training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
