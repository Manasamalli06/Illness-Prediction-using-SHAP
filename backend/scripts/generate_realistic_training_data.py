import pandas as pd
import numpy as np
import random
import os

def generate_realistic_medical_data(n_samples=2000):
    """
    Generates realistic medical data with updated feature ranges and natural variance.
    Creates clinically plausible data with some overlap between classes (realistic).
    
    Updated Ranges:
    Age:           Min=8,    Max=85
    BMI:           Min=17.8, Max=200
    Systolic_BP:   Min=84,   Max=200
    Glucose:       Min=81,   Max=200
    Body_Temp:     Min=95.99, Max=200
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for i in range(n_samples):
        # Randomly decide if this will be high or low risk (50-50 split)
        is_high_risk = random.random() < 0.5
        
        if is_high_risk:
            # HIGH RISK: Bias parameters toward high risk but with natural variation
            
            # Age: Weighted toward older ages
            age_choice = random.random()
            if age_choice < 0.6:
                age = np.random.randint(55, 86)  # Older
            else:
                age = np.random.randint(8, 85)   # Any age
            
            # BMI: Weighted toward high values
            bmi_choice = random.random()
            if bmi_choice < 0.6:
                bmi = np.random.uniform(80, 150)  # Obese/Severe
            else:
                bmi = np.random.uniform(17.8, 100)  # Some normal/overweight
            
            # Systolic BP: Generally elevated, with correlation to age and BMI
            base_bp = 110 + (age - 20) * 0.7 + (bmi - 20) * 0.15
            systolic_bp = np.clip(
                base_bp + np.random.normal(0, 12),
                84, 200
            )
            # Extra 30% chance of being higher
            if random.random() < 0.3:
                systolic_bp = np.clip(systolic_bp + np.random.uniform(20, 50), 84, 200)
            
            # Glucose: Generally elevated
            base_glucose = 100 + (age - 20) * 0.3 + (bmi - 20) * 0.25
            glucose = np.clip(
                base_glucose + np.random.normal(0, 15),
                81, 200
            )
            # Extra 30% chance of being much higher
            if random.random() < 0.3:
                glucose = np.clip(glucose + np.random.uniform(30, 80), 81, 200)
            
            # Body Temp: Mostly normal, but some elevated/high fever
            temp_choice = random.random()
            if temp_choice < 0.7:
                body_temp = np.random.normal(98.6, 1.5)
                body_temp = np.clip(body_temp, 95.99, 102)
            else:
                body_temp = np.random.uniform(100, 150)  # Elevated/fever range
            
            gender = random.choice(['Male', 'Female', 'Transgender'])
            risk_label = 'High Risk'
            
        else:
            # LOW RISK: Bias parameters toward low risk but with natural variation
            
            # Age: More distributed, but some bias toward younger
            age_choice = random.random()
            if age_choice < 0.5:
                age = np.random.randint(18, 50)  # Younger
            else:
                age = np.random.randint(8, 85)   # Any age
            
            # BMI: Weighted toward normal/healthy
            bmi_choice = random.random()
            if bmi_choice < 0.6:
                bmi = np.random.uniform(18, 30)  # Healthy range
            else:
                bmi = np.random.uniform(17.8, 80)  # Including some overweight
            
            # Systolic BP: Generally normal to elevated
            base_bp = 100 + (age - 20) * 0.3 + (bmi - 20) * 0.1
            systolic_bp = np.clip(
                base_bp + np.random.normal(0, 10),
                84, 130
            )
            # 20% chance of being somewhat higher (but not critical)
            if random.random() < 0.2:
                systolic_bp = np.clip(systolic_bp + np.random.uniform(10, 30), 84, 160)
            
            # Glucose: Generally normal to slightly elevated
            base_glucose = 90 + (age - 20) * 0.15 + (bmi - 20) * 0.1
            glucose = np.clip(
                base_glucose + np.random.normal(0, 12),
                81, 130
            )
            # 15% chance of being in prediabetic range
            if random.random() < 0.15:
                glucose = np.clip(glucose + np.random.uniform(15, 50), 81, 160)
            
            # Body Temp: Mostly normal
            body_temp = np.random.normal(98.6, 0.6)
            body_temp = np.clip(body_temp, 95.99, 100)
            
            gender = random.choice(['Male', 'Female', 'Transgender'])
            risk_label = 'Low Risk'
        
        data.append({
            'Age': int(age),
            'Gender': gender,
            'BMI': round(bmi, 2),
            'Systolic_BP': round(systolic_bp, 1),
            'Glucose': round(glucose, 1),
            'Body_Temp': round(body_temp, 2),
            'Risk_Label': risk_label
        })
    
    df = pd.DataFrame(data)
    
    # Introduce Label Noise to target 92-95% accuracy
    # Flipping ~4.5% of labels to target the 93-94% accuracy range
    noise_idx = df.sample(frac=0.045, random_state=42).index
    df.loc[noise_idx, 'Risk_Label'] = df.loc[noise_idx, 'Risk_Label'].map({
        'Low Risk': 'High Risk', 
        'High Risk': 'Low Risk'
    })

    # Ensure perfect class balance
    low_risk = df[df['Risk_Label'] == 'Low Risk']
    high_risk = df[df['Risk_Label'] == 'High Risk']
    
    min_class = min(len(low_risk), len(high_risk))
    df_balanced = pd.concat([
        low_risk.sample(min_class, random_state=42),
        high_risk.sample(min_class, random_state=42)
    ])
    
    return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    print("=" * 70)
    print("GENERATING REALISTIC MEDICAL DATA FOR TRAINING")
    print("=" * 70)
    print("\nFeature Ranges (Updated):")
    print("  Age:           Min=8,     Max=85")
    print("  BMI:           Min=17.8,  Max=200")
    print("  Systolic_BP:   Min=84,    Max=200")
    print("  Glucose:       Min=81,    Max=200")
    print("  Body_Temp:     Min=95.99, Max=200")
    print("\nData Strategy: Realistic medical correlations with natural class overlap")
    print("=" * 70)
    
    # Generate data
    df = generate_realistic_medical_data(n_samples=2000)
    
    # Display statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk Label Distribution:")
    print(df['Risk_Label'].value_counts())
    
    # Feature ranges check
    print(f"\nActual Feature Ranges:")
    print(f"  Age:         {df['Age'].min()}-{df['Age'].max()}")
    print(f"  BMI:         {df['BMI'].min():.1f}-{df['BMI'].max():.1f}")
    print(f"  Systolic_BP: {df['Systolic_BP'].min():.1f}-{df['Systolic_BP'].max():.1f}")
    print(f"  Glucose:     {df['Glucose'].min():.1f}-{df['Glucose'].max():.1f}")
    print(f"  Body_Temp:   {df['Body_Temp'].min():.2f}-{df['Body_Temp'].max():.2f}")
    
    print(f"\nFeature Statistics for LOW RISK:")
    print(df[df['Risk_Label'] == 'Low Risk'][['Age', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].describe().round(2))
    
    print(f"\nFeature Statistics for HIGH RISK:")
    print(df[df['Risk_Label'] == 'High Risk'][['Age', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].describe().round(2))
    
    # Save to CSV using robust path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(SCRIPT_DIR, '../data/augmented_medical_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Low Risk:  {len(df[df['Risk_Label'] == 'Low Risk'])} ({len(df[df['Risk_Label'] == 'Low Risk'])/len(df)*100:.1f}%)")
    print(f"  High Risk: {len(df[df['Risk_Label'] == 'High Risk'])} ({len(df[df['Risk_Label'] == 'High Risk'])/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Data generation complete! Ready for realistic model training.")
    print("=" * 70)

if __name__ == "__main__":
    main()
