import pandas as pd
import numpy as np
import random

def generate_perfect_medical_data(n_samples=3000):
    """
    Generates medical data with STRONG medical correlations for perfect predictions.
    Updated feature ranges:
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
        # Decide risk category first (helps create better separation)
        is_high_risk = random.random() < 0.5
        
        if is_high_risk:
            # HIGH RISK: Generate parameters that indicate high risk
            age = np.random.choice(
                np.concatenate([
                    np.random.randint(60, 86, size=60),  # Elderly weighted
                    np.random.randint(8, 60, size=40)    # Some younger
                ])
            )
            
            # BMI: Higher values for high risk
            bmi = np.random.choice(
                np.concatenate([
                    np.random.uniform(80, 200, size=60),   # Obese range
                    np.random.uniform(17.8, 80, size=40)   # Some normal
                ])
            )
            
            # Systolic BP: Higher values
            base_bp = 150 + (age - 8) * 0.5 + (bmi - 17.8) * 0.2
            systolic_bp = np.clip(
                base_bp + np.random.normal(0, 10),
                84, 200
            )
            
            # Glucose: Higher values
            base_glucose = 140 + (age - 8) * 0.3 + (bmi - 17.8) * 0.3
            glucose = np.clip(
                base_glucose + np.random.normal(0, 12),
                81, 200
            )
            
            # Body Temp: Higher values (fever/infection)
            body_temp = np.random.choice(
                np.concatenate([
                    np.random.uniform(100, 200, size=70),   # Elevated
                    np.random.uniform(95.99, 100, size=30)  # Some normal
                ])
            )
            
            gender = random.choice(['Male', 'Female', 'Transgender'])
            risk_label = 'High Risk'
        
        else:
            # LOW RISK: Generate parameters that indicate low risk
            age = np.random.choice(
                np.concatenate([
                    np.random.randint(18, 50, size=70),     # Younger weighted
                    np.random.randint(8, 18, size=15),      # Some pediatric
                    np.random.randint(50, 86, size=15)      # Some older
                ])
            )
            
            # BMI: Lower values for low risk
            bmi = np.random.choice(
                np.concatenate([
                    np.random.uniform(18, 30, size=70),     # Healthy range
                    np.random.uniform(30, 50, size=20),     # Some overweight
                    np.random.uniform(50, 80, size=10)      # Rare high
                ])
            )
            
            # Systolic BP: Lower values
            base_bp = 100 + (age - 8) * 0.3 + (bmi - 17.8) * 0.1
            systolic_bp = np.clip(
                base_bp + np.random.normal(0, 8),
                84, 120
            )
            
            # Glucose: Lower values
            base_glucose = 90 + (age - 8) * 0.2 + (bmi - 17.8) * 0.15
            glucose = np.clip(
                base_glucose + np.random.normal(0, 10),
                81, 130
            )
            
            # Body Temp: Normal range
            body_temp = np.random.normal(98.4, 0.8)
            body_temp = np.clip(body_temp, 95.99, 102)
            
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
    
    # Perfect balance
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
    print("GENERATING PERFECT MEDICAL DATA FOR TRAINING")
    print("=" * 70)
    print("\nFeature Ranges (Updated):")
    print("  Age:           Min=8,     Max=85")
    print("  BMI:           Min=17.8,  Max=200")
    print("  Systolic_BP:   Min=84,    Max=200")
    print("  Glucose:       Min=81,    Max=200")
    print("  Body_Temp:     Min=95.99, Max=200")
    print("\nStrategy: STRONG medical correlations for clear class separation")
    print("=" * 70)
    
    # Generate data
    df = generate_perfect_medical_data(n_samples=3000)
    
    # Display statistics
    print(f"\nDataset shape: {df.shape}")
    print(f"\nRisk Label Distribution:")
    print(df['Risk_Label'].value_counts())
    print(f"\nClass proportions:")
    props = df['Risk_Label'].value_counts(normalize=True)
    print(f"  Low Risk:  {props['Low Risk']*100:.1f}%")
    print(f"  High Risk: {props['High Risk']*100:.1f}%")
    
    print(f"\nFeature Statistics for LOW RISK:")
    low_risk_stats = df[df['Risk_Label'] == 'Low Risk'][['Age', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].describe()
    print(low_risk_stats)
    
    print(f"\nFeature Statistics for HIGH RISK:")
    high_risk_stats = df[df['Risk_Label'] == 'High Risk'][['Age', 'BMI', 'Systolic_BP', 'Glucose', 'Body_Temp']].describe()
    print(high_risk_stats)
    
    # Save to CSV
    output_path = 'augmented_medical_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Data saved to: {output_path}")
    print(f"  Total records: {len(df)}")
    print(f"  Low Risk:  {len(df[df['Risk_Label'] == 'Low Risk'])} ({len(df[df['Risk_Label'] == 'Low Risk'])/len(df)*100:.1f}%)")
    print(f"  High Risk: {len(df[df['Risk_Label'] == 'High Risk'])} ({len(df[df['Risk_Label'] == 'High Risk'])/len(df)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Data generation complete! Ready for training with strong separation.")
    print("=" * 70)

if __name__ == "__main__":
    main()
