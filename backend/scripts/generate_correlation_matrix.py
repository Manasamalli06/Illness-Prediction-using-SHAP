import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_correlation_matrix():
    # Load the data
    data_path = "backend/data/augmented_medical_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path)

    # Select relevant features
    features = ['Age', 'BMI', 'Glucose', 'Systolic_BP', 'Body_Temp']
    # Rename for cleaner display
    display_names = {
        'Age': 'Age',
        'BMI': 'BMI',
        'Glucose': 'Glucose Level',
        'Systolic_BP': 'Systolic BP',
        'Body_Temp': 'Body Temperature'
    }
    
    correlation_df = df[features].rename(columns=display_names)

    # Calculate Pearson Correlation
    corr_matrix = correlation_df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Custom color palette (modern/premium)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap=cmap, 
                center=0,
                square=True, 
                linewidths=.5, 
                cbar_kws={"shrink": .8},
                annot_kws={"size": 12, "weight": "bold"})

    plt.title('Fig. 4. Pearson Correlation Matrix', fontsize=16, pad=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    
    # Save the plot
    output_path = "correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation matrix saved to {output_path}")

if __name__ == "__main__":
    generate_correlation_matrix()
