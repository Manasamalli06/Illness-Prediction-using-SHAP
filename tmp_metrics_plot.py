
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set paths
RESULTS_PATH = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\nhanes_results.json'
OUTPUT_IMAGE = r'c:\Users\MANASA\OneDrive\Desktop\Illness-Prediction-using-SHAP\metrics_comparison.png'

def generate_metrics_bar_chart():
    # Load results
    with open(RESULTS_PATH, 'r') as f:
        results = json.load(f)
    
    # Prepare data
    models = [r['Model'] for r in results]
    accuracy = [r['Accuracy'] * 100 for r in results]
    precision = [r['Precision'] * 100 for r in results]
    recall = [r['Recall'] * 100 for r in results]
    f1 = [r['F1-Score'] * 100 for r in results]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#3498db', alpha=0.8)
    rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#2ecc71', alpha=0.8)
    rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#e67e22', alpha=0.8)
    rects4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#9b59b6', alpha=0.8)
    
    # Labeling
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Performance Metrics Comparison - NHANES Dataset', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    
    # Add values on top
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    ax.set_ylim(0, 115) # Room for labels and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Metrics comparison image saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    generate_metrics_bar_chart()
