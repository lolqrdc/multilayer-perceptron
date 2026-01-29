# GOAL: Understand & visualize breast cancer dataset (569 samples, 30 features).
# Focus on TOP 10 MEAN features only (ignore SE/Worst variants for clarity)

import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MAIN_FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
]

def readData(data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    filepath = os.path.join(project_dir, data)
    
    features = []
    diagnosis = []
    with open(filepath, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            diagnosis.append(row[1])      
            features.append([float(x) for x in row[2:]])  
    return np.array(features), np.array(diagnosis)

def basicInfo(X, y):
    print("Dataset shape:", X.shape)
    print(f"Classes: {np.sum(y=='M')} Malignant, {np.sum(y=='B')} Benign")
    
    df_stats = pd.DataFrame(X[:, :10], columns=MAIN_FEATURES)
    print("\n10 main features stats...")

def distributions_plot(X, y, save_path):
    df = pd.DataFrame(X[:, :10], columns=MAIN_FEATURES)
    df['diagnosis'] = y
    
    palette = {"M": "#e74c3c", "B": "#3498db"}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(MAIN_FEATURES[:6]):
        sns.histplot(data=df, x=feat, hue="diagnosis", ax=axes[i],
                     bins=25, kde=True, stat="density", palette=palette)
        axes[i].set_title(feat.replace('_mean', ''))
    
    plt.suptitle("Feature distributions")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def correlation_scatter_plot(X, y, save_path):
    df = pd.DataFrame(X[:, :10], columns=MAIN_FEATURES)
    df['diagnosis'] = y
    
    palette = {"M": "#e74c3c", "B": "#3498db"}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feat in enumerate(MAIN_FEATURES[:6]):
        sns.violinplot(data=df, x="diagnosis", y=feat, hue="diagnosis", 
                      ax=axes[i], palette=palette, legend=False)
        axes[i].set_title(feat.replace('_mean', ''))
        axes[i].set_xlabel("Diagnosis")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    data_file = "data/data.csv"
    save_dir = "outputs"
    
    if len(sys.argv) > 1: data_file = sys.argv[1]
    if len(sys.argv) > 2: save_dir = sys.argv[2]
    
    os.makedirs(save_dir, exist_ok=True)
    
    X, y = readData(data_file)
    
    basicInfo(X, y)
    
    distributions_plot(X, y, save_dir + "/distributions.png")
    correlation_scatter_plot(X, y, save_dir + "/analysis.png")
    
    print("\nPlots saved.")

if __name__ == "__main__":
    main()
