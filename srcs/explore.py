"""
Exploration and Visualization Module

This module provides data exploration and visualization for the breast cancer
dataset (569 samples, 30 features). It focuses on the top 10 mean features,
ignoring SE and Worst variants for clarity. Generates distribution plots and
feature analysis visualizations.
"""

import csv
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

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
    print("\n10 main features stats (mean ± std):")
    for feat in MAIN_FEATURES:
        mean = df_stats[feat].mean()
        std = df_stats[feat].std()
        print(f"  {feat:25s} {mean:8.3f} ± {std:.3f}")

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

def violin_plot(X, y, save_path):
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

def correlation_heatmap(X, save_path):
    df = pd.DataFrame(X[:, :10], columns=MAIN_FEATURES)
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True, linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Explore and visualize dataset")
    parser.add_argument("--dataset", default="data/data.csv", help="Path to input dataset")
    parser.add_argument("--output", default="outputs", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    X, y = readData(args.dataset)
    
    basicInfo(X, y)
    
    distributions_plot(X, y, os.path.join(args.output, "distributions.png"))
    violin_plot(X, y, os.path.join(args.output, "analysis.png"))
    correlation_heatmap(X, os.path.join(args.output, "correlation.png"))
    
    print("\nPlots saved.")

if __name__ == "__main__":
    main()
