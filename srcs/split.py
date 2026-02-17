"""
Dataset Splitting Module

This module splits the breast cancer dataset into training and validation sets
using an 80/20 ratio (standard ML practice). Implements shuffle with seed=42
for reproducibility. Outputs two CSV files: data_train.csv (~455 samples) and
data_valid.csv (~114 samples).
"""

import csv
import os
import numpy as np
import pandas as pd
import argparse

def readData(data):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    filepath = os.path.join(project_dir, data)
    
    full_data = []
    with open(filepath, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            full_data.append(row)
    return np.array(full_data, dtype=str)

def writeData(data, filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    filepath = os.path.join(project_dir, "data", filename)
    
    pd.DataFrame(data).to_csv(filepath, index=False, header=False)    
    print(f"Saved {data.shape[0]} samples to data/{filename}")

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train/validation sets")
    parser.add_argument("--dataset", default="data/data.csv", help="Path to input dataset")
    args = parser.parse_args()

    data = readData(args.dataset)
    np.random.seed(42)
    np.random.shuffle(data)
    
    # split: 455 train, 114 validation)
    n_samples = data.shape[0]
    n_train = int(0.8 * n_samples)
    n_valid = n_samples - n_train
    
    train_data = data[:n_train]
    valid_data = data[n_train:]
    
    print(f"Train split: {train_data.shape}")
    print(f"Valid split: {valid_data.shape}")
    
    writeData(train_data, "data_train.csv")
    writeData(valid_data, "data_valid.csv")
    
    print("Split completed.")

if __name__ == "__main__":
    main()
