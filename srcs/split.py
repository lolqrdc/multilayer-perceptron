# GOAL: Split dataset into train/validation sets. 
# 80/20 ratio (standard ML practice) & shuffle+seed=42 for reproducibility

import csv
import os
import numpy as np

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
    
    np.savetxt(filepath, data, fmt='%s', delimiter=',')
    print(f"Saved {data.shape[0]} samples to data/{filename}")

def main():
    data = readData("data/data.csv")
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
