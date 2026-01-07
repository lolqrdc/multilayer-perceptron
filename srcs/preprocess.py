import pandas 
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

script_dir = Path(__file__).parent
data_path = script_dir.parent / "data" / "data.csv"
result_dir = script_dir.parent / "results"

df = pandas.read_csv(data_path, header=None)
COL_NAMES = ['id', 'diagnosis']
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
           'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

for feature in features:
    COL_NAMES.append(feature + '_mean')
    COL_NAMES.append(feature + '_se')
    COL_NAMES.append(feature + '_worst')

df.columns = COL_NAMES

x = df.drop(['id', 'diagnosis'], axis=1).values # les caracteristiques en suppr id + diagnosis
y = (df['diagnosis'] == 'M').astype(int).values # la prediction mais en B=0 et M=1

x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

total_patients = len(df)
print(f"Data_apprentissage : {x_train.shape[0]} patientes × {x_train.shape[1]} features") # les mesures pour determiner le type
print(f"Data_validation  : {x_val.shape[0]} patientes × {x_val.shape[1]} features")
print()
print(f"Classe_apprentissage : {np.bincount(y_train)[0]} benins + {np.bincount(y_train)[1]} malins") # le diagnostic (B ou M)
print(f"Classe_validation    : {np.bincount(y_val)[0]} benins + {np.bincount(y_val)[1]} malins")
print()
print(f"Data normalized with StandardScaler")
print(f"Saved file as data_preprocessed.pkl")


with open(result_dir / 'data_preprocessed.pkl', 'wb') as f:
    pickle.dump({
        'donnees_apprentissage': x_train,      # 455 patientes × 30 features
        'donnees_validation': x_val,           # 114 patientes × 30 features
        'classes_apprentissage': y_train,      # 285 B + 170 M
        'classes_validation': y_val,           # 72 B + 42 M
        'normaliseur': scaler                  # Pour standardiser
    }, f)