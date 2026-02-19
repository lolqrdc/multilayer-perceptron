### Configuration

```bash
python3 -m venv venv

source venv/bin/activate

pip install numpy pandas matplotlib seaborn

deactivate
```

### Exploration des données

Visualisations du dataset.

```bash
python srcs/explore.py [OPTIONS]
```

**Arguments :**

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--dataset` | str | `data/data.csv` | Chemin vers le dataset à analyser |
| `--output` | str | `outputs` | Dossier de sortie pour les graphiques |

**Exemple :**

```bash
# Explorer le dataset original
python srcs/explore.py

# Explorer le dataset d'entraînement
python srcs/explore.py --dataset data/data_train.csv --output outputs_train

# Explorer le dataset de validation
python srcs/explore.py --dataset data/data_valid.csv --output outputs_valid
```

**Sortie :**
- Statistiques dans le terminal (mean ± std pour 10 features)
- `outputs/distributions.png` - Distributions des features
- `outputs/analysis.png` - Violin plots
- `outputs/correlation.png` - Matrice de corrélation

---

### Split du dataset

Divise le dataset en ensembles d'entraînement (80%) et de validation (20%).

```bash
python srcs/split.py [OPTIONS]
```

**Arguments :**

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--dataset` | str | `data/data.csv` | Chemin vers le dataset à diviser |

**Exemple :**

```bash
# Valeurs par défaut
python srcs/split.py

# Dataset personnalisé
python srcs/split.py --dataset data/autre_dataset.csv
```

**Sortie :**
- `data/data_train.csv`
- `data/data_valid.csv`

---

### Entraînement du modèle

Entraîne le réseau de neurones multicouche.

```bash
python srcs/train.py [OPTIONS]
```

**Arguments :**

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--train` | str | `data/data_train.csv` | Chemin vers le dataset d'entraînement |
| `--valid` | str | `data/data_valid.csv` | Chemin vers le dataset de validation |
| `--layers` | int+ | `24 24` | Architecture du réseau (couches cachées seulement) |
| `--epochs` | int | `84` | Nombre d'epochs d'entraînement |
| `--model` | str | `model.npy` | Chemin de sauvegarde du modèle |

**Paramètres fixes :**
- Learning rate : `0.0314`
- Batch size : `8`

**Exemples :**

```bash
# Entraînement par défaut
python srcs/train.py

# Utiliser des datasets personnalisés
python srcs/train.py --train data_train.csv --valid data_perfect.csv

# Modifier l'architecture (2 couches cachées de 24 neurones chacune)
python srcs/train.py --layers 24 24

# Architecture avec 3 couches cachées
python srcs/train.py --layers 32 24 16

# Plus epochs
python srcs/train.py --epochs 150

```

**Sortie :**
- `model.npy` - Modèle entraîné
- `outputs/learning_curves.png` - Courbes d'apprentissage
- Métriques affichées pour chaque epochs :
  ```
  epoch 01/84 - loss: 0.6557 - val_loss: 0.6627 - acc: 0.6347 - val_acc: 0.6056
  ...
  epoch 84/84 - loss: 0.0564 - val_loss: 0.0605 - acc: 0.9836 - val_acc: 0.9789
  ```

---

### Prédiction

Évalue le modèle entraîné sur un dataset de test.

```bash
python srcs/predict.py [OPTIONS]
```

**Arguments :**

| Argument | Type | Défaut | Description |
|----------|------|--------|-------------|
| `--model` | str | `model.npy` | Chemin vers le modèle entraîné |
| `--dataset` | str | `data/data_valid.csv` | Dataset de test |

**Exemples :**

```bash
# Prédiction sur la validation (par défaut)
python srcs/predict.py

# Prédiction sur l'entraînement (vérifier l'overfitting)
python srcs/predict.py --dataset data/data_train.csv

# Utiliser un modèle spécifique
python srcs/predict.py --model mon_modele.npy --dataset data/data_valid.csv
```

**Sortie :**
```
Loading model from model.npy...
Model loaded: 3 layers
Test data shape: (114, 30)

Binary Cross-Entropy Loss: 0.060469
Accuracy: 0.9789 (97.89%)
```

---

### Comparaison de modèles

```bash
python srcs/compare.py
```

**Personnalisation :**
Éditer `srcs/compare.py` pour modifier :
- Les fichiers à comparer : `load_history('models/VOTRE_MODEL_history.json')`
- Les labels : `labels=['Modèle 1', 'Modèle 2']`

**Exemple :**

```bash
python srcs/train.py --model model_32.npy --layers 32 32 --epochs 84
python srcs/train.py --model model_128.npy --layers 128 128 --epochs 100

# Comparer
python srcs/compare.py
```

**Sortie :**
- `outputs/comparison.png` - Graphique comparatif avec 2 subplots (loss et accuracy)

---
