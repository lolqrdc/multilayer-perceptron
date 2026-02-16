# Multilayer Perceptron - Breast Cancer Classification

Implémentation d'un réseau de neurones multicouche (MLP) pour la classification binaire du cancer du sein (Bénin/Malin) à partir de caractéristiques médicales.

## Dataset

Le dataset contient **569 échantillons** avec **30 caractéristiques** par échantillon :
- Classes : Malignant (M) / Benign (B)
- Données : `data/data.csv`
- Format : ID, Diagnostic, 30 features (radius, texture, perimeter, area, etc.)

## Structure du Projet

```
multilayer-perceptron/
├── data/
│   ├── data.csv            # Dataset original (569 samples)
│   ├── data_train.csv      # Dataset d'entraînement (455 samples)
│   └── data_valid.csv      # Dataset de validation (114 samples)
├── srcs/
│   ├── explore.py          # Exploration et visualisation
│   ├── split.py            # Division du dataset
│   ├── train.py            # 
│   └── predict.py          # 
└── README.md
```

### 1. `explore.py` - Exploration et Visualisation

**Fonctionnalités :**
- Lecture et analyse du dataset
- Statistiques descriptives sur les 10 features principales
- Génération de visualisations :
  - Distributions des features par diagnostic (histogrammes + KDE)
  - Violin plots pour analyse comparative

**Utilisation :**
```bash
python srcs/explore.py [data_file] [output_dir]
# Exemple :
python srcs/explore.py data/data.csv outputs
```

**Outputs :**
- `outputs/distributions.png` : Distributions des 6 premières features
- `outputs/analysis.png` : Violin plots pour analyse comparative

**Features analysées :**
- radius_mean, texture_mean, perimeter_mean
- area_mean, smoothness_mean, compactness_mean
- concavity_mean, concave_points_mean
- symmetry_mean, fractal_dimension_mean

---

### 2. `split.py` - Division du Dataset

**Fonctionnalités :**
- Split 80/20 (Train/Validation) suivant les bonnes pratiques ML
- Shuffle avec seed=42 pour reproductibilité
- Sauvegarde automatique des splits

**Utilisation :**
```bash
python srcs/split.py
```

**Résultats :**
- Dataset d'entraînement : 455 échantillons (80%)
- Dataset de validation : 114 échantillons (20%)
- Fichiers générés : `data/data_train.csv` et `data/data_valid.csv`

---

### `train.py` (En cours)
- Implémentation de la backpropagation
- Gradient descent / optimisation
- Boucle d'entraînement complète
- Sauvegarde du modèle entraîné

### `predict.py` (À faire)
- Chargement du modèle entraîné
- Prédiction sur de nouvelles données
- Évaluation sur le dataset de validation

---

## Techno utilisées

- **Python 3.x**
- **NumPy** : calculs matriciels et opérations numériques
- **Pandas** : manipulation de données
- **Matplotlib** : visualisations
- **Seaborn** : graphiques statistiques
- **CSV** : lecture des données

---

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install numpy pandas matplotlib seaborn
```

---

## Objectifs du Projet

1. Explorer et visualiser les données
2. Préparer les données (split, normalisation)
3. Construire l'architecture MLP
4. Implémenter le forward pass et backpropagation
5. Entraîner le modèle
6. Créer un système de prédiction

---

## Notes Techniques

---

## 📈 Performances Attendues

---
