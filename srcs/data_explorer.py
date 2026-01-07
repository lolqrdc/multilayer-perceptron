# data_explorer.py permet de comprendre le dataset fourni, comment detecter un cancer benin et malin
import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('default')
sns.set_palette("husl")
script_dir = Path(__file__).parent
data_path = script_dir.parent / "data" / "data.csv"
results_dir = script_dir.parent / "results"
results_dir.mkdir(exist_ok=True)

# Charger et nommer les colonnes (Breast Cancer Wisconsin Standard)
df = pandas.read_csv(data_path, header=None)
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
           'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
stats = ['mean', 'se', 'worst']
COL_NAMES = ['id', 'diagnosis']
for feature in features:
    COL_NAMES.append(feature + '_mean')
    COL_NAMES.append(feature + '_se') 
    COL_NAMES.append(feature + '_worst')

df.columns = COL_NAMES

# Afficher les infos du dataset
print(f" Dataset : {df.shape}")
print(f"Classes : {df['diagnosis'].value_counts()}")

# Statistiques par classe
print("\n Moyenne par classe :")
print(df.groupby('diagnosis')[df.columns[2:]].mean().round(3))

# 8 graphiques pour voir si les deux classes (bénin/malin) sont bien distinguées
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Distribution des classes
df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0], color=['skyblue', 'salmon'])
axes[0,0].set_title("Nombre de cas : bénin vs malin")
axes[0,0].set_xlabel("Diagnostic")
axes[0,0].set_ylabel("Nombre de patients")

# Histogrammes côte à côte : taille des tumeurs bénignes vs malignes
benin = df[df['diagnosis'] == 'B']['radius_mean']
malin = df[df['diagnosis'] == 'M']['radius_mean']
axes[0,1].hist(benin, bins=30, alpha=0.6, label='Bénin', color='skyblue', edgecolor='blue')
axes[0,1].hist(malin, bins=30, alpha=0.6, label='Malin', color='salmon', edgecolor='red')
axes[0,1].set_title('Taille des tumeurs : bénin vs malin')
axes[0,1].set_xlabel('Taille (radius_mean)')
axes[0,1].set_ylabel('Nombre de cas')
axes[0,1].legend()

# Heatmap des corrélations
corr = df.iloc[:, 2:].corr().abs().unstack().sort_values(ascending=False)
top_corr = corr[1:11]
print(f"\nTop 10 correlations: {top_corr}")

sns.heatmap(df.iloc[:, 2:12].corr(), annot=False, cmap='coolwarm', ax=axes[0,2])
axes[0,2].set_title('Corrélations entre caractéristiques')

# Violin plot : comparaison des pires valeurs (worst)
sns.violinplot(data=df, x='diagnosis', y='radius_worst', ax=axes[0,3], palette=['skyblue', 'salmon'])
axes[0,3].set_title('Pires valeurs de taille : bénin vs malin')
axes[0,3].set_ylabel('Taille (radius_worst)')
axes[0,3].set_xlabel('Diagnostic')

# Violin plot : taille moyenne
sns.violinplot(data=df, x='diagnosis', y='radius_mean', ax=axes[1,0], palette=['skyblue', 'salmon'])
axes[1,0].set_title('Taille : bénin vs malin (distribution)')
axes[1,0].set_ylabel('Taille (radius_mean)')
axes[1,0].set_xlabel('Diagnostic')

# Scatter 1 : Taille vs Concavité
sns.scatterplot(data=df, x='radius_mean', y='concave_points_worst', hue='diagnosis', 
                palette={'B': 'skyblue', 'M': 'salmon'}, s=50, ax=axes[1,1])
axes[1,1].set_title('Taille vs Concavité\n(bleu=bénin, rouge=malin)')

# Scatter 2 : Texture vs Périmètre
sns.scatterplot(data=df, x='texture_mean', y='perimeter_mean', hue='diagnosis',
                palette={'B': 'skyblue', 'M': 'salmon'}, s=50, ax=axes[1,2])
axes[1,2].set_title('Texture vs Périmètre\n(bleu=bénin, rouge=malin)')

# Scatter 3 : Concavité moyenne vs Concavité worst
sns.scatterplot(data=df, x='concavity_mean', y='concave_points_worst', hue='diagnosis',
                palette={'B': 'skyblue', 'M': 'salmon'}, s=50, ax=axes[1,3])
axes[1,3].set_title('Concavité moyenne vs Pire concavité\n(bleu=bénin, rouge=malin)')
plt.tight_layout()
plt.savefig(results_dir / 'data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("voir results/understand_data.png")

# Réduire les données en 2 dimensions avec PCA pour voir si les classes se chevauchent
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop(['id','diagnosis'], axis=1))

# Afficher la projection PCA
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=(df['diagnosis']=='M').astype(int), cmap='RdYlBu')
plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Séparabilité : les classes se chevauchent-elles ? (PCA)')
plt.colorbar(scatter, label='Malin (1) / Benin (0)')
plt.savefig(results_dir / 'pca_separation.png', dpi=300)
plt.show()

print("voir results/pca_separation.png")
