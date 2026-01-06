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

# Noms des colonnes du standard de Breast Cancer Wisconsin
df = pandas.read_csv(data_path, header=None)
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
           'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
stats = ['mean', 'se', 'worst']
COL_NAMES = ['id', 'diagnosis'] + [f'{f}_{s}' for f in features for s in stats]
df.columns = COL_NAMES

print(f" Dataset : {df.shape}")
print(f"Classes : {df['diagnosis'].value_counts()}")

# Benign ou Malignant 
print("\n Moyenne par classe :")
print(df.groupby('diagnosis')[df.columns[2:]].mean().round(3))

# 6 graphiques
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title("Distribution classes")

sns.boxplot(data=df, x='diagnosis', y='radius_mean', ax=axes[0,1])
axes[0,1].set_title('Radius mean')

corr = df.iloc[:, 2:].corr().abs().unstack().sort_values(ascending=False)
top_corr = corr[1:11]
print(f"\nTop 10 correlations: {top_corr}")

sns.heatmap(df.iloc[:, 2:12].corr(), annot=False, cmap='coolwarm', ax=axes[0,2])
axes[0,2].set_title('Correlations')

df['radius_mean'].hist(bins=30, alpha=0.7, ax=axes[1,0])
axes[1,0].set_title('radius_mean')

sns.pairplot(df[['diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean', 'concave_points_worst']], 
             hue='diagnosis', diag_kind='hist')
plt.tight_layout()
plt.savefig(results_dir / 'data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("voir results/understand_data.png")

# À ajouter à la fin de ton data_expl.py
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df.drop(['id','diagnosis'], axis=1))
plt.figure(figsize=(8,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=(df['diagnosis']=='M').astype(int), cmap='RdYlBu')
plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Séparabilité des classes (PCA)')
plt.colorbar(scatter, label='Maligne (1) / Bénigne (0)')
plt.savefig(results_dir / 'pca_separation.png', dpi=300)
plt.show()
