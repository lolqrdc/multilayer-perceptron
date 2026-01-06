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
df = pandas.read_csv(data_path, header=None)

features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
           'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']
stats = ['mean', 'se', 'worst']
COL_NAMES = ['id', 'diagnosis'] + [f'{f}_{s}' for f in features for s in stats]

df.columns = COL_NAMES
print("The dataset is charged.")
print(f" DImensions: {df.shape}")
print(f"Classes: {df['diagnosis'].value_counts()}")

print("\n Stats per classes:")
print(df.groupby('diagnosis')[df.columns[2:]].mean().round(3))

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

df['diagnosis'].value_counts().plot(kind='bar', ax=axes[0,0])
axes[0,0].set_title("Distribution of classes")

sns.boxplot(data=df, x='diagnosis', y='radius_mean', ax=axes[0,1])
axes[0,1].set_title('Radius mean per classes')

corr = df.iloc[:, 2:].corr().abs().unstack().sort_values(ascending=False)
top_corr = corr[1:11]
print(f"\nTop 10 correlations: {top_corr}")

sns.heatmap(df.iloc[:, 2:12].corr(), annot=False, cmap='coolwarm', ax=axes[0,2])
axes[0,2].set_title('Correlations (10 first features)')

df['radius_mean'].hist(bins=30, alpha=0.7, ax=axes[1,0])
axes[1,0].set_title('Distribution radius_mean')

sns.pairplot(df[['diagnosis', 'radius_mean', 'perimeter_mean', 'area_mean', 'concave_points_worst']], 
             hue='diagnosis', diag_kind='hist')
plt.tight_layout()
plt.savefig(results_dir / 'data_exploration.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Exploration terminée - voir results/data_exploration.png")