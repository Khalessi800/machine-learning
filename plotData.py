sns.set_style("whitegrid")
plt.figure(figsize = (5, 2))
plt.title('Income Distribution of Adults', fontsize=18, fontweight='bold')
# Calculer la répartition par pourcentage de la colonne 'Income'
eda_percentage = df_train_balanced['Income'].value_counts(normalize=True).rename_axis('Income').reset_index(name='Percentage')
# Tracer le graphique avec 'hue' pour éviter l'avertissement
ax = sns.barplot(x='Income', y='Percentage', data=eda_percentage.head(10), palette='Greens_r', hue='Income', legend=False)
# Ajouter les annotations
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height:.0%}', (x + width/2, y + height*1.02), ha='center', fontweight='bold')
plt.show()
