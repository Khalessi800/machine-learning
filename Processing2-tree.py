for column in df_train_cleaned.columns:
    if df_train_cleaned[column].dtype == 'object':  # Vérifier uniquement les colonnes qualitatives
        missing_count = (df_train_cleaned[column] =='?').sum()
        if missing_count > 0:
            print(f"Colonne '{column}' contient {missing_count} valeurs manquantes ('?').")

# Vérifier la présence de '?' dans chaque colonne
for column in df_test_cleaned.columns:
    if df_test_cleaned[column].dtype == 'object':  # Vérifier uniquement les colonnes qualitatives
        missing_count = (df_test_cleaned[column] =='?').sum()
        if missing_count > 0:
            print(f"Colonne '{column}' contient {missing_count} valeurs manquantes ('?').")


# Remplacer '?' par NaN (NaN est l'indicateur de valeur manquante en pandas)
df_train_cleaned = df_train_cleaned.replace('?', pd.NA)
df_test_cleaned = df_test_cleaned.replace('?', pd.NA)

# Imputer les valeurs manquantes dans les colonnes spécifiques avec la valeur la plus fréquente
for col in ['workclass', 'occupation', 'native-country']:
    df_train_cleaned[col] = df_train_cleaned[col].fillna(df_train_cleaned[col].mode()[0])

# Imputer les valeurs manquantes dans les colonnes spécifiques avec la valeur la plus fréquente
for col in ['workclass', 'occupation', 'native-country']:
    df_test_cleaned[col] = df_test_cleaned[col].fillna(df_test_cleaned[col].mode()[0])
# Afficher les premières lignes pour vérifier
print(df_test_cleaned.head())

df_train_cleaned = df_train_cleaned.drop(['education'], axis=1)
df_test_cleaned = df_test_cleaned.drop(['education'], axis=1)


df_train_cleaned.loc[:, 'Income'] = df_train_cleaned['Income'].map({'<=50K': 0, '>50K': 1})
df_test_cleaned.loc[:, 'Income'] = df_test_cleaned['Income'].map({'<=50K.': 0, '>50K.': 1})


# Sélectionner uniquement les colonnes numériques
numeric_df = df_train_cleaned.select_dtypes(include=['float64', 'int64'])
# Créer la matrice de corrélation
corr_matrix = numeric_df.corr()
# Afficher la heatmap
plt.figure(figsize = (6, 5))
plt.title("Correlation between different features of the dataset", fontsize = 10, fontweight = 'bold')
sns.heatmap(corr_matrix, cmap = 'Reds' , annot = True)
plt.xticks(fontsize=9, rotation = 90)
plt.yticks(fontsize=9, rotation = 90)
plt.show()

df_train_cleaned=df_train_cleaned.drop(columns=['Unnamed: 2'])
df_test_cleaned=df_test_cleaned.drop(columns=['Unnamed: 2'])
