# Importer mannwhitneyu de scipy.stats
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
# Identifier les colonnes numériques
numeric_columns = df_train_balanced.select_dtypes(include=['float64', 'int64']).drop(columns=['Income']).columns
# Calcul des p-values pour chaque colonne numérique
for col in numeric_columns:
    group0 = df_train_balanced[df_train_balanced['Income'] == 0][col]
    group1 = df_train_balanced[df_train_balanced['Income'] == 1][col]
    # Test de Mann-Whitney
    _, p_value = mannwhitneyu(group0, group1)
    print(f"p-value de Mann-Whitney pour {col} : {p_value:.4f}")

variables_qualitatives =df_train_balanced.select_dtypes(include=['object']).columns
# 2. Effectuer le test du kruskall centre chaque variable qualitative et la variable cible 'Income'
for col in variables_qualitatives:
  groups=[df_train_balanced[df_train_balanced[col]==cat]['Income'] for cat in df_train_balanced[col].unique()]
  stat, p = kruskal(*groups)
  print(f"kruskall wallis p-value pour {col}:{p:.5f}")


# Identifier les colonnes numériques et catégoriques pour l'ensemble d'entraînement
numeric_features_train = df_train_balanced.select_dtypes(include=['int64', 'float64']).drop(columns=['Income']).columns
categorical_features_train = df_train_balanced.select_dtypes(include=['object']).columns
# Identifier les colonnes numériques et catégoriques pour l'ensemble de test
numeric_features_test = df_test_balanced.select_dtypes(include=['int64', 'float64']).drop(columns=['Income'], errors='ignore').columns
categorical_features_test = df_test_balanced.select_dtypes(include=['object']).columns
# Vérification que les colonnes sont cohérentes entre train et test
assert list(numeric_features_train) == list(numeric_features_test), "Les colonnes numériques ne correspondent pas entre train et test."
assert list(categorical_features_train) == list(categorical_features_test), "Les colonnes catégoriques ne correspondent pas entre train et test."
