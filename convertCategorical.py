from imblearn.over_sampling import RandomOverSampler

# Vérifier si la colonne 'Income' est de type numérique, sinon la convertir
if df_train_cleaned['Income'].dtype != 'int64' and df_train_cleaned['Income'].dtype != 'float64':
    df_train_cleaned['Income'] = df_train_cleaned['Income'].astype(int)

# Séparer les variables numériques
X_numeric = df_train_cleaned.select_dtypes(include=['float64', 'int64']).drop(columns=['Income'], errors='ignore')

# Séparer les variables catégoriques
X_categorical = df_train_cleaned.select_dtypes(include=['object'])

# Concaténer les deux parties
X = pd.concat([X_numeric, X_categorical], axis=1)

# Extraire la variable cible
y = df_train_cleaned['Income']

# Rééquilibrer les données avec RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Créer un DataFrame équilibré
df_train_balanced = pd.DataFrame(X_resampled, columns=X.columns)
df_train_balanced['Income'] = y_resampled

from imblearn.over_sampling import RandomOverSampler
# Vérifier si la colonne 'Income' est de type numérique, sinon la convertir
if df_test_cleaned['Income'].dtype != 'int64' and df_test_cleaned['Income'].dtype != 'float64':
    df_test_cleaned['Income'] = df_test_cleaned['Income'].astype(int)

# Séparer les variables numériques
X_test_numeric = df_test_cleaned.select_dtypes(include=['float64', 'int64']).drop(columns=['Income'], errors='ignore')

# Séparer les variables catégoriques
X_test_categorical = df_test_cleaned.select_dtypes(include=['object'])

# Concaténer les deux parties
X_test = pd.concat([X_test_numeric, X_test_categorical], axis=1)

# Extraire la variable cible
y_test = df_test_cleaned['Income']

# Rééquilibrer les données avec RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_test_resampled, y_test_resampled = ros.fit_resample(X_test, y_test)

# Créer un DataFrame équilibré
df_test_balanced = pd.DataFrame(X_test_resampled, columns=X_test.columns)
df_test_balanced['Income'] = y_test_resampled

# Vérification des dimensions
print(f"Taille avant équilibrage : {df_test_cleaned.shape}")
print(f"Taille après équilibrage : {df_test_balanced.shape}")
