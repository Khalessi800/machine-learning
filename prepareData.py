# Préparer le préprocesseur pour les colonnes numériques et catégoriques
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features_train),  # Standardisation des colonnes numériques
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_train)  # Encodage des colonnes catégoriques
])
# Préparer le préprocesseur pour les colonnes numériques et catégoriques
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features_test),  # Standardisation des colonnes numériques
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_test)  # Encodage des colonnes catégoriques
])

# Définir X (caractéristiques) et y (variable cible) pour l'entraînement
X_train = df_train_balanced.drop(columns=['Income'])
y_train = df_train_balanced['Income']
# Définir X (caractéristiques) et y (variable cible) pour le test
X_test = df_test_balanced.drop(columns=['Income'])
y_test = df_test_balanced['Income']
print("Dimensions des ensembles :")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# Construire le pipeline avec le modèle DecisionTreeClassifier
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42))
])
