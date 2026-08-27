# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Criação do dataframe
df = pd.read_csv('bank.csv', sep=',')
print("Tamanho do DataFrame (linhas, colunas):", df.shape)
print(df.head())

# Separação de features e variável alvo (y)
X = df.drop(columns=['y'])
y = df['y'].apply(lambda x: 1 if x == 'yes' else 0).values

# Separação de dados de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Construção do pré-processador e pipeline (Otimização)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
])

# Treinamento do modelo
pipeline.fit(X_train, y_train)

# Gere o array de previsões e apresente a acurácia
y_predict = pipeline.predict(X_test)
acc_rf = accuracy_score(y_test, y_predict)
print(f"\nAcurácia do Modelo (Random Forest): {acc_rf * 100:.2f}%")

# Exportação do arquivo .pkl
with open('modelo.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Modelo salvo como modelo.pkl")