import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Carregamento dos dados
df = pd.read_csv('bank.csv', sep=',') 

# Definição de Features (X) e Target (y)
X = df.drop(columns=['y'])
y = df['y']

# Mapeamento de colunas numéricas e categóricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'category']).columns

# Construção do Pré-processador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Montagem do Pipeline Completo
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Separação em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento
print("Iniciando o treinamento do modelo...")
pipeline.fit(X_train, y_train)

# Avaliação
y_pred = pipeline.predict(X_test)
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Exportação
with open('modelo.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("\nModelo exportado com sucesso como 'modelo.pkl'")