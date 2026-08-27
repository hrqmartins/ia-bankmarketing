from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Carregamento do modelo treinado ao iniciar a API
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

# Inicialização do app Flask
app = Flask(__name__)

# Criação da rota /predict acessível via método POST
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Recebe os dados da requisição no formato JSON
        dados = request.get_json()
        
        # Converte o JSON recebido em um DataFrame do Pandas
        df = pd.DataFrame(dados)
        
        # Realiza a predição usando o modelo carregado
        # O nosso pipeline aplica automaticamente o StandardScaler e o OneHotEncoder aqui!
        pred = modelo.predict(df)

        # Retorna a previsão em formato JSON
        return jsonify({
            "predicao": pred.tolist(),
            "mensagem": "Sucesso"
        })
        
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

# Execução do servidor
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)