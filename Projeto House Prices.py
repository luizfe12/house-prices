import pandas as pd
import numpy as np


def ComputeCategorizartion(X, Coluna):
    # Cria as colunas "dummy" (binárias) a partir da coluna especificada.
    D = pd.get_dummies(X[:, Coluna], dtype=int)

     # Remove a coluna categórica original do array.
    X = np.delete(X, Coluna, axis = 1)

     # Concatena o array original (sem a coluna) com as novas colunas dummy.
    X = np.concatenate([X, D.values], axis = 1)
    return X


def SplitTest(X, Y, Testesize):
    # Divide os dados de features (X) e alvo (y) em conjuntos de treino e teste.
    from sklearn.model_selection import train_test_split
    XTrain, XTeste, YTrain, YTeste = train_test_split(X, Y, test_size = 0.2, random_state=12)
    return XTrain, XTeste, YTrain, YTeste


def ComputingScalling(train, teste):
    #Aplica a padronização (Standard Scaling) nos conjuntos de treino e teste
    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    teste = scaleX.transform(teste)
    return train, teste



#Carregando a base de dados
BaseDeDados = pd.read_csv('house-prices.csv', delimiter = ',')
# X: Pega todas as colunas a partir da terceira (índice 2).
# y: Pega a segunda coluna (índice 1), que contém os preços.
X = BaseDeDados.iloc[:, 2:].values
y = BaseDeDados.iloc[:, 1].values

# Aplica One-Hot Encoding na coluna de índice 5
X = ComputeCategorizartion(X, 5)

# Aplica Label Encoding na coluna de índice 4
from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 4] = LabelEncoder_X.fit_transform(X[:, 4])

# Divide o dataset em 80% para treino e 20% para teste
X_treino, X_teste, y_treino, y_teste = SplitTest(X, y, 0.2)

# Aplica a padronização nos dados após a divisão, para evitar data leakage.
X_treino, X_teste = ComputingScalling(X_treino, X_teste)

#treinando os modelos de regressao e randomforest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
modelo_regressao = LinearRegression()
modelo_forest = RandomForestRegressor(n_estimators=100)

modelo_regressao.fit(X_treino, y_treino)
modelo_forest.fit(X_treino, y_treino)

#vendo as previsoes do modelo
previsoes = modelo_regressao.predict(X_teste)
previsoesm2 = modelo_forest.predict(X_teste)

# Calcula o Erro Médio Absoluto (MAE) para cada modelo.
from sklearn.metrics import mean_absolute_error
erro_medio = mean_absolute_error(y_teste, previsoes)
erro_medio2 = mean_absolute_error(y_teste, previsoesm2)

# Calcula o erro percentual para colocar o resultado em contexto.
media_dos_precos = y.mean()
erro_percentual_medio = (erro_medio / media_dos_precos) * 100
erro_percentual_medio2 = (erro_medio2 / media_dos_precos) * 100

print(f"O preço médio das casas no dataset é: R$ {media_dos_precos:,.2f}")
print(f"O erro médio do modelo é: R$ {erro_medio:,.2f}")
print(f"O erro médio do modelo é: R$ {erro_medio2:,.2f}")
print(f"Isso representa um erro percentual médio de: {erro_percentual_medio:.2f}%")
print(f"Isso representa um erro percentual médio de: {erro_percentual_medio2:.2f}%")
