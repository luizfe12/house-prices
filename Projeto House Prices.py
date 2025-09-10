import pandas as pd
import numpy as np

#carregando a base de dados, removendo a coluna de índices
BaseDeDados = pd.read_csv('house-prices.csv', delimiter = ',')
X = BaseDeDados.iloc[:, 2:].values
y = BaseDeDados.iloc[:, 1].values


def ComputeCategorizartion(X, Coluna):
    D = pd.get_dummies(X[:, Coluna], dtype=int)
    X = np.delete(X, Coluna, axis = 1)
    X = np.concatenate([X, D.values], axis = 1)
    return X


X = ComputeCategorizartion(X, 5)


from sklearn.preprocessing import LabelEncoder
LabelEncoder_X = LabelEncoder()
X[:, 4] = LabelEncoder_X.fit_transform(X[:, 4])


def SplitTest(X, Y, Testesize):
    from sklearn.model_selection import train_test_split
    XTrain, XTeste, YTrain, YTeste = train_test_split(X, Y, test_size = 0.2)
    return XTrain, XTeste, YTrain, YTeste

X_treino, X_teste, y_treino, y_teste = SplitTest(X, y, 0.2)


def ComputingScalling(train, teste):
    from sklearn.preprocessing import StandardScaler
    scaleX = StandardScaler()
    train = scaleX.fit_transform(train)
    teste = scaleX.transform(teste)
    return train, teste

X_treino, X_teste = ComputingScalling(X_treino, X_teste)

#treinando os modelos de regressao e randomforest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
modelo_regressao = LinearRegression()
modelo_forest = RandomForestRegressor()

modelo_regressao.fit(X_treino, y_treino)
modelo_forest.fit(X_treino, y_treino)

#vendo as previsoes do modelo
previsoes = modelo_regressao.predict(X_teste)
previsoesm2 = modelo_forest.predict(X_teste)


from sklearn.metrics import mean_absolute_error
erro_medio = mean_absolute_error(y_teste, previsoes)
erro_medio2 = mean_absolute_error(y_teste, previsoesm2)

media_dos_precos = y.mean()

erro_percentual_medio = (erro_medio / media_dos_precos) * 100
erro_percentual_medio2 = (erro_medio2 / media_dos_precos) * 100

print(f"O preço médio das casas no dataset é: R$ {media_dos_precos:,.2f}")
print(f"O erro médio do modelo é: R$ {erro_medio:,.2f}")
print(f"O erro médio do modelo é: R$ {erro_medio2:,.2f}")
print(f"Isso representa um erro percentual médio de: {erro_percentual_medio:.2f}%")
print(f"Isso representa um erro percentual médio de: {erro_percentual_medio2:.2f}%")