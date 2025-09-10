   Previsão de Preços de Imóveis
Este é um projeto de Machine Learning para iniciantes cujo objetivo é prever o preço de imóveis com base em suas características. O script implementa um pipeline completo, desde o carregamento e pré-processamento dos dados até o treinamento e avaliação comparativa de dois modelos de regressão.

📜 Sobre o Projeto
O modelo utiliza um conjunto de dados de imóveis para aprender a relação entre features como área, número de quartos, localização, etc., e seu preço de venda. Foram treinados e avaliados dois algoritmos distintos para comparar suas performances:

Regressão Linear: Um modelo linear simples e interpretável.

Random Forest Regressor: Um modelo de conjunto (ensemble) mais complexo e poderoso, capaz de aprender relações não-lineares.

A métrica de avaliação utilizada foi o Erro Médio Absoluto (MAE), que nos informa, em média, o quanto as previsões do modelo erram em relação ao preço real.

✨ Funcionalidades
Carregamento de Dados: Carrega os dados a partir de um arquivo house-prices.csv.

Pré-processamento:

Label Encoding: Transforma features binárias (como 'sim'/'não') em 0 e 1.

One-Hot Encoding: Transforma features categóricas com múltiplos valores (como 'localização') em colunas binárias para evitar a criação de uma ordem artificial.

Feature Scaling: Padroniza as features numéricas usando StandardScaler para que todas tenham a mesma escala, otimizando o desempenho dos modelos.

Treinamento de Modelos: Treina e compara a performance de uma Regressão Linear e um Random Forest.

Avaliação: Calcula o erro médio absoluto e o erro percentual médio para ambos os modelos, facilitando a interpretação e comparação dos resultados.

🛠️ Tecnologias Utilizadas
O projeto foi desenvolvido inteiramente em Python, utilizando as seguintes bibliotecas:

Python 3.x

Pandas: Para manipulação e análise de dados.

NumPy: Para operações numéricas eficientes.

Scikit-learn: Para o pré-processamento dos dados e a implementação dos modelos de Machine Learning.

🚀 Como Executar o Projeto
Siga os passos abaixo para executar o projeto em sua máquina local.

Pré-requisitos
Python 3.8 ou superior

Git (para clonar o repositório)

Passos de Instalação
Clone o repositório:

Bash

git clone https://github.com/SEU_USUARIO/NOME_DO_SEU_REPOSITORIO.git
cd NOME_DO_SEU_REPOSITORIO
Crie e ative um ambiente virtual (Recomendado):

Bash

# Cria o ambiente
python -m venv venv

# Ativa no Windows
.\venv\Scripts\activate

# Ativa no macOS/Linux
source venv/bin/activate
Crie o arquivo requirements.txt:
Crie um arquivo chamado requirements.txt na raiz do projeto e adicione o seguinte conteúdo:

Plaintext

pandas
numpy
scikit-learn
Instale as dependências:

Bash

pip install -r requirements.txt
Executando o Script
Com o ambiente virtual ativado e as dependências instaladas, execute o script principal:

Bash

python nome_do_seu_script.py
📊 Exemplo de Saída
Ao executar o script, você verá a seguinte saída no terminal com a avaliação de performance dos modelos:

========================================
      RESULTADOS FINAIS DE PERFORMANCE      
========================================
Preço médio dos imóveis: R$ 130,427.34

--- Modelo: Regressão Linear ---
Erro Médio Absoluto: R$ 7,512.37
Erro Percentual Médio: 5.76%

--- Modelo: Random Forest Regressor ---
Erro Médio Absoluto: R$ 9,721.97
Erro Percentual Médio: 7.45%

========================================
📈 Análise dos Resultados
========================================
É interessante notar que, para este conjunto de dados e com os parâmetros padrão, o modelo de Regressão Linear (erro de 5.76%) superou o Random Forest (erro de 7.45%).

Isso é um fenômeno comum em Machine Learning e sugere que a simplicidade do modelo linear foi mais robusta e generalizou melhor para os dados de teste. O modelo Random Forest, por ser mais complexo, pode ter sofrido um leve overfitting (quando o modelo "decora" os dados de treino em vez de aprender a tendência geral), o que é comum em datasets com poucas amostras.

🔮 Próximos Passos e Melhorias
Este projeto serve como uma base sólida. Algumas ideias para expandi-lo incluem:

[ ] Realizar a otimização de hiperparâmetros do Random Forest (usando GridSearchCV ou RandomizedSearchCV) para tentar superar o resultado da Regressão Linear.

[ ] Implementar mais modelos de regressão (ex: XGBoost, LightGBM).

[ ] Criar novas features (engenharia de features) para melhorar a precisão.

[ ] Adicionar mais visualizações de dados, como gráficos de distribuição e matriz de correlação.

[ ] Envolver o melhor modelo treinado em uma API simples usando Flask ou FastAPI.
