import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset Iris direto do sklearn
iris = load_iris()

# Matriz n x 4
x = iris.data

# Vetor com valores {0, 1, 2}
y = iris.target

# Transforma em um vetor booleano
# (setosa é 0, se for setosa é 1, que é true)
# Daí transforma em um problema binário
# y = (y == 0).astype(int)
# Como aqui a execução agora é em one-vs-all, não faz sentido fazer isso
# Senão quebra a lógica de treinar vários modelos hahahaha

# Aqui faz o split e a estraficação
# 70 treino, 30 teste
# (O random=42 é só pra reprodutabilidade)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42, stratify=y
)

# Escaler faz a escala do dados e ajuda na otimização
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Função Sigmoide, como nos slides
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Função de perda, como nos slides
def loss(y, y_hat):
    # epsilon assim evita o log(0)
    eps = 1e-15
    return -np.mean(
        y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps)
    )

# Treinamento por gradiente descendente, como nos slides
# (lr: learning rate; epochs: iterações)
def train_logistic_gd(x, y, lr=0.1, epochs=1000):
    n_samples, n_features = x.shape

    # Inicialização
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        # Loop de otimização
        linear = x @ w + b
        y_hat = sigmoid(linear)

        # gradientes
        dw = (1 / n_samples) * (x.T @ (y_hat - y))
        db = (1 / n_samples) * np.sum(y_hat - y)

        # atualização
        w -= lr * dw
        b -= lr * db

    return w, b

# Treinamento do one vs all
# Decompoe o problema multiclasse em vários problemas independentes
def train_one_vs_all(x, y, num_classes):
    models = []

    for k in range(num_classes):
        # Classe k vs o resto
        y_binary = (y == k).astype(int)

        # Aqui executa a regressão logística pra cada classe
        w, b = train_logistic_gd(x, y_binary)
        models.append((w,b))

    return models

#  Treino Antigo
#w, b = train_logistic_gd(x_train, y_train)

def predict_ova(x, models):
    probs = []

    for w, b in models:
        prob = sigmoid(x @ w + b)
        probs.append(prob)

    # shape: (n_classes, n_samples) -> transpor
    probs = np.vstack(probs).T

    # pega a classe com maior prob
    return np.argmax(probs, axis=1)

# Função de predição antiga
# Threshold de 0.5
# Retorna 0 ou 1
def predict(x, w, b):
    return (sigmoid(x @ w + b) >= 0.5).astype(int)
# Lembrando:
#   x -> Dados de entradas (features)
#   w -> os pesos aprendidos pelo modelo
#   b -> o viés (bias/intercepto)

# Predição antiga
# y_pred = predict(x_test, w, b)

# Numero de classes é a quantidade de itens no vetor y
num_classes = len(np.unique(y))

# treina o modelo
models = train_one_vs_all(x_train, y_train, num_classes)

# Realiza a predição com base no one vs all
y_pred = predict_ova(x_test, models)

acc = (y_pred == y_test).mean()
print("Acurácia:", acc)
