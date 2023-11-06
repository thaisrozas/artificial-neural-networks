import numpy as np
import matplotlib.pyplot as plt

# Carregando dados
data = np.loadtxt("binary-classification/Data.csv", delimiter=",")
inicio = 800
fim = 2200

# Extrair as linhas entre 'inicio' e 'fim' e as colunas desejadas
subset = data[inicio:fim, :]
X = subset[:, :2]  # Duas primeiras colunas
Y = subset[:, 2]  # Terceira coluna
N, p = X.shape

plt.scatter(X[:, 0], X[:, 1], c=Y, linewidths=0.4, edgecolors="k")
plt.title("Gráfico de Espalhamento")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

X = np.concatenate((-np.ones((N, 1)), X), axis=1)


def sign(x):
    return 1 if x >= 0 else -1


def mean_squared_error(d, u):
    return np.mean((d - u) ** 2)


def train_adaline(X, d, eta, max_epochs, epsilon):
    num_samples, num_features = X.shape
    w = np.random.rand(num_features)

    for epoch in range(max_epochs):
        EQM_anterior = mean_squared_error(d, np.dot(X, w))

        for i in range(num_samples):
            u_t = np.dot(X[i], w)
            error = d[i] - u_t
            w += eta * error * X[i]

        EQM_atual = mean_squared_error(d, np.dot(X, w))

        if abs(EQM_atual - EQM_anterior) <= epsilon:
            break

    return w


def adaline_test(w, x_unknown):
    u = np.dot(w, x_unknown)
    y_t = sign(u)
    # if y_t == -1:
    #     print("ADALINE: A amostra pertence à classe A")
    # else:
    #     print("ADALINE: A amostra pertence à classe B")
    return y_t


# Funções para métricas de avaliação
def acuracia(y_true, y_pred):
    return sum(y_true == y_pred) / len(y_true)


def sensibilidade(y_true, y_pred):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == -1))
    if TP + FN == 0:
        return 0  # Evita divisão por zero
    return TP / (TP + FN)


def especificidade(y_true, y_pred):
    TN = sum((y_true == -1) & (y_pred == -1))
    FP = sum((y_true == -1) & (y_pred == 1))
    if TN + FP == 0:
        return 0  # Evita divisão por zero
    return TN / (TN + FP)


# Função para criar gráfico de matriz de confusão
def plot_confusion_matrix(y_true, y_pred, title):
    TP = sum((y_true == 1) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == -1))
    FP = sum((y_true == -1) & (y_pred == 1))
    TN = sum((y_true == -1) & (y_pred == -1))

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Classe Negativa", "Classe Positiva"])
    plt.yticks(tick_marks, ["Classe Negativa", "Classe Positiva"])
    plt.xlabel("Previsões")
    plt.ylabel("Rótulos Reais")
    plt.show()


accuracies_adaline = []
sensitivities_adaline = []
specificities_adaline = []


best_accuracy_adaline = -1
worst_accuracy_adaline = 2


Y_teste_best_accuracy_adaline = []
X_teste_best_accuracy_adaline = []

Y_teste_worst_accuracy_adaline = []
X_teste_worst_accuracy_adaline = []

for round in range(100):
    s = np.random.permutation(N)
    X_random = X[s, :]
    y_random = Y[s]

    # DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

    X_treino = X_random[0 : int(N * 0.8), :]
    Y_treino = y_random[0 : int(N * 0.8)]

    X_teste = X_random[int(N * 0.8) :, :]
    Y_teste = y_random[int(N * 0.8) :]
    w_adaline = train_adaline(X_treino, Y_treino, 0.001, 100, 0.7)

    y_pred_adaline = np.zeros(len(X_teste))

    for i in range(len(X_teste)):
        y_pred_adaline[i] = adaline_test(w_adaline, X_teste[i])

    accuracy_adaline = acuracia(Y_teste, y_pred_adaline)

    accuracies_adaline.append(accuracy_adaline)

    sensitivity_adaline = sensibilidade(Y_teste, y_pred_adaline)

    sensitivities_adaline.append(sensitivity_adaline)

    specificity_adaline = especificidade(Y_teste, y_pred_adaline)

    specificities_adaline.append(specificity_adaline)

    best_accuracy_adaline = max(best_accuracy_adaline, accuracy_adaline)
    worst_accuracy_adaline = min(worst_accuracy_adaline, accuracy_adaline)

    if accuracy_adaline == best_accuracy_adaline:
        Y_teste_best_accuracy_adaline = Y_teste
        X_teste_best_accuracy_adaline = X_teste
    if accuracy_adaline == worst_accuracy_adaline:
        Y_teste_worst_accuracy_adaline = Y_teste
        X_teste_worst_accuracy_adaline = X_teste


# Calculate statistic

mean_accuracy_adaline = np.mean(accuracies_adaline)
std_accuracy_adaline = np.std(accuracies_adaline)
max_accuracy_adaline = max(accuracies_adaline)
min_accuracy_adaline = min(accuracies_adaline)

mean_sensitivity_adaline = np.mean(sensitivities_adaline)
std_sensitivity_adaline = np.std(sensitivities_adaline)
max_sensitivity_adaline = max(sensitivities_adaline)
min_sensitivity_adaline = min(sensitivities_adaline)

mean_specificity_adaline = np.mean(specificities_adaline)
std_specificity_adaline = np.std(specificities_adaline)
max_specificity_adaline = max(specificities_adaline)
min_specificity_adaline = min(specificities_adaline)

print("Resultados do Adaline:")
print(f"Acurácia Média: {mean_accuracy_adaline * 100:.2f}")
print(f"Desvio Padrão: {std_accuracy_adaline * 100:.2f}")
print(f"Acurácia Máxima: {max_accuracy_adaline * 100:.2f}")
print(f"Acurácia Mínima: {min_accuracy_adaline * 100:.2f}")
print("")
print(f"Sensibilidade Média: {mean_sensitivity_adaline:.2f}")
print(f"Desvio Padrão da Sensibilidade: {std_sensitivity_adaline:.2f}")
print(f"Sensibilidade Máxima: {max_sensitivity_adaline:.2f}")
print(f"Sensibilidade Mínima: {min_sensitivity_adaline:.2f}")
print("")
print(f"Especificidade Média: {mean_specificity_adaline:.2f}")
print(f"Desvio Padrão da Especificidade: {std_specificity_adaline:.2f}")
print(f"Especificidade Máxima: {max_specificity_adaline:.2f}")
print(f"Especificidade Mínima: {min_specificity_adaline:.2f}")


plot_confusion_matrix(
    Y_teste_best_accuracy_adaline,
    y_pred_adaline,
    "Matriz de Confusão (Melhor caso adaline)",
)
plot_confusion_matrix(
    Y_teste_worst_accuracy_adaline,
    y_pred_adaline,
    "Matriz de Confusão (Pior caso adaline)",
)

plt.scatter(
    X_teste_best_accuracy_adaline[:, 1],
    X_teste_best_accuracy_adaline[:, 2],
    c=Y_teste_best_accuracy_adaline,
    linewidths=0.4,
    edgecolors="k",
)
plt.title("Gráfico de hiperplano (Melhor caso adaline) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.scatter(
    X_teste_worst_accuracy_adaline[:, 1],
    X_teste_worst_accuracy_adaline[:, 2],
    c=Y_teste_worst_accuracy_adaline,
    linewidths=0.4,
    edgecolors="k",
)
plt.title("Gráfico de hiperplano (Pior caso adaline) ")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
