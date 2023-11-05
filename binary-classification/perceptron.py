import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def sign(u):
    return 1 if u >= 0 else -1

def perceptron_pseudocode(X, Y, learning_rate):
    # Add a bias term to the input data
    X = np.c_[np.ones(X.shape[0]), X]
    
    # Initialize weights with zeros or random small values
    w = np.zeros(X.shape[1])
    
    error_exists = True
    t = 0
    
    while error_exists:
        error_exists = False
        
        for i in range(X.shape[0]):
            u_t = np.dot(w, X[i])
            y_t = sign(u_t)
            w_new = w + learning_rate * (Y[i] - y_t) * X[i]
            
            if (Y[i] != y_t):
                error_exists = True
                
            w = w_new

        t += 1

    return w

# Load data from an external CSV file
data = np.loadtxt("binary-classification/Data.csv", delimiter=',')

# Split data into features and labels
#X = data[:, :-1]
#Y = data[:, -1]

inicio = 800
fim = 2200

# Extrair as linhas entre 'inicio' e 'fim' e as colunas desejadas
subset = data[inicio:fim, :]
X = subset[:, :2]  # Duas primeiras colunas
Y = subset[:, 2]   # Terceira coluna
N = X.shape[0]

# Set a learning rate
learning_rate = 0.01

# Train the perceptron
#final_weights = perceptron_pseudocode(X, Y, learning_rate)


#TREINAMENTO E TESTE DO PERCEPTRON EM RODADAS

# Inicialize as variáveis para armazenar resultados
accuracies = []
sensitivities = []
specificities = []
confusion_matrices = []

# Realize o processo de treinamento e teste em 100 rodadas
R = 5

for r in range(R):

    # EMBARALHAR AS AMOSTRAS

    s = np.random.permutation(N)

    X_random = X[s,:]
    y_random = Y[s]

    #DIVIDIR TESTE E TREINO (EM X E Y) NA PROPORÇÃO 80-20

    X_treino = X_random[0:int(N*.8),:]  
    y_treino = y_random[0:int(N*.8)]  

    X_teste = X_random[int(N*.8):,:]
    y_teste = y_random[int(N*.8):]

    trained_weights = perceptron_pseudocode(X_treino, y_treino, learning_rate)

    # Create a scatter plot of the data points
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='green', edgecolors='k', label='Class 1')
    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='blue', edgecolors='k', label='Class -1')

    # Plot the decision boundary
    x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2 = (-trained_weights[0] - trained_weights[1] * x1) / trained_weights[2]
    plt.plot(x1, x2, color='red', linestyle='--', label='Decision Boundary')

    plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
    plt.show()

    # Testar o Perceptron no conjunto de teste
    correct_predictions = 0
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for i in range(X_teste.shape[0]):
        X_teste_with_bias = np.insert(X_teste[i], 0, 1)  # Adiciona o termo de viés
        u_t = np.dot(trained_weights, X_teste_with_bias)
        y_t = sign(u_t)

        if y_t == y_teste[i]:
            correct_predictions += 1

        if y_t == 1 and y_teste[i] == 1:
            true_positives += 1
        elif y_t == -1 and y_teste[i] == 1:
            false_negatives += 1
        elif y_t == -1 and y_teste[i] == -1:
            true_negatives += 1
        elif y_t == 1 and y_teste[i] == -1:
            false_positives += 1

    accuracy = correct_predictions / X_teste.shape[0]
    accuracies.append(accuracy)

    # Sensibilidade e Especificidade
    sensitivity = true_positives / (true_positives + false_negatives)
    specificity = true_negatives / (true_negatives + false_positives)
    sensitivities.append(sensitivity)
    specificities.append(specificity)

    # Matriz de Confusão
    confusion_matrix_data = np.array([[true_negatives, false_positives], [false_negatives, true_positives]])
    confusion_matrices.append(confusion_matrix_data)

# (a) Acurácia Média, com desvio padrão, maior e menor valor
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
max_accuracy = max(accuracies)
min_accuracy = min(accuracies)

print(f"(a) Acurácia Média: {mean_accuracy * 100:.2f}%")

# (b) Sensibilidade Média, com desvio padrão, maior e menor valor
mean_sensitivity = np.mean(sensitivities)
std_sensitivity = np.std(sensitivities)
max_sensitivity = max(sensitivities)
min_sensitivity = min(sensitivities)

print(f"(b) Sensibilidade Média: {mean_sensitivity * 100:.2f}%")

# (c) Especificidade Média, com desvio padrão, maior e menor valor
mean_specificity = np.mean(specificities)
std_specificity = np.std(specificities)
max_specificity = max(specificities)
min_specificity = min(specificities)

print(f"(c) Especificidade Média: {mean_specificity * 100:.2f}%")

# (d) Construir uma matriz de confusão (gráfico) para a rodada com a melhor acurácia
best_accuracy_round = accuracies.index(max_accuracy)
best_confusion_matrix = confusion_matrices[best_accuracy_round]

print("(d) Matriz de Confusão para a Melhor Acurácia:")
print(best_confusion_matrix)

# (e) Construir uma matriz de confusão (gráfico) para a rodada com a pior acurácia
worst_accuracy_round = accuracies.index(min_accuracy)
worst_confusion_matrix = confusion_matrices[worst_accuracy_round]

print("(e) Matriz de Confusão para a Pior Acurácia:")
print(worst_confusion_matrix)  




