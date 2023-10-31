import numpy as np
import matplotlib.pyplot as plt

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

inicio = 1000
fim = 2000

# Extrair as linhas entre 'inicio' e 'fim' e as colunas desejadas
subset = data[inicio:fim, :]
X = subset[:, :2]  # Duas primeiras colunas
Y = subset[:, 2]   # Terceira coluna


# Set a learning rate
learning_rate = 0.01

# Train the perceptron
final_weights = perceptron_pseudocode(X, Y, learning_rate)

# Create a scatter plot of the data points
plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='green', edgecolors='k', label='Class 1')
plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1], color='blue', edgecolors='k', label='Class -1')

# Plot the decision boundary
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
x2 = (-final_weights[0] - final_weights[1] * x1) / final_weights[2]
plt.plot(x1, x2, color='red', linestyle='--', label='Decision Boundary')

plt.xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
plt.ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Perceptron Decision Boundary')
plt.show()