import numpy as np
import matplotlib.pyplot as plt

def sign(u):
    return 1 if u>=0 else -1

Data = np.loadtxt("/home/thais/Desktop/artificial-neural-networks/binary-classification/Data.csv",delimiter=',', skiprows=1)

plt.scatter(Data[0:1500,0], Data[0:1500,1], color='green', edgecolors='k')
plt.scatter(Data[1500:,0], Data[1500:,1], color='blue', edgecolors='k')
plt.show()