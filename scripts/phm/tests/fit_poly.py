import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import numpy as np
from scipy.interpolate import lagrange
import torch
# Datos proporcionados
def f(x):
    x_data = np.array([0,  1,   2, 3,   4, 5,   6, 7,   8, 9, 10, 11])
    y_data = np.array([-1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    polynomial = lagrange(x_data, y_data)
    coefficients = polynomial.c
    def polynomial(x):
        pows = len(coefficients)-1
        return np.sum(coefficients * np.power(x, np.arange(pows, -1, -1)))
    
    # Obtener el polinomio como tensor de torch
    y_interp = np.array(list(map(lambda x: polynomial(x), x)))
    y_interp[x > 10.5] = 4.75
    return y_interp

# Datos para graficar
x = np.linspace(0, 15, 100)
y = f(x)

# Datos para interpolar
x_interp = np.linspace(0, 15, 100)
y_interp = f(x_interp)


# Graficar los datos originales y la interpolación de Lagrange
plt.scatter(x, y, label='Datos originales')
plt.plot(x_interp, y_interp, label='Interpolación de Lagrange', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# New custom loss
import torch
def custom_reg_loss(y_hat, y):
    y = torch.tensor(y, dtype=torch.float32)
    y_hat = torch.tensor(y_hat, dtype=torch.float32)
    coefficients = torch.tensor([
    1.253e-07, -7.165e-06, 0.0001791, -0.002575, 0.02357, 
    -0.1434, 0.5894, -1.624, 2.918, -3.23, 2.469, -1
    ], dtype=torch.float32)
    distance = torch.abs(y_hat - y)
    costs = list(map(lambda x: torch.sum(coefficients * torch.pow(x, torch.arange(11, -1, -1))), distance))
    loss = torch.tensor(costs)
    loss[distance.squeeze() > 9.5] = 4.15
    # Pasar a lista
    return loss.numpy()

# Graficar la función
y_hat = np.linspace(0, 15, 100)
y = np.zeros_like(y_hat)
loss = custom_reg_loss(y_hat, y)
plt.plot(y_hat, loss)
plt.xlabel("Distancia")
plt.ylabel("Loss")
plt.title("Custom regression loss")
plt.show()