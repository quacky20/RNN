import numpy as np
import matplotlib.pyplot as plt
from RNN import *

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape((len(X_t), 1))
Y_t =  np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

# plt.plot(X_t, Y_t)
# plt.show()  

n_neurons = 500

rnn = RNN(X_t, n_neurons, Tanh())
optimizer= Optimizer_SGD(learning_rate = 1e-5, momentum = 0.95, decay = 0.01)

T = rnn.T
n_epochs = 400

Monitor = np.zeros((n_epochs, 1))

for n in range(n_epochs):
    
    rnn.forward()
    
    Y_hat = rnn.Y_hat
    dY = Y_hat - Y_t
    L = 0.5 * np.dot(dY.T, dY) / T

    Monitor[n] = L

    if (n%10 == 0):
        print(f"Loss after {n} epochs: ", L)

    if (n%100 == 0):
        plt.plot(X_t, Y_t)
        plt.plot(X_t, Y_hat)
        plt.legend(['y', '$\hat{y}$'])
        plt.show()

    rnn.backward(dY)

    optimizer.pre_update_params()
    optimizer.update_params(rnn)
    optimizer.post_update_params()
    

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()

plt.plot(range(n_epochs), Monitor)
plt.xlabel('Epochs')
plt.ylabel('MSSE')
plt.yscale('log')
plt.show()