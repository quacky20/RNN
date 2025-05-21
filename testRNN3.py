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

T = rnn.T
n_epochs = 400
alpha = 1e-5

for n in range(n_epochs):
    
    rnn.forward()
    
    Y_hat = rnn.Y_hat
    dY = Y_hat - Y_t
    L = 0.5 * np.dot(dY.T, dY) / T

    if (n%10==0): print(f"Loss after epoch {n}: ", float(L))

    rnn.backward(dY)

    rnn.Wx -= alpha*rnn.dWx
    rnn.Wy -= alpha*rnn.dWy
    rnn.Wh -= alpha*rnn.dWh
    rnn.biases -= alpha*rnn.dbiases

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()