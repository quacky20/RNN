import numpy as np
import matplotlib.pyplot as plt
from RNN import *

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape((len(X_t), 1))
Y_t =  np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1)

plt.plot(X_t, Y_t)
plt.show()  

n_neurons = 500
rnn = RNN(X_t, n_neurons)

Y_hat = rnn.Y_hat
H = rnn.H
T = rnn.T
ht = H[0]

for t, xt in enumerate(X_t):
    xt = xt.reshape(1, 1)
    [ht, y_hat_t, out] = rnn.forward(xt, ht)

    H[t+1] = ht
    Y_hat[t] = y_hat_t

for h in H:
    plt.plot(np.arange(20), h[0:20], '-k', linewidth = 1, alpha = 0.05)
plt.show()

plt.plot(X_t, Y_t)
plt.plot(X_t, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()