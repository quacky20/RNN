import numpy as np
import matplotlib.pyplot as plt
from RNN import *

X_t = np.arange(-10, 10, 0.1)
X_t = X_t.reshape((len(X_t), 1))
Y_t =  np.sin(X_t) + 0.1*np.random.randn(len(X_t), 1) + 0.02 * np.exp(0.25 * X_t)

# rnn = runRNN(X_t, Y_t, Tanh(), n_epochs = 400, learning_rate = 1e-5, momentum = 0.95, decay = 0.01, n_neurons = 500)

# X_new = np.arange(0, 20, 0.3)
# X_new = X_new.reshape((len(X_new), 1))

# Y_hat = applyRNN(X_new, rnn)

# plt.plot(X_t, Y_t)
# plt.plot(X_new, Y_hat) 
# plt.legend(['y', '$\hat{y}$'])
# plt.show()

dt = 120

rnn = runRNN(Y_t, Y_t, Tanh(), n_epochs = 2000, learning_rate = 1e-6, momentum = 0.95, decay = 0.01, n_neurons = 100, dt = dt, plot_each = 500)

Y_hat = applyRNN(Y_t, rnn)

X_t = np.arange(len(Y_t))

plt.plot(X_t, Y_t)
plt.plot(X_t + dt, Y_hat)
plt.legend(['y', '$\hat{y}$'])
plt.show()