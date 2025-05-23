import numpy as np
import matplotlib.pyplot as plt

def runRNN(X_t, Y_t, Activation, n_epochs = 500, n_neurons = 500, learning_rate = 1e-5, decay = 0.01, momentum = 0.95, dt = 0, plot_each = 50):

    rnn = RNN(n_neurons, Activation)
    optimizer= Optimizer_SGD(learning_rate=learning_rate, momentum=momentum, decay=decay)
    T = max(X_t.shape)

    X_plot = np.arange(0,T)

    if dt != 0:
        X_t_dt = Y_t[0:-dt]
        Y_t_dt = Y_t[dt:]
        X_plots = X_plot[dt:]
    
    else:
        X_t_dt = X_t
        Y_t_dt = Y_t
        X_plots = X_plot

    Monitor = np.zeros((n_epochs, 1))

    print("RNN is running...")

    for n in range(n_epochs):
        
        rnn.forward(X_t_dt)
        
        Y_hat = rnn.Y_hat
        dY = Y_hat - Y_t_dt     

        rnn.backward(dY)

        optimizer.pre_update_params()
        optimizer.update_params(rnn)
        optimizer.post_update_params()

        # if (log):
        #     if (n%10 == 0):
        #         print(f"Loss after {n} epochs: ", L)

        #     if (n%100 == 0):
        #         plt.plot(X_t, Y_t)
        #         plt.plot(X_t, Y_hat)
        #         plt.legend(['y', '$\hat{y}$'])
        #         plt.show()

        if (n % plot_each == 0):

            rnn.forward(X_t)

            M = np.max(np.vstack((rnn.Y_hat, Y_t)))
            m = np.min(np.vstack((rnn.Y_hat, Y_t)))

            L = 0.5 * np.dot(dY.T, dY) / (T-dt)

            plt.plot(X_plot, Y_t)
            plt.plot(X_plot + dt, rnn.Y_hat)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend(['y', '$\hat{y}$'])
            plt.title('epoch' + str(n))

            if dt != 0:
                plt.fill_between([X_plot[-1], X_plots[-1] + dt], m, M, color = 'k', alpha = 0.1)
                plt.plot([X_plot[-1], X_plot[-1]], [m, M], 'k-', linewidth = 3)
            plt.show()

            L = float(L)
            print(f"MSSE = {L:.3f}")
        
    rnn.forward(X_t)

    if dt != 0:
        dY = rnn.Y_hat[:-dt] - Y_t[dt:]

    else:
        dY = rnn.Y_hat - Y_t

    L = 0.5 * np.dot(dY.T, dY) / (T-dt)

    plt.plot(X_plot, Y_t)
    plt.plot(X_plot + dt, rnn.Y_hat)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['y', '$\hat{y}$'])
    plt.title('epoch' + str(n))

    if dt != 0:
        plt.fill_between([X_plot[-1], X_plots[-1] + dt], m, M, color = 'k', alpha = 0.1)
        plt.plot([X_plot[-1], X_plot[-1]], [m, M], 'k-', linewidth = 3)
    plt.show()

    L = float(L)
    print(f"Done! MSSE = {L:.3f}")

    # plt.plot(X_t, Y_t)
    # plt.plot(X_t, Y_hat)
    # plt.legend(['y', '$\hat{y}$'])
    # plt.show()

    # plt.plot(range(n_epochs), Monitor)
    # plt.xlabel('Epochs')
    # plt.ylabel('MSSE')
    # plt.yscale('log')
    # plt.show()

    return rnn

def applyRNN(X_t, rnn):
    T = max(X_t.shape)
    Y_hat = np.zeros((T,1))

    H = rnn.H
    ht = H[0]

    H = [np.zeros((rnn.neurons, 1)) for t in range(T+1)]

    [_,_, Y_hat] = rnn.RNNCell(X_t, ht, rnn.ACT, H, Y_hat)

    return Y_hat


class RNN():

    def __init__(self, n_neurons, Activation):

        self.neurons = n_neurons

        self.Wx = 0.1 * np.random.randn(n_neurons, 1)
        self.Wh = 0.1 * np.random.randn(n_neurons, n_neurons)
        self.Wy = 0.1 * np.random.randn(1, n_neurons)
        self.biases = 0.1 * np.random.randn(n_neurons, 1)

        self.Activation = Activation

    def forward(self, X_t):
        self.T = max(X_t.shape)
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1))
        self.H = [np.zeros((self.neurons, 1)) for t in range(self.T + 1)]

        # initializing d_weights:
        self.dWx = 0.1 * np.zeros((self.neurons, 1))
        self.dWh = 0.1 * np.zeros((self.neurons, self.neurons))
        self.dWy = 0.1 * np.zeros((1, self.neurons))
        self.dbiases = 0.1 * np.zeros((self.neurons, 1))

        X_t = self.X_t
        H = self.H
        Y_hat = self.Y_hat
        ht = H[0]

        Activation = self.Activation

        ACT = [Activation for t in range(self.T)]

        [ACT, H, Y_hat] = self.RNNCell(X_t, ht, ACT, H, Y_hat)

        self.Y_hat = Y_hat
        self.H = H
        self.ACT = ACT

    def RNNCell(self, X_t, ht, ACT, H, Y_hat):

        for t, xt in enumerate(X_t):
            xt = xt.reshape(1,1)
            out = np.dot(self.Wx, xt) + np.dot(self.Wh, ht) + self.biases
            
            ACT[t].forward(out)
            ht = ACT[t].output

            y_hat_t = np.dot(self.Wy, ht)

            H[t+1] = ht
            Y_hat[t] = y_hat_t

        return ACT, H, Y_hat
    

    def backward(self, dvalues):
        # dvalues = dy

        T = self.T
        H = self.H
        X_t = self.X_t

        ACT = self.ACT

        dWx = self.dWx
        dWy = self.dWy
        dWh = self.dWh
        dbiases = self.dbiases

        Wy = self.Wy
        Wh = self.Wh
        dht = np.dot(Wy.T, dvalues[-1].reshape(1,1))

        for t in reversed(range(T)):
            dy = dvalues[t].reshape(1,1)
            xt = X_t[t].reshape(1,1)

            ACT[t].backward(dht)
            dtanh = ACT[t].dinput

            dWx += np.dot(dtanh, xt)
            dWy += np.dot(H[t+1], dy).T
            dWh += np.dot(H[t], dtanh.T)
            dbiases += dtanh

            dht = np.dot(Wh, dtanh) + np.dot(Wy.T, dy)

        self.dWx = dWx
        self.dWy = dWy
        self.dWh = dWh
        self.dbiases = dbiases


class Tanh():

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        deriv = 1 - self.output**2
        self.dinput = np.multiply(deriv, dvalues)

class Optimizer_SGD:

    def __init__(self, learning_rate = 1, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay*self.iterations))
    
    def update_params(self, layer):
        
        if self.momentum:

            if not hasattr(layer, 'Wx_momentums'):
                layer.Wx_momentums = np.zeros_like(layer.Wx)
                layer.Wy_momentums = np.zeros_like(layer.Wy)
                layer.Wh_momentums = np.zeros_like(layer.Wh)
                layer.bias_momentums = np.zeros_like(layer.biases)

            Wx_updates = self.momentum * layer.Wx_momentums - self.current_learning_rate * layer.dWx
            layer.Wx_momentums = Wx_updates

            Wy_updates = self.momentum * layer.Wy_momentums - self.current_learning_rate * layer.dWy
            layer.Wy_momentums = Wy_updates

            Wh_updates = self.momentum * layer.Wh_momentums - self.current_learning_rate * layer.dWh
            layer.Wh_momentums = Wh_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:
            Wx_updates = -self.current_learning_rate * layer.dWx
            Wy_updates = -self.current_learning_rate * layer.dWy
            Wh_updates = -self.current_learning_rate * layer.dWh
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.Wx += Wx_updates
        layer.Wy += Wy_updates
        layer.Wh += Wh_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1