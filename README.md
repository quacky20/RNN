# Simple RNN from Scratch 🧠🔄

Welcome to this **Recurrent Neural Network (RNN)** implementation built from the ground up using only NumPy! Perfect for learning how RNNs work with time series prediction. 📈✨

## Features 🚀

- Customizable number of neurons and epochs
- Gradient descent optimizer with momentum & learning rate decay ⚙️
- Support for activation functions like Tanh 🔄
- Visualization of predictions vs actual data during training 📊
- Handles time-shifted (autoregressive) sequence learning ⏳

## How to Use 🛠️

1. Prepare your time series data as input `X_t` and output `Y_t` (usually the same series shifted by some time step `dt`).
2. Call `runRNN(X_t, Y_t, Activation=Tanh(), ...)` with your desired parameters.
3. Watch the training progress and prediction plots every few epochs! 🎉
4. Use the returned RNN model to predict new sequences.

## Why This Project? 🤔

Building an RNN from scratch helped me to understand the *inner workings of sequence models* before jumping into complex architectures like LSTMs or Transformers. It’s a great stepping stone for music generation, language modeling, and more! 🎶📚
