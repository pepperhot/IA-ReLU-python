import numpy as np
import matplotlib.pyplot as plt
from data_ia import *
import time


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_val = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_val / np.sum(exp_val, axis=1, keepdims=True)

class Loss_Categorical_Cross_Entropy():
    def forward(self, y_pred, y_true):
        sample = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(sample), y_true]
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct_confidence)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]
        self.dinputs = (dvalues - y_true) / samples

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iterations = 0

    def update_params(self, layer):
        if not hasattr(layer, 'm_w'):
            layer.m_w = np.zeros_like(layer.weights)
            layer.v_w = np.zeros_like(layer.weights)
            layer.m_b = np.zeros_like(layer.biases)
            layer.v_b = np.zeros_like(layer.biases)
        
        self.iterations += 1
        layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.dweights
        layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * (layer.dweights ** 2)
        layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.dbiases
        layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * (layer.dbiases ** 2)
        
        layer.weights -= self.learning_rate * layer.m_w / (1 - self.beta1 ** self.iterations) / (np.sqrt(layer.v_w / (1 - self.beta2 ** self.iterations)) + self.epsilon)
        layer.biases -= self.learning_rate * layer.m_b / (1 - self.beta1 ** self.iterations) / (np.sqrt(layer.v_b / (1 - self.beta2 ** self.iterations)) + self.epsilon)

X, y = triangles(300, 7)
print("attention ça arrive !")
plt.scatter(X[:, 0], X[:, 1])
plt.show()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="autumn", vmin=-2, vmax=np.max(y))
plt.show()

dense1 = Layer_Dense(2, 40)
activation1 = Activation_ReLu()

dense2 = Layer_Dense(40, 35)
activation2 = Activation_ReLu()

dense3 = Layer_Dense(35, 30)
activation3 = Activation_ReLu()

dense4 = Layer_Dense(30, 182)
softmax_loss = Activation_Softmax()

loss_function = Loss_Categorical_Cross_Entropy()
optimizer = Optimizer_Adam(learning_rate=0.02)

best_loss = float('inf')
for epoch in range(5000):
    start_time = time.time()
    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    dense4.forward(activation3.output)
    softmax_loss.forward(dense4.output)
    
    loss = np.mean(loss_function.forward(softmax_loss.output, y))
    predictions = np.argmax(softmax_loss.output, axis=1)
    accuracy = np.mean(predictions == y) * 100
    
    if loss < best_loss:
        best_loss = loss
        best_epoch = epoch
        best_accuracy = accuracy
        print(f'perte : {loss:.4f} | Précision : {accuracy:.2f}% | Itération {epoch}')
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="autumn")
        plt.title(f"Epoch {epoch} - Accuracy: {accuracy:.2f}%")
        plt.pause(0.01)

    if accuracy == 100:
        print("accuracy 100%")
        break
    
    loss_function.backward(softmax_loss.output, y)
    dense4.backward(loss_function.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    for layer in [dense1, dense2, dense3, dense4]:
            optimizer.update_params(layer)

end_time = time.time()
print(f"{(end_time - start_time)} secondes")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="autumn", vmin=-2, vmax=np.max(y))
plt.title("Données de référence")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap="autumn", vmin=-2, vmax=np.max(y))
plt.title(f"Prédictions (Epoch {epoch})")
plt.show()
