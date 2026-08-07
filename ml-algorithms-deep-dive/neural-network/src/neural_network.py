"""
Neural Network Implementation from Scratch
Multi-layer perceptron with backpropagation
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

class NeuralNetworkClassifier:
    """Multi-layer perceptron for classification."""
    
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01, random_state=None):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.loss_history = []
        
    def _initialize_weights(self):
        """Xavier initialization."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        return self
    
    def _forward(self, X):
        """Forward pass through network."""
        activations = [X]
        pre_activations = []
        
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = current @ w + b
            pre_activations.append(z)
            
            # Apply activation
            if i < len(self.weights) - 1:  # Hidden layers
                if self.activation == 'relu':
                    current = relu(z)
                else:
                    current = sigmoid(z)
            else:  # Output layer (softmax)
                current = softmax(z)
            
            activations.append(current)
        
        return activations, pre_activations
    
    def _backward(self, activations, pre_activations, y_true):
        """Backpropagation to compute gradients."""
        m = y_true.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer error
        delta = activations[-1] - y_true
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            grad_w = activations[i].T @ delta / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = (delta @ self.weights[i].T) * relu_derivative(pre_activations[i-1])
        
        return gradients_w, gradients_b
    
    def fit(self, X, y, epochs=1000, verbose=False):
        """Train the neural network."""
        # One-hot encode labels
        n_classes = len(np.unique(y))
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y] = 1
        
        self._initialize_weights()
        
        for epoch in range(epochs):
            # Forward pass
            activations, pre_activations = self._forward(X)
            
            # Compute loss
            loss = cross_entropy_loss(activations[-1], y_onehot)
            self.loss_history.append(loss)
            
            # Backward pass
            gradients_w, gradients_b = self._backward(activations, pre_activations, y_onehot)
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * gradients_w[i]
                self.biases[i] -= self.learning_rate * gradients_b[i]
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        activations, _ = self._forward(X)
        return activations[-1]
    
    def score(self, X, y):
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.neural_network import MLPClassifier
    
    X, y = make_classification(n_samples=300, n_features=10, n_informative=8, n_classes=3, random_state=42)
    
    # Custom NN
    nn_custom = NeuralNetworkClassifier(layer_sizes=[10, 16, 8, 3], learning_rate=0.1, random_state=42)
    nn_custom.fit(X, y, epochs=500)
    acc_custom = nn_custom.score(X, y)
    
    # Sklearn NN
    nn_sklearn = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42, learning_rate_init=0.1)
    nn_sklearn.fit(X, y)
    acc_sklearn = nn_sklearn.score(X, y)
    
    print(f"Custom NN Accuracy: {acc_custom:.3f}")
    print(f"Sklearn NN Accuracy: {acc_sklearn:.3f}")
    print(f"Loss history: [{nn_custom.loss_history[0]:.3f} -> {nn_custom.loss_history[-1]:.3f}]")
