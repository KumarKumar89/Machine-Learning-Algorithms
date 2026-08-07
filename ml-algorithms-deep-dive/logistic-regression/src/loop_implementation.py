"""
Logistic Regression - Loop Implementation (Educational)

This module implements Logistic Regression using explicit Python loops
for maximum clarity and educational value. See vectorized_implementation.py
for production-ready code.

Mathematical Foundation:
- Prediction: ŷ = σ(w^T x + b) where σ(z) = 1/(1+e^(-z))
- Loss: J = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m)||w||²
- Gradient: ∂J/∂wⱼ = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾ + (λ/m)wⱼ
- Update: w := w - α * ∂J/∂w
"""

import numpy as np


def sigmoid(z):
    """
    Sigmoid activation function.
    
    ↳ σ(z) = 1 / (1 + e^(-z))
    
    Maps any real number to (0, 1) for probability interpretation.
    
    Args:
        z: Input value(s) (scalar or numpy array)
    
    Returns:
        Sigmoid of input, same shape as z
    """
    return 1.0 / (1.0 + np.exp(-z))


class LogisticRegressionLoop:
    """
    Logistic Regression using explicit loops - educational clarity.
    
    This implementation prioritizes readability over performance to help
    understand the algorithm's mechanics. For production use, see
    LogisticRegressionVectorized.
    
    Attributes:
        lr (float): Learning rate for gradient descent
        n_iterations (int): Number of gradient descent iterations
        lambda_reg (float): L2 regularization strength (λ)
        weights (np.ndarray): Learned feature coefficients (w)
        bias (float): Learned intercept term (b)
        loss_history (list): Training loss at each iteration
    """
    
    def __init__(self, learning_rate=0.1, n_iterations=1000, lambda_reg=0.0):
        """
        Initialize logistic regression model.
        
        Args:
            learning_rate: Step size for gradient descent (α)
            n_iterations: Number of iterations to run
            lambda_reg: L2 regularization parameter (λ). 
                       0 = no regularization, higher = stronger regularization
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def _compute_loss(self, X, y):
        """
        Compute cross-entropy loss with L2 regularization.
        
        ↳ J = -(1/m) Σ[y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m)||w||²
        
        Args:
            X: Feature matrix (m samples × n features)
            y: Target vector (m binary labels)
        
        Returns:
            Average loss across all samples
        """
        m = len(y)
        epsilon = 1e-15  # Prevent log(0) numerical error
        
        total_loss = 0.0
        for i in range(m):
            # Forward pass: z = w^T x + b
            z = self.bias
            for j in range(len(self.weights)):
                z += self.weights[j] * X[i, j]
            
            # ŷ = σ(z)
            y_hat = sigmoid(z)
            
            # Clip to avoid log(0)
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
            
            # Cross-entropy: -(y log(ŷ) + (1-y) log(1-ŷ))
            if y[i] == 1:
                total_loss -= np.log(y_hat)
            else:
                total_loss -= np.log(1 - y_hat)
        
        # Average over samples
        avg_loss = total_loss / m
        
        # Add L2 regularization: (λ/2m)||w||²
        if self.lambda_reg > 0:
            l2_term = 0.0
            for j in range(len(self.weights)):
                l2_term += self.weights[j] ** 2
            avg_loss += (self.lambda_reg / (2 * m)) * l2_term
        
        return avg_loss
    
    def fit(self, X, y):
        """
        Fit logistic regression model using gradient descent.
        
        ↳ Implements:
           w := w - α * [(1/m) X^T(ŷ - y) + (λ/m)w]
           b := b - α * (1/m) Σ(ŷ - y)
        
        Args:
            X: Training features (m samples × n features)
            y: Training labels (m binary labels: 0 or 1)
        
        Returns:
            self (for method chaining)
        """
        m, n = X.shape
        
        # Initialize parameters to zero
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        for iteration in range(self.n_iterations):
            # Forward pass: compute all predictions
            y_hat = np.zeros(m)
            for i in range(m):
                z = self.bias
                for j in range(n):
                    z += self.weights[j] * X[i, j]
                y_hat[i] = sigmoid(z)
            
            # Compute and store loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Compute gradients via loops
            dw = np.zeros(n)
            db = 0.0
            
            for i in range(m):
                error = y_hat[i] - y[i]  # ↳ (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) from gradient derivation
                
                for j in range(n):
                    # ↳ ∂J/∂wⱼ = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) xⱼ⁽ⁱ⁾ + (λ/m)wⱼ
                    dw[j] += error * X[i, j]
                
                # ↳ ∂J/∂b = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)
                db += error
            
            # Average gradients
            dw /= m
            db /= m
            
            # Add L2 regularization gradient: (λ/m)w
            if self.lambda_reg > 0:
                for j in range(n):
                    dw[j] += (self.lambda_reg / m) * self.weights[j]
            
            # Update parameters: gradient descent step
            # ↳ w := w - α * ∇w J (gradient descent with regularization)
            for j in range(n):
                self.weights[j] -= self.lr * dw[j]
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        """
        Return probability of class 1 for each sample.
        
        ↳ P(y=1|x) = σ(w^T x + b)
        
        Args:
            X: Feature matrix (m samples × n features)
        
        Returns:
            Probability of class 1 for each sample (array of length m)
        """
        m = X.shape[0]
        probs = np.zeros(m)
        
        for i in range(m):
            z = self.bias
            for j in range(len(self.weights)):
                z += self.weights[j] * X[i, j]
            probs[i] = sigmoid(z)
        
        return probs
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels using threshold on probabilities.
        
        ↳ ŷ = 1 if σ(w^T x + b) ≥ threshold, else 0
        
        Args:
            X: Feature matrix (m samples × n features)
            threshold: Decision threshold (default 0.5)
        
        Returns:
            Predicted class labels (0 or 1) for each sample
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Compute classification accuracy.
        
        ↳ Accuracy = (1/m) Σ(ŷ⁽ⁱ⁾ == y⁽ⁱ⁾)
        
        Args:
            X: Feature matrix
            y: True labels
        
        Returns:
            Fraction of correctly classified samples
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Test the loop version when run directly
if __name__ == "__main__":
    print("="*70)
    print("LOGISTIC REGRESSION - LOOP IMPLEMENTATION TEST")
    print("="*70)
    
    np.random.seed(42)
    
    # Generate synthetic binary classification data
    m = 100
    X_loop = np.random.randn(m, 2)
    # True decision boundary: 2*x1 + 1.5*x2 - 1 = 0
    true_z = 2 * X_loop[:, 0] + 1.5 * X_loop[:, 1] - 1
    y_loop = (true_z > 0).astype(int)
    
    print(f"\nDataset: {m} samples, 2 features")
    print(f"Class distribution: {np.sum(y_loop==0)} negative, {np.sum(y_loop==1)} positive")
    
    # Train model
    model_loop = LogisticRegressionLoop(learning_rate=0.5, n_iterations=1000)
    model_loop.fit(X_loop, y_loop)
    
    print("\n=== TRAINING RESULTS ===")
    print(f"Weights: [{model_loop.weights[0]:.4f}, {model_loop.weights[1]:.4f}]")
    print(f"Bias: {model_loop.bias:.4f}")
    print(f"Training Accuracy: {model_loop.score(X_loop, y_loop):.4f}")
    print(f"Final Loss: {model_loop.loss_history[-1]:.6f}")
    
    # Show convergence
    print(f"\nLoss at iteration 0: {model_loop.loss_history[0]:.6f}")
    print(f"Loss at iteration 500: {model_loop.loss_history[500]:.6f}")
    print(f"Loss at iteration 999: {model_loop.loss_history[-1]:.6f}")
    
    # Test predictions
    print("\n=== SAMPLE PREDICTIONS ===")
    test_samples = np.array([[0.0, 0.0], [2.0, 1.0], [-2.0, -1.0]])
    probs = model_loop.predict_proba(test_samples)
    preds = model_loop.predict(test_samples)
    
    for i, (sample, prob, pred) in enumerate(zip(test_samples, probs, preds)):
        print(f"Sample {i+1}: x={sample}, P(class=1)={prob:.3f}, predicted={pred}")
    
    print("\n" + "="*70)
    print("✓ Loop implementation test complete!")
    print("="*70)
