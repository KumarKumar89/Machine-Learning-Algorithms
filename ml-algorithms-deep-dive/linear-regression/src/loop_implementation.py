"""
Linear Regression using explicit loops - educational clarity.

This implementation prioritizes understanding over performance.
Every operation is visible and traceable to the mathematical equations.

Mathematical Foundation:
- Cost: J(θ) = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
- Gradient: ∂J/∂wⱼ = (2/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾
- Update: wⱼ := wⱼ - α * ∂J/∂wⱼ
"""

import numpy as np
from typing import Tuple, List


class LinearRegressionLoop:
    """
    Linear Regression using explicit loops for educational purposes.
    
    This version makes every computation step visible, perfect for
    understanding how gradient descent actually works under the hood.
    
    Attributes:
        learning_rate (float): Step size for gradient descent (α in equations)
        n_iterations (int): Number of gradient descent iterations
        weights (np.ndarray): Learned feature coefficients (w)
        bias (float): Learned intercept term (b)
        loss_history (List[float]): MSE loss at each iteration
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the model.
        
        Args:
            learning_rate: Step size for gradient updates (α)
            n_iterations: Number of passes through the data
        """
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history: List[float] = []
    
    def _compute_prediction(self, X: np.ndarray, i: int) -> float:
        """
        Compute prediction for a single sample using loops.
        
        ↳ Implements: ŷ⁽ⁱ⁾ = w^T x⁽ⁱ⁾ + b = Σ wⱼ * xⱼ⁽ⁱ⁾ + b
        
        Args:
            X: Feature matrix (m samples, n features)
            i: Index of sample to predict
            
        Returns:
            Predicted value for sample i
        """
        prediction = self.bias  # Start with bias term
        
        # Add contribution from each feature
        for j in range(X.shape[1]):
            prediction += self.weights[j] * X[i, j]
        
        return prediction
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute Mean Squared Error loss.
        
        ↳ Implements: J(θ) = (1/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)²
        
        Args:
            y_pred: Predicted values
            y_true: True target values
            
        Returns:
            Mean squared error
        """
        m = len(y_true)
        total_squared_error = 0.0
        
        for i in range(m):
            error = y_pred[i] - y_true[i]
            total_squared_error += error ** 2
        
        return total_squared_error / m
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionLoop':
        """
        Fit the model using batch gradient descent with explicit loops.
        
        This method implements the full training loop where each iteration:
        1. Computes predictions for all samples (forward pass)
        2. Calculates the MSE loss
        3. Computes gradients for each parameter
        4. Updates parameters by stepping opposite to gradient
        
        ↳ Update rule: θ := θ - α * ∇θ J(θ)
           where ∇θ J(θ) = (2/m) * X^T (Xθ - y)
        
        Args:
            X: Feature matrix of shape (m samples, n features)
            y: Target vector of shape (m,)
            
        Returns:
            Self for method chaining
        """
        m, n = X.shape
        
        # Initialize parameters to zero
        # Starting point: all weights = 0, bias = 0
        self.weights = np.zeros(n)
        self.bias = 0.0
        
        # Training loop
        for iteration in range(self.n_iterations):
            # ────────────────────────────────────────────────────────
            # STEP 1: Forward Pass - Compute all predictions
            # ────────────────────────────────────────────────────────
            y_pred = np.zeros(m)
            
            for i in range(m):
                # ↳ ŷ⁽ⁱ⁾ = w^T x⁽ⁱ⁾ + b (Eq. 1 from Section 3)
                y_pred[i] = self._compute_prediction(X, i)
            
            # ────────────────────────────────────────────────────────
            # STEP 2: Compute Loss
            # ────────────────────────────────────────────────────────
            errors = y_pred - y
            loss = self._compute_loss(y_pred, y)
            self.loss_history.append(loss)
            
            # ────────────────────────────────────────────────────────
            # STEP 3: Compute Gradients via Loops
            # ────────────────────────────────────────────────────────
            dw = np.zeros(n)  # Gradient for weights
            db = 0.0          # Gradient for bias
            
            for i in range(m):
                error = errors[i]  # (ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)
                
                # Gradient for each weight
                for j in range(n):
                    # ↳ ∂J/∂wⱼ = (2/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾) * xⱼ⁽ⁱ⁾
                    #   This measures how much the error changes when we adjust wⱼ
                    dw[j] += (2.0 / m) * error * X[i, j]
                
                # Gradient for bias
                # ↳ ∂J/∂b = (2/m) Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)
                #   Bias gradient doesn't depend on input features
                db += (2.0 / m) * error
            
            # ────────────────────────────────────────────────────────
            # STEP 4: Update Parameters (Gradient Descent Step)
            # ────────────────────────────────────────────────────────
            # ↳ θ := θ - α * ∇θ J(θ)
            #   Move parameters in direction that reduces loss
            
            for j in range(n):
                self.weights[j] -= self.lr * dw[j]
            
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for new data.
        
        ↳ Implements: ŷ = Xw + b
        
        Args:
            X: Feature matrix of shape (m samples, n features)
            
        Returns:
            Predictions of shape (m,)
        """
        m = X.shape[0]
        predictions = np.zeros(m)
        
        for i in range(m):
            predictions[i] = self._compute_prediction(X, i)
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination.
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res = Σ(y - ŷ)² (residual sum of squares)
        - SS_tot = Σ(y - ȳ)² (total sum of squares)
        
        Interpretation:
        - R² = 1.0: Perfect fit
        - R² = 0.0: Model predicts mean no better than horizontal line
        - R² < 0.0: Model worse than predicting mean
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        
        # Residual sum of squares
        ss_res = 0.0
        for i in range(len(y)):
            ss_res += (y[i] - y_pred[i]) ** 2
        
        # Total sum of squares
        y_mean = np.mean(y)
        ss_tot = 0.0
        for i in range(len(y)):
            ss_tot += (y[i] - y_mean) ** 2
        
        return 1.0 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Return learned parameters."""
        return {
            'weights': self.weights.copy() if self.weights is not None else None,
            'bias': self.bias,
            'loss_history': self.loss_history.copy()
        }


def generate_sample_data(
    n_samples: int = 100,
    n_features: int = 1,
    true_weights: np.ndarray = None,
    true_bias: float = 5.0,
    noise_std: float = 1.0,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data for testing.
    
    Creates data following: y = X @ true_weights + true_bias + noise
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (ignored if true_weights provided)
        true_weights: Actual weights to use (generates random if None)
        true_bias: True intercept value
        noise_std: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    """
    np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, n_features)
    
    if true_weights is None:
        true_weights = np.random.randn(n_features)
    
    # Generate targets: y = Xw + b + ε
    y = X @ true_weights + true_bias + np.random.randn(n_samples) * noise_std
    
    return X, y


# ═══════════════════════════════════════════════════════════════
# MAIN: Demonstration and Testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("LINEAR REGRESSION - LOOP IMPLEMENTATION (Educational)")
    print("=" * 70)
    
    # Generate simple 1D dataset
    print("\n📊 Generating synthetic data...")
    X, y = generate_sample_data(
        n_samples=50,
        n_features=1,
        true_weights=np.array([2.5]),
        true_bias=3.0,
        noise_std=0.5,
        random_seed=42
    )
    
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   True weights: [2.5]")
    print(f"   True bias: 3.0")
    
    # Create and train model
    print("\n🔧 Training model with gradient descent...")
    model = LinearRegressionLoop(
        learning_rate=0.1,
        n_iterations=1000
    )
    
    model.fit(X, y)
    
    # Display results
    print("\n✅ Training complete!")
    print(f"\n📈 Learned parameters:")
    print(f"   Weights: {model.weights}")
    print(f"   Bias: {model.bias:.4f}")
    
    print(f"\n📊 Performance metrics:")
    r2 = model.score(X, y)
    print(f"   R² Score: {r2:.4f}")
    print(f"   Final Loss: {model.loss_history[-1]:.6f}")
    print(f"   Initial Loss: {model.loss_history[0]:.6f}")
    print(f"   Improvement: {(1 - model.loss_history[-1]/model.loss_history[0])*100:.2f}%")
    
    # Test predictions
    print("\n🎯 Sample predictions:")
    test_points = np.array([[0.0], [1.0], [-1.0]])
    predictions = model.predict(test_points)
    
    for i, (x, pred) in enumerate(zip(test_points, predictions)):
        print(f"   X={x[0]:5.2f} → ŷ={pred:6.2f}")
    
    # Show convergence
    print("\n📉 Convergence (loss every 100 iterations):")
    for i in range(0, len(model.loss_history), 100):
        print(f"   Iteration {i:4d}: Loss = {model.loss_history[i]:.6f}")
    
    print("\n" + "=" * 70)
    print("✓ Loop implementation demonstration complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Run vectorized_implementation.py for 100x faster version")
    print("   2. Run sklearn_comparison.py to verify against industry standard")
    print("   3. Open notebooks/visualization.ipynb for interactive plots")
