# K-Means Clustering: Complete Technical Explanation

## SECTION 1 — INTUITION FIRST (The "Bar Napkin" Explanation)

K-Means is like organizing a crowded party into K conversation groups. Imagine 100 people at a conference mixer, and you want to form 5 discussion circles. You randomly pick 5 people as "group starters." Everyone else walks to the starter they're closest to physically. Now each group has members. You notice some groups are lopsided, so you ask each group to pick a new center person—the one who's most "average" in position relative to all members. People re-evaluate: "Wait, I'm actually closer to Group 3's new center!" and shuffle around. This repeats until nobody wants to switch groups anymore. That's K-Means: iteratively assigning points to nearest centers, then recomputing centers until convergence.

```
Initial State (K=3):          After 1st Iteration:        Converged:
                                 
    *    ●                      *    ●                     *    ●
         ▲                       ▲  ▲                      ▲▲ ▲
   *  ●   ○  ●              *  ●   ○  ●                *  ●   ○  ●
      \  |  /                   \ | /                      \|/
       \ | /                     \|/                        ●●●
        \|/                       ●                         ***
         ●                       **                         
                                    
    Centroids: ▲▲▲            Centroids: ▲▲▲             Centroids: ▲▲▲
    Points: * ● ○             Reassigned                 Stable clusters
```

**Goal:** Partition $m$ samples into $K$ clusters minimizing within-cluster variance: $\min \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2$

## SECTION 2 — BLACK-BOX API (Track A)

```python
from sklearn.cluster import KMeans
import numpy as np

# Sample data: customer spending scores vs annual income
X = np.array([
    [15, 80], [20, 75], [18, 85],  # Cluster 1: Low income, high spending
    [80, 20], [90, 15], [85, 25],  # Cluster 2: High income, low spending
    [50, 50], [55, 45], [45, 55],  # Cluster 3: Balanced
])

# Fit K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print(f"Cluster assignments: {labels}")
print(f"Centroids:\n{centroids}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")

# Predict new customer
new_customer = np.array([[60, 40]])
prediction = kmeans.predict(new_customer)
print(f"New customer belongs to cluster: {prediction[0]}")
```

**Key Hyperparameters:**
| Parameter | Default | What It Controls |
|-----------|---------|------------------|
| `n_clusters` | 8 | Number of clusters K (must specify) |
| `init` | 'k-means++' | Initialization method (random vs smart) |
| `n_init` | 10 | Number of runs with different centroid seeds |
| `max_iter` | 300 | Maximum iterations per run |
| `tol` | 1e-4 | Convergence tolerance |

**When NOT to use K-Means:**
❌ Non-spherical clusters (moons, rings, spirals)
❌ Clusters of very different sizes
❌ Unknown K (use DBSCAN or hierarchical clustering)
❌ Categorical data (use K-Modes)
❌ Outliers present (use K-Medoids)

## SECTION 3 — THE MATHEMATICAL ENGINE (Track B, Part 1)

### Objective Function

Given data $X = \{x^{(1)}, x^{(2)}, ..., x^{(m)}\}$ where $x^{(i)} \in \mathbb{R}^n$, find $K$ centroids $\mu_1, \mu_2, ..., \mu_K$ and assignments $c^{(i)} \in \{1, 2, ..., K\}$ minimizing:

$$J(\mu, c) = \sum_{i=1}^m \|x^{(i)} - \mu_{c^{(i)}}\|^2 = \sum_{k=1}^K \sum_{i: c^{(i)}=k} \|x^{(i)} - \mu_k\|^2$$

**Plain English:** Sum of squared distances from each point to its assigned cluster center. We want points close to their center.

### Algorithm Steps

**Step 1: Initialize centroids** (K-Means++ algorithm)
- Pick first centroid randomly from data points
- For each subsequent centroid $j = 2, ..., K$:
  - Compute distance $D(x)$ from each point $x$ to nearest existing centroid
  - Select next centroid with probability proportional to $D(x)^2$

**Plain English:** K-Means++ spreads initial centroids apart, avoiding bad local minima.

**Step 2: Assignment step (E-step)**
$$c^{(i)} := \arg\min_{k} \|x^{(i)} - \mu_k\|^2$$

**Plain English:** Assign each point to the nearest centroid (Euclidean distance).

**Step 3: Update step (M-step)**
$$\mu_k := \frac{\sum_{i: c^{(i)}=k} x^{(i)}}{\sum_{i: c^{(i)}=k} 1}$$

**Plain English:** Move each centroid to the mean (average) of all points assigned to it.

**Step 4: Repeat** steps 2-3 until convergence (centroids don't move or max iterations reached).

### Worked Numerical Example

**Data:** 6 points in 1D: [1, 2, 3, 8, 9, 10], K=2

**Initialization (random):** $\mu_1 = 2$, $\mu_2 = 9$

**Iteration 1:**
- Assignments:
  - Point 1: $\|1-2\|=1 < \|1-9\|=8$ → Cluster 1
  - Point 2: $\|2-2\|=0 < \|2-9\|=7$ → Cluster 1
  - Point 3: $\|3-2\|=1 < \|3-9\|=6$ → Cluster 1
  - Point 8: $\|8-2\|=6 > \|8-9\|=1$ → Cluster 2
  - Point 9: $\|9-2\|=7 > \|9-9\|=0$ → Cluster 2
  - Point 10: $\|10-2\|=8 > \|10-9\|=1$ → Cluster 2
  
- Update centroids:
  - $\mu_1 = \text{mean}([1, 2, 3]) = 2$
  - $\mu_2 = \text{mean}([8, 9, 10]) = 9$

**Convergence:** Centroids unchanged → Stop!

**Final:** Cluster 1: {1,2,3}, Cluster 2: {8,9,10}

## SECTION 4 — BARE-METAL IMPLEMENTATION (Track B, Part 2)

```python
import numpy as np
from collections import Counter

class KMeansScratch:
    """
    K-Means Clustering from scratch using NumPy.
    
    Mathematical foundation:
    - Objective: min Σ||x⁽ⁱ⁾ - μ_c⁽ⁱ⁾||²
    - E-step: c⁽ⁱ⁾ := argmin_k ||x⁽ⁱ⁾ - μ_k||²
    - M-step: μ_k := mean of points in cluster k
    """
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, init='kmeans++'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        
    def _initialize_centroids(self, X):
        """
        Initialize centroids using K-Means++ algorithm.
        
        ↳ Probability of selection ∝ D(x)² where D(x) = distance to nearest centroid
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        m, n = X.shape
        
        if self.init == 'random':
            # Random initialization
            indices = np.random.choice(m, self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'kmeans++':
            # K-Means++ initialization
            centroids = []
            
            # First centroid: random point
            first_idx = np.random.randint(m)
            centroids.append(X[first_idx].copy())
            
            # Remaining centroids
            for _ in range(1, self.n_clusters):
                # Compute distances to nearest existing centroid
                distances = np.zeros(m)
                for i in range(m):
                    min_dist = float('inf')
                    for c in centroids:
                        dist = np.sum((X[i] - c) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                    distances[i] = min_dist
                
                # Select next centroid with probability proportional to D²
                probabilities = distances / np.sum(distances)
                cumulative_probs = np.cumsum(probabilities)
                r = np.random.rand()
                next_idx = np.searchsorted(cumulative_probs, r)
                centroids.append(X[next_idx].copy())
            
            return np.array(centroids)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X):
        """
        Assign each point to nearest centroid (E-step).
        
        ↳ c⁽ⁱ⁾ := argmin_k ||x⁽ⁱ⁾ - μ_k||²
        """
        m = X.shape[0]
        labels = np.zeros(m, dtype=int)
        
        for i in range(m):
            distances = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                # Euclidean distance squared
                distances[k] = np.sum((X[i] - self.centroids[k]) ** 2)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroids to mean of assigned points (M-step).
        
        ↳ μ_k := (Σ_{i: c⁽ⁱ⁾=k} x⁽ⁱ⁾) / (Σ_{i: c⁽ⁱ⁾=k} 1)
        """
        n = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n))
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # Empty cluster: reinitialize randomly
                new_centroids[k] = X[np.random.randint(X.shape[0])]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels):
        """
        Compute within-cluster sum of squares.
        
        ↳ J = Σ_k Σ_{i: c⁽ⁱ⁾=k} ||x⁽ⁱ⁾ - μ_k||²
        """
        inertia = 0.0
        for i in range(len(X)):
            inertia += np.sum((X[i] - self.centroids[labels[i]]) ** 2)
        return inertia
    
    def fit(self, X):
        """
        Fit K-Means clustering.
        
        ↳ Alternates between E-step and M-step until convergence
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # E-step: assign clusters
            labels = self._assign_clusters(X)
            
            # M-step: update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            centroid_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        
        self.n_iter_ = iteration + 1
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels)
        
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)
    
    def transform(self, X):
        """Transform X to cluster-distance space."""
        m = X.shape[0]
        distances = np.zeros((m, self.n_clusters))
        
        for i in range(m):
            for k in range(self.n_clusters):
                distances[i, k] = np.sqrt(np.sum((X[i] - self.centroids[k]) ** 2))
        
        return distances


# Vectorized version for production
class KMeansVectorized:
    """Fully vectorized K-Means using broadcasting."""
    
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def _initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        m = X.shape[0]
        
        # K-Means++ initialization (vectorized)
        centroids = []
        first_idx = np.random.randint(m)
        centroids.append(X[first_idx].copy())
        
        for _ in range(1, self.n_clusters):
            # Compute squared distances to all existing centroids (vectorized)
            distances = np.zeros(m)
            for c in centroids:
                dist = np.sum((X - c) ** 2, axis=1)
                distances = np.minimum(distances, dist)
            
            # Select next centroid
            probs = distances / np.sum(distances)
            next_idx = np.random.choice(m, p=probs)
            centroids.append(X[next_idx].copy())
        
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # E-step: compute all pairwise distances (vectorized)
            # Shape: (m, K)
            distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # M-step: update centroids (vectorized)
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if np.sum(mask) > 0:
                    new_centroids[k] = np.mean(X[mask], axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(X.shape[0])]
            
            # Check convergence
            shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids
            
            if shift < self.tol:
                self.n_iter_ = iteration + 1
                break
        
        self.n_iter_ = iteration + 1
        self.labels_ = labels
        self.inertia_ = np.sum((X - self.centroids[labels]) ** 2)
        
        return self
    
    def predict(self, X):
        distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score
    
    # Generate synthetic data
    X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
    
    # Test scratch implementation
    kmeans_scratch = KMeansScratch(n_clusters=4, random_state=42)
    kmeans_scratch.fit(X)
    
    print("=== K-MEANS SCRATCH IMPLEMENTATION ===")
    print(f"Iterations: {kmeans_scratch.n_iter_}")
    print(f"Inertia: {kmeans_scratch.inertia_:.2f}")
    print(f"Adjusted Rand Score: {adjusted_rand_score(y_true, kmeans_scratch.labels_):.3f}")
    
    # Test vectorized implementation
    kmeans_vec = KMeansVectorized(n_clusters=4, random_state=42)
    kmeans_vec.fit(X)
    
    print("\n=== K-MEANS VECTORIZED IMPLEMENTATION ===")
    print(f"Iterations: {kmeans_vec.n_iter_}")
    print(f"Inertia: {kmeans_vec.inertia_:.2f}")
    print(f"Adjusted Rand Score: {adjusted_rand_score(y_true, kmeans_vec.labels_):.3f}")
    
    # Compare with sklearn
    from sklearn.cluster import KMeans
    kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_sklearn.fit(X)
    
    print("\n=== SKLEARN (for reference) ===")
    print(f"Iterations: {kmeans_sklearn.n_iter_}")
    print(f"Inertia: {kmeans_sklearn.inertia_:.2f}")
    print(f"Adjusted Rand Score: {adjusted_rand_score(y_true, kmeans_sklearn.labels_):.3f}")
```

## SECTION 5 — VISUAL EXPLANATION

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate data
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# Fit model
kmeans = KMeansVectorized(n_clusters=4, random_state=42)
kmeans.fit(X)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial state (first iteration)
ax1 = axes[0, 0]
ax1.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=50)
initial_centroids = kmeans._initialize_centroids(X)
ax1.scatter(initial_centroids[:, 0], initial_centroids[:, 1], 
            c='red', s=200, marker='X', label='Initial Centroids')
ax1.set_title('Step 1: Initialize Centroids (K-Means++)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: After first iteration
ax2 = axes[0, 1]
# Simulate one iteration
temp_labels = kmeans.predict(X)
ax2.scatter(X[:, 0], X[:, 1], c=temp_labels, cmap='viridis', alpha=0.7, s=50)
ax2.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', s=200, marker='X', label='Centroids')
ax2.set_title('Step 2: Assign & Update (Iteration 1)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Final clustering
ax3 = axes[1, 0]
ax3.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7, s=50)
ax3.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
            c='red', s=200, marker='X', label='Final Centroids')
ax3.set_title(f'Final Clustering (K={kmeans.n_clusters})', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Elbow method
ax4 = axes[1, 1]
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeansVectorized(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

ax4.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax4.axvline(x=4, color='red', linestyle='--', label='Optimal K (Elbow)')
ax4.set_xlabel('Number of Clusters (K)', fontsize=12)
ax4.set_ylabel('Inertia (Within-cluster SS)', fontsize=12)
ax4.set_title('Elbow Method: Finding Optimal K', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/workspace/ml-algorithms-deep-dive/k-means/visualizations/kmeans_demo.png', dpi=150)
plt.show()
```

## SECTION 6 — HARDWARE SYMPATHY & COMPLEXITY

| Operation | Time Complexity | Space Complexity | Bottleneck |
|-----------|----------------|------------------|------------|
| **Training (per iteration)** | O(m × K × n) | O(K × n) | Memory bandwidth |
| **Inference (single point)** | O(K × n) | O(1) | Compute |
| **K-Means++ Init** | O(m × K × n) | O(K × n) | Memory bandwidth |

**Hardware Behavior:**
- **Memory Bandwidth Bound:** Each iteration reads entire dataset (m samples)
- **GPU Parallelization:** Embarrassingly parallel—distance computations independent
- **Cache Efficiency:** Process in batches fitting L2/L3 cache
- **Minimum Hardware:** Can cluster 1M samples on laptop with 8GB RAM

## SECTION 7 — FAILURE MODES & TELEMETRY

| Failure Mode | Detection | Fix |
|--------------|-----------|-----|
| **Bad Local Minimum** | High inertia across multiple runs | Use K-Means++, increase n_init |
| **Empty Clusters** | Some centroids never updated | Reinitialize empty centroids |
| **Wrong K Selection** | No clear elbow in inertia plot | Use silhouette score, domain knowledge |
| **Non-convergence** | Max iterations reached without stability | Increase max_iter, check tol |

**Telemetry Checklist:**
- Log inertia per iteration (should decrease monotonically)
- Track cluster sizes (watch for imbalance)
- Monitor centroid movement (convergence rate)
- Compute silhouette score for cluster quality

## SECTION 8 — REAL-WORLD INTEGRATION

**Use Case: Customer Segmentation Pipeline**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Transaction │     │  Feature     │     │   K-Means   │
│   Data      │────▶│  Engineering │────▶│ Clustering  │
│ (Raw Logs)  │     │  (RFM Score) │     │  (K=5)      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                      ┌─────────────────┐
                                      │ Marketing Action│
                                      │ • Cluster 0: VIP│
                                      │ • Cluster 1: At-│
                                      │   risk          │
                                      │ • Cluster 2: New│
                                      └─────────────────┘
```

**Modern Usage:** Often replaced by DBSCAN for irregular clusters or deep embedding clustering for high-dimensional data.
