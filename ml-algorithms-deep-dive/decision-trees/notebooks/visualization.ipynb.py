"""
Decision Trees Visualization Notebook

Demonstrates:
1. Tree structure visualization
2. Decision boundary plotting
3. Feature importance analysis
4. Overfitting analysis across depths
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import sys
sys.path.append('src')

from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

print("Generating Decision Tree Visualizations...")
print("="*70)

# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 1: Tree Structure Diagram
# ═══════════════════════════════════════════════════════════════

def plot_tree_structure(node, depth=0, pos_x=0, pos_y=0, ax=None, 
                        horizontal_spacing=4, vertical_spacing=2):
    """Recursively draw tree structure."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, -15)
        ax.axis('off')
    
    if node.is_leaf:
        # Draw leaf node (circle)
        circle = Circle((pos_x, pos_y), 0.6, color='lightgreen', 
                       edgecolor='darkgreen', linewidth=2)
        ax.add_patch(circle)
        label = f'{node.value}'
        ax.text(pos_x, pos_y, label, ha='center', va='center', 
               fontsize=9, fontweight='bold', color='darkgreen')
    else:
        # Draw internal node (rectangle)
        rect = Rectangle((pos_x-0.8, pos_y-0.4), 1.6, 0.8, 
                        color='lightblue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(rect)
        label = f'X[{node.feature_index}]≤{node.threshold:.2f}'
        ax.text(pos_x, pos_y, label, ha='center', va='center', 
               fontsize=8, fontweight='bold', color='darkblue')
        
        # Draw connections to children
        left_x = pos_x - horizontal_spacing / (depth + 1)
        right_x = pos_x + horizontal_spacing / (depth + 1)
        child_y = pos_y + vertical_spacing
        
        if node.left:
            ax.plot([pos_x, left_x], [pos_y, child_y], 'b-', linewidth=1.5, alpha=0.7)
            plot_tree_structure(node.left, depth+1, left_x, child_y, ax=ax,
                              horizontal_spacing=horizontal_spacing, 
                              vertical_spacing=vertical_spacing)
        
        if node.right:
            ax.plot([pos_x, right_x], [pos_y, child_y], 'r-', linewidth=1.5, alpha=0.7)
            plot_tree_structure(node.right, depth+1, right_x, child_y, ax=ax,
                              horizontal_spacing=horizontal_spacing, 
                              vertical_spacing=vertical_spacing)
    
    return ax


# Create simple dataset for visualization
np.random.seed(42)
X_viz = np.random.randn(100, 2)
y_viz = ((X_viz[:, 0] > 0) & (X_viz[:, 1] > 0)).astype(int)

clf_viz = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
clf_viz.fit(X_viz, y_viz)

fig1 = plt.figure(figsize=(14, 8))
ax1 = fig1.add_subplot(111)
plot_tree_structure(clf_viz.root, ax=ax1)
ax1.set_title('Decision Tree Structure (Depth=3)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/tree_structure.png', dpi=150, bbox_inches='tight')
print("✓ Created: visualizations/tree_structure.png")
plt.close()


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 2: Decision Boundaries
# ═══════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate 2D dataset
np.random.seed(42)
n_samples = 300
X_boundary = np.random.randn(n_samples, 2)
y_boundary = ((X_boundary[:, 0]**2 + X_boundary[:, 1]**2) < 2).astype(int)

# Fit model
clf_boundary = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_boundary.fit(X_boundary, y_boundary)

# Create mesh grid
x_min, x_max = X_boundary[:, 0].min() - 0.5, X_boundary[:, 0].max() + 0.5
y_min, y_max = X_boundary[:, 1].min() - 0.5, X_boundary[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on grid
Z = clf_boundary.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision regions
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=20)
scatter = axes[0].scatter(X_boundary[:, 0], X_boundary[:, 1], c=y_boundary, 
                         cmap='viridis', edgecolors='black', linewidth=0.5, s=50)
axes[0].set_xlabel('Feature 0', fontsize=12)
axes[0].set_ylabel('Feature 1', fontsize=12)
axes[0].set_title('Decision Boundaries in 2D Feature Space', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Add annotation
axes[0].annotate('Axis-aligned splits\ncreate rectangular regions', 
                xy=(0.5, 0.5), xytext=(0.7, 0.8),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

# Plot feature importance
if hasattr(clf_boundary, 'feature_importances_'):
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, 2))
    bars = axes[1].barh(['Feature 0', 'Feature 1'], 
                       clf_boundary.feature_importances_, color=colors)
    axes[1].set_xlabel('Importance Score', fontsize=12)
    axes[1].set_title('Feature Importances', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    
    # Add value labels
    for bar, val in zip(bars, clf_boundary.feature_importances_):
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    axes[1].set_xlim(0, 1.1)
    axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/decision_boundaries.png', dpi=150, bbox_inches='tight')
print("✓ Created: visualizations/decision_boundaries.png")
plt.close()


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 3: Overfitting Analysis
# ═══════════════════════════════════════════════════════════════

# Generate dataset with noise
np.random.seed(42)
n_train = 200
n_test = 100

X_overfit = np.random.randn(n_train + n_test, 5)
true_relationship = (X_overfit[:, 0] > 0).astype(int) ^ (X_overfit[:, 1] > 0.5).astype(int)
noise = np.random.rand(n_train + n_test) < 0.1  # 10% label noise
y_overfit = true_relationship.copy()
y_overfit[noise] = 1 - y_overfit[noise]

X_train, X_test = X_overfit[:n_train], X_overfit[n_train:]
y_train, y_test = y_overfit[:n_train], y_overfit[n_train:]

# Train trees with different depths
max_depths = range(1, 15)
train_scores = []
test_scores = []

for depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

fig3, ax = plt.subplots(figsize=(10, 6))

ax.plot(max_depths, train_scores, 'bo-', linewidth=2, markersize=8, label='Training Accuracy')
ax.plot(max_depths, test_scores, 'rs-', linewidth=2, markersize=8, label='Testing Accuracy')

# Highlight optimal depth
optimal_idx = np.argmax(test_scores)
optimal_depth = max_depths[optimal_idx]
ax.axvline(optimal_depth, color='green', linestyle='--', linewidth=2, 
           label=f'Optimal Depth = {optimal_depth}')

# Shade overfitting region
ax.fill_between(max_depths, test_scores, train_scores, alpha=0.2, color='red', 
               label='Overfitting Gap')

ax.set_xlabel('Maximum Tree Depth', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Decision Tree: Training vs Testing Performance\n(Illustrating Overfitting)', 
            fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(max_depths)

plt.tight_layout()
plt.savefig('visualizations/overfitting_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Created: visualizations/overfitting_analysis.png")
plt.close()


# ═══════════════════════════════════════════════════════════════
# VISUALIZATION 4: Regression Tree Example
# ═══════════════════════════════════════════════════════════════

fig4, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate regression data
np.random.seed(42)
X_reg = np.sort(np.random.randn(200, 1), axis=0) * 2
y_reg = np.sin(X_reg.flatten()) + np.random.randn(200) * 0.1

# Fit regression tree
reg_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
reg_tree.fit(X_reg, y_reg)

# Plot data and predictions
X_plot = np.linspace(-3, 3, 500).reshape(-1, 1)
y_pred = reg_tree.predict(X_plot)

axes[0].scatter(X_reg, y_reg, alpha=0.6, color='steelblue', s=40, label='Training Data')
axes[0].plot(X_plot, y_pred, 'r-', linewidth=2.5, label='Tree Prediction')
axes[0].plot(X_plot, np.sin(X_plot), 'g--', linewidth=2, alpha=0.7, label='True Function (sin)')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('y', fontsize=12)
axes[0].set_title('Decision Tree Regression: Step Function Approximation', 
                 fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Show step nature
axes[0].annotate('Piecewise constant\npredictions', 
                xy=(0, reg_tree.predict([[0]])[0]),
                xytext=(1.5, reg_tree.predict([[0]])[0] + 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='darkred')

# Residual plot
residuals = y_reg.flatten() - reg_tree.predict(X_reg).flatten()
axes[1].scatter(reg_tree.predict(X_reg), residuals, alpha=0.6, color='coral', s=40)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
axes[1].set_xlabel('Predicted Value', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Residual Analysis', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/regression_example.png', dpi=150, bbox_inches='tight')
print("✓ Created: visualizations/regression_example.png")
plt.close()


# ═══════════════════════════════════════════════════════════════
# ASCII DIAGRAM: How Tree Makes Decisions
# ═══════════════════════════════════════════════════════════════

ascii_diagram = """
======================================================================
DECISION TREE INFERENCE FLOW
======================================================================

Input Sample: [Petal Length=4.5, Petal Width=1.3, ...]
                      │
                      ▼
            ┌─────────────────┐
            │ Petal Width ≤   │
            │     0.8?        │
            └────────┬────────┘
                     │
          ┌──────────┴──────────┐
          │ NO                  │ YES
          ▼                     ▼
    ┌─────────────┐       ┌─────────────┐
    │ Petal Length│       │ Class:      │
    │   ≤ 2.5?    │       │ Setosa      │
    └──────┬──────┘       └─────────────┘
           │
    ┌──────┴──────┐
    │ YES         │ NO
    ▼             ▼
┌─────────┐   ┌──────────┐
│ Class:  │   │ Class:   │
│ Versicol│   │ Virginica│
└─────────┘   └──────────┘

→ Path length = tree depth for this sample
→ Each split eliminates ~50% of remaining classes (ideally)
→ Axis-aligned splits create hyperrectangular decision regions
======================================================================
"""

print(ascii_diagram)

print("\n" + "="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files:")
print("  • visualizations/tree_structure.png")
print("  • visualizations/decision_boundaries.png")
print("  • visualizations/overfitting_analysis.png")
print("  • visualizations/regression_example.png")
print("\nTo view: open the PNG files or run this notebook in Jupyter.")
