# Generate a complete All Machine Learning Models Colab Notebook

try:
    import nbformat as nbf
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat"])
    import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source.strip()))


def code(source):
    cells.append(nbf.v4.new_code_cell(source.strip()))


# ----------------------------------------------------------------------------------
# TITLE
# ----------------------------------------------------------------------------------

md(r'''
# 🤖 All Machine Learning Models — Colab Project

This notebook demonstrates a complete machine learning project using many popular ML models.

## Included Tasks

1. Classification
2. Regression
3. Clustering
4. Dimensionality Reduction
5. Model Comparison
6. Best Model Selection

## Models Used

### Classification

- Logistic Regression
- Ridge Classifier
- SGD Classifier
- KNN Classifier
- Decision Tree Classifier
- Random Forest Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- Hist Gradient Boosting Classifier
- AdaBoost Classifier
- XGBoost Classifier
- LightGBM Classifier
- Support Vector Classifier
- Gaussian Naive Bayes
- MLP Classifier

### Regression

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Bayesian Ridge
- Huber Regressor
- SGD Regressor
- KNN Regressor
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- Hist Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- SVR
- Linear SVR
- MLP Regressor

### Clustering

- KMeans
- MiniBatchKMeans
- DBSCAN
- Agglomerative Clustering
- Gaussian Mixture Model
- Birch

### Dimensionality Reduction

- PCA
''')


# ----------------------------------------------------------------------------------
# SETUP
# ----------------------------------------------------------------------------------

md(r'''
## 1. Install Required Libraries

This cell installs the required packages.
''')


code(r'''
!pip install -q scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm
''')


code(r'''
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
    adjusted_rand_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    RidgeClassifier,
    Lasso,
    ElasticNet,
    BayesianRidge,
    HuberRegressor,
    SGDClassifier,
    SGDRegressor
)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor
)

from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# Optional boosting libraries
xgb_available = False
lgbm_available = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    lgbm_available = True
except Exception:
    pass


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_style("whitegrid")

print("Setup completed.")
print("XGBoost available:", xgb_available)
print("LightGBM available:", lgbm_available)
''')


# ----------------------------------------------------------------------------------
# CLASSIFICATION
# ----------------------------------------------------------------------------------

md(r'''
## 2. Classification Project

We use the **Breast Cancer Wisconsin dataset** for classification.

Target:

- 0 = Malignant
- 1 = Benign

We will train many classification models and compare their accuracy.
''')


code(r'''
class_data = load_breast_cancer()

X_class = pd.DataFrame(class_data.data, columns=class_data.feature_names)
y_class = pd.Series(class_data.target, name="target")

print("Classification dataset shape:", X_class.shape)
print("\nTarget distribution:")
print(y_class.value_counts())

X_class.head()
''')


code(r'''
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.20,
    stratify=y_class,
    random_state=RANDOM_STATE
)

print("Training shape:", X_train_class.shape)
print("Testing shape:", X_test_class.shape)
''')


code(r'''
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Ridge Classifier": RidgeClassifier(max_iter=5000),
    "SGD Classifier": SGDClassifier(max_iter=2000, random_state=RANDOM_STATE),
    "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting Classifier": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "AdaBoost Classifier": AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE),
    "Support Vector Classifier": SVC(probability=True, random_state=RANDOM_STATE),
    "Gaussian Naive Bayes": GaussianNB(),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE)
}

if xgb_available:
    classifiers["XGBoost Classifier"] = XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

if lgbm_available:
    classifiers["LightGBM Classifier"] = LGBMClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbose=-1
    )

print("Total classification models:", len(classifiers))
''')


code(r'''
trained_classifiers = {}
class_results = []

for name, model in classifiers.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    pipe.fit(X_train_class, y_train_class)

    train_pred = pipe.predict(X_train_class)
    test_pred = pipe.predict(X_test_class)

    trained_classifiers[name] = pipe

    class_results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train_class, train_pred),
        "Test Accuracy": accuracy_score(y_test_class, test_pred),
        "F1 Score": f1_score(y_test_class, test_pred, average="weighted", zero_division=0)
    })

class_results_df = pd.DataFrame(class_results)
class_results_df = class_results_df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

class_results_df
''')


code(r'''
plt.figure(figsize=(10, 6))

sns.barplot(
    data=class_results_df,
    x="Test Accuracy",
    y="Model",
    palette="viridis"
)

plt.title("Classification Model Comparison")
plt.xlabel("Test Accuracy")
plt.ylabel("Model")
plt.xlim(0, 1)

plt.tight_layout()
plt.show()
''')


code(r'''
best_class_model_name = class_results_df.iloc[0]["Model"]
best_class_model = trained_classifiers[best_class_model_name]

best_class_pred = best_class_model.predict(X_test_class)

print("Best Classification Model:", best_class_model_name)
print("\nClassification Report:")
print(classification_report(y_test_class, best_class_pred))

cm = confusion_matrix(y_test_class, best_class_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
''')


# ----------------------------------------------------------------------------------
# REGRESSION
# ----------------------------------------------------------------------------------

md(r'''
## 3. Regression Project

We use the **Diabetes dataset** for regression.

Target:

- Disease progression after one year

We will train many regression models and compare them using:

- R² Score
- MAE
- RMSE
''')


code(r'''
reg_data = load_diabetes()

X_reg = pd.DataFrame(reg_data.data, columns=reg_data.feature_names)
y_reg = pd.Series(reg_data.target, name="target")

print("Regression dataset shape:", X_reg.shape)

X_reg.head()
''')


code(r'''
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("Training shape:", X_train_reg.shape)
print("Testing shape:", X_test_reg.shape)
''')


code(r'''
regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=10000),
    "ElasticNet Regression": ElasticNet(max_iter=10000),
    "Bayesian Ridge": BayesianRidge(),
    "Huber Regressor": HuberRegressor(max_iter=2000),
    "SGD Regressor": SGDRegressor(max_iter=2000, random_state=RANDOM_STATE),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Hist Gradient Boosting Regressor": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(),
    "Linear SVR": LinearSVR(max_iter=10000, random_state=RANDOM_STATE),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=RANDOM_STATE)
}

if xgb_available:
    regressors["XGBoost Regressor"] = XGBRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE
    )

if lgbm_available:
    regressors["LightGBM Regressor"] = LGBMRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        verbose=-1
    )

print("Total regression models:", len(regressors))
''')


code(r'''
trained_regressors = {}
reg_results = []

for name, model in regressors.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", model)
    ])

    pipe.fit(X_train_reg, y_train_reg)

    train_pred = pipe.predict(X_train_reg)
    test_pred = pipe.predict(X_test_reg)

    trained_regressors[name] = pipe

    reg_results.append({
        "Model": name,
        "Train R2": r2_score(y_train_reg, train_pred),
        "Test R2": r2_score(y_test_reg, test_pred),
        "MAE": mean_absolute_error(y_test_reg, test_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test_reg, test_pred))
    })

reg_results_df = pd.DataFrame(reg_results)
reg_results_df = reg_results_df.sort_values("Test R2", ascending=False).reset_index(drop=True)

reg_results_df
''')


code(r'''
plt.figure(figsize=(10, 6))

sns.barplot(
    data=reg_results_df,
    x="Test R2",
    y="Model",
    palette="coolwarm"
)

plt.title("Regression Model Comparison")
plt.xlabel("Test R² Score")
plt.ylabel("Model")
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.tight_layout()
plt.show()
''')


code(r'''
best_reg_model_name = reg_results_df.iloc[0]["Model"]
best_reg_model = trained_regressors[best_reg_model_name]

best_reg_pred = best_reg_model.predict(X_test_reg)

print("Best Regression Model:", best_reg_model_name)
print("\nTest R² Score:", round(r2_score(y_test_reg, best_reg_pred), 4))
print("Test MAE:", round(mean_absolute_error(y_test_reg, best_reg_pred), 4))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test_reg, best_reg_pred)), 4))

plt.figure(figsize=(6, 5))

plt.scatter(y_test_reg, best_reg_pred, alpha=0.6)
plt.plot(
    [y_test_reg.min(), y_test_reg.max()],
    [y_test_reg.min(), y_test_reg.max()],
    "r--"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.tight_layout()
plt.show()
''')


# ----------------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------------

md(r'''
## 4. Clustering Project

We use the **Iris dataset** for clustering.

Even though Iris has labels, we will use clustering algorithms in unsupervised mode and then compare the clusters with the original labels using:

- Silhouette Score
- Adjusted Rand Index
''')


code(r'''
cluster_data = load_iris()

X_cluster = pd.DataFrame(cluster_data.data, columns=cluster_data.feature_names)
y_cluster = cluster_data.target

X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

print("Clustering dataset shape:", X_cluster.shape)

X_cluster.head()
''')


code(r'''
cluster_models = {
    "KMeans": KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10),
    "MiniBatchKMeans": MiniBatchKMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10),
    "DBSCAN": DBSCAN(eps=0.5, min_samples=5),
    "AgglomerativeClustering": AgglomerativeClustering(n_clusters=3),
    "GaussianMixture": GaussianMixture(n_components=3, random_state=RANDOM_STATE),
    "Birch": Birch(n_clusters=3)
}

cluster_results = []

for name, model in cluster_models.items():
    labels = model.fit_predict(X_cluster_scaled)

    unique_labels = np.unique(labels)
    non_noise_labels = [label for label in unique_labels if label != -1]

    if len(non_noise_labels) >= 2:
        sil_score = silhouette_score(X_cluster_scaled, labels)
    else:
        sil_score = np.nan

    ari_score = adjusted_rand_score(y_cluster, labels)

    cluster_results.append({
        "Model": name,
        "Number of Clusters": len(non_noise_labels),
        "Silhouette Score": sil_score,
        "Adjusted Rand Index": ari_score
    })

cluster_results_df = pd.DataFrame(cluster_results)
cluster_results_df
''')


code(r'''
plt.figure(figsize=(10, 5))

sns.barplot(
    data=cluster_results_df,
    x="Model",
    y="Adjusted Rand Index",
    palette="magma"
)

plt.title("Clustering Model Comparison")
plt.xlabel("Model")
plt.ylabel("Adjusted Rand Index")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
''')


# ----------------------------------------------------------------------------------
# PCA
# ----------------------------------------------------------------------------------

md(r'''
## 5. Dimensionality Reduction Using PCA

We reduce the Iris dataset to 2 dimensions and visualize the classes.

We also compare true labels with KMeans cluster labels.
''')


code(r'''
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)

kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y_cluster,
    cmap="viridis",
    alpha=0.7
)
axes[0].set_title("PCA: True Iris Classes")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    alpha=0.7
)
axes[1].set_title("PCA: KMeans Clusters")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

plt.tight_layout()
plt.show()
''')


# ----------------------------------------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------------------------------------

md(r'''
## 6. Final Summary

This notebook demonstrated a complete machine learning project using many different models.

### Classification

We compared multiple classifiers on the Breast Cancer dataset.

Best Classification Model:

Check the classification results table above.

### Regression

We compared multiple regressors on the Diabetes dataset.

Best Regression Model:

Check the regression results table above.

### Clustering

We compared multiple clustering algorithms on the Iris dataset.

### Dimensionality Reduction

We used PCA to visualize the Iris dataset in 2D.

---

## Next Improvements

You can improve this project by:

- Adding cross-validation
- Adding hyperparameter tuning with GridSearchCV or Optuna
- Adding feature selection
- Adding more datasets
- Adding deep learning models using TensorFlow or PyTorch
- Adding model saving with joblib
- Adding Streamlit deployment
''')


# ----------------------------------------------------------------------------------
# SAVE NOTEBOOK
# ----------------------------------------------------------------------------------

nb["cells"] = cells

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11"
    },
    "colab": {
        "name": "All_Machine_Learning_Models_Colab_Project.ipynb",
        "provenance": []
    }
}

output_file = "All_Machine_Learning_Models_Colab_Project.ipynb"

with open(output_file, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created {output_file}")

try:
    from google.colab import files
    files.download(output_file)
    print("Download started. Upload this notebook to Google Colab and run Run All.")
except Exception:
    print("Notebook file created. Upload it manually to Google Colab.")