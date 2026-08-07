# Machine Learning Algorithms: Deep Dive Repository

A comprehensive collection of machine learning algorithms explained with **dual-track methodology**: Track A (Black Box API) for rapid prototyping and Track B (Open Heart Surgery) for deep mathematical understanding and from-scratch implementation.

## 📚 Repository Structure

```
ml-algorithms-deep-dive/
├── README.md                    # This file
├── linear-regression/           # ✓ Complete
│   ├── README.md               # Full 8-section explanation
│   ├── src/
│   │   ├── loop_implementation.py
│   │   ├── vectorized_implementation.py
│   │   └── sklearn_comparison.py
│   ├── notebooks/
│   │   └── visualization.ipynb
│   └── tests/
│       └── test_implementation.py
├── logistic-regression/         # Next
├── k-nearest-neighbors/         # Planned
├── decision-trees/              # Planned
├── random-forests/              # Planned
├── gradient-boosting/           # Planned
├── support-vector-machines/     # Planned
├── k-means-clustering/          # Planned
├── pca/                         # Planned
└── neural-networks/             # Planned
```

## 🎯 Dual-Track Explanation Contract

Every algorithm follows the **8-section structure**:

### TRACK A — "THE BLACK BOX"
High-level analogy, scikit-learn / PyTorch API one-liner, when to use it, when NOT to use it. Written for an engineer who needs to ship something in 10 minutes.

### TRACK B — "THE OPEN HEART SURGERY"
Full mathematical derivation, from-scratch implementation, hardware-level behavior. Written for an engineer who needs to debug, optimize, or extend it at 2 AM.

## 📋 Mandatory 8-Section Structure

1. **Intuition First** - The "Bar Napkin" explanation with concrete analogies
2. **Black-Box API** - 5-line scikit-learn version with runnable code
3. **The Mathematical Engine** - Formal math, derivations, worked numerical examples
4. **Bare-Metal Implementation** - From-scratch NumPy implementation with inline math references
5. **Visual Explanation** - Matplotlib visualizations, ASCII diagrams, interactive concepts
6. **Hardware Sympathy & Complexity** - Big-O, CPU/GPU behavior, memory hierarchy
7. **Failure Modes & Production Telemetry** - What fails, why, detection, fixes
8. **Real-World Integration** - Modern system architecture, where it fits in production

## ✅ Completed Algorithms

| Algorithm | Status | Sections | Implementations | Visualizations |
|-----------|--------|----------|-----------------|----------------|
| Linear Regression | ✅ Complete | 8/8 | Loop + Vectorized + Normal Eq | 4 plots + ASCII |

## 🚧 In Progress

| Algorithm | Status | ETA |
|-----------|--------|-----|
| Logistic Regression | 📝 Writing | Next |
| K-Nearest Neighbors | ⏳ Planned | - |
| Decision Trees | ⏳ Planned | - |

## 🛠️ How to Use This Repository

### For Rapid Prototyping (Track A)
```bash
cd linear-regression
python src/sklearn_comparison.py
```

### For Deep Understanding (Track B)
```bash
cd linear-regrosis
python src/loop_implementation.py    # Start here for clarity
python src/vectorized_implementation.py  # Then optimize
jupyter notebook notebooks/visualization.ipynb
```

### For Testing
```bash
cd linear-regression
pytest tests/test_implementation.py -v
```

## 📊 Comparison Framework

Each algorithm includes:
- **Complexity tables** (time/space for train vs inference)
- **Hardware analysis** (CPU cache, GPU parallelization, memory bandwidth)
- **Failure mode catalog** with telemetry checklists
- **Production integration patterns** with architectural diagrams

## 🤝 Contributing

This repository follows a strict format to ensure consistency:

1. Every algorithm MUST have all 8 sections
2. All code must be runnable (no pseudocode)
3. Math must include "Plain English" translations
4. Implementations must reference equations by name
5. Visualizations must have clear labels and legends

## 📖 Learning Path

**Beginner:** Start with Track A sections → Run sklearn examples → Modify hyperparameters

**Intermediate:** Read Track B math → Implement from scratch → Compare with sklearn

**Advanced:** Optimize implementations → Add features → Contribute new algorithms

## 🔧 Requirements

```bash
pip install numpy pandas matplotlib scikit-learn pytest jupyter
```

Optional for GPU acceleration:
```bash
pip install torch torchvision torchaudio
```

## 📝 License

MIT License - Feel free to use in your projects, courses, or production systems.

## 👨‍💻 About

Created for senior software engineers who want to build genuine mathematical intuition, not just call APIs. Each algorithm is explained as if teaching a colleague over coffee, then dissected as if preparing for a 2 AM production incident.

---

**Last Updated:** $(date +%Y-%m-%d)
**Total Algorithms:** 1 complete, 9 planned
