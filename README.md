# Income Prediction usingMulti-Layer Perceptron (MLP) with framework

This project implements a Feedforward Neural Network (MLP) using the **PyTorch** framework to predict whether an individual's income exceeds $50,000 per year based on census data.

## Methodology

The pipeline involves a standard machine learning workflow, from data ingestion to model evaluation:

### 1. Data Preprocessing
* **Dataset:** Adult Census Income dataset from the UCI Machine Learning Repository.
* **Cleaning:** Removed missing values (marked as "?") and performed feature-target separation.
* **Encoding:** * **Target:** Binary encoding (1 for `>50K`, 0 for `<=50K`).
    * **Features:** One-hot encoding for categorical variables.
* **Scaling:** Applied `StandardScaler` to normalize numerical features for stable gradient descent.
* **Splitting:** 80/20 train-test split with stratification to maintain class proportions.
###     Why Feature Scaling?

Feature scaling was critical to:
- Improve convergence speed  
- Prevent unstable gradient updates  
- Ensure efficient optimization  

### 2. Model Architecture
The model is a 3-layer fully connected network:
* **Input Layer:** Dynamic size based on encoded features.
* **Hidden Layer 1:** 128 units with ReLU activation.
* **Hidden Layer 2:** 64 units with ReLU activation.
* **Output Layer:** 1 unit with Sigmoid activation for probability output.

### 3. Training Configuration
* **Loss Function:** Binary Cross-Entropy Loss (`BCELoss`).
* **Optimizer:** Adam Optimizer with a learning rate of `0.01`.
* **Epochs:** 50 full passes over the training set.

---

## Results

The model was evaluated on a held-out test set. To account for class imbalance, a custom classification threshold of **0.35** was applied to the probability outputs.

### Performance Metrics

| Metric    | Value  |
| :---      | :---   |
| **Accuracy** | 83.69% |
| **Precision** | 65.02% |
| **Recall** | 74.63% |
| **F1-Score** | 69.50% |

### Interpretation

- **83.69% Accuracy** indicates strong overall classification performance.  
- **Precision (0.6502)** shows moderate control over false positives.  
- **Recall (0.7463)** demonstrates solid ability to capture positive instances.  
- **F1-Score (0.6950)** reflects a balanced trade-off between precision and recall.  

The model achieves a stable and competitive classification performance suitable for real-world applications.

---

## Key Takeaways

- Implemented a neural network using a deep learning framework.  
- Applied proper preprocessing techniques including feature scaling.  
- Achieved stable convergence through backpropagation.  
- Evaluated performance using multiple classification metrics.  
- Built a complete end-to-end ML pipeline.  

