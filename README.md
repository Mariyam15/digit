# DIGIT

A **Neural Network-based Digit Classifier** built from scratch using Python and NumPy.
This project trains a deep feedforward neural network on the MNIST dataset to recognize handwritten digits (0–9).

---

##  Project Info

* **Repository Name:** `digit`
* **Main File:** `digit_classifier.py`
* **Language:** Python
* **Dataset:** MNIST (via `sklearn.datasets.fetch_openml`)

---

## 🚀 Features

*  Fully connected neural network (from scratch)
*  Multi-class classification (digits 0–9)
*  Configurable architecture (multiple hidden layers)
*  Supports:

  * Sigmoid & Tanh activation functions
  * MSE & Cross-Entropy loss
*  Mini-batch gradient descent training
*  Performance evaluation:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix
*  Visualization using Matplotlib & Seaborn

---

## 🛠️ Tech Stack

* Python
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## ▶️ How to Run

###  Install dependencies

```bash
pip install numpy matplotlib seaborn scikit-learn
```

###  Run the script

```bash
python digit_classifier.py
```

---

##  Model Architecture

Default architecture:

```
784 → 256 → 128 → 64 → 10
```

* Input layer: 784 (28×28 image pixels)
* Output layer: 10 classes (digits 0–9)
* Softmax activation at output

---

##  Experiments Included

###  Learning Rate Comparison

* 0.1
* 0.01
* 0.001

###  Activation Function Test

* Tanh vs Sigmoid

###  Architecture Variants

* Deep network: `784 → 512 → 256 → 128 → 10`
* Smaller network: `784 → 128 → 64 → 32 → 10`

###  Model Complexity Comparison

* Simple: `784 → 10`
* Medium: `784 → 64 → 10`
* Better: `784 → 128 → 10`

---

##  Evaluation Metrics

*  Accuracy
*  Precision (Macro Average)
*  Recall (Macro Average)
*  F1 Score (Macro Average)
*  Confusion Matrix (visualized)

---

##  Sample Output

* Random digit visualization
* Training logs per epoch:

```
Epoch 1: Loss=0.1234, Accuracy=0.89
```

* Confusion Matrix Heatmap

---

##  Known Issues

* No GPU acceleration (CPU only)
* Training can be slow
* No model saving/loading
* Minor code artifacts (e.g. `+++++++` line should be removed)

---

##  Future Improvements

* Add model saving/loading
* Implement optimizers (Adam, RMSprop)
* Add dropout regularization
* Use PyTorch or TensorFlow backend
* Improve training speed

---

##  License

Free to use for learning and educational purposes.

---

##  Author
MARIYAM FATIMA
https://github.com/Mariyam15/digit


---
