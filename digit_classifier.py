import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

print("Loading dataset...")
data = fetch_openml('mnist_784', version=1, as_frame=False)
features = data['data']
labels = data['target'].astype(int)

def display_samples(features, labels, count=5):
    for i in range(count):
        idx = np.random.randint(0, len(features))
        plt.subplot(1, count, i + 1)
        plt.imshow(features[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Digit: {labels[idx]}")
        plt.axis('off')
    plt.show()

display_samples(features, labels)
features = features / 255.0
X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=60000, test_size=10000)

def act_sigmoid(z): return 1 / (1 + np.exp(-z))
def der_sigmoid(a): return a * (1 - a)
def act_tanh(z): return np.tanh(z)
def der_tanh(z): return 1 - np.tanh(z) ** 2
def softmax_fn(z):
    exps = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
def loss_mse(actual, pred): return np.mean((actual - pred) ** 2)
def loss_ce(actual, pred):
    eps = 1e-12
    pred = np.clip(pred, eps, 1. - eps)
    return -np.mean(np.sum(actual * np.log(pred), axis=1))

def encode_labels(y_vals, cls=10):
    encoded = np.zeros((len(y_vals), cls))
    encoded[np.arange(len(y_vals)), y_vals] = 1
    return encoded

y_train_enc = encode_labels(y_train)
y_test_enc = encode_labels(y_test)

def init_weights(in_dim=784, h1=256, h2=128, h3=64, out_dim=10):
    np.random.seed(0)
    w1 = np.random.randn(in_dim, h1) * 0.01
    w2 = np.random.randn(h1, h2) * 0.01
    w3 = np.random.randn(h2, h3) * 0.01
    w4 = np.random.randn(h3, out_dim) * 0.01
    return w1, w2, w3, w4

def forward(X, w1, w2, w3, w4, mode='sigmoid'):
    act = act_sigmoid if mode == 'sigmoid' else act_tanh
    z1 = X @ w1
    a1 = act(z1)
    z2 = a1 @ w2
    a2 = act(z2)
    z3 = a2 @ w3
    a3 = act(z3)
    z4 = a3 @ w4
    a4 = softmax_fn(z4)
    return (a1, a2, a3, a4), (z1, z2, z3, z4)

def backward(X, y, weights, activations, z_vals, mode='sigmoid'):
    w1, w2, w3, w4 = weights
    a1, a2, a3, a4 = activations
    z1, z2, z3, z4 = z_vals
    der = der_sigmoid if mode == 'sigmoid' else der_tanh

    d4 = a4 - y
    dw4 = a3.T @ d4

    d3 = (d4 @ w4.T) * der(z3)
    dw3 = a2.T @ d3

    d2 = (d3 @ w3.T) * der(z2)
    dw2 = a1.T @ d2

    d1 = (d2 @ w2.T) * der(z1)
    dw1 = X.T @ d1

    return dw1, dw2, dw3, dw4

def train(X, y, ep=5, bs=64, alpha=0.1, mode='sigmoid', loss='mse'):
    w1, w2, w3, w4 = init_weights()
    for epoch in range(ep):
        perm = np.random.permutation(len(X))
        X, y = X[perm], y[perm]
        for i in range(0, len(X), bs):
            xb = X[i:i+bs]
            yb = y[i:i+bs]
            acts, zs = forward(xb, w1, w2, w3, w4, mode)
            grads = backward(xb, yb, (w1, w2, w3, w4), acts, zs, mode)
            w1 -= alpha * grads[0] / bs
            w2 -= alpha * grads[1] / bs
            w3 -= alpha * grads[2] / bs
            w4 -= alpha * grads[3] / bs

        A, _ = forward(X, w1, w2, w3, w4, mode)
        y_pred = np.argmax(A[-1], axis=1)
        y_true = np.argmax(y, axis=1)
        acc = np.mean(y_pred == y_true)
        l = loss_mse(y, A[-1]) if loss == 'mse' else loss_ce(y, A[-1])
        print(f"Epoch {epoch+1}: Loss={l:.4f}, Accuracy={acc:.4f}")
    return w1, w2, w3, w4

def evaluate(X, w1, w2, w3, w4, mode='sigmoid'):
    A, _ = forward(X, w1, w2, w3, w4, mode)
    return np.argmax(A[-1], axis=1)

print("\n--- Learning Rate Variants ---")
for lr_val in [0.1, 0.01, 0.001]:
    print(f"\nLearning Rate = {lr_val}")
    weights = train(X_train, y_train_enc, ep=5, alpha=lr_val, mode='tanh', loss='cross')
    preds = evaluate(X_test, *weights, mode='tanh')
    accuracy = np.mean(preds == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

print("\n--- Tanh Activation Test ---")
weights = train(X_train, y_train_enc, ep=5, alpha=0.1, mode='tanh', loss='cross')
preds = evaluate(X_test, *weights, mode='tanh')
print(f"Tanh Accuracy: {np.mean(preds == y_test):.4f}")

def arch_test(h1, h2, h3, desc):
    print(f"\nTesting: {desc}")
    w1, w2, w3, w4 = init_weights(h1=h1, h2=h2, h3=h3)
    for e in range(5):
        perm = np.random.permutation(len(X_train))
        xb, yb = X_train[perm], y_train_enc[perm]
        for i in range(0, len(xb), 64):
            x_sub = xb[i:i+64]
            y_sub = yb[i:i+64]
            acts, zs = forward(x_sub, w1, w2, w3, w4, 'tanh')
            grads = backward(x_sub, y_sub, (w1, w2, w3, w4), acts, zs, 'tanh')
            w1 -= 0.1 * grads[0] / 64
            w2 -= 0.1 * grads[1] / 64
            w3 -= 0.1 * grads[2] / 64
            w4 -= 0.1 * grads[3] / 64
    y_pred = evaluate(X_test, w1, w2, w3, w4, 'tanh')
    print(f"{desc} ➤ Accuracy: {np.mean(y_pred == y_test):.4f}")

print("\n--- Architecture Variants ---")
arch_test(512, 256, 128, "Arch 784 → 512 → 256 → 128 → 10")
arch_test(128, 64, 32, "Arch 784 → 128 → 64 → 32 → 10")

print("\n--- Simple vs Deep ---")
arch_test(10, 10, 10, "Simple: 784 → 10")
arch_test(64, 10, 10, "Medium: 784 → 64 → 10")
arch_test(128, 10, 10, "Better: 784 → 128 → 10")

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
+++++++
# Compute confusion matrix
conf_mat = confusion_matrix(y_test, preds)
print("\nConfusion Matrix:")
print(conf_mat)

# Compute overall accuracy
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.4f}")

# Compute Precision, Recall, and F1-Score (Macro Average)
precision = precision_score(y_test, preds, average='macro')
recall = recall_score(y_test, preds, average='macro')
f1 = f1_score(y_test, preds, average='macro')

print(f"\nPrecision (Macro Avg): {precision:.4f}")
print(f"Recall (Macro Avg): {recall:.4f}")
print(f"F1-Score (Macro Avg): {f1:.4f}")

# Optional: Full classification report
print("\nClassification Report:")
print(classification_report(y_test, preds))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Calculate confusion matrix again if not already
conf_mat = confusion_matrix(y_test, preds)

# Plot the confusion matrix
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
