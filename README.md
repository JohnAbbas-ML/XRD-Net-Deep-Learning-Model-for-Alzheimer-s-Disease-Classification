
### 🧠 XRD-Net: Deep Learning Model for Alzheimer's Disease Classification

This repository contains the implementation of **XRD-Net**, a novel convolutional neural network designed for **Alzheimer’s Disease stage classification** from medical images. The architecture integrates **Dense Blocks**, **Residual Blocks**, and a custom **Spatial Context Fusion (SCF) Block** with attention to achieve rich multi-scale feature learning.

---

### ✅ **Key Features**

* **Dense Blocks** for feature reuse and efficient representation learning
* **SCF Blocks** for multi-scale spatial context fusion with attention
* **Residual Connections** for stable gradient flow
* **Global Average Pooling + FC** classification head
* **Dropout regularization** to reduce overfitting
* Supports **4-class AD stage classification**

---

### 📌 **Model Architecture Overview**

The model follows a multi-stage structure:

1. Initial Convolution + MaxPool
2. **Stage 1:** Dense Block → SCF Block → Residual Block
3. **Stage 2:** Dense Block → SCF Block → Residual Block
4. **Stage 3:** Dense Block → SCF Block → Residual Block
5. Global Avg Pool → Fully Connected Layers → Softmax (logits)

---

### 🏗️ **Core Components**

| Component               | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| **DenseBlock**          | Feature reuse via layer concatenation      |
| **SCF Block**           | Multi-scale fusion (3×3 + 5×5) + Attention |
| **ResidualBlock**       | Enhanced gradient flow (skip connections)  |
| **Classification Head** | GAP + FC + Dropout for final prediction    |

---

### 🚀 **Usage**

Create the model:

```python
model = XRDNet(num_classes=4, growth_rate=32, dropout_rate=0.5)
```

Test forward pass:

```python
dummy_input = torch.randn(2, 3, 128, 128)
output = model(dummy_input)
```

Extract features:

```python
features = model.extract_features(dummy_input)
```

---
