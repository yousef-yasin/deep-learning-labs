# deep-learning-labs
Deep Learning projects including CNNs, RNNs, LSTMs, Transformers, and neural networks.
# Student Performance Classifier using PyTorch

## Overview

This project is a simple Deep Learning classification model built with PyTorch.

The model predicts a student's performance level based on academic-related features.

The output classes are:

- `0` → Low Performance
- `1` → Medium Performance
- `2` → High Performance

This project was built to practice the core Deep Learning pipeline using PyTorch.

---

## Project Goal

The goal of this project is to build a neural network that can classify student performance using basic input features such as:

- Study hours
- Sleep hours
- Attendance percentage
- Completed assignments

---

## Features Used

Each student is represented using 4 input features:

| Feature | Description |
|--------|-------------|
| Study Hours | Number of daily study hours |
| Sleep Hours | Number of daily sleep hours |
| Attendance | Student attendance percentage |
| Assignments Completed | Number of completed assignments |

---

## Classes

| Class | Label |
|------|-------|
| 0 | Low |
| 1 | Medium |
| 2 | High |

---

## Technologies Used

- Python
- PyTorch
- Torch Dataset
- Torch DataLoader
- Neural Networks
- Matplotlib

---

## Deep Learning Concepts Applied

This project applies the following concepts:

- Custom Dataset
- DataLoader
- Mini-batch training
- Neural Network using `nn.Module`
- Linear Layers
- ReLU Activation Function
- CrossEntropyLoss
- SGD Optimizer
- Training Loop
- Backpropagation
- Accuracy Evaluation
- Softmax Probabilities
- Model Saving
- Loss Visualization

---

## Model Architecture

```text
Input Layer: 4 features
        ↓
Linear(4 → 16)
        ↓
ReLU
        ↓
Linear(16 → 8)
        ↓
ReLU
        ↓
Linear(8 → 3)
        ↓
Output Classes: Low / Medium / High.