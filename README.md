# 🎓 Student AI Performance Classifier

A Deep Learning project built with **PyTorch** that predicts a student's academic performance based on study-related features.

---

# 📌 Project Overview

This project demonstrates how to build a complete Deep Learning classification pipeline using **PyTorch**.

The model receives several student-related features and predicts one of three performance levels:

* 🟥 Low
* 🟨 Medium
* 🟩 High

This project was developed as part of my Deep Learning learning journey while practicing core PyTorch concepts.

---

# 🚀 Features

The model predicts student performance using:

* 📚 Study Hours
* 😴 Sleep Hours
* 🏫 Attendance
* 📝 Assignments Completed
* 📊 Previous Score

Output:

```
0 → Low
1 → Medium
2 → High
```

---

# 🧠 Deep Learning Concepts Used

This project covers the following PyTorch concepts:

* Custom Dataset
* DataLoader
* Mini Batch Training
* Neural Networks
* nn.Module
* Linear Layers
* ReLU
* GELU
* CrossEntropyLoss
* SGD Optimizer
* Forward Propagation
* Backpropagation
* Training Loop
* Accuracy Evaluation
* Softmax
* Argmax
* Model Evaluation
* torch.no_grad()
* model.eval()
* Model Saving
* Loss Visualization

---

# 🏗 Model Architecture

```
Input (5 Features)
        │
        ▼
Linear (5 → 32)
        │
      ReLU
        │
        ▼
Linear (32 → 16)
        │
      GELU
        │
        ▼
Linear (16 → 8)
        │
      ReLU
        │
        ▼
Linear (8 → 3)
        │
        ▼
Output Classes
```

---

# 📂 Project Structure

```
02-first-neural-network
│
├── images
│   └── student_ai_training_loss.png
│
├── models
│   └── student_ai_model.pth
│
├── notebooks
│
├── src
│   ├── student_ai_project.py
│   └── practice
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 📈 Training Result

Final Training Accuracy

```
Accuracy: 100%
```

Training Loss decreases steadily during training.

---

# 📊 Example Prediction

Input

```
Study Hours: 7
Sleep Hours: 8
Attendance: 90
Assignments: 4
Previous Score: 88
```

Prediction

```
Medium
```

Example probabilities

```
Low    : 0.000000002
Medium : 99.05%
High   : 0.94%
```

---

# 💾 Saved Outputs

After training, the project automatically saves:

```
models/student_ai_model.pth
images/student_ai_training_loss.png
```

---

# ▶️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python src/student_ai_project.py
```

---

# 🛠 Technologies

* Python
* PyTorch
* Torch Dataset
* Torch DataLoader
* Matplotlib

---

# 📚 Learning Outcomes

Through this project I practiced:

* Building custom datasets
* Creating neural networks
* Training deep learning models
* Multi-class classification
* Model evaluation
* Saving trained models
* Visualizing loss curves
* Making predictions using Softmax

---

# 🚀 Future Improvements

* Use a real-world dataset
* Add Train/Test split
* Add Validation set
* Try Adam optimizer
* Add Dropout
* Add Batch Normalization
* Export predictions to CSV
* Build a Streamlit web application
* Deploy the model

---

# 👨‍💻 Author

Deep Learning Project using **PyTorch**

Built as part of my AI Engineer roadmap.
...
...