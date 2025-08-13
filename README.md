
# Weather Temperature Prediction using Gradient Descent Linear Regression

## Overview

This repository contains a **machine learning project** that predicts **temperature** from historical weather data.
Unlike standard libraries, the model here is implemented **from scratch** using **Gradient Descent** for linear regression, providing full control over the learning process.

The project covers:

* Data preprocessing & feature engineering
* One-hot encoding for categorical variables
* Training a custom-built gradient descent model
* Model evaluation using **MSE**, **MAE**, and **R² score**

---

## Technical Details

### **1. Model Hypothesis**

The linear regression model predicts the target value as:

$$
\hat{y} = \mathbf{X} \cdot \mathbf{w} + b
$$

Where:

* $\mathbf{X}$ = feature matrix
* $\mathbf{w}$ = weight vector
* $b$ = bias term

---

### **2. Loss Function**

We use **Mean Squared Error (MSE)** as the cost function:

$$
J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \left( y^{(i)} - \hat{y}^{(i)} \right)^2
$$

Where:

* $n$ = number of samples
* $y^{(i)}$ = actual value
* $\hat{y}^{(i)}$ = predicted value

---

### **3. Gradient Descent Update Rules**

The model parameters are updated as:

$$
w_j := w_j - \alpha \cdot \frac{\partial J}{\partial w_j}
$$

$$
b := b - \alpha \cdot \frac{\partial J}{\partial b}
$$

Where:

* $\alpha$ = learning rate

Gradients are computed as:

$$
\frac{\partial J}{\partial w} = -\frac{2}{n} \mathbf{X}^T \left( \mathbf{y} - \hat{\mathbf{y}} \right)
$$

$$
\frac{\partial J}{\partial b} = -\frac{2}{n} \sum_{i=1}^n \left( y^{(i)} - \hat{y}^{(i)} \right)
$$


##  Installation

```bash
# Clone the repository
git clone https://github.com/iiioooiso/Linear-Regression-from-Scratch.git
cd Linear-Regression-from-Scratch
# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Python script:

```bash
python src/main.py --data_path data/data.csv
```

### Or run the notebook:

```bash
jupyter notebook notebooks/app.ipynb
```

---

## Metrics Used

* **MSE (Mean Squared Error)** – Measures average squared prediction error.
* **MAE (Mean Absolute Error)** – Measures average absolute prediction error.
* **R² Score** – Proportion of variance in the dependent variable explained by the model.

---

##  Technologies

* **Python** (Pandas, NumPy, scikit-learn)
* **Jupyter Notebook**
* **Custom ML implementation** (no sklearn model fitting)

---

## License

NULL.
