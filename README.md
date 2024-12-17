# Hiring Decision Predictor

## Overview  
The **Hiring Decision Predictor** is a machine learning-based project designed to predict hiring decisions based on candidate attributes. The project leverages various supervised machine learning algorithms and data science techniques to build a robust, accurate prediction model.

---

## Table of Contents  
- [Project Objectives](#project-objectives)  
- [Technologies Used](#technologies-used)  
- [Data Preprocessing](#data-preprocessing)  
- [Model Development](#model-development)  
- [Model Evaluation](#model-evaluation)  
- [Key Insights](#key-insights)  
- [Conclusions](#conclusions)  
- [How to Run](#how-to-run)

---

## Project Objectives  
1. To predict hiring decisions based on candidate features using machine learning models.  
2. To compare the performance of multiple algorithms and determine the best model.  
3. To gain insights into which features impact hiring decisions the most.  

---

## Technologies Used  
The following technologies and libraries were used:  

- **Python**: Programming language  
- **pandas**: Data manipulation and analysis  
- **numpy**: Numerical computation  
- **scikit-learn**: Machine learning model building  
- **matplotlib**: Data visualization  

---

## Data Preprocessing  
The dataset underwent the following preprocessing steps to ensure model readiness:  

1. **Data Cleaning**:  
   - Handled missing values and inconsistencies in the dataset.  
2. **Feature Engineering**:  
   - Engineered features for better model performance (e.g., scaling and encoding).  
3. **Data Splitting**:  
   - Split the dataset into training and testing sets using an 80/20 ratio.  
4. **Feature Scaling**:  
   - Applied normalization or standardization to ensure consistent feature scales.  

---

## Model Development  
The following machine learning models were implemented and compared:  

1. **Logistic Regression**  
2. **Support Vector Machines (SVM)**  
3. **K-Nearest Neighbors (KNN)**  
4. **Ridge Regression**  
5. **Lasso Regression**  

### Workflow:  
- **Model Training**: Used the training set to fit the models.  
- **Hyperparameter Tuning**: Applied grid search to find the optimal parameters for each model.  
- **Cross-Validation**: Used k-fold cross-validation to avoid overfitting.  

---

## Model Evaluation  
Each model was evaluated using key performance metrics:  

- Accuracy  
- Precision  
- Recall  
- F1-Score  

A comparison table was created to identify the best-performing model based on these metrics.  

---

## Key Insights  
- Logistic Regression and Support Vector Machines (SVM) provided the best accuracy among the models tested.  
- KNN and Ridge Regression performed decently but were slightly less accurate.  
- Lasso Regression highlighted feature importance through its regularization technique.  

---

## Conclusions  
The project successfully achieved the goal of predicting hiring decisions using machine learning. Key conclusions include:  

1. **Logistic Regression** and **SVM** are effective models for binary classification in this context.  
2. Feature preprocessing (scaling and encoding) plays a significant role in improving model performance.  
3. Ridge and Lasso Regression provide additional insights into feature importance while maintaining good predictive power.  

---

## How to Run  

### Prerequisites  
Ensure you have the following installed:  
- Python 3.x  
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Steps  
1. Clone the repository:  
   ```bash
   git clone git@github.com:jon-paino/hiringpred.git
   cd HiringDecisionPredictor
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook Hiring_Decision_Predictor.ipynb
   ```

4. Follow the notebook workflow to preprocess the data, train models, and evaluate results.

---

## Future Work  
- Expand the model to predict multi-class hiring outcomes (e.g., job roles).  
- Integrate advanced techniques such as ensemble models (e.g., Random Forests, Gradient Boosting).  
- Visualize feature importance using SHAP or LIME.  

---

## Acknowledgments  
This project was built for educational purposes to explore machine learning in hiring predictions.
