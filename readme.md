# Titanic Survival Prediction - Task 3

Welcome to **Task 3** of the AI & ML internship! In this project, we predict the survival of Titanic passengers using various machine learning models including Logistic Regression and Random Forest.

# Folder Structure
TASK 3/
│
├── train.csv
├── test.csv
├── gender_submission.csv
├── task3_linear_model.py
├── task3_logistic_model.py
├── task3_random_forest.py
├── confusion_matrix.png
├── submission.csv
└── README.md

 Objective

To apply **Supervised Learning** techniques to predict whether a passenger survived the Titanic disaster based on their features.

---

# Models Implemented

### 1. Linear Regression
- Used initially for numeric predictions (for learning).
- Evaluated using MAE, MSE, R².

### 2. Logistic Regression
- Suitable for binary classification (survived or not).
- Evaluation metrics: **Accuracy**, **Confusion Matrix**, **Classification Report**.

### 3. Random Forest Classifier
- Powerful ensemble model.
- Tuned using **GridSearchCV** for best parameters.

---

# Features Used

- `Pclass` (Ticket class)
- `Sex`
- `Age`
- `SibSp` (Siblings/Spouses aboard)
- `Parch` (Parents/Children aboard)
- `Fare`

Categorical features like `Sex` and `Embarked` were encoded numerically.

---

# Data Preprocessing

- Missing `Age`, `Fare`, `Embarked` values were filled with median or mode.
- Categorical encoding done using label mapping.
- Data split into train/test using `train_test_split`.

---

# Evaluation

- **Accuracy Score**
- **Confusion Matrix** (Saved as `confusion_matrix.png`)
- **Classification Report** (Precision, Recall, F1-Score)

---

# Submission

Final predictions on the `test.csv` were saved in:

using the best model after tuning.

---

##  Highlights

- Learned and implemented **Supervised Learning**.
- Understood and visualized **model performance**.
- Applied **GridSearchCV** to tune hyperparameters.

---

##  Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

##  Conclusion

This task helped in gaining practical experience with:
- Data cleaning
- Model building
- Evaluation
- Model tuning
- Real-world submission workflow

---

##  Author

**Vaidika**  
Part of AI & ML Internship 🚀  
Task Completed on: `June 26, 2025` (My Birthday! 🎂)
