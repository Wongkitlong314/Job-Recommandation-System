from model_data_preprocessing import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
from joblib import dump, load
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import recall_score, f1_score, confusion_matrix, classification_report


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Read data
df_2020 = pd.read_csv('/Users/lc/Documents/IT5006/group_assignment/kaggle_survey_2020_responses.csv', low_memory=False)
df_2021 = pd.read_csv('/Users/lc/Documents/IT5006/group_assignment/kaggle_survey_2021_responses.csv', low_memory=False)
df_2022 = pd.read_csv('/Users/lc/Documents/IT5006/group_assignment/kaggle_survey_2022_responses.csv', low_memory=False)

model_data =data_cleaning(df_2020, df_2021, df_2022)

# seperate label and features
X = model_data.drop([col for col in model_data.columns if col.startswith('role_title')], axis = 1)
y = model_data["role_title"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print(label_mapping)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_test.columns)

# test different classifiers
classifiers = [
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("Support Vector Machine", SVC()),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, min_samples_leaf=20)),
    ("AdaBoost", AdaBoostClassifier(n_estimators=100, learning_rate=0.5)),
    ("Neural Network", MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, alpha=0.001, solver='adam', random_state=42)),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("Quadratic Discriminant Analysis", QuadraticDiscriminantAnalysis()),
    ("Gaussian Process", GaussianProcessClassifier(n_restarts_optimizer=10, max_iter_predict=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=5, min_samples_leaf=20)),
    ("XGBoost", xgb.XGBClassifier(n_estimators=100, learning_rate=0.5, max_depth=5, min_samples_leaf=20)),
    ("LightGBM", LGBMClassifier(num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=200)),
    ("CatBoost", CatBoostClassifier(verbose=0, n_estimators=100))

]

# Test and compare the classifiers
results = []
for name, classifier in classifiers:
    print(name)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((name, accuracy))

# Sort the results by accuracy in descending order
results.sort(key=lambda x: x[1], reverse=True)

# Print the results
for name, score in results:
    print(f"{name}: {score}")
# LGBMClassifier is the best classifier


# Test different hyperparameters for LGBMClassifier
# RandomizedSearchCV find the most suitable parameters
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': sp_randint(6, 50),
             'min_child_samples': sp_randint(100, 500),
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8),
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

# 这里的n_iter和cv可以根据你的需求进行调整
clf = RandomizedSearchCV(LGBMClassifier(),
                         param_distributions=param_test,
                         n_iter=100,
                         scoring='accuracy',
                         cv=5)

clf.fit(X_train, y_train)

print("Best parameters found: ", clf.best_params_)
print("Best accuracy found: ", clf.best_score_)

# GridSearchCV find the most suitable parameters
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': [25, 30, 35],
                'min_child_samples': [100, 120, 140],
                'min_child_weight': [0.01, 0.1, 1, 10],
                'subsample': [0.2, 0.3, 0.4],
                'colsample_bytree': [0.5, 0.6, 0.7],
                'reg_alpha': [5, 7, 10],
                'reg_lambda': [5, 7, 10]}

# here n_iter and cv can be adjusted according to your needs
clf = GridSearchCV(LGBMClassifier(),
                        param_grid=param_test,
                        scoring='accuracy',
                        cv=5)

clf.fit(X_train, y_train)

print("Best parameters found: ", clf.best_params_)
print("Best accuracy found: ", clf.best_score_)

# use optuna to find the most suitable parameters
def objective(trial: Trial) -> float:
    params_lgb = {
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "binary",
        "metric": "auc",
        "num_leaves": trial.suggest_int("num_leaves", 2, 5000),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 500),
        "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 0.1),
        "subsample": trial.suggest_discrete_uniform("subsample", 0.1, 1.0, 0.01),
        "colsample_bytree": trial.suggest_discrete_uniform("colsample_bytree", 0.1, 1.0, 0.01),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    model = LGBMClassifier(**params_lgb)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

sampler = TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

print("Best trial:")
trial_ = study.best_trial

print(f"  Value: {trial_.value}")
print("  Params: ")
for key, value in trial_.params.items():
    print(f"    {key}: {value}")

# define the classifier with the best parameters
classifier = LGBMClassifier(colsample_bytree=0.5741776857102016, min_child_samples=134, min_child_weight=0.01, num_leaves = 25, reg_alpha=5, reg_lambda=10, subsample=0.29839819133454243)

# train the model
classifier.fit(X_train, y_train)

# save the model
dump(classifier, 'classifier_LGBM.joblib') 

# test the model
y_pred = classifier.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)


# print the accuracy
print(f"LightGBM: {accuracy}")

# calculate the recall
recall = recall_score(y_test, y_pred, average='macro')

# calculate the F1 score
f1 = f1_score(y_test, y_pred, average='macro')

# print the recall and F1 score
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# print the confusion matrix
print(f"Confusion Matrix: \n{cm}")

# generate the classification report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

# print the classification report
print(f"Classification Report: \n{report}")

#return top 2 accuracy
# get the probability of each class
y_proba = classifier.predict_proba(X_test)

# get the top 2 predictions
top2_pred = np.argsort(y_proba, axis=1)[:, -2:]

# caculate the top 2 accuracy
accuracy_top2 = np.mean(np.array([y_test[i] in top2_pred[i] for i in range(len(y_test))]))

# print the top 2 accuracy
print(f"Top-2 Accuracy: {accuracy_top2}")