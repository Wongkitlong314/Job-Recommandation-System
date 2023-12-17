from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
import numpy as np
from model_data_preprocessing import data_cleaning

def model_XGB(input_df,n):
    data = data_cleaning()

    # Keep region in data to calculate standarize_salary 
    # # but region is not used in Model
    data_model = data.drop(columns=['region','algo_other'])
    X = data_model.drop([col for col in data_model.columns if col.startswith('role_title')], axis = 1) #.drop(['completion_time', 'salary_int', axis = 1)
    y = data_model["role_title"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Your existing code for class weights and training the classifier
    num_classes = len(set(y_train))
    class_weights = [len(y_train) / (np.sum(y_train == i) + 1) for i in range(num_classes)]

    # Create and train an XGBoost classifier
    xgb_clf = XGBClassifier(scale_pos_weight=class_weights)
    xgb_clf.fit(X_train, y_train)

    # Make predictions on the test set probabilities
    y_pred_proba = xgb_clf.predict_proba(input_df)
    label_mapping = {0: 'Data Analyst', 1: 'Data Engineer', 2: 'Data Scientist', 3: 'Machine Learning Engineer', 4: 'Manager', 5: 'Other', 6: 'Research Scientist', 7: 'Software Engineer'}
    top = np.argsort(y_pred_proba, axis=1)[:, -n:]
    
    # Inverse the label to get the top first
    inverse_label = np.flip(top,axis=1)
    inverse_label#概率最大的标签

    recommendations = [[label_mapping[i] for i in row] for row in inverse_label]

    
    return recommendations