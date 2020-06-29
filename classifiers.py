import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import catboost as cb


def _get_train_test_values():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    X_train = df_train.drop("Target", axis=1)
    y_train = df_train["Target"]
    X_test = df_test.drop("Target", axis=1)
    y_test = df_test["Target"]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def knn_classifier():
    BEST_K = 3  # Look the jupyter notebook for more explanations about the process
    print("Running KNN Classifier...")
    X_train, X_test, y_train, y_test = _get_train_test_values()
    classifier = KNeighborsClassifier(n_neighbors=BEST_K)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred)
    print("the KNN cfm was: \n", cfm)
    score = classifier.score(X_test, y_test)
    print("the KNN score was:", score)
    return (score, 'KNN')


def logistic_regression_classifier():
    print("Running Logistic Regression Classifier...")
    X_train, X_test, y_train, y_test = _get_train_test_values()
    classifier = LogisticRegression(
        penalty="l2", dual=True, max_iter=480, multi_class="auto", solver="liblinear", C=10)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred)
    print("the Logistic Regression cfm was: \n", cfm)
    score = classifier.score(X_test, y_test)
    print("the Logistic Regression score was:", score)
    return (score, 'Logistic Regression')


def random_forest_classifier():
    print("Running Random Forest Classifier...")
    X_train, X_test, y_train, y_test = _get_train_test_values()
    classifier = RandomForestClassifier(criterion='gini', max_depth=None, max_features='log2',
                                        max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, n_estimators=120)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred)
    print("the Random Forest cfm was: \n", cfm)
    score = classifier.score(X_test, y_test)
    print("the Random Forest score was:", score)
    return (score, 'Random Forest')


def Catboost_classifier():
    print("Running Catboost Classifier...")
    X_train, X_test, y_train, y_test = _get_train_test_values()
    classifier = cb.CatBoostClassifier(
        depth=7, iterations=300, l2_leaf_reg=9, learning_rate=0.1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    cfm = confusion_matrix(y_test, y_pred)
    print("the Catboost cfm was: \n", cfm)
    score = classifier.score(X_test, y_test)
    print("the Catboost score was:", score)
    return (score, 'Catboost')
