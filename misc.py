"""
misc.py
Generic, reusable helper functions for the ML workflow.
Both train.py (DecisionTreeRegressor) and train2.py (KernelRidge)
import these functions, so they must be model-agnostic.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


def load_data():
    """Load the Boston Housing dataset manually (sklearn deprecated it).
    Method taken verbatim from the assignment PDF / sklearn 1.0 docs."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target  # MEDV is the target variable
    return df


def preprocess_data(df, target_col='MEDV'):
    """Split a dataframe into features X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Generic train/test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(model, X_train, y_train):
    """Generic training: fit ANY scikit-learn regressor and return it."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return the MSE of a fitted model on the test set."""
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)


def average_mse(model, X, y, cv=5):
    """Average MSE via k-fold cross-validation (honours 'average MSE')."""
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring='neg_mean_squared_error')
    return -scores.mean()