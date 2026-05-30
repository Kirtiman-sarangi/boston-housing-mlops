"""
train.py
Trains a DecisionTreeRegressor on the Boston Housing data.
All operations use the generic functions in misc.py.
"""

from sklearn.tree import DecisionTreeRegressor
import misc


def main():
    df = misc.load_data()                 # data loading
    X, y = misc.preprocess_data(df)       # preprocessing
    X_train, X_test, y_train, y_test = misc.split_data(X, y)  # split

    model = DecisionTreeRegressor(random_state=42)
    model = misc.train_model(model, X_train, y_train)         # training

    test_mse = misc.evaluate_model(model, X_test, y_test)     # testing
    avg_mse = misc.average_mse(model, X, y)                   # avg MSE (CV)

    print("Model: DecisionTreeRegressor")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Average MSE (5-fold CV): {avg_mse:.4f}")


if __name__ == "__main__":
    main()