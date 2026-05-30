"""
train2.py
Trains a KernelRidge model on the Boston Housing data.
Reuses the SAME generic functions from misc.py.
"""

from sklearn.kernel_ridge import KernelRidge
import misc


def main():
    df = misc.load_data()                 # data loading
    X, y = misc.preprocess_data(df)       # preprocessing
    X_train, X_test, y_train, y_test = misc.split_data(X, y)  # split

    model = KernelRidge(alpha=1.0)
    model = misc.train_model(model, X_train, y_train)         # training

    test_mse = misc.evaluate_model(model, X_test, y_test)     # testing
    avg_mse = misc.average_mse(model, X, y)                   # avg MSE (CV)

    print("Model: KernelRidge")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Average MSE (5-fold CV): {avg_mse:.4f}")


if __name__ == "__main__":
    main()