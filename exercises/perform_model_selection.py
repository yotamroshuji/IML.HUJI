from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_indexes = np.random.permutation(len(X))[:n_samples]
    test_mask = np.ones(len(X), dtype=bool)
    test_mask[train_indexes] = False
    train_X = X[train_indexes]
    train_y = y[train_indexes]
    test_X = X[test_mask]
    test_y = y[test_mask]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # Ridge CV
    v_cross_validate = np.vectorize(cross_validate, excluded=['X', 'y'])
    ridge_lambdas = np.linspace(start=0.001, stop=1, num=n_evaluations)
    lasso_lambdas = np.linspace(start=0.05, stop=1, num=n_evaluations)
    figure = make_subplots(rows=1, cols=2, subplot_titles=['Ridge Regression', 'Lasso'])
    figure.update_layout(title=r"$\text{Model Errors, Given } \lambda$", title_x=0.5)
    ridge_validation_errors, lasso_validation_errors = [], []

    for idx, (model_class, lambdas, validation_errors) in enumerate([
        (RidgeRegression, ridge_lambdas, ridge_validation_errors),
        (partial(Lasso, max_iter=3000), lasso_lambdas, lasso_validation_errors)
    ]):
        models = [model_class(lambda_param) for lambda_param in lambdas]
        train_errors, temp_validation_errors = v_cross_validate(models, X=train_X, y=train_y, scoring=mean_square_error)
        figure.add_traces(
            [
                go.Scatter(x=lambdas, y=train_errors, name='Train Error'),
                go.Scatter(x=lambdas, y=temp_validation_errors, name='Validation Error')
            ],
            rows=1, cols=idx + 1
        )
        figure.update_xaxes(title_text=r"$\lambda$", row=1, col=idx + 1)
        figure.update_yaxes(title_text=r"$\text{Mean Square Error}$", row=1, col=idx + 1)

        # Update the validation errors (used later)
        validation_errors.extend(temp_validation_errors)

    figure.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = ridge_lambdas[np.argmin(ridge_validation_errors)]
    best_lasso_lambda = lasso_lambdas[np.argmin(lasso_validation_errors)]

    # Print the errors for each of the models: Ridge, Lasso, LinearRegression
    print(
        f"Ridge Regression error with lambda: {best_ridge_lambda} - "
        f"{RidgeRegression(best_ridge_lambda).fit(train_X, train_y).loss(test_X, test_y)}")

    print(
        f"Lasso error with lambda: {best_lasso_lambda} - "
        f"{mean_square_error(test_y, Lasso(best_lasso_lambda).fit(train_X, train_y).predict(test_X))}")

    print(
        f"Linear Regression error - "
        f"{LinearRegression().fit(train_X, train_y).loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
