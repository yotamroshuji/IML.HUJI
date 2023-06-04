from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Not shuffling indexes (results in noisy test/validation errors)
    sample_indexes = np.arange(X.shape[0])
    index_subsets = np.array_split(sample_indexes, cv)
    train_score, validation_score = 0.0, 0.0
    for validation_index_subset in index_subsets:
        train_subset_mask = np.ones_like(sample_indexes, dtype=bool)
        train_subset_mask[validation_index_subset] = False

        X_train = X[train_subset_mask]
        y_train = y[train_subset_mask]
        X_validation = X[validation_index_subset]
        y_validation = y[validation_index_subset]

        fitted_estimator = deepcopy(estimator).fit(X_train, y_train)
        train_score += scoring(y_train, fitted_estimator.predict(X_train))
        validation_score += scoring(y_validation, fitted_estimator.predict(X_validation))

    return train_score / cv, validation_score / cv
