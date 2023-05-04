import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    if len(y_true) == 0:
        return 0.0

    return (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    hinge_losses = np.maximum(0, 1 - y_true * y_pred)
    loss = hinge_losses.sum()
    if normalize:
        loss /= len(hinge_losses)
    return loss


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    # Calculate the accuracy as (True Positives + True Negatives) / (Positives + Negatives)
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions / len(y_pred)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, n_features)

    Returns:
    --------
    output: ndarray of shape (n_samples, n_features)
        Softmax(x) for every sample x in given data X
    """
    raise NotImplementedError()


if __name__ == '__main__':
    assert accuracy(np.array([1, 1, 1, -1, -1, -1]), np.array([1, 1, 1, -1, -1, -1])) == 1
    assert accuracy(np.array([1, 1, 1, -1, -1, -1]), np.array([1, 1, 1, -1, -1, 1])) == 5 / 6
