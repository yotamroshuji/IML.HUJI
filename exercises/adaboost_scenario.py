from functools import partial

import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ada_boost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_errors = [ada_boost.partial_loss(train_X, train_y, t) for t in range(1, n_learners + 1)]
    test_errors = [ada_boost.partial_loss(test_X, test_y, t) for t in range(1, n_learners + 1)]
    figure = go.Figure(
        [
            go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, mode='lines', name='Train error'),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_errors, mode='lines', name='Test error')
        ]
    )
    figure.update_xaxes(range=[0, n_learners * 1.05])
    figure.update_layout(title="Train and Test errors of AdaBoost Ensemble",
                         xaxis_title="Ensemble Size",
                         yaxis_title="Misclassification Error")
    figure.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    figure2 = make_subplots(rows=2, cols=2,
                            subplot_titles=[rf"$\textbf{{AdaBoost ({t} learners)}}$" for t in T])

    for idx, t in enumerate(T):
        figure2.add_traces([
            decision_surface(partial(ada_boost.partial_predict, T=t), lims[0], lims[1], showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                       mode='markers',
                       marker=go.scatter.Marker(color=test_y, colorscale=(custom[0], custom[1])))
        ],
            rows=idx // 2 + 1,
            cols=idx % 2 + 1
        )
    figure2.update_layout(title="Decision Surface of AdaBoost Ensemble", showlegend=False)
    figure2.show()

    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_errors) + 1
    figure3 = go.Figure([
        decision_surface(partial(ada_boost.partial_predict, T=best_t), lims[0], lims[1], showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers',
                   marker=go.scatter.Marker(color=test_y, colorscale=(custom[0], custom[1])))
    ])
    figure3.update_layout(
        title=rf"Best performing AdaBoost Ensemble <br>"
              rf"(Size: {best_t}, "
              rf"Accuracy: {accuracy(test_y, ada_boost.partial_predict(test_X, best_t))}, "
              rf"Error: {ada_boost.partial_loss(test_X, test_y, best_t)})"
    )
    figure3.show()

    # Question 4: Decision surface with weighted samples
    point_weights = ada_boost.D_[-1] / np.max(ada_boost.D_[-1]) * 30
    figure4 = go.Figure([
        decision_surface(ada_boost.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                   marker=go.scatter.Marker(color=train_y, size=point_weights))
    ])
    figure4.update_layout(title="AdaBoost Decision Surface and Train Sample Distribution")
    figure4.show()


if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise, 250, 5000, 500)
