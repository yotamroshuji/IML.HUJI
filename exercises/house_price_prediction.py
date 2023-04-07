import os

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # Make sure we don't ruin the entered data
    X = X.copy()

    # Add "prices" to X to make it easier to handle indexes
    if y is not None:
        X['price'] = y

    # Remove unnecessary columns (ID)
    X = X.drop(["id", "lat", "long"], axis=1)

    # Convert the date column into seconds
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT', errors="coerce").apply(
        lambda x: x.timestamp() if pd.notna(x) else pd.NA)

    # Remove empty rows and duplicates (there aren't any right here, but this should be taken care of)
    X = X.dropna().drop_duplicates()

    # Remove rows with negative values for any of the following columns
    non_negative_cols = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above',
                         'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'sqft_living15', 'sqft_lot15']

    X = X[(X[non_negative_cols] >= 0).all(axis=1)]

    non_zero_cols = ['date', 'price', 'sqft_lot', 'floors', 'yr_built', 'zipcode', 'sqft_lot15']

    X = X[(X[non_zero_cols] != 0).all(axis=1)]

    # Make sure categories are only in the correct span
    X = X[X.waterfront.isin({0, 1}) &
          X.view.between(0, 4) &
          X.condition.between(1, 5) &
          X.grade.between(1, 13)]

    # Make sure sqft sum up correctly
    X = X[X.sqft_above + X.sqft_basement == X.sqft_living]

    # Make sure renovation isn't before the house was built
    X = X[(X.yr_built <= X.yr_renovated) | (X.yr_renovated == 0)]

    # Fix yr_renovated and yr_built: There is a large difference between "never renovated" (0) and the year renovated
    # (value around 2000), and not so much a difference between renovated in 1950 and 2010.
    # In addition, yr_built and yr_renovated are very closely related to when the actual sale occurred (a house sold
    # in 1950 and built in 1949, is better than built in 1949 and sold 2015).

    years_since_renovation = pd.to_datetime(X.date, unit='s').dt.year - X.yr_renovated
    years_since_built = pd.to_datetime(X.date, unit='s').dt.year - X.yr_built

    # There can be negative values, for example if a house was sold before it was built.
    # Using a large range of buckets, to captures all feasible possibilities.
    bins = np.append(np.linspace(-10, 90, 11), np.inf)

    X["years_since_renovation"] = pd.cut(years_since_renovation, bins=bins, labels=bins[1:], include_lowest=True)
    X["years_since_built"] = pd.cut(years_since_built, bins=bins, labels=bins[1:], include_lowest=True)
    X = pd.get_dummies(X, columns=["years_since_renovation"], drop_first=True)
    X = pd.get_dummies(X, columns=["years_since_built"], drop_first=True)
    X = X.drop(["yr_renovated", "yr_built"], axis=1)

    # Set the categorical variables correctly
    # No need to fix waterfront/view/grade/condition (because these are correctly sorted)
    X.zipcode = X.zipcode.astype('category')
    X = pd.get_dummies(X, columns=['zipcode'], drop_first=True)

    # Return the prices back to the y column
    y = X['price']
    X = X.drop("price", axis=1)

    # Returning all values as floats
    return X.astype(float), y.astype(float)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    # Remove columns with a std of zero
    std_mult = np.std(y) * np.std(X, axis=0)
    std_mult = std_mult[std_mult != 0]
    X = X.loc[:, std_mult.index]

    # Calculate pearson correlation for all the features at once
    # Use the covariance equation: E( (X-E(X)) * (Y-E(Y)) ) to get a df where each column stores the covariance
    # of the feature with y
    centered_y = y - np.mean(y)
    centered_X = X - np.mean(X, axis=0)
    covariances = np.mean(centered_X.mul(centered_y, axis=0), axis=0)

    pearson_correlations = covariances / std_mult

    # Don't write all the zipcode files (like 70 of them and takes forever)
    for feature, pearson_cor in pearson_correlations.iteritems():
        if feature.startswith("zipcode"):
            continue

        figure = px.scatter(
            pd.DataFrame({feature: X[feature], y.name: y}),
            x=feature, y=y.name,
            trendline='ols',
            trendline_color_override="red",
            title=f"Correlation between feature: {feature} and {y.name}, "
                  f"<br>Pearson Correlation: {pearson_cor}"
        )
        figure.write_image(os.path.join(output_path, f"{feature}_to_{y.name}.png"))


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(X=df.loc[:, df.columns != 'price'], y=df['price'])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)

    # Question 3 - Feature evaluation with respect to response
    # TODO: change this path
    feature_evaluation(train_X, train_y,
                       r"C:\Users\Yotam\PycharmProjects\IML.HUJI\exercises\graphs_house_evaluation")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    test_X, test_y = preprocess_data(test_X, test_y)
    model = LinearRegression()
    plot_df = pd.DataFrame(index=range(10, 101, 1), columns=['loss_mean', 'loss_variance'], dtype=float)
    for p in plot_df.index:
        loss_values = np.zeros((10, 1))
        for idx in range(10):
            train_X_sample = train_X.sample(frac=p / 100.0)
            train_y_sample = train_y.loc[train_X_sample.index]
            loss_values[idx] = model.fit(train_X_sample, train_y_sample).loss(test_X, test_y)

        plot_df.loc[p, ['loss_mean', 'loss_variance']] = [np.mean(loss_values), np.var(loss_values)]

    figure = go.Figure([
        go.Scatter(
            name="Upper Bound",
            x=plot_df.index,
            y=plot_df['loss_mean'] + 2 * np.sqrt(plot_df['loss_variance']),
            mode='lines',
            line=dict(width=0),

        ),
        go.Scatter(
            name="Lower Bound",
            x=plot_df.index,
            y=plot_df['loss_mean'] - 2 * np.sqrt(plot_df['loss_variance']),
            mode='lines',
            line=dict(width=0),
            fillcolor='lightgray',
            fill='tonexty'
        ),
        go.Scatter(
            name="Average loss as function of training size",
            x=plot_df.index,
            y=plot_df['loss_mean'],
            mode='lines+markers'
        )
    ])
    figure.update_layout(
        xaxis_title=f'Percent of Training Set Size (of {len(train_X)} samples)',
        yaxis_title=f'Average Loss from Test Set',
        autosize=False,
        width=1024,
        height=512,
        showlegend=False
    )

    figure.write_image("average_loss_over_training_set.png")
