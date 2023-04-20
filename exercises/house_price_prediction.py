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


class TrainPreprocessMetadata:
    """
    A "dataclass" containing metadata about the training preprocess operation.
    This is used to preprocess the test data.
    """

    def __init__(self) -> None:
        self.col_median_values: Optional[pd.DataFrame] = None


train_metadata = TrainPreprocessMetadata()


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    Preprocess sample data.
    Preprocess is different for train data (y supplied) and test data (y not supplied).

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

    # If a y vector (prices) was supplied, the data inputted is training data.
    is_train_data = y is not None

    # Remove unnecessary columns (ID)
    X = X.drop(["id", "lat", "long"], axis=1)

    # Convert the date column into seconds
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT', errors="coerce").apply(
        lambda x: x.timestamp() if pd.notna(x) else pd.NA)

    # Update the metadata after making sure all cols are numeric (date) and removing unneeded cols
    if is_train_data:
        _collect_train_metadata(X, y)

    # Replace invalid values in rows to ones that make sense.
    is_zero_or_nan = lambda x: x.isna() | x == 0
    is_negative_or_nan = lambda x: x.isna() | x.lt(0)
    is_negative_or_zero_or_nan = lambda x: is_zero_or_nan(x) | is_negative_or_nan(x)
    is_outside_or_nan = lambda lower, upper: lambda x: (x.isna()) | (x.gt(upper) | x.lt(lower))
    train_medians = train_metadata.col_median_values

    col_replace_condition = [
        # Date can be negative if the date is before 1.1.1970
        ('date', pd.isna, train_medians['date']),

        ('bedrooms', is_negative_or_nan, train_medians['bedrooms']),
        ('bathrooms', is_negative_or_nan, train_medians['bathrooms']),
        ('sqft_living', is_negative_or_nan, train_medians['sqft_living']),
        ('sqft_lot', is_negative_or_nan, train_medians['sqft_lot']),
        ('floors', is_negative_or_zero_or_nan, train_medians['floors']),

        ('waterfront', is_outside_or_nan(0, 1), 0),
        ('view', is_outside_or_nan(0, 4), train_medians['view']),
        ('condition', is_outside_or_nan(1, 5), train_medians['condition']),
        ('grade', is_outside_or_nan(1, 13), train_medians['grade']),

        ('sqft_above', is_negative_or_nan, train_medians['sqft_above']),
        ('sqft_basement', is_negative_or_nan, train_medians['sqft_basement']),

        # Assume all houses were built after 1900, and for anything else
        # give a value of -np.inf, so the year the house was built is set to a long time ago
        ('yr_built', lambda x: x.isna(), train_medians['yr_built']),
        ('yr_built', lambda x: x.lt(1900), -np.inf),

        # If the renovation is smaller than 1900, set it to zero
        ('yr_renovated', lambda x: x.isna() | x.lt(1900), 0),

        # If the zipcode stated in this dataset does not exist in the train dataset (which
        # deals with illegal zipcodes), it will not appear in the final dataset, so we can ignore it for now.

        ('sqft_living15', is_negative_or_nan, train_medians['sqft_living15']),
        ('sqft_lot15', is_negative_or_nan, train_medians['sqft_lot15']),
    ]

    for column, condition, replacement_value in col_replace_condition:
        condition_mask = condition(X[column])
        X.loc[condition_mask, column] = replacement_value

    # After fixing all values, make sure that:
    # Sqft sum up correctly, otherwise, change the sqft_living to match the sum of sqft_above + sqft_basement

    invalid_sum_mask = X.sqft_above + X.sqft_basement != X.sqft_living
    X.loc[invalid_sum_mask, 'sqft_living'] = X.loc[invalid_sum_mask, 'sqft_basement'] + \
                                             X.loc[invalid_sum_mask, 'sqft_above']

    # If the renovation happened before the house was built, change the renovation year to 0 (not renovated)
    X.loc[(X.yr_built > X.yr_renovated), 'yr_renovated'] = 0

    # Add the categorical columns and make sure to keep those from the train data/add missing ones
    X = _preprocess_categorical_fields(X)

    # Make train specific changes
    if is_train_data:
        # Drop rows with negative/zero/nan prices
        y = y[~(y.isna() | y.lt(0))]

        # Remove outliers in house price
        extreme_high_price = y.quantile(0.999)
        extreme_low_price = y.quantile(0.001)
        y = y[y.between(extreme_low_price, extreme_high_price)]

        # Drop duplicate rows
        X = X.drop_duplicates()

        # Keep only indexes than remain in X and y
        shared_indexes = y.index.intersection(X.index)
        X = X.loc[shared_indexes]
        y = y.loc[shared_indexes]

    # Returning all values as floats
    return X.astype(float), y


def _collect_train_metadata(X: pd.DataFrame, y: pd.Series):
    train_metadata.col_median_values = X.median(axis=0)


def _preprocess_categorical_fields(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create one-hot fields for the sample dataframe.
    Fields added: years_since_renovation, years_since_built, zipcode
    :param X: DataFrame of shape (n_samples, n_features)
              Design matrix of regression problem
    :return: Matrix with the columns.
    """
    # Fix yr_renovated and yr_built: There is a large difference between "never renovated" (0) and the year renovated
    # (value around 2000), and not so much a difference between renovated in 1950 and 2010.
    # In addition, yr_built and yr_renovated are very closely related to when the actual sale occurred (a house sold
    # in 1950 and built in 1949, is better than built in 1949 and sold 2015).

    years_since_renovation = pd.to_datetime(X.date, unit='s').dt.year - X.yr_renovated
    years_since_built = pd.to_datetime(X.date, unit='s').dt.year - X.yr_built

    # There can be negative values, for example if a house was sold before it was built.
    # Using only a small amount of features, after experimenting with the best loss.
    bins = np.append(np.array([-5, 0, 1, 5, 10]), np.inf)

    # Note: using .astype(float) to not include all columns if not needed (float to have np.inf).
    X["years_since_renovation"] = pd.cut(years_since_renovation, bins=bins, labels=bins[1:],
                                         include_lowest=True).astype(float)
    X["years_since_built"] = pd.cut(years_since_built, bins=bins, labels=bins[1:],
                                    include_lowest=True).astype(float)

    # Notice not using the drop_first=True, because we want every zipcode in the train data to have a column,
    # so missing columns in the test data don't get a column and are ignored.
    X = pd.get_dummies(X, columns=["years_since_renovation"], drop_first=False, prefix='years_since_renovation_')
    X = pd.get_dummies(X, columns=["years_since_built"], drop_first=False, prefix='years_since_built_')
    X = X.drop(["yr_renovated", "yr_built"], axis=1)

    # Set the categorical variables correctly
    X.zipcode = X.zipcode.astype('category')
    X = pd.get_dummies(X, columns=['zipcode'], drop_first=False)
    return X


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

        fig = px.scatter(
            pd.DataFrame({feature: X[feature], y.name: y}),
            x=feature, y=y.name,
            trendline='ols',
            trendline_color_override="red",
            title=f"Correlation between feature: {feature} and {y.name}, "
                  f"<br>Pearson Correlation: {pearson_cor}"
        )
        fig.write_image(os.path.join(output_path, f"{feature}_to_{y.name}.png"))


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
    test_X, _ = preprocess_data(test_X)

    # Make sure that test_X has the same columns as the train_X
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

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

    figure.show()
