import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    data_frame = pd.read_csv(filename, parse_dates=['Date'])
    data_frame = data_frame.dropna().drop_duplicates()

    # Drop temperatures that don't make sense (e.g. minus 72)
    data_frame = data_frame[(60 > data_frame.Temp) & (data_frame.Temp > -50)]

    # Make sure the Date column matches the other columns in the dataset (there were none in the original)
    # This also validates all the dates are correct (months between 1 and 12, days, etc.)
    data_frame = data_frame[~
                            (data_frame.Date.dt.year != data_frame.Year) |
                            (data_frame.Date.dt.month != data_frame.Month) |
                            (data_frame.Date.dt.day != data_frame.Day)]

    # Adding day of the year column
    data_frame['DayOfYear'] = data_frame.Date.dt.day_of_year

    return data_frame


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_df = df[df.Country == 'Israel']
    fig = px.scatter(x=israel_df.DayOfYear, y=israel_df.Temp, color=israel_df.Year.astype(str))
    fig.update_layout(xaxis_title='Day of the Year', yaxis_title='Temperature', title='Temperature by Day in Israel')
    fig.write_image('temp_by_day_in_israel.png')

    # Question 3 - Exploring differences between countries
    std_of_temp_per_month = israel_df.groupby(['Month'], as_index=False).agg(temp_std=('Temp', 'std'))
    fig = px.bar(x=std_of_temp_per_month.Month, y=std_of_temp_per_month.temp_std)
    fig.update_layout(xaxis_title='Month', yaxis_title='Standard Deviation',
                      title='Temperature Standard Deviation During Multiple Years')
    fig.write_image('temp_std_deviation_during_multiple_years.png')

    country_month_temps = df.groupby(by=['Country', 'Month'], as_index=False).agg(avg_temp=('Temp', 'mean'),
                                                                                  std_temp=('Temp', 'std'))
    fig = px.line(x=country_month_temps['Month'], y=country_month_temps['avg_temp'],
                  color=country_month_temps['Country'], error_y=country_month_temps['std_temp'])
    fig.update_layout(xaxis_title="Month", yaxis_title="Average Temperature",
                      title="Average Temperature with Deviation Over Multiple Years")
    fig.write_image("avg_temp_with_deviation_over_multiple_years.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df['DayOfYear'], israel_df['Temp'])

    loss_values = []
    for k_degree in range(1, 11):
        loss = PolynomialFitting(k_degree).fit(train_X, train_y).loss(test_X, test_y)
        loss_values.append((k_degree, loss))
        print(f"k value: {k_degree}, loss: {np.round(loss, 2)}")

    k_degree_to_loss = pd.DataFrame(loss_values, columns=['k_degree', 'loss'])
    fig = px.bar(x=k_degree_to_loss.k_degree, y=k_degree_to_loss.loss)
    fig.update_layout(xaxis_title="Polynomial Degree (k)", yaxis_title="Test error (loss)",
                      title="Error of Temperature Prediction Using DayOfYear by Polynomial Degree (k)")
    fig.write_image('temp_predict_err_by_polynomial_deg.png')

    # Question 5 - Evaluating fitted model on different countries
    # I chose to use k=4
    CHOSEN_K = 4
    model = PolynomialFitting(k=CHOSEN_K).fit(israel_df['DayOfYear'], israel_df['Temp'])
    not_israel_df = df[~df.index.isin(israel_df.index)]
    country_to_loss = not_israel_df.groupby(['Country'], as_index=False).apply(
        lambda country_df: pd.Series(
            dict(
                loss=model.loss(country_df['DayOfYear'], country_df['Temp'])
            )
        )
    )
    fig = px.bar(country_to_loss, x='Country', y='loss', color='Country')
    fig.update_layout(yaxis_title="Model Prediction Error (loss)",
                      title="Temperature Prediction of Model Trained on Israel Data")
    fig.write_image('temp_in_different_countries_with_israel_model.png')
