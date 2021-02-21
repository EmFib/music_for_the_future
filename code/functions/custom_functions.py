import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.model_selection import train_test_split

import statsmodels.api as sm

import warnings
warnings.simplefilter(action="ignore")
# We are required to do this in order to avoid "FutureWarning" issues.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

%matplotlib inline

def find_p_and_q(df, feature, arima_dict, n=6):

    train = df[feature][0:162]

    d = ndiff_df.loc[ndiff_df['audio_feature'] == feature, 'ndiffs for stationarity'].iloc[0]

    # starting with large start AIC
    best_aic = 99 * (10 * 16)
    # creating variables to store best values ofd p and q
    best_p = 0
    best_q = 0

    # use nested for loop to iterate over values of p and q
    for p in range(n):

        for q in range(n):

            # insert try and and except statements
            try:

                # fitting on ARIMA(p, 1, q) model
                print(f'Attempting to fit ARIMA({p}, {d}, {q})')

                # instantiate ARIMA model
                arima = ARIMA(train, order=(p,d,q))

                # fit ARIMA model
                model = arima.fit()

                # print out AIC for ARIMA(p, 1, q) model
                print(f'For {feature}, the AIC for ARIMA({p},{d},{q}) is: {model.aic}')

                # Is this current model's AIC better than the OF best_aic?
                if model.aic < best_aic:
                    # we want aic to be lower so we are setting a high aic and hoping for something lower

                    # if it is, we overwrite the best_aic, best_p, and best_q
                    best_aic = model.aic
                    best_p = p
                    best_q = q

            except:
                pass

        order = (best_p, d, best_q)

    arima_dict['audio_feature'].append(feature)
    arima_dict['ndiffs(d)'].append(d)
    arima_dict['best_p'].append(best_p)
    arima_dict['best_q'].append(best_q)
    arima_dict['order'].append(order)
    arima_dict['ARIMA_model'].append(f'ARIMA({best_p},{d},{best_q})')
    arima_dict['ARIMA_AIC'].append(best_aic)

    print()
    print(f'{feature.upper()} MODEL FINISHED!')
    print(f'The model for {feature} that minimizes AIC on the training data is the ARIMA({best_p},{d},{best_q}).')
    print(f'The model has an aIC of {best_aic}.')
    print()


def find_sarima_parameters(df, feature, param_df, n_rows=47):

    import time
    t0 = time.time()
    final_mae = 1000000000000
    final_S = 0
    final_D = 0
    final_P = 0
    final_Q = 0

    # find order from arima parameters dataframe
    order = param_df.loc[param_df['audio_feature'] == feature, 'order'].iloc[0]

    train_values = df[feature][0:n_rows]
    test_values = df[feature][n_rows:]

    for S in range(48,53):
        for D in range(2):
            for P in range(4):
                for Q in range(4):
                    print(f'Checking ({P}, {D}, {Q}, {S}) at {round(time.time() - t0)} seconds.')
                    try:
                        sarima = SARIMAX(endog = train_values,
                                         order = order,
                                         seasonal_order = (P, D, Q, S)).fit()

                        sarima_pred = sarima.predict(start=test_values.index[0], end=test_values.index[-1], typ='levels')

                        if mean_absolute_error(test_values, sarima_pred) < final_mae:
                            final_mae = mean_absolute_error(test_values, sarima_pred)
                            final_S = S
                            final_D = D
                            final_P = P
                            final_Q = Q

                        print(f'We just fit a SARIMAX(2, 0, 2)x({P}, {D}, {Q}, {S}) model with {mean_absolute_error(test_values, sarima_pred)} MAE and {mean_squared_error(test_values, sarima_pred)**0.5} RMSE.')

                    except:
                        print('problem!')
                        raise

    print()
    print(f'The final model for {feature} is SARIMAX(2, 0, 2)x({final_P}, {final_D}, {final_Q}, {final_S}).')
    print()


def arima_predict_plot(df, feature, year, ndiff_df, param_df, title='title', figsize=(15,5), order=None, d=None, ci=True):

    # create train and test sets
    n_rows = round(len(df)*0.9)
    train = df[feature][0:n_rows]
    test = df[feature][n_rows:]

    # find ndiffs for stationarity from ndiff dataframe
    if d is None:
        d = ndiff_df.loc[ndiff_df['audio_feature'] == feature, 'ndiffs for stationarity'].iloc[0]
    print(f'd = {d}')

    if order is None:
        # find order from arima parameters dataframe
        order = param_df.loc[param_df['audio_feature'] == feature, 'order'].iloc[0]
    print(f'order = {order}')

    try:
        # instantiate ARIMA model
        model = ARIMA(train, order=order)

        # fit ARIMA model
        arima = model.fit()

        # get predictions for train and test sets
        preds_train = model.predict(params=arima.params, start=train.index[d], end=train.index[-1], typ='levels')
        preds_test = model.predict(params=arima.params, start=test.index[0], end=test.index[-1], typ='levels')

        # calculate and print RMSE for train and test setes
        train_rmse = mean_squared_error(train[d::], preds_train)**0.5
        print(f'{feature.capitalize()} train RMSE ({year}) - ARIMA({order}): {train_rmse}')

        test_rmse = mean_squared_error(test, preds_test)**0.5
        print(f'{feature.capitalize()} test RMSE ({year}) - ARIMA({order}): {test_rmse}')

        # add RMSEs to arima parameters dataframe
        param_df.loc[param_df['audio_feature'] == feature, 'arima_train_rmse'] = train_rmse
        param_df.loc[param_df['audio_feature'] == feature, 'arima_test_rmse'] = test_rmse

          # set up plot
        plt.figure(figsize=figsize)

        # plot training data
        plt.plot(train, color='blue')

        # plot testing data
        plt.plot(test.index, test, color='orange')

        # plot predicted values for test set
        plt.plot(test.index, preds_test, color='green')

        # add line for the baseline model (mean value of feature)
        plt.hlines(df[feature].mean(), train.index[0], test.index[-1], color = 'grey')

        # plot confidence interval
        if ci:
            ci = 1.96 * np.std(preds_test)/np.mean(preds_test)
            plt.fill_between(test.index, (preds_test - ci), (preds_test + ci), color='blue', alpha=.1)

        # make plot with title!
        plt.title(title, fontsize=16)
        plt.show() ;

    except:
        print(ValueError)
        pass

def sarima_predict_plot_seasonal(df, feature, year, ndiff_df, param_df, title='title', figsize=(15,5), order=None, d=None, seasonal_order=None, ci=True):

    # create train and test sets
    n_rows = round(len(df)*0.9)
    train = df[feature][0:n_rows]
    test = df[feature][n_rows:]

    # find ndiffs for stationarity from ndiff dataframe
    if d is None:
        d = ndiff_df.loc[ndiff_df['audio_feature'] == feature, 'ndiffs for stationarity'].iloc[0]
    print(f'd = {d}')

    # find order from arima parameters dataframe
    if order is None:
        order = param_df.loc[param_df['audio_feature'] == feature, 'order'].iloc[0]
    print(f'order = {order}')

    # find seasonal order from arima parameters dataframe
    if seasonal_order is None:
        sea_string = param_df.loc[param_df['audio_feature'] == feature, 'seasonal_order'].iloc[0]
        seasonal_order = tuple(map(int, sea_string.split(', ')))
    print(f'seasonal order = {seasonal_order}')

    try:

        # instantiate and fit SARIMAX model
        sarima = SARIMAX(endog=train, order=order, seasonal_order=seasonal_order).fit()

        # get predictions for train and test sets
        preds_train = sarima.predict(start=train.index[d], end=train.index[-1], typ='levels')
        preds_test = sarima.predict(start=test.index[0], end=test.index[-1], typ='levels')

        # calculate and print RMSE for train and test setes
        train_rmse = mean_squared_error(train[d::], preds_train)**0.5
        print(f'{feature.capitalize()} train RMSE ({year}) - SARIMA({order}): {train_rmse}')

        test_rmse = mean_squared_error(test, preds_test)**0.5
        print(f'{feature.capitalize()} test RMSE ({year}) - SARIMA({order}): {test_rmse}')

        # add RMSEs to arima parameters dataframe
        param_df.loc[param_df['audio_feature'] == feature, 'sarima_train_rmse'] = train_rmse
        param_df.loc[param_df['audio_feature'] == feature, 'sarima_test_rmse'] = test_rmse

        # calculate residuals
        # residuals = test - preds_test

        # set up plot
        plt.figure(figsize=figsize)

        # plot training data
        plt.plot(train, color='blue')

        # plot testing data
        plt.plot(test.index, test, color='orange')

        # plot predicted values for test set
        plt.plot(test.index, preds_test, color='green')

        # add line for the baseline model (mean value of feature)
        plt.hlines(df[feature].mean(), train.index[0], test.index[-1], color = 'grey')

        # plot confidence interval
        if ci:
            ci = 1.96 * np.std(preds_test)/np.mean(preds_test)
            plt.fill_between(test.index, (preds_test - ci), (preds_test + ci), color='blue', alpha=.1)

        # make plot with title!
        plt.title(title, fontsize=16)
        plt.show() ;


    except ValueError as ve:
        # print(ve)
        # pass
        raise


def sarima_predict_plot_exog(df, feature, year, ndiff_df, param_df, exog_var, title='title', figsize=(15,5), order=None, d=None, seasonal_order=None, ci=True):

    # find ndiffs for stationarity from ndiff dataframe
    if d is None:
        d = ndiff_df.loc[ndiff_df['audio_feature'] == feature, 'ndiffs for stationarity'].iloc[0]
    print(f'd = {d}')

    # find order from arima parameters dataframe
    if order is None:
        order = param_df.loc[param_df['audio_feature'] == feature, 'order'].iloc[0]
    print(f'order = {order}')

    # find seasonal order from arima parameters dataframe
    if seasonal_order is None:
        sea_string = param_df.loc[param_df['audio_feature'] == feature, 'seasonal_order'].iloc[0]
        seasonal_order = tuple(map(int, sea_string.split(', ')))
    print(f'seasonal order = {seasonal_order}')

    # reshape exogenous features to pass to the model
    exog = df.loc[:, exog_var]

    # create train and test sets
    n_rows = round(len(df)*0.9)
    train = df[feature][0:n_rows]
    test = df[feature][n_rows:]

    try:
        # instantiate and fit SARIMAX model
        sarima = SARIMAX(endog=train, exog=exog[0:n_rows], order=order, seasonal_order=seasonal_order).fit()

        # get predictions for train and test sets
        preds_train = sarima.predict(start=train.index[d], end=train.index[-1], typ='levels', exog=exog[0:n_rows])
        preds_test = sarima.predict(start=test.index[0], end=test.index[-1], typ='levels', exog=exog[n_rows:])

        # calculate and print RMSE for train and test setes
        train_rmse = mean_squared_error(train[d::], preds_train)**0.5
        print(f'{feature.capitalize()} train RMSE ({year}) - SARIMAX({seasonal_order}) w/ exogenous variables: {train_rmse}')

        test_rmse = mean_squared_error(test, preds_test)**0.5
        print(f'{feature.capitalize()} test RMSE ({year}) - SARIMAX({seasonal_order}) w/ exogenous variables: {test_rmse}')

        # add RMSEs to arima parameters dataframe
        param_df.loc[param_df['audio_feature'] == feature, 'exog_train_rmse'] = train_rmse
        param_df.loc[param_df['audio_feature'] == feature, 'exog_test_rmse'] = test_rmse

        # calculate residuals
        # residuals = test - preds_test

        # set up plot
        plt.figure(figsize=figsize)

        # plot training data
        plt.plot(train, color='blue')

        # plot testing data
        plt.plot(test.index, test, color='orange')

        # plot predicted values for test set
        plt.plot(test.index, preds_test, color='green')

        # add line for the baseline model (mean value of feature)
        plt.hlines(df[feature].mean(), train.index[0], test.index[-1], color = 'grey')

        # plot confidence interval
        if ci:
            ci = 1.96 * np.std(preds_test)/np.mean(preds_test)
            plt.fill_between(test.index, (preds_test - ci), (preds_test + ci), color='blue', alpha=.1)

        # make plot with title!
        plt.title(title, fontsize=16)
        plt.show() ;

    except ValueError as ve:
        print(ve)
        pass
