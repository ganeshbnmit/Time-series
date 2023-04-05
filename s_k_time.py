from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.compose import make_reduction
from xgboost import XGBRegressor
import datetime as dt
import json
import numpy as np
import pandas as pd
from datetime import datetime
y = load_airline()
#y = pd.read_csv(r"C:\Users\admin\Downloads\Electric_Production.csv")
#y['DATE']=pd.to_datetime(y['DATE'],infer_datetime_format=True)
#indexedy = y.set_index(['DATE'])

class TimeSeriesAnalysis:
    def forecast_naive(self,timeseries_data, horizon, seasonality_period):
        y = timeseries_data
        fh = np.arange(1, horizon+1)
        forecaster = NaiveForecaster(strategy="last", sp=seasonality_period)
        forecaster.fit(y, fh=fh)
        y_pred = forecaster.predict()
        plot_series(y, y_pred, labels=["y", "y_pred"])
    
    def forecast_kneighbors_test(self,timeseries_data, test_size):
        y = timeseries_data
        y_train, y_test = temporal_train_test_split(y, test_size=test_size)
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        regressor = KNeighborsRegressor(n_neighbors=1)
        forecaster = make_reduction(regressor, window_length=15, strategy="recursive")
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)
        plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
        print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_test, y_pred, symmetric=False))
        
    def forecast_arima_test(self,timeseries_data, test_size):
        y = timeseries_data
        y_train, y_test = temporal_train_test_split(y, test_size=test_size)
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        forecaster = ARIMA(
            order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True
        )

        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)
        plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
        print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_test, y_pred, symmetric=False))
        
    def xgboost(self,timeseries_data, test_size):
        y = timeseries_data
        y_train, y_test = temporal_train_test_split(y, test_size=test_size)
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        
        regressor = XGBRegressor(objective='reg:squarederror', random_state=42)

        forecaster = make_reduction(regressor, strategy="recursive", window_length=52)
        forecaster.fit(y_train)
        y_pred = forecaster.predict(fh)
        plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"])
        print("Mean Absolute Percentage Error (MAPE):", mean_absolute_percentage_error(y_test, y_pred, symmetric=False))

passengers=TimeSeriesAnalysis()
passengers.forecast_arima_test(timeseries_data=y, test_size=36)

passengers=TimeSeriesAnalysis()
passengers.forecast_naive(timeseries_data=y, horizon=36, seasonality_period=12)

passengers=TimeSeriesAnalysis()
passengers.forecast_kneighbors_test(timeseries_data=y, test_size=36)

passengers=TimeSeriesAnalysis()
passengers.xgboost(timeseries_data=y, test_size=36)
