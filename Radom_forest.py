import pandas as pd
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as mean_absolute_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sktime.forecasting.model_selection import temporal_train_test_split
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from prophet import Prophet
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error as smape_loss
from sktime.forecasting.arima import AutoARIMA
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestRegressor

class TimeSeriesAnalysis:
    def Randomforest(self):
        data = pd.read_csv(r"C:\Users\admin\Downloads\Electric_Production.csv")
        productions = data['IPG2211A2N'].values
        productions = scale(productions)

        last_days = []
        today = []
        step = 1
        max_len = 5
        for idx in range(0, len(productions) - max_len, step):
            last_days.append(productions[idx: idx + max_len])
            today.append(productions[idx + max_len])

        x = np.array(last_days)
        y = np.array(today)

        split = x.shape[0] // 5
        X_train = x[:-split]
        X_test = x[-split:]
        y_train = y[:-split]
        y_test = y[-split:]


        regr = RandomForestRegressor(random_state=42, n_estimators=50)

        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        
        ant = X_test.shape[0] + 300
        x_axis = np.arange(ant) + len(productions) - ant
        x_axis_nxt = np.arange(len(y_pred)) + y_train.shape[0]
        plt.plot(x_axis, productions[-ant:], linewidth=2)
        plt.plot(x_axis_nxt, y_pred, linewidth=2)
        plt.show()
        
production = TimeSeriesAnalysis()
#production.get_time_series('IPG2211A2N')
production.Randomforest()
