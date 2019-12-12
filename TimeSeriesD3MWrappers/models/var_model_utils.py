from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class Arima:
    def __init__(self, seasonal=True, seasonal_differencing=1, max_order=5):
        """initialize ARIMA class
        
        Keyword Arguments:
            seasonal {bool} -- whether time series has seasonal component (default: {True})
            seasonal_differencing {int} -- period for seasonal differencing (default: {1})
            max_order {int} -- maximum order of p and q terms on which to fit model (default: {5})
        """

        self.seasonal = seasonal
        self.seasonal_differencing = seasonal_differencing
        self.max_order = max_order

    def fit(self, train):
        """fit ARIMA model on training data, automatically selecting p (AR), q (MA), 
            P (AR - seasonal), Q (MA - seasonal), d, and D (differencing) amongst other parameters
            based on AIC
        
        Arguments:
            np array -- endogenous time series on which model should select parameters and fit
        """

        self.arima_model = auto_arima(
            train,
            start_p=1,
            start_q=1,
            max_p=self.max_order,
            max_q=self.max_order,
            m=self.seasonal_differencing,
            seasonal=self.seasonal,
            stepwise=True,
            suppress_warnings=True,
        )
        self.arima_model.fit(train)

    def predict(self, n_periods=1, return_conf_int=False, alpha=0.05):
        """forecasts the time series n_periods into the future
        
        Keyword Arguments:
            n_periods {int} -- number of periods to forecast into the future (default: {1})
            return_conf_int {bool} -- whether to return confidence intervals instead of 
                forecasts
            alpha {float} -- significance level for confidence interval, i.e. alpha = 0.05 
                returns a 95% confdience interval from alpha / 2 to 1 - (alpha / 2) 
                (default: {0.05})
        
        Returns:
            np array -- (n, 1) time series forecast n_periods into the future
                OR (n, 2) if returning confidence interval forecast
        """
        if return_conf_int:
            forecast, interval = self.arima_model.predict(
                n_periods=n_periods, return_conf_int=True, alpha=alpha
            )
            return (
                forecast,
                interval[:, 0],
                interval[:, 1],
            )
        else:
            return self.arima_model.predict(n_periods=n_periods)

    def get_absolute_value_params(self):
        """get absolute value of trend, AR, and MA parameters of 
            fit ARIMA model (no exogenous variables)
        
        Returns:
            pandas df -- df with column for each parameter
        """

        try:
            ar_count = self.arima_model.arparams().shape[0]
        except AttributeError:
            logger.debug("There are no ar parameters in this model")
            ar_count = 0
        try:
            ma_count = self.arima_model.maparams().shape[0]
        except AttributeError:
            logger.debug("There are no ma parameters in this model")
            ma_count = 0
        trend_count = self.arima_model.df_model() - ar_count - ma_count

        ar_cols = ["ar_" + str(i + 1) for i in range(ar_count)]
        ma_cols = ["ma_" + str(i + 1) for i in range(ma_count)]
        trend_cols = ["trend_" + str(i) for i in range(trend_count)]

        return pd.DataFrame(
            np.absolute(self.arima_model.params().reshape(1, -1)),
            columns=trend_cols + ar_cols + ma_cols,
        )
