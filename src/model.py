import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd


class Model:
    """Base class of models"""

    def __init__(self, run_sett: dict, data_obj):
        self.data_obj = data_obj
        self._run_sett = run_sett


class VAM(Model):
    """Vector Asset Model (VAM)"""

    def __init__(self, run_sett: dict, data_obj):
        super().__init__(run_sett, data_obj)
        self.cutoff = "20210104"
        self.model_sett = self._run_sett["models"]["VAM"]
        self.model_lag = self.model_sett["p"]
        self.sgn_forecasts = self.calculate(delta=self.model_sett["delta"])

    def calculate(self, delta):
        all_data = self.data_obj.data
        train = all_data.iloc[
            :, pd.to_datetime(all_data.columns) < pd.to_datetime(self.cutoff)
        ]  # Par
        test = all_data.iloc[
            :, pd.to_datetime(all_data.columns) >= pd.to_datetime(self.cutoff)
        ]  # Paralelise the training and testing phase and look lookbackwindow and refit
        # Start by using a Lookback window = 252
        # S = 21 (refit every month)
        # use new data to predict each new day

        adj_train = train.transpose().iloc[:, :10]  # for now due to computational time
        model = VAR(adj_train)
        results = model.fit(self.model_lag)
        lag_order = results.k_ar
        forecast_input = adj_train.values[-lag_order:]
        forecasts = results.forecast(forecast_input, steps=test.shape[1])
        sgn_forecasts = np.where(forecasts > delta, 1, -1)

        return sgn_forecasts


class SAM(Model):
    """Single Asset Model (SAM)"""

    def __init__(self, data_obj, name, truncate=False):
        super().__init__(data_obj, name)
        self._truncate = truncate

    def calculate(self, confidence):
        time_series = self.data_obj.yearly_ts()
        a = np.sqrt(5)
        b = np.zeros(5)


class CAM(Model):
    """Cluster Asset Model (CAM)"""

    def __init__(self, data_obj, name, truncate=False):
        super().__init__(data_obj, name)
        self._truncate = truncate

    def calculate(self, confidence):
        time_series = self.data_obj.yearly_ts()
        a = np.sqrt(5)
        b = np.zeros(5)
