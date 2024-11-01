import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd


class Model:
    """Base class of models"""

    def __init__(self, data_obj):
        self._data_obj = data_obj


class VAM(Model):
    """Vector Asset Model (VAM)"""

    def __init(self, data_obj):
        super().__init__(data_obj)

    def calculate(self):
        all_data = self._data_obj.data
        train = all_data.iloc[
            :, pd.to_datetime(all_data.columns) < pd.to_datetime("20210601")
        ]
        test = all_data.iloc[
            :, pd.to_datetime(all_data.columns) >= pd.to_datetime("20210601")
        ]

        adj_train = train.transpose().iloc[:, :10]  # for now due to computational time
        model = VAR(adj_train)
        results = model.fit(1)
        lag_order = results.k_ar
        forecast_input = adj_train.values[-lag_order:]
        forecast = results.forecast(forecast_input, steps=test.shape[1])
        sgn_forecast = np.sign(forecast)

        return sgn_forecast


class SAM(Model):
    """Single Asset Model (SAM)"""

    def __init(self, data_obj, name, truncate=False):
        super().__init__(data_obj, name)
        self._truncate = truncate

    def calculate(self, confidence):
        time_series = self._data_obj.yearly_ts()
        a = np.sqrt(5)
        b = np.zeros(5)


class CAM(Model):
    """Cluster Asset Model (CAM)"""

    def __init(self, data_obj, name, truncate=False):
        super().__init__(data_obj, name)
        self._truncate = truncate

    def calculate(self, confidence):
        time_series = self._data_obj.yearly_ts()
        a = np.sqrt(5)
        b = np.zeros(5)
