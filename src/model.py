import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd
import multiprocessing as mp


class Model:
    """Base class of models"""

    def __init__(self, run_sett: dict, data_obj):
        self.data_obj = data_obj
        self._run_sett = run_sett


class UAM(Model):
    """Universal Asset Model (UAM)"""

    def __init__(self, run_sett: dict, data_obj):
        super().__init__(run_sett, data_obj)
        self.cutoff = "20210104"
        self.model_sett = self._run_sett["models"]["UAM"]
        self.model_lag = self.model_sett["p"]
        self.sgn_forecasts = self.calculate(delta=self.model_sett["delta"])

    def calculate(self, delta):
        all_data = self.data_obj.data

        sgn_forecasts = self.rolling_training_testing(
            self.model_sett["L"], self.model_sett["S"], all_data
        )

        return sgn_forecasts

    def rolling_training_testing(self, L, S, all_data):
        """Overarching training function, calls the parralisation function.

        Attributes:
            L (int): Int, lookback window. {252}
            S (int): Int, days after which the model needs to be refitted. {21}
            data (np.array): Array with all the training data.
        """
        nr_models = all_data.shape[1] // S
        dates = pd.to_datetime(all_data.columns)

        train_date_sets = [dates[i : i + L] for i in range(0, len(dates) - L, S)][
            :nr_models
        ]
        train_data_sets = [
            all_data.loc[:, dates.isin(date_set)] for date_set in train_date_sets
        ]

        test_date_sets = [dates[i : i + S] for i in range(L - 1, len(dates), S)]
        test_data_sets = [
            all_data.loc[:, dates.isin(date_set)] for date_set in test_date_sets
        ]

        pool = mp.Pool(mp.cpu_count())
        results = [
            pool.apply(self.train, args=(data_set)) for data_set in train_data_sets
        ]
        sgn_forecasts = [
            pool.apply(self.train, args=(result, data_set))
            for data_set, result in zip(test_data_sets, results)
        ]
        pool.close()

        return sgn_forecasts

    def train(self, data):
        """General training function, used within parallelisation.

        Attributes:
            data (np.array): Array with the training data.
        """
        adj_train = data.transpose().iloc[:, :10]  # for now due to computational time
        model = VAR(adj_train)
        result = model.fit(self.model_lag)

        return result

    def test(self, result, data):
        """General testing function, used within parallelisation.

        Attributes:
            result (VAR.fit): Result from VAR fitting model
            data (np.array): Array with the training data.
        """
        lag_order = result.k_ar
        all_forecasts = []
        for step in range(data.shape[1]):
            adj_test = data.transpose().iloc[
                :, :10
            ]  # for now due to computational time
            forecast = result.forecast(adj_test.values[step : step + 1], steps=1)
            sgn_forecast = np.where(
                forecast > 0.00005, 1, -1
            )  # can change delta (e.g. betsize)
            all_forecasts.append(*sgn_forecast)

        return np.array(all_forecasts)

        forecast_input = adj_train.values[-lag_order:]
        forecasts = results.forecast(forecast_input, steps=test.shape[1])
        sgn_forecasts = np.where(forecasts > delta, 1, -1)
