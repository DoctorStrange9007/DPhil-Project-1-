import numpy as np
from statsmodels.tsa.api import VAR
import pandas as pd
import multiprocessing as mp
import warnings
from statsmodels.tsa.base.tsa_model import ValueWarning

warnings.filterwarnings("ignore", category=ValueWarning)


class Model:
    """Base class for all financial models.

    Provides common functionality and structure for derived model classes.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs : dict
        Dictionary of DataFrames containing clustered financial data
    """

    def __init__(self, run_sett: dict, clustered_dfs):
        self.clustered_dfs = clustered_dfs
        self._run_sett = run_sett


class UAM(Model):
    """Universal Asset Model (UAM) for multi-cluster financial modeling.

    Manages multiple Cluster Asset Models (CAM) to provide consolidated
    predictions across different asset clusters.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs : dict
        Dictionary of DataFrames containing clustered financial data
    """

    def __init__(self, run_sett: dict, clustered_dfs):
        super().__init__(run_sett, clustered_dfs)
        self.cam_objs = self.combine_cams()
        self.model_sett = self._run_sett["models"]["UAM"]

    def combine_cams(self):
        cam_objs = {}
        for cluster_label, clustered_df in self.clustered_dfs.items():
            cam_obj = CAM(self._run_sett, self.clustered_dfs, clustered_df)
            cam_objs[cluster_label] = cam_obj

        return cam_objs


class CAM(Model):
    """Cluster Asset Model (CAM) for single-cluster financial modeling.

    Implements a Vector Autoregression (VAR) model for predicting asset
    movements within a specific cluster using rolling window analysis.

    Parameters
    ----------
    run_sett : dict
        Dictionary containing all runtime settings and parameters
    clustered_dfs : dict
        Dictionary of DataFrames containing clustered financial data
    cluster_data : pd.DataFrame
        Data for the specific cluster being modeled
    """

    def __init__(self, run_sett: dict, clustered_dfs, cluster_data):
        super().__init__(run_sett, clustered_dfs)
        self.cutoff = "20210104"
        self.model_sett = self._run_sett["models"]["CAM"]
        self.model_lag = self.model_sett["p"]
        self.cluster_data = cluster_data
        self.sgn_forecasts, self.forecasted_dates = self.calculate()

    def calculate(self):
        """Execute the main calculation pipeline for the CAM model.

        Returns
        -------
        tuple
            - sgn_forecasts : list of numpy.ndarray
                Binary predictions (-1 or 1) for asset movements
            - forecasted_dates : pandas.DatetimeIndex
                Dates corresponding to the forecasts
        """
        all_data = self.cluster_data.copy()

        sgn_forecasts, forcasted_date_sets = self.rolling_training_testing(
            self.model_sett["L"], self.model_sett["S"], all_data
        )

        return sgn_forecasts, forcasted_date_sets

    def rolling_training_testing(self, L, S, all_data):
        """Perform rolling window training and testing using parallel processing.

        Parameters
        ----------
        L : int
            Lookback window size (e.g., 252 trading days)
        S : int
            Step size for model refitting (e.g., 21 trading days)
        all_data : pd.DataFrame
            Complete dataset for training and testing

        Returns
        -------
        tuple
            - sgn_forecasts : list of numpy.ndarray
                Binary predictions (-1 or 1) for each rolling window
            - forecasted_dates : pandas.DatetimeIndex
                Dates corresponding to the forecasts
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
        forecasted_dates = dates[L:]
        test_data_sets = [
            all_data.loc[:, dates.isin(date_set)] for date_set in test_date_sets
        ]

        pool = mp.Pool(mp.cpu_count())
        results = pool.map(self.train, [data_set for data_set in train_data_sets])
        sgn_forecasts = [
            pool.apply(test, args=(test_data_set, result))
            for test_data_set, result in zip(test_data_sets, results)
        ]
        pool.close()

        return sgn_forecasts, forecasted_dates

    def train(self, data):
        """Train a VAR model on a single window of data.

        Parameters
        ----------
        data : pd.DataFrame
            Training data for the current window

        Returns
        -------
        statsmodels.tsa.vector_ar.var_model.VARResults
            Fitted VAR model results

        Notes
        -----
        Currently limited to first 20 columns for computational efficiency
        """
        adj_train = data.transpose().iloc[:, :20]  # for now due to computational time
        model = VAR(adj_train)
        result = model.fit(self.model_lag)

        return result


def test(data, result):
    """Generate predictions using a trained VAR model.

    Parameters
    ----------
    data : pd.DataFrame
        Test data to generate predictions for
    result : statsmodels.tsa.vector_ar.var_model.VARResults
        Fitted VAR model

    Returns
    -------
    numpy.ndarray
        Binary predictions (-1 or 1) for asset movements

    Notes
    -----
    - Currently limited to first 20 columns for computational efficiency
    - Uses 0.00005 as the threshold for positive/negative prediction
    """
    all_forecasts = []
    for step in range(data.shape[1]):
        adj_test = data.transpose().iloc[:, :20]  # for now due to computational time
        forecast = result.forecast(
            adj_test.values[step : step + 1], steps=1
        )  # step + 1 := step + p
        sgn_forecast = np.where(
            forecast > 0.00005, 1, -1
        )  # can change delta (e.g. betsize)
        all_forecasts.append(*sgn_forecast)

    return np.array(all_forecasts)
