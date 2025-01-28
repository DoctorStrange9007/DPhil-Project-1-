import numpy as np
from statsmodels.tsa.api import VAR
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
    clustered_dfs_sets : list
        List of dictionaries of DataFrames containing clustered financial data
    """

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        forecasted_dates,
    ):
        self.clustered_dfs_train_sets = clustered_dfs_train_sets
        self.clustered_dfs_test_sets = clustered_dfs_test_sets
        self.forecasted_dates = forecasted_dates
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

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        forecasted_dates,
    ):
        super().__init__(
            run_sett,
            clustered_dfs_train_sets,
            clustered_dfs_test_sets,
            forecasted_dates,
        )
        (
            self.train_data_per_cam,
            self.test_data_per_cam,
        ) = self.create_data_sets_per_cam()
        self.cam_objs = self.combine_cams()
        self.model_sett = self._run_sett["models"]["UAM"]

    def create_data_sets_per_cam(self):
        train_data_per_cam = {}
        test_data_per_cam = {}
        for cluster_label in list(self.clustered_dfs_train_sets[0].keys()):
            train_data_per_cam[cluster_label] = [
                d[cluster_label] for d in self.clustered_dfs_train_sets
            ]
            test_data_per_cam[cluster_label] = [
                d[cluster_label] for d in self.clustered_dfs_test_sets
            ]

        return train_data_per_cam, test_data_per_cam

    def combine_cams(self):
        cam_objs = {}
        for cluster_label in list(self.train_data_per_cam.keys()):
            cluster_train_sets = self.train_data_per_cam[cluster_label]
            cluster_test_sets = self.test_data_per_cam[cluster_label]
            cam_obj = CAM(
                self._run_sett,
                self.clustered_dfs_train_sets,
                self.clustered_dfs_test_sets,
                cluster_train_sets,
                cluster_test_sets,
                self.forecasted_dates,
            )
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

    def __init__(
        self,
        run_sett: dict,
        clustered_dfs_train_sets,
        clustered_dfs_test_sets,
        cluster_train_sets,
        cluster_test_sets,
        forecasted_dates,
    ):
        super().__init__(
            run_sett,
            clustered_dfs_train_sets,
            clustered_dfs_test_sets,
            forecasted_dates,
        )
        self.model_sett = self._run_sett["models"]["CAM"]
        self.model_lag = self.model_sett["p"]
        self.forecasted_dates = forecasted_dates
        self.cluster_train_data = cluster_train_sets
        self.cluster_test_data = cluster_test_sets
        self.sgn_forecasts = self.calculate()

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

        pool = mp.Pool(mp.cpu_count())
        results = pool.map(
            self.train, [data_set for data_set in self.cluster_train_data]
        )
        sgn_forecasts = [
            pool.apply(test, args=(test_data_set, result))
            for test_data_set, result in zip(self.cluster_test_data, results)
        ]
        pool.close()

        return sgn_forecasts

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
        adj_train = data.transpose()
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
        adj_test = data.transpose()
        forecast = result.forecast(
            adj_test.values[step : step + 1], steps=1
        )  # step + 1 := step + p
        sgn_forecast = np.where(
            forecast > 0.00005, 1, -1
        )  # can change delta (e.g. betsize)
        all_forecasts.append(*sgn_forecast)

    return np.array(all_forecasts)
