import numpy as np
import pandas as pd


class Performance:
    """Base class for financial performance metrics calculation.

    Provides a common interface for different types of performance analysis
    of predicted trading signals.

    Parameters
    ----------
    data_obj : object
        Object containing the original financial data
    pred_obj : object
        Object containing the model predictions and trading signals
    """

    def __init__(self, data_obj, pred_obj):
        self._data_obj = data_obj
        self.pred_obj = pred_obj


class PnL(Performance):
    """Calculate and analyze Profit and Loss (PnL) metrics for trading signals.

    Computes PnL metrics across different prediction series and assets,
    including both aggregate and per-asset performance measures.

    Parameters
    ----------
    data_obj : object
        Object containing the original financial data
    pred_obj : object
        Object containing the model predictions and trading signals

    Attributes
    ----------
    pnl_res : dict
        Dictionary containing PnL results for each CAM model
    """

    def __init__(self, data_obj, pred_obj):
        super().__init__(data_obj, pred_obj)
        self.pnl_res = self.combine_cams()

    def combine_cams(self):
        """Combine PnL results across all Cluster Asset Models (CAMs).

        Returns
        -------
        dict
            Dictionary where keys are CAM identifiers and values are
            dictionaries containing PnL metrics for each CAM

        Notes
        -----
        Processes each CAM separately and combines results into a single
        dictionary for easy comparison and analysis
        """
        cams_res = {}

        for cam_key, cam_obj in self.pred_obj.cam_objs.items():
            cam_res = self.calculate_cam(cam_obj)
            cams_res[cam_key] = cam_res

        return cams_res

    def calculate_cam(self, cam_obj):
        """Calculate PnL metrics for a single Cluster Asset Model.

        Parameters
        ----------
        cam_obj : object
            CAM object containing cluster data and forecasts

        Returns
        -------
        dict
            Dictionary containing the following PnL metrics:
            - pnl_across_assets : numpy.ndarray
                Cumulative PnL summed across all assets
            - daily_profit_or_loss_across_assets : numpy.ndarray
                Daily PnL summed across all assets
            - daily_profit_or_loss_per_asset : numpy.ndarray
                Daily PnL for each individual asset
            - pnl_per_asset : numpy.ndarray
                Cumulative PnL for each individual asset
            - dates : array-like
                Dates corresponding to the PnL calculations
            - asset_names : list
                Names of assets in the analysis

        Notes
        -----
        PnL is calculated by multiplying signal signs (-1/1) with actual returns.
        """
        all_data = cam_obj.cluster_data.copy()
        dates = pd.to_datetime(all_data.columns)
        forecasted_dates = cam_obj.forecasted_dates
        true_data = all_data.loc[:, dates.isin(forecasted_dates)]
        sgn_forecasts = cam_obj.sgn_forecasts
        sgn_forecasts = np.concatenate(sgn_forecasts, axis=0)
        sgn_forecasts = sgn_forecasts[:-1, :]
        asset_names = all_data.index.to_list()
        res = sgn_forecasts * true_data.transpose().values
        res_per_day_across_assets = res.sum(axis=1)
        pnl_per_asset = np.cumsum(res, axis=0)
        pnl_across_assets = pnl_per_asset.sum(axis=1)

        return {
            "pnl_across_assets": pnl_across_assets,
            "daily_profit_or_loss_across_assets": res_per_day_across_assets,
            "daily_profit_or_loss_per_asset": res,
            "pnl_per_asset": pnl_per_asset,
            "dates": forecasted_dates,
            "asset_names": asset_names,
        }
