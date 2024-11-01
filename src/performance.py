import numpy as np
import pandas as pd


class Performance:
    """Base class of performance metrics"""

    def __init__(self, data_obj, pred_obj):
        self._data_obj = data_obj
        self._pred_obj = pred_obj


class PnL(Performance):
    """Calculates PnL of different prediciton series"""

    def __init(self, data_obj, pred_obj):
        super().__init__(data_obj, pred_obj)

    def calculate(self):
        all_data = self._data_obj.data
        test = all_data.iloc[
            :, pd.to_datetime(all_data.columns) >= pd.to_datetime("20210601")
        ]

        res = self._pred_obj * test.transpose().iloc[:, :10].values
        pnl_per_asset = np.cumsum(res, axis=0)
        pnl_across_assets = pnl_per_asset.sum(axis=1)

        return pnl_across_assets, pd.to_datetime(test.columns)


class SR(PnL):
    """Calculates SR of different prediciton series"""

    def __init(self, data_obj, pred_obj, pnl):
        super().__init__(data_obj, pred_obj, pnl)

    def calculate(self):
        return (np.mean(self._pnl) / np.std(self._pnl)) * np.sqrt(252)
