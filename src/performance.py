import numpy as np
import pandas as pd


class Performance:
    """Base class of performance metrics"""

    def __init__(self, data_obj, pred_obj):
        self.data_obj = data_obj
        self.pred_obj = pred_obj


class PnL(Performance):
    """Calculates PnL of different prediciton series"""

    def __init__(self, data_obj, pred_obj):
        super().__init__(data_obj, pred_obj)
        self.pnl_res = self.calculate()
        self.sr_across_assets = yearly_sharpe_ratio(
            self.pnl_res["daily_profit_or_loss_across_assets"]
        )
        self.sr_per_asset = yearly_sharpe_ratio(
            self.pnl_res["daily_profit_or_loss_per_asset"]
        )

    def calculate(self):
        all_data = self.data_obj.data
        test = all_data.iloc[
            :, pd.to_datetime(all_data.columns) >= pd.to_datetime(self.pred_obj.cutoff)
        ]
        sgn_forecasts = self.pred_obj.sgn_forecasts
        asset_names = test.index.to_list()[:10]  # for now due to computational time
        res = (
            sgn_forecasts * test.transpose().iloc[:, :10].values
        )  # for now due to computational time
        res_per_day_across_assets = res.sum(axis=1)
        pnl_per_asset = np.cumsum(res, axis=0)
        pnl_across_assets = pnl_per_asset.sum(axis=1)

        return {
            "pnl_across_assets": pnl_across_assets,
            "daily_profit_or_loss_across_assets": res_per_day_across_assets,
            "daily_profit_or_loss_per_asset": res,
            "pnl_per_asset": pnl_per_asset,
            "dates": pd.to_datetime(test.columns),
            "asset_names": asset_names,
        }


def yearly_sharpe_ratio(pnl_ts):
    return (np.mean(pnl_ts, axis=0) / np.std(pnl_ts, axis=0)) * np.sqrt(252)
