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
        all_data = self.data_obj.data[:10]  # for now due to computational time
        dates = pd.to_datetime(all_data.columns)
        forecasted_dates = self.pred_obj.forecasted_dates
        true_data = all_data.loc[:, dates.isin(forecasted_dates)]
        sgn_forecasts = self.pred_obj.sgn_forecasts
        sgn_forecasts = np.concatenate(sgn_forecasts, axis=0)
        sgn_forecasts = sgn_forecasts[:-1, :]
        asset_names = all_data.index.to_list()[:10]  # for now due to computational time
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


def yearly_sharpe_ratio(pnl_ts):
    return (np.mean(pnl_ts, axis=0) / np.std(pnl_ts, axis=0)) * np.sqrt(pnl_ts.shape[0])
