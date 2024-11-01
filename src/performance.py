import numpy as np


class Performance:
    """Base class of performance metrics"""

    def __init__(self, data_obj, pred_obj):
        self._data_obj = data_obj
        self._pred_obj = pred_obj


class PnL(Performance):
    """Calculates PnL of different prediciton series"""

    def __init(self, data_obj, pred_obj):
        super().__init__(data_obj, pred_obj)
        self._pnl = self.calculate()

    def calculate(self):
        frets = self._data_obj.data

        sign_preds = np.sign(self._pred_obj)

        pnl = sign_preds * frets

        return pnl


class SR(PnL):
    """Calculates SR of different prediciton series"""

    def __init(self, data_obj, pred_obj, pnl):
        super().__init__(data_obj, pred_obj, pnl)

    def calculate(self):
        return (np.mean(self._pnl) / np.std(self._pnl)) * np.sqrt(252)
