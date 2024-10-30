import numpy as np


class Model:
    """Base class of models"""

    def __init__(self, data_obj, name):
        self._data_obj = data_obj
        self._name = name


class VAM(Model):
    """Vector Asset Model (VAM)"""

    def __init(self, data_obj, name, truncate=False):
        super().__init__(data_obj, name)
        self._truncate = truncate

    def calculate(self, confidence):
        time_series = self._data_obj.yearly_ts()
        a = np.sqrt(5)
        b = np.zeros(5)


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
