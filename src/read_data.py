import os
import pandas as pd


class ReadData:
    """This class loads and preprocesses the data.

    First the data is loaded from Excel and converted to CSV. This is done only once. Next some global preprocessing of
    the input dataframe is applied. Features of the data, such as time_series, are calculated on request and saved in a
    library.

    Attributes:
        data (pd.DataFrame): parsed input dataset
        data_library (dict): contains many different data structures, such as time series
    """

    def __init__(self, run_sett: dict):
        """constructs a ReadData object"""
        self._run_sett = run_sett

        # self.data = self.parse_data(data=raw_data)
        self.data = self.read_raw()
        self.data_library = {}

    def read_raw(self):
        sorted_strings = sorted(
            list_visible_files_with_list_comprehension(self._run_sett["input_dir"]),
            key=lambda x: int(x[:8]),
        )
        for day_file in sorted_strings:
            input_path = os.path.join(self._run_sett["input_dir"], day_file)
            raw_data = pd.read_csv(
                input_path, index_col="ticker", compression="gzip"
            ).loc[:, "close"]
            # now keep merging days to that series in dfs

    def parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies some global cleaning of the data.

        Error fixes:
            ...
            ...
            ...

        Args:
            data: raw input data

        Returns:
            data: parsed input data
        """


def list_visible_files_with_list_comprehension(directory):
    visible_files = [file for file in os.listdir(directory) if not file.startswith(".")]
    return visible_files
