import os
import pandas as pd
import numpy as np


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
        self._raw_data = self.read_raw()
        self.data = self.parse_data(data=self._raw_data)

    def read_raw(self):
        """Retrieves the raw return data.

        Retrieving steps:
            all data in yearly folders and seperate daily csv.gz files
            need to loop over folders
            calls read_raw_year function which loops over seperate daily csv.gz files
            calculates return values r_t=log(p_{t}/p_{t-1})

        Args:
            self

        Returns:
            raw_data: pd.DataFrame with all the days concatenated and calculated return values
        """
        years = self._run_sett["years"]
        raw_data = self.read_raw_year(years[0])

        for year in years[1:]:
            raw_data_temp = self.read_raw_year(year)
            raw_data = pd.concat([raw_data, raw_data_temp], axis=1, join="inner")

        return raw_data

    def read_raw_year(self, year):
        """Retrieves the raw return data.

        Retrieving steps:
            need to loop over folders and over seperate daily csv.gz files
            calculates return values r_t=log(p_{t}/p_{t-1})

        Args:
            self
            year: int of year

        Returns:
            raw_data: pd.DataFrame with all the yearly days concatenated and calculated return values
        """
        sorted_strings = sorted(
            list_visible_files_with_list_comprehension(
                self._run_sett["input_dir"] + str(year)
            ),
            key=lambda x: int(x[:8]),
        )
        input_path_init = os.path.join(
            self._run_sett["input_dir"] + str(year), sorted_strings[0]
        )
        raw_data_year = (
            pd.read_csv(input_path_init, index_col="ticker", compression="gzip")
            .loc[:, "close"]
            .rename(sorted_strings[0][:8])
            .to_frame()
        )

        for day_file in sorted_strings[1:]:
            input_path = os.path.join(self._run_sett["input_dir"] + str(year), day_file)
            raw_data_year_temp = (
                pd.read_csv(input_path, index_col="ticker", compression="gzip")
                .loc[:, "close"]
                .rename(day_file[:8])
            )
            raw_data_year = pd.concat(
                [raw_data_year, raw_data_year_temp], axis=1, join="inner"
            )

        raw_data_year = (
            np.log(raw_data_year.transpose() / raw_data_year.transpose().shift(1))
            .iloc[1:,]
            .transpose()
        )

        return raw_data_year

    def parse_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Applies some global cleaning of the data.

        Error fixes:
            drop days for which more than n/10 stocks have zero return values
            drop stocks for which more than d/2 days have zero return values
            drop days for which more than n/10 stocks have higher than 1 return values
            drop stocks for which more than d/2 days have higher than 1 return values

        Args:
            data: raw input data

        Returns:
            data: parsed input data
        """
        thresh_n = data.shape[0] // 10
        thresh_d = data.shape[1] // 2
        zero_return_df = data == 0
        bigger_than_one_return_df = data > 1

        valid_days = ~(
            (zero_return_df.sum() > thresh_n)
            | (bigger_than_one_return_df.sum() > thresh_n)
        )
        valid_stocks = ~(
            (zero_return_df.sum(axis=1) > thresh_d)
            | (bigger_than_one_return_df.sum(axis=1) > thresh_d)
        )
        parsed_data = data.loc[valid_stocks, valid_days]

        return parsed_data


def list_visible_files_with_list_comprehension(directory):
    visible_files = [file for file in os.listdir(directory) if not file.startswith(".")]
    return visible_files
