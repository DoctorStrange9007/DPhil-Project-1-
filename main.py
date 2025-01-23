import os
import yaml
from src.read_data import ReadData
from src.model import UAM
from src.performance import PnL
from src import utils
from src.embedding import Spectral

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])
    data_obj = ReadData(run_sett=run_sett)
    if run_sett["embedding"] == "spectral":
        clustered_dfs = Spectral(run_sett, data_obj).clustered_dfs
    pred_obj = UAM(run_sett, clustered_dfs)
    pnl_obj = PnL(data_obj, pred_obj)
    utils.plot_pnl_with_sharpe(pnl_obj)
    a = 6
