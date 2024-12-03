import os
import yaml
from src.read_data import ReadData
from src.model import VAM
from src.performance import PnL
from src import utils

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])
    data_obj = ReadData(run_sett=run_sett)
    pred_obj = VAM(run_sett, data_obj)  # change to general model Class when more models
    pnl_obj = PnL(data_obj, pred_obj)
    utils.plot_pnl_with_sharpe(pnl_obj)
    a = 5
