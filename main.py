import os
import yaml
from src.read_data import ReadData
from src.model import VAM
from src.performance import PnL
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])
    data_obj = ReadData(run_sett=run_sett)
    pred_obj = VAM(data_obj).calculate()
    pnl, dates = PnL(data_obj, pred_obj).calculate()
    plt.plot(dates, pnl)
    a = 5
