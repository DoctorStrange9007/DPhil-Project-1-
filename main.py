import os
import yaml
from src.read_data import ReadData

if __name__ == "__main__":
    with open("settings.yaml", "r") as f:
        run_sett = yaml.safe_load(f)
    if not os.path.exists(run_sett["output_dir"]):
        os.makedirs(run_sett["output_dir"])
    data_obj = ReadData(run_sett=run_sett)
    print(data_obj.data)
