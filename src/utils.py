import matplotlib.pyplot as plt
import pandas as pd


def plot_pnl_with_sharpe(pnl_obj):
    """Plots a PnL curve and annotates it with the Sharpe ratio.

    Attributes:
        pnl_values (np.array): Array of PnL values (floats or ints).
        dates (np.array): Array of dates (datetime objects or strings).
        sharpe_ratio (float): Float value representing the Sharpe ratio.
    """
    asset_names = pnl_obj.pnl_res["asset_names"]
    pnl_values_across_assets = pnl_obj.pnl_res["pnl_across_assets"]
    pnl_values_per_asset = pnl_obj.pnl_res["pnl_per_asset"]
    pnl_dates = pnl_obj.pnl_res["dates"]
    sharpe_ratio_across_assets = pnl_obj.sr_across_assets
    sharpe_ratio_per_asset = pnl_obj.sr_per_asset
    model_name = pnl_obj.pred_obj.model_sett["name"]
    model_lag = pnl_obj.pred_obj.model_lag

    df = pd.DataFrame({"Date": pnl_dates, "PnL_VAM": pnl_values_across_assets})
    df["Date"] = pd.to_datetime(df["Date"])
    for asset, asset_pnl in zip(asset_names, pnl_values_per_asset.transpose()):
        df["PnL_" + asset] = asset_pnl

    plt.figure(figsize=(12, 6))
    label = f"{df.columns[1]} ({sharpe_ratio_across_assets})"
    plt.plot(df["Date"], df[df.columns[1]], label=label, linewidth=2)

    for column, sharpe_ratio in zip(df.columns[2:], sharpe_ratio_per_asset):
        label = f"{column} ({sharpe_ratio})"
        plt.plot(df["Date"], df[column], label=label, linewidth=2)

    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    plt.title(
        "PnL Curves of " + model_name + " predictions with p = " + str(model_lag),
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("PnL", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()

    # Show the plot
    plt.show()
