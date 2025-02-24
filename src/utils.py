import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_pnl_with_sharpe(pnl_obj):
    """Plot PnL curves for all CAMs and the combined UAM with Sharpe ratios.

    Creates a comprehensive visualization showing the performance of individual
    Cluster Asset Models (CAMs) and the Universal Asset Model (UAM), with
    each curve annotated with its Sharpe ratio.

    Parameters
    ----------
    pnl_obj : object
        Object containing PnL results with attributes:
        - pred_obj.model_sett['p'] : int
            Model lag parameter
        - pnl_res : dict
            Dictionary of PnL results for each CAM

    Returns
    -------
    None
        Displays the plot using matplotlib

    Notes
    -----
    - Combines PnL across all CAMs to create the UAM curve
    - Includes a horizontal line at y=0 for reference
    - Grid lines are added for better readability
    """
    model_lag = pnl_obj.pred_obj.model_sett["p"]
    plt.figure(figsize=(12, 6))

    for i, (cam_label, cam_res) in enumerate(pnl_obj.pnl_res.items()):
        if i == 0:
            df_UAM = plot_pnl_with_sharpe_per_cam(cam_label, cam_res)
            df_UAM["Dates"] = cam_res["dates"]
        else:
            df_UAM["PnL_CAM_" + str(cam_label)] = plot_pnl_with_sharpe_per_cam(
                cam_label, cam_res
            )

    # plot UAM
    label = f"PnL_UAM_(SR:{yearly_sharpe_ratio(df_UAM.drop(columns=['Dates']).sum(axis=1)):.2f})_(SoR:{yearly_sortino_ratio(df_UAM.drop(columns=['Dates']).sum(axis=1)):.2f})"
    plt.plot(
        df_UAM["Dates"],
        df_UAM.drop(columns=["Dates"]).sum(axis=1),
        label=label,
        linewidth=2,
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    plt.title(
        "PnL Curves of VAR(" + str(model_lag) + ") model",
        fontsize=16,
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("PnL", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_pnl_with_sharpe_per_cam(cam_label, cam_pnl_obj):
    """Plot PnL curves for a single CAM and its constituent assets.

    Creates a visualization showing both the aggregate CAM performance
    and individual asset performances within the cluster, with each
    curve annotated with its Sharpe ratio.

    Parameters
    ----------
    cam_label : int
        Identifier for the CAM being plotted
    cam_pnl_obj : dict
        Dictionary containing:
        - asset_names : list
            Names of assets in the cluster
        - pnl_across_assets : numpy.ndarray
            Aggregate PnL values across all assets
        - pnl_per_asset : numpy.ndarray
            Individual PnL values for each asset
        - dates : array-like
            Dates corresponding to the PnL values

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame containing the CAM's aggregate PnL values

    Notes
    -----
    - Creates separate curves for the CAM and each individual asset
    - Each curve is labeled with its annualized Sharpe ratio
    - Returns only the CAM's aggregate PnL for use in UAM calculations
    """
    pnl_values_across_assets = cam_pnl_obj["pnl_across_assets"]
    pnl_dates = cam_pnl_obj["dates"]
    sharpe_ratio_across_assets = yearly_sharpe_ratio(pnl_values_across_assets)
    sortino_ratio_across_assets = yearly_sortino_ratio(pnl_values_across_assets)

    df = pd.DataFrame(
        {"Date": pnl_dates, "PnL_CAM_" + str(cam_label): pnl_values_across_assets}
    )
    df["Date"] = pd.to_datetime(df["Date"])

    label = f"{df.columns[1]} (SR:{sharpe_ratio_across_assets:.2f}), (SoR:{sortino_ratio_across_assets:.2f})"
    plt.plot(df["Date"], df[df.columns[1]], label=label, linewidth=2)

    return df[["PnL_CAM_" + str(cam_label)]]


def yearly_sharpe_ratio(pnl_ts):
    """Calculate the annualized Sharpe ratio for a PnL time series.

    Computes the Sharpe ratio by taking the mean return divided by
    the standard deviation, then annualizing based on the number
    of observations.

    Parameters
    ----------
    pnl_ts : numpy.ndarray
        Array of PnL values

    Returns
    -------
    float
        Annualized Sharpe ratio

    Notes
    -----
    - Assumes returns are already excess returns (above risk-free rate)
    - Annualization is done by multiplying by sqrt(n) where n is the
      number of observations in the time series
    """
    return (np.mean(pnl_ts, axis=0) / np.std(pnl_ts, axis=0)) * np.sqrt(
        pnl_ts.shape[0]
    )  # assume for risk free rate is 0


def yearly_sortino_ratio(pnl_ts):
    """Calculate the annualized Sortino ratio for a PnL time series.

    Computes the Sortino ratio by taking the mean return divided by
    the standard deviation of negative returns, then annualizing based on the number"""
    pnl_ts_down = pnl_ts[pnl_ts < 0]
    return (np.mean(pnl_ts, axis=0) / np.std(pnl_ts_down, axis=0)) * np.sqrt(
        pnl_ts.shape[0]
    )  # assume for risk free rate is 0
