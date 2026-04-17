import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def run_forecast(df: pd.DataFrame, model_artifacts: dict) -> pd.DataFrame:
    scaler = model_artifacts["scaler"]
    artifacts = model_artifacts["hybrid_artifacts"]
    history_y = model_artifacts["history_y"]
    history_X = model_artifacts["history_X"]
    feature_history = model_artifacts["feature_history"]
    lstm_model = model_artifacts["lstm_model"]

    best_order = tuple(artifacts["best_order"])
    lookback = int(artifacts["lookback"])
    best_weight = float(artifacts["best_weight"])

    result_df = df.copy()

    # Build Date column from year and month
    result_df["Date"] = pd.to_datetime(
        result_df["year"].astype(int).astype(str) + "-" +
        result_df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    )
    result_df = result_df.sort_values("Date").set_index("Date")

    # Convert future exogenous inputs to log scale
    future_log = np.log(result_df[["WTI", "Exchange_Rate"]]).copy()
    future_log.columns = ["log_WTI_Price", "log_Exchange_Rate"]

    # Local rolling copies
    history_y_local = history_y.copy()
    history_X_local = history_X.copy()
    feature_history_local = feature_history.copy()

    arimax_log_list = []
    hybrid_log_list = []
    weighted_log_list = []
    resid_scaled_list = []
    resid_list = []
    dates = []

    for i in range(len(future_log)):
        x_next = future_log.iloc[[i]]
        date = future_log.index[i]

        model = SARIMAX(
            history_y_local,
            exog=history_X_local,
            order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fit = model.fit(disp=False)

        arimax_pred = float(
            fit.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0]
        )

        last_window = feature_history_local.iloc[-lookback:].copy()
        scaled_window = scaler.transform(last_window)
        X_input = scaled_window.reshape(1, lookback, scaled_window.shape[1])

        pred_resid_scaled = float(lstm_model.predict(X_input, verbose=0)[0, 0])

        # Inverse transform only the residual column
        dummy = np.zeros((1, feature_history_local.shape[1]))
        dummy[0, 0] = pred_resid_scaled
        resid = float(scaler.inverse_transform(dummy)[0, 0])

        hybrid_pred = float(arimax_pred + resid)
        weighted_pred = float(best_weight * hybrid_pred + (1 - best_weight) * arimax_pred)

        arimax_log_list.append(arimax_pred)
        hybrid_log_list.append(hybrid_pred)
        weighted_log_list.append(weighted_pred)
        resid_scaled_list.append(pred_resid_scaled)
        resid_list.append(resid)
        dates.append(date)

        # Rolling update for next step
        history_y_local = pd.concat([
            history_y_local,
            pd.Series([weighted_pred], index=[date], name=history_y_local.name)
        ])
        history_X_local = pd.concat([history_X_local, x_next])

        new_row = pd.DataFrame({
            "residual": [resid],
            "log_WTI_Price": [x_next.iloc[0, 0]],
            "log_Exchange_Rate": [x_next.iloc[0, 1]]
        }, index=[date])

        feature_history_local = pd.concat([feature_history_local, new_row])

    output = result_df.copy()
    output["ARIMAX_Log_Forecast"] = arimax_log_list
    output["LSTM_Residual_Scaled"] = resid_scaled_list
    output["LSTM_Residual_Forecast"] = resid_list
    output["Hybrid_Log_Forecast"] = hybrid_log_list
    output["Weighted_Log_Forecast"] = weighted_log_list

    # Convert back to level scale
    output["ARIMAX_Forecast"] = np.exp(output["ARIMAX_Log_Forecast"])
    output["Hybrid_Forecast"] = np.exp(output["Hybrid_Log_Forecast"])
    output["Weighted_Hybrid_Forecast"] = np.exp(output["Weighted_Log_Forecast"])

    output = output.reset_index()
    output["year"] = output["Date"].dt.year
    output["month"] = output["Date"].dt.month

    return output[[
        "Date",
        "year",
        "month",
        "WTI",
        "Exchange_Rate",
        "ARIMAX_Forecast",
        "Hybrid_Forecast",
        "Weighted_Hybrid_Forecast"
    ]]