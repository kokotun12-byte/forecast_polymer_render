import io
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from app.model_loader import load_artifacts
from app.predictor import run_forecast

CURRENT_YEAR = datetime.now().year

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")


def create_forecast_plot(result_df: pd.DataFrame, model_artifacts: dict) -> str:
    history_y = model_artifacts["history_y"]
    historical_level = np.exp(history_y)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        historical_level.index,
        historical_level.values,
        label="Historical",
        linewidth=2
    )
    ax.plot(
        result_df["Date"],
        result_df["ARIMAX_Forecast"],
        "--",
        label="ARIMAX Forecast"
    )
    ax.plot(
        result_df["Date"],
        result_df["Hybrid_Forecast"],
        ":",
        label="Hybrid Forecast"
    )
    ax.plot(
        result_df["Date"],
        result_df["Weighted_Hybrid_Forecast"],
        linewidth=2,
        label="Weighted Hybrid Forecast"
    )

    ax.set_title("Historical Series and Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Polymer Import")
    ax.legend()
    ax.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def dataframe_to_excel_bytes(df: pd.DataFrame) -> io.BytesIO:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Forecast")
    output.seek(0)
    return output


@app.on_event("startup")
def startup_event():
    global model_artifacts
    model_artifacts = load_artifacts(lookback=12, n_features=3)
    print("Artifacts loaded successfully")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "current_year": CURRENT_YEAR,
            "result": None,
            "result_records": None,
            "result_json": None,
            "plot_url": None,
        }
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/download-template")
def download_template():
    template_df = pd.DataFrame({
        "year": [CURRENT_YEAR, CURRENT_YEAR, CURRENT_YEAR],
        "month": [1, 2, 3],
        "WTI": [65.00, 67.00, 68.50],
        "Exchange_Rate": [3589, 3616, 3650]
    })

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        template_df.to_excel(writer, index=False, sheet_name="Template")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=forecast_input_template.xlsx"}
    )


@app.post("/download-excel")
async def download_excel(request: Request):
    form = await request.form()
    data_json = form.get("result_json")

    if not data_json:
        return HTMLResponse("No forecast result available for download.", status_code=400)

    df = pd.read_json(io.StringIO(data_json))
    excel_file = dataframe_to_excel_bytes(df)

    return StreamingResponse(
        excel_file,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=forecast_results.xlsx"}
    )


@app.post("/predict/manual", response_class=HTMLResponse)
def predict_manual(
    request: Request,
    year1: str = Form(""),
    month1: str = Form(""),
    wti1: str = Form(""),
    exchange_rate1: str = Form(""),
    year2: str = Form(""),
    month2: str = Form(""),
    wti2: str = Form(""),
    exchange_rate2: str = Form(""),
    year3: str = Form(""),
    month3: str = Form(""),
    wti3: str = Form(""),
    exchange_rate3: str = Form("")
):
    rows = []

    manual_inputs = [
        (year1, month1, wti1, exchange_rate1),
        (year2, month2, wti2, exchange_rate2),
        (year3, month3, wti3, exchange_rate3),
    ]

    for idx, (year, month, wti, exchange_rate) in enumerate(manual_inputs, start=1):
        filled = any([year.strip(), month.strip(), wti.strip(), exchange_rate.strip()])

        if filled:
            if not all([year.strip(), month.strip(), wti.strip(), exchange_rate.strip()]):
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "result": f"Error: Row {idx} is incomplete. Please fill all four fields.",
                        "current_year": CURRENT_YEAR
                    }
                )

            try:
                row = {
                    "year": int(year),
                    "month": int(month),
                    "WTI": float(str(wti).replace(",", "").strip()),
                    "Exchange_Rate": float(str(exchange_rate).replace(",", "").strip())
                }
            except ValueError:
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "result": f"Error: Row {idx} has invalid numeric values.",
                        "current_year": CURRENT_YEAR
                    }
                )

            if row["month"] < 1 or row["month"] > 12:
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "result": f"Error: Row {idx} month must be between 1 and 12.",
                        "current_year": CURRENT_YEAR
                    }
                )

            if row["year"] != CURRENT_YEAR:
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "result": f"Error: Row {idx} year must be {CURRENT_YEAR}. Forecast is allowed only within the current year.",
                        "current_year": CURRENT_YEAR
                    }
                )

            if row["WTI"] <= 0 or row["Exchange_Rate"] <= 0:
                return templates.TemplateResponse(
                    request=request,
                    name="index.html",
                    context={
                        "result": f"Error: Row {idx} WTI and Exchange Rate must be positive values.",
                        "current_year": CURRENT_YEAR
                    }
                )

            rows.append(row)

    if len(rows) == 0:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": "Error: Please fill at least one month.",
                "current_year": CURRENT_YEAR
            }
        )

    df = pd.DataFrame(rows)

    if df.duplicated(subset=["year", "month"]).any():
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": "Error: Duplicate year-month rows are not allowed.",
                "current_year": CURRENT_YEAR
            }
        )

    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df = run_forecast(df, model_artifacts)
    plot_url = create_forecast_plot(df, model_artifacts)

    download_df = df.copy()
    download_df["Date"] = download_df["Date"].astype(str)
    result_records = df.fillna("").to_dict(orient="records")

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "result": df.to_html(index=False, float_format="%.2f", classes="result-table"),
            "result_records": result_records,
            "plot_url": plot_url,
            "result_json": download_df.to_json(orient="records"),
            "current_year": CURRENT_YEAR
        }
    )


@app.post("/predict/excel", response_class=HTMLResponse)
def predict_excel(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_excel(file.file)
        df.columns = df.columns.str.strip()

        required_cols = ["year", "month", "WTI", "Exchange_Rate"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": f"Error: Missing required columns: {missing_cols}",
                    "current_year": CURRENT_YEAR
                }
            )

        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
            "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }

        df["month"] = df["month"].astype(str).str.strip().replace(month_map)
        df["WTI"] = df["WTI"].astype(str).str.replace(",", "", regex=False).str.strip()
        df["Exchange_Rate"] = df["Exchange_Rate"].astype(str).str.replace(",", "", regex=False).str.strip()

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["month"] = pd.to_numeric(df["month"], errors="coerce")
        df["WTI"] = pd.to_numeric(df["WTI"], errors="coerce")
        df["Exchange_Rate"] = pd.to_numeric(df["Exchange_Rate"], errors="coerce")

        if df[["year", "month", "WTI", "Exchange_Rate"]].isnull().any().any():
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": "Error: Excel file contains invalid or missing values.",
                    "current_year": CURRENT_YEAR
                }
            )

        bad_months = df[(df["month"] < 1) | (df["month"] > 12)]
        if not bad_months.empty:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": "Error: Month must be between 1 and 12.",
                    "current_year": CURRENT_YEAR
                }
            )

        bad_years = df[df["year"] != CURRENT_YEAR]
        if not bad_years.empty:
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": f"Error: Forecast is allowed only for year {CURRENT_YEAR}.",
                    "current_year": CURRENT_YEAR
                }
            )

        if (df[["WTI", "Exchange_Rate"]] <= 0).any().any():
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": "Error: WTI and Exchange Rate must be positive values.",
                    "current_year": CURRENT_YEAR
                }
            )

        if df.duplicated(subset=["year", "month"]).any():
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": "Error: Duplicate year-month rows are not allowed.",
                    "current_year": CURRENT_YEAR
                }
            )

        df = df.sort_values(["year", "month"]).reset_index(drop=True)
        df = run_forecast(df, model_artifacts)
        plot_url = create_forecast_plot(df, model_artifacts)

        download_df = df.copy()
        download_df["Date"] = download_df["Date"].astype(str)
        result_records = df.fillna("").to_dict(orient="records")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": df.to_html(index=False, float_format="%.2f", classes="result-table"),
                "result_records": result_records,
                "plot_url": plot_url,
                "result_json": download_df.to_json(orient="records"),
                "current_year": CURRENT_YEAR
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "result": f"Error: Failed to read Excel file: {str(e)}",
                "current_year": CURRENT_YEAR
            }
        )