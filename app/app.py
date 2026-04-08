import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import joblib
from datetime import datetime, date
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="InvestIQ — Smart Asset Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data_backup")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_backup")


# ==========================================================
# STARTUP CHECK
# ==========================================================
def check_required_folders():
    missing = []

    if not os.path.exists(PROCESSED_DIR):
        missing.append(f"Processed data folder not found: {PROCESSED_DIR}")

    if not os.path.exists(MODEL_DIR):
        missing.append(f"Model folder not found: {MODEL_DIR}")

    return missing


# ==========================================================
# LOCATION CONFIG
# ==========================================================
LOCATION_CONFIG = {
    "🇮🇳 India": {
        "currency": "₹",
        "currency_code": "INR",
        "assets": {
            "Gold": {"file": "clean_gold_inr.csv", "unit": "INR/gram", "color": "#F4A32F"},
            "Stocks": {"file": "clean_nifty.csv", "unit": "Points", "color": "#3B82F6"},
            "Bitcoin": {"file": "clean_btc_inr.csv", "unit": "INR/BTC", "color": "#F97316"},
            "Ethereum": {"file": "clean_eth_inr.csv", "unit": "INR/ETH", "color": "#8B5CF6"},
        },
        "stock_label": "NIFTY 50",
    },
    "🇺🇸 USA": {
        "currency": "$",
        "currency_code": "USD",
        "assets": {
            "Gold": {"file": "clean_gold_usd.csv", "unit": "USD/troy oz", "color": "#F4A32F"},
            "Stocks": {"file": "clean_sp500.csv", "unit": "Points", "color": "#3B82F6"},
            "Bitcoin": {"file": "clean_btc_usd.csv", "unit": "USD/BTC", "color": "#F97316"},
            "Ethereum": {"file": "clean_eth_usd.csv", "unit": "USD/ETH", "color": "#8B5CF6"},
        },
        "stock_label": "S&P 500",
    },
    "🌍 Global": {
        "currency": "$",
        "currency_code": "USD",
        "assets": {
            "Gold": {"file": "clean_gold_usd.csv", "unit": "USD/troy oz", "color": "#F4A32F"},
            "NIFTY": {"file": "clean_nifty.csv", "unit": "Points", "color": "#3B82F6"},
            "S&P 500": {"file": "clean_sp500.csv", "unit": "Points", "color": "#10B981"},
            "Bitcoin": {"file": "clean_btc_usd.csv", "unit": "USD/BTC", "color": "#F97316"},
        },
        "stock_label": "Indices",
    },
}


# ==========================================================
# HELPERS
# ==========================================================
def get_model_key(asset_name: str, location: str) -> str:
    if asset_name == "Gold":
        return "gold"
    elif asset_name == "Stocks":
        return "nifty" if "India" in location else "sp500"
    elif asset_name == "NIFTY":
        return "nifty"
    elif asset_name == "S&P 500":
        return "sp500"
    elif asset_name == "Bitcoin":
        return "btc"
    elif asset_name == "Ethereum":
        return "eth"
    return asset_name.lower()


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ==========================================================
# DATA LOADING
# ==========================================================
@st.cache_data(ttl=3600)
def load_asset_data(filepath: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, filepath)

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Remove junk columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return pd.DataFrame()

    df = df[~df.index.isna()].copy()
    df.sort_index(inplace=True)

    # Convert all columns safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with Close if available
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    return df


@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def rebuild_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty or "Close" not in df.columns:
        return df

    # Lags
    for lag in [1, 7, 30]:
        df[f"lag_{lag}d"] = df["Close"].shift(lag)

    # Moving averages
    for w in [7, 30, 90, 200]:
        df[f"ma_{w}d"] = df["Close"].rolling(w).mean()

    # Std
    for w in [7, 30]:
        df[f"std_{w}d"] = df["Close"].rolling(w).std()

    # Percent vs MA
    if "ma_30d" in df.columns:
        df["pct_vs_ma30"] = (df["Close"] - df["ma_30d"]) / (df["ma_30d"] + 1e-9)

    # Bollinger position
    if "ma_30d" in df.columns and "std_30d" in df.columns:
        upper = df["ma_30d"] + 2 * df["std_30d"]
        lower = df["ma_30d"] - 2 * df["std_30d"]
        df["BB_pos"] = (df["Close"] - lower) / ((upper - lower) + 1e-9)

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Volatility
    returns = df["Close"].pct_change()
    df["volatility_30d"] = returns.rolling(30).std()

    # Time features
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)

    return df


# ==========================================================
# FORECASTING
# ==========================================================
def get_xgb_forecast(asset_key: str, df: pd.DataFrame, n_days: int = 180) -> pd.DataFrame:
    model_path = os.path.join(MODEL_DIR, "xgboost", f"{asset_key}_xgb_reg.pkl")
    bundle = load_model(model_path)

    if df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    if bundle is None:
        return pd.DataFrame()

    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        feat_cols = bundle.get("feature_cols", [])
    else:
        model = bundle
        feat_cols = [c for c in df.columns if c != "Close"]

    if model is None:
        return pd.DataFrame()

    df_copy = rebuild_features(df.copy())
    if df_copy.empty:
        return pd.DataFrame()

    last_date = df_copy.index.max()
    results = []

    for i in range(n_days):
        available_feat = [c for c in feat_cols if c in df_copy.columns]
        if not available_feat:
            break

        row = df_copy.iloc[-1:][available_feat].copy()
        row = row.ffill(axis=1).fillna(0)

        try:
            next_price = float(model.predict(row)[0])
        except Exception:
            break

        next_date = last_date + pd.Timedelta(days=i + 1)
        results.append({"Date": next_date, "forecast": next_price})

        new_row = df_copy.iloc[-1].copy()
        new_row["Close"] = next_price
        new_row.name = next_date

        df_copy = pd.concat([df_copy, new_row.to_frame().T])
        df_copy = rebuild_features(df_copy)

    if not results:
        return pd.DataFrame()

    forecast_df = pd.DataFrame(results).set_index("Date")

    recent_vol = df["Close"].pct_change().dropna().tail(60).std()
    if pd.notna(recent_vol):
        forecast_df["lower_ci"] = forecast_df["forecast"] * (1 - 1.96 * recent_vol)
        forecast_df["upper_ci"] = forecast_df["forecast"] * (1 + 1.96 * recent_vol)

    return forecast_df


def get_direction_prediction(asset_key: str, df: pd.DataFrame) -> dict:
    model_path = os.path.join(MODEL_DIR, "xgboost", f"{asset_key}_xgb_cls.pkl")
    bundle = load_model(model_path)

    if bundle is None or df.empty or "Close" not in df.columns:
        return {"direction": "Unknown", "confidence": 0}

    if isinstance(bundle, dict):
        model = bundle.get("model", None)
        feat_cols = bundle.get("feature_cols", [])
        label_map = bundle.get("label_map", {0: "Down", 1: "Stable", 2: "Up"})
    else:
        model = bundle
        feat_cols = [c for c in df.columns if c != "Close"]
        label_map = {0: "Down", 1: "Stable", 2: "Up"}

    if model is None:
        return {"direction": "Unknown", "confidence": 0}

    df_feat = rebuild_features(df.copy())
    available = [c for c in feat_cols if c in df_feat.columns]
    if not available:
        return {"direction": "Unknown", "confidence": 0}

    last_row = df_feat.iloc[-1:][available].copy()
    last_row = last_row.ffill(axis=1).fillna(0)

    try:
        pred = model.predict(last_row)[0]
        proba = model.predict_proba(last_row)[0]
        return {
            "direction": label_map.get(pred, "Unknown"),
            "confidence": float(np.max(proba)) * 100,
            "proba_down": float(proba[0]) * 100 if len(proba) > 0 else 0,
            "proba_stable": float(proba[1]) * 100 if len(proba) > 1 else 0,
            "proba_up": float(proba[2]) * 100 if len(proba) > 2 else 0,
        }
    except Exception:
        return {"direction": "Unknown", "confidence": 0}


# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================
def compute_recommendation(budget: float, target_date: date, assets_data: dict, currency: str) -> dict:
    today = pd.Timestamp.today().normalize()
    days_ahead = (pd.Timestamp(target_date) - today).days

    if days_ahead <= 0:
        return {"error": "Target date must be in the future."}

    scores = {}
    summaries = {}

    for asset_name, data in assets_data.items():
        df = data.get("df", pd.DataFrame())
        forecast = data.get("forecast_df", pd.DataFrame())
        direction_info = data.get("direction", {})

        if df.empty or forecast.empty or "Close" not in df.columns:
            continue

        current_price = float(df["Close"].dropna().iloc[-1])
        target_ts = pd.Timestamp(target_date)

        try:
            nearest_idx = forecast.index.get_indexer([target_ts], method="nearest")[0]
            future_price = float(forecast.iloc[nearest_idx]["forecast"])
        except Exception:
            future_price = current_price

        expected_return_pct = ((future_price - current_price) / current_price) * 100 if current_price > 0 else 0
        units_can_buy = budget / current_price if current_price > 0 else 0

        affordability_score = min(1.0, units_can_buy)

        vol = float(df["volatility_30d"].dropna().iloc[-1]) if "volatility_30d" in df.columns and not df["volatility_30d"].dropna().empty else 0.3
        vol_score = max(0, 1 - min(vol * 10, 1))

        rsi = float(df["RSI"].dropna().iloc[-1]) if "RSI" in df.columns and not df["RSI"].dropna().empty else 50
        if rsi < 30:
            trend_score = 0.8
        elif rsi > 70:
            trend_score = 0.6
        else:
            trend_score = 1.0

        direction = direction_info.get("direction", "Stable")
        confidence = direction_info.get("confidence", 50)

        if direction == "Up":
            cls_score = min(1.0, confidence / 100)
        elif direction == "Stable":
            cls_score = 0.6
        else:
            cls_score = 0.2

        return_score = max(0, min(1, (expected_return_pct + 20) / 40))

        composite = (
            0.35 * return_score +
            0.20 * affordability_score +
            0.15 * trend_score +
            0.15 * vol_score +
            0.15 * cls_score
        )

        years = max(days_ahead / 365, 0.01)
        cagr = ((future_price / current_price) ** (1 / years) - 1) * 100 if current_price > 0 else 0

        scores[asset_name] = composite
        summaries[asset_name] = {
            "current_price": current_price,
            "future_price": future_price,
            "expected_return_pct": expected_return_pct,
            "units_can_buy": units_can_buy,
            "value_at_target": units_can_buy * future_price,
            "profit": units_can_buy * future_price - budget,
            "rsi": rsi,
            "volatility": vol,
            "composite_score": composite,
            "cagr": cagr,
            "direction": direction,
            "confidence": confidence,
        }

    if not scores:
        return {"error": "No valid forecast data available."}

    best_asset = max(scores, key=scores.get)
    best = summaries[best_asset]

    total_score = sum(scores.values())
    allocation = {a: (s / total_score) * budget for a, s in scores.items()} if total_score > 0 else {}

    if best["expected_return_pct"] > 15:
        verdict = "🟢 Strong Buy"
        description = f"{best_asset} shows the strongest upside for your selected time horizon."
    elif best["expected_return_pct"] > 5:
        verdict = "🟡 Buy / Accumulate"
        description = f"{best_asset} has moderate upside. Consider phased investing."
    elif best["expected_return_pct"] > -5:
        verdict = "⚪ Hold / Wait"
        description = f"Markets appear range-bound. Waiting for a better entry may help."
    else:
        verdict = "🔴 Avoid / Caution"
        description = f"Most assets show weak outlook under current forecast assumptions."

    return {
        "best_asset": best_asset,
        "verdict": verdict,
        "description": description,
        "scores": scores,
        "summaries": summaries,
        "allocation": allocation,
        "days_ahead": days_ahead,
        "budget": budget,
        "currency": currency,
    }


# ==========================================================
# CHARTS
# ==========================================================
def plot_historical_with_forecast(df, forecast_df, asset_name, color, currency, unit):
    fig = go.Figure()

    if not df.empty and "Close" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name=f"{asset_name} Historical",
            line=dict(color=color, width=2),
            hovertemplate=f"%{{x|%d %b %Y}}<br>{currency}%{{y:,.2f}}<extra></extra>",
        ))

    if not forecast_df.empty:
        if "upper_ci" in forecast_df.columns and "lower_ci" in forecast_df.columns:
            rgb = _hex_to_rgb(color)
            fig.add_trace(go.Scatter(
                x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                y=list(forecast_df["upper_ci"]) + list(forecast_df["lower_ci"][::-1]),
                fill="toself",
                fillcolor=f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Confidence Band",
            ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df["forecast"],
            name=f"{asset_name} Forecast",
            line=dict(color=color, width=2, dash="dot"),
            hovertemplate=f"%{{x|%d %b %Y}}<br>Forecast: {currency}%{{y:,.2f}}<extra></extra>",
        ))

    for window, dash_style in [(30, "dash"), (200, "longdash")]:
        col = f"ma_{window}d"
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                name=f"MA {window}",
                line=dict(width=1, dash=dash_style, color="gray"),
                opacity=0.6,
            ))

    fig.update_layout(
        title=f"{asset_name} — Historical Price + Forecast ({unit})",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=1.02),
    )
    return fig


def plot_comparison_normalised(assets_dict, selected_period="3Y"):
    period_map = {"1Y": 365, "3Y": 1095, "5Y": 1825, "10Y": 3650, "All": 99999}
    days = period_map.get(selected_period, 1095)
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)

    colors = {
        "Gold": "#F4A32F",
        "Stocks": "#3B82F6",
        "Bitcoin": "#F97316",
        "Ethereum": "#8B5CF6",
        "NIFTY": "#3B82F6",
        "S&P 500": "#10B981",
    }

    fig = go.Figure()

    for asset_name, df in assets_dict.items():
        if df.empty or "Close" not in df.columns:
            continue
        sub = df[df.index >= cutoff]["Close"].dropna()
        if len(sub) < 2:
            continue
        norm = (sub / sub.iloc[0]) * 100
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm,
            name=asset_name,
            line=dict(width=2, color=colors.get(asset_name, "#888")),
        ))

    fig.add_hline(y=100, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Normalised Returns Comparison — {selected_period}",
        xaxis_title="Date",
        yaxis_title="Base = 100",
        hovermode="x unified",
        height=450,
    )
    return fig


def plot_rsi_macd(df, asset_name):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=[f"{asset_name} Close", "RSI (14)", "MACD"],
    )

    if not df.empty and "Close" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"), row=1, col=1)

    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    if "MACD" in df.columns and "MACD_signal" in df.columns and "MACD_hist" in df.columns:
        colors_hist = ["green" if v >= 0 else "red" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD Hist", marker_color=colors_hist), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"), row=3, col=1)

    fig.update_layout(height=650, showlegend=False)
    return fig


def plot_budget_bar(reco, currency):
    summaries = reco.get("summaries", {})
    assets = list(summaries.keys())
    values = [summaries[a]["value_at_target"] for a in assets]
    profits = [summaries[a]["profit"] for a in assets]
    colors = ["green" if p >= 0 else "red" for p in profits]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=assets,
        y=values,
        marker_color=colors,
        text=[f"{currency}{v:,.0f}" for v in values],
        textposition="outside",
    ))
    fig.add_hline(y=reco["budget"], line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Projected Portfolio Value by Asset",
        yaxis_title=f"Value ({currency})",
        height=400,
    )
    return fig


def plot_allocation_pie(reco):
    alloc = reco.get("allocation", {})
    if not alloc:
        return go.Figure()

    fig = px.pie(
        names=list(alloc.keys()),
        values=list(alloc.values()),
        title="Suggested Budget Allocation"
    )
    return fig


# ==========================================================
# SIDEBAR
# ==========================================================
def render_sidebar():
    with st.sidebar:
        st.title("📈 InvestIQ")
        st.caption("Smart Asset Forecasting")
        st.divider()

        location = st.selectbox("🌍 Location", list(LOCATION_CONFIG.keys()), index=0)

        cfg = LOCATION_CONFIG[location]
        currency = cfg["currency"]
        default_budget = 50000 if currency == "₹" else 1000

        budget = st.number_input(
            f"💰 Budget ({currency})",
            min_value=100,
            max_value=10_000_000,
            value=default_budget,
            step=1000,
        )

        target_date = st.date_input(
            "📅 Target Date",
            value=date(date.today().year + 1, 6, 1),
            min_value=date.today(),
        )

        preferred_assets = st.multiselect(
            "🏦 Assets to Compare",
            list(cfg["assets"].keys()),
            default=list(cfg["assets"].keys())[:3],
        )

        risk_profile = st.select_slider(
            "⚖️ Risk Profile",
            options=["Conservative", "Moderate", "Aggressive"],
            value="Moderate",
        )

        st.divider()
        st.caption(f"Data refreshed: {datetime.today().strftime('%d %b %Y')}")
        st.caption("Models: XGBoost Regressor + Classifier")

    return {
        "location": location,
        "budget": budget,
        "target_date": target_date,
        "preferred_assets": preferred_assets,
        "risk_profile": risk_profile,
        "currency": currency,
        "config": cfg,
    }


# ==========================================================
# MAIN APP
# ==========================================================
def main():
    # -------------------------------
    # Startup validation
    # -------------------------------
    issues = check_required_folders()
    if issues:
        st.error("⚠ Project folders are missing:")
        for issue in issues:
            st.write(f"- {issue}")
        st.stop()

    params = render_sidebar()
    cfg = params["config"]

    st.title("📈 InvestIQ — Smart Asset Forecasting")
    st.caption(
        f"Location: {params['location']} | "
        f"Budget: {params['currency']}{params['budget']:,} | "
        f"Target Date: {params['target_date']}"
    )

    # -------------------------------
    # Load Data
    # -------------------------------
    assets_data = {}
    missing_files = []

    for asset_name in params["preferred_assets"]:
        asset_cfg = cfg["assets"].get(asset_name, {})
        fname = asset_cfg.get("file", "")
        df = load_asset_data(fname)

        if df.empty:
            missing_files.append(fname)
            continue

        df = rebuild_features(df)
        assets_data[asset_name] = {
            "df": df,
            "unit": asset_cfg.get("unit", ""),
            "color": asset_cfg.get("color", "#888"),
        }

    if missing_files:
        with st.expander("⚠ Missing / Empty Data Files", expanded=False):
            for f in missing_files:
                st.write(f"- {f}")

    if not assets_data:
        st.error("⚠ No usable data found. Please ensure your processed CSV files are inside processed_data_backup/")
        st.stop()

    # -------------------------------
    # Model Status
    # -------------------------------
    with st.expander("🧠 Model Status", expanded=False):
        for asset_name in assets_data.keys():
            model_key = get_model_key(asset_name, params["location"])
            reg_path = os.path.join(MODEL_DIR, "xgboost", f"{model_key}_xgb_reg.pkl")
            cls_path = os.path.join(MODEL_DIR, "xgboost", f"{model_key}_xgb_cls.pkl")

            reg_ok = "✅" if os.path.exists(reg_path) else "❌"
            cls_ok = "✅" if os.path.exists(cls_path) else "❌"

            st.write(f"**{asset_name}** → Regressor: {reg_ok} | Classifier: {cls_ok}")

    # -------------------------------
    # Forecasting
    # -------------------------------
    days_ahead = max(1, (pd.Timestamp(params["target_date"]) - pd.Timestamp.today()).days)

    with st.spinner("Running forecasts..."):
        for asset_name, data in assets_data.items():
            model_key = get_model_key(asset_name, params["location"])
            forecast = get_xgb_forecast(model_key, data["df"], min(days_ahead, 365))

            if forecast.empty:
                last_price = data["df"]["Close"].dropna().iloc[-1]
                future_dates = pd.date_range(
                    start=pd.Timestamp.today() + pd.Timedelta(days=1),
                    periods=min(days_ahead, 365),
                    freq="D"
                )
                forecast = pd.DataFrame({"forecast": [last_price] * len(future_dates)}, index=future_dates)

            assets_data[asset_name]["forecast_df"] = forecast
            assets_data[asset_name]["direction"] = get_direction_prediction(model_key, data["df"])

    # -------------------------------
    # Tabs
    # -------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", "📉 Historical", "🔮 Forecast", "⚖️ Compare", "💡 Recommendation"
    ])

    # ======================================================
    # TAB 1 — OVERVIEW
    # ======================================================
    with tab1:
        st.subheader("Current Market Snapshot")

        cols = st.columns(len(assets_data))
        for i, (asset_name, data) in enumerate(assets_data.items()):
            df = data["df"]
            direction = data["direction"]

            current = float(df["Close"].dropna().iloc[-1])
            prev = float(df["Close"].dropna().iloc[-2]) if len(df) >= 2 else current
            change_pct = ((current - prev) / prev) * 100 if prev != 0 else 0
            last_dt = df.index.max().strftime("%d %b %Y")

            with cols[i]:
                st.metric(
                    label=f"{asset_name}",
                    value=f"{params['currency']}{current:,.2f}",
                    delta=f"{change_pct:+.2f}%"
                )
                dir_icon = {"Up": "🟢", "Down": "🔴", "Stable": "🟡"}.get(direction.get("direction", ""), "⚪")
                st.caption(f"{dir_icon} Next Day: {direction.get('direction','—')} ({direction.get('confidence',0):.0f}%)")
                st.caption(f"Last updated: {last_dt}")

        st.divider()
        st.subheader("Head-to-Head Performance")
        period = st.radio("Period", ["1Y", "3Y", "5Y", "10Y", "All"], horizontal=True, index=1)
        st.plotly_chart(
            plot_comparison_normalised({a: d["df"] for a, d in assets_data.items()}, period),
            use_container_width=True
        )

    # ======================================================
    # TAB 2 — HISTORICAL
    # ======================================================
    with tab2:
        st.subheader("Historical Price & Technical Indicators")
        selected_asset = st.selectbox("Select Asset", list(assets_data.keys()), key="hist_asset")

        data = assets_data[selected_asset]
        df = data["df"]

        col1, col2 = st.columns(2)
        with col1:
            hist_start = st.date_input("From", value=max(date(2018, 1, 1), df.index.min().date()), key="hist_from")
        with col2:
            hist_end = st.date_input("To", value=min(date.today(), df.index.max().date()), key="hist_to")

        df_filtered = df[(df.index >= pd.Timestamp(hist_start)) & (df.index <= pd.Timestamp(hist_end))]

        st.plotly_chart(plot_rsi_macd(df_filtered, selected_asset), use_container_width=True)

        close = df_filtered["Close"].dropna()
        if len(close) > 1:
            ann_return = ((close.iloc[-1] / close.iloc[0]) ** (252 / len(close)) - 1) * 100
        else:
            ann_return = 0

        stats = {
            "Min": f"{params['currency']}{close.min():,.2f}" if not close.empty else "—",
            "Max": f"{params['currency']}{close.max():,.2f}" if not close.empty else "—",
            "Mean": f"{params['currency']}{close.mean():,.2f}" if not close.empty else "—",
            "Current": f"{params['currency']}{close.iloc[-1]:,.2f}" if not close.empty else "—",
            "Annualised Return": f"{ann_return:.2f}%",
            "30d Volatility": f"{df_filtered['volatility_30d'].dropna().mean() * 100:.2f}%" if "volatility_30d" in df_filtered.columns and not df_filtered['volatility_30d'].dropna().empty else "—",
        }
        st.dataframe(pd.DataFrame([stats]), use_container_width=True, hide_index=True)

    # ======================================================
    # TAB 3 — FORECAST
    # ======================================================
    with tab3:
        st.subheader(f"Forecast Until {params['target_date']}")
        selected_asset = st.selectbox("Select Asset", list(assets_data.keys()), key="fc_asset")

        data = assets_data[selected_asset]
        df = data["df"]
        forecast_df = data["forecast_df"]

        hist_cutoff = pd.Timestamp.today() - pd.Timedelta(days=2 * 365)
        df_recent = df[df.index >= hist_cutoff]

        st.plotly_chart(
            plot_historical_with_forecast(
                df_recent, forecast_df, selected_asset, data["color"], params["currency"], data["unit"]
            ),
            use_container_width=True
        )

        if not forecast_df.empty:
            current = float(df["Close"].dropna().iloc[-1])
            milestones = {}
            for label, d in [("30 days", 30), ("90 days", 90), ("180 days", 180), ("1 year", 365)]:
                ts = pd.Timestamp.today() + pd.Timedelta(days=d)
                try:
                    nearest_idx = forecast_df.index.get_indexer([ts], method="nearest")[0]
                    p = float(forecast_df.iloc[nearest_idx]["forecast"])
                    milestones[label] = {
                        "Forecast Price": f"{params['currency']}{p:,.2f}",
                        "Expected Change": f"{(p - current) / current * 100:+.1f}%"
                    }
                except Exception:
                    pass

            if milestones:
                st.subheader("Forecast Milestones")
                st.dataframe(pd.DataFrame(milestones).T, use_container_width=True)

    # ======================================================
    # TAB 4 — COMPARE
    # ======================================================
    with tab4:
        st.subheader("Asset Comparison")

        compare_assets = st.multiselect(
            "Select assets",
            list(assets_data.keys()),
            default=list(assets_data.keys()),
            key="compare_multi",
        )

        if compare_assets:
            target_ts = pd.Timestamp(params["target_date"])
            rows = []

            for a in compare_assets:
                data = assets_data[a]
                df = data["df"]
                forecast_df = data["forecast_df"]
                current = float(df["Close"].dropna().iloc[-1])

                try:
                    nearest_idx = forecast_df.index.get_indexer([target_ts], method="nearest")[0]
                    future = float(forecast_df.iloc[nearest_idx]["forecast"])
                except Exception:
                    future = current

                ret_pct = ((future - current) / current) * 100 if current > 0 else 0
                units = params["budget"] / current if current > 0 else 0
                final_val = units * future
                years = max(days_ahead / 365, 0.01)
                cagr = ((future / current) ** (1 / years) - 1) * 100 if current > 0 else 0

                rows.append({
                    "Asset": a,
                    "Current Price": f"{params['currency']}{current:,.2f}",
                    "Forecast Price": f"{params['currency']}{future:,.2f}",
                    "Expected Return": f"{ret_pct:+.1f}%",
                    "CAGR": f"{cagr:.2f}%",
                    "Units for Budget": f"{units:.4f}",
                    "Projected Value": f"{params['currency']}{final_val:,.0f}",
                    "Signal": data["direction"].get("direction", "—"),
                })

            compare_df = pd.DataFrame(rows)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            ret_df = pd.DataFrame({
                "Asset": compare_df["Asset"],
                "Expected Return (%)": compare_df["Expected Return"].str.replace("%", "", regex=False).str.replace("+", "", regex=False).astype(float)
            })

            fig = px.bar(
                ret_df,
                x="Asset",
                y="Expected Return (%)",
                color="Expected Return (%)",
                color_continuous_scale="RdYlGn",
                title="Expected Return Comparison"
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # TAB 5 — RECOMMENDATION
    # ======================================================
    with tab5:
        st.subheader("💡 Personalised Investment Recommendation")

        reco = compute_recommendation(
            budget=float(params["budget"]),
            target_date=params["target_date"],
            assets_data=assets_data,
            currency=params["currency"],
        )

        if "error" in reco:
            st.error(reco["error"])
        else:
            st.markdown(f"## {reco['verdict']}")
            st.info(reco["description"])

            best = reco["summaries"][reco["best_asset"]]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Asset", reco["best_asset"])
            with col2:
                st.metric("Expected Return", f"{best['expected_return_pct']:+.1f}%")
            with col3:
                st.metric("Projected Value", f"{params['currency']}{best['value_at_target']:,.0f}")
            with col4:
                st.metric("CAGR", f"{best['cagr']:.2f}%")

            st.divider()

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_budget_bar(reco, params["currency"]), use_container_width=True)
            with col2:
                st.plotly_chart(plot_allocation_pie(reco), use_container_width=True)

            st.subheader("Detailed Breakdown")
            breakdown_rows = []
            for asset, s in reco["summaries"].items():
                breakdown_rows.append({
                    "Asset": asset,
                    "Current Price": f"{params['currency']}{s['current_price']:,.2f}",
                    "Forecast Price": f"{params['currency']}{s['future_price']:,.2f}",
                    "Expected Return": f"{s['expected_return_pct']:+.1f}%",
                    "CAGR": f"{s['cagr']:.2f}%",
                    "Units You Can Buy": f"{s['units_can_buy']:.4f}",
                    "Projected Value": f"{params['currency']}{s['value_at_target']:,.0f}",
                    "Profit/Loss": f"{params['currency']}{s['profit']:+,.0f}",
                    "RSI": f"{s['rsi']:.1f}",
                    "Volatility": f"{s['volatility'] * 100:.1f}%",
                    "Signal": f"{s['direction']} ({s['confidence']:.0f}%)",
                    "Score": f"{s['composite_score']:.3f}",
                })

            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True, hide_index=True)

            st.caption(
                "⚠ Disclaimer: This is a model-based forecast and not financial advice. "
                "Markets are uncertain. Please consult a registered financial advisor before investing."
            )


# ==========================================================
# RUN APP
# ==========================================================
if __name__ == "__main__":
    main()
