import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Market Dashboard", layout="wide")

st.title("📈 Market Dashboard")
st.caption("Daily data via Alpha Vantage. Live snapshot via Finnhub.")

# -----------------------------
# Secrets
# -----------------------------
AV_KEY = st.secrets.get("ALPHAVANTAGE_API_KEY")
FINNHUB_KEY = st.secrets.get("FINNHUB_API_KEY", None)

if not AV_KEY:
    st.error("Missing ALPHAVANTAGE_API_KEY. Add it to .streamlit/secrets.toml")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def safe_pct(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:,.2f}%"

def safe_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.{digits}f}"

# -----------------------------
# Data fetchers
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def av_time_series(symbol: str, mode: str) -> pd.DataFrame:
    """
    mode:
      - "daily_compact": TIME_SERIES_DAILY with compact (free-friendly)
      - "weekly": TIME_SERIES_WEEKLY (free-friendly, multi-year)
    """
    url = "https://www.alphavantage.co/query"

    if mode == "daily_compact":
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact",
            "apikey": AV_KEY,
        }
        series_key = "Time Series (Daily)"
        colmap = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }

    elif mode == "weekly":
        params = {
            "function": "TIME_SERIES_WEEKLY",
            "symbol": symbol,
            "apikey": AV_KEY,
        }
        series_key = "Weekly Time Series"
        colmap = {
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        }

    else:
        raise ValueError("Invalid mode")

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    if "Information" in j or "Note" in j:
        raise ValueError("ALPHAVANTAGE_LIMIT")
    if "Error Message" in j:
        raise ValueError("ALPHAVANTAGE_ERROR")
    if "Error Message" in j:
        raise ValueError(j["Error Message"])

    if series_key not in j:
        raise ValueError(f"Unexpected response keys: {list(j.keys())}")

    ts = j[series_key]
    df = pd.DataFrame.from_dict(ts, orient="index").rename(columns=colmap)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=10 * 60, show_spinner=False)  # 10 min cache
def finnhub_quote(symbol: str) -> dict:
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": FINNHUB_KEY}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def compute_metrics(close: pd.Series) -> dict:
    rets = close.pct_change().dropna()
    if len(rets) < 30:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}

    ann = 252.0
    years = len(close) / ann
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = rets.std() * np.sqrt(ann)
    sharpe = (rets.mean() * ann) / (rets.std() * np.sqrt(ann)) if rets.std() != 0 else np.nan

    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    maxdd = dd.min()

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd}

def drawdown_series(close: pd.Series) -> pd.Series:
    rets = close.pct_change().dropna()
    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    return dd

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    symbols_text = st.text_area(
        "Symbols (one per line)",
        "AAPL\nMSFT\nSPY\nQQQ\nGLD",
        help="Alpha Vantage free tier is rate-limited. Start with 3–5 symbols max.",
    )
    symbols = [s.strip().upper() for s in symbols_text.splitlines() if s.strip()]

    lookback = st.selectbox("Lookback", ["1M", "3M", "6M", "1Y", "3Y"], index=3)
    primary = st.selectbox("Primary symbol", symbols if symbols else ["AAPL"])

    st.caption("Tip: Avoid repeated refreshes; Alpha Vantage free tier is strict.")

    if st.button("Hard refresh (clear cache)"):
        st.cache_data.clear()
        st.success("Cache cleared.")

days_map = {"1M": 31, "3M": 93, "6M": 186, "1Y": 365, "3Y": 365 * 3}
days = days_map[lookback]

is_3y = (lookback == "3Y")
mode = "weekly" if is_3y else "daily_compact"
periods_per_year = 52.0 if is_3y else 252.0

# For <=6M, compact is enough; for 1Y/3Y use full
outputsize = "compact" if lookback in ["1M", "3M", "6M"] else "full"

# -----------------------------
# Load data
# -----------------------------
errors = {}
data = {}

with st.spinner("Fetching daily price data..."):
    for i, sym in enumerate(symbols):
        try:
            df = av_time_series(sym, mode=mode)
            df = df.tail(days)
            if not df.empty:
                data[sym] = df
        except Exception as e:
            err = str(e)
            if "ALPHAVANTAGE_LIMIT" in err:
                errors[sym] = "Rate limit hit. Wait and try again (or reduce symbols)."
                break
            elif "ALPHAVANTAGE_ERROR" in err:
                errors[sym] = "Provider error (symbol / endpoint)."
            else:
                errors[sym] = "Provider error."

        # Free-tier friendly pacing for multi-symbol loads (only matters on cold cache)
        if i < len(symbols) - 1:
            time.sleep(12)

if errors:
    st.warning("Some symbols failed to load (provider limits / access).")
    for sym in errors.keys():
        st.write(f"- **{sym}**: Provider returned an error. Try fewer symbols or wait and refresh.")

if not data or primary not in data:
    st.error("No usable data returned. Try fewer symbols (e.g., AAPL/MSFT/SPY) and avoid rapid refreshes.")
    st.stop()

# Align closes across symbols
close_df = pd.DataFrame({sym: df["close"] for sym, df in data.items()}).dropna()

# Returns + rolling vol
rets = close_df.pct_change().dropna()
vol20 = rets.rolling(20).std() * np.sqrt(252)
dd = drawdown_series(close_df[primary])

# -----------------------------
# Snapshot row (optional Finnhub)
# -----------------------------
st.subheader("Snapshot")
snap_left, snap_right = st.columns([3, 2])

with snap_left:
    if FINNHUB_KEY:
        cols = st.columns(min(6, len(symbols)))
        for i, sym in enumerate(symbols[:6]):
            with cols[i]:
                try:
                    q = finnhub_quote(sym)
                    st.metric(
                        sym,
                        safe_num(q.get("c", np.nan), 2),
                        f"{safe_num(q.get('d', np.nan), 2)} ({safe_num(q.get('dp', np.nan), 2)}%)",
                    )
                except Exception:
                    st.metric(sym, "—", "—")
    else:
        st.info("Add FINNHUB_API_KEY to show live quote snapshots (optional).")

with snap_right:
    st.metric("Data last updated (UTC)", utc_now_str())
    st.caption(f"Alpha Vantage outputsize: **{outputsize}** | Lookback: **{lookback}**")

st.divider()

# -----------------------------
# Primary charts
# -----------------------------
st.subheader(f"{primary} overview")

c1, c2 = st.columns([2, 1])

with c1:
    price_df = close_df[primary].reset_index()
    price_df.columns = ["date", "close"]
    fig_price = px.line(price_df, x="date", y="close", title="Close Price")
    fig_price.update_layout(xaxis_title="Date", yaxis_title="Close")
    st.plotly_chart(fig_price, use_container_width=True)

with c2:
    m = compute_metrics(close_df[primary])
    st.markdown("### Risk metrics")
    st.metric("CAGR", safe_pct(m["CAGR"]))
    st.metric("Annualized Vol", safe_pct(m["Vol"]))
    st.metric("Sharpe (rf=0)", safe_num(m["Sharpe"], 2))
    st.metric("Max Drawdown", safe_pct(m["MaxDD"]))

st.divider()

# -----------------------------
# Returns / Vol / Drawdown
# -----------------------------
r1, r2, r3 = st.columns(3)

with r1:
    rr = rets[primary].reset_index()
    rr.columns = ["date", "ret"]
    fig_r = px.line(rr, x="date", y="ret", title="Daily Returns")
    fig_r.update_layout(xaxis_title="Date", yaxis_title="Return")
    st.plotly_chart(fig_r, use_container_width=True)

with r2:
    vv = vol20[primary].reset_index()
    vv.columns = ["date", "vol"]
    fig_v = px.line(vv, x="date", y="vol", title="Rolling Volatility (20D, ann.)")
    fig_v.update_layout(xaxis_title="Date", yaxis_title="Vol (annualized)")
    st.plotly_chart(fig_v, use_container_width=True)

with r3:
    dd_df = dd.reset_index()
    dd_df.columns = ["date", "dd"]
    fig_dd = px.line(dd_df, x="date", y="dd", title="Drawdown")
    fig_dd.update_layout(xaxis_title="Date", yaxis_title="Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# -----------------------------
# Correlation + data export
# -----------------------------
st.subheader("Cross-asset analysis")

a1, a2 = st.columns([2, 1])

with a1:
    corr = rets.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation (Daily Returns)")
    st.plotly_chart(fig_corr, use_container_width=True)

with a2:
    st.markdown("### Data export")
    st.download_button(
        "Download aligned closes (CSV)",
        close_df.to_csv().encode("utf-8"),
        file_name="close_prices.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download returns (CSV)",
        rets.to_csv().encode("utf-8"),
        file_name="returns.csv",
        mime="text/csv",
    )
    st.caption("Aligned data drops dates where any selected symbol is missing.")