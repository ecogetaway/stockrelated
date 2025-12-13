import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ----------------------------
# Universe: from your midcap file (symbols without .NS)
# ----------------------------
MIDCAP50 = [
    "APLAPOLLO","AUBANK","ASHOKLEY","AUROPHARMA","BSE","BHARATFORG","BHEL","COFORGE",
    "COLPAL","CUMMINSIND","DABUR","DIXON","FEDERALBNK","FORTIS","GMRAIRPORT","GODREJPROP",
    "HDFCAMC","HEROMOTOCO","HINDPETRO","IDFCFIRSTB","IRCTC","INDUSTOWER","INDUSINDBK",
    "JUBLFOOD","LUPIN","MANKIND","MARICO","MFSL","MPHASIS","MUTHOOTFIN","NHPC","NMDC",
    "OBEROIRLTY","OIL","PAYTM","OFSS","POLICYBZR","PIIND","PAGEIND","PERSISTENT",
    "PHOENIXLTD","POLYCAB","PRESTIGE","SBICARD","SRF","SUPREMEIND","SUZLON","TIINDIA",
    "UPL","YESBANK"
]

# ----------------------------
# Indicators (adapted from your script)
# ----------------------------
def calc_rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return float(v) if pd.notna(v) else 50.0

def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    v = atr.iloc[-1]
    return float(v) if pd.notna(v) else 0.0

def calc_vol(close: pd.Series) -> float:
    ret = close.pct_change().dropna()
    return float(ret.std()) if len(ret) else 0.0

def build_alerts(row, vol_alert=0.035, vol_mult=2.0, rsi_ob=70, rsi_os=30) -> list[str]:
    alerts = []
    if row["daily_volatility"] >= vol_alert:
        alerts.append(f"HIGH_VOL({row['daily_volatility']:.1%})")
    if row["volume_ratio"] >= vol_mult:
        alerts.append(f"VOL_SPIKE({row['volume_ratio']:.1f}x)")
    if row["rsi"] >= rsi_ob:
        alerts.append(f"OVERBOUGHT({row['rsi']:.0f})")
    elif row["rsi"] <= rsi_os:
        alerts.append(f"OVERSOLD({row['rsi']:.0f})")
    if abs(row["price_change"]) >= 0.03:
        alerts.append(f"BIG_MOVE({row['price_change']:+.1%})")
    return alerts

# ----------------------------
# Cached data download (reduces Yahoo calls)
# ----------------------------
@st.cache_data(ttl=600)  # cache 10 minutes
def fetch_bulk(tickers: list[str], period: str, interval: str) -> pd.DataFrame:
    # yfinance accepts space-separated tickers or list
    df = yf.download(
        tickers=[t + ".NS" for t in tickers],
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    return df

def scan(period="1mo", interval="1d", rsi_period=14, atr_period=14,
         vol_high=0.030, vol_med=0.020) -> pd.DataFrame:
    raw = fetch_bulk(MIDCAP50, period=period, interval=interval)
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows = []
    for sym in MIDCAP50:
        key = sym + ".NS"
        if key not in raw.columns.get_level_values(0):
            continue

        d = raw[key].dropna()
        if len(d) < max(atr_period, rsi_period) + 2:
            continue

        close = d["Close"]
        vol = calc_vol(close)
        rsi = calc_rsi(close, period=rsi_period)
        atr = calc_atr(d, period=atr_period)

        cur_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        price_change = (cur_close - prev_close) / prev_close if prev_close else 0.0

        cur_vol = float(d["Volume"].iloc[-1])
        avg_vol = float(d["Volume"].mean())
        vol_ratio = cur_vol / avg_vol if avg_vol else 1.0

        if vol >= vol_high:
            bucket = "HIGH"
        elif vol >= vol_med:
            bucket = "MED"
        else:
            bucket = "LOW"

        rows.append({
            "symbol": sym,
            "bucket": bucket,
            "price": cur_close,
            "price_change": price_change,
            "daily_volatility": vol,
            "rsi": rsi,
            "atr": atr,
            "volume_ratio": vol_ratio,
            "data_points": int(len(d)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["alerts"] = out.apply(lambda r: ", ".join(build_alerts(r)), axis=1)
    out = out.sort_values(["bucket", "daily_volatility"], ascending=[True, False])
    return out

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Midcap 50 Volatility Scanner", layout="wide")
st.title("Midcap 50 Volatility Scanner (Yahoo/yfinance)")

with st.sidebar:
    st.header("Settings")
    interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
    period = st.selectbox("Period", ["1mo", "3mo", "6mo"], index=0)

    rsi_period = st.number_input("RSI period", 5, 50, 14)
    atr_period = st.number_input("ATR period", 5, 50, 14)

    vol_high = st.number_input("High volatility threshold (std of returns)", 0.0, 0.2, 0.03, step=0.005)
    vol_med = st.number_input("Medium volatility threshold", 0.0, 0.2, 0.02, step=0.005)

run = st.button("Run scan")

if run:
    t0 = time.time()
    df = scan(period=period, interval=interval, rsi_period=rsi_period, atr_period=atr_period,
              vol_high=vol_high, vol_med=vol_med)
    st.caption(f"Run time: {(time.time()-t0):.2f}s • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if df.empty:
        st.error("No data returned (Yahoo throttling or temporary issue). Try later or switch interval/period.")
    else:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"midcap50_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

st.info("Note: Yahoo/yfinance can rate-limit; caching is enabled to reduce repeated requests.")
