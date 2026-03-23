import json
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import requests
from math import sqrt
import time
import random
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

from langchain_community.tools import DuckDuckGoSearchRun

ACCOUNT_FILE = Path(__file__).resolve().parent / "paper_account.json"
TREASURY_FILE = Path(__file__).resolve().parent / "treasury_holdings.json"
TREASURY_UNIVERSE_FILE = Path(__file__).resolve().parent / "treasury_universe.json"
BACKTEST_PROTOCOL_FILE = Path(__file__).resolve().parent / "backtest_protocol_history.json"

STRATEGY_CONFIG = {
    "watchlist": ["MSTR", "MARA", "XXI", "3350.T", "CEPO", "RIOT", "COIN", "HUT", "CLSK", "TSLA", "GLXY.TO", "GDC", "DJT", "BMNR", "DFDV", "SBET", "BTBT", "CIFR", "CORZ"],
    "nav_threshold_pct": 0.05,
    "deviation_threshold": 0.10,
    "relaxed_deviation_threshold": 0.10,
    "pricing_lag_threshold_pct": 0.02,
    "options_iv_threshold": 1.0,
    "options_iv_over_hv_gap": 0.20,
    "crypto_rally_daily_gain": 0.03,
    "beta_min": 1.2,
    "portfolio_sector_max": 0.60,
    "single_name_max": 0.20,
    "risk_per_trade_var95": 0.01,
    "max_new_trades_per_run": 4,
    "adaptive_relaxation_timer_minutes": 30,
    "enable_auto_relax": os.getenv("ENABLE_AUTO_RELAX", "true").strip().lower() in {"1", "true", "yes", "on"},
    "estimated_units_mode": os.getenv("ESTIMATED_UNITS_MODE", "false").strip().lower() in {"1", "true", "yes", "on"},
    "missing_holdings_abort_threshold_pct": 0.30,
    "missing_holdings_action": os.getenv("MISSING_HOLDINGS_ACTION", "alert").strip().lower() or "alert",
    "holdings_stale_after_days": 42,
    "live_watchlist_min_holdings_usd": 100_000_000,
    "live_watchlist_min_avg_dollar_volume": 5_000_000,
    "live_watchlist_limit": 12,
    "auto_refresh_holdings_on_startup": True,
    "enable_llm_company_site_mnav_fallback": os.getenv("ENABLE_LLM_COMPANY_SITE_MNAV_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"},
    "high_beta_mode": False,
}

HIGH_BETA_PRESET = {
    "portfolio_sector_max": 0.75,
    "single_name_max": 0.25,
    "risk_per_trade_var95": 0.04,
    "max_new_trades_per_run": 6,
}

BASE_PRESET = {
    "portfolio_sector_max": 0.60,
    "single_name_max": 0.20,
    "risk_per_trade_var95": 0.025,
    "max_new_trades_per_run": 4,
}

UNDERLYING_MAP = {
    "SBET": "ETH-USD",
    "DFDV": "BTC-USD",
    "BMNR": "BTC-USD",
    "BTBT": "BTC-USD",
    "MARA": "BTC-USD",
    "RIOT": "BTC-USD",
    "HUT": "BTC-USD",
    "CIFR": "BTC-USD",
    "CLSK": "BTC-USD",
    "CORZ": "BTC-USD",
}

HEDGE_ETF_MAP = {
    "BTC-USD": "BITI",   # inverse BTC strategy ETF proxy
    "ETH-USD": "ETHE",   # ETH trust proxy
    "SOL-USD": "SOLQ",   # may not be available on all brokers
}

DEFAULT_TREASURY_UNIVERSE = [
    {"symbol": "MSTR", "asset": "BTC", "units": 720000, "market_cap_hint_usd": 95_000_000_000},
    {"symbol": "MARA", "asset": "BTC", "units": 53000, "market_cap_hint_usd": 6_000_000_000},
    {"symbol": "XXI", "asset": "BTC", "units": 43000, "market_cap_hint_usd": None},
    {"symbol": "3350.T", "asset": "BTC", "units": 35000, "market_cap_hint_usd": None},
    {"symbol": "CEPO", "asset": "BTC", "units": 30000, "market_cap_hint_usd": None},
    {"symbol": "RIOT", "asset": "BTC", "units": 18000, "market_cap_hint_usd": 5_000_000_000},
    {"symbol": "COIN", "asset": "BTC", "units": 15000, "market_cap_hint_usd": 78_000_000_000},
    {"symbol": "HUT", "asset": "BTC", "units": 14000, "market_cap_hint_usd": 3_000_000_000},
    {"symbol": "CLSK", "asset": "BTC", "units": 13000, "market_cap_hint_usd": 3_000_000_000},
    {"symbol": "TSLA", "asset": "BTC", "units": 12000, "market_cap_hint_usd": 1_000_000_000_000},
    {"symbol": "XYZ", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "GLXY.TO", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "GDC", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "DJT", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "BMNR", "asset": "ETH", "units": None, "market_cap_hint_usd": None},
    {"symbol": "DFDV", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "SBET", "asset": "ETH", "units": None, "market_cap_hint_usd": None},
    {"symbol": "BTBT", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "CIFR", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
    {"symbol": "CORZ", "asset": "BTC", "units": None, "market_cap_hint_usd": None},
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_account() -> dict:
    return {
        "cash": 100000.0,
        "positions": {},
        "trades": [],
        "last_updated": _now_iso(),
    }


def _load_account() -> dict:
    if not ACCOUNT_FILE.exists():
        account = _default_account()
        _save_account(account)
        return account
    try:
        return json.loads(ACCOUNT_FILE.read_text(encoding="utf-8"))
    except Exception:
        account = _default_account()
        _save_account(account)
        return account


def _save_account(account: dict) -> None:
    account["last_updated"] = _now_iso()
    ACCOUNT_FILE.write_text(json.dumps(account, indent=2), encoding="utf-8")


def _get_market_deps():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        import yfinance as yf
        import yfinance.cache as yf_cache
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install with `pip install numpy yfinance pandas scikit-learn`."
        ) from exc
    cache_dir = Path(__file__).resolve().parent / ".yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf_cache.set_cache_location(str(cache_dir))
    except Exception:
        pass
    return np, pd, yf, LinearRegression


def _load_treasury_holdings() -> dict:
    if TREASURY_FILE.exists():
        try:
            data = json.loads(TREASURY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                universe_map = {str(row.get("symbol", "")).upper(): row for row in _load_treasury_universe()}
                return {
                    str(symbol).upper(): _normalize_holding_record(str(symbol).upper(), row, universe_map)
                    for symbol, row in data.items()
                    if isinstance(row, dict)
                }
        except Exception:
            pass
    return {}


def _save_treasury_holdings(payload: dict) -> None:
    TREASURY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_treasury_universe() -> list[dict]:
    if TREASURY_UNIVERSE_FILE.exists():
        try:
            data = json.loads(TREASURY_UNIVERSE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return list(DEFAULT_TREASURY_UNIVERSE)


def _parse_iso_date(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _holding_staleness_days(as_of_date: str | None) -> int | None:
    dt = _parse_iso_date(as_of_date)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - dt).days))


def _normalize_holding_record(symbol: str, row: dict | None, universe_map: dict | None = None) -> dict:
    row = row or {}
    universe_map = universe_map or {}
    urow = universe_map.get(symbol, {})
    asset = row.get("asset") or urow.get("asset") or "BTC"
    normalized = {
        "units": None if row.get("units") is None else float(row.get("units")),
        "shares_outstanding": None if row.get("shares_outstanding") is None else float(row.get("shares_outstanding")),
        "asset": str(asset).upper().strip(),
        "updated_at_utc": row.get("updated_at_utc") or _now_iso(),
        "source": row.get("source") or ("manual" if row.get("units") is not None or row.get("shares_outstanding") is not None else "unknown"),
        "as_of_date": row.get("as_of_date"),
        "confidence": float(row.get("confidence")) if row.get("confidence") is not None else (0.9 if row.get("units") is not None else 0.0),
    }
    staleness_days = _holding_staleness_days(normalized.get("as_of_date"))
    normalized["staleness_days"] = staleness_days
    return normalized


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _treasury_reference_cache_path() -> Path:
    return Path(__file__).resolve().parent / "treasury_reference_cache.json"


def _load_reference_cache() -> dict:
    path = _treasury_reference_cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_reference_cache(payload: dict) -> None:
    _treasury_reference_cache_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_treasury_universe(rows: list[dict]) -> None:
    TREASURY_UNIVERSE_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _macd_signal(prices, np_module):
    arr = prices.to_numpy(dtype=float)
    if len(arr) < 40:
        return False
    alpha12 = 2.0 / 13.0
    alpha26 = 2.0 / 27.0
    alpha9 = 2.0 / 10.0
    ema12 = arr[0]
    ema26 = arr[0]
    macd_series = []
    for value in arr:
        ema12 = alpha12 * value + (1 - alpha12) * ema12
        ema26 = alpha26 * value + (1 - alpha26) * ema26
        macd_series.append(ema12 - ema26)
    signal = macd_series[0]
    signal_series = []
    for value in macd_series:
        signal = alpha9 * value + (1 - alpha9) * signal
        signal_series.append(signal)
    if len(macd_series) < 2:
        return False
    return macd_series[-1] > signal_series[-1] and macd_series[-2] <= signal_series[-2]


def _ema_series(values, period: int):
    alpha = 2.0 / (period + 1.0)
    out = []
    prev = None
    for v in values:
        fv = float(v)
        prev = fv if prev is None else (alpha * fv + (1.0 - alpha) * prev)
        out.append(prev)
    return out


def _bx_trender_metrics(hist_df, np_module) -> dict:
    close = hist_df["Close"].dropna()
    if len(close) < 30:
        return {"error": "not enough bars for BX Trender"}

    sma9 = float(close.tail(9).mean())
    ema11 = float(_ema_series(close.to_numpy(), 11)[-1])
    ema11_prev = float(_ema_series(close.to_numpy(), 11)[-2])
    bb_mid = float(close.tail(20).mean())
    bb_std = float(close.tail(20).std(ddof=1))
    bb_up = bb_mid + 2.0 * bb_std
    bb_lo = bb_mid - 2.0 * bb_std
    last_close = float(close.iloc[-1])

    # Weighted indicator score: trend + momentum + Bollinger location
    score = 0.0
    score += 0.40 if last_close > sma9 else -0.40
    score += 0.35 if last_close > ema11 else -0.35
    score += 0.15 if ema11 > ema11_prev else -0.15
    if bb_up > bb_lo:
        bb_pos = (last_close - bb_lo) / (bb_up - bb_lo)
        score += 0.10 if bb_pos > 0.55 else -0.10 if bb_pos < 0.45 else 0.0
    else:
        bb_pos = 0.5

    return {
        "sma9": round(sma9, 4),
        "ema11": round(ema11, 4),
        "ema11_prev": round(ema11_prev, 4),
        "bb20_mid": round(bb_mid, 4),
        "bb20_upper": round(bb_up, 4),
        "bb20_lower": round(bb_lo, 4),
        "close": round(last_close, 4),
        "bb_position_0_1": round(float(bb_pos), 4),
        "bx_score": round(float(score), 4),
    }


def _weekly_ema11_red(hist_df) -> bool:
    close = hist_df["Close"].dropna()
    if len(close) < 15:
        return False
    ema = _ema_series(close.to_numpy(), 11)
    return float(ema[-1]) < float(ema[-2])


def _estimate_var95(returns, np_module, n_sims: int = 2000):
    if len(returns) < 30:
        return 0.03
    mu = float(np_module.mean(returns))
    sigma = float(np_module.std(returns, ddof=1))
    sims = np_module.random.normal(mu, sigma, size=n_sims)
    return float(max(0.005, -np_module.quantile(sims, 0.05)))


def _realized_vol_30d(returns, np_module):
    if len(returns) < 10:
        return 0.0
    return float(np_module.std(returns.tail(30), ddof=1) * sqrt(252))


def _atm_implied_vol(ticker):
    return None


def _get_alpaca_credentials() -> tuple[str, str, str]:
    key = os.getenv("ALPACA_API_KEY", "").strip()
    secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env."
        )
    return key, secret, base_url.rstrip("/")


def _send_discord_message(content: str) -> dict:
    webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook:
        return {"sent": False, "reason": "DISCORD_WEBHOOK_URL not set"}

    mention_user_id = os.getenv("DISCORD_USER_ID", "").strip()
    mention_prefix = f"<@{mention_user_id}> " if mention_user_id else ""
    body = {"content": f"{mention_prefix}{content}"[:1900]}
    try:
        resp = requests.post(webhook, json=body, timeout=10)
        if resp.status_code >= 400:
            return {
                "sent": False,
                "reason": f"Discord webhook error {resp.status_code}: {resp.text[:200]}",
            }
        return {"sent": True}
    except Exception as exc:
        return {"sent": False, "reason": str(exc)}


def _alpaca_request(method: str, path: str, payload: dict | None = None) -> dict:
    key, secret, base_url = _get_alpaca_credentials()
    url = f"{base_url}{path}"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }

    response = requests.request(
        method=method.upper(),
        url=url,
        headers=headers,
        json=payload,
        timeout=20,
    )
    if response.status_code >= 400:
        try:
            details = response.json()
        except Exception:
            details = {"message": response.text}
        raise RuntimeError(f"Alpaca API error {response.status_code}: {details}")

    if not response.text:
        return {"status": "ok"}
    return response.json()


def _alpaca_data_request(path: str, params: dict | None = None) -> dict:
    key, secret, _ = _get_alpaca_credentials()
    url = f"https://data.alpaca.markets{path}"
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }
    response = requests.get(url, headers=headers, params=params or {}, timeout=20)
    if response.status_code >= 400:
        try:
            details = response.json()
        except Exception:
            details = {"message": response.text}
        raise RuntimeError(f"Alpaca Market Data error {response.status_code}: {details}")
    return response.json() if response.text else {}


def _finnhub_request(path: str, params: dict | None = None) -> dict:
    api_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing Finnhub credentials. Set FINNHUB_API_KEY in .env.")
    query = dict(params or {})
    query["token"] = api_key
    url = f"https://finnhub.io/api{path}"
    response = requests.get(url, params=query, timeout=20)
    if response.status_code >= 400:
        try:
            details = response.json()
        except Exception:
            details = {"message": response.text}
        raise RuntimeError(f"Finnhub API error {response.status_code}: {details}")
    return response.json() if response.text else {}


def _finnhub_company_profile2(symbol: str) -> dict:
    return _finnhub_request("/v1/stock/profile2", {"symbol": symbol.upper().strip()})


_LLM_FALLBACK = None


def _get_llm_fallback():
    global _LLM_FALLBACK
    if _LLM_FALLBACK is not None:
        return _LLM_FALLBACK
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    _LLM_FALLBACK = ChatOpenAI(model="gpt-5-mini-2025-08-07", api_key=api_key, temperature=0)
    return _LLM_FALLBACK


def _fetch_company_website_context(base_url: str) -> list[dict]:
    if not base_url:
        return []
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    targets = [base_url]
    pages = []
    seen = set()
    domain = urlparse(base_url).netloc.lower()

    for url in list(targets):
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = session.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
        except Exception:
            continue
        html = resp.text
        text = html[:8000]
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(" ", strip=True)[:12000]
            if len(targets) < 3:
                for a in soup.find_all("a", href=True):
                    href = a.get("href", "")
                    label = a.get_text(" ", strip=True).lower()
                    full_url = urljoin(url, href)
                    parsed = urlparse(full_url)
                    if parsed.netloc.lower() != domain:
                        continue
                    joined = f"{label} {href}".lower()
                    if any(key in joined for key in ["investor", "press", "news", "treasury", "bitcoin", "ethereum", "digital asset"]):
                        if full_url not in targets:
                            targets.append(full_url)
                    if len(targets) >= 3:
                        break
        except Exception:
            pass
        pages.append({"url": url, "text": text})
        if len(pages) >= 3:
            break
    return pages


def _llm_extract_treasury_inputs_from_company_website(
    symbol: str,
    asset: str,
    finnhub_profile: dict | None = None,
) -> dict:
    if not bool(STRATEGY_CONFIG.get("enable_llm_company_site_mnav_fallback", True)):
        return {}
    llm = _get_llm_fallback()
    if llm is None:
        return {}
    finnhub_profile = finnhub_profile or {}
    website = str(finnhub_profile.get("weburl") or "").strip()
    if not website:
        return {}

    pages = _fetch_company_website_context(website)
    if not pages:
        return {}

    page_blob = "\n\n".join(
        f"URL: {page['url']}\nTEXT:\n{page['text']}"
        for page in pages
        if page.get("text")
    )[:24000]
    if not page_blob:
        return {}

    prompt = (
        "Extract treasury-holdings inputs for a public company from its own website text.\n"
        f"Symbol: {symbol}\n"
        f"Expected crypto asset focus: {asset}\n\n"
        "Rules:\n"
        "- Only use facts clearly stated in the website text.\n"
        "- If the company's digital asset units are not explicitly stated, return null for units.\n"
        "- If shares outstanding are not explicitly stated, return null for shares_outstanding.\n"
        "- Return numeric values only, not words.\n"
        "- confidence must be between 0 and 1 and low unless the statement is explicit.\n"
        "- Return strict JSON with keys: units, shares_outstanding, asset, as_of_date, confidence, evidence.\n\n"
        f"Website text:\n{page_blob}"
    )
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        if isinstance(content, list):
            content = "\n".join(str(x) for x in content)
        cleaned = str(content).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            return {}
        units = payload.get("units")
        shares = payload.get("shares_outstanding")
        return {
            "units": None if units in (None, "") else float(units),
            "shares_outstanding": None if shares in (None, "") else float(shares),
            "asset": str(payload.get("asset") or asset).upper().strip(),
            "as_of_date": payload.get("as_of_date"),
            "confidence": max(0.0, min(1.0, _safe_float(payload.get("confidence"), 0.2))),
            "evidence": str(payload.get("evidence") or "")[:500],
            "source": "llm_company_website",
            "website": website,
        }
    except Exception:
        return {}


def _is_crypto_symbol(symbol: str) -> bool:
    text = str(symbol).upper()
    return text.endswith("-USD") or "/" in text


def _alpaca_symbol(symbol: str) -> str:
    text = str(symbol).upper().strip()
    if text.endswith("-USD"):
        return text.replace("-USD", "/USD")
    return text


def _period_to_start(period: str) -> datetime:
    period = str(period).strip().lower()
    now = datetime.now(timezone.utc)
    mapping = {
        "10d": timedelta(days=10),
        "1mo": timedelta(days=31),
        "3mo": timedelta(days=92),
        "6mo": timedelta(days=183),
        "1y": timedelta(days=366),
        "2y": timedelta(days=732),
        "5y": timedelta(days=1830),
        "10y": timedelta(days=3650),
    }
    return now - mapping.get(period, timedelta(days=183))


def _interval_to_timeframe(interval: str) -> str:
    mapping = {
        "1d": "1Day",
        "1wk": "1Week",
        "1mo": "1Month",
    }
    return mapping.get(str(interval).strip().lower(), "1Day")


def _bars_to_df(pd_module, bars: list[dict]) -> "pd.DataFrame":
    rows = []
    for bar in bars:
        rows.append(
            {
                "Date": bar.get("t"),
                "Open": _safe_float(bar.get("o"), 0.0),
                "High": _safe_float(bar.get("h"), 0.0),
                "Low": _safe_float(bar.get("l"), 0.0),
                "Close": _safe_float(bar.get("c"), 0.0),
                "Volume": _safe_float(bar.get("v"), 0.0),
            }
        )
    if not rows:
        return pd_module.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = pd_module.DataFrame(rows)
    df["Date"] = pd_module.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_market_history(symbol: str, period: str = "6mo", interval: str = "1d"):
    _, pd, _, _ = _get_market_deps()
    symbol = symbol.upper().strip()
    timeframe = _interval_to_timeframe(interval)
    start = _period_to_start(period).isoformat()
    end = datetime.now(timezone.utc).isoformat()

    if _is_crypto_symbol(symbol):
        payload = _alpaca_data_request(
            "/v1beta3/crypto/us/bars",
            {
                "symbols": _alpaca_symbol(symbol),
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "sort": "asc",
                "limit": 10000,
            },
        )
        bars = (payload.get("bars") or {}).get(_alpaca_symbol(symbol), [])
        return _bars_to_df(pd, bars)

    payload = _alpaca_data_request(
        f"/v2/stocks/{symbol}/bars",
        {
            "timeframe": timeframe,
            "start": start,
            "end": end,
            "sort": "asc",
            "adjustment": "raw",
            "feed": "iex",
            "limit": 10000,
        },
    )
    bars = payload.get("bars", []) if isinstance(payload, dict) else []
    return _bars_to_df(pd, bars)


def _latest_close(yf_module, symbol: str) -> float:
    history = _fetch_market_history(symbol, period="10d", interval="1d")
    if history.empty:
        raise ValueError(f"No market data for {symbol}")
    return float(history["Close"].dropna().iloc[-1])


def fetch_treasury_data(
    urls: list[str] | None = None,
    retries: int = 2,
    timeout: int = 20,
    use_cache: bool = True,
) -> list[dict]:
    urls = urls or [
        "https://bitcointreasuries.net/",
        "https://coinmarketcap.com/view/bitcoin-treasuries/",
        "https://coinmarketcap.com/view/ethereum-treasuries/",
    ]
    cache = _load_reference_cache() if use_cache else {}
    fetched = []

    for url in urls:
        cached = cache.get(url)
        if use_cache and isinstance(cached, dict):
            fetched.append(dict(cached, cache_hit=True))
            continue

        last_error = None
        for _ in range(max(1, int(retries))):
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                item = {
                    "url": url,
                    "status": "ok",
                    "fetched_at_utc": _now_iso(),
                    "content_sample": resp.text[:1200],
                }
                fetched.append(item)
                if use_cache:
                    cache[url] = item
                last_error = None
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.5)
        if last_error:
            fallback = {"url": url, "status": "error", "error": last_error}
            if use_cache and isinstance(cached, dict):
                fallback["cached_fallback"] = cached
            fetched.append(fallback)

    if use_cache:
        _save_reference_cache(cache)
    return fetched


def _missing_units_symbols(symbols: list[str]) -> list[str]:
    holdings_map = _load_treasury_holdings()
    universe_map = {str(row.get("symbol", "")).upper(): row for row in _load_treasury_universe()}
    missing = []
    for symbol in symbols:
        hold = holdings_map.get(symbol, {})
        units = hold.get("units")
        if units is None:
            units = universe_map.get(symbol, {}).get("units")
        if units is None:
            missing.append(symbol)
    return missing


def _resolve_treasury_inputs(
    symbol: str,
    info: dict | None = None,
    yf_module=None,
    estimated_units_mode: bool = False,
    finnhub_profile: dict | None = None,
) -> dict:
    symbol = symbol.upper().strip()
    holdings_map = _load_treasury_holdings()
    universe_map = {str(row.get("symbol", "")).upper(): row for row in _load_treasury_universe()}
    hold = holdings_map.get(symbol, {})
    urow = universe_map.get(symbol, {})
    info = info or {}

    asset = str(hold.get("asset") or urow.get("asset") or "BTC").upper()
    as_of_date = hold.get("as_of_date")
    confidence = _safe_float(hold.get("confidence"), 0.0)
    warnings_list = []

    units = hold.get("units")
    units_source = hold.get("source") if units is not None else None
    if units is None and urow.get("units") is not None:
        units = float(urow.get("units"))
        units_source = "universe"
        confidence = max(confidence, 0.65)
        if not as_of_date:
            as_of_date = hold.get("updated_at_utc")

    market_cap_hint = _safe_float(urow.get("market_cap_hint_usd"), 0.0)
    if units is None and estimated_units_mode and yf_module is not None and market_cap_hint > 0:
        crypto_symbol = f"{asset}-USD"
        try:
            crypto_px = _latest_close(yf_module, crypto_symbol)
            units = market_cap_hint / max(float(crypto_px), 1e-9)
            units_source = "estimated_market_cap"
            confidence = max(confidence, 0.25)
            warnings_list.append(f"{symbol}: estimated treasury units from market cap hint.")
        except Exception:
            pass

    shares = hold.get("shares_outstanding")
    shares_source = "manual" if shares is not None else None
    if shares is None:
        profile = finnhub_profile or {}
        finnhub_shares = _safe_float(
            profile.get("shareOutstanding")
            if isinstance(profile, dict)
            else None,
            0.0,
        )
        if finnhub_shares > 0:
            shares = finnhub_shares
            shares_source = "finnhub"
            confidence = max(confidence, 0.8 if units is not None else 0.6)
        else:
            profile = {}
        yahoo_shares = _safe_float(info.get("sharesOutstanding"), 0.0)
        if shares is None and yahoo_shares > 0:
            shares = yahoo_shares
            shares_source = "yahoo"
            confidence = max(confidence, 0.55 if units is not None else 0.35)

    llm_fallback = {}
    if units is None or shares is None:
        llm_fallback = _llm_extract_treasury_inputs_from_company_website(
            symbol=symbol,
            asset=asset,
            finnhub_profile=finnhub_profile,
        )
        if units is None and llm_fallback.get("units") is not None:
            units = float(llm_fallback["units"])
            units_source = str(llm_fallback.get("source") or "llm_company_website")
            confidence = max(confidence, min(0.45, _safe_float(llm_fallback.get("confidence"), 0.2)))
            warnings_list.append(f"{symbol}: treasury units inferred from company website via LLM fallback.")
        if shares is None and llm_fallback.get("shares_outstanding") is not None:
            shares = float(llm_fallback["shares_outstanding"])
            shares_source = str(llm_fallback.get("source") or "llm_company_website")
            confidence = max(confidence, min(0.4, _safe_float(llm_fallback.get("confidence"), 0.2)))
            warnings_list.append(f"{symbol}: shares outstanding inferred from company website via LLM fallback.")

    if not as_of_date and isinstance(finnhub_profile, dict):
        ipo = finnhub_profile.get("ipo")
        if ipo:
            as_of_date = ipo
    if not as_of_date and llm_fallback.get("as_of_date"):
        as_of_date = llm_fallback.get("as_of_date")

    staleness_days = _holding_staleness_days(as_of_date)
    return {
        "symbol": symbol,
        "asset": asset,
        "units": None if units is None else float(units),
        "units_source": units_source,
        "shares_outstanding": None if shares is None else float(shares),
        "shares_source": shares_source,
        "as_of_date": as_of_date,
        "staleness_days": staleness_days,
        "confidence": round(float(confidence), 4),
        "llm_fallback": llm_fallback,
        "warnings": warnings_list,
    }


def refresh_treasury_holdings(
    watchlist: str = "",
    use_reference_pages: bool = True,
    missing_only: bool = False,
) -> dict:
    symbols = [s.strip().upper() for s in (watchlist or ",".join(STRATEGY_CONFIG["watchlist"])).split(",") if s.strip()]
    _, _, yf, _ = _get_market_deps()
    universe_map = {str(row.get("symbol", "")).upper(): row for row in _load_treasury_universe()}
    holdings = _load_treasury_holdings()
    refreshed = {}
    updated = []
    errors = []
    fetched_pages = []

    if use_reference_pages:
        try:
            fetched_pages = fetch_treasury_data()
        except Exception as exc:
            errors.append(f"reference_fetch_failed: {exc}")

    for symbol in symbols:
        existing = holdings.get(symbol, {})
        if missing_only and existing.get("units") is not None and existing.get("shares_outstanding") is not None:
            refreshed[symbol] = existing
            continue

        finnhub_profile = {}
        try:
            finnhub_profile = _finnhub_company_profile2(symbol)
        except Exception as exc:
            errors.append(f"{symbol}: finnhub_profile_failed: {exc}")

        resolved = _resolve_treasury_inputs(
            symbol,
            info={},
            yf_module=yf,
            estimated_units_mode=bool(STRATEGY_CONFIG.get("estimated_units_mode", False)),
            finnhub_profile=finnhub_profile,
        )
        has_any_value = resolved.get("units") is not None or resolved.get("shares_outstanding") is not None
        merged = {
            "units": resolved.get("units"),
            "shares_outstanding": resolved.get("shares_outstanding"),
            "asset": resolved.get("asset") or universe_map.get(symbol, {}).get("asset") or "BTC",
            "updated_at_utc": _now_iso(),
            "source": ",".join(
                part for part in [resolved.get("units_source"), resolved.get("shares_source")] if part
            ) or existing.get("source") or "unknown",
            "as_of_date": resolved.get("as_of_date") or (datetime.now(timezone.utc).date().isoformat() if has_any_value else existing.get("as_of_date")),
            "confidence": resolved.get("confidence"),
        }
        llm_fallback = resolved.get("llm_fallback") or {}
        if llm_fallback.get("evidence"):
            merged["llm_evidence"] = llm_fallback.get("evidence")
        if llm_fallback.get("website"):
            merged["website"] = llm_fallback.get("website")
        refreshed[symbol] = _normalize_holding_record(symbol, merged, universe_map)
        updated.append(
            {
                "symbol": symbol,
                "units": refreshed[symbol].get("units"),
                "shares_outstanding": refreshed[symbol].get("shares_outstanding"),
                "source": refreshed[symbol].get("source"),
                "confidence": refreshed[symbol].get("confidence"),
                "staleness_days": refreshed[symbol].get("staleness_days"),
            }
        )

    for symbol, row in holdings.items():
        if symbol not in refreshed:
            refreshed[symbol] = _normalize_holding_record(symbol, row, universe_map)

    _save_treasury_holdings(refreshed)
    return {
        "status": "ok",
        "symbols_checked": symbols,
        "updated_records": updated,
        "reference_pages_checked": len(fetched_pages),
        "errors": errors,
        "generated_at_utc": _now_iso(),
    }


def validate_data_quality(
    watchlist: str = "",
    max_missing_holdings_pct: float | None = None,
) -> dict:
    symbols = [s.strip().upper() for s in (watchlist or ",".join(STRATEGY_CONFIG["watchlist"])).split(",") if s.strip()]
    holdings_map = _load_treasury_holdings()
    max_missing = (
        float(max_missing_holdings_pct)
        if max_missing_holdings_pct is not None
        else float(STRATEGY_CONFIG.get("missing_holdings_abort_threshold_pct", 0.30))
    )
    missing_units = _missing_units_symbols(symbols)
    stale_after_days = int(STRATEGY_CONFIG.get("holdings_stale_after_days", 42))
    stale_symbols = [
        symbol
        for symbol in symbols
        if (holdings_map.get(symbol, {}).get("staleness_days") or 0) > stale_after_days
    ]
    missing_shares = [
        symbol
        for symbol in symbols
        if holdings_map.get(symbol, {}).get("shares_outstanding") is None
    ]
    missing_pct = (len(missing_units) / len(symbols)) if symbols else 0.0
    report = {
        "watchlist": symbols,
        "checked_symbols": len(symbols),
        "missing_units_symbols": missing_units,
        "missing_units_count": len(missing_units),
        "missing_units_pct": round(missing_pct, 4),
        "missing_shares_symbols": missing_shares,
        "missing_shares_count": len(missing_shares),
        "stale_symbols": stale_symbols,
        "stale_symbols_count": len(stale_symbols),
        "stale_after_days": stale_after_days,
        "threshold_pct": round(max_missing, 4),
        "abort_recommended": bool(symbols and missing_pct > max_missing),
        "generated_at_utc": _now_iso(),
    }
    return report


def _model_from_history(np_module, history) -> dict:
    closes = history["Close"].dropna().tail(252)
    if len(closes) < 80:
        raise ValueError("Not enough history for modeling")

    returns = closes.pct_change().dropna()
    if len(returns) < 60:
        raise ValueError("Not enough return observations")

    price = float(closes.iloc[-1])
    mom_20 = float((closes.iloc[-1] / closes.iloc[-21]) - 1.0)
    mom_60 = float((closes.iloc[-1] / closes.iloc[-61]) - 1.0)

    x = np_module.arange(60)
    y = np_module.log(closes.tail(60).to_numpy())
    slope, _ = np_module.polyfit(x, y, 1)
    trend_component = float(np_module.exp(slope) - 1.0)

    ma_20 = float(closes.tail(20).mean())
    mean_reversion_component = float(-0.35 * ((price / ma_20) - 1.0))

    predicted_1d_return = float(
        0.55 * trend_component + 0.20 * (mom_20 / 20.0) + 0.10 * (mom_60 / 60.0) + mean_reversion_component
    )

    vol_20 = float(returns.tail(20).std() * np_module.sqrt(252.0))
    confidence = float(max(0.05, min(0.95, 1.0 - (vol_20 / 1.5))))

    predicted_price = float(price * (1.0 + predicted_1d_return))
    stop_buffer = float(max(0.015, min(0.08, vol_20 / 6.0)))
    take_profit_buffer = float(max(0.02, min(0.15, abs(predicted_1d_return) * 3.0)))

    signal = "HOLD"
    if predicted_1d_return > 0.004 and confidence > 0.45:
        signal = "BUY"
    elif predicted_1d_return < -0.004 and confidence > 0.45:
        signal = "SELL"

    return {
        "price": round(price, 4),
        "predicted_1d_return": round(predicted_1d_return, 6),
        "predicted_price": round(predicted_price, 4),
        "confidence": round(confidence, 4),
        "volatility_annualized": round(vol_20, 4),
        "momentum_20d": round(mom_20, 4),
        "momentum_60d": round(mom_60, 4),
        "stop_loss_price": round(price * (1.0 - stop_buffer), 4),
        "take_profit_price": round(price * (1.0 + take_profit_buffer), 4),
        "signal": signal,
    }


def search_tool(query: str) -> str:
    """Search the web for information."""
    try:
        search = DuckDuckGoSearchRun()
    except ImportError:
        return "Missing dependency: ddgs. Install with `pip install -U ddgs`."
    return search.run(query)


def yahoo_finance_tool(symbol: str, period: str = "6mo", interval: str = "1d") -> str:
    """Fetch historical market data for a ticker using Alpaca market data."""
    try:
        history = _fetch_market_history(symbol.upper(), period=period, interval=interval)
    except RuntimeError as exc:
        return str(exc)
    if history.empty:
        return f"No Alpaca market data found for {symbol.upper()}."

    latest = history.tail(5)[["Open", "High", "Low", "Close", "Volume"]]
    return (
        f"Alpaca market data for {symbol.upper()} ({period}, {interval})\n"
        f"{latest.to_string()}"
    )


def model_analyst_tool(symbol: str, lookback_days: int = 252) -> str:
    """Build a quantitative model and return analyst-style metrics for a ticker."""
    try:
        np, _, _, _ = _get_market_deps()
        history = _fetch_market_history(symbol.upper(), period="2y", interval="1d")
        if history.empty:
            return json.dumps({"error": f"No market data for {symbol.upper()}"})
        modeled = _model_from_history(np, history.tail(max(lookback_days, 80)))
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    payload = {
        "symbol": symbol.upper(),
        "model": "hybrid trend + mean-reversion",
        "timeframe": "next trading day",
        "metrics": modeled,
        "generated_at_utc": _now_iso(),
    }
    return json.dumps(payload, indent=2)


def _rsi_series(close_series, period: int = 14):
    delta = close_series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_snapshot(close_series, np_module):
    if len(close_series) < 40:
        return {"error": "not enough bars for MACD"}
    ema12 = _ema_series(close_series.to_numpy(), 12)
    ema26 = _ema_series(close_series.to_numpy(), 26)
    macd = np_module.array(ema12) - np_module.array(ema26)
    signal = np_module.array(_ema_series(macd, 9))
    hist = macd - signal
    return {
        "macd": round(float(macd[-1]), 4),
        "signal": round(float(signal[-1]), 4),
        "hist": round(float(hist[-1]), 4),
        "slope_up": bool(macd[-1] > macd[-2]),
        "hist_green": bool(hist[-1] > 0),
        "hist_increasing": bool(hist[-1] > hist[-2]),
        "bullish_cross": bool(macd[-1] > signal[-1] and macd[-2] <= signal[-2]),
    }


def _bollinger_snapshot(close_series):
    if len(close_series) < 40:
        return {"error": "not enough bars for Bollinger"}
    mid = close_series.rolling(20).mean()
    std = close_series.rolling(20).std(ddof=1)
    up = mid + (2.0 * std)
    lo = mid - (2.0 * std)
    width = (up - lo) / (mid + 1e-9)
    w_now = float(width.iloc[-1])
    w_avg = float(width.tail(20).mean())
    close_now = float(close_series.iloc[-1])
    return {
        "mid": round(float(mid.iloc[-1]), 4),
        "upper": round(float(up.iloc[-1]), 4),
        "lower": round(float(lo.iloc[-1]), 4),
        "width": round(w_now, 4),
        "width_expanding": bool(w_now > w_avg * 1.05),
        "near_upper_band": bool(close_now >= float(up.iloc[-1]) * 0.985),
        "near_lower_band": bool(close_now <= float(lo.iloc[-1]) * 1.015),
    }


def _ichimoku_state(df):
    if len(df) < 120:
        return {"state": "insufficient_data"}
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    conv = (high.rolling(9).max() + low.rolling(9).min()) / 2.0
    base = (high.rolling(26).max() + low.rolling(26).min()) / 2.0
    span_a = ((conv + base) / 2.0).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2.0).shift(26)
    ca = float(span_a.iloc[-1]) if not span_a.empty else 0.0
    cb = float(span_b.iloc[-1]) if not span_b.empty else 0.0
    top = max(ca, cb)
    bot = min(ca, cb)
    c = float(close.iloc[-1])
    state = "inside"
    if c > top:
        state = "above"
    elif c < bot:
        state = "below"
    return {
        "state": state,
        "conversion": round(float(conv.iloc[-1]), 4),
        "base": round(float(base.iloc[-1]), 4),
        "cloud_top": round(top, 4),
        "cloud_bottom": round(bot, 4),
    }


def _latest_candle_pattern(df):
    if len(df) < 2:
        return {"pattern": "unknown"}
    row = df.iloc[-1]
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    body = abs(c - o)
    rng = max(1e-9, h - l)
    lower = min(o, c) - l
    upper = h - max(o, c)
    hammer = body <= 0.35 * rng and lower >= 2.0 * body and upper <= body
    inv_hammer = body <= 0.35 * rng and upper >= 2.0 * body and lower <= body
    return {
        "pattern": "hammer" if hammer else "inverted_hammer" if inv_hammer else "none",
        "red_candle": bool(c < o),
    }


def _fib_extensions(df):
    if len(df) < 120:
        return {"error": "not enough bars for fib"}
    swing = df.tail(180)
    hi = float(swing["High"].max())
    lo = float(swing["Low"].min())
    diff = max(1e-9, hi - lo)
    return {
        "swing_low": round(lo, 4),
        "swing_high": round(hi, 4),
        "1.272": round(hi + 0.272 * diff, 4),
        "1.414": round(hi + 0.414 * diff, 4),
        "1.618": round(hi + 0.618 * diff, 4),
        "0.886_retracement": round(hi - 0.886 * diff, 4),
    }


def technical_stock_analysis_tool(symbol: str) -> str:
    """
    Multi-timeframe technical analysis:
    consolidation, gap, Ichimoku, MACD, RSI, Bollinger, candles, FIB.
    """
    try:
        np, _, _, _ = _get_market_deps()
        daily = _fetch_market_history(symbol.upper(), period="2y", interval="1d")
        weekly = _fetch_market_history(symbol.upper(), period="5y", interval="1wk")
        monthly = _fetch_market_history(symbol.upper(), period="10y", interval="1mo")
        if daily.empty or weekly.empty:
            return json.dumps({"error": f"insufficient price data for {symbol.upper()}"}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    # consolidation estimate by range compression
    cons_6m = daily.tail(126)
    cons_1y = daily.tail(252)
    cons_5y = weekly.tail(260)
    r6 = (float(cons_6m["High"].max()) - float(cons_6m["Low"].min())) / max(1e-9, float(cons_6m["Low"].min()))
    r1y = (float(cons_1y["High"].max()) - float(cons_1y["Low"].min())) / max(1e-9, float(cons_1y["Low"].min()))
    r5y = (float(cons_5y["High"].max()) - float(cons_5y["Low"].min())) / max(1e-9, float(cons_5y["Low"].min()))
    consolidation = "none"
    if r5y < 1.2:
        consolidation = "long_5y"
    elif r1y < 0.8:
        consolidation = "long_1y"
    elif r6 < 0.5:
        consolidation = "mid_6m"

    # breakaway gap
    gap = {"exists": False, "direction": "none"}
    if len(daily) >= 2:
        prev = daily.iloc[-2]
        cur = daily.iloc[-1]
        if float(cur["Low"]) > float(prev["High"]) * 1.01:
            gap = {"exists": True, "direction": "up"}
        elif float(cur["High"]) < float(prev["Low"]) * 0.99:
            gap = {"exists": True, "direction": "down"}

    d_close = daily["Close"].dropna()
    w_close = weekly["Close"].dropna()
    m_close = monthly["Close"].dropna() if not monthly.empty else d_close

    macd_d = _macd_snapshot(d_close, np)
    macd_w = _macd_snapshot(w_close, np)
    rsi_d = float(_rsi_series(d_close).iloc[-1]) if len(d_close) > 20 else 50.0
    rsi_w = float(_rsi_series(w_close).iloc[-1]) if len(w_close) > 20 else 50.0
    rsi_m = float(_rsi_series(m_close).iloc[-1]) if len(m_close) > 20 else 50.0
    boll_d = _bollinger_snapshot(d_close)
    boll_w = _bollinger_snapshot(w_close) if len(w_close) >= 40 else {"error": "insufficient_weekly"}
    ichi_d = _ichimoku_state(daily)
    ichi_w = _ichimoku_state(weekly)
    candle_d = _latest_candle_pattern(daily)
    candle_w = _latest_candle_pattern(weekly)
    candle_m = _latest_candle_pattern(monthly) if not monthly.empty else {"pattern": "unknown"}
    fib = _fib_extensions(daily)

    verdict_points = []
    if gap["exists"] and gap["direction"] == "up":
        verdict_points.append("breakaway gap up")
    if macd_d.get("slope_up") and macd_d.get("hist_green"):
        verdict_points.append("daily MACD trend improving")
    if macd_w.get("slope_up") and macd_w.get("hist_green"):
        verdict_points.append("weekly MACD supportive")
    if ichi_d.get("state") == "above":
        verdict_points.append("daily above Ichimoku cloud")
    if rsi_d > 50:
        verdict_points.append("daily RSI > 50")
    if boll_d.get("width_expanding"):
        verdict_points.append("volatility expansion at bands")
    if candle_w.get("red_candle"):
        verdict_points.append("weekly active red candle risk")

    verdict = "neutral"
    bullish_count = sum(
        [
            bool(gap["exists"] and gap["direction"] == "up"),
            bool(macd_d.get("slope_up")),
            bool(macd_d.get("hist_green")),
            bool(ichi_d.get("state") == "above"),
            bool(rsi_d > 50),
            bool(boll_d.get("near_upper_band")),
        ]
    )
    bearish_count = sum(
        [
            bool(candle_w.get("red_candle")),
            bool(ichi_d.get("state") == "below"),
            bool(rsi_d < 50),
            bool(macd_d.get("slope_up") is False),
        ]
    )
    if bullish_count >= 4 and bearish_count <= 1:
        verdict = "bullish"
    elif bearish_count >= 3:
        verdict = "bearish"

    payload = {
        "symbol": symbol.upper(),
        "generated_at_utc": _now_iso(),
        "consolidation_state": consolidation,
        "breakaway_gap": gap,
        "ichimoku_daily": ichi_d,
        "ichimoku_weekly": ichi_w,
        "macd_daily": macd_d,
        "macd_weekly": macd_w,
        "rsi_daily": round(rsi_d, 2),
        "rsi_weekly": round(rsi_w, 2),
        "rsi_monthly": round(rsi_m, 2),
        "bollinger_daily": boll_d,
        "bollinger_weekly": boll_w,
        "candle_daily": candle_d,
        "candle_weekly": candle_w,
        "candle_monthly": candle_m,
        "fib_levels": fib,
        "verdict": verdict,
        "key_points": verdict_points,
    }
    return json.dumps(payload, indent=2)


def technical_watchlist_analysis_tool(watchlist: str) -> str:
    """Run multi-timeframe technical analysis across a watchlist."""
    symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    results = []
    for symbol in symbols:
        raw = technical_stock_analysis_tool(symbol)
        try:
            results.append(json.loads(raw))
        except Exception:
            results.append({"symbol": symbol, "error": raw})
    return json.dumps({"generated_at_utc": _now_iso(), "results": results}, indent=2)


def paper_account_status_tool() -> str:
    """Return the current fake portfolio state and mark-to-market equity."""
    account = _load_account()

    try:
        _, _, yf, _ = _get_market_deps()
    except RuntimeError as exc:
        return str(exc)

    positions_value = 0.0
    enriched_positions = {}
    for symbol, pos in account["positions"].items():
        qty = float(pos.get("quantity", 0.0))
        if qty <= 0:
            continue
        try:
            last_price = _latest_close(yf, symbol)
        except Exception:
            last_price = float(pos.get("avg_price", 0.0))
        market_value = qty * last_price
        positions_value += market_value
        enriched_positions[symbol] = {
            "quantity": round(qty, 4),
            "avg_price": round(float(pos.get("avg_price", 0.0)), 4),
            "last_price": round(last_price, 4),
            "market_value": round(market_value, 2),
        }

    equity = float(account["cash"]) + positions_value
    payload = {
        "cash": round(float(account["cash"]), 2),
        "positions_value": round(positions_value, 2),
        "equity": round(equity, 2),
        "positions": enriched_positions,
        "trade_count": len(account.get("trades", [])),
        "last_updated": account.get("last_updated"),
    }
    return json.dumps(payload, indent=2)


def trade_signal_tool(symbol: str, risk_budget_pct: float = 0.01) -> str:
    """Generate model-driven trade signal and position size for fake account trading."""
    symbol = symbol.upper()
    risk_budget_pct = float(max(0.001, min(0.05, risk_budget_pct)))

    try:
        np, _, _, _ = _get_market_deps()
        history = _fetch_market_history(symbol, period="2y", interval="1d")
        modeled = _model_from_history(np, history)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    account = _load_account()
    status = json.loads(paper_account_status_tool())
    equity = float(status.get("equity", account["cash"]))
    cash = float(account["cash"])

    signal = modeled["signal"]
    price = float(modeled["price"])
    stop_loss = float(modeled["stop_loss_price"])

    per_share_risk = max(0.01, abs(price - stop_loss))
    risk_budget_dollars = equity * risk_budget_pct
    base_qty = int(risk_budget_dollars // per_share_risk)

    held = int(float(account["positions"].get(symbol, {}).get("quantity", 0.0)))
    if signal == "BUY":
        affordable_qty = int(cash // price)
        qty = max(0, min(base_qty, affordable_qty))
    elif signal == "SELL":
        qty = held
    else:
        qty = 0

    payload = {
        "symbol": symbol,
        "signal": signal,
        "recommended_quantity": qty,
        "risk_budget_pct": risk_budget_pct,
        "risk_budget_dollars": round(risk_budget_dollars, 2),
        "model_metrics": modeled,
        "position_before": account["positions"].get(symbol, {"quantity": 0, "avg_price": 0}),
        "generated_at_utc": _now_iso(),
    }
    return json.dumps(payload, indent=2)


def execute_paper_trade_tool(symbol: str, side: str, quantity: int) -> str:
    """Execute a BUY/SELL against a fake account and persist updated balances."""
    symbol = symbol.upper()
    side = side.upper().strip()
    quantity = int(quantity)

    if side not in {"BUY", "SELL"}:
        return json.dumps({"error": "side must be BUY or SELL"})
    if quantity <= 0:
        return json.dumps({"error": "quantity must be > 0"})

    try:
        _, _, yf, _ = _get_market_deps()
        market_price = _latest_close(yf, symbol)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    account = _load_account()
    positions = account.setdefault("positions", {})
    trades = account.setdefault("trades", [])

    slippage = 0.0005
    fill_price = market_price * (1.0 + slippage if side == "BUY" else 1.0 - slippage)
    notional = fill_price * quantity

    position = positions.get(symbol, {"quantity": 0.0, "avg_price": 0.0})
    old_qty = float(position.get("quantity", 0.0))
    old_avg = float(position.get("avg_price", 0.0))

    if side == "BUY":
        if account["cash"] < notional:
            return json.dumps(
                {
                    "error": "Insufficient cash",
                    "cash": round(float(account["cash"]), 2),
                    "required": round(notional, 2),
                },
                indent=2,
            )
        new_qty = old_qty + quantity
        new_avg = ((old_qty * old_avg) + notional) / new_qty
        account["cash"] = float(account["cash"]) - notional
        positions[symbol] = {"quantity": new_qty, "avg_price": new_avg}
        realized_pnl = 0.0
    else:
        if old_qty < quantity:
            return json.dumps(
                {
                    "error": "Insufficient shares",
                    "held": round(old_qty, 4),
                    "requested": quantity,
                },
                indent=2,
            )
        proceeds = notional
        realized_pnl = (fill_price - old_avg) * quantity
        new_qty = old_qty - quantity
        account["cash"] = float(account["cash"]) + proceeds
        if new_qty <= 0:
            positions.pop(symbol, None)
        else:
            positions[symbol] = {"quantity": new_qty, "avg_price": old_avg}

    trade = {
        "timestamp_utc": _now_iso(),
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "fill_price": round(fill_price, 4),
        "notional": round(notional, 2),
        "realized_pnl": round(realized_pnl, 2),
    }
    trades.append(trade)

    _save_account(account)
    account_snapshot = json.loads(paper_account_status_tool())
    discord = _send_discord_message(
        f"[LOCAL PAPER TRADE] {side} {quantity} {symbol} @ {round(fill_price, 4)} | Notional ${round(notional, 2)}"
    )

    return json.dumps(
        {
            "status": "filled",
            "trade": trade,
            "account_snapshot": account_snapshot,
            "discord_notification": discord,
        },
        indent=2,
    )


def auto_trade_watchlist_tool(
    watchlist: str = "MSTR,COIN,MARA,RIOT,TSLA,NVDA",
    risk_budget_pct: float = 0.01,
    max_new_trades: int = 2,
) -> str:
    """Run signal generation on a watchlist and execute paper trades automatically."""
    symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    max_new_trades = max(0, int(max_new_trades))

    decisions = []
    trades_placed = 0

    for symbol in symbols:
        signal_payload = json.loads(trade_signal_tool(symbol, risk_budget_pct))
        if signal_payload.get("error"):
            decisions.append({"symbol": symbol, "error": signal_payload["error"]})
            continue

        signal = signal_payload["signal"]
        qty = int(signal_payload.get("recommended_quantity", 0))

        if signal == "BUY" and qty > 0 and trades_placed < max_new_trades:
            order = json.loads(execute_paper_trade_tool(symbol, "BUY", qty))
            decisions.append({"symbol": symbol, "decision": "BUY", "order": order})
            if not order.get("error"):
                trades_placed += 1
        elif signal == "SELL" and qty > 0:
            order = json.loads(execute_paper_trade_tool(symbol, "SELL", qty))
            decisions.append({"symbol": symbol, "decision": "SELL", "order": order})
        else:
            decisions.append(
                {
                    "symbol": symbol,
                    "decision": "HOLD",
                    "model_signal": signal,
                }
            )

    return json.dumps(
        {
            "watchlist": symbols,
            "decisions": decisions,
            "completed_at_utc": _now_iso(),
        },
        indent=2,
    )


def broker_paper_account_status_tool() -> str:
    """Return real broker paper account status from Alpaca API."""
    try:
        account = _alpaca_request("GET", "/v2/account")
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    payload = {
        "broker": "alpaca_paper",
        "status": account.get("status"),
        "buying_power": account.get("buying_power"),
        "cash": account.get("cash"),
        "equity": account.get("equity"),
        "portfolio_value": account.get("portfolio_value"),
        "long_market_value": account.get("long_market_value"),
        "short_market_value": account.get("short_market_value"),
        "pattern_day_trader": account.get("pattern_day_trader"),
        "trading_blocked": account.get("trading_blocked"),
        "updated_at_utc": _now_iso(),
    }
    return json.dumps(payload, indent=2)


def broker_positions_tool() -> str:
    """List current real broker paper positions from Alpaca API."""
    try:
        positions = _alpaca_request("GET", "/v2/positions")
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    trimmed = []
    for p in positions:
        trimmed.append(
            {
                "symbol": p.get("symbol"),
                "qty": p.get("qty"),
                "avg_entry_price": p.get("avg_entry_price"),
                "market_value": p.get("market_value"),
                "unrealized_pl": p.get("unrealized_pl"),
                "unrealized_plpc": p.get("unrealized_plpc"),
                "side": p.get("side"),
            }
        )
    return json.dumps({"broker": "alpaca_paper", "positions": trimmed}, indent=2)


def broker_submit_order_tool(
    symbol: str,
    side: str,
    quantity: float,
    order_type: str = "market",
    time_in_force: str = "day",
) -> str:
    """Submit an order to Alpaca paper account."""
    symbol = symbol.upper().strip()
    side = side.lower().strip()
    order_type = order_type.lower().strip()
    time_in_force = time_in_force.lower().strip()
    quantity = float(quantity)

    if side not in {"buy", "sell"}:
        return json.dumps({"error": "side must be buy or sell"}, indent=2)
    if quantity <= 0:
        return json.dumps({"error": "quantity must be > 0"}, indent=2)
    if order_type not in {"market", "limit"}:
        return json.dumps({"error": "order_type must be market or limit"}, indent=2)

    payload = {
        "symbol": symbol,
        "qty": str(quantity),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }

    try:
        order = _alpaca_request("POST", "/v2/orders", payload)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    trimmed = {
        "id": order.get("id"),
        "client_order_id": order.get("client_order_id"),
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "qty": order.get("qty"),
        "type": order.get("type"),
        "time_in_force": order.get("time_in_force"),
        "status": order.get("status"),
        "submitted_at": order.get("submitted_at"),
    }
    discord = _send_discord_message(
        "[ALPACA PAPER ORDER] "
        f"{trimmed.get('side', side).upper()} {trimmed.get('qty', quantity)} {symbol} "
        f"| type={trimmed.get('type', order_type)} tif={trimmed.get('time_in_force', time_in_force)} "
        f"| status={trimmed.get('status', 'unknown')}"
    )
    return json.dumps(
        {"broker": "alpaca_paper", "order": trimmed, "discord_notification": discord},
        indent=2,
    )


def broker_cancel_all_orders_tool() -> str:
    """Cancel all open Alpaca paper orders."""
    try:
        result = _alpaca_request("DELETE", "/v2/orders")
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    return json.dumps(
        {
            "broker": "alpaca_paper",
            "status": "cancel_requested",
            "result": result,
            "updated_at_utc": _now_iso(),
        },
        indent=2,
    )


def broker_daily_summary_tool(report_date_utc: str = "") -> str:
    """
    Return daily PnL proxy and today's fills for Alpaca paper account.
    PnL proxy uses account.equity - account.last_equity.
    """
    try:
        account = _alpaca_request("GET", "/v2/account")
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)

    date_text = report_date_utc.strip() if report_date_utc else datetime.now(timezone.utc).date().isoformat()
    pnl = _safe_float(account.get("equity"), 0.0) - _safe_float(account.get("last_equity"), 0.0)

    fills = []
    fill_errors = None
    try:
        acts = _alpaca_request("GET", f"/v2/account/activities/FILL?date={date_text}")
        if isinstance(acts, list):
            for a in acts:
                fills.append(
                    {
                        "symbol": a.get("symbol"),
                        "side": a.get("side"),
                        "qty": a.get("qty"),
                        "price": a.get("price"),
                        "transaction_time": a.get("transaction_time"),
                        "order_id": a.get("order_id"),
                    }
                )
    except Exception as exc:
        fill_errors = str(exc)

    return json.dumps(
        {
            "broker": "alpaca_paper",
            "report_date_utc": date_text,
            "daily_pnl_usd": round(pnl, 2),
            "fills_count": len(fills),
            "fills": fills,
            "fills_error": fill_errors,
            "generated_at_utc": _now_iso(),
        },
        indent=2,
    )


def send_discord_report_tool(content: str) -> str:
    """Send a report message to Discord webhook."""
    result = _send_discord_message(content)
    return json.dumps({"status": "ok", "discord": result}, indent=2)


def broker_auto_trade_watchlist_tool(
    watchlist: str = "MSTR,COIN,MARA,RIOT,CLSK,HUT",
    risk_budget_pct: float = 0.01,
    max_new_trades: int = 2,
) -> str:
    """
    Generate model signals and place orders in Alpaca paper account.
    Uses market orders and only places up to max_new_trades buys per run.
    """
    symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    max_new_trades = max(0, int(max_new_trades))
    trades_placed = 0
    decisions = []

    for symbol in symbols:
        signal_payload = json.loads(trade_signal_tool(symbol, risk_budget_pct))
        if signal_payload.get("error"):
            decisions.append({"symbol": symbol, "error": signal_payload["error"]})
            continue

        signal = signal_payload.get("signal", "HOLD")
        qty = float(signal_payload.get("recommended_quantity", 0))

        if signal == "BUY" and qty > 0 and trades_placed < max_new_trades:
            order = json.loads(broker_submit_order_tool(symbol, "buy", qty))
            decisions.append({"symbol": symbol, "decision": "BUY", "order": order})
            if not order.get("error"):
                trades_placed += 1
        elif signal == "SELL" and qty > 0:
            order = json.loads(broker_submit_order_tool(symbol, "sell", qty))
            decisions.append({"symbol": symbol, "decision": "SELL", "order": order})
        else:
            decisions.append({"symbol": symbol, "decision": "HOLD", "model_signal": signal})

    return json.dumps(
        {
            "broker": "alpaca_paper",
            "watchlist": symbols,
            "decisions": decisions,
            "completed_at_utc": _now_iso(),
        },
        indent=2,
    )


def strategy_config_tool() -> str:
    """Return currently configured strategy parameters."""
    payload = {
        "strategy": "crypto_treasury_arb_momentum",
        "config": STRATEGY_CONFIG,
        "underlying_map": UNDERLYING_MAP,
        "hedge_etf_map": HEDGE_ETF_MAP,
        "treasury_holdings_file": str(TREASURY_FILE),
    }
    return json.dumps(payload, indent=2)


def update_strategy_config_tool(config_json: str) -> str:
    """Update strategy config keys from a JSON object string."""
    try:
        updates = json.loads(config_json)
        if not isinstance(updates, dict):
            return json.dumps({"error": "config_json must decode to an object"}, indent=2)
        for key, value in updates.items():
            if key in STRATEGY_CONFIG:
                STRATEGY_CONFIG[key] = value
        return json.dumps({"status": "ok", "config": STRATEGY_CONFIG}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)}, indent=2)


def set_high_beta_mode_tool(enabled: bool = True) -> str:
    """Toggle high-beta mode preset for more aggressive risk-taking."""
    enabled = bool(enabled)
    STRATEGY_CONFIG["high_beta_mode"] = enabled
    preset = HIGH_BETA_PRESET if enabled else BASE_PRESET
    for key, value in preset.items():
        STRATEGY_CONFIG[key] = value
    return json.dumps(
        {
            "status": "ok",
            "high_beta_mode": enabled,
            "applied_preset": preset,
            "config": STRATEGY_CONFIG,
        },
        indent=2,
    )


def _candidate_from_config() -> dict:
    return {
        "pricing_lag_threshold_pct": float(STRATEGY_CONFIG.get("pricing_lag_threshold_pct", 0.02)),
        "beta_min": float(STRATEGY_CONFIG.get("beta_min", 1.2)),
        "risk_per_trade_var95": float(STRATEGY_CONFIG.get("risk_per_trade_var95", 0.025)),
        "crypto_rally_daily_gain": float(STRATEGY_CONFIG.get("crypto_rally_daily_gain", 0.03)),
    }


def _mutate_candidate(base: dict) -> dict:
    c = dict(base)
    c["pricing_lag_threshold_pct"] = max(0.005, min(0.06, c["pricing_lag_threshold_pct"] + random.uniform(-0.006, 0.006)))
    c["beta_min"] = max(0.8, min(2.0, c["beta_min"] + random.uniform(-0.2, 0.2)))
    c["risk_per_trade_var95"] = max(0.005, min(0.06, c["risk_per_trade_var95"] + random.uniform(-0.01, 0.01)))
    c["crypto_rally_daily_gain"] = max(0.005, min(0.08, c["crypto_rally_daily_gain"] + random.uniform(-0.01, 0.01)))
    return c


def _metrics_from_returns(np_module, returns_arr):
    if len(returns_arr) < 30:
        return {
            "trades": int(len(returns_arr)),
            "mean_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "win_rate": 0.0,
            "score": -999.0,
        }
    mean_r = float(np_module.mean(returns_arr))
    std_r = float(np_module.std(returns_arr) + 1e-9)
    sharpe = float((mean_r / std_r) * np_module.sqrt(252.0))
    equity = np_module.cumprod(1.0 + returns_arr)
    running_max = np_module.maximum.accumulate(equity)
    dd = (running_max - equity) / np_module.maximum(running_max, 1e-9)
    max_dd = float(np_module.max(dd))
    win_rate = float(np_module.mean(returns_arr > 0))
    score = float(sharpe - (2.5 * max_dd) + (0.2 * win_rate))
    return {
        "trades": int(len(returns_arr)),
        "mean_return": round(mean_r, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "score": round(score, 4),
    }


def _simulate_candidate_returns(candidate: dict, symbols: list[str], lookback_days: int = 220):
    np, _, _, _ = _get_market_deps()
    all_returns = []
    for symbol in symbols:
        symbol = symbol.upper().strip()
        crypto = UNDERLYING_MAP.get(symbol, "BTC-USD")
        s_hist = _fetch_market_history(symbol, period="2y", interval="1d")
        c_hist = _fetch_market_history(crypto, period="2y", interval="1d")
        if s_hist.empty or c_hist.empty:
            continue
        s_close = s_hist["Close"].dropna().tail(lookback_days + 80)
        c_close = c_hist["Close"].dropna().tail(lookback_days + 80)
        joined = s_close.to_frame("stock").join(c_close.to_frame("crypto"), how="inner").dropna()
        if len(joined) < 100:
            continue
        s_rets = joined["stock"].pct_change().dropna()
        c_rets = joined["crypto"].pct_change().dropna()
        close_vals = joined["stock"]

        for i in range(40, len(s_rets) - 1):
            sr = float(s_rets.iloc[i])
            cr = float(c_rets.iloc[i])
            win_s = s_rets.iloc[i - 20 : i]
            win_c = c_rets.iloc[i - 20 : i]
            beta = float((win_s.cov(win_c)) / (win_c.var() + 1e-9))
            expected = beta * cr
            lag = expected - sr

            # BX Trender on trailing window
            tail_close = close_vals.iloc[: i + 1].tail(60)
            sma9 = float(tail_close.tail(9).mean())
            ema11_series = _ema_series(tail_close.to_numpy(), 11)
            ema11 = float(ema11_series[-1])
            ema11_prev = float(ema11_series[-2]) if len(ema11_series) > 1 else ema11
            bb_mid = float(tail_close.tail(20).mean())
            bb_std = float(tail_close.tail(20).std(ddof=1))
            bb_up = bb_mid + 2.0 * bb_std
            bb_lo = bb_mid - 2.0 * bb_std
            px = float(tail_close.iloc[-1])
            bx_score = (0.4 if px > sma9 else -0.4) + (0.35 if px > ema11 else -0.35) + (0.15 if ema11 > ema11_prev else -0.15)
            if bb_up > bb_lo:
                bb_pos = (px - bb_lo) / (bb_up - bb_lo)
                bx_score += 0.10 if bb_pos > 0.55 else -0.10 if bb_pos < 0.45 else 0.0

            rally = cr > float(candidate["crypto_rally_daily_gain"])

            # Weekly EMA11 red gate for shorts
            weekly = close_vals.iloc[: i + 1].resample("W-FRI").last().dropna()
            weekly_red = False
            if len(weekly) >= 15:
                wema = _ema_series(weekly.to_numpy(), 11)
                weekly_red = float(wema[-1]) < float(wema[-2])

            long_ok = lag > float(candidate["pricing_lag_threshold_pct"]) and beta > float(candidate["beta_min"]) and bx_score > -0.15
            short_ok = lag < -float(candidate["pricing_lag_threshold_pct"]) and weekly_red and bx_score < 0.15 and not rally

            next_ret = float(s_rets.iloc[i + 1])
            if long_ok:
                all_returns.append(next_ret)
            elif short_ok:
                all_returns.append(-next_ret)

    if not all_returns:
        return np.array([])
    return np.array(all_returns, dtype=float)


def _classify_regime(np_module, returns_arr):
    if len(returns_arr) < 20:
        return "unknown"
    mean_r = float(np_module.mean(returns_arr))
    vol = float(np_module.std(returns_arr))
    trend = "bull" if mean_r > 0.001 else "bear" if mean_r < -0.001 else "sideways"
    vol_tag = "high_vol" if vol > 0.03 else "low_vol"
    return f"{trend}_{vol_tag}"


def _build_joined_map(symbols: list[str], years: int = 6):
    _get_market_deps()
    joined_map = {}
    for symbol in symbols:
        symbol = symbol.upper().strip()
        crypto = UNDERLYING_MAP.get(symbol, "BTC-USD")
        s_hist = _fetch_market_history(symbol, period=f"{years}y", interval="1d")
        c_hist = _fetch_market_history(crypto, period=f"{years}y", interval="1d")
        if s_hist.empty or c_hist.empty:
            continue
        joined = (
            s_hist["Close"].dropna().to_frame("stock")
            .join(c_hist["Close"].dropna().to_frame("crypto"), how="inner")
            .dropna()
        )
        if len(joined) < 400:
            continue
        joined_map[symbol] = joined
    return joined_map


def _simulate_candidate_on_window(candidate: dict, joined_map: dict, start_idx: int, end_idx: int):
    np, _, _, _ = _get_market_deps()
    rets = []
    for _, joined in joined_map.items():
        s_rets = joined["stock"].pct_change().dropna()
        c_rets = joined["crypto"].pct_change().dropna()
        close_vals = joined["stock"]
        local_end = min(end_idx, len(s_rets) - 1)
        local_start = max(40, start_idx)
        for i in range(local_start, local_end):
            if i + 1 >= len(s_rets):
                break
            sr = float(s_rets.iloc[i])
            cr = float(c_rets.iloc[i])
            win_s = s_rets.iloc[max(0, i - 20) : i]
            win_c = c_rets.iloc[max(0, i - 20) : i]
            if len(win_s) < 10 or len(win_c) < 10:
                continue
            beta = float((win_s.cov(win_c)) / (win_c.var() + 1e-9))
            expected = beta * cr
            lag = expected - sr

            tail_close = close_vals.iloc[: i + 1].tail(60)
            if len(tail_close) < 30:
                continue
            sma9 = float(tail_close.tail(9).mean())
            ema11_series = _ema_series(tail_close.to_numpy(), 11)
            ema11 = float(ema11_series[-1])
            ema11_prev = float(ema11_series[-2]) if len(ema11_series) > 1 else ema11
            bb_mid = float(tail_close.tail(20).mean())
            bb_std = float(tail_close.tail(20).std(ddof=1))
            bb_up = bb_mid + 2.0 * bb_std
            bb_lo = bb_mid - 2.0 * bb_std
            px = float(tail_close.iloc[-1])
            bx_score = (0.4 if px > sma9 else -0.4) + (0.35 if px > ema11 else -0.35) + (0.15 if ema11 > ema11_prev else -0.15)
            if bb_up > bb_lo:
                bb_pos = (px - bb_lo) / (bb_up - bb_lo)
                bx_score += 0.10 if bb_pos > 0.55 else -0.10 if bb_pos < 0.45 else 0.0

            rally = cr > float(candidate["crypto_rally_daily_gain"])
            weekly = close_vals.iloc[: i + 1].resample("W-FRI").last().dropna()
            weekly_red = False
            if len(weekly) >= 15:
                wema = _ema_series(weekly.to_numpy(), 11)
                weekly_red = float(wema[-1]) < float(wema[-2])

            long_ok = lag > float(candidate["pricing_lag_threshold_pct"]) and beta > float(candidate["beta_min"]) and bx_score > -0.15
            short_ok = lag < -float(candidate["pricing_lag_threshold_pct"]) and weekly_red and bx_score < 0.15 and not rally

            next_ret = float(s_rets.iloc[i + 1])
            if long_ok:
                rets.append(next_ret)
            elif short_ok:
                rets.append(-next_ret)
    if not rets:
        return np.array([])
    return np.array(rets, dtype=float)


def optimize_strategy_parameters_tool(
    iterations: int = 40,
    lookback_days: int = 220,
    watchlist: str = "",
    promote_if_improved: bool = True,
    min_score_improvement: float = 0.05,
) -> str:
    """
    Optimize key strategy parameters using a simple train/validation backtest loop.
    """
    np, _, _, _ = _get_market_deps()
    symbols = [s.strip().upper() for s in (watchlist or ",".join(STRATEGY_CONFIG["watchlist"])).split(",") if s.strip()]
    iterations = max(5, min(int(iterations), 300))

    base = _candidate_from_config()
    base_returns = _simulate_candidate_returns(base, symbols, lookback_days=lookback_days)
    split = int(len(base_returns) * 0.7) if len(base_returns) > 10 else len(base_returns)
    base_train = _metrics_from_returns(np, base_returns[:split]) if split > 0 else _metrics_from_returns(np, base_returns)
    base_val = _metrics_from_returns(np, base_returns[split:]) if split < len(base_returns) else base_train

    best = {"candidate": dict(base), "train": base_train, "val": base_val}
    tested = []

    for _ in range(iterations):
        cand = _mutate_candidate(base)
        ret = _simulate_candidate_returns(cand, symbols, lookback_days=lookback_days)
        split = int(len(ret) * 0.7) if len(ret) > 10 else len(ret)
        train_m = _metrics_from_returns(np, ret[:split]) if split > 0 else _metrics_from_returns(np, ret)
        val_m = _metrics_from_returns(np, ret[split:]) if split < len(ret) else train_m
        tested.append({"candidate": cand, "train_score": train_m["score"], "val_score": val_m["score"], "trades": val_m["trades"]})
        if val_m["score"] > best["val"]["score"] and train_m["score"] > -1.0:
            best = {"candidate": cand, "train": train_m, "val": val_m}

    improved = float(best["val"]["score"]) - float(base_val["score"])
    promoted = False
    if promote_if_improved and improved >= float(min_score_improvement):
        STRATEGY_CONFIG["pricing_lag_threshold_pct"] = round(float(best["candidate"]["pricing_lag_threshold_pct"]), 6)
        STRATEGY_CONFIG["beta_min"] = round(float(best["candidate"]["beta_min"]), 4)
        STRATEGY_CONFIG["risk_per_trade_var95"] = round(float(best["candidate"]["risk_per_trade_var95"]), 6)
        STRATEGY_CONFIG["crypto_rally_daily_gain"] = round(float(best["candidate"]["crypto_rally_daily_gain"]), 6)
        promoted = True

    return json.dumps(
        {
            "status": "ok",
            "symbols": symbols,
            "iterations": iterations,
            "base_candidate": base,
            "base_train_metrics": base_train,
            "base_val_metrics": base_val,
            "best_candidate": best["candidate"],
            "best_train_metrics": best["train"],
            "best_val_metrics": best["val"],
            "score_improvement": round(improved, 4),
            "promoted": promoted,
            "updated_config": STRATEGY_CONFIG,
            "top_trials": sorted(tested, key=lambda x: x["val_score"], reverse=True)[:10],
            "generated_at_utc": _now_iso(),
        },
        indent=2,
    )


def run_optimizer_loop_tool(
    cycles: int = 6,
    interval_seconds: int = 3600,
    iterations_per_cycle: int = 30,
    lookback_days: int = 220,
    promote_if_improved: bool = True,
    notify_discord: bool = True,
) -> str:
    """
    Repeatedly optimize strategy parameters over time and optionally notify Discord.
    """
    cycles = max(1, min(int(cycles), 200))
    interval_seconds = max(5, min(int(interval_seconds), 86400))
    logs = []

    for idx in range(cycles):
        result = json.loads(
            optimize_strategy_parameters_tool(
                iterations=iterations_per_cycle,
                lookback_days=lookback_days,
                promote_if_improved=promote_if_improved,
            )
        )
        logs.append(
            {
                "cycle": idx + 1,
                "timestamp_utc": _now_iso(),
                "score_improvement": result.get("score_improvement"),
                "promoted": result.get("promoted"),
                "best_candidate": result.get("best_candidate"),
            }
        )
        if notify_discord:
            _send_discord_message(
                f"[Optimizer cycle {idx + 1}/{cycles}] "
                f"improvement={result.get('score_improvement')} promoted={result.get('promoted')} "
                f"candidate={result.get('best_candidate')}"
            )
        if idx < cycles - 1:
            time.sleep(interval_seconds)

    return json.dumps(
        {
            "status": "ok",
            "cycles": cycles,
            "interval_seconds": interval_seconds,
            "logs": logs,
            "final_config": STRATEGY_CONFIG,
            "completed_at_utc": _now_iso(),
        },
        indent=2,
    )


def run_long_backtest_protocol_tool(
    years: int = 6,
    train_days: int = 252,
    test_days: int = 63,
    step_days: int = 63,
    iterations_per_window: int = 25,
    watchlist: str = "",
    auto_adjust: bool = True,
    min_validation_improvement: float = 0.05,
    notify_discord: bool = True,
) -> str:
    """
    Long walk-forward backtest protocol:
    - analyzes historical trend regimes
    - optimizes parameters on each training window
    - validates on forward test window
    - optionally adjusts live config if aggregate validation improves
    """
    np, _, _, _ = _get_market_deps()
    symbols = [s.strip().upper() for s in (watchlist or ",".join(STRATEGY_CONFIG["watchlist"])).split(",") if s.strip()]
    years = max(2, min(int(years), 12))
    train_days = max(126, min(int(train_days), 756))
    test_days = max(21, min(int(test_days), 252))
    step_days = max(10, min(int(step_days), test_days))
    iterations_per_window = max(5, min(int(iterations_per_window), 120))

    joined_map = _build_joined_map(symbols, years=years)
    if not joined_map:
        return json.dumps({"error": "No sufficient historical data for selected watchlist."}, indent=2)

    min_len = min(len(j) for j in joined_map.values())
    start_anchor = max(80, train_days)
    window_logs = []
    candidate_pool = []

    for anchor in range(start_anchor, min_len - test_days - 1, step_days):
        train_start = max(40, anchor - train_days)
        train_end = anchor
        test_start = anchor
        test_end = anchor + test_days

        base = _candidate_from_config()
        base_train_rets = _simulate_candidate_on_window(base, joined_map, train_start, train_end)
        base_test_rets = _simulate_candidate_on_window(base, joined_map, test_start, test_end)
        base_train = _metrics_from_returns(np, base_train_rets)
        base_test = _metrics_from_returns(np, base_test_rets)

        best = {"candidate": dict(base), "train": base_train, "test": base_test}
        for _ in range(iterations_per_window):
            cand = _mutate_candidate(base)
            train_rets = _simulate_candidate_on_window(cand, joined_map, train_start, train_end)
            test_rets = _simulate_candidate_on_window(cand, joined_map, test_start, test_end)
            train_m = _metrics_from_returns(np, train_rets)
            test_m = _metrics_from_returns(np, test_rets)
            if train_m["score"] > -1.0 and test_m["score"] > best["test"]["score"]:
                best = {"candidate": cand, "train": train_m, "test": test_m}

        btc_key = next((k for k in joined_map.keys() if UNDERLYING_MAP.get(k, "BTC-USD") == "BTC-USD"), None)
        regime = "unknown"
        if btc_key:
            btc_test = joined_map[btc_key]["crypto"].pct_change().dropna().iloc[test_start:test_end]
            regime = _classify_regime(np, btc_test.to_numpy()) if len(btc_test) > 0 else "unknown"

        improvement = float(best["test"]["score"]) - float(base_test["score"])
        entry = {
            "anchor_index": anchor,
            "train_range": [train_start, train_end],
            "test_range": [test_start, test_end],
            "regime": regime,
            "base_test": base_test,
            "best_test": best["test"],
            "best_candidate": best["candidate"],
            "test_score_improvement": round(improvement, 4),
        }
        window_logs.append(entry)
        candidate_pool.append((best["candidate"], improvement, best["test"]["score"]))

    if not window_logs:
        return json.dumps({"error": "Not enough data length for protocol windows."}, indent=2)

    avg_improvement = float(sum(w["test_score_improvement"] for w in window_logs) / len(window_logs))
    valid_improvements = [w["test_score_improvement"] for w in window_logs if w["test_score_improvement"] > 0]
    win_rate = float(len(valid_improvements) / len(window_logs))

    promoted = False
    selected = _candidate_from_config()
    if candidate_pool:
        candidate_pool.sort(key=lambda x: x[2], reverse=True)
        top = [x[0] for x in candidate_pool[: max(1, min(5, len(candidate_pool)))]]
        selected = {
            "pricing_lag_threshold_pct": round(float(sum(c["pricing_lag_threshold_pct"] for c in top) / len(top)), 6),
            "beta_min": round(float(sum(c["beta_min"] for c in top) / len(top)), 4),
            "risk_per_trade_var95": round(float(sum(c["risk_per_trade_var95"] for c in top) / len(top)), 6),
            "crypto_rally_daily_gain": round(float(sum(c["crypto_rally_daily_gain"] for c in top) / len(top)), 6),
        }

    if auto_adjust and avg_improvement >= float(min_validation_improvement) and win_rate >= 0.55:
        STRATEGY_CONFIG["pricing_lag_threshold_pct"] = selected["pricing_lag_threshold_pct"]
        STRATEGY_CONFIG["beta_min"] = selected["beta_min"]
        STRATEGY_CONFIG["risk_per_trade_var95"] = selected["risk_per_trade_var95"]
        STRATEGY_CONFIG["crypto_rally_daily_gain"] = selected["crypto_rally_daily_gain"]
        promoted = True

    history = []
    if BACKTEST_PROTOCOL_FILE.exists():
        try:
            history = json.loads(BACKTEST_PROTOCOL_FILE.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    session_summary = {
        "timestamp_utc": _now_iso(),
        "years": years,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "iterations_per_window": iterations_per_window,
        "avg_validation_improvement": round(avg_improvement, 4),
        "validation_win_rate": round(win_rate, 4),
        "promoted": promoted,
        "selected_candidate": selected,
    }
    history.append(session_summary)
    BACKTEST_PROTOCOL_FILE.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")

    if notify_discord:
        _send_discord_message(
            f"[Backtest protocol] windows={len(window_logs)} avg_improvement={round(avg_improvement,4)} "
            f"win_rate={round(win_rate,4)} promoted={promoted} candidate={selected}"
        )

    return json.dumps(
        {
            "status": "ok",
            "protocol": "walk_forward_long_backtest",
            "symbols": list(joined_map.keys()),
            "windows": len(window_logs),
            "avg_validation_improvement": round(avg_improvement, 4),
            "validation_win_rate": round(win_rate, 4),
            "promoted": promoted,
            "selected_candidate": selected,
            "updated_config": STRATEGY_CONFIG,
            "window_logs": window_logs[-30:],
            "history_file": str(BACKTEST_PROTOCOL_FILE),
            "completed_at_utc": _now_iso(),
        },
        indent=2,
    )


def upsert_treasury_holding_tool(
    symbol: str,
    units: float,
    shares_outstanding: float,
    asset: str = "BTC",
) -> str:
    """Update treasury holdings inputs for NAV calculations."""
    symbol = symbol.upper().strip()
    payload = _load_treasury_holdings()
    payload[symbol] = _normalize_holding_record(symbol, {
        "units": float(units),
        "shares_outstanding": float(shares_outstanding),
        "asset": asset.upper().strip(),
        "updated_at_utc": _now_iso(),
        "source": "manual",
        "as_of_date": datetime.now(timezone.utc).date().isoformat(),
        "confidence": 0.95,
    }, {str(row.get("symbol", "")).upper(): row for row in _load_treasury_universe()})
    _save_treasury_holdings(payload)
    return json.dumps({"status": "ok", "symbol": symbol, "record": payload[symbol]}, indent=2)


def seed_crypto_treasury_universe_tool() -> str:
    """Initialize the tracked crypto treasury universe."""
    _save_treasury_universe(list(DEFAULT_TREASURY_UNIVERSE))
    return json.dumps(
        {
            "status": "ok",
            "count": len(DEFAULT_TREASURY_UNIVERSE),
            "file": str(TREASURY_UNIVERSE_FILE),
        },
        indent=2,
    )


def upsert_crypto_treasury_company_tool(
    symbol: str,
    asset: str = "BTC",
    units: float | None = None,
    market_cap_hint_usd: float | None = None,
) -> str:
    """Add or update one company row in the treasury universe file."""
    symbol = symbol.upper().strip()
    rows = _load_treasury_universe()
    record = {
        "symbol": symbol,
        "asset": asset.upper().strip(),
        "units": None if units is None else float(units),
        "market_cap_hint_usd": None if market_cap_hint_usd is None else float(market_cap_hint_usd),
        "updated_at_utc": _now_iso(),
    }
    replaced = False
    for idx, row in enumerate(rows):
        if str(row.get("symbol", "")).upper() == symbol:
            rows[idx] = {**row, **record}
            replaced = True
            break
    if not replaced:
        rows.append(record)
    _save_treasury_universe(rows)
    return json.dumps({"status": "ok", "symbol": symbol, "replaced": replaced, "record": record}, indent=2)


def get_crypto_treasury_watchlist_tool(
    min_holdings_usd: float = 100_000_000,
    min_avg_dollar_volume: float = 5_000_000,
    limit: int = 20,
) -> str:
    """
    Build a liquidity-focused watchlist from tracked treasury companies.
    """
    _get_market_deps()
    rows = _load_treasury_universe()
    enriched = []
    for row in rows:
        symbol = str(row.get("symbol", "")).upper().strip()
        if not symbol:
            continue
        try:
            hist = _fetch_market_history(symbol, period="3mo", interval="1d")
        except Exception:
            continue
        if hist.empty:
            continue
        finnhub_profile = {}
        try:
            finnhub_profile = _finnhub_company_profile2(symbol)
        except Exception:
            finnhub_profile = {}
        resolved = _resolve_treasury_inputs(
            symbol,
            info={},
            yf_module=None,
            estimated_units_mode=False,
            finnhub_profile=finnhub_profile,
        )
        units = resolved.get("units")
        asset = resolved.get("asset")
        if units is None:
            continue
        crypto_ticker = f"{asset}-USD"
        try:
            crypto_px = _latest_close(None, crypto_ticker)
        except Exception:
            continue
        close = _safe_float(hist["Close"].dropna().iloc[-1], 0.0)
        avg_volume = _safe_float(hist["Volume"].tail(20).mean(), 0.0)
        avg_dollar_volume = close * avg_volume
        holdings_usd = float(units) * float(crypto_px)
        if holdings_usd < float(min_holdings_usd) or avg_dollar_volume < float(min_avg_dollar_volume):
            continue
        enriched.append(
            {
                "symbol": symbol,
                "asset": asset,
                "units": float(units),
                "holdings_usd": round(holdings_usd, 2),
                "avg_dollar_volume_20d": round(avg_dollar_volume, 2),
                "shares_outstanding": resolved.get("shares_outstanding"),
                "staleness_days": resolved.get("staleness_days"),
                "confidence": resolved.get("confidence"),
            }
        )

    enriched.sort(key=lambda x: (x["holdings_usd"], x["avg_dollar_volume_20d"]), reverse=True)
    selected = enriched[: max(1, int(limit))]
    fallback_selected = []
    if not selected:
        for row in rows:
            symbol = str(row.get("symbol", "")).upper().strip()
            if not symbol:
                continue
            resolved = _resolve_treasury_inputs(symbol, info={}, yf_module=None, estimated_units_mode=False)
            if resolved.get("units") is None:
                continue
            fallback_selected.append(
                {
                    "symbol": symbol,
                    "asset": resolved.get("asset"),
                    "units": resolved.get("units"),
                    "holdings_usd": None,
                    "avg_dollar_volume_20d": None,
                    "shares_outstanding": resolved.get("shares_outstanding"),
                    "staleness_days": resolved.get("staleness_days"),
                    "confidence": resolved.get("confidence"),
                    "selection_mode": "fallback_units_only",
                }
            )
        selected = fallback_selected[: max(1, int(limit))]
    return json.dumps(
        {
            "min_holdings_usd": min_holdings_usd,
            "min_avg_dollar_volume": min_avg_dollar_volume,
            "count": len(selected),
            "watchlist": selected,
            "selection_mode": "liquidity_filtered" if enriched else "fallback_units_only",
            "generated_at_utc": _now_iso(),
        },
        indent=2,
    )


def _active_watchlist_symbols(watchlist: str = "") -> list[str]:
    if watchlist.strip():
        return [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    payload = json.loads(
        get_crypto_treasury_watchlist_tool(
            min_holdings_usd=float(STRATEGY_CONFIG.get("live_watchlist_min_holdings_usd", 100_000_000)),
            min_avg_dollar_volume=float(STRATEGY_CONFIG.get("live_watchlist_min_avg_dollar_volume", 5_000_000)),
            limit=int(STRATEGY_CONFIG.get("live_watchlist_limit", 12)),
        )
    )
    return [str(row.get("symbol", "")).upper() for row in payload.get("watchlist", []) if row.get("symbol")]


def refresh_treasury_holdings_tool(
    watchlist: str = "",
    use_reference_pages: bool = True,
    missing_only: bool = False,
) -> str:
    """Refresh canonical treasury holdings using existing records, universe units, Yahoo shares, and optional references."""
    return json.dumps(
        refresh_treasury_holdings(
            watchlist=watchlist,
            use_reference_pages=use_reference_pages,
            missing_only=missing_only,
        ),
        indent=2,
    )


def fetch_treasury_reference_pages_tool() -> str:
    """
    Fetch treasury reference pages for manual/LLM-assisted holdings updates.
    """
    results = fetch_treasury_data(
        urls=[
            "https://bitcointreasuries.net/",
            "https://www.theblock.co/data/crypto-markets/bitcoin-etf/bitcoin-treasuries",
            "https://coinmarketcap.com/view/bitcoin-treasuries/",
            "https://coinmarketcap.com/view/ethereum-treasuries/",
        ]
    )
    return json.dumps({"fetched_at_utc": _now_iso(), "results": results}, indent=2)


def compute_mnav_snapshot_tool(
    watchlist: str = "",
    min_volume: int = 500000,
    estimated_units_mode: bool | None = None,
) -> str:
    """
    Compute mNAV snapshot and peer deviations for tracked treasury companies.
    mNAV = market_cap / NAV, where NAV ~= holdings + cash - debt.
    """
    _get_market_deps()
    universe = _load_treasury_universe()
    estimated_units_mode = (
        bool(estimated_units_mode)
        if estimated_units_mode is not None
        else bool(STRATEGY_CONFIG.get("estimated_units_mode", False))
    )

    if watchlist.strip():
        symbols = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    else:
        symbols = _active_watchlist_symbols()

    rows = []
    warnings_list = []
    for symbol in symbols:
        try:
            hist = _fetch_market_history(symbol, period="3mo", interval="1d")
        except Exception as exc:
            rows.append({"symbol": symbol, "error": f"price history unavailable: {exc}"})
            continue
        if hist.empty:
            rows.append({"symbol": symbol, "error": "no price history"})
            continue

        close = _safe_float(hist["Close"].dropna().iloc[-1], 0.0)
        avg_volume = _safe_float(hist["Volume"].tail(20).mean(), 0.0)
        returns = hist["Close"].pct_change().dropna()
        vol_7d = _safe_float(returns.tail(7).std() * sqrt(252), 0.0)

        info = {}
        market_cap = 0.0
        cash = 0.0
        debt = 0.0

        finnhub_profile = {}
        try:
            finnhub_profile = _finnhub_company_profile2(symbol)
        except Exception:
            finnhub_profile = {}
        resolved = _resolve_treasury_inputs(
            symbol,
            info=info,
            yf_module=None,
            estimated_units_mode=estimated_units_mode,
            finnhub_profile=finnhub_profile,
        )
        units = resolved.get("units")
        shares = _safe_float(resolved.get("shares_outstanding"), 0.0)
        asset = resolved.get("asset")
        units_source = resolved.get("units_source") or "unknown"
        warnings_list.extend(resolved.get("warnings", []))
        if units is None:
            rows.append({"symbol": symbol, "error": "missing treasury units (update holdings/universe)"})
            warnings_list.append(f"{symbol}: skipped mNAV because treasury units are missing.")
            continue

        crypto_symbol = f"{str(asset).upper()}-USD"
        try:
            crypto_px = _latest_close(None, crypto_symbol)
        except Exception:
            rows.append({"symbol": symbol, "error": f"missing crypto price for {crypto_symbol}"})
            continue

        holdings_value = float(units) * float(crypto_px)
        nav = holdings_value + cash - debt
        nav_per_share = nav / shares if shares > 0 else 0.0

        if market_cap <= 0 and shares > 0 and close > 0:
            market_cap = shares * close

        if nav <= 0 or market_cap <= 0:
            rows.append({"symbol": symbol, "error": "insufficient NAV/market cap inputs"})
            continue

        mnav = market_cap / nav
        sector_filter = avg_volume >= float(min_volume)

        # Approximate 30d mean mNAV using price history with fixed share/nav assumptions.
        mnav_30d = None
        try:
            if shares > 0 and nav > 0:
                market_caps_hist = hist["Close"].tail(30).dropna() * shares
                mnav_30d = float(market_caps_hist.mean() / nav)
        except Exception:
            mnav_30d = None

        rows.append(
            {
                "symbol": symbol,
                "asset": str(asset).upper(),
                "units": float(units),
                "units_source": units_source,
                "shares_outstanding": None if shares <= 0 else float(shares),
                "shares_source": resolved.get("shares_source"),
                "staleness_days": resolved.get("staleness_days"),
                "confidence": resolved.get("confidence"),
                "price": round(close, 4),
                "crypto_symbol": crypto_symbol,
                "crypto_price": round(float(crypto_px), 4),
                "holdings_value_usd": round(holdings_value, 2),
                "cash_usd": round(cash, 2),
                "debt_usd": round(debt, 2),
                "nav_usd": round(nav, 2),
                "nav_per_share": round(nav_per_share, 4),
                "market_cap_usd": round(market_cap, 2),
                "mnav": round(mnav, 4),
                "mnav_30d_avg": None if mnav_30d is None else round(mnav_30d, 4),
                "avg_volume_20d": round(avg_volume, 0),
                "volatility_7d_ann": round(vol_7d, 4),
                "liquidity_ok": sector_filter,
            }
        )

    valid = [r for r in rows if "mnav" in r]
    sector_avg = sum(r["mnav"] for r in valid) / len(valid) if valid else None
    for row in valid:
        row["deviation_from_sector_avg"] = (
            None if sector_avg in (None, 0) else round((row["mnav"] / sector_avg) - 1.0, 4)
        )
        base_30d = row.get("mnav_30d_avg")
        row["deviation_from_30d_avg"] = (
            None if not base_30d or base_30d == 0 else round((row["mnav"] / base_30d) - 1.0, 4)
        )

    return json.dumps(
        {
            "generated_at_utc": _now_iso(),
            "sector_avg_mnav": None if sector_avg is None else round(sector_avg, 4),
            "warnings": warnings_list,
            "estimated_units_mode": estimated_units_mode,
            "rows": rows,
        },
        indent=2,
    )


def detect_mnav_arbitrage_tool(
    watchlist: str = "",
    deviation_threshold: float = 0.15,
    min_volume: int = 500000,
    max_vol_7d: float = 0.50,
    discount_mnav: float = 0.8,
    premium_mnav: float = 2.0,
) -> str:
    """
    Detect mNAV inefficiencies and suggest long/short or relative-value trades.
    """
    snapshot = json.loads(compute_mnav_snapshot_tool(watchlist=watchlist, min_volume=min_volume))
    rows = [r for r in snapshot.get("rows", []) if isinstance(r, dict) and "mnav" in r]

    candidates = []
    for row in rows:
        if not row.get("liquidity_ok"):
            continue
        if _safe_float(row.get("volatility_7d_ann"), 0.0) > float(max_vol_7d):
            continue

        dev_sector = abs(_safe_float(row.get("deviation_from_sector_avg"), 0.0))
        dev_30d = abs(_safe_float(row.get("deviation_from_30d_avg"), 0.0))
        if dev_sector < float(deviation_threshold) and dev_30d < float(deviation_threshold):
            continue
        candidates.append(row)

    discounts = [r for r in candidates if _safe_float(r.get("mnav"), 9.9) <= float(discount_mnav)]
    premiums = [r for r in candidates if _safe_float(r.get("mnav"), 0.0) >= float(premium_mnav)]
    discounts.sort(key=lambda x: x.get("mnav", 99))
    premiums.sort(key=lambda x: x.get("mnav", -1), reverse=True)

    pairs = []
    for long_row, short_row in zip(discounts[:5], premiums[:5]):
        pairs.append(
            {
                "long_symbol": long_row["symbol"],
                "short_symbol": short_row["symbol"],
                "long_mnav": long_row["mnav"],
                "short_mnav": short_row["mnav"],
                "thesis": "mean-reversion of relative mNAV spread",
            }
        )

    return json.dumps(
        {
            "generated_at_utc": _now_iso(),
            "filters": {
                "deviation_threshold": deviation_threshold,
                "min_volume": min_volume,
                "max_vol_7d": max_vol_7d,
                "discount_mnav": discount_mnav,
                "premium_mnav": premium_mnav,
            },
            "sector_avg_mnav": snapshot.get("sector_avg_mnav"),
            "discount_candidates": discounts,
            "premium_candidates": premiums,
            "pairs": pairs,
        },
        indent=2,
    )


def execute_mnav_pairs_trade_tool(
    long_symbol: str,
    short_symbol: str,
    dollars_per_leg: float = 5000,
) -> str:
    """
    Execute a simple dollar-neutral pair in Alpaca paper account.
    """
    _, _, _, _ = _get_market_deps()
    long_symbol = long_symbol.upper().strip()
    short_symbol = short_symbol.upper().strip()
    dollars_per_leg = max(100.0, float(dollars_per_leg))

    long_px = _latest_close(None, long_symbol)
    short_px = _latest_close(None, short_symbol)
    long_qty = max(1, int(dollars_per_leg // max(long_px, 0.01)))
    short_qty = max(1, int(dollars_per_leg // max(short_px, 0.01)))

    long_order = json.loads(broker_submit_order_tool(long_symbol, "buy", long_qty))
    short_order = json.loads(broker_submit_order_tool(short_symbol, "sell", short_qty))

    return json.dumps(
        {
            "long_leg": {
                "symbol": long_symbol,
                "price": round(long_px, 4),
                "qty": long_qty,
                "order": long_order,
            },
            "short_leg": {
                "symbol": short_symbol,
                "price": round(short_px, 4),
                "qty": short_qty,
                "order": short_order,
            },
            "dollars_per_leg": dollars_per_leg,
            "generated_at_utc": _now_iso(),
        },
        indent=2,
    )


def _evaluate_symbol_signal(symbol: str, risk_budget_pct: float = 0.025) -> dict:
    np, _, _, LinearRegression = _get_market_deps()
    symbol = symbol.upper()
    crypto = UNDERLYING_MAP.get(symbol, "BTC-USD")

    try:
        s_hist = _fetch_market_history(symbol, period="6mo", interval="1d")
        s_hist_weekly = _fetch_market_history(symbol, period="2y", interval="1wk")
        c_hist = _fetch_market_history(crypto, period="6mo", interval="1d")
    except Exception as exc:
        return {"symbol": symbol, "error": f"market data unavailable: {exc}"}
    if s_hist.empty or c_hist.empty or s_hist_weekly.empty:
        return {"symbol": symbol, "error": "missing market data"}

    s_close = s_hist["Close"].dropna()
    c_close = c_hist["Close"].dropna()
    joined = s_close.to_frame("stock").join(c_close.to_frame("crypto"), how="inner")
    if len(joined) < 80:
        return {"symbol": symbol, "error": "not enough aligned history"}

    s_rets = joined["stock"].pct_change().dropna()
    c_rets = joined["crypto"].pct_change().dropna()
    beta = float((s_rets.cov(c_rets)) / (c_rets.var() + 1e-9))
    crypto_daily_gain = float(c_rets.iloc[-1])

    # Linear regression convergence model: stock return explained by crypto return.
    x = c_rets.to_numpy().reshape(-1, 1)
    y = s_rets.to_numpy()
    model = LinearRegression()
    model.fit(x, y)
    predicted = float(model.predict([[float(c_rets.iloc[-1])]])[0])
    lag = predicted - float(s_rets.iloc[-1])

    hv30 = _realized_vol_30d(s_rets, np)
    iv = _atm_implied_vol(None)
    iv_trigger = bool(iv is not None and iv > STRATEGY_CONFIG["options_iv_threshold"] and (iv - hv30) > STRATEGY_CONFIG["options_iv_over_hv_gap"])

    macd_ok = _macd_signal(c_close.tail(120), np)
    rally = crypto_daily_gain > STRATEGY_CONFIG["crypto_rally_daily_gain"] and macd_ok
    bx = _bx_trender_metrics(s_hist, np)
    weekly_ema11_red = _weekly_ema11_red(s_hist_weekly)

    finnhub_profile = {}
    try:
        finnhub_profile = _finnhub_company_profile2(symbol)
    except Exception:
        finnhub_profile = {}
    resolved = _resolve_treasury_inputs(
        symbol,
        info={},
        yf_module=None,
        estimated_units_mode=bool(STRATEGY_CONFIG.get("estimated_units_mode", False)),
        finnhub_profile=finnhub_profile,
    )
    nav_gap = None
    shares_outstanding = resolved.get("shares_outstanding")
    units = resolved.get("units")
    if shares_outstanding and units:
        px = float(joined["crypto"].iloc[-1])
        nav_per_share = (float(units) * px) / float(shares_outstanding)
        nav_gap = (float(joined["stock"].iloc[-1]) - nav_per_share) / max(nav_per_share, 1e-9)

    long_trigger = (lag > STRATEGY_CONFIG["pricing_lag_threshold_pct"]) or (nav_gap is not None and nav_gap < -STRATEGY_CONFIG["nav_threshold_pct"])
    short_trigger = (lag < -STRATEGY_CONFIG["pricing_lag_threshold_pct"]) or (nav_gap is not None and nav_gap > STRATEGY_CONFIG["nav_threshold_pct"])

    signal = "HOLD"
    bx_score = _safe_float(bx.get("bx_score"), 0.0)
    if long_trigger and beta > STRATEGY_CONFIG["beta_min"] and (rally or nav_gap is not None) and bx_score > -0.15:
        signal = "BUY"
    # Per user rule: only allow shorts when weekly EMA11 is red.
    if short_trigger and not rally and weekly_ema11_red and bx_score < 0.15:
        signal = "SELL"

    var95 = _estimate_var95(s_rets.tail(120), np)
    account = json.loads(broker_paper_account_status_tool())
    equity = float(account.get("equity") or 100000.0)
    max_risk = 0.05 if STRATEGY_CONFIG.get("high_beta_mode") else 0.04
    base_risk_budget = equity * float(min(max(risk_budget_pct, 0.015), max_risk))
    bx_size_multiplier = max(0.70, min(1.30, 1.0 + (bx_score * 0.25)))
    risk_budget = base_risk_budget * bx_size_multiplier
    last_px = float(joined["stock"].iloc[-1])
    stop_pct = 0.05 if (iv is not None and iv > 1.5) else 0.12
    per_share_risk = max(last_px * stop_pct, last_px * max(var95, 0.01))
    qty = int(risk_budget // max(per_share_risk, 0.01))

    return {
        "symbol": symbol,
        "crypto_underlying": crypto,
        "signal": signal,
        "recommended_quantity": max(0, qty),
        "beta": round(beta, 4),
        "pricing_lag": round(lag, 4),
        "crypto_daily_gain": round(crypto_daily_gain, 4),
        "macd_crossover": macd_ok,
        "rally_condition": rally,
        "nav_gap": None if nav_gap is None else round(nav_gap, 4),
        "nav_inputs": {
            "units": units,
            "units_source": resolved.get("units_source"),
            "shares_outstanding": shares_outstanding,
            "shares_source": resolved.get("shares_source"),
            "staleness_days": resolved.get("staleness_days"),
            "confidence": resolved.get("confidence"),
        },
        "iv": None if iv is None else round(iv, 4),
        "hv30": round(hv30, 4),
        "iv_trigger": iv_trigger,
        "bx_trender": bx,
        "weekly_ema11_red": weekly_ema11_red,
        "var95_1d": round(var95, 4),
        "base_risk_budget_usd": round(base_risk_budget, 2),
        "bx_size_multiplier": round(bx_size_multiplier, 4),
        "risk_budget_usd": round(risk_budget, 2),
        "last_price": round(last_px, 4),
        "stop_pct": round(stop_pct, 4),
    }


def run_crypto_treasury_strategy_tool(
    watchlist: str = "",
    risk_budget_pct: float = 0.025,
    execute_orders: bool = False,
    max_new_trades: int = 2,
) -> str:
    """
    Scan crypto treasury equities and optionally execute Alpaca paper orders.
    """
    symbols = _active_watchlist_symbols(watchlist)
    if not symbols:
        return json.dumps(
            {
                "strategy": "crypto_treasury_arb_momentum",
                "error": "no tradable symbols available after live watchlist filtering",
                "watchlist": symbols,
                "generated_at_utc": _now_iso(),
            },
            indent=2,
        )
    max_new_trades = max(0, int(max_new_trades))
    decisions = []
    orders = []
    buys = 0
    threshold = float(STRATEGY_CONFIG.get("deviation_threshold", 0.15))
    mnav_snapshot = json.loads(
        compute_mnav_snapshot_tool(
            watchlist=",".join(symbols),
            estimated_units_mode=bool(STRATEGY_CONFIG.get("estimated_units_mode", False)),
        )
    )
    mnav_map = {
        str(row.get("symbol", "")).upper(): row
        for row in mnav_snapshot.get("rows", [])
        if isinstance(row, dict) and row.get("symbol")
    }

    for symbol in symbols:
        try:
            snap = _evaluate_symbol_signal(symbol, risk_budget_pct=risk_budget_pct)
        except Exception as exc:
            decisions.append({"symbol": symbol, "error": str(exc)})
            continue

        decision = {"symbol": symbol, "analysis": snap}
        signal = snap.get("signal", "HOLD")
        decision["raw_signal"] = signal
        qty = int(snap.get("recommended_quantity", 0))
        mnav_row = mnav_map.get(symbol, {})
        decision["mnav_context"] = mnav_row

        if mnav_row.get("error"):
            decision["mnav_gate"] = "unavailable_non_blocking"
        elif "mnav" in mnav_row:
            dev_sector = _safe_float(mnav_row.get("deviation_from_sector_avg"), 0.0)
            dev_30d = _safe_float(mnav_row.get("deviation_from_30d_avg"), 0.0)
            supports_buy = dev_sector <= -threshold or dev_30d <= -threshold
            supports_sell = dev_sector >= threshold or dev_30d >= threshold
            if signal == "BUY" and not supports_buy:
                signal = "HOLD"
                decision["mnav_gate"] = "blocked_no_discount"
            elif signal == "SELL" and not supports_sell:
                signal = "HOLD"
                decision["mnav_gate"] = "blocked_no_premium"
            else:
                decision["mnav_gate"] = "passed"

        decision["effective_signal"] = signal

        if execute_orders and signal == "BUY" and qty > 0 and buys < max_new_trades:
            order = json.loads(broker_submit_order_tool(symbol, "buy", qty))
            decision["order"] = order
            orders.append(order)
            if not order.get("error"):
                buys += 1
        elif execute_orders and signal == "SELL" and qty > 0:
            order = json.loads(broker_submit_order_tool(symbol, "sell", qty))
            decision["order"] = order
            orders.append(order)

        decisions.append(decision)

    account = json.loads(broker_paper_account_status_tool())
    positions = json.loads(broker_positions_tool())

    payload = {
        "strategy": "crypto_treasury_arb_momentum",
        "config": STRATEGY_CONFIG,
        "data_quality": validate_data_quality(watchlist=",".join(symbols)),
        "mnav_snapshot": mnav_snapshot,
        "watchlist": symbols,
        "execute_orders": bool(execute_orders),
        "decisions": decisions,
        "orders": orders,
        "account": account,
        "positions": positions,
        "generated_at_utc": _now_iso(),
    }
    return json.dumps(payload, indent=2)


def place_auto_hedge_tool(stock_symbol: str, quantity: float) -> str:
    """
    Place a simple hedge order against the stock's mapped crypto proxy ETF.
    """
    stock_symbol = stock_symbol.upper().strip()
    quantity = float(quantity)
    if quantity <= 0:
        return json.dumps({"error": "quantity must be > 0"}, indent=2)

    crypto = UNDERLYING_MAP.get(stock_symbol, "BTC-USD")
    hedge_symbol = HEDGE_ETF_MAP.get(crypto, "BITI")
    hedge_ratio = 1.0
    hedge_qty = max(1.0, round(quantity * hedge_ratio, 2))

    order = json.loads(broker_submit_order_tool(hedge_symbol, "buy", hedge_qty))
    return json.dumps(
        {
            "stock_symbol": stock_symbol,
            "crypto_underlying": crypto,
            "hedge_symbol": hedge_symbol,
            "hedge_ratio": hedge_ratio,
            "hedge_quantity": hedge_qty,
            "hedge_order": order,
        },
        indent=2,
    )


def autonomous_broker_trading_session_tool(
    duration_minutes: int = 60,
    max_trades: int = 100,
    poll_seconds: int = 60,
    watchlist: str = "",
) -> str:
    """
    Run an autonomous Alpaca paper trading session:
    - scans configured watchlist
    - executes strategy decisions each cycle
    - stops at duration or max_trades
    """
    duration_minutes = max(1, min(int(duration_minutes), 240))
    max_trades = max(1, min(int(max_trades), 500))
    poll_seconds = max(15, min(int(poll_seconds), 900))

    if bool(STRATEGY_CONFIG.get("auto_refresh_holdings_on_startup", True)):
        try:
            refresh_treasury_holdings(watchlist=watchlist, use_reference_pages=True, missing_only=False)
        except Exception:
            pass

    symbols = _active_watchlist_symbols(watchlist)

    start_ts = time.time()
    end_ts = start_ts + (duration_minutes * 60)
    total_orders = 0
    cycles = 0
    cycle_logs = []
    relaxed_applied = False
    original_deviation_threshold = float(STRATEGY_CONFIG.get("deviation_threshold", 0.15))
    quality_report = validate_data_quality(watchlist=",".join(symbols))
    missing_action = str(STRATEGY_CONFIG.get("missing_holdings_action", "alert")).lower()

    if not symbols:
        return json.dumps(
            {
                "status": "aborted",
                "reason": "empty_live_watchlist",
                "data_quality": quality_report,
                "watchlist": symbols,
                "session_started_utc": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                "session_ended_utc": _now_iso(),
            },
            indent=2,
        )

    if quality_report.get("abort_recommended"):
        if missing_action == "alert":
            _send_discord_message(
                "Data quality gate blocked session start: "
                f"{quality_report.get('missing_units_count')}/{quality_report.get('checked_symbols')} "
                "symbols are missing treasury units."
            )
        return json.dumps(
            {
                "status": "aborted",
                "reason": "data_quality_gate",
                "data_quality": quality_report,
                "watchlist": symbols,
                "session_started_utc": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
                "session_ended_utc": _now_iso(),
            },
            indent=2,
        )

    while time.time() < end_ts and total_orders < max_trades:
        cycles += 1
        remaining_trades = max_trades - total_orders
        elapsed_minutes = (time.time() - start_ts) / 60.0

        if (
            not relaxed_applied
            and bool(STRATEGY_CONFIG.get("enable_auto_relax", True))
            and elapsed_minutes >= float(STRATEGY_CONFIG.get("adaptive_relaxation_timer_minutes", 30))
        ):
            STRATEGY_CONFIG["deviation_threshold"] = float(
                STRATEGY_CONFIG.get("relaxed_deviation_threshold", 0.10)
            )
            relaxed_applied = True

        is_open = True
        clock_error = None
        try:
            clock = _alpaca_request("GET", "/v2/clock")
            is_open = bool(clock.get("is_open", True))
        except Exception as exc:
            clock_error = str(exc)

        placed_this_cycle = 0
        cycle_error = None
        if is_open:
            try:
                result = json.loads(
                    run_crypto_treasury_strategy_tool(
                        watchlist=",".join(symbols),
                        risk_budget_pct=float(STRATEGY_CONFIG.get("risk_per_trade_var95", 0.025)),
                        execute_orders=True,
                        max_new_trades=min(remaining_trades, int(STRATEGY_CONFIG.get("max_new_trades_per_run", 4))),
                    )
                )
                for item in result.get("orders", []):
                    if isinstance(item, dict) and not item.get("error"):
                        order_data = item.get("order", {})
                        if order_data.get("id"):
                            placed_this_cycle += 1
            except Exception as exc:
                cycle_error = str(exc)

        total_orders += placed_this_cycle
        cycle_logs.append(
            {
                "cycle": cycles,
                "timestamp_utc": _now_iso(),
                "elapsed_minutes": round(elapsed_minutes, 2),
                "market_open": is_open,
                "placed_orders": placed_this_cycle,
                "total_orders": total_orders,
                "deviation_threshold": STRATEGY_CONFIG.get("deviation_threshold"),
                "adaptive_relaxation_applied": relaxed_applied,
                "clock_error": clock_error,
                "cycle_error": cycle_error,
            }
        )

        if time.time() >= end_ts or total_orders >= max_trades:
            break

        sleep_for = min(poll_seconds, max(1, int(end_ts - time.time())))
        time.sleep(sleep_for)

    account = json.loads(broker_paper_account_status_tool())
    positions = json.loads(broker_positions_tool())
    STRATEGY_CONFIG["deviation_threshold"] = original_deviation_threshold

    return json.dumps(
        {
            "status": "completed",
            "duration_minutes": duration_minutes,
            "max_trades": max_trades,
            "poll_seconds": poll_seconds,
            "watchlist": symbols,
            "data_quality": quality_report,
            "adaptive_relaxation_applied": relaxed_applied,
            "cycles_run": cycles,
            "total_orders_placed": total_orders,
            "session_started_utc": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "session_ended_utc": _now_iso(),
            "account": account,
            "positions": positions,
            "cycle_logs": cycle_logs[-30:],
        },
        indent=2,
    )
