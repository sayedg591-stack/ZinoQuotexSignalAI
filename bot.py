```python
import os
import json
import time
import asyncio
import logging
import threading
import math
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# ============================================================
# OPTIONAL GEMINI
# ============================================================

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

try:
    OWNER_ID = int(os.getenv("OWNER_ID", "0").strip())
except Exception:
    OWNER_ID = 0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

if not MT4_API_KEY:
    MT4_API_KEY = os.getenv(
        "ZINO_API_KEY",
        ""
    ).strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")


# ============================================================
# TRADING CONFIG
# ============================================================

TIMEFRAME = "M1"

MIN_CANDLES = 50
REQUIRED_TOTAL_CANDLES = MIN_CANDLES + 1

ENTRY_DELAY_SECONDS = 60

MIN_SCORE = 13
MAX_SCORE = 20

MIN_CONFIDENCE = 76
MAX_CONFIDENCE = 89

MIN_SCORE_GAP = 2

MIN_ADX = 20.0
STRONG_ADX = 25.0

MIN_CANDLE_RANGE_RATIO = 0.25
MAX_CANDLE_RANGE_RATIO = 2.8

MAX_DATA_AGE_SECONDS = 180

RECOVERY_LIMIT = 1
RECOVERY_WAIT_SECONDS = 60

# Same BASE pair protection.
SETUP_REPEAT_BLOCK_SECONDS = 300

# Minimum meaningful ATR movement before allowing
# an old pair/setup to return.
FRESH_SETUP_ATR_RATIO = 0.35


# ============================================================
# INDICATOR SETTINGS
# ============================================================

EMA_FAST = 9
EMA_SLOW = 21

RSI_PERIOD = 14

ADX_PERIOD = 14

ATR_PERIOD = 14

STOCH_K_PERIOD = 14
STOCH_SMOOTH = 3
STOCH_D_PERIOD = 3

CCI_PERIOD = 20

MOMENTUM_PERIOD = 10

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

STOCH_RSI_RSI_PERIOD = 14
STOCH_RSI_STOCH_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3

WILLIAMS_PERIOD = 14

ULTIMATE_FAST = 7
ULTIMATE_MIDDLE = 14
ULTIMATE_SLOW = 28


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBAL STATE
# ============================================================

data_lock = threading.RLock()
cycle_lock = threading.RLock()
analysis_lock = threading.Lock()

data_store = {}
history = []

stats = {
    "base_win": 0,
    "base_loss": 0,
    "recovery_win": 0,
    "recovery_loss": 0,
}

# Last BASE setup per symbol.
# Recovery is intentionally NOT blocked by this memory.
recent_base_setups = {}


cycle = {
    "active": False,
    "trade_type": None,
    "symbol": None,
    "direction": None,
    "pending": False,
    "recovery_used": False,
    "recovery_requested": False,
    "recovery_requested_at": 0,
    "generating": False,
    "setup_key": None,
    "last_signal_time": 0,
    "base_symbol": None,
    "base_direction": None,
    "last_signal": None,
}


telegram_loop = None
application = None


# ============================================================
# BASIC HELPERS
# ============================================================

def now_algiers():
    return datetime.now(TIMEZONE)


def now_timestamp():
    return time.time()


def fmt_time(dt=None):
    if dt is None:
        dt = now_algiers()
    return dt.strftime("%H:%M:%S")


def is_owner(update: Update):
    if not update.effective_user:
        return False
    return update.effective_user.id == OWNER_ID


async def owner_only(update: Update):
    if not is_owner(update):
        if update.message:
            await update.message.reply_text(
                "⛔ هذا الأمر متاح للمالك فقط."
            )
        return False
    return True


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def normalize_direction(direction):
    direction = str(direction or "").upper().strip()

    if direction in ("UP", "CALL", "BUY"):
        return "UP"

    if direction in ("DOWN", "PUT", "SELL"):
        return "DOWN"

    return None


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):

    if not isinstance(c, dict):
        return None

    try:
        t = int(
            c.get("time")
            or c.get("timestamp")
            or c.get("t")
            or 0
        )

        o = safe_float(
            c.get("open")
            if c.get("open") is not None
            else c.get("o")
        )

        h = safe_float(
            c.get("high")
            if c.get("high") is not None
            else c.get("h")
        )

        l = safe_float(
            c.get("low")
            if c.get("low") is not None
            else c.get("l")
        )

        cl = safe_float(
            c.get("close")
            if c.get("close") is not None
            else c.get("c")
        )

        if t <= 0:
            return None

        if min(o, h, l, cl) <= 0:
            return None

        if h < max(o, cl):
            return None

        if l > min(o, cl):
            return None

        return {
            "time": t,
            "open": o,
            "high": h,
            "low": l,
            "close": cl,
        }

    except Exception:
        return None


def normalize_candles(raw):

    if not isinstance(raw, list):
        return []

    result = []

    for c in raw:
        nc = normalize_candle(c)
        if nc:
            result.append(nc)

    result.sort(key=lambda x: x["time"])

    clean = []
    seen = set()

    for c in result:
        if c["time"] in seen:
            continue

        seen.add(c["time"])
        clean.append(c)

    return clean


# ============================================================
# CLOSED CANDLE ENGINE
# ============================================================

def get_closed_candles(candles):

    if not isinstance(candles, list):
        return None

    if len(candles) < REQUIRED_TOTAL_CANDLES:
        return None

    # MT4 last candle = currently forming M1 candle.
    closed = candles[:-1]

    if len(closed) < MIN_CANDLES:
        return None

    return closed


# ============================================================
# MATH / SERIES HELPERS
# ============================================================

def sma(values, period):

    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def ema_series(values, period):

    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)

    current = sum(values[:period]) / period

    result = [None] * (period - 1)
    result.append(current)

    for value in values[period:]:
        current = (
            (value - current) * multiplier
            + current
        )
        result.append(current)

    return result


def ema(values, period):

    series = ema_series(values, period)

    if not series:
        return None

    return series[-1]


# ============================================================
# RSI
# ============================================================

def rsi_series(values, period=14):

    if len(values) < period + 1:
        return []

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    result = [None] * period

    if avg_loss == 0:
        result.append(100.0)
    else:
        rs = avg_gain / avg_loss
        result.append(100.0 - 100.0 / (1.0 + rs))

    for i in range(period, len(gains)):

        avg_gain = (
            avg_gain * (period - 1)
            + gains[i]
        ) / period

        avg_loss = (
            avg_loss * (period - 1)
            + losses[i]
        ) / period

        if avg_loss == 0:
            value = 100.0
        else:
            rs = avg_gain / avg_loss
            value = 100.0 - 100.0 / (1.0 + rs)

        result.append(value)

    return result


def rsi(values, period=14):

    series = rsi_series(values, period)

    return series[-1] if series else None


# ============================================================
# ATR
# ============================================================

def true_ranges(candles):

    result = []

    for i, c in enumerate(candles):

        if i == 0:
            tr = c["high"] - c["low"]

        else:
            prev_close = candles[i - 1]["close"]

            tr = max(
                c["high"] - c["low"],
                abs(c["high"] - prev_close),
                abs(c["low"] - prev_close),
            )

        result.append(tr)

    return result


def atr_series(candles, period=14):

    trs = true_ranges(candles)

    if len(trs) < period:
        return []

    value = sum(trs[:period]) / period

    result = [None] * (period - 1)
    result.append(value)

    for tr in trs[period:]:
        value = (
            value * (period - 1)
            + tr
        ) / period

        result.append(value)

    return result


def atr(candles, period=14):

    series = atr_series(candles, period)

    return series[-1] if series else None


# ============================================================
# ADX + DI
# ============================================================

def adx_di(candles, period=14):

    if len(candles) < period * 2 + 5:
        return None

    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):

        current = candles[i]
        prev = candles[i - 1]

        up_move = current["high"] - prev["high"]
        down_move = prev["low"] - current["low"]

        plus = (
            up_move
            if up_move > down_move and up_move > 0
            else 0.0
        )

        minus = (
            down_move
            if down_move > up_move and down_move > 0
            else 0.0
        )

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - prev["close"]),
            abs(current["low"] - prev["close"]),
        )

        tr_values.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(tr_values) < period * 2:
        return None

    tr_avg = sum(tr_values[:period]) / period
    plus_avg = sum(plus_dm[:period]) / period
    minus_avg = sum(minus_dm[:period]) / period

    dx_values = []

    last_plus_di = 0.0
    last_minus_di = 0.0

    for i in range(period, len(tr_values)):

        tr_avg = (
            tr_avg * (period - 1)
            + tr_values[i]
        ) / period

        plus_avg = (
            plus_avg * (period - 1)
            + plus_dm[i]
        ) / period

        minus_avg = (
            minus_avg * (period - 1)
            + minus_dm[i]
        ) / period

        if tr_avg <= 0:
            continue

        plus_di = 100.0 * plus_avg / tr_avg
        minus_di = 100.0 * minus_avg / tr_avg

        last_plus_di = plus_di
        last_minus_di = minus_di

        denominator = plus_di + minus_di

        if denominator <= 0:
            continue

        dx = (
            100.0
            * abs(plus_di - minus_di)
            / denominator
        )

        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    adx_value = sum(dx_values[:period]) / period

    for value in dx_values[period:]:
        adx_value = (
            adx_value * (period - 1)
            + value
        ) / period

    return {
        "adx": adx_value,
        "plus_di": last_plus_di,
        "minus_di": last_minus_di,
    }


# ============================================================
# STOCHASTIC 14,3,3
# ============================================================

def stochastic(candles):

    if len(candles) < STOCH_K_PERIOD + STOCH_SMOOTH + STOCH_D_PERIOD:
        return None

    raw_k = []

    for i in range(STOCH_K_PERIOD - 1, len(candles)):

        window = candles[
            i - STOCH_K_PERIOD + 1:
            i + 1
        ]

        highest = max(c["high"] for c in window)
        lowest = min(c["low"] for c in window)

        if highest == lowest:
            k = 50.0
        else:
            k = (
                100.0
                * (
                    candles[i]["close"] - lowest
                )
                / (highest - lowest)
            )

        raw_k.append(k)

    if len(raw_k) < STOCH_SMOOTH:
        return None

    smooth_k_series = []

    for i in range(STOCH_SMOOTH - 1, len(raw_k)):
        smooth_k_series.append(
            sum(
                raw_k[
                    i - STOCH_SMOOTH + 1:
                    i + 1
                ]
            ) / STOCH_SMOOTH
        )

    if len(smooth_k_series) < STOCH_D_PERIOD:
        return None

    k = smooth_k_series[-1]

    d = sum(
        smooth_k_series[-STOCH_D_PERIOD:]
    ) / STOCH_D_PERIOD

    return {
        "k": k,
        "d": d,
    }


# ============================================================
# CCI 20
# ============================================================

def cci(candles, period=20):

    if len(candles) < period:
        return None

    typical = [
        (
            c["high"]
            + c["low"]
            + c["close"]
        ) / 3.0
        for c in candles
    ]

    current = typical[-1]

    window = typical[-period:]

    mean = sum(window) / period

    deviation = (
        sum(abs(x - mean) for x in window)
        / period
    )

    if deviation <= 0:
        return 0.0

    return (
        current - mean
    ) / (
        0.015 * deviation
    )


# ============================================================
# AWESOME OSCILLATOR
# ============================================================

def awesome_oscillator(candles):

    if len(candles) < 34:
        return None

    median = [
        (c["high"] + c["low"]) / 2.0
        for c in candles
    ]

    fast = sma(median, 5)
    slow = sma(median, 34)

    if fast is None or slow is None:
        return None

    return fast - slow


# ============================================================
# MOMENTUM 10
# ============================================================

def momentum(values, period=10):

    if len(values) <= period:
        return None

    return values[-1] - values[-1 - period]


# ============================================================
# MACD 12,26,9
# ============================================================

def macd(values):

    if len(values) < MACD_SLOW + MACD_SIGNAL:
        return None

    fast_series = ema_series(values, MACD_FAST)
    slow_series = ema_series(values, MACD_SLOW)

    macd_values = []

    for i in range(len(values)):

        if (
            fast_series[i] is None
            or slow_series[i] is None
        ):
            continue

        macd_values.append(
            fast_series[i]
            - slow_series[i]
        )

    if len(macd_values) < MACD_SIGNAL:
        return None

    signal_series = ema_series(
        macd_values,
        MACD_SIGNAL
    )

    signal = signal_series[-1]

    level = macd_values[-1]

    return {
        "level": level,
        "signal": signal,
        "histogram": level - signal,
    }


# ============================================================
# STOCHASTIC RSI FAST 3,3,14,14
# ============================================================

def stochastic_rsi(values):

    rsi_values = rsi_series(
        values,
        STOCH_RSI_RSI_PERIOD
    )

    clean = [
        x for x in rsi_values
        if x is not None
    ]

    if len(clean) < STOCH_RSI_STOCH_PERIOD:
        return None

    raw = []

    for i in range(
        STOCH_RSI_STOCH_PERIOD - 1,
        len(clean)
    ):

        window = clean[
            i - STOCH_RSI_STOCH_PERIOD + 1:
            i + 1
        ]

        lowest = min(window)
        highest = max(window)

        if highest == lowest:
            value = 50.0
        else:
            value = (
                100.0
                * (clean[i] - lowest)
                / (highest - lowest)
            )

        raw.append(value)

    if len(raw) < STOCH_RSI_K:
        return None

    k_values = []

    for i in range(STOCH_RSI_K - 1, len(raw)):
        k_values.append(
            sum(
                raw[
                    i - STOCH_RSI_K + 1:
                    i + 1
                ]
            ) / STOCH_RSI_K
        )

    if len(k_values) < STOCH_RSI_D:
        return None

    k = k_values[-1]

    d = sum(
        k_values[-STOCH_RSI_D:]
    ) / STOCH_RSI_D

    return {
        "k": k,
        "d": d,
    }


# ============================================================
# WILLIAMS %R 14
# ============================================================

def williams_r(candles, period=14):

    if len(candles) < period:
        return None

    window = candles[-period:]

    highest = max(c["high"] for c in window)
    lowest = min(c["low"] for c in window)

    if highest == lowest:
        return -50.0

    return (
        -100.0
        * (
            highest - candles[-1]["close"]
        )
        / (highest - lowest)
    )


# ============================================================
# BULL / BEAR POWER
# ============================================================

def bull_bear_power(candles, period=13):

    closes = [
        c["close"]
        for c in candles
    ]

    ema_value = ema(
        closes,
        period
    )

    if ema_value is None:
        return None

    last = candles[-1]

    bull = last["high"] - ema_value
    bear = last["low"] - ema_value

    return {
        "bull": bull,
        "bear": bear,
        "net": bull + bear,
    }


# ============================================================
# ULTIMATE OSCILLATOR 7,14,28
# ============================================================

def ultimate_oscillator(candles):

    required = ULTIMATE_SLOW + 1

    if len(candles) < required:
        return None

    bp = []
    tr = []

    for i in range(1, len(candles)):

        c = candles[i]
        prev = candles[i - 1]

        true_low = min(
            c["low"],
            prev["close"]
        )

        true_high = max(
            c["high"],
            prev["close"]
        )

        bp.append(
            c["close"] - true_low
        )

        tr.append(
            true_high - true_low
        )

    def average(period):

        b = sum(bp[-period:])
        t = sum(tr[-period:])

        if t <= 0:
            return 0.0

        return b / t

    a7 = average(7)
    a14 = average(14)
    a28 = average(28)

    return 100.0 * (
        4.0 * a7
        + 2.0 * a14
        + a28
    ) / 7.0


# ============================================================
# MOVING AVERAGE SUMMARY
# ============================================================

def moving_average_summary(values):

    periods = [5, 10, 20, 50, 100, 200]

    close = values[-1]

    buy = 0
    sell = 0

    details = []

    for period in periods:

        if len(values) < period:
            continue

        sma_value = sma(
            values,
            period
        )

        ema_value = ema(
            values,
            period
        )

        if close > sma_value:
            buy += 1
            sma_state = "BUY"
        else:
            sell += 1
            sma_state = "SELL"

        if close > ema_value:
            buy += 1
            ema_state = "BUY"
        else:
            sell += 1
            ema_state = "SELL"

        details.append({
            "period": period,
            "sma": sma_value,
            "ema": ema_value,
            "sma_state": sma_state,
            "ema_state": ema_state,
        })

    total = buy + sell

    if total == 0:
        summary = "NEUTRAL"
    elif sell > buy:
        summary = "SELL"
    elif buy > sell:
        summary = "BUY"
    else:
        summary = "NEUTRAL"

    return {
        "buy": buy,
        "sell": sell,
        "summary": summary,
        "details": details,
    }


# ============================================================
# OSCILLATOR SUMMARY
# ============================================================

def oscillator_summary(
    rsi_value,
    stoch_data,
    cci_value,
    ao_value,
    momentum_value,
    macd_data,
    stoch_rsi_data,
    williams_value,
    bull_bear_data,
    ultimate_value
):

    up = 0
    down = 0
    neutral = 0

    states = {}

    # RSI
    if rsi_value > 52:
        states["RSI"] = "BUY"
        up += 1
    elif rsi_value < 48:
        states["RSI"] = "SELL"
        down += 1
    else:
        states["RSI"] = "NEUTRAL"
        neutral += 1

    # Stochastic
    if stoch_data:
        if (
            stoch_data["k"] > stoch_data["d"]
            and stoch_data["k"] > 50
        ):
            states["STOCH"] = "BUY"
            up += 1
        elif (
            stoch_data["k"] < stoch_data["d"]
            and stoch_data["k"] < 50
        ):
            states["STOCH"] = "SELL"
            down += 1
        else:
            states["STOCH"] = "NEUTRAL"
            neutral += 1

    # CCI
    if cci_value is not None:
        if cci_value > 100:
            states["CCI"] = "BUY"
            up += 1
        elif cci_value < -100:
            states["CCI"] = "SELL"
            down += 1
        elif cci_value > 0:
            states["CCI"] = "BUY"
            up += 1
        elif cci_value < 0:
            states["CCI"] = "SELL"
            down += 1
        else:
            states["CCI"] = "NEUTRAL"
            neutral += 1

    # Awesome Oscillator
    if ao_value is not None:
        if ao_value > 0:
            states["AO"] = "BUY"
            up += 1
        elif ao_value < 0:
            states["AO"] = "SELL"
            down += 1
        else:
            states["AO"] = "NEUTRAL"
            neutral += 1

    # Momentum
    if momentum_value is not None:
        if momentum_value > 0:
            states["MOMENTUM"] = "BUY"
            up += 1
        elif momentum_value < 0:
            states["MOMENTUM"] = "SELL"
            down += 1
        else:
            states["MOMENTUM"] = "NEUTRAL"
            neutral += 1

    # MACD
    if macd_data:
        if macd_data["level"] > macd_data["signal"]:
            states["MACD"] = "BUY"
            up += 1
        elif macd_data["level"] < macd_data["signal"]:
            states["MACD"] = "SELL"
            down += 1
        else:
            states["MACD"] = "NEUTRAL"
            neutral += 1

    # Stoch RSI
    if stoch_rsi_data:
        if stoch_rsi_data["k"] > stoch_rsi_data["d"]:
            states["STOCH_RSI"] = "BUY"
            up += 1
        elif stoch_rsi_data["k"] < stoch_rsi_data["d"]:
            states["STOCH_RSI"] = "SELL"
            down += 1
        else:
            states["STOCH_RSI"] = "NEUTRAL"
            neutral += 1

    # Williams
    if williams_value is not None:
        if williams_value > -50:
            states["WILLIAMS"] = "BUY"
            up += 1
        elif williams_value < -50:
            states["WILLIAMS"] = "SELL"
            down += 1
        else:
            states["WILLIAMS"] = "NEUTRAL"
            neutral += 1

    # Bull/Bear
    if bull_bear_data:
        if bull_bear_data["net"] > 0:
            states["BULL_BEAR"] = "BUY"
            up += 1
        elif bull_bear_data["net"] < 0:
            states["BULL_BEAR"] = "SELL"
            down += 1
        else:
            states["BULL_BEAR"] = "NEUTRAL"
            neutral += 1

    # Ultimate
    if ultimate_value is not None:
        if ultimate_value > 50:
            states["ULTIMATE"] = "BUY"
            up += 1
        elif ultimate_value < 50:
            states["ULTIMATE"] = "SELL"
            down += 1
        else:
            states["ULTIMATE"] = "NEUTRAL"
            neutral += 1

    total = up + down + neutral

    if down > up:
        summary = "SELL"
    elif up > down:
        summary = "BUY"
    else:
        summary = "NEUTRAL"

    return {
        "buy": up,
        "sell": down,
        "neutral": neutral,
        "total": total,
        "summary": summary,
        "states": states,
    }


# ============================================================
# PRICE ACTION
# ============================================================

def candle_features(c):

    body = abs(c["close"] - c["open"])
    total = c["high"] - c["low"]

    if total <= 0:
        return {
            "body": 0.0,
            "range": 0.0,
            "body_ratio": 0.0,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
        }

    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]

    return {
        "body": body,
        "range": total,
        "body_ratio": body / total,
        "upper_wick": upper,
        "lower_wick": lower,
    }


def recent_average_range(candles, count=14):

    subset = candles[-count:]

    if not subset:
        return 0.0

    return sum(
        c["high"] - c["low"]
        for c in subset
    ) / len(subset)


def candle_direction(c):

    if c["close"] > c["open"]:
        return "UP"

    if c["close"] < c["open"]:
        return "DOWN"

    return "NEUTRAL"


def market_structure(candles):

    if len(candles) < 12:
        return "NEUTRAL"

    recent = candles[-8:]

    first = recent[:4]
    last = recent[4:]

    first_high = max(c["high"] for c in first)
    first_low = min(c["low"] for c in first)

    last_high = max(c["high"] for c in last)
    last_low = min(c["low"] for c in last)

    if last_high > first_high and last_low > first_low:
        return "UP"

    if last_high < first_high and last_low < first_low:
        return "DOWN"

    return "NEUTRAL"


def breakout_status(candles):

    if len(candles) < 12:
        return "NONE"

    last = candles[-1]
    previous = candles[-9:-1]

    previous_high = max(c["high"] for c in previous)
    previous_low = min(c["low"] for c in previous)

    if last["close"] > previous_high:
        return "UP"

    if last["close"] < previous_low:
        return "DOWN"

    return "NONE"


def liquidity_status(candles):

    if len(candles) < 12:
        return "NONE"

    last = candles[-1]
    previous = candles[-9:-1]

    high = max(c["high"] for c in previous)
    low = min(c["low"] for c in previous)

    features = candle_features(last)

    if (
        last["high"] > high
        and last["close"] < high
        and features["upper_wick"] > features["body"]
    ):
        return "DOWN"

    if (
        last["low"] < low
        and last["close"] > low
        and features["lower_wick"] > features["body"]
    ):
        return "UP"

    return "NONE"


def abnormal_candle(candles):

    if len(candles) < 16:
        return False

    avg_range = recent_average_range(
        candles[-16:-1],
        15
    )

    if avg_range <= 0:
        return False

    last_range = (
        candles[-1]["high"]
        - candles[-1]["low"]
    )

    return (
        last_range
        > avg_range * MAX_CANDLE_RANGE_RATIO
    )


# ============================================================
# PRIMARY CORE
# ============================================================

def primary_indicator_states(
    ema9,
    ema21,
    rsi14,
    adx_value,
    plus_di,
    minus_di
):

    if ema9 > ema21:
        ema_state = "UP"
    elif ema9 < ema21:
        ema_state = "DOWN"
    else:
        ema_state = "NEUTRAL"

    if rsi14 > 52:
        rsi_state = "UP"
    elif rsi14 < 48:
        rsi_state = "DOWN"
    else:
        rsi_state = "NEUTRAL"

    if adx_value >= MIN_ADX:
        if plus_di > minus_di:
            di_state = "UP"
        elif minus_di > plus_di:
            di_state = "DOWN"
        else:
            di_state = "NEUTRAL"
    else:
        di_state = "NEUTRAL"

    return {
        "ema": ema_state,
        "rsi": rsi_state,
        "adx_di": di_state,
    }


def primary_support_count(direction, states):

    return sum(
        1
        for state in states.values()
        if state == direction
    )


def primary_opposition_count(direction, states):

    opposite = (
        "DOWN"
        if direction == "UP"
        else "UP"
    )

    return sum(
        1
        for state in states.values()
        if state == opposite
    )


# ============================================================
# SUPPORT SCORING
# ============================================================

def support_direction_scores(
    direction,
    oscillator,
    ma_summary
):

    opposite = (
        "DOWN"
        if direction == "UP"
        else "UP"
    )

    support = 0
    opposition = 0

    for state in oscillator["states"].values():

        if state == (
            "BUY" if direction == "UP" else "SELL"
        ):
            support += 1

        elif state == (
            "SELL" if direction == "UP" else "BUY"
        ):
            opposition += 1

    if ma_summary["summary"] == (
        "BUY" if direction == "UP" else "SELL"
    ):
        support += 2

    elif ma_summary["summary"] == (
        "SELL" if direction == "UP" else "BUY"
    ):
        opposition += 2

    return support, opposition


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def calculate_analysis(candles):

    closed = get_closed_candles(candles)

    if closed is None:
        return None

    closes = [
        c["close"]
        for c in closed
    ]

    # --------------------------------------------------------
    # CORE
    # --------------------------------------------------------

    ema9 = ema(closes, EMA_FAST)
    ema21 = ema(closes, EMA_SLOW)

    rsi14 = rsi(
        closes,
        RSI_PERIOD
    )

    adx_data = adx_di(
        closed,
        ADX_PERIOD
    )

    atr14 = atr(
        closed,
        ATR_PERIOD
    )

    if (
        ema9 is None
        or ema21 is None
        or rsi14 is None
        or adx_data is None
        or atr14 is None
    ):
        return None

    adx_value = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]

    # --------------------------------------------------------
    # EXTRA INDICATORS
    # --------------------------------------------------------

    stoch_data = stochastic(closed)

    cci_value = cci(
        closed,
        CCI_PERIOD
    )

    ao_value = awesome_oscillator(closed)

    momentum_value = momentum(
        closes,
        MOMENTUM_PERIOD
    )

    macd_data = macd(closes)

    stoch_rsi_data = stochastic_rsi(
        closes
    )

    williams_value = williams_r(
        closed,
        WILLIAMS_PERIOD
    )

    bull_bear_data = bull_bear_power(
        closed,
        13
    )

    ultimate_value = ultimate_oscillator(
        closed
    )

    ma_summary = moving_average_summary(
        closes
    )

    oscillator = oscillator_summary(
        rsi14,
        stoch_data,
        cci_value,
        ao_value,
        momentum_value,
        macd_data,
        stoch_rsi_data,
        williams_value,
        bull_bear_data,
        ultimate_value
    )

    # --------------------------------------------------------
    # LAST CLOSED CANDLE
    # --------------------------------------------------------

    last = closed[-1]

    candle = candle_features(last)

    structure = market_structure(closed)
    breakout = breakout_status(closed)
    liquidity = liquidity_status(closed)

    candle_dir = candle_direction(last)

    avg_range = recent_average_range(
        closed,
        14
    )

    if avg_range <= 0:
        return None

    volatility_ratio = (
        candle["range"] / avg_range
    )

    volatility_ok = (
        candle["range"]
        >= avg_range * MIN_CANDLE_RANGE_RATIO
        and
        candle["range"]
        <= avg_range * MAX_CANDLE_RANGE_RATIO
        and
        atr14 > 0
    )

    abnormal = abnormal_candle(closed)

    # --------------------------------------------------------
    # PRIMARY STATES
    # --------------------------------------------------------

    primary_states = primary_indicator_states(
        ema9,
        ema21,
        rsi14,
        adx_value,
        plus_di,
        minus_di
    )

    up_primary = primary_support_count(
        "UP",
        primary_states
    )

    down_primary = primary_support_count(
        "DOWN",
        primary_states
    )

    # --------------------------------------------------------
    # STRICT CORE VALIDATION
    #
    # We require 2/3 primary support.
    # One neutral is allowed.
    # One opposite is allowed only if the other
    # two clearly support the direction.
    # --------------------------------------------------------

    up_core = (
        up_primary >= 2
        and down_primary <= 1
    )

    down_core = (
        down_primary >= 2
        and up_primary <= 1
    )

    # --------------------------------------------------------
    # INITIAL DIRECTION
    # --------------------------------------------------------

    up_score = 0
    down_score = 0

    # ========================================================
    # CORE SCORE = 10 POINTS
    # ========================================================

    # EMA 3
    if ema9 > ema21:
        up_score += 3
    elif ema9 < ema21:
        down_score += 3

    # RSI 2
    if rsi14 < 48:
        down_score += 2
    elif rsi14 > 52:
        up_score += 2

    # ADX + DI 3
    if adx_value >= STRONG_ADX:
        if plus_di > minus_di:
            up_score += 3
        elif minus_di > plus_di:
            down_score += 3

    elif adx_value >= MIN_ADX:
        if plus_di > minus_di:
            up_score += 2
        elif minus_di > plus_di:
            down_score += 2

    # ATR / volatility 1
    if volatility_ok:
        # Neutral point: this is quality, not direction.
        pass

    # Price above/below both EMA = 2
    if last["close"] > ema9 and last["close"] > ema21:
        up_score += 2

    elif last["close"] < ema9 and last["close"] < ema21:
        down_score += 2

    # ========================================================
    # CONFIRMATION SCORE
    # Maximum contribution is controlled below.
    # ========================================================

    up_support, up_opposition = support_direction_scores(
        "UP",
        oscillator,
        ma_summary
    )

    down_support, down_opposition = support_direction_scores(
        "DOWN",
        oscillator,
        ma_summary
    )

    # Convert many indicators into max 4 points per side.
    if up_support >= 8:
        up_score += 4
    elif up_support >= 6:
        up_score += 3
    elif up_support >= 4:
        up_score += 2
    elif up_support >= 2:
        up_score += 1

    if down_support >= 8:
        down_score += 4
    elif down_support >= 6:
        down_score += 3
    elif down_support >= 4:
        down_score += 2
    elif down_support >= 2:
        down_score += 1

    # ========================================================
    # PRICE ACTION = MAX 4
    # ========================================================

    if structure == "UP":
        up_score += 2
    elif structure == "DOWN":
        down_score += 2

    if breakout == "UP":
        up_score += 1
    elif breakout == "DOWN":
        down_score += 1

    if (
        candle["body_ratio"] >= 0.55
        and candle_dir == "UP"
    ):
        up_score += 1

    elif (
        candle["body_ratio"] >= 0.55
        and candle_dir == "DOWN"
    ):
        down_score += 1

    # --------------------------------------------------------
    # MAX SCORE = 20
    # --------------------------------------------------------

    up_score = min(up_score, MAX_SCORE)
    down_score = min(down_score, MAX_SCORE)

    if up_score > down_score:
        direction = "UP"
        raw_score = up_score
    elif down_score > up_score:
        direction = "DOWN"
        raw_score = down_score
    else:
        direction = None
        raw_score = 0

    if direction == "UP":
        core_support = up_primary
        core_opposition = down_primary
    elif direction == "DOWN":
        core_support = down_primary
        core_opposition = up_primary
    else:
        core_support = 0
        core_opposition = 0

    # --------------------------------------------------------
    # CORE MUST PASS
    # --------------------------------------------------------

    core_aligned = (
        (
            direction == "UP"
            and up_core
        )
        or
        (
            direction == "DOWN"
            and down_core
        )
    )

    # --------------------------------------------------------
    # STRONG PRIMARY CONFLICT
    # --------------------------------------------------------

    strong_primary_conflict = False

    if direction == "UP":

        if (
            ema9 < ema21
            and rsi14 < 48
        ):
            strong_primary_conflict = True

        if (
            adx_value >= STRONG_ADX
            and minus_di > plus_di
            and ema9 < ema21
        ):
            strong_primary_conflict = True

        if (
            adx_value >= STRONG_ADX
            and minus_di > plus_di
            and rsi14 < 45
        ):
            strong_primary_conflict = True

    elif direction == "DOWN":

        if (
            ema9 > ema21
            and rsi14 > 52
        ):
            strong_primary_conflict = True

        if (
            adx_value >= STRONG_ADX
            and plus_di > minus_di
            and ema9 > ema21
        ):
            strong_primary_conflict = True

        if (
            adx_value >= STRONG_ADX
            and plus_di > minus_di
            and rsi14 > 55
        ):
            strong_primary_conflict = True

    # --------------------------------------------------------
    # SECONDARY PRICE CONFLICT
    # --------------------------------------------------------

    secondary_conflict = False

    if direction == "UP":

        if (
            structure == "DOWN"
            and liquidity == "DOWN"
        ):
            secondary_conflict = True

    elif direction == "DOWN":

        if (
            structure == "UP"
            and liquidity == "UP"
        ):
            secondary_conflict = True

    # --------------------------------------------------------
    # SUPPORT CONTRADICTION
    # --------------------------------------------------------

    support_conflict = False

    if direction == "UP":

        if (
            up_opposition >= 7
            and up_support <= 3
        ):
            support_conflict = True

    elif direction == "DOWN":

        if (
            down_opposition >= 7
            and down_support <= 3
        ):
            support_conflict = True

    # --------------------------------------------------------
    # STOCH RSI EXTREME FILTER
    #
    # Extreme alone does NOT create a signal.
    # It prevents blindly chasing.
    # --------------------------------------------------------

    extreme_chase = False

    if direction == "DOWN":

        if (
            stoch_rsi_data
            and stoch_rsi_data["k"] <= 5
            and rsi14 <= 35
            and cci_value is not None
            and cci_value < -150
        ):
            extreme_chase = True

    elif direction == "UP":

        if (
            stoch_rsi_data
            and stoch_rsi_data["k"] >= 95
            and rsi14 >= 65
            and cci_value is not None
            and cci_value > 150
        ):
            extreme_chase = True

    # --------------------------------------------------------
    # CONFLICT REASONS
    # --------------------------------------------------------

    conflict_reasons = []

    if strong_primary_conflict:
        conflict_reasons.append("PRIMARY")

    if secondary_conflict:
        conflict_reasons.append("SECONDARY")

    if support_conflict:
        conflict_reasons.append("SUPPORT")

    if abnormal:
        conflict_reasons.append("ABNORMAL")

    if not volatility_ok:
        conflict_reasons.append("VOLATILITY")

    if extreme_chase:
        conflict_reasons.append("EXTREME")

    conflict = bool(conflict_reasons)

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = 0

    if direction:

        gap = abs(
            up_score - down_score
        )

        confidence = (
            70
            + raw_score * 0.75
            + gap * 0.80
        )

        if core_support == 3:
            confidence += 4
        elif core_support == 2:
            confidence += 1

        if adx_value >= 30:
            confidence += 3
        elif adx_value >= 25:
            confidence += 2

        if (
            oscillator["summary"]
            == ("BUY" if direction == "UP" else "SELL")
        ):
            confidence += 2

        if (
            ma_summary["summary"]
            == ("BUY" if direction == "UP" else "SELL")
        ):
            confidence += 2

        if structure == direction:
            confidence += 1

        if breakout == direction:
            confidence += 1

        confidence = int(
            clamp(
                confidence,
                MIN_CONFIDENCE,
                MAX_CONFIDENCE
            )
        )

    # --------------------------------------------------------
    # HUMAN-READABLE REASON
    # --------------------------------------------------------

    if direction == "UP":
        core_text = (
            f"EMA {'UP' if ema9 > ema21 else 'mixed'}, "
            f"RSI {rsi14:.1f}, "
            f"ADX {adx_value:.1f}, "
            f"DI+ {plus_di:.1f} > DI- {minus_di:.1f}"
        )
    elif direction == "DOWN":
        core_text = (
            f"EMA {'DOWN' if ema9 < ema21 else 'mixed'}, "
            f"RSI {rsi14:.1f}, "
            f"ADX {adx_value:.1f}, "
            f"DI- {minus_di:.1f} > DI+ {plus_di:.1f}"
        )
    else:
        core_text = "No clear primary direction."

    reason = (
        f"{core_text}. "
        f"Support {('BUY' if direction == 'UP' else 'SELL') if direction else 'NONE'} "
        f"{up_support if direction == 'UP' else down_support}. "
        f"MA={ma_summary['summary']}, "
        f"Osc={oscillator['summary']}, "
        f"Structure={structure}."
    )

    return {
        "direction": direction,

        "up_score": up_score,
        "down_score": down_score,
        "score": raw_score,

        "confidence": confidence,

        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi14,

        "adx": adx_value,
        "plus_di": plus_di,
        "minus_di": minus_di,

        "atr": atr14,

        "stoch": stoch_data,
        "cci": cci_value,
        "ao": ao_value,
        "momentum": momentum_value,
        "macd": macd_data,
        "stoch_rsi": stoch_rsi_data,
        "williams": williams_value,
        "bull_bear": bull_bear_data,
        "ultimate": ultimate_value,

        "ma_summary": ma_summary,
        "oscillator_summary": oscillator,

        "support_up": up_support,
        "support_down": down_support,

        "opposition_up": up_opposition,
        "opposition_down": down_opposition,

        "structure": structure,
        "breakout": breakout,
        "liquidity": liquidity,

        "candle_direction": candle_dir,
        "body_ratio": candle["body_ratio"],
        "candle_range": candle["range"],
        "average_range": avg_range,
        "volatility_ratio": volatility_ratio,

        "abnormal": abnormal,
        "volatility_ok": volatility_ok,

        "primary_states": primary_states,
        "primary_support": core_support,
        "primary_opposition": core_opposition,
        "core_aligned": core_aligned,

        "strong_primary_conflict":
            strong_primary_conflict,

        "secondary_conflict":
            secondary_conflict,

        "support_conflict":
            support_conflict,

        "extreme_chase":
            extreme_chase,

        "conflict_reasons":
            conflict_reasons,

        "conflict": conflict,

        "reason": reason,

        "price": last["close"],
        "candle_time": last["time"],
    }


# ============================================================
# CANCELLATION
# ============================================================

def cancellation_price(candles, direction):

    closed = get_closed_candles(candles)

    if closed is None:
        closed = candles

    recent = closed[-8:]

    if direction == "UP":
        return min(c["low"] for c in recent)

    return max(c["high"] for c in recent)


# ============================================================
# SETUP / FRESHNESS
# ============================================================

def make_setup_key(symbol, analysis):

    return (
        f"{symbol}|"
        f"{analysis['direction']}|"
        f"{analysis['candle_time']}"
    )


def fresh_base_setup(symbol, analysis):

    record = recent_base_setups.get(symbol)

    if not record:
        return True

    elapsed = (
        now_timestamp()
        - record["time"]
    )

    if elapsed >= SETUP_REPEAT_BLOCK_SECONDS:
        return True

    # Direction change is considered a genuinely new setup.
    if analysis["direction"] != record["direction"]:
        return True

    # Require meaningful price movement before reusing same pair.
    old_price = record["price"]
    new_price = analysis["price"]

    atr_value = analysis.get("atr", 0.0)

    if (
        atr_value > 0
        and abs(new_price - old_price)
        >= atr_value * FRESH_SETUP_ATR_RATIO
    ):
        return True

    # Strong indicator change can also create a new setup.
    old_rsi = record.get("rsi", 50.0)

    if abs(
        analysis["rsi"] - old_rsi
    ) >= 5.0:
        return True

    logger.info(
        "FRESH SETUP BLOCK | %s | "
        "elapsed=%.0fs/%ss | same direction | "
        "price movement too small",
        symbol,
        elapsed,
        SETUP_REPEAT_BLOCK_SECONDS
    )

    return False


# ============================================================
# VALIDATION
# ============================================================

def validate_signal(symbol, candles):

    analysis = calculate_analysis(candles)

    if not analysis:
        return None

    if not analysis["direction"]:
        return None

    # Core is mandatory.
    if not analysis["core_aligned"]:

        logger.info(
            "REJECT | %s | CORE | "
            "support=%d/3 opposition=%d/3 | "
            "EMA=%s RSI=%s ADXDI=%s",
            symbol,
            analysis["primary_support"],
            analysis["primary_opposition"],
            analysis["primary_states"]["ema"],
            analysis["primary_states"]["rsi"],
            analysis["primary_states"]["adx_di"],
        )

        return None

    # Conflict is mandatory rejection.
    if analysis["conflict"]:

        logger.info(
            "REJECT | %s | CONFLICT=%s | "
            "score=%d/20 | "
            "range_ratio=%.2f | "
            "MA=%s | OSC=%s",
            symbol,
            ",".join(
                analysis["conflict_reasons"]
            ),
            analysis["score"],
            analysis["volatility_ratio"],
            analysis["ma_summary"]["summary"],
            analysis["oscillator_summary"]["summary"],
        )

        return None

    # Minimum score.
    if analysis["score"] < MIN_SCORE:

        logger.info(
            "REJECT | %s | SCORE=%d/20 < %d",
            symbol,
            analysis["score"],
            MIN_SCORE
        )

        return None

    # Score gap.
    score_gap = abs(
        analysis["up_score"]
        - analysis["down_score"]
    )

    if score_gap < MIN_SCORE_GAP:

        logger.info(
            "REJECT | %s | GAP=%d < %d",
            symbol,
            score_gap,
            MIN_SCORE_GAP
        )

        return None

    # ADX must be valid when DI is directional.
    if (
        analysis["primary_states"]["adx_di"]
        == analysis["direction"]
        and analysis["adx"] < MIN_ADX
    ):

        logger.info(
            "REJECT | %s | ADX %.2f < %.2f",
            symbol,
            analysis["adx"],
            MIN_ADX
        )

        return None

    # Final directional contradiction.
    if analysis["direction"] == "UP":

        if (
            analysis["ema9"] < analysis["ema21"]
            and analysis["rsi"] < 48
        ):
            return None

        if (
            analysis["adx"] >= STRONG_ADX
            and analysis["minus_di"]
            > analysis["plus_di"]
            and analysis["ema9"]
            < analysis["ema21"]
        ):
            return None

    elif analysis["direction"] == "DOWN":

        if (
            analysis["ema9"] > analysis["ema21"]
            and analysis["rsi"] > 52
        ):
            return None

        if (
            analysis["adx"] >= STRONG_ADX
            and analysis["plus_di"]
            > analysis["minus_di"]
            and analysis["ema9"]
            > analysis["ema21"]
        ):
            return None

    analysis["cancellation"] = cancellation_price(
        candles,
        analysis["direction"]
    )

    return analysis


# ============================================================
# BEST PAIR
# ============================================================

def choose_best_pair():

    candidates = []

    current_time = now_timestamp()

    with data_lock:
        items = list(data_store.items())

    for symbol, info in items:

        candles = info.get("candles", [])
        received = info.get("received_at", 0)

        if not candles:
            continue

        if current_time - received > MAX_DATA_AGE_SECONDS:
            continue

        if len(candles) < REQUIRED_TOTAL_CANDLES:

            logger.info(
                "CANDIDATE SKIP | %s | candles=%d need=%d",
                symbol,
                len(candles),
                REQUIRED_TOTAL_CANDLES
            )

            continue

        analysis = calculate_analysis(candles)

        if not analysis:
            continue

        if not analysis["direction"]:

            logger.info(
                "CANDIDATE SKIP | %s | NO DIRECTION",
                symbol
            )

            continue

        if not analysis["core_aligned"]:

            logger.info(
                "CANDIDATE SKIP | %s | "
                "CORE=%d/3 OPP=%d/3 | "
                "EMA=%s RSI=%s ADXDI=%s",
                symbol,
                analysis["primary_support"],
                analysis["primary_opposition"],
                analysis["primary_states"]["ema"],
                analysis["primary_states"]["rsi"],
                analysis["primary_states"]["adx_di"],
            )

            continue

        if analysis["conflict"]:

            logger.info(
                "CANDIDATE SKIP | %s | "
                "CONFLICT=%s | "
                "score=%d/20 | "
                "MA=%s OSC=%s",
                symbol,
                ",".join(
                    analysis["conflict_reasons"]
                ),
                analysis["score"],
                analysis["ma_summary"]["summary"],
                analysis["oscillator_summary"]["summary"],
            )

            continue

        if analysis["score"] < MIN_SCORE:

            logger.info(
                "CANDIDATE SKIP | %s | SCORE=%d/20",
                symbol,
                analysis["score"]
            )

            continue

        gap = abs(
            analysis["up_score"]
            - analysis["down_score"]
        )

        if gap < MIN_SCORE_GAP:

            logger.info(
                "CANDIDATE SKIP | %s | GAP=%d",
                symbol,
                gap
            )

            continue

        # Fresh setup applies to BASE only.
        if not fresh_base_setup(
            symbol,
            analysis
        ):
            continue

        # Quality ranking.
        quality = float(
            analysis["score"]
        )

        quality += gap * 0.8

        if analysis["primary_support"] == 3:
            quality += 4.0

        elif analysis["primary_support"] == 2:
            quality += 1.0

        if analysis["adx"] >= 30:
            quality += 2.0

        elif analysis["adx"] >= 25:
            quality += 1.0

        if (
            analysis["oscillator_summary"]["summary"]
            == (
                "BUY"
                if analysis["direction"] == "UP"
                else "SELL"
            )
        ):
            quality += 2.0

        if (
            analysis["ma_summary"]["summary"]
            == (
                "BUY"
                if analysis["direction"] == "UP"
                else "SELL"
            )
        ):
            quality += 2.0

        if analysis["structure"] == analysis["direction"]:
            quality += 1.5

        if analysis["breakout"] == analysis["direction"]:
            quality += 1.0

        if (
            analysis["candle_direction"]
            == analysis["direction"]
            and analysis["body_ratio"] >= 0.45
        ):
            quality += 1.0

        if analysis["liquidity"] == analysis["direction"]:
            quality += 1.0

        candidates.append({
            "symbol": symbol,
            "candles": candles,
            "analysis": analysis,
            "quality": quality,
        })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    top = candidates[:8]

    logger.info(
        "TOP CANDIDATES: %s",
        " | ".join(
            (
                f"{x['symbol']} "
                f"{x['analysis']['direction']} "
                f"{x['analysis']['score']}/20 "
                f"CORE={x['analysis']['primary_support']}/3 "
                f"OSC={x['analysis']['oscillator_summary']['summary']} "
                f"MA={x['analysis']['ma_summary']['summary']} "
                f"Q={x['quality']:.1f}"
            )
            for x in top
        )
    )

    return candidates[0]


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_message(text):

    global application

    if application is None:
        return False

    try:

        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=text,
            parse_mode="Markdown"
        )

        return True

    except Exception as e:

        logger.error(
            "Telegram send failed: %s",
            e
        )

        try:

            await application.bot.send_message(
                chat_id=OWNER_ID,
                text=text
            )

            return True

        except Exception:
            return False


def send_message_threadsafe(text):

    if telegram_loop is None:
        return False

    try:

        future = asyncio.run_coroutine_threadsafe(
            send_message(text),
            telegram_loop
        )

        future.result(timeout=20)

        return True

    except Exception as e:

        logger.error(
            "Threadsafe Telegram send failed: %s",
            e
        )

        return False


# ============================================================
# SIGNAL CARD
# ============================================================

def build_signal_text(
    symbol,
    analysis,
    trade_type
):

    direction = analysis["direction"]

    emoji = (
        "🟢 UP"
        if direction == "UP"
        else
        "🔴 DOWN"
    )

    entry_time = (
        now_algiers()
        + timedelta(
            seconds=ENTRY_DELAY_SECONDS
        )
    )

    title = (
        "🎯 BASE TRADE"
        if trade_type == "BASE"
        else
        "♻️ RECOVERY 1/1"
    )

    osc = analysis["oscillator_summary"]
    ma = analysis["ma_summary"]

    text = (
        "🎓 *ZinoProSignalAI*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 *{symbol} | M1*\n\n"
        f"{title}\n"
        f"➡️ *{emoji}*\n\n"
        f"🔥 Confidence: *{analysis['confidence']}%*\n"
        f"📈 UP Score: *{analysis['up_score']}/20*\n"
        f"📉 DOWN Score: *{analysis['down_score']}/20*\n\n"
        f"⏱️ Entry after: *1 minute*\n"
        f"🕐 *ENTRY TIME: "
        f"{fmt_time(entry_time)} 🇩🇿*\n\n"
        f"💰 Price: *{analysis['price']:.6f}*\n"
        f"❌ Cancellation: "
        f"*{analysis['cancellation']:.6f}*\n\n"
        "🧠 *Confirmation:*\n"
        f"EMA: {analysis['primary_states']['ema']} | "
        f"RSI: {analysis['primary_states']['rsi']} | "
        f"ADX/DI: {analysis['primary_states']['adx_di']}\n"
        f"ADX: {analysis['adx']:.1f}\n"
        f"Support: "
        f"{analysis['primary_support']}/3\n"
        f"Oscillators: "
        f"{osc['summary']} "
        f"({osc['buy']}B/{osc['sell']}S)\n"
        f"Moving Averages: "
        f"{ma['summary']} "
        f"({ma['buy']}B/{ma['sell']}S)\n\n"
        f"📐 Structure: {analysis['structure']}\n"
        f"💧 Liquidity: {analysis['liquidity']}\n"
        f"⚡ ATR: {analysis['atr']:.6f}\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "⚠️ One trade only — waiting for result."
    )

    return text


# ============================================================
# SEND ONE SIGNAL
# ============================================================

def send_one_signal(
    symbol,
    candles,
    analysis,
    trade_type
):

    global cycle

    with cycle_lock:

        if cycle["pending"]:
            return False

        if trade_type == "BASE":

            if cycle["active"]:
                return False

        elif trade_type == "RECOVERY":

            if not cycle["active"]:
                return False

            if cycle["recovery_used"]:
                return False

            if cycle["trade_type"] != "RECOVERY":
                return False

        setup_key = make_setup_key(
            symbol,
            analysis
        )

        if cycle["setup_key"] == setup_key:
            return False

        cycle["active"] = True
        cycle["trade_type"] = trade_type
        cycle["symbol"] = symbol
        cycle["direction"] = analysis["direction"]
        cycle["pending"] = True
        cycle["generating"] = False
        cycle["setup_key"] = setup_key
        cycle["last_signal_time"] = now_timestamp()

        cycle["last_signal"] = {
            "symbol": symbol,
            "direction": analysis["direction"],
            "trade_type": trade_type,
            "score": analysis["score"],
            "confidence": analysis["confidence"],
            "price": analysis["price"],
            "candle_time": analysis["candle_time"],
            "sent_at": now_timestamp(),
        }

        if trade_type == "BASE":

            cycle["base_symbol"] = symbol
            cycle["base_direction"] = analysis["direction"]

            recent_base_setups[symbol] = {
                "time": now_timestamp(),
                "direction": analysis["direction"],
                "price": analysis["price"],
                "rsi": analysis["rsi"],
                "candle_time": analysis["candle_time"],
            }

        else:

            cycle["recovery_used"] = True
            cycle["recovery_requested"] = False
            cycle["recovery_requested_at"] = 0

    text = build_signal_text(
        symbol,
        analysis,
        trade_type
    )

    success = send_message_threadsafe(text)

    if not success:

        with cycle_lock:

            cycle["pending"] = False

            if trade_type == "BASE":

                cycle["active"] = False
                cycle["trade_type"] = None
                cycle["symbol"] = None
                cycle["direction"] = None
                cycle["setup_key"] = None

            else:

                cycle["recovery_used"] = False
                cycle["recovery_requested"] = True
                cycle["recovery_requested_at"] = now_timestamp()

        return False

    with data_lock:

        history.append({
            "time": now_algiers().isoformat(),
            "symbol": symbol,
            "trade_type": trade_type,
            "direction": analysis["direction"],
            "score": analysis["score"],
            "confidence": analysis["confidence"],
            "price": analysis["price"],
            "result": "PENDING",
        })

        if len(history) > 100:
            del history[:-100]

    logger.info(
        "SIGNAL SENT | %s | %s | %s | "
        "%d/20 | confidence=%d%% | "
        "CORE=%d/3 | OSC=%s | MA=%s",
        symbol,
        trade_type,
        analysis["direction"],
        analysis["score"],
        analysis["confidence"],
        analysis["primary_support"],
        analysis["oscillator_summary"]["summary"],
        analysis["ma_summary"]["summary"],
    )

    return True


# ============================================================
# BASE
# ============================================================

def analyze_and_send_base():

    with cycle_lock:

        if cycle["pending"]:
            return False

        if cycle["active"]:
            return False

        if cycle["generating"]:
            return False

        cycle["generating"] = True

    try:

        candidate = choose_best_pair()

        if not candidate:

            logger.info(
                "No valid candidate found."
            )

            return False

        symbol = candidate["symbol"]
        candles = candidate["candles"]

        analysis = validate_signal(
            symbol,
            candles
        )

        if not analysis:
            return False

        return send_one_signal(
            symbol,
            candles,
            analysis,
            "BASE"
        )

    finally:

        with cycle_lock:
            cycle["generating"] = False


# ============================================================
# RECOVERY
# ============================================================

def analyze_and_send_recovery():

    with cycle_lock:

        if not cycle["active"]:
            return False

        if cycle["pending"]:
            return False

        if cycle["recovery_used"]:
            return False

        if cycle["trade_type"] != "RECOVERY":
            return False

        requested_at = cycle["recovery_requested_at"]

        if requested_at > 0:

            elapsed = (
                now_timestamp()
                - requested_at
            )

            if elapsed < RECOVERY_WAIT_SECONDS:

                logger.info(
                    "Recovery waiting %.0fs/%ss",
                    elapsed,
                    RECOVERY_WAIT_SECONDS
                )

                return False

    # Recovery may use any pair.
    # Fresh BASE protection does not block it.
    candidate = choose_best_pair()

    if not candidate:
        return False

    symbol = candidate["symbol"]
    candles = candidate["candles"]

    analysis = validate_signal(
        symbol,
        candles
    )

    if not analysis:
        return False

    return send_one_signal(
        symbol,
        candles,
        analysis,
        "RECOVERY"
    )


# ============================================================
# MT4 PAYLOAD
# ============================================================

def process_mt4_payload(payload):

    if not isinstance(payload, dict):

        return {
            "ok": False,
            "error": "Invalid JSON object"
        }

    symbol = str(
        payload.get("symbol") or ""
    ).strip().upper()

    timeframe = str(
        payload.get("timeframe")
        or payload.get("tf")
        or "M1"
    ).upper()

    candles = normalize_candles(
        payload.get("candles") or []
    )

    if not symbol:

        return {
            "ok": False,
            "error": "Missing symbol"
        }

    if timeframe != "M1":

        return {
            "ok": False,
            "error": "Only M1 is accepted"
        }

    if len(candles) < 10:

        return {
            "ok": False,
            "error": "Not enough candles"
        }

    with data_lock:

        data_store[symbol] = {
            "symbol": symbol,
            "timeframe": "M1",
            "candles": candles[-150:],
            "received_at": now_timestamp(),
        }

    logger.info(
        "MT4 DATA | %s | candles=%d | "
        "forming_excluded=yes",
        symbol,
        len(candles)
    )

    def background_signal_check():

        if not analysis_lock.acquire(
            blocking=False
        ):
            return

        try:

            with cycle_lock:

                active = cycle["active"]
                pending = cycle["pending"]

                recovery_requested = (
                    cycle["recovery_requested"]
                )

                trade_type = cycle["trade_type"]

            if pending:
                return

            if (
                active
                and trade_type == "RECOVERY"
                and recovery_requested
            ):

                analyze_and_send_recovery()
                return

            if active:
                return

            analyze_and_send_base()

        finally:
            analysis_lock.release()

    threading.Thread(
        target=background_signal_check,
        daemon=True
    ).start()

    return {
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(candles),
        "closed_candles_used": max(
            0,
            len(candles) - 1
        ),
        "forming_candle_excluded": True,
    }


# ============================================================
# HTTP SERVER
# ============================================================

class MT4Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def _send_json(self, code, data):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        if path in (
            "/",
            "/health",
            "/healthz"
        ):

            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "ZinoProSignalAI",
                    "status": "running",
                    "mode": "MT4-M1",
                }
            )

            return

        if path in (
            "/mt4status",
            "/status"
        ):

            with data_lock:

                symbols = {}

                for symbol, info in data_store.items():

                    symbols[symbol] = {
                        "candles": len(
                            info.get(
                                "candles",
                                []
                            )
                        ),
                        "age_seconds": round(
                            now_timestamp()
                            - info.get(
                                "received_at",
                                now_timestamp()
                            ),
                            1
                        ),
                    }

            with cycle_lock:

                cycle_copy = {
                    "active": cycle["active"],
                    "trade_type": cycle["trade_type"],
                    "symbol": cycle["symbol"],
                    "direction": cycle["direction"],
                    "pending": cycle["pending"],
                    "recovery_used": cycle["recovery_used"],
                    "recovery_requested":
                        cycle["recovery_requested"],
                }

            self._send_json(
                200,
                {
                    "ok": True,
                    "cycle": cycle_copy,
                    "symbols": symbols,
                }
            )

            return

        self._send_json(
            404,
            {
                "ok": False,
                "error": "Not found"
            }
        )

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        if path not in (
            "/mt4",
            "/api/mt4"
        ):

            self._send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found"
                }
            )

            return

        header_key = (
            self.headers.get("X-MT4-API-Key")
            or self.headers.get("X-API-Key")
            or ""
        ).strip()

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )
        except Exception:
            content_length = 0

        if content_length <= 0:

            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "Empty body"
                }
            )

            return

        raw = self.rfile.read(
            content_length
        )

        try:

            payload = json.loads(
                raw.decode("utf-8")
            )

        except Exception:

            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "Invalid JSON"
                }
            )

            return

        json_key = str(
            payload.get("api_key") or ""
        ).strip()

        received_key = (
            header_key
            or json_key
        )

        if MT4_API_KEY:

            if received_key != MT4_API_KEY:

                self._send_json(
                    401,
                    {
                        "ok": False,
                        "error": "Unauthorized"
                    }
                )

                return

        result = process_mt4_payload(
            payload
        )

        if result.get("ok"):

            result["accepted"] = True
            result["batch_complete"] = True

            result["batch_id"] = (
                f"{result['symbol']}_"
                f"{int(time.time())}"
            )

            self._send_json(
                200,
                result
            )

        else:

            self._send_json(
                400,
                result
            )


def start_http_server():

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        MT4Handler
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ MT4 M1 engine ready.\n"
        "🎯 Strongest setup only.\n"
        "♻️ One recovery maximum.\n"
        "🕯️ Closed-candle analysis.\n\n"
        "CORE:\n"
        "EMA 9/21 + RSI 14 + ADX/DI 14 + ATR 14\n\n"
        "SUPPORT:\n"
        "Stoch 14,3,3\n"
        "CCI 20\n"
        "Awesome Oscillator\n"
        "Momentum 10\n"
        "MACD 12,26,9\n"
        "Stoch RSI 3,3,14,14\n"
        "Williams %R 14\n"
        "Bull/Bear Power\n"
        "Ultimate 7,14,28\n"
        "MA Summary\n\n"
        "Commands:\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/analyze SYMBOL M1"
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with cycle_lock:

        current = (
            "NONE"
            if not cycle["active"]
            else
            f"{cycle['trade_type']} "
            f"{cycle['symbol']} "
            f"{cycle['direction']}"
        )

        pending = cycle["pending"]
        recovery_used = cycle["recovery_used"]
        recovery_requested = cycle["recovery_requested"]

    total_wins = (
        stats["base_win"]
        + stats["recovery_win"]
    )

    total_losses = (
        stats["base_loss"]
        + stats["recovery_loss"]
    )

    total = total_wins + total_losses

    winrate = (
        total_wins / total * 100
        if total > 0
        else 0
    )

    text = (
        "📊 *ZinoProSignalAI STATS*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 BASE WIN: {stats['base_win']}\n"
        f"🔴 BASE LOSS: {stats['base_loss']}\n"
        f"♻️ RECOVERY WIN: {stats['recovery_win']}\n"
        f"♻️ RECOVERY LOSS: {stats['recovery_loss']}\n\n"
        f"🏆 TOTAL WIN: {total_wins}\n"
        f"❌ TOTAL LOSS: {total_losses}\n"
        f"📈 WIN RATE: {winrate:.1f}%\n\n"
        f"🔄 CURRENT CYCLE: {current}\n"
        f"⏳ PENDING: {'YES' if pending else 'NO'}\n"
        f"♻️ RECOVERY USED: {'YES' if recovery_used else 'NO'}\n"
        f"🔎 RECOVERY WAITING: "
        f"{'YES' if recovery_requested else 'NO'}"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown"
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with data_lock:
        items = history[-10:]

    if not items:

        await update.message.reply_text(
            "📚 History empty."
        )

        return

    lines = [
        "📚 *ZinoProSignalAI HISTORY*",
        "━━━━━━━━━━━━━━━━━━"
    ]

    for item in reversed(items):

        icon = (
            "🟢"
            if item["direction"] == "UP"
            else
            "🔴"
        )

        lines.append(
            f"{icon} {item['symbol']} "
            f"{item['trade_type']}\n"
            f"   {item['direction']} | "
            f"{item['score']}/20 | "
            f"{item['confidence']}%\n"
            f"   Result: {item['result']}"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown"
    )


# ============================================================
# WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with cycle_lock:

        if not cycle["pending"]:

            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة حالياً."
            )

            return

        trade_type = cycle["trade_type"]
        symbol = cycle["symbol"]
        direction = cycle["direction"]

        if trade_type == "BASE":
            stats["base_win"] += 1
        else:
            stats["recovery_win"] += 1

        with data_lock:

            for item in reversed(history):

                if (
                    item["result"] == "PENDING"
                    and item["symbol"] == symbol
                    and item["trade_type"] == trade_type
                ):

                    item["result"] = "WIN"
                    break

        cycle["active"] = False
        cycle["trade_type"] = None
        cycle["symbol"] = None
        cycle["direction"] = None
        cycle["pending"] = False
        cycle["recovery_used"] = False
        cycle["recovery_requested"] = False
        cycle["recovery_requested_at"] = 0
        cycle["setup_key"] = None
        cycle["last_signal"] = None

    await update.message.reply_text(
        f"🟢 *{trade_type} WIN*\n\n"
        f"📊 {symbol} | {direction}\n\n"
        "✅ الدورة انتهت.\n"
        "🔎 البوت يبحث الآن عن أقوى BASE جديدة.",
        parse_mode="Markdown"
    )


# ============================================================
# LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with cycle_lock:

        if not cycle["pending"]:

            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة حالياً."
            )

            return

        trade_type = cycle["trade_type"]
        symbol = cycle["symbol"]
        direction = cycle["direction"]

        if trade_type == "BASE":

            stats["base_loss"] += 1

            with data_lock:

                for item in reversed(history):

                    if (
                        item["result"] == "PENDING"
                        and item["symbol"] == symbol
                        and item["trade_type"] == "BASE"
                    ):

                        item["result"] = "LOSS"
                        break

            cycle["trade_type"] = "RECOVERY"
            cycle["pending"] = False
            cycle["recovery_used"] = False
            cycle["recovery_requested"] = True
            cycle["recovery_requested_at"] = now_timestamp()
            cycle["last_signal"] = None

        else:

            stats["recovery_loss"] += 1

            with data_lock:

                for item in reversed(history):

                    if (
                        item["result"] == "PENDING"
                        and item["symbol"] == symbol
                        and item["trade_type"] == "RECOVERY"
                    ):

                        item["result"] = "LOSS"
                        break

            cycle["active"] = False
            cycle["trade_type"] = None
            cycle["symbol"] = None
            cycle["direction"] = None
            cycle["pending"] = False
            cycle["recovery_used"] = False
            cycle["recovery_requested"] = False
            cycle["recovery_requested_at"] = 0
            cycle["setup_key"] = None
            cycle["last_signal"] = None

    if trade_type == "BASE":

        await update.message.reply_text(
            "🔴 *BASE LOSS*\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {direction}\n\n"
            "♻️ Recovery 1/1 مسموحة.\n"
            "⏳ انتظار دقيقة للحركة الجديدة.\n"
            "🔍 بعدها نفحص جميع الأزواج.\n"
            "⚠️ إذا لا توجد فرصة قوية، لن نرسل Recovery.",
            parse_mode="Markdown"
        )

        await asyncio.sleep(
            RECOVERY_WAIT_SECONDS
        )

        def recovery_worker():

            if not analysis_lock.acquire(
                blocking=False
            ):
                return

            try:
                analyze_and_send_recovery()
            finally:
                analysis_lock.release()

        threading.Thread(
            target=recovery_worker,
            daemon=True
        ).start()

    else:

        await update.message.reply_text(
            "🔴 *RECOVERY LOSS*\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {direction}\n\n"
            "⛔ Recovery 1/1 انتهت.\n"
            "❌ لا توجد Recovery ثانية.\n"
            "🔎 الدورة انتهت والبحث القادم BASE جديدة.",
            parse_mode="Markdown"
        )


# ============================================================
# RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with cycle_lock:

        cycle["active"] = False
        cycle["trade_type"] = None
        cycle["symbol"] = None
        cycle["direction"] = None
        cycle["pending"] = False
        cycle["recovery_used"] = False
        cycle["recovery_requested"] = False
        cycle["recovery_requested_at"] = 0
        cycle["generating"] = False
        cycle["setup_key"] = None
        cycle["last_signal"] = None

    with data_lock:

        history.clear()

        stats["base_win"] = 0
        stats["base_loss"] = 0
        stats["recovery_win"] = 0
        stats["recovery_loss"] = 0

    recent_base_setups.clear()

    await update.message.reply_text(
        "♻️ تم Reset بالكامل.\n"
        "🎯 البوت جاهز للبحث عن أقوى BASE."
    )


# ============================================================
# MT4 STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with data_lock:

        if not data_store:

            await update.message.reply_text(
                "🔴 لا توجد بيانات MT4 حتى الآن."
            )

            return

        lines = [
            "📡 *MT4 STATUS*",
            "━━━━━━━━━━━━━━━━━━"
        ]

        for symbol, info in data_store.items():

            candles = len(
                info.get(
                    "candles",
                    []
                )
            )

            age = (
                now_timestamp()
                - info.get(
                    "received_at",
                    now_timestamp()
                )
            )

            status = (
                "🟢"
                if age <= MAX_DATA_AGE_SECONDS
                else
                "🔴"
            )

            lines.append(
                f"{status} {symbol} | M1 | "
                f"candles={candles} | "
                f"age={age:.0f}s"
            )

    with cycle_lock:

        lines.append("")
        lines.append(
            f"🔄 Cycle: "
            f"{cycle['trade_type'] or 'NONE'}"
        )
        lines.append(
            f"⏳ Pending: "
            f"{'YES' if cycle['pending'] else 'NO'}"
        )
        lines.append(
            f"♻️ Recovery waiting: "
            f"{'YES' if cycle['recovery_requested'] else 'NO'}"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown"
    )


# ============================================================
# MANUAL ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with cycle_lock:

        if cycle["pending"]:

            await update.message.reply_text(
                "⛔ توجد صفقة معلقة. استعمل /win أو /loss."
            )

            return

        if cycle["active"]:

            await update.message.reply_text(
                "⛔ توجد دورة نشطة."
            )

            return

    args = context.args

    if not args:

        await update.message.reply_text(
            "استعمل:\n/analyze EURUSD M1"
        )

        return

    symbol = args[0].upper()

    timeframe = (
        args[1].upper()
        if len(args) > 1
        else "M1"
    )

    if timeframe != "M1":

        await update.message.reply_text(
            "⛔ هذا الإصدار يعمل على M1 فقط."
        )

        return

    with data_lock:
        info = data_store.get(symbol)

    if not info:

        await update.message.reply_text(
            f"🔴 لا توجد بيانات MT4 لـ {symbol}."
        )

        return

    candles = info.get("candles", [])

    if len(candles) < REQUIRED_TOTAL_CANDLES:

        await update.message.reply_text(
            f"⚠️ {symbol}: "
            f"{len(candles)} candles.\n"
            f"المطلوب {REQUIRED_TOTAL_CANDLES}."
        )

        return

    analysis = validate_signal(
        symbol,
        candles
    )

    if not analysis:

        await update.message.reply_text(
            f"❌ {symbol} لا يملك حالياً "
            "setup مطابق للشروط.\n\n"
            f"Core: 2/3 minimum\n"
            f"Score: {MIN_SCORE}/20 minimum\n"
            "EMA + RSI + ADX/DI + ATR\n"
            "ثم confirmation indicators."
        )

        return

    sent = send_one_signal(
        symbol,
        candles,
        analysis,
        "BASE"
    )

    await update.message.reply_text(
        "✅ تم إرسال BASE واحدة فقط."
        if sent
        else
        "⚠️ لم يتم إرسال الإشارة."
    )


# ============================================================
# TELEGRAM MAIN
# ============================================================

async def telegram_main():

    global application
    global telegram_loop

    telegram_loop = asyncio.get_running_loop()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start_command)
    )

    application.add_handler(
        CommandHandler("stats", stats_command)
    )

    application.add_handler(
        CommandHandler("history", history_command)
    )

    application.add_handler(
        CommandHandler("win", win_command)
    )

    application.add_handler(
        CommandHandler("loss", loss_command)
    )

    application.add_handler(
        CommandHandler("reset", reset_command)
    )

    application.add_handler(
        CommandHandler("mt4status", mt4status_command)
    )

    application.add_handler(
        CommandHandler("analyze", analyze_command)
    )

    await application.initialize()
    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True
    )

    logger.info(
        "Telegram polling started."
    )

    while True:
        await asyncio.sleep(3600)


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_config():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing."
        )

    if OWNER_ID <= 0:
        raise RuntimeError(
            "OWNER_ID is missing or invalid."
        )

    if not MT4_API_KEY:

        logger.warning(
            "MT4_API_KEY is not configured. "
            "MT4 authentication disabled."
        )

    if not GEMINI_API_KEY:

        logger.warning(
            "GEMINI_API_KEY is not configured. "
            "Technical engine works without Gemini."
        )

    if not GEMINI_AVAILABLE:

        logger.warning(
            "google-genai is not installed. "
            "Gemini disabled."
        )


# ============================================================
# STARTUP
# ============================================================

def main():

    validate_config()

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "Strategy: "
        "EMA9/21 + RSI14 + ADX14/DI + ATR14 | "
        "Stoch14,3,3 | CCI20 | AO | Momentum10 | "
        "MACD12,26,9 | StochRSI3,3,14,14 | "
        "Williams14 | BullBear | Ultimate7,14,28 | "
        "MA confirmation"
    )

    asyncio.run(
        telegram_main()
    )


if __name__ == "__main__":
    main()
```
