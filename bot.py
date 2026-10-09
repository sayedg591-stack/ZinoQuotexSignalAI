import os
import json
import logging
import threading
import asyncio
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

try:
    OWNER_ID = int(os.getenv("OWNER_ID", "0"))
except Exception:
    OWNER_ID = 0

MT4_API_KEY = (
    os.getenv("MT4_API_KEY", "").strip()
    or os.getenv("ZINO_API_KEY", "").strip()
)

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")

TIMEFRAME = "M1"

MIN_CANDLES = 50
REQUIRED_TOTAL_CANDLES = 51

# Entry is calculated from the actual incoming timeframe:
# M1 = 1 minute, M2 = 2 minutes, M3 = 3 minutes.
RECOVERY_WAIT_SECONDS = 120

MIN_SCORE = 14
MAX_SCORE = 20

MIN_CONFIDENCE = 72
MAX_CONFIDENCE = 89

MIN_SCORE_GAP = 3

MIN_ADX = 18.0
STRONG_ADX = 25.0

MIN_CANDLE_RANGE_RATIO = 0.25
MAX_CANDLE_RANGE_RATIO = 2.8

MAX_DATA_AGE_SECONDS = 180

RECOVERY_LIMIT = 1

SETUP_REPEAT_BLOCK_SECONDS = 300

FRESH_SETUP_ATR_RATIO = 0.35

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")

# ============================================================
# GLOBAL STATE
# ============================================================

data_store = {}

data_lock = threading.RLock()
cycle_lock = threading.RLock()

telegram_app = None
telegram_loop = None

shutdown_event = threading.Event()

cycle = {
    "active": False,
    "stage": "IDLE",

    "base_symbol": None,
    "base_direction": None,
    "base_price": None,
    "base_entry_time": None,

    "recovery_count": 0,

    "pending_signal": None,

    "last_result": None,
    "recovery_ready_at": None,
}

stats = {
    "signals": 0,
    "wins": 0,
    "losses": 0,

    "base_signals": 0,
    "base_wins": 0,
    "base_losses": 0,

    "recovery_signals": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}

history = []

recent_base_setups = {}

# ============================================================
# BASIC HELPERS
# ============================================================


def now_algeria():
    return datetime.now(TIMEZONE)


def now_string():
    return now_algeria().strftime("%Y-%m-%d %H:%M:%S")


def time_string(dt):
    if not dt:
        return "--:--:--"

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TIMEZONE)

    return dt.astimezone(TIMEZONE).strftime("%H:%M:%S")


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def fmt_price(price):
    if price is None:
        return "--"

    try:
        p = float(price)

        if abs(p) >= 100:
            return f"{p:.3f}"

        if abs(p) >= 10:
            return f"{p:.4f}"

        return f"{p:.5f}"

    except Exception:
        return str(price)


def direction_emoji(direction):
    return "🟢" if direction == "UP" else "🔴"


def opposite_direction(direction):
    if direction == "UP":
        return "DOWN"

    if direction == "DOWN":
        return "UP"

    return None


def is_owner(update):
    if not update or not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


# ============================================================
# CANDLE NORMALIZATION
# ============================================================


def normalize_candle(c):
    """
    Accepts common MT4/MT5 candle formats.

    Expected fields:
        time / timestamp
        open
        high
        low
        close
        volume
    """

    if not isinstance(c, dict):
        return None

    t = (
        c.get("time")
        if c.get("time") is not None
        else c.get("timestamp")
    )

    o = safe_float(c.get("open"))
    h = safe_float(c.get("high"))
    l = safe_float(c.get("low"))
    cl = safe_float(c.get("close"))

    if o is None or h is None or l is None or cl is None:
        return None

    volume = safe_float(
        c.get("volume"),
        safe_float(c.get("tick_volume"), 0.0),
    )

    return {
        "time": t,
        "open": o,
        "high": h,
        "low": l,
        "close": cl,
        "volume": volume or 0.0,
    }


def normalize_candles(raw):
    if not isinstance(raw, list):
        return []

    result = []

    for c in raw:
        n = normalize_candle(c)

        if n:
            result.append(n)

    result.sort(
        key=lambda x: safe_float(x.get("time"), 0)
        if isinstance(x.get("time"), (int, float))
        else str(x.get("time", ""))
    )

    return result


def get_closed_candles(candles):
    """
    MT4 sends candles where the last candle can still be forming.
    We intentionally remove the last candle.
    """

    if len(candles) < 3:
        return []

    return candles[:-1]


# ============================================================
# INDICATORS
# ============================================================


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)

    first = sum(values[:period]) / period

    result = [first]

    previous = first

    for value in values[period:]:
        current = (value - previous) * multiplier + previous
        result.append(current)
        previous = current

    return result


def ema(values, period):
    series = ema_series(values, period)

    if not series:
        return None

    return series[-1]


def true_ranges(candles):
    if not candles:
        return []

    result = []

    previous_close = None

    for c in candles:
        high = c["high"]
        low = c["low"]

        if previous_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        result.append(tr)

        previous_close = c["close"]

    return result


def atr(candles, period=14):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def rsi(candles, period=14):
    if len(candles) < period + 1:
        return None

    closes = [c["close"] for c in candles]

    gains = []
    losses = []

    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]

        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1)) + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(candles, period=14):
    if len(candles) < period:
        return None, None

    window = candles[-period:]

    highest = max(c["high"] for c in window)
    lowest = min(c["low"] for c in window)

    if highest == lowest:
        return 50.0, 50.0

    k = (
        (candles[-1]["close"] - lowest)
        / (highest - lowest)
    ) * 100.0

    recent_k = []

    for i in range(max(0, len(candles) - 3), len(candles)):
        sub = candles[max(0, i - period + 1): i + 1]

        hi = max(x["high"] for x in sub)
        lo = min(x["low"] for x in sub)

        if hi == lo:
            recent_k.append(50.0)
        else:
            recent_k.append(
                ((candles[i]["close"] - lo) / (hi - lo)) * 100
            )

    d = sum(recent_k) / len(recent_k)

    return k, d


def cci(candles, period=20):
    if len(candles) < period:
        return None

    typical = [
        (c["high"] + c["low"] + c["close"]) / 3
        for c in candles
    ]

    window = typical[-period:]

    mean = sum(window) / period

    deviation = sum(abs(x - mean) for x in window) / period

    if deviation == 0:
        return 0.0

    return (typical[-1] - mean) / (0.015 * deviation)


def momentum(candles, period=10):
    if len(candles) <= period:
        return None

    return candles[-1]["close"] - candles[-1 - period]["close"]


def macd(candles):
    closes = [c["close"] for c in candles]

    fast = ema(closes, 12)
    slow = ema(closes, 26)

    if fast is None or slow is None:
        return None, None

    macd_line = fast - slow

    series_fast = ema_series(closes, 12)
    series_slow = ema_series(closes, 26)

    length = min(len(series_fast), len(series_slow))

    if length < 9:
        return macd_line, None

    macd_series = []

    for i in range(length):
        macd_series.append(
            series_fast[-length + i]
            - series_slow[-length + i]
        )

    signal = ema(macd_series, 9)

    return macd_line, signal


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    window = candles[-period:]

    highest = max(c["high"] for c in window)
    lowest = min(c["low"] for c in window)

    if highest == lowest:
        return -50.0

    return (
        (highest - candles[-1]["close"])
        / (highest - lowest)
    ) * -100.0


def ultimate_oscillator(candles):
    if len(candles) < 30:
        return None

    bp = []
    tr = []

    for i in range(1, len(candles)):
        c = candles[i]
        prev = candles[i - 1]

        buying_pressure = (
            c["close"]
            - min(c["low"], prev["close"])
        )

        true_range = (
            max(c["high"], prev["close"])
            - min(c["low"], prev["close"])
        )

        bp.append(buying_pressure)
        tr.append(true_range)

    def avg(period):
        if len(bp) < period:
            return None

        b = sum(bp[-period:])
        t = sum(tr[-period:])

        if t == 0:
            return 0

        return b / t

    a7 = avg(7)
    a14 = avg(14)
    a28 = avg(28)

    if a7 is None or a14 is None or a28 is None:
        return None

    return 100 * (
        (4 * a7) +
        (2 * a14) +
        a28
    ) / 7


def awesome_oscillator(candles):
    median = [
        (c["high"] + c["low"]) / 2
        for c in candles
    ]

    if len(median) < 34:
        return None

    fast = sum(median[-5:]) / 5
    slow = sum(median[-34:]) / 34

    return fast - slow


def adx_di(candles, period=14):
    if len(candles) < period * 2 + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        plus = (
            up_move
            if up_move > down_move and up_move > 0
            else 0
        )

        minus = (
            down_move
            if down_move > up_move and down_move > 0
            else 0
        )

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(trs) < period:
        return None, None, None

    atr_values = []
    plus_di_values = []
    minus_di_values = []

    for i in range(period, len(trs) + 1):
        tr_sum = sum(trs[i - period:i])
        plus_sum = sum(plus_dm[i - period:i])
        minus_sum = sum(minus_dm[i - period:i])

        if tr_sum == 0:
            plus_di = 0
            minus_di = 0
        else:
            plus_di = 100 * plus_sum / tr_sum
            minus_di = 100 * minus_sum / tr_sum

        dx_den = plus_di + minus_di

        if dx_den == 0:
            dx = 0
        else:
            dx = (
                abs(plus_di - minus_di)
                / dx_den
            ) * 100

        atr_values.append(dx)
        plus_di_values.append(plus_di)
        minus_di_values.append(minus_di)

    if len(atr_values) < period:
        return None, None, None

    adx = sum(atr_values[-period:]) / period

    return (
        adx,
        plus_di_values[-1],
        minus_di_values[-1],
    )


def stoch_rsi(candles, period=14):
    if len(candles) < period * 2:
        return None

    rsi_values = []

    for i in range(period, len(candles)):
        sub = candles[:i + 1]

        value = rsi(sub, period)

        if value is not None:
            rsi_values.append(value)

    if len(rsi_values) < period:
        return None

    window = rsi_values[-period:]

    low = min(window)
    high = max(window)

    if high == low:
        return 50.0

    return (
        (rsi_values[-1] - low)
        / (high - low)
    ) * 100


def bull_bear_power(candles, period=13):
    if len(candles) < period:
        return None, None

    closes = [c["close"] for c in candles]

    ema_value = ema(closes, period)

    if ema_value is None:
        return None, None

    current = candles[-1]

    bull = current["high"] - ema_value
    bear = current["low"] - ema_value

    return bull, bear


# ============================================================
# PRICE ACTION
# ============================================================


def candle_metrics(candles):
    if not candles:
        return {}

    c = candles[-1]

    body = abs(c["close"] - c["open"])

    upper_wick = c["high"] - max(
        c["open"],
        c["close"],
    )

    lower_wick = min(
        c["open"],
        c["close"],
    ) - c["low"]

    total_range = c["high"] - c["low"]

    if total_range <= 0:
        body_ratio = 0
    else:
        body_ratio = body / total_range

    return {
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "range": total_range,
        "body_ratio": body_ratio,
        "bullish": c["close"] > c["open"],
        "bearish": c["close"] < c["open"],
    }


def structure_direction(candles):
    if len(candles) < 8:
        return None

    recent = candles[-6:]

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    if (
        highs[-1] > highs[-2]
        and lows[-1] > lows[-2]
    ):
        return "UP"

    if (
        highs[-1] < highs[-2]
        and lows[-1] < lows[-2]
    ):
        return "DOWN"

    return None


# ============================================================
# ADVANCED BREAKOUT
# ============================================================


def breakout_analysis(candles):
    """
    Advanced breakout detector.

    Returns:
        direction:
            UP / DOWN / None

        strength:
            STRONG / WEAK / NONE

        retest:
            True / False

        fake:
            True / False

        level:
            breakout level

        score:
            internal breakout score
    """

    result = {
        "direction": None,
        "strength": "NONE",
        "retest": False,
        "fake": False,
        "level": None,
        "score": 0,
    }

    if len(candles) < 25:
        return result

    last = candles[-1]

    previous = candles[-21:-1]

    if len(previous) < 15:
        return result

    highest = max(c["high"] for c in previous)
    lowest = min(c["low"] for c in previous)

    result["level"] = (
        highest
        if last["close"] > highest
        else lowest
        if last["close"] < lowest
        else None
    )

    atr_value = atr(candles, 14)

    if atr_value is None or atr_value <= 0:
        atr_value = (
            sum(
                c["high"] - c["low"]
                for c in candles[-14:]
            )
            / 14
        )

    close = last["close"]
    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return result

    # --------------------------------------------------------
    # REAL UP BREAKOUT
    # --------------------------------------------------------

    if close > highest:

        result["direction"] = "UP"

        distance = close - highest

        if distance >= atr_value * 0.30:
            result["strength"] = "STRONG"
            result["score"] = 2
        else:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # REAL DOWN BREAKOUT
    # --------------------------------------------------------

    elif close < lowest:

        result["direction"] = "DOWN"

        distance = lowest - close

        if distance >= atr_value * 0.30:
            result["strength"] = "STRONG"
            result["score"] = 2
        else:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # FAKE UP BREAKOUT
    # --------------------------------------------------------

    elif last["high"] > highest and close <= highest:

        result["direction"] = "UP"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = highest
        result["score"] = -2

    # --------------------------------------------------------
    # FAKE DOWN BREAKOUT
    # --------------------------------------------------------

    elif last["low"] < lowest and close >= lowest:

        result["direction"] = "DOWN"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = lowest
        result["score"] = -2

    # --------------------------------------------------------
    # RETEST DETECTION
    # --------------------------------------------------------

    if result["direction"] in ("UP", "DOWN") and not result["fake"]:

        direction = result["direction"]
        level = result["level"]

        if level is not None:

            scan_start = max(
                2,
                len(candles) - 7,
            )

            for i in range(scan_start, len(candles) - 1):

                current = candles[i]

                local_start = max(
                    0,
                    i - 20,
                )

                local_window = candles[local_start:i]

                if len(local_window) < 10:
                    continue

                local_high = max(
                    x["high"]
                    for x in local_window
                )

                local_low = min(
                    x["low"]
                    for x in local_window
                )

                tolerance = atr_value * 0.18

                if direction == "UP":

                    if (
                        current["high"] > local_high
                        and abs(current["low"] - local_high)
                        <= tolerance
                    ):
                        result["retest"] = True
                        break

                elif direction == "DOWN":

                    if (
                        current["low"] < local_low
                        and abs(current["high"] - local_low)
                        <= tolerance
                    ):
                        result["retest"] = True
                        break

    if result["retest"]:
        result["score"] += 1

    return result


def breakout_direction(candles):
    """
    Compatibility helper.
    """

    return breakout_analysis(candles)["direction"]


def liquidity_direction(candles):
    if len(candles) < 12:
        return None

    previous = candles[-11:-1]
    last = candles[-1]

    highest = max(c["high"] for c in previous)
    lowest = min(c["low"] for c in previous)

    # Sweep above liquidity and close back below
    if (
        last["high"] > highest
        and last["close"] < highest
    ):
        return "DOWN"

    # Sweep below liquidity and close back above
    if (
        last["low"] < lowest
        and last["close"] > lowest
    ):
        return "UP"

    return None


# ============================================================
# ANALYSIS
# ============================================================


def normalize_timeframe(value, candles=None):
    """Normalize M1/M2/M3 timeframe from MT4/MT5 payload or candle spacing."""
    if value is not None:
        text = str(value).strip().upper().replace(" ", "")
        if text in {"1", "1M", "M1", "60", "60S"}:
            return "M1"
        if text in {"2", "2M", "M2", "120", "120S"}:
            return "M2"
        if text in {"3", "3M", "M3", "180", "180S"}:
            return "M3"

    if candles and len(candles) >= 3:
        times = []
        for c in candles[-20:]:
            t = safe_float(c.get("time"))
            if t is not None:
                times.append(t)
        diffs = []
        for a, b in zip(times, times[1:]):
            d = b - a
            if 30 <= d <= 600:
                diffs.append(d)
        if diffs:
            median = sorted(diffs)[len(diffs)//2]
            if 90 <= median < 150:
                return "M2"
            if 150 <= median < 240:
                return "M3"
            if 45 <= median < 90:
                return "M1"

    return TIMEFRAME


def timeframe_minutes(timeframe):
    tf = normalize_timeframe(timeframe)
    return {"M1": 1, "M2": 2, "M3": 3}.get(tf, 1)


def calculate_entry_time(timeframe, created_at=None):
    """Return an Algiers entry time based on M1/M2/M3, not a fixed delay."""
    minutes = timeframe_minutes(timeframe)
    base = now_algeria() if created_at is None else created_at
    if base.tzinfo is None:
        base = base.replace(tzinfo=TIMEZONE)
    return base.astimezone(TIMEZONE) + timedelta(minutes=minutes)


def _sign(value, eps=0.0):
    if value is None:
        return 0
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0



def bollinger_bands(candles, period=20, std_mult=2.0):
    if len(candles) < period:
        return None, None, None
    closes = [c["close"] for c in candles[-period:]]
    mid = sum(closes) / period
    variance = sum((x - mid) ** 2 for x in closes) / period
    sd = variance ** 0.5
    return mid, mid + std_mult * sd, mid - std_mult * sd


def macd_details(candles):
    closes = [c["close"] for c in candles]
    if len(closes) < 35:
        return {"direction": None, "histogram": None, "cross": None}
    ef = ema_series(closes, 12)
    es = ema_series(closes, 26)
    n = min(len(ef), len(es))
    if n < 10:
        return {"direction": None, "histogram": None, "cross": None}
    lines = [ef[-n+i] - es[-n+i] for i in range(n)]
    sig = ema_series(lines, 9)
    if not sig:
        return {"direction": None, "histogram": None, "cross": None}
    line = lines[-1]
    signal = sig[-1]
    hist = line - signal
    prev_line = lines[-2] if len(lines) >= 2 else line
    prev_sig = sig[-2] if len(sig) >= 2 else signal
    cross = "UP" if prev_line <= prev_sig and line > signal else "DOWN" if prev_line >= prev_sig and line < signal else None
    direction = "UP" if line > signal and hist > 0 else "DOWN" if line < signal and hist < 0 else None
    return {"direction": direction, "line": line, "signal": signal, "histogram": hist, "cross": cross}


def wick_pattern(candles):
    if len(candles) < 2:
        return {"direction": None, "pattern": "NONE", "strength": 0}
    c = candles[-1]
    body = abs(c["close"] - c["open"])
    rng = c["high"] - c["low"]
    if rng <= 0:
        return {"direction": None, "pattern": "NONE", "strength": 0}
    upper = c["high"] - max(c["open"], c["close"])
    lower = min(c["open"], c["close"]) - c["low"]
    body_safe = max(body, rng * 0.05)
    direction = None
    pattern = "NONE"
    strength = 0
    if lower >= body_safe * 2.0 and lower >= rng * 0.45 and c["close"] >= c["low"] + rng * 0.60:
        direction, pattern, strength = "UP", "BULLISH PIN/WICK", 2
    elif upper >= body_safe * 2.0 and upper >= rng * 0.45 and c["close"] <= c["low"] + rng * 0.40:
        direction, pattern, strength = "DOWN", "BEARISH PIN/WICK", 2
    if len(candles) >= 2:
        p = candles[-2]
        if c["close"] > c["open"] and p["close"] < p["open"] and c["close"] >= p["open"] and c["open"] <= p["close"]:
            direction, pattern, strength = "UP", "BULLISH ENGULFING", max(strength, 2)
        elif c["close"] < c["open"] and p["close"] > p["open"] and c["close"] <= p["open"] and c["open"] >= p["close"]:
            direction, pattern, strength = "DOWN", "BEARISH ENGULFING", max(strength, 2)
    return {"direction": direction, "pattern": pattern, "strength": strength, "upper_wick": upper, "lower_wick": lower, "range": rng}


def fake_breakout_details(candles):
    result = {"direction": None, "level": None, "strength": 0, "type": "NONE"}
    if len(candles) < 12:
        return result
    last = candles[-1]
    window = candles[-11:-1]
    high = max(c["high"] for c in window)
    low = min(c["low"] for c in window)
    a = atr(candles, 14) or 0
    rng = last["high"] - last["low"]
    if rng <= 0:
        return result
    upper = last["high"] - max(last["open"], last["close"])
    lower = min(last["open"], last["close"]) - last["low"]
    # Sweep support -> bullish rejection.
    if last["low"] < low and last["close"] > low and lower >= rng * 0.35:
        strength = 2 if (a <= 0 or (low-last["low"]) >= a*0.10) else 1
        result.update(direction="UP", level=low, strength=strength, type="BULLISH FAKE BREAKOUT")
    # Sweep resistance -> bearish rejection.
    elif last["high"] > high and last["close"] < high and upper >= rng * 0.35:
        strength = 2 if (a <= 0 or (last["high"]-high) >= a*0.10) else 1
        result.update(direction="DOWN", level=high, strength=strength, type="BEARISH FAKE BREAKOUT")
    return result


def news_risk_from_item(item):
    if not isinstance(item, dict):
        return False, "NONE"
    raw = item.get("news_risk", item.get("high_impact_news", item.get("news_blocked", item.get("news"))))
    if isinstance(raw, dict):
        blocked = bool(raw.get("blocked") or raw.get("high_impact") or raw.get("active"))
        level = str(raw.get("level", raw.get("risk", "HIGH" if blocked else "NONE"))).upper()
        return blocked, level
    if isinstance(raw, bool):
        return raw, "HIGH" if raw else "NONE"
    if raw is not None:
        text = str(raw).strip().upper()
        blocked = text in {"HIGH", "HIGH_IMPACT", "BLOCK", "BLOCKED", "TRUE", "1", "RED"}
        return blocked, text or "NONE"
    return False, "NONE"


def calculate_analysis(candles, timeframe="M1"):
    """Confluence analysis capped at 20 points."""
    if len(candles) < MIN_CANDLES:
        return None
    timeframe = normalize_timeframe(timeframe, candles)
    closes = [c["close"] for c in candles]
    last = candles[-1]
    ema9, ema21, ema50 = ema(closes, 9), ema(closes, 21), ema(closes, 50)
    rsi14, williams = rsi(candles, 14), williams_r(candles, 14)
    adx, plus_di, minus_di = adx_di(candles, 14)
    atr14 = atr(candles, 14)
    structure = structure_direction(candles)
    wick = wick_pattern(candles)
    fake = fake_breakout_details(candles)
    bo = breakout_analysis(candles)
    mac = macd_details(candles)
    bb_mid, bb_upper, bb_lower = bollinger_bands(candles)
    liquidity = liquidity_direction(candles)
    up = down = 0
    reasons_up, reasons_down, conflicts = [], [], []

    # Trend / EMA = 3
    trend_votes = 0
    if ema9 is not None and ema21 is not None: trend_votes += 1 if ema9 > ema21 else -1 if ema9 < ema21 else 0
    if ema21 is not None and ema50 is not None: trend_votes += 1 if ema21 > ema50 else -1 if ema21 < ema50 else 0
    if ema21 is not None: trend_votes += 1 if last["close"] > ema21 else -1 if last["close"] < ema21 else 0
    trend = "UP" if trend_votes >= 2 else "DOWN" if trend_votes <= -2 else None
    if trend == "UP": up += 3; reasons_up.append("EMA 9/21/50 trend UP")
    elif trend == "DOWN": down += 3; reasons_down.append("EMA 9/21/50 trend DOWN")

    # Structure = 3
    if structure == "UP": up += 3; reasons_up.append("HH/HL structure")
    elif structure == "DOWN": down += 3; reasons_down.append("LH/LL structure")

    # Price action / wicks = 3
    if wick["direction"] == "UP": up += min(3, wick["strength"] + (1 if fake["direction"] == "UP" else 0)); reasons_up.append(wick["pattern"])
    elif wick["direction"] == "DOWN": down += min(3, wick["strength"] + (1 if fake["direction"] == "DOWN" else 0)); reasons_down.append(wick["pattern"])

    # Momentum / ADX-DI = 3
    mom = "UP" if plus_di is not None and minus_di is not None and plus_di > minus_di else "DOWN" if plus_di is not None and minus_di is not None and minus_di > plus_di else None
    if mom == "UP" and adx is not None and adx >= MIN_ADX: up += 3 if adx >= STRONG_ADX else 2; reasons_up.append(f"ADX/DI UP {adx:.1f}")
    elif mom == "DOWN" and adx is not None and adx >= MIN_ADX: down += 3 if adx >= STRONG_ADX else 2; reasons_down.append(f"ADX/DI DOWN {adx:.1f}")

    # Fake breakout / liquidity = 3. Real breakout is secondary; fake rejection can be the trigger.
    if fake["direction"] == "UP": up += min(3, fake["strength"] + 1); reasons_up.append("Bullish fake breakout")
    elif fake["direction"] == "DOWN": down += min(3, fake["strength"] + 1); reasons_down.append("Bearish fake breakout")
    elif bo["direction"] == "UP" and bo["strength"] == "STRONG": up += 3 if bo["retest"] else 2; reasons_up.append("Strong breakout UP" + (" + retest" if bo["retest"] else ""))
    elif bo["direction"] == "DOWN" and bo["strength"] == "STRONG": down += 3 if bo["retest"] else 2; reasons_down.append("Strong breakout DOWN" + (" + retest" if bo["retest"] else ""))
    elif liquidity == "UP": up += 1; reasons_up.append("Liquidity sweep UP")
    elif liquidity == "DOWN": down += 1; reasons_down.append("Liquidity sweep DOWN")

    # MACD = 2
    if mac["direction"] == "UP": up += 2 if mac.get("cross") == "UP" else 1; reasons_up.append("MACD bullish")
    elif mac["direction"] == "DOWN": down += 2 if mac.get("cross") == "DOWN" else 1; reasons_down.append("MACD bearish")

    # Bollinger = 2. Touch alone never scores.
    bb_dir = None
    if bb_mid is not None and bb_upper is not None and bb_lower is not None:
        rng = max(bb_upper - bb_lower, 1e-12)
        pos = (last["close"] - bb_lower) / rng
        if last["low"] < bb_lower and last["close"] > bb_lower: bb_dir = "UP"
        elif last["high"] > bb_upper and last["close"] < bb_upper: bb_dir = "DOWN"
        elif trend == "UP" and pos > 0.55 and last["close"] > bb_mid: bb_dir = "UP"
        elif trend == "DOWN" and pos < 0.45 and last["close"] < bb_mid: bb_dir = "DOWN"
    if bb_dir == "UP": up += 2; reasons_up.append("Bollinger confirmation")
    elif bb_dir == "DOWN": down += 2; reasons_down.append("Bollinger confirmation")

    # RSI + Williams = 1
    osc = None
    if rsi14 is not None and williams is not None:
        if 52 <= rsi14 <= 68 and -50 < williams < -15: osc = "UP"
        elif 32 <= rsi14 <= 48 and -85 < williams < -50: osc = "DOWN"
    if osc == "UP": up += 1; reasons_up.append("RSI/W%R")
    elif osc == "DOWN": down += 1; reasons_down.append("RSI/W%R")

    up = min(20, int(up)); down = min(20, int(down))
    direction = "UP" if up > down else "DOWN" if down > up else trend or mom or wick.get("direction") or "UP"
    score = max(up, down); gap = abs(up-down)

    extension = abs(last["close"]-ema21)/atr14 if ema21 is not None and atr14 and atr14 > 0 else 0
    abnormal = False
    if len(candles) >= 21:
        avg = sum(c["high"]-c["low"] for c in candles[-21:-1]) / 20
        abnormal = avg > 0 and (last["high"]-last["low"]) > avg * MAX_CANDLE_RANGE_RATIO
    extreme = (direction == "UP" and rsi14 is not None and rsi14 >= 76) or (direction == "DOWN" and rsi14 is not None and rsi14 <= 24)
    overextended = extension > 1.35
    if trend and structure and trend != structure: conflicts.append("Trend/structure disagreement")
    if mom and trend and mom != trend and (adx or 0) >= STRONG_ADX: conflicts.append("Strong momentum against trend")
    if extreme: conflicts.append("RSI extreme")
    if overextended: conflicts.append("EMA21 overextension")
    if abnormal: conflicts.append("Abnormal candle")

    independent = sum([
        trend == direction,
        structure == direction,
        wick.get("direction") == direction,
        mom == direction and (adx or 0) >= MIN_ADX,
        fake.get("direction") == direction or (bo.get("direction") == direction and not fake.get("direction")),
        mac.get("direction") == direction,
        bb_dir == direction,
        osc == direction,
    ])
    confidence = 68 + min(12, gap * 2) + independent * 2
    if adx is not None and adx >= STRONG_ADX: confidence += 3
    if conflicts: confidence -= min(12, len(conflicts)*4)
    if extreme: confidence -= 6
    if overextended: confidence -= 5
    if abnormal: confidence -= 8
    confidence = int(clamp(confidence, MIN_CONFIDENCE, MAX_CONFIDENCE))

    primary_votes = (3 if trend == "UP" else -3 if trend == "DOWN" else 0) + (2 if structure == "UP" else -2 if structure == "DOWN" else 0) + (1 if mom == "UP" else -1 if mom == "DOWN" else 0)
    primary = "UP" if primary_votes > 0 else "DOWN" if primary_votes < 0 else None
    if primary and primary != direction: conflicts.append("Primary trend conflict")
    return {
        "direction": direction, "up_score": up, "down_score": down, "max_score": score, "score_gap": gap,
        "confidence": confidence, "primary_direction": primary, "trend_direction": trend, "momentum_direction": mom,
        "strong_confluence": independent, "ema9": ema9, "ema21": ema21, "ema50": ema50, "rsi": rsi14,
        "adx": adx, "plus_di": plus_di, "minus_di": minus_di, "atr": atr14, "williams": williams,
        "structure": structure, "breakout": bo["direction"], "breakout_strength": bo["strength"], "breakout_retest": bo["retest"],
        "breakout_fake": bool(fake["direction"]), "breakout_score": bo["score"], "breakout_level": bo["level"],
        "fake_breakout_direction": fake["direction"], "fake_breakout_type": fake["type"], "liquidity": liquidity,
        "wick_pattern": wick["pattern"], "wick_direction": wick["direction"], "macd_direction": mac["direction"],
        "macd_cross": mac.get("cross"), "macd_histogram": mac.get("histogram"), "bollinger_direction": bb_dir,
        "bollinger_mid": bb_mid, "bollinger_upper": bb_upper, "bollinger_lower": bb_lower,
        "body_ratio": candle_metrics(candles).get("body_ratio", 0), "candle_direction": "UP" if last["close"] > last["open"] else "DOWN" if last["close"] < last["open"] else None,
        "abnormal_candle": abnormal, "volatility_ok": atr14 is not None and atr14 > 0, "extreme_chase": extreme,
        "overextended": overextended, "extension_atr": extension, "conflict": bool(conflicts), "conflict_reasons": conflicts,
        "reasons_up": reasons_up, "reasons_down": reasons_down, "candle_time": last.get("time"), "price": last["close"], "timeframe": timeframe,
    }


# ============================================================
# DATA FRESHNESS
# ============================================================


def data_age_seconds(symbol):
    item = data_store.get(symbol)

    if not item:
        return None

    received = item.get("received_at")

    if not received:
        return None

    return max(
        0,
        time.time() - received,
    )


def is_data_fresh(symbol):
    age = data_age_seconds(symbol)

    if age is None:
        return False

    return age <= MAX_DATA_AGE_SECONDS


# ============================================================
# SIGNAL VALIDATION
# ============================================================


def validate_signal(analysis):
    if not analysis or analysis.get("direction") not in ("UP", "DOWN"):
        return False, "no direction"
    if analysis.get("news_blocked"):
        return False, "high impact news"
    if analysis.get("max_score", 0) < MIN_SCORE:
        return False, "score too low"
    if analysis.get("score_gap", 0) < MIN_SCORE_GAP:
        return False, "score gap too small"
    if analysis.get("abnormal_candle") or analysis.get("overextended"):
        return False, "bad price location"
    if not analysis.get("volatility_ok"):
        return False, "ATR unavailable"
    if analysis.get("extreme_chase"):
        return False, "extreme RSI chase"
    direction = analysis["direction"]
    if analysis.get("primary_direction") and analysis.get("primary_direction") != direction:
        return False, "trend conflict"
    if analysis.get("strong_confluence", 0) < 4:
        return False, "insufficient confluence"
    # A fake breakout is valid only when its rejection direction matches the signal.
    fake_dir = analysis.get("fake_breakout_direction")
    if fake_dir and fake_dir != direction:
        return False, "fake breakout against direction"
    if analysis.get("confidence", 0) < MIN_CONFIDENCE:
        return False, "confidence too low"
    return True, "valid"


# ============================================================
# FRESH SETUP CONTROL
# ============================================================


def fresh_base_setup(symbol, direction, analysis):
    key = f"{symbol}:{direction}"

    previous = recent_base_setups.get(key)

    if not previous:
        return True

    previous_time = previous.get("time", 0)

    elapsed = time.time() - previous_time

    if elapsed >= SETUP_REPEAT_BLOCK_SECONDS:
        return True

    previous_price = previous.get("price")

    current_price = analysis.get("price")

    atr_value = analysis.get("atr")

    if (
        previous_price is not None
        and current_price is not None
        and atr_value is not None
        and atr_value > 0
    ):

        price_move = abs(
            current_price
            - previous_price
        )

        if price_move >= (
            atr_value * FRESH_SETUP_ATR_RATIO
        ):
            return True

    previous_rsi = previous.get("rsi")
    current_rsi = analysis.get("rsi")

    if (
        previous_rsi is not None
        and current_rsi is not None
        and abs(current_rsi - previous_rsi) >= 5
    ):
        return True

    return False


# ============================================================
# SYMBOL EVALUATION
# ============================================================


def evaluate_symbol(symbol):
    item = data_store.get(symbol)

    if not item:
        return None

    if not is_data_fresh(symbol):
        return None

    candles = item.get("candles", [])

    if len(candles) < REQUIRED_TOTAL_CANDLES:
        return None

    closed = get_closed_candles(candles)

    if len(closed) < MIN_CANDLES:
        return None

    timeframe = normalize_timeframe(item.get("timeframe"), candles)

    analysis = calculate_analysis(
        closed,
        timeframe,
    )

    if not analysis:
        return None

    news_blocked, news_level = news_risk_from_item(item)
    analysis["news_blocked"] = news_blocked
    analysis["news_level"] = news_level

    valid, reason = validate_signal(analysis)

    if not valid:
        return None

    direction = analysis["direction"]

    return {
        "symbol": symbol,
        "analysis": analysis,
        "direction": direction,
        "validation": reason,
    }


# ============================================================
# BEST BASE PAIR
# ============================================================


def choose_best_base_pair():
    candidates = []

    with data_lock:
        symbols = list(data_store.keys())

    current_symbol = cycle.get("base_symbol")
    current_direction = cycle.get("base_direction")

    recent_symbols = [
        x.get("symbol")
        for x in history[-10:]
        if x.get("symbol")
    ]

    for symbol in symbols:

        candidate = evaluate_symbol(symbol)

        if not candidate:
            continue

        analysis = candidate["analysis"]
        direction = candidate["direction"]

        if not fresh_base_setup(
            symbol,
            direction,
            analysis,
        ):
            continue

        # Quality is driven by confluence, not by a single indicator.
        quality = (
            analysis.get("max_score", 0) * 8
            + analysis.get("score_gap", 0) * 7
        )

        adx = analysis.get("adx") or 0
        quality += min(20, adx * 0.55)

        if analysis.get("primary_direction") == direction:
            quality += 10

        if analysis.get("trend_direction") == direction:
            quality += 8

        if analysis.get("structure") == direction:
            quality += 10

        if analysis.get("strong_confluence", 0) >= 5:
            quality += 8
        elif analysis.get("strong_confluence", 0) >= 4:
            quality += 4

        if analysis.get("breakout") == direction and not analysis.get("breakout_fake"):
            quality += 5
            if analysis.get("breakout_strength") == "STRONG":
                quality += 4
            if analysis.get("breakout_retest"):
                quality += 7

        if analysis.get("breakout_fake"):
            quality -= 30

        if analysis.get("overextended"):
            quality -= 25

        if analysis.get("liquidity") == direction:
            quality += 2

        if analysis.get("body_ratio", 0) >= 0.60:
            quality += 3

        if current_symbol == symbol:
            quality -= 15

        if current_direction == direction:
            quality -= 5

        if symbol in recent_symbols:
            quality -= 4

        # Prefer stronger confidence
        quality += (
            analysis.get("confidence", 0)
            * 0.25
        )

        candidates.append(
            (
                quality,
                candidate,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# BEST RECOVERY PAIR
# ============================================================


def choose_best_recovery_pair():
    candidates = []

    with data_lock:
        symbols = list(data_store.keys())

    base_symbol = cycle.get("base_symbol")
    base_direction = cycle.get("base_direction")

    for symbol in symbols:

        candidate = evaluate_symbol(symbol)

        if not candidate:
            continue

        analysis = candidate["analysis"]
        direction = candidate["direction"]

        quality = (
            analysis["max_score"] * 10
            + analysis["score_gap"] * 5
        )

        adx = analysis.get("adx") or 0

        quality += adx

        if (
            analysis.get("primary_direction")
            == direction
        ):
            quality += 8

        if analysis.get("structure") == direction:
            quality += 5

        if (
            analysis.get("breakout") == direction
            and not analysis.get("breakout_fake")
        ):

            if analysis.get("breakout_strength") == "STRONG":
                quality += 8

            elif analysis.get("breakout_strength") == "WEAK":
                quality += 3

            if analysis.get("breakout_retest"):
                quality += 5

        if analysis.get("breakout_fake"):
            quality -= 20

        if analysis.get("liquidity") == direction:
            quality += 2

        if symbol == base_symbol:
            quality -= 18

        if direction == base_direction:
            quality -= 6
        else:
            quality += 4

        quality += (
            analysis.get("confidence", 0)
            * 0.25
        )

        candidates.append(
            (
                quality,
                candidate,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return candidates[0][1]


# ============================================================
# HISTORY
# ============================================================


def add_history(
    symbol,
    direction,
    stage,
    analysis,
    result="PENDING",
    entry_time=None,
):
    item = {
        "time": now_string(),

        "symbol": symbol,

        "direction": direction,

        "stage": stage,

        "result": result,

        "confidence": analysis.get(
            "confidence"
        ),

        "up_score": analysis.get(
            "up_score"
        ),

        "down_score": analysis.get(
            "down_score"
        ),

        "score": analysis.get(
            "max_score"
        ),

        "adx": analysis.get(
            "adx"
        ),

        "rsi": analysis.get(
            "rsi"
        ),

        "structure": analysis.get(
            "structure"
        ),

        "breakout": analysis.get(
            "breakout"
        ),

        "breakout_strength": analysis.get(
            "breakout_strength"
        ),

        "breakout_retest": analysis.get(
            "breakout_retest"
        ),

        "breakout_fake": analysis.get(
            "breakout_fake"
        ),

        "timeframe": analysis.get("timeframe", TIMEFRAME),

        "entry_time": (
            entry_time.isoformat()
            if isinstance(entry_time, datetime)
            else entry_time
        ),
    }

    history.append(item)

    if len(history) > 100:
        del history[:-100]

    return item


# ============================================================
# TELEGRAM MESSAGE
# ============================================================


def build_signal_text(
    symbol,
    direction,
    stage,
    analysis,
    entry_time,
):
    confidence = analysis.get(
        "confidence",
        0,
    )

    up_score = analysis.get(
        "up_score",
        0,
    )

    down_score = analysis.get(
        "down_score",
        0,
    )

    price = analysis.get(
        "price"
    )

    breakout = analysis.get(
        "breakout"
    )

    breakout_strength = analysis.get(
        "breakout_strength",
        "NONE",
    )

    breakout_retest = analysis.get(
        "breakout_retest",
        False,
    )

    structure = analysis.get(
        "structure"
    )

    adx = analysis.get(
        "adx"
    )

    rsi_value = analysis.get(
        "rsi"
    )

    timeframe = normalize_timeframe(
        analysis.get("timeframe", TIMEFRAME)
    )

    if breakout is None:
        breakout_text = "NONE"
    else:
        breakout_text = breakout

    if breakout_strength == "STRONG":
        quality_text = "🔥 STRONG"

    elif breakout_strength == "WEAK":
        quality_text = "🟡 WEAK"

    elif breakout_strength == "FAKE":
        quality_text = "⚠️ FAKE"

    else:
        quality_text = "⚪ NONE"

    retest_text = (
        "✅ YES"
        if breakout_retest
        else "❌ NO"
    )

    atr_value = analysis.get("atr") or 0
    cancel_buffer = atr_value * 0.35 if atr_value > 0 else 0
    cancel_price = None
    if price is not None and cancel_buffer > 0:
        if direction == "UP":
            cancel_price = price - cancel_buffer
        else:
            cancel_price = price + cancel_buffer

    stage_text = (
        "🎯 BASE TRADE"
        if stage == "BASE"
        else "♻️ RECOVERY 1/1"
    )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {analysis.get('timeframe', TIMEFRAME)}\n\n"
        f"{stage_text}\n"
        f"{direction_emoji(direction)} "
        f"**{direction}**\n\n"
        f"🔥 Confidence: **{confidence}%**\n"
        f"🟢 UP Score: **{up_score}/{MAX_SCORE}**\n"
        f"🔴 DOWN Score: **{down_score}/{MAX_SCORE}**\n\n"
        f"📐 Structure: "
        f"**{structure or 'NONE'}**\n"
        f"🚀 Breakout: "
        f"**{breakout_text}**\n"
        f"💥 Breakout Quality: "
        f"**{quality_text}**\n"
        f"🔄 Retest: "
        f"**{retest_text}**\n"
        f"🕯️ Wick: **{analysis.get('wick_pattern', 'NONE')}**\n"
        f"📉 MACD: **{analysis.get('macd_direction') or 'NONE'}**"
        f" | BB: **{analysis.get('bollinger_direction') or 'NONE'}**\n"
        f"🧪 Fake Breakout: **{analysis.get('fake_breakout_type') or 'NONE'}**\n\n"
        f"📈 ADX: "
        f"**{adx:.1f}**\n"
        if adx is not None
        else
        "📈 ADX: **--**\n"
    ) + (
        f"📊 RSI: **{rsi_value:.1f}**\n"
        if rsi_value is not None
        else
        "📊 RSI: **--**\n"
    ) + (
        "\n"
        f"💰 Price: **{fmt_price(price)}**\n"
        f"🛑 Cancellation: **{fmt_price(cancel_price)}**\n"
        f"   {'إلغاء إذا أغلقت شمعة تحت' if direction == 'UP' else 'إلغاء إذا أغلقت شمعة فوق'} **{fmt_price(cancel_price)}**\n\n"
        f"⏱️ Entry after: **{timeframe_minutes(timeframe)} minute(s)**\n"
        f"🕐 **ENTRY TIME: "
        f"{time_string(entry_time)} 🇩🇿**\n\n"
        "━━━━━━━━━━━━━━━━━━"
    )


def build_result_text(
    signal,
    result,
):
    if not signal:
        return (
            "❌ No pending signal found."
        )

    symbol = signal.get("symbol")
    direction = signal.get("direction")
    stage = signal.get("stage")

    if result == "WIN":

        if stage == "BASE":

            return (
                "✅ **BASE WIN**\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | {signal.get('analysis', {}).get('timeframe', TIMEFRAME)}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "🎯 Base signal completed successfully.\n"
                "🔎 Searching for a new BASE setup..."
            )

        return (
            "✅ **RECOVERY WIN**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {signal.get('analysis', {}).get('timeframe', TIMEFRAME)}\n"
            f"{direction_emoji(direction)} "
            f"{direction}\n\n"
            "♻️ Recovery compensated the BASE loss.\n"
            "🔎 Searching for a new BASE setup..."
        )

    if result == "LOSS":

        if stage == "BASE":

            return (
                "❌ **BASE LOSS**\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | {signal.get('analysis', {}).get('timeframe', TIMEFRAME)}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "♻️ Recovery **1/1 ALLOWED**\n"
                "⏳ Waiting 2 minutes before recovery search..."
            )

        return (
            "❌ **RECOVERY LOSS**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {signal.get('analysis', {}).get('timeframe', TIMEFRAME)}\n"
            f"{direction_emoji(direction)} "
            f"{direction}\n\n"
            "⛔ No second martingale.\n"
            "🔎 Cycle ended. Searching for a new BASE setup..."
        )

    return "Unknown result."


# ============================================================
# SEND TELEGRAM
# ============================================================


async def send_telegram_message(text_message):
    global telegram_app

    if telegram_app is None:
        logger.warning(
            "Telegram application not ready"
        )
        return False

    try:
        await telegram_app.bot.send_message(
            chat_id=OWNER_ID,
            text=text_message,
            parse_mode="Markdown",
        )

        return True

    except Exception as exc:
        logger.exception(
            "Telegram send error: %s",
            exc,
        )

        return False


def send_telegram_sync(text_message):
    global telegram_loop

    if telegram_loop is None:
        return False

    try:
        future = asyncio.run_coroutine_threadsafe(
            send_telegram_message(text_message),
            telegram_loop,
        )

        future.result(
            timeout=30
        )

        return True

    except Exception as exc:
        logger.exception(
            "Telegram sync send error: %s",
            exc,
        )

        return False


# ============================================================
# SIGNAL CREATION
# ============================================================


def send_one_signal(candidate, stage):
    global cycle

    if not candidate:
        return False

    symbol = candidate["symbol"]
    direction = candidate["direction"]
    analysis = candidate["analysis"]

    valid, reason = validate_signal(
        analysis
    )

    if not valid:
        logger.info(
            "Signal rejected %s %s: %s",
            symbol,
            direction,
            reason,
        )
        return False

    with cycle_lock:

        if stage == "BASE":

            if cycle.get("active"):
                return False

            cycle["active"] = True
            cycle["stage"] = "BASE"

            cycle["base_symbol"] = symbol
            cycle["base_direction"] = direction
            cycle["base_price"] = analysis.get(
                "price"
            )

            cycle["recovery_count"] = 0

        elif stage == "RECOVERY":

            if not cycle.get("active"):
                return False

            if cycle.get("recovery_count", 0) >= RECOVERY_LIMIT:
                return False

            cycle["stage"] = "RECOVERY"

            cycle["recovery_count"] += 1

        else:
            return False

        timeframe = normalize_timeframe(
            analysis.get("timeframe"),
        )

        entry_time = calculate_entry_time(
            timeframe,
            now_algeria(),
        )

        cycle["base_entry_time"] = entry_time

        cycle["pending_signal"] = {
            "symbol": symbol,
            "direction": direction,
            "stage": stage,
            "analysis": analysis,
            "entry_time": entry_time,
            "created_at": time.time(),
        }

    text_message = build_signal_text(
        symbol,
        direction,
        stage,
        analysis,
        entry_time,
    )

    sent = send_telegram_sync(
        text_message
    )

    if not sent:
        with cycle_lock:
            cycle["pending_signal"] = None

            if stage == "BASE":
                cycle["active"] = False
                cycle["stage"] = "IDLE"

        return False

    stats["signals"] += 1

    if stage == "BASE":
        stats["base_signals"] += 1

        recent_base_setups[
            f"{symbol}:{direction}"
        ] = {
            "time": time.time(),
            "price": analysis.get(
                "price"
            ),
            "rsi": analysis.get(
                "rsi"
            ),
        }

    else:
        stats["recovery_signals"] += 1

    add_history(
        symbol,
        direction,
        stage,
        analysis,
        "PENDING",
        entry_time,
    )

    logger.info(
        "Signal sent | %s | %s | %s | score=%s/%s confidence=%s",
        stage,
        symbol,
        direction,
        analysis.get("max_score"),
        MAX_SCORE,
        analysis.get("confidence"),
    )

    return True


# ============================================================
# RESULT MANAGEMENT
# ============================================================


def mark_history_result(
    symbol,
    direction,
    stage,
    result,
):
    for item in reversed(history):

        if (
            item.get("symbol") == symbol
            and item.get("direction") == direction
            and item.get("stage") == stage
            and item.get("result") == "PENDING"
        ):

            item["result"] = result

            return True

    return False


def handle_win():
    global cycle

    with cycle_lock:

        pending = cycle.get(
            "pending_signal"
        )

        if not pending:
            return False, "NO_PENDING"

        symbol = pending["symbol"]
        direction = pending["direction"]
        stage = pending["stage"]

        mark_history_result(
            symbol,
            direction,
            stage,
            "WIN",
        )

        stats["wins"] += 1

        if stage == "BASE":
            stats["base_wins"] += 1

        else:
            stats["recovery_wins"] += 1

        cycle["last_result"] = "WIN"

        cycle["pending_signal"] = None

        cycle["active"] = False
        cycle["stage"] = "IDLE"

        cycle["base_symbol"] = None
        cycle["base_direction"] = None
        cycle["base_price"] = None
        cycle["recovery_count"] = 0
        cycle["recovery_ready_at"] = None

    return True, "WIN"


def handle_loss():
    global cycle

    with cycle_lock:

        pending = cycle.get(
            "pending_signal"
        )

        if not pending:
            return False, "NO_PENDING"

        symbol = pending["symbol"]
        direction = pending["direction"]
        stage = pending["stage"]

        mark_history_result(
            symbol,
            direction,
            stage,
            "LOSS",
        )

        stats["losses"] += 1

        if stage == "BASE":
            stats["base_losses"] += 1

        else:
            stats["recovery_losses"] += 1

        cycle["last_result"] = "LOSS"

        cycle["pending_signal"] = None

        if stage == "BASE":

            if (
                cycle.get("recovery_count", 0)
                < RECOVERY_LIMIT
            ):

                cycle["stage"] = "WAIT_RECOVERY"

                cycle["recovery_ready_at"] = (
                    time.time()
                    + RECOVERY_WAIT_SECONDS
                )

                return True, "BASE_LOSS"

        # Recovery loss or recovery unavailable
        cycle["active"] = False
        cycle["stage"] = "IDLE"

        cycle["base_symbol"] = None
        cycle["base_direction"] = None
        cycle["base_price"] = None

        cycle["recovery_count"] = 0
        cycle["recovery_ready_at"] = None

    return True, "RECOVERY_LOSS"


# ============================================================
# AUTO SIGNAL WORKER
# ============================================================


def signal_worker():
    logger.info(
        "Signal worker started"
    )

    last_attempt = 0

    while not shutdown_event.is_set():

        try:

            current = time.time()

            if (
                current - last_attempt
                < 10
            ):
                time.sleep(1)
                continue

            last_attempt = current

            with cycle_lock:
                current_stage = cycle.get(
                    "stage"
                )

                active = cycle.get(
                    "active"
                )

                pending = cycle.get(
                    "pending_signal"
                )

                recovery_ready_at = cycle.get(
                    "recovery_ready_at"
                )

            # ------------------------------------------------
            # WAITING RECOVERY
            # ------------------------------------------------

            if (
                current_stage == "WAIT_RECOVERY"
                and recovery_ready_at is not None
            ):

                if current >= recovery_ready_at:

                    candidate = (
                        choose_best_recovery_pair()
                    )

                    if candidate:

                        sent = send_one_signal(
                            candidate,
                            "RECOVERY",
                        )

                        if sent:
                            continue

                    time.sleep(2)
                    continue

            # ------------------------------------------------
            # ACTIVE SIGNAL
            # ------------------------------------------------

            if active and pending:

                time.sleep(2)
                continue

            # ------------------------------------------------
            # NEW BASE
            # ------------------------------------------------

            if not active:

                candidate = (
                    choose_best_base_pair()
                )

                if candidate:

                    sent = send_one_signal(
                        candidate,
                        "BASE",
                    )

                    if sent:
                        continue

            time.sleep(2)

        except Exception as exc:

            logger.exception(
                "Signal worker error: %s",
                exc,
            )

            time.sleep(3)


# ============================================================
# MT4 HTTP SERVER
# ============================================================


class RequestHandler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format_string,
        *args,
    ):
        return

    def send_json(
        self,
        status,
        payload,
    ):
        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path in (
            "/",
            "/health",
            "/healthz",
        ):

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "time": now_string(),
                },
            )

            return

        if path in (
            "/mt4status",
            "/status",
        ):

            with data_lock:

                symbols = {}

                for symbol, item in data_store.items():

                    candles = item.get(
                        "candles",
                        [],
                    )

                    symbols[symbol] = {
                        "candles": len(candles),
                        "age_seconds": data_age_seconds(
                            symbol
                        ),
                    }

            with cycle_lock:

                cycle_copy = {
                    "active": cycle.get(
                        "active"
                    ),
                    "stage": cycle.get(
                        "stage"
                    ),
                    "base_symbol": cycle.get(
                        "base_symbol"
                    ),
                    "base_direction": cycle.get(
                        "base_direction"
                    ),
                    "recovery_count": cycle.get(
                        "recovery_count"
                    ),
                    "pending": bool(
                        cycle.get(
                            "pending_signal"
                        )
                    ),
                }

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "time": now_string(),
                    "symbols": symbols,
                    "cycle": cycle_copy,
                    "stats": stats,
                },
            )

            return

        self.send_json(
            404,
            {
                "error": "not found"
            },
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path not in (
            "/mt4",
            "/api/mt4",
            "/mt5",
            "/api/mt5",
        ):

            self.send_json(
                404,
                {
                    "error": "not found"
                },
            )

            return

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        header_key = (
            self.headers.get(
                "X-MT4-API-Key"
            )
            or self.headers.get(
                "X-API-Key"
            )
            or ""
        ).strip()

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

        except Exception:

            content_length = 0

        raw_body = self.rfile.read(
            content_length
        )

        try:

            payload = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:

            self.send_json(
                400,
                {
                    "error": "invalid json",
                    "details": str(exc),
                },
            )

            return

        body_key = str(
            payload.get(
                "api_key",
                ""
            )
        ).strip()

        supplied_key = (
            header_key
            or body_key
        )

        if MT4_API_KEY:

            if supplied_key != MT4_API_KEY:

                self.send_json(
                    401,
                    {
                        "error": "unauthorized"
                    },
                )

                return

        # ----------------------------------------------------
        # SYMBOL
        # ----------------------------------------------------

        symbol = (
            payload.get("symbol")
            or payload.get("Symbol")
            or payload.get("pair")
            or payload.get("Pair")
        )

        if not symbol:

            self.send_json(
                400,
                {
                    "error": "symbol missing"
                },
            )

            return

        symbol = str(
            symbol
        ).strip().upper()

        # ----------------------------------------------------
        # CANDLES
        # ----------------------------------------------------

        raw_candles = (
            payload.get("candles")
            or payload.get("data")
            or payload.get("bars")
            or []
        )

        candles = normalize_candles(
            raw_candles
        )

        if len(candles) < REQUIRED_TOTAL_CANDLES:

            self.send_json(
                400,
                {
                    "error": "not enough candles",
                    "symbol": symbol,
                    "candles": len(candles),
                    "required": REQUIRED_TOTAL_CANDLES,
                },
            )

            return

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        payload_timeframe = (
            payload.get("timeframe")
            or payload.get("Timeframe")
            or payload.get("tf")
            or payload.get("TF")
        )
        detected_timeframe = normalize_timeframe(
            payload_timeframe,
            candles,
        )

        with data_lock:

            data_store[symbol] = {
                "candles": candles,
                "received_at": time.time(),
                "updated": now_string(),
                "timeframe": detected_timeframe,
                "news_risk": payload.get("news_risk", payload.get("high_impact_news", payload.get("news_blocked", payload.get("news")))),
                "news_minutes_to": payload.get("news_minutes_to"),
            }

        batch_id = (
            f"{symbol}_"
            f"{int(time.time())}"
        )

        logger.info(
            "MT data accepted | %s | %s | candles=%s",
            symbol,
            detected_timeframe,
            len(candles),
        )

        self.send_json(
            200,
            {
                "status": "accepted",
                "symbol": symbol,
                "candles": len(candles),
                "timeframe": detected_timeframe,
                "batch_id": batch_id,
                "batch_complete": True,
            },
        )


# ============================================================
# HTTP SERVER
# ============================================================


def start_http_server():
    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT,
        ),
        RequestHandler,
    )

    logger.info(
        "HTTP server started on port %s",
        PORT,
    )

    try:
        server.serve_forever()

    except Exception as exc:

        logger.exception(
            "HTTP server error: %s",
            exc,
        )

    finally:

        server.server_close()


# ============================================================
# TELEGRAM COMMANDS
# ============================================================


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Bot is online.\n\n"
        "📡 MT4/MT5 data feed: active\n"
        "📊 Timeframe: M1 / M2 / M3\n"
        "🎯 Automatic signals: ON\n"
        "♻️ Recovery: 1/1\n\n"
        "Commands:\n"
        "/stats\n"
        "/history\n"
        "/mt4status\n"
        "/analyze SYMBOL\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/breakoutstats"
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    total = stats["signals"]
    wins = stats["wins"]
    losses = stats["losses"]

    if wins + losses > 0:
        winrate = (
            wins
            / (wins + losses)
        ) * 100
    else:
        winrate = 0

    with cycle_lock:

        stage = cycle.get(
            "stage"
        )

        active = cycle.get(
            "active"
        )

        recovery_count = cycle.get(
            "recovery_count"
        )

    text_message = (
        "📊 **ZinoProSignalAI STATS**\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📡 Signals: **{total}**\n"
        f"✅ Wins: **{wins}**\n"
        f"❌ Losses: **{losses}**\n"
        f"🎯 Winrate: **{winrate:.1f}%**\n\n"
        f"🟢 BASE Signals: "
        f"**{stats['base_signals']}**\n"
        f"🟢 BASE Wins: "
        f"**{stats['base_wins']}**\n"
        f"🔴 BASE Losses: "
        f"**{stats['base_losses']}**\n\n"
        f"♻️ Recovery Signals: "
        f"**{stats['recovery_signals']}**\n"
        f"♻️ Recovery Wins: "
        f"**{stats['recovery_wins']}**\n"
        f"♻️ Recovery Losses: "
        f"**{stats['recovery_losses']}**\n\n"
        f"⚙️ Cycle: "
        f"**{stage}**\n"
        f"Active: **{active}**\n"
        f"Recovery count: "
        f"**{recovery_count}/1**"
    )

    await update.message.reply_text(
        text_message,
        parse_mode="Markdown",
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not history:

        await update.message.reply_text(
            "📚 History is empty."
        )

        return

    lines = [
        "📚 **ZinoProSignalAI HISTORY**",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in history[-15:][::-1]:

        result = item.get(
            "result",
            "PENDING",
        )

        if result == "WIN":
            icon = "🟢"
        elif result == "LOSS":
            icon = "🔴"
        else:
            icon = "🟡"

        breakout = item.get(
            "breakout"
        )

        strength = item.get(
            "breakout_strength",
            "NONE",
        )

        retest = (
            "Y"
            if item.get(
                "breakout_retest"
            )
            else "N"
        )

        lines.append(
            f"{icon} "
            f"{item.get('symbol')} "
            f"{item.get('stage')}\n"
            f"   {item.get('direction')} "
            f"| {result}\n"
            f"   Score "
            f"{item.get('score')}/{MAX_SCORE} "
            f"| Conf "
            f"{item.get('confidence')}%\n"
            f"   Breakout: "
            f"{breakout or 'NONE'} "
            f"{strength} "
            f"| Retest {retest}\n"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )


async def breakoutstats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    categories = {
        "UP": {
            "wins": 0,
            "losses": 0,
        },
        "DOWN": {
            "wins": 0,
            "losses": 0,
        },
        "NONE": {
            "wins": 0,
            "losses": 0,
        },
        "FAKE": {
            "wins": 0,
            "losses": 0,
        },
    }

    for item in history:

        result = item.get(
            "result"
        )

        if result not in (
            "WIN",
            "LOSS",
        ):
            continue

        if item.get(
            "breakout_fake"
        ):
            category = "FAKE"

        else:
            category = (
                item.get(
                    "breakout"
                )
                or "NONE"
            )

            if category not in categories:
                category = "NONE"

        if result == "WIN":
            categories[category]["wins"] += 1

        else:
            categories[category]["losses"] += 1

    lines = [
        "🚀 **BREAKOUT STATISTICS**",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for category, values in categories.items():

        wins = values["wins"]
        losses = values["losses"]

        total = wins + losses

        rate = (
            (wins / total) * 100
            if total > 0
            else 0
        )

        lines.append(
            f"\n**{category}**\n"
            f"🟢 Wins: {wins}\n"
            f"🔴 Losses: {losses}\n"
            f"🎯 Winrate: {rate:.1f}%"
        )

    lines.append(
        "\n📌 كلما زاد عدد الإشارات، "
        "الإحصائية تولي أكثر فائدة."
    )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    with data_lock:

        if not data_store:

            await update.message.reply_text(
                "📡 MT4 STATUS\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "❌ No MT4 data received."
            )

            return

        lines = [
            "📡 **MT4 STATUS**",
            "━━━━━━━━━━━━━━━━━━",
        ]

        for symbol, item in data_store.items():

            candles = len(
                item.get(
                    "candles",
                    []
                )
            )

            age = data_age_seconds(
                symbol
            )

            if age is None:
                age_text = "--"
            else:
                age_text = (
                    f"{age:.0f}s"
                )

            lines.append(
                f"📊 {symbol} "
                f"| candles={candles} "
                f"| TF={item.get('timeframe', TIMEFRAME)} "
                f"| age={age_text}"
            )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not context.args:

        await update.message.reply_text(
            "Usage:\n"
            "/analyze EURUSD"
        )

        return

    symbol = (
        context.args[0]
        .strip()
        .upper()
    )

    with data_lock:
        item = data_store.get(
            symbol
        )

    if not item:

        await update.message.reply_text(
            f"❌ No data for {symbol}"
        )

        return

    candles = get_closed_candles(
        item.get(
            "candles",
            []
        )
    )

    analysis = calculate_analysis(
        candles,
        normalize_timeframe(item.get("timeframe"), item.get("candles", [])),
    )

    if not analysis:

        await update.message.reply_text(
            "❌ Not enough data."
        )

        return

    direction = (
        analysis.get("direction")
        or "NONE"
    )

    breakout = (
        analysis.get("breakout")
        or "NONE"
    )

    strength = analysis.get(
        "breakout_strength",
        "NONE",
    )

    retest = (
        "YES"
        if analysis.get(
            "breakout_retest"
        )
        else "NO"
    )

    fake = (
        "YES"
        if analysis.get(
            "breakout_fake"
        )
        else "NO"
    )

    valid, reason = validate_signal(
        analysis
    )

    text_message = (
        "🔬 **ANALYSIS**\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {analysis.get('timeframe', TIMEFRAME)}\n\n"
        f"🎯 Direction: **{direction}**\n"
        f"🟢 UP: "
        f"**{analysis['up_score']}/{MAX_SCORE}**\n"
        f"🔴 DOWN: "
        f"**{analysis['down_score']}/{MAX_SCORE}**\n"
        f"🔥 Confidence: "
        f"**{analysis['confidence']}%**\n\n"
        f"📐 Structure: "
        f"**{analysis.get('structure') or 'NONE'}**\n"
        f"🚀 Breakout: **{breakout}**\n"
        f"💥 Quality: **{strength}**\n"
        f"🔄 Retest: **{retest}**\n"
        f"⚠️ Fake: **{fake}**\n\n"
        f"📈 ADX: "
        f"**{analysis.get('adx'):.1f}**\n"
        if analysis.get("adx") is not None
        else
        "📈 ADX: **--**\n"
    ) + (
        f"📊 RSI: "
        f"**{analysis.get('rsi'):.1f}**\n\n"
        if analysis.get("rsi") is not None
        else
        "📊 RSI: **--**\n\n"
    ) + (
        f"✅ Validation: **{valid}**\n"
        f"Reason: `{reason}`"
    )

    await update.message.reply_text(
        text_message,
        parse_mode="Markdown",
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    with cycle_lock:

        pending = cycle.get(
            "pending_signal"
        )

    if not pending:

        await update.message.reply_text(
            "⚠️ No pending signal."
        )

        return

    stage = pending["stage"]

    success, result = handle_win()

    if not success:

        await update.message.reply_text(
            "⚠️ Could not register WIN."
        )

        return

    text_message = build_result_text(
        pending,
        "WIN",
    )

    await update.message.reply_text(
        text_message,
        parse_mode="Markdown",
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    with cycle_lock:

        pending = cycle.get(
            "pending_signal"
        )

    if not pending:

        await update.message.reply_text(
            "⚠️ No pending signal."
        )

        return

    success, result = handle_loss()

    if not success:

        await update.message.reply_text(
            "⚠️ Could not register LOSS."
        )

        return

    text_message = build_result_text(
        pending,
        "LOSS",
    )

    await update.message.reply_text(
        text_message,
        parse_mode="Markdown",
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    global cycle

    with cycle_lock:

        cycle = {
            "active": False,
            "stage": "IDLE",

            "base_symbol": None,
            "base_direction": None,
            "base_price": None,
            "base_entry_time": None,

            "recovery_count": 0,

            "pending_signal": None,

            "last_result": None,
            "recovery_ready_at": None,
        }

    stats.update(
        {
            "signals": 0,
            "wins": 0,
            "losses": 0,

            "base_signals": 0,
            "base_wins": 0,
            "base_losses": 0,

            "recovery_signals": 0,
            "recovery_wins": 0,
            "recovery_losses": 0,
        }
    )

    history.clear()
    recent_base_setups.clear()

    await update.message.reply_text(
        "♻️ **ZinoProSignalAI RESET**\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Cycle reset\n"
        "✅ Statistics reset\n"
        "✅ History reset\n"
        "✅ Recovery reset\n"
        "🔎 Ready for a new BASE setup.",
        parse_mode="Markdown",
    )


# ============================================================
# ERROR HANDLER
# ============================================================


async def error_handler(
    update,
    context,
):

    logger.exception(
        "Telegram error: %s",
        context.error,
    )


# ============================================================
# TELEGRAM APPLICATION
# ============================================================


async def run_telegram():
    global telegram_app
    global telegram_loop

    telegram_loop = asyncio.get_running_loop()

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "stats",
            stats_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "history",
            history_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "breakoutstats",
            breakoutstats_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "mt4status",
            mt4status_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "status",
            mt4status_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "analyze",
            analyze_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "win",
            win_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "loss",
            loss_command,
        )
    )

    telegram_app.add_handler(
        CommandHandler(
            "reset",
            reset_command,
        )
    )

    telegram_app.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram bot starting..."
    )

    await telegram_app.initialize()

    await telegram_app.start()

    await telegram_app.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=["message"],
    )

    logger.info(
        "Telegram polling started"
    )

    while not shutdown_event.is_set():

        await asyncio.sleep(1)

    await telegram_app.updater.stop()

    await telegram_app.stop()

    await telegram_app.shutdown()


# ============================================================
# STARTUP
# ============================================================


def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "TIMEFRAME=%s",
        TIMEFRAME,
    )

    logger.info(
        "MIN_SCORE=%s/%s",
        MIN_SCORE,
        MAX_SCORE,
    )

    logger.info(
        "ENTRY_DELAY=M1:60s M2:120s M3:180s",
    )

    logger.info(
        "RECOVERY_LIMIT=%s",
        RECOVERY_LIMIT,
    )

    if not BOT_TOKEN:

        logger.error(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:

        logger.warning(
            "OWNER_ID is missing or 0"
        )

    if not MT4_API_KEY:

        logger.warning(
            "MT4_API_KEY is missing"
        )

    # --------------------------------------------------------
    # HTTP SERVER
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        name="HTTPServer",
        daemon=True,
    )

    http_thread.start()

    # --------------------------------------------------------
    # SIGNAL WORKER
    # --------------------------------------------------------

    worker_thread = threading.Thread(
        target=signal_worker,
        name="SignalWorker",
        daemon=True,
    )

    worker_thread.start()

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    try:

        asyncio.run(
            run_telegram()
        )

    except KeyboardInterrupt:

        logger.info(
            "Shutdown requested"
        )

    except Exception as exc:

        logger.exception(
            "Fatal Telegram error: %s",
            exc,
        )

    finally:

        shutdown_event.set()

        logger.info(
            "ZinoProSignalAI stopped"
        )


if __name__ == "__main__":
    main()
