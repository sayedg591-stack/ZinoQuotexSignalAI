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

MIN_SCORE = 13
MAX_SCORE = 20

MIN_CONFIDENCE = 70
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


def calculate_analysis(candles, timeframe="M1"):
    """
    Accuracy-focused analysis.

    The 20 points are deliberately balanced so one indicator cannot dominate:
      Trend/EMA      3
      Structure      3
      Price action   3
      Momentum/ADX   3
      Breakout       3
      RSI/W%R        2
      Liquidity      1
      Context        2
    """
    if len(candles) < MIN_CANDLES:
        return None

    timeframe = normalize_timeframe(timeframe, candles)
    closes = [c["close"] for c in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)
    rsi14 = rsi(candles, 14)
    adx, plus_di, minus_di = adx_di(candles, 14)
    atr14 = atr(candles, 14)
    williams = williams_r(candles, 14)
    structure = structure_direction(candles)
    breakout_info = breakout_analysis(candles)
    liquidity = liquidity_direction(candles)
    metrics = candle_metrics(candles)
    last = candles[-1]

    breakout = breakout_info["direction"]
    breakout_strength = breakout_info["strength"]
    breakout_retest = breakout_info["retest"]
    breakout_fake = breakout_info["fake"]
    breakout_level = breakout_info["level"]
    breakout_score = breakout_info["score"]

    up_score = 0
    down_score = 0
    reasons_up = []
    reasons_down = []
    conflict_reasons = []

    # -------------------------
    # 1) TREND / EMA (3)
    # -------------------------
    trend_direction = None
    trend_up = 0
    trend_down = 0

    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            trend_up += 1
        elif ema9 < ema21:
            trend_down += 1

    if ema21 is not None and ema50 is not None:
        if ema21 > ema50:
            trend_up += 1
        elif ema21 < ema50:
            trend_down += 1

    if ema21 is not None:
        if last["close"] > ema21:
            trend_up += 1
        elif last["close"] < ema21:
            trend_down += 1

    if trend_up > trend_down:
        trend_direction = "UP"
        up_score += min(3, trend_up)
        reasons_up.append("EMA trend UP")
    elif trend_down > trend_up:
        trend_direction = "DOWN"
        down_score += min(3, trend_down)
        reasons_down.append("EMA trend DOWN")

    # -------------------------
    # 2) STRUCTURE (3)
    # -------------------------
    structure_score = 0
    if len(candles) >= 6:
        a, b, c = candles[-3:]
        higher_highs = c["high"] > b["high"] > a["high"]
        higher_lows = c["low"] > b["low"] > a["low"]
        lower_highs = c["high"] < b["high"] < a["high"]
        lower_lows = c["low"] < b["low"] < a["low"]
        close_up = c["close"] > b["close"] > a["close"]
        close_down = c["close"] < b["close"] < a["close"]

        if higher_highs and higher_lows:
            structure = "UP"
            structure_score = 3 if close_up else 2
        elif lower_highs and lower_lows:
            structure = "DOWN"
            structure_score = 3 if close_down else 2

    if structure == "UP":
        up_score += structure_score
        reasons_up.append("3-candle structure UP")
    elif structure == "DOWN":
        down_score += structure_score
        reasons_down.append("3-candle structure DOWN")

    # -------------------------
    # 3) PRICE ACTION (3)
    # -------------------------
    body_ratio = metrics.get("body_ratio", 0)
    candle_dir = "UP" if metrics.get("bullish") else "DOWN" if metrics.get("bearish") else None
    pa_score = 0

    if candle_dir:
        if body_ratio >= 0.60:
            pa_score += 2
        elif body_ratio >= 0.45:
            pa_score += 1

        previous = candles[-2]
        previous_dir = "UP" if previous["close"] > previous["open"] else "DOWN" if previous["close"] < previous["open"] else None
        if previous_dir == candle_dir:
            pa_score += 1

        if candle_dir == "UP":
            up_score += pa_score
            reasons_up.append("Price action UP")
        else:
            down_score += pa_score
            reasons_down.append("Price action DOWN")

    # -------------------------
    # 4) MOMENTUM / ADX / DI (3)
    # -------------------------
    momentum_score = 0
    momentum_direction = None

    if plus_di is not None and minus_di is not None:
        if plus_di > minus_di:
            momentum_direction = "UP"
        elif minus_di > plus_di:
            momentum_direction = "DOWN"

    if adx is not None and momentum_direction:
        if adx >= STRONG_ADX:
            momentum_score += 2
        elif adx >= MIN_ADX:
            momentum_score += 1

        if momentum_direction == "UP":
            momentum_score += 1 if plus_di > minus_di + 2 else 0
            up_score += momentum_score
            reasons_up.append(f"Momentum UP ADX {adx:.1f}")
        else:
            momentum_score += 1 if minus_di > plus_di + 2 else 0
            down_score += momentum_score
            reasons_down.append(f"Momentum DOWN ADX {adx:.1f}")

    # -------------------------
    # 5) BREAKOUT / RETEST (3)
    # -------------------------
    if breakout_fake:
        if breakout == "UP":
            down_score += 2
            reasons_down.append("Failed UP breakout")
        elif breakout == "DOWN":
            up_score += 2
            reasons_up.append("Failed DOWN breakout")
    elif breakout == "UP":
        add = 2 if breakout_strength == "STRONG" else 1
        if breakout_retest:
            add += 1
        up_score += min(3, add)
        reasons_up.append("Breakout UP" + (" + retest" if breakout_retest else ""))
    elif breakout == "DOWN":
        add = 2 if breakout_strength == "STRONG" else 1
        if breakout_retest:
            add += 1
        down_score += min(3, add)
        reasons_down.append("Breakout DOWN" + (" + retest" if breakout_retest else ""))

    # -------------------------
    # 6) RSI + WILLIAMS %R (2)
    # -------------------------
    osc_up = 0
    osc_down = 0

    # Do not chase extreme RSI. In a trend, mid-zone confirmation is safer.
    if rsi14 is not None:
        if 52 <= rsi14 <= 68:
            osc_up += 1
        elif 32 <= rsi14 <= 48:
            osc_down += 1

    if williams is not None:
        if -50 < williams < -15:
            osc_up += 1
        elif -85 < williams < -50:
            osc_down += 1

    if osc_up > osc_down:
        up_score += min(2, osc_up)
        reasons_up.append("RSI/W%R confirm UP")
    elif osc_down > osc_up:
        down_score += min(2, osc_down)
        reasons_down.append("RSI/W%R confirm DOWN")

    # -------------------------
    # 7) LIQUIDITY (1)
    # -------------------------
    if liquidity == "UP":
        up_score += 1
        reasons_up.append("Liquidity sweep UP")
    elif liquidity == "DOWN":
        down_score += 1
        reasons_down.append("Liquidity sweep DOWN")

    # -------------------------
    # 8) CONTEXT / EXTENSION (2)
    # -------------------------
    extension = 0.0
    if atr14 and atr14 > 0 and ema9 is not None:
        extension = abs(last["close"] - ema9) / atr14

    # One point for healthy location relative to EMA9; one for trend agreement.
    context_up = 0
    context_down = 0
    if ema9 is not None:
        if last["close"] > ema9 and extension <= 1.15:
            context_up += 1
        elif last["close"] < ema9 and extension <= 1.15:
            context_down += 1

    if trend_direction == "UP" and structure == "UP":
        context_up += 1
    elif trend_direction == "DOWN" and structure == "DOWN":
        context_down += 1

    up_score += min(2, context_up)
    down_score += min(2, context_down)

    # -------------------------
    # Symmetric conflict / anti-chase logic
    # -------------------------
    up_score = clamp(int(round(up_score)), 0, MAX_SCORE)
    down_score = clamp(int(round(down_score)), 0, MAX_SCORE)

    if up_score > down_score:
        direction = "UP"
    elif down_score > up_score:
        direction = "DOWN"
    else:
        # User wants an actual direction, never WAIT/NEUTRAL.
        # Use the strongest structural/trend side only when scores tie.
        direction = trend_direction or structure or candle_dir or "UP"

    max_score = max(up_score, down_score)
    score_gap = abs(up_score - down_score)

    # Primary direction is intentionally price/trend based, not RSI based.
    primary_votes = 0
    if trend_direction == "UP":
        primary_votes += 2
    elif trend_direction == "DOWN":
        primary_votes -= 2

    if structure == "UP":
        primary_votes += 2
    elif structure == "DOWN":
        primary_votes -= 2

    if momentum_direction == "UP" and adx is not None and adx >= MIN_ADX:
        primary_votes += 1
    elif momentum_direction == "DOWN" and adx is not None and adx >= MIN_ADX:
        primary_votes -= 1

    if candle_dir == "UP":
        primary_votes += 1
    elif candle_dir == "DOWN":
        primary_votes -= 1

    primary_direction = "UP" if primary_votes > 0 else "DOWN" if primary_votes < 0 else None

    if primary_direction and direction != primary_direction:
        conflict_reasons.append("Trend/structure conflict")

    if trend_direction and structure and trend_direction != structure:
        conflict_reasons.append("EMA structure disagreement")

    if momentum_direction and trend_direction and momentum_direction != trend_direction and (adx or 0) >= STRONG_ADX:
        conflict_reasons.append("Strong momentum against trend")

    # Two previous closed candles should not strongly oppose the selected side.
    last3 = candles[-3:]
    if direction == "UP":
        same_dir = sum(1 for c in last3 if c["close"] > c["open"])
        opposite_dir = sum(1 for c in last3 if c["close"] < c["open"])
    else:
        same_dir = sum(1 for c in last3 if c["close"] < c["open"])
        opposite_dir = sum(1 for c in last3 if c["close"] > c["open"])
    if opposite_dir >= 2 and same_dir <= 1:
        conflict_reasons.append("Recent candles oppose direction")

    abnormal_candle = False
    average_range = None
    if len(candles) >= 21:
        recent_ranges = [c["high"] - c["low"] for c in candles[-21:-1]]
        average_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else None
        current_range = last["high"] - last["low"]
        if average_range and average_range > 0:
            ratio = current_range / average_range
            if ratio > MAX_CANDLE_RANGE_RATIO:
                abnormal_candle = True
                conflict_reasons.append("Abnormal candle")
            elif ratio < MIN_CANDLE_RANGE_RATIO:
                conflict_reasons.append("Very low volatility")

    volatility_ok = atr14 is not None and atr14 > 0
    extreme_chase = (
        (direction == "UP" and rsi14 is not None and rsi14 >= 76)
        or (direction == "DOWN" and rsi14 is not None and rsi14 <= 24)
    )
    if extreme_chase:
        conflict_reasons.append("RSI extreme chase")

    overextended = extension > 1.35 if atr14 and atr14 > 0 else False
    if overextended:
        conflict_reasons.append("Price overextended from EMA9")

    # Confidence is calibrated from independent confluence, not score alone.
    confidence = 64
    confidence += min(8, score_gap * 2)
    if trend_direction == direction:
        confidence += 4
    if structure == direction:
        confidence += 5
    if momentum_direction == direction and (adx or 0) >= STRONG_ADX:
        confidence += 4
    elif momentum_direction == direction and (adx or 0) >= MIN_ADX:
        confidence += 2
    if breakout == direction and not breakout_fake:
        confidence += 2 if breakout_strength == "STRONG" else 1
        if breakout_retest:
            confidence += 3
    if candle_dir == direction and body_ratio >= 0.60:
        confidence += 2
    if osc_up == 2 or osc_down == 2:
        confidence += 2
    if conflict_reasons:
        confidence -= min(12, len(conflict_reasons) * 4)
    if extreme_chase:
        confidence -= 6
    if overextended:
        confidence -= 5
    if abnormal_candle:
        confidence -= 8

    confidence = int(clamp(confidence, MIN_CONFIDENCE, MAX_CONFIDENCE))

    # Quality flags used by validation/selection.
    strong_confluence = sum([
        trend_direction == direction,
        structure == direction,
        momentum_direction == direction and (adx or 0) >= MIN_ADX,
        breakout == direction and not breakout_fake,
        candle_dir == direction,
        (osc_up >= 1 if direction == "UP" else osc_down >= 1),
    ])

    return {
        "direction": direction,
        "up_score": up_score,
        "down_score": down_score,
        "max_score": max_score,
        "score_gap": score_gap,
        "confidence": confidence,
        "primary_direction": primary_direction,
        "trend_direction": trend_direction,
        "momentum_direction": momentum_direction,
        "strong_confluence": strong_confluence,
        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "rsi": rsi14,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "atr": atr14,
        "williams": williams,
        "structure": structure,
        "breakout": breakout,
        "breakout_strength": breakout_strength,
        "breakout_retest": breakout_retest,
        "breakout_fake": breakout_fake,
        "breakout_score": breakout_score,
        "breakout_level": breakout_level,
        "liquidity": liquidity,
        "body_ratio": body_ratio,
        "candle_direction": candle_dir,
        "abnormal_candle": abnormal_candle,
        "volatility_ok": volatility_ok,
        "extreme_chase": extreme_chase,
        "overextended": overextended,
        "extension_atr": extension,
        "conflict": bool(conflict_reasons),
        "conflict_reasons": conflict_reasons,
        "reasons_up": reasons_up,
        "reasons_down": reasons_down,
        "candle_time": last.get("time"),
        "price": last["close"],
        "timeframe": timeframe,
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
    if not analysis:
        return False, "no analysis"

    direction = analysis.get("direction")
    if direction not in ("UP", "DOWN"):
        return False, "no direction"

    if analysis.get("max_score", 0) < MIN_SCORE:
        return False, "score too low"

    if analysis.get("score_gap", 0) < MIN_SCORE_GAP:
        return False, "score gap too small"

    if analysis.get("primary_direction") != direction:
        return False, "trend/structure conflict"

    if analysis.get("abnormal_candle"):
        return False, "abnormal candle"

    if analysis.get("overextended"):
        return False, "price overextended"

    if not analysis.get("volatility_ok"):
        return False, "volatility unavailable"

    adx = analysis.get("adx")
    if adx is None or adx < MIN_ADX:
        return False, "ADX too weak"

    if analysis.get("extreme_chase"):
        return False, "extreme RSI chase"

    if analysis.get("breakout_fake"):
        return False, "fake breakout"

    # Precision gate: either clean structure or a strong confirmed breakout.
    structure_ok = analysis.get("structure") == direction
    breakout_ok = (
        analysis.get("breakout") == direction
        and analysis.get("breakout_strength") == "STRONG"
        and analysis.get("breakout_retest")
    )
    if not structure_ok and not breakout_ok:
        return False, "no structural confirmation"

    if analysis.get("strong_confluence", 0) < 4:
        return False, "insufficient confluence"

    # Never let a marginal 13/20 signal through with a tiny directional edge.
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
        f"**{retest_text}**\n\n"
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
