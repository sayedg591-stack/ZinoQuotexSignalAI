import os
import json
import time
import math
import logging
import threading
import asyncio

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)


# ============================================================
# ZinoProSignalAI - MT4 VERSION
# ============================================================

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

OWNER_ID = int(os.getenv("OWNER_ID", "0"))

MT4_API_KEY = (
    os.getenv("MT4_API_KEY", "").strip()
    or os.getenv("ZINO_API_KEY", "").strip()
)

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")

TIMEFRAME = "M1"

# MT4 sends candles including the currently forming candle.
# We remove the last candle before analysis.
MIN_CANDLES = 50
REQUIRED_TOTAL_CANDLES = 51

# Signal timing
ENTRY_DELAY_SECONDS = 120
RECOVERY_WAIT_SECONDS = 120

# Quality filters
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

# One recovery only
RECOVERY_LIMIT = 1

# BASE repeat protection
SETUP_REPEAT_BLOCK_SECONDS = 300

# A new setup on same pair/direction should have meaningful movement
FRESH_SETUP_ATR_RATIO = 0.35


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBAL STATE
# ============================================================

data_store = {}

data_lock = threading.RLock()
analysis_lock = threading.Lock()

telegram_loop = None
telegram_app = None

http_server = None


cycle = {
    "active": False,

    "trade_type": None,          # BASE / RECOVERY
    "symbol": None,
    "direction": None,

    "pending": False,

    "recovery_used": False,
    "recovery_requested": False,
    "recovery_requested_at": None,

    "generating": False,

    "setup_key": None,

    "last_signal_time": None,

    "base_symbol": None,
    "base_direction": None,

    "last_signal": None,
}


stats = {
    "base_win": 0,
    "base_loss": 0,

    "recovery_win": 0,
    "recovery_loss": 0,

    "signals": 0,
}


history = []

# Last BASE setup per symbol
recent_base_setups = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def now_local():
    return datetime.now(TIMEZONE)


def now_ts():
    return time.time()


def fmt_time(dt):
    return dt.astimezone(TIMEZONE).strftime("%H:%M:%S")


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def sign(value, tolerance=0.0):
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def normalize_direction(direction):
    if not direction:
        return None

    d = str(direction).upper().strip()

    if d in ("UP", "BUY", "CALL"):
        return "UP"

    if d in ("DOWN", "SELL", "PUT"):
        return "DOWN"

    return None


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):
    if not isinstance(c, dict):
        return None

    try:
        timestamp = (
            c.get("time")
            or c.get("timestamp")
            or c.get("datetime")
            or c.get("date")
        )

        if timestamp is None:
            timestamp = time.time()

        if isinstance(timestamp, str):
            try:
                timestamp = float(timestamp)
            except Exception:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp.replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    timestamp = time.time()

        timestamp = float(timestamp)

        # MT4 sometimes sends milliseconds
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0

        o = safe_float(c.get("open"))
        h = safe_float(c.get("high"))
        l = safe_float(c.get("low"))
        close = safe_float(c.get("close"))

        if h <= 0 or l <= 0:
            return None

        return {
            "time": timestamp,
            "open": o,
            "high": h,
            "low": l,
            "close": close,
            "volume": safe_float(
                c.get("volume", c.get("tick_volume", 0))
            ),
        }

    except Exception:
        return None


def normalize_candles(candles):
    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:
        n = normalize_candle(candle)

        if n:
            result.append(n)

    result.sort(key=lambda x: x["time"])

    # Remove duplicate timestamps
    unique = {}

    for c in result:
        unique[c["time"]] = c

    result = list(unique.values())
    result.sort(key=lambda x: x["time"])

    return result[-150:]


def get_closed_candles(candles):
    candles = normalize_candles(candles)

    if len(candles) < REQUIRED_TOTAL_CANDLES:
        return []

    # Last MT4 candle is normally the forming candle.
    closed = candles[:-1]

    if len(closed) < MIN_CANDLES:
        return []

    return closed[-120:]


# ============================================================
# MATH
# ============================================================

def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def ema_series(values, period):
    if len(values) < period:
        return []

    alpha = 2.0 / (period + 1.0)

    first = sum(values[:period]) / period

    result = [first]

    prev = first

    for value in values[period:]:
        current = (value - prev) * alpha + prev
        result.append(current)
        prev = current

    return result


def ema(values, period):
    series = ema_series(values, period)

    if not series:
        return None

    return series[-1]


def true_ranges(candles):
    result = []

    previous_close = None

    for c in candles:
        h = c["high"]
        l = c["low"]

        if previous_close is None:
            tr = h - l
        else:
            tr = max(
                h - l,
                abs(h - previous_close),
                abs(l - previous_close),
            )

        result.append(max(tr, 0.0))

        previous_close = c["close"]

    return result


def atr(candles, period=14):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def stochastic(candles, period=14):
    if len(candles) < period:
        return None

    section = candles[-period:]

    highest = max(c["high"] for c in section)
    lowest = min(c["low"] for c in section)

    if highest == lowest:
        return 50.0

    close = candles[-1]["close"]

    return 100.0 * (close - lowest) / (highest - lowest)


def cci(candles, period=20):
    if len(candles) < period:
        return None

    typical = [
        (c["high"] + c["low"] + c["close"]) / 3
        for c in candles
    ]

    current = typical[-1]

    avg = sum(typical[-period:]) / period

    mean_dev = (
        sum(abs(x - avg) for x in typical[-period:])
        / period
    )

    if mean_dev == 0:
        return 0.0

    return (current - avg) / (0.015 * mean_dev)


def momentum(values, period=10):
    if len(values) <= period:
        return None

    base = values[-period - 1]

    if base == 0:
        return None

    return (values[-1] / base) * 100.0


def macd(values, fast=12, slow=26, signal_period=9):
    fast_series = ema_series(values, fast)
    slow_series = ema_series(values, slow)

    if not fast_series or not slow_series:
        return None, None, None

    # Align using actual EMA values from the tail.
    slow_start_index = slow - fast

    aligned_fast = fast_series[slow_start_index:]

    length = min(len(aligned_fast), len(slow_series))

    macd_values = []

    for i in range(length):
        macd_values.append(
            aligned_fast[i] - slow_series[i]
        )

    if len(macd_values) < signal_period:
        return None, None, None

    signal_values = ema_series(
        macd_values,
        signal_period,
    )

    if not signal_values:
        return None, None, None

    macd_value = macd_values[-1]
    signal_value = signal_values[-1]

    return (
        macd_value,
        signal_value,
        macd_value - signal_value,
    )


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    section = candles[-period:]

    highest = max(c["high"] for c in section)
    lowest = min(c["low"] for c in section)

    if highest == lowest:
        return -50.0

    close = candles[-1]["close"]

    return -100.0 * (
        (highest - close) / (highest - lowest)
    )


def ultimate_oscillator(candles):
    if len(candles) < 30:
        return None

    bp = []
    tr = []

    for i, c in enumerate(candles):
        if i == 0:
            previous_close = c["close"]
        else:
            previous_close = candles[i - 1]["close"]

        bp.append(
            c["close"] - min(c["low"], previous_close)
        )

        tr.append(
            max(
                c["high"],
                previous_close,
            )
            - min(
                c["low"],
                previous_close,
            )
        )

    def avg_ratio(period):
        if len(bp) < period:
            return 0.0

        bp_sum = sum(bp[-period:])
        tr_sum = sum(tr[-period:])

        if tr_sum == 0:
            return 0.0

        return bp_sum / tr_sum

    a7 = avg_ratio(7)
    a14 = avg_ratio(14)
    a28 = avg_ratio(28)

    return 100.0 * (
        4.0 * a7 +
        2.0 * a14 +
        a28
    ) / 7.0


def awesome_oscillator(candles):
    if len(candles) < 34:
        return None

    median = [
        (c["high"] + c["low"]) / 2
        for c in candles
    ]

    fast = sma(median, 5)
    slow = sma(median, 34)

    if fast is None or slow is None:
        return None

    return fast - slow


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        if up_move > down_move and up_move > 0:
            pdm = up_move
        else:
            pdm = 0.0

        if down_move > up_move and down_move > 0:
            mdm = down_move
        else:
            mdm = 0.0

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)
        plus_dm.append(pdm)
        minus_dm.append(mdm)

    if len(trs) < period:
        return None, None, None

    tr_avg = sum(trs[:period]) / period
    plus_avg = sum(plus_dm[:period]) / period
    minus_avg = sum(minus_dm[:period]) / period

    dx_values = []
    plus_di_last = None
    minus_di_last = None

    for i in range(period, len(trs) + 1):
        if i > period:
            tr_avg = (
                (tr_avg * (period - 1)) + trs[i - 1]
            ) / period

            plus_avg = (
                (plus_avg * (period - 1))
                + plus_dm[i - 1]
            ) / period

            minus_avg = (
                (minus_avg * (period - 1))
                + minus_dm[i - 1]
            ) / period

        if tr_avg == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100.0 * plus_avg / tr_avg
            minus_di = 100.0 * minus_avg / tr_avg

        denominator = plus_di + minus_di

        if denominator == 0:
            dx = 0.0
        else:
            dx = (
                100.0
                * abs(plus_di - minus_di)
                / denominator
            )

        dx_values.append(dx)

        plus_di_last = plus_di
        minus_di_last = minus_di

    if len(dx_values) < period:
        adx = sum(dx_values) / len(dx_values)
    else:
        adx = sum(dx_values[-period:]) / period

    return (
        adx,
        plus_di_last,
        minus_di_last,
    )


def stoch_rsi(values, period=14):
    if len(values) < period + 14:
        return None

    rsi_values = []

    for i in range(14, len(values) + 1):
        section = values[:i]

        value = rsi(section, 14)

        if value is not None:
            rsi_values.append(value)

    if len(rsi_values) < period:
        return None

    recent = rsi_values[-period:]

    lowest = min(recent)
    highest = max(recent)

    if highest == lowest:
        return 50.0

    return (
        100.0
        * (recent[-1] - lowest)
        / (highest - lowest)
    )


def bull_bear_power(candles):
    closes = [c["close"] for c in candles]

    e20 = ema(closes, 20)

    if e20 is None:
        return None, None

    last = candles[-1]

    bull = last["high"] - e20
    bear = last["low"] - e20

    return bull, bear


# ============================================================
# PRICE ACTION
# ============================================================

def candle_metrics(candles):
    last = candles[-1]

    candle_range = last["high"] - last["low"]

    if candle_range <= 0:
        return {
            "range": 0.0,
            "body": 0.0,
            "body_ratio": 0.0,
            "upper_wick": 0.0,
            "lower_wick": 0.0,
        }

    body = abs(last["close"] - last["open"])

    upper_wick = (
        last["high"]
        - max(last["open"], last["close"])
    )

    lower_wick = (
        min(last["open"], last["close"])
        - last["low"]
    )

    return {
        "range": candle_range,
        "body": body,
        "body_ratio": body / candle_range,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
    }


def structure_direction(candles):
    if len(candles) < 8:
        return None

    recent = candles[-8:]

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    first_high = max(highs[:4])
    second_high = max(highs[4:])

    first_low = min(lows[:4])
    second_low = min(lows[4:])

    if second_high > first_high and second_low >= first_low:
        return "UP"

    if second_low < first_low and second_high <= first_high:
        return "DOWN"

    return None


def breakout_direction(candles):
    if len(candles) < 21:
        return None

    last = candles[-1]

    previous = candles[-21:-1]

    highest = max(c["high"] for c in previous)
    lowest = min(c["low"] for c in previous)

    if last["close"] > highest:
        return "UP"

    if last["close"] < lowest:
        return "DOWN"

    return None


def liquidity_direction(candles):
    if len(candles) < 10:
        return None

    last = candles[-1]

    previous = candles[-10:-1]

    previous_high = max(c["high"] for c in previous)
    previous_low = min(c["low"] for c in previous)

    # Sweep below then close back above
    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        return "UP"

    # Sweep above then close back below
    if (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        return "DOWN"

    return None


# ============================================================
# ANALYSIS ENGINE
# ============================================================

def calculate_analysis(candles):
    candles = normalize_candles(candles)

    if len(candles) < MIN_CANDLES:
        return None

    closes = [c["close"] for c in candles]

    current_price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    ema50 = ema(closes, 50)
    ema100 = ema(closes, 100)
    ema200 = ema(closes, 200)

    rsi14 = rsi(closes, 14)

    adx14, plus_di, minus_di = adx_di(
        candles,
        14,
    )

    atr14 = atr(candles, 14)

    stoch14 = stochastic(
        candles,
        14,
    )

    cci20 = cci(
        candles,
        20,
    )

    mom10 = momentum(
        closes,
        10,
    )

    macd_value, macd_signal, macd_hist = macd(
        closes
    )

    srsi = stoch_rsi(closes)

    williams = williams_r(
        candles,
        14,
    )

    ultimate = ultimate_oscillator(
        candles
    )

    ao = awesome_oscillator(
        candles
    )

    bull, bear = bull_bear_power(
        candles
    )

    structure = structure_direction(
        candles
    )

    breakout = breakout_direction(
        candles
    )

    liquidity = liquidity_direction(
        candles
    )

    metrics = candle_metrics(candles)

    # --------------------------------------------------------
    # Primary direction votes
    # --------------------------------------------------------

    up_votes = 0
    down_votes = 0

    # EMA 9/21
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            up_votes += 1
        elif ema9 < ema21:
            down_votes += 1

    # RSI
    if rsi14 is not None:
        if rsi14 >= 52:
            up_votes += 1
        elif rsi14 <= 48:
            down_votes += 1

    # ADX + DI
    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
        and adx14 >= MIN_ADX
    ):
        if plus_di > minus_di:
            up_votes += 1
        elif minus_di > plus_di:
            down_votes += 1

    primary_direction = None

    if up_votes >= 2 and down_votes <= 1:
        primary_direction = "UP"

    elif down_votes >= 2 and up_votes <= 1:
        primary_direction = "DOWN"

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    up_score = 0
    down_score = 0

    # EMA = 3
    if ema9 is not None and ema21 is not None:

        if ema9 > ema21:
            up_score += 3

        elif ema9 < ema21:
            down_score += 3

    # RSI = 2
    if rsi14 is not None:

        if 52 <= rsi14 <= 68:
            up_score += 2

        elif 32 <= rsi14 <= 48:
            down_score += 2

        elif rsi14 > 68:
            up_score += 1

        elif rsi14 < 32:
            down_score += 1

    # ADX / DI = 3
    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
    ):

        if adx14 >= STRONG_ADX:

            if plus_di > minus_di:
                up_score += 3

            elif minus_di > plus_di:
                down_score += 3

        elif adx14 >= MIN_ADX:

            if plus_di > minus_di:
                up_score += 2

            elif minus_di > plus_di:
                down_score += 2

    # Price vs EMA21 / EMA50 = 2
    if ema21 is not None:

        if current_price > ema21:
            up_score += 1

        elif current_price < ema21:
            down_score += 1

    if ema50 is not None:

        if current_price > ema50:
            up_score += 1

        elif current_price < ema50:
            down_score += 1

    # --------------------------------------------------------
    # Secondary indicators
    # max 4 points per direction
    # --------------------------------------------------------

    secondary_up = 0
    secondary_down = 0

    # Stochastic
    if stoch14 is not None:

        if stoch14 >= 55:
            secondary_up += 1

        elif stoch14 <= 45:
            secondary_down += 1

    # CCI
    if cci20 is not None:

        if cci20 > 50:
            secondary_up += 1

        elif cci20 < -50:
            secondary_down += 1

    # Momentum
    if mom10 is not None:

        if mom10 > 100:
            secondary_up += 1

        elif mom10 < 100:
            secondary_down += 1

    # MACD
    if macd_hist is not None:

        if macd_hist > 0:
            secondary_up += 1

        elif macd_hist < 0:
            secondary_down += 1

    # Williams
    if williams is not None:

        if williams > -50:
            secondary_up += 1

        elif williams < -50:
            secondary_down += 1

    # Ultimate Oscillator
    if ultimate is not None:

        if ultimate > 50:
            secondary_up += 1

        elif ultimate < 50:
            secondary_down += 1

    # AO
    if ao is not None:

        if ao > 0:
            secondary_up += 1

        elif ao < 0:
            secondary_down += 1

    # Bull/Bear power
    if bull is not None and bear is not None:

        if bull > 0 and bear > 0:
            secondary_up += 1

        elif bull < 0 and bear < 0:
            secondary_down += 1

    secondary_up = min(secondary_up, 4)
    secondary_down = min(secondary_down, 4)

    up_score += secondary_up
    down_score += secondary_down

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if structure == "UP":
        up_score += 2

    elif structure == "DOWN":
        down_score += 2

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    if breakout == "UP":
        up_score += 1

    elif breakout == "DOWN":
        down_score += 1

    # --------------------------------------------------------
    # Candle body
    # --------------------------------------------------------

    last = candles[-1]

    if metrics["body_ratio"] >= 0.45:

        if last["close"] > last["open"]:
            up_score += 1

        elif last["close"] < last["open"]:
            down_score += 1

    # --------------------------------------------------------
    # Cap at 20
    # --------------------------------------------------------

    up_score = min(up_score, MAX_SCORE)
    down_score = min(down_score, MAX_SCORE)

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if up_score > down_score:
        direction = "UP"

    elif down_score > up_score:
        direction = "DOWN"

    else:
        direction = None

    gap = abs(up_score - down_score)

    # --------------------------------------------------------
    # Conflicts
    # --------------------------------------------------------

    conflict = False
    conflict_reasons = []

    if primary_direction and direction:
        if primary_direction != direction:
            conflict = True
            conflict_reasons.append(
                "primary direction conflict"
            )

    if structure and direction:
        if structure != direction:
            # Structure contradiction is serious
            if abs(
                up_score - down_score
            ) <= 3:
                conflict = True
                conflict_reasons.append(
                    "structure contradiction"
                )

    if (
        adx14 is not None
        and adx14 >= STRONG_ADX
        and plus_di is not None
        and minus_di is not None
        and direction
    ):
        di_direction = (
            "UP"
            if plus_di > minus_di
            else "DOWN"
        )

        if di_direction != direction:
            conflict = True
            conflict_reasons.append(
                "ADX/DI contradiction"
            )

    # --------------------------------------------------------
    # Abnormal candle filter
    # --------------------------------------------------------

    recent_ranges = [
        c["high"] - c["low"]
        for c in candles[-20:]
    ]

    average_range = (
        sum(recent_ranges)
        / len(recent_ranges)
        if recent_ranges
        else 0
    )

    abnormal_candle = False

    if (
        average_range > 0
        and metrics["range"]
        > average_range * MAX_CANDLE_RANGE_RATIO
    ):
        abnormal_candle = True
        conflict_reasons.append(
            "abnormal candle"
        )

    # Very tiny candle
    if (
        average_range > 0
        and metrics["range"]
        < average_range * MIN_CANDLE_RANGE_RATIO
    ):
        conflict_reasons.append(
            "very low volatility"
        )

    # --------------------------------------------------------
    # Chase filter
    # --------------------------------------------------------

    extreme_chase = False

    if rsi14 is not None:

        if direction == "UP" and rsi14 >= 78:
            extreme_chase = True

        if direction == "DOWN" and rsi14 <= 22:
            extreme_chase = True

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    volatility_ok = True

    if atr14 is None or atr14 <= 0:
        volatility_ok = False

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = 70

    confidence += max(0, gap) * 2

    confidence += max(
        0,
        min(4, abs(up_votes - down_votes)),
    )

    if adx14 is not None:
        if adx14 >= 30:
            confidence += 5
        elif adx14 >= 25:
            confidence += 3
        elif adx14 >= MIN_ADX:
            confidence += 1

    if direction == structure:
        confidence += 2

    if direction == breakout:
        confidence += 2

    if direction == liquidity:
        confidence += 1

    if conflict:
        confidence -= 8

    if extreme_chase:
        confidence -= 5

    confidence = int(
        clamp(
            confidence,
            MIN_CONFIDENCE,
            MAX_CONFIDENCE,
        )
    )

    return {
        "direction": direction,

        "up_score": up_score,
        "down_score": down_score,

        "gap": gap,

        "confidence": confidence,

        "primary_direction": primary_direction,

        "up_votes": up_votes,
        "down_votes": down_votes,

        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,
        "ema100": ema100,
        "ema200": ema200,

        "rsi": rsi14,

        "adx": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,

        "atr": atr14,

        "stoch": stoch14,
        "cci": cci20,
        "momentum": mom10,

        "macd": macd_value,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,

        "stoch_rsi": srsi,

        "williams": williams,
        "ultimate": ultimate,

        "ao": ao,

        "bull": bull,
        "bear": bear,

        "structure": structure,
        "breakout": breakout,
        "liquidity": liquidity,

        "candle_range": metrics["range"],
        "body_ratio": metrics["body_ratio"],

        "abnormal_candle": abnormal_candle,
        "extreme_chase": extreme_chase,

        "volatility_ok": volatility_ok,

        "conflict": conflict,
        "conflict_reasons": conflict_reasons,

        "price": current_price,

        "candle_time": candles[-1]["time"],
    }


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(analysis):
    if not analysis:
        return False, "no analysis"

    direction = analysis.get("direction")

    if direction not in ("UP", "DOWN"):
        return False, "no clear direction"

    if analysis.get("primary_direction") != direction:
        return False, "primary trend conflict"

    if analysis.get("conflict"):
        return False, ", ".join(
            analysis.get(
                "conflict_reasons",
                ["conflict"],
            )
        )

    if analysis.get("abnormal_candle"):
        return False, "abnormal candle"

    if not analysis.get("volatility_ok"):
        return False, "bad volatility"

    score = max(
        analysis.get("up_score", 0),
        analysis.get("down_score", 0),
    )

    if score < MIN_SCORE:
        return False, f"score {score}/{MAX_SCORE}"

    if analysis.get("gap", 0) < MIN_SCORE_GAP:
        return False, "score gap too small"

    if analysis.get("adx") is None:
        return False, "ADX unavailable"

    if analysis["adx"] < MIN_ADX:
        return False, "ADX too weak"

    # Avoid extreme chasing
    if analysis.get("extreme_chase"):
        return False, "extreme RSI chase"

    return True, "OK"


# ============================================================
# FRESH BASE SETUP
# ============================================================

def fresh_base_setup(
    symbol,
    direction,
    analysis,
):
    current_time = now_ts()

    previous = recent_base_setups.get(symbol)

    if not previous:
        return True

    previous_direction = previous.get(
        "direction"
    )

    previous_time = previous.get(
        "time",
        0,
    )

    # Direction changed -> fresh setup
    if previous_direction != direction:
        return True

    elapsed = current_time - previous_time

    if elapsed >= SETUP_REPEAT_BLOCK_SECONDS:
        return True

    previous_price = previous.get(
        "price"
    )

    current_price = analysis.get(
        "price"
    )

    atr_value = analysis.get(
        "atr"
    )

    if (
        previous_price is not None
        and current_price is not None
        and atr_value
        and atr_value > 0
    ):
        movement = abs(
            current_price - previous_price
        )

        if movement >= (
            atr_value * FRESH_SETUP_ATR_RATIO
        ):
            return True

    previous_rsi = previous.get(
        "rsi"
    )

    current_rsi = analysis.get(
        "rsi"
    )

    if (
        previous_rsi is not None
        and current_rsi is not None
        and abs(
            current_rsi - previous_rsi
        ) >= 5
    ):
        return True

    return False


# ============================================================
# DATA QUALITY
# ============================================================

def data_is_fresh(symbol, data):
    if not data:
        return False

    candles = data.get("candles", [])

    if len(candles) < REQUIRED_TOTAL_CANDLES:
        return False

    received_at = data.get(
        "received_at",
        0,
    )

    if received_at:
        age = now_ts() - received_at

        if age > MAX_DATA_AGE_SECONDS:
            return False

    return True


# ============================================================
# CANDIDATE CREATION
# ============================================================

def evaluate_symbol(symbol, data):
    if not data_is_fresh(symbol, data):
        return None

    candles = data.get("candles", [])

    closed = get_closed_candles(
        candles
    )

    if len(closed) < MIN_CANDLES:
        return None

    analysis = calculate_analysis(
        closed
    )

    if not analysis:
        return None

    valid, reason = validate_signal(
        analysis
    )

    if not valid:
        return None

    direction = analysis["direction"]

    return {
        "symbol": symbol,
        "direction": direction,
        "analysis": analysis,
        "reason": reason,
        "candle_time": analysis[
            "candle_time"
        ],
    }


# ============================================================
# BASE CANDIDATE SELECTION
# ============================================================

def choose_best_base_pair():
    candidates = []

    with data_lock:
        items = list(
            data_store.items()
        )

    for symbol, data in items:

        candidate = evaluate_symbol(
            symbol,
            data,
        )

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

        # ----------------------------------------------------
        # Quality ranking
        # ----------------------------------------------------

        score = max(
            analysis["up_score"],
            analysis["down_score"],
        )

        gap = analysis["gap"]

        quality = (
            score * 10
            + gap * 5
        )

        # ADX
        adx_value = analysis.get("adx")

        if adx_value:
            quality += min(
                adx_value,
                40
            ) * 0.5

        # Primary agreement
        if (
            analysis.get(
                "primary_direction"
            )
            == direction
        ):
            quality += 8

        # Structure
        if (
            analysis.get("structure")
            == direction
        ):
            quality += 5

        # Breakout
        if (
            analysis.get("breakout")
            == direction
        ):
            quality += 4

        # Liquidity
        if (
            analysis.get("liquidity")
            == direction
        ):
            quality += 2

        # Candle body
        if analysis.get(
            "body_ratio",
            0,
        ) >= 0.45:
            quality += 2

        # ----------------------------------------------------
        # Avoid same symbol after previous signal
        # ----------------------------------------------------

        if (
            cycle.get("symbol")
            == symbol
        ):
            quality -= 15

        # Avoid same direction streak
        if (
            cycle.get("direction")
            == direction
        ):
            quality -= 5

        # History penalty
        recent_symbols = [
            x.get("symbol")
            for x in history[-5:]
            if x.get("symbol")
        ]

        if symbol in recent_symbols:
            quality -= 4

        candidate["quality"] = quality

        candidates.append(
            candidate
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# RECOVERY CANDIDATE SELECTION
# ============================================================

def choose_best_recovery_pair():
    """
    Recovery is NOT a BASE.

    Therefore:
    - Do NOT apply fresh_base_setup()
    - Search all valid pairs again
    - Prefer another symbol
    - Prefer a new direction
    - Only fall back to base symbol if needed
    """

    candidates = []

    base_symbol = cycle.get(
        "base_symbol"
    )

    base_direction = cycle.get(
        "base_direction"
    )

    with data_lock:
        items = list(
            data_store.items()
        )

    for symbol, data in items:

        candidate = evaluate_symbol(
            symbol,
            data,
        )

        if not candidate:
            continue

        analysis = candidate["analysis"]

        direction = candidate["direction"]

        score = max(
            analysis["up_score"],
            analysis["down_score"],
        )

        gap = analysis["gap"]

        quality = (
            score * 10
            + gap * 5
        )

        adx_value = analysis.get("adx")

        if adx_value:
            quality += min(
                adx_value,
                40
            ) * 0.5

        if (
            analysis.get(
                "primary_direction"
            )
            == direction
        ):
            quality += 8

        if (
            analysis.get("structure")
            == direction
        ):
            quality += 5

        if (
            analysis.get("breakout")
            == direction
        ):
            quality += 4

        if (
            analysis.get("liquidity")
            == direction
        ):
            quality += 2

        # ----------------------------------------------------
        # Recovery preference
        # ----------------------------------------------------

        # Strongly prefer different pair
        if (
            base_symbol
            and symbol == base_symbol
        ):
            quality -= 18

        # Prefer direction change
        if (
            base_direction
            and direction == base_direction
        ):
            quality -= 6
        else:
            quality += 4

        candidate["quality"] = quality

        candidates.append(
            candidate
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True,
    )

    return candidates[0]


# ============================================================
# TELEGRAM
# ============================================================

async def telegram_send(text):
    global telegram_app

    if not telegram_app:
        logger.error(
            "Telegram application unavailable"
        )
        return False

    try:
        await telegram_app.bot.send_message(
            chat_id=OWNER_ID,
            text=text,
            parse_mode="Markdown",
        )

        return True

    except Exception as exc:
        logger.exception(
            "Telegram send failed: %s",
            exc,
        )

        return False


def send_telegram_sync(text):
    global telegram_loop

    if telegram_loop is None:
        logger.error(
            "Telegram loop unavailable"
        )
        return False

    future = asyncio.run_coroutine_threadsafe(
        telegram_send(text),
        telegram_loop,
    )

    try:
        return future.result(
            timeout=15
        )
    except Exception as exc:
        logger.exception(
            "Telegram future error: %s",
            exc,
        )

        return False


# ============================================================
# SIGNAL TEXT
# ============================================================

def build_signal_text(
    candidate,
    trade_type,
):
    symbol = candidate["symbol"]

    direction = candidate["direction"]

    analysis = candidate["analysis"]

    score = max(
        analysis["up_score"],
        analysis["down_score"],
    )

    confidence = analysis[
        "confidence"
    ]

    price = analysis["price"]

    entry_dt = (
        now_local()
        + timedelta(
            seconds=ENTRY_DELAY_SECONDS
        )
    )

    direction_emoji = (
        "🟢"
        if direction == "UP"
        else "🔴"
    )

    if trade_type == "RECOVERY":
        title = (
            "♻️ RECOVERY 1/1"
        )
    else:
        title = (
            "🎯 BASE TRADE"
        )

    cancellation = (
        price * 0.9995
        if direction == "UP"
        else price * 1.0005
    )

    lines = [
        "🎓 *ZinoProSignalAI*",
        "━━━━━━━━━━━━━━━━━━",
        f"📊 *{symbol}* | M1",
        "",
        f"{title}",
        "",
        f"{direction_emoji} *{direction}*",
        "",
        f"🔥 Confidence: *{confidence}%*",
        (
            f"🟢 UP Score: "
            f"{analysis['up_score']}/{MAX_SCORE}"
        ),
        (
            f"🔴 DOWN Score: "
            f"{analysis['down_score']}/{MAX_SCORE}"
        ),
        "",
        (
            f"⏱️ Entry after: "
            f"*{ENTRY_DELAY_SECONDS // 60} minutes*"
        ),
        "",
        (
            f"🕐 *ENTRY TIME: "
            f"{fmt_time(entry_dt)} 🇩🇿*"
        ),
        "",
        f"💰 Price: `{price:.5f}`",
        f"🚫 Cancellation: `{cancellation:.5f}`",
        "",
        "━━━━━━━━━━━━━━━━━━",
        "🧠 *CONFIRMATION*",
        (
            f"EMA 9/21: "
            f"{'UP' if analysis['ema9'] and analysis['ema21'] and analysis['ema9'] > analysis['ema21'] else 'DOWN'}"
        ),
        (
            f"RSI: "
            f"{analysis['rsi']:.1f}"
            if analysis.get("rsi") is not None
            else "RSI: N/A"
        ),
        (
            f"ADX: "
            f"{analysis['adx']:.1f}"
            if analysis.get("adx") is not None
            else "ADX: N/A"
        ),
        (
            f"Structure: "
            f"{analysis.get('structure') or 'NEUTRAL'}"
        ),
        (
            f"Breakout: "
            f"{analysis.get('breakout') or 'NONE'}"
        ),
        "",
        "⚠️ Trade only at the displayed entry time.",
    ]

    return "\n".join(lines)


# ============================================================
# HISTORY
# ============================================================

def add_history(
    symbol,
    direction,
    trade_type,
    result="PENDING",
    confidence=None,
    score=None,
):
    item = {
        "time": now_local().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "symbol": symbol,
        "direction": direction,
        "trade_type": trade_type,
        "result": result,
        "confidence": confidence,
        "score": score,
    }

    history.append(item)

    if len(history) > 100:
        del history[:-100]

    return item


def mark_last_pending_result(result):
    for item in reversed(history):

        if (
            item.get("result")
            == "PENDING"
        ):
            item["result"] = result
            return item

    return None


# ============================================================
# SIGNAL SENDING
# ============================================================

def send_one_signal(
    candidate,
    trade_type,
):
    global cycle

    if not candidate:
        return False

    symbol = candidate["symbol"]
    direction = candidate["direction"]

    analysis = candidate["analysis"]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    if trade_type == "BASE":

        if cycle["active"]:
            return False

        if cycle["recovery_used"]:
            return False

    # --------------------------------------------------------
    # RECOVERY
    # --------------------------------------------------------

    elif trade_type == "RECOVERY":

        if not cycle["active"]:
            return False

        if cycle["recovery_used"]:
            logger.warning(
                "Recovery already used"
            )
            return False

        if (
            cycle["trade_type"]
            != "RECOVERY"
        ):
            logger.warning(
                "Recovery requested but cycle is not RECOVERY"
            )
            return False

    else:
        return False

    # --------------------------------------------------------
    # Prevent duplicate setup
    # --------------------------------------------------------

    candle_time = analysis[
        "candle_time"
    ]

    setup_key = (
        f"{symbol}|"
        f"{direction}|"
        f"{int(candle_time)}"
    )

    if cycle.get(
        "setup_key"
    ) == setup_key:
        return False

    # --------------------------------------------------------
    # Build message
    # --------------------------------------------------------

    text = build_signal_text(
        candidate,
        trade_type,
    )

    # --------------------------------------------------------
    # Send
    # --------------------------------------------------------

    cycle_backup = dict(
        cycle
    )

    cycle["active"] = True
    cycle["pending"] = True

    cycle["trade_type"] = trade_type
    cycle["symbol"] = symbol
    cycle["direction"] = direction

    cycle["setup_key"] = setup_key

    cycle["last_signal_time"] = now_ts()

    cycle["last_signal"] = candidate

    if trade_type == "BASE":

        cycle["base_symbol"] = symbol
        cycle["base_direction"] = direction

    elif trade_type == "RECOVERY":

        cycle["recovery_used"] = True
        cycle["recovery_requested"] = False
        cycle["recovery_requested_at"] = None

    success = send_telegram_sync(
        text
    )

    if not success:

        cycle.clear()
        cycle.update(
            cycle_backup
        )

        return False

    # --------------------------------------------------------
    # Store BASE setup
    # --------------------------------------------------------

    if trade_type == "BASE":

        recent_base_setups[
            symbol
        ] = {
            "direction": direction,
            "time": now_ts(),
            "price": analysis[
                "price"
            ],
            "rsi": analysis.get(
                "rsi"
            ),
        }

    add_history(
        symbol=symbol,
        direction=direction,
        trade_type=trade_type,
        result="PENDING",
        confidence=analysis[
            "confidence"
        ],
        score=max(
            analysis["up_score"],
            analysis["down_score"],
        ),
    )

    stats["signals"] += 1

    logger.info(
        "SIGNAL SENT | %s | %s | %s | %s%% | %s/20",
        trade_type,
        symbol,
        direction,
        analysis["confidence"],
        max(
            analysis["up_score"],
            analysis["down_score"],
        ),
    )

    return True


# ============================================================
# BASE ANALYSIS
# ============================================================

def analyze_and_send_base():
    with analysis_lock:

        if cycle["active"]:
            return False

        candidate = choose_best_base_pair()

        if not candidate:
            logger.info(
                "No valid BASE setup found"
            )
            return False

        return send_one_signal(
            candidate,
            "BASE",
        )


# ============================================================
# RECOVERY ANALYSIS
# ============================================================

def analyze_and_send_recovery():
    with analysis_lock:

        if not cycle["active"]:
            return False

        if cycle["pending"]:
            return False

        if cycle["recovery_used"]:
            return False

        if (
            cycle["trade_type"]
            != "RECOVERY"
        ):
            return False

        requested_at = cycle.get(
            "recovery_requested_at"
        )

        if not requested_at:
            return False

        elapsed = (
            now_ts() - requested_at
        )

        if elapsed < RECOVERY_WAIT_SECONDS:
            return False

        candidate = (
            choose_best_recovery_pair()
        )

        if not candidate:

            logger.info(
                "No valid RECOVERY setup found"
            )

            # Keep waiting for next candle/data
            return False

        return send_one_signal(
            candidate,
            "RECOVERY",
        )


# ============================================================
# SIGNAL WORKER
# ============================================================

def signal_worker():
    try:

        # Recovery first
        if (
            cycle["active"]
            and cycle["trade_type"]
            == "RECOVERY"
            and not cycle["pending"]
            and not cycle["recovery_used"]
        ):
            analyze_and_send_recovery()

            return

        # New BASE
        if not cycle["active"]:
            analyze_and_send_base()

    except Exception as exc:
        logger.exception(
            "Signal worker error: %s",
            exc,
        )


# ============================================================
# MT4 PAYLOAD
# ============================================================

def process_mt4_payload(payload):
    if not isinstance(payload, dict):
        return {
            "accepted": False,
            "error": "invalid JSON",
        }

    symbol = (
        payload.get("symbol")
        or payload.get("Symbol")
    )

    timeframe = (
        payload.get("timeframe")
        or payload.get("tf")
        or payload.get("Timeframe")
        or "M1"
    )

    candles = payload.get(
        "candles"
    )

    if not symbol:
        return {
            "accepted": False,
            "error": "symbol missing",
        }

    if not candles:
        return {
            "accepted": False,
            "error": "candles missing",
        }

    symbol = str(
        symbol
    ).upper().strip()

    timeframe = str(
        timeframe
    ).upper().strip()

    normalized = normalize_candles(
        candles
    )

    if len(normalized) < REQUIRED_TOTAL_CANDLES:
        return {
            "accepted": False,
            "error": (
                f"not enough candles: "
                f"{len(normalized)}"
            ),
        }

    with data_lock:

        data_store[
            symbol
        ] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": normalized,
            "received_at": now_ts(),
            "last_candle_time": normalized[
                -1
            ]["time"],
        }

    # Start signal engine in background
    threading.Thread(
        target=signal_worker,
        daemon=True,
        name="SignalWorker",
    ).start()

    return {
        "accepted": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(normalized),
        "batch_complete": True,
        "batch_id": (
            f"{symbol}_"
            f"{int(now_ts())}"
        ),
    }


# ============================================================
# HTTP SERVER
# ============================================================

class MT4Handler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format_string,
        *args,
    ):
        logger.info(
            "HTTP | " + format_string,
            *args,
        )

    def send_json(
        self,
        code,
        data,
    ):
        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    def check_api_key(
        self,
        payload=None,
    ):
        # If no API key is configured,
        # allow requests.
        if not MT4_API_KEY:
            return True

        header_key = (
            self.headers.get(
                "X-MT4-API-Key"
            )
            or self.headers.get(
                "X-API-Key"
            )
        )

        json_key = None

        if isinstance(payload, dict):
            json_key = payload.get(
                "api_key"
            )

        supplied = (
            header_key
            or json_key
            or ""
        )

        return supplied == MT4_API_KEY

    def do_GET(self):

        parsed = self.path.split(
            "?",
            1,
        )[0]

        if parsed in (
            "/",
            "/health",
            "/healthz",
        ):

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": (
                        "ZinoProSignalAI"
                    ),
                    "mode": "MT4",
                    "timeframe": TIMEFRAME,
                    "time": now_local().isoformat(),
                },
            )

            return

        if parsed in (
            "/mt4status",
            "/status",
        ):

            with data_lock:

                symbols = {}

                for symbol, data in data_store.items():

                    symbols[symbol] = {
                        "timeframe": data.get(
                            "timeframe"
                        ),
                        "candles": len(
                            data.get(
                                "candles",
                                [],
                            )
                        ),
                        "age_seconds": round(
                            now_ts()
                            - data.get(
                                "received_at",
                                now_ts(),
                            ),
                            1,
                        ),
                    }

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": (
                        "ZinoProSignalAI"
                    ),
                    "mode": "MT4",
                    "symbols": symbols,
                    "cycle": cycle,
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

        parsed = self.path.split(
            "?",
            1,
        )[0]

        if parsed not in (
            "/mt4",
            "/api/mt4",
        ):
            self.send_json(
                404,
                {
                    "error": "not found"
                },
            )

            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            raw = self.rfile.read(
                length
            )

            payload = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except Exception as exc:

            logger.error(
                "Invalid JSON: %s",
                exc,
            )

            self.send_json(
                400,
                {
                    "accepted": False,
                    "error": "invalid JSON",
                },
            )

            return

        if not self.check_api_key(
            payload
        ):

            self.send_json(
                401,
                {
                    "accepted": False,
                    "error": "unauthorized",
                },
            )

            return

        try:

            result = (
                process_mt4_payload(
                    payload
                )
            )

            code = (
                200
                if result.get(
                    "accepted"
                )
                else 400
            )

            self.send_json(
                code,
                result,
            )

        except Exception as exc:

            logger.exception(
                "MT4 processing error"
            )

            self.send_json(
                500,
                {
                    "accepted": False,
                    "error": str(exc),
                },
            )


def start_http_server():
    global http_server

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        MT4Handler,
    )

    http_server = server

    logger.info(
        "HTTP server listening on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

def owner_only(update):
    if not update or not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID
    )


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    text = (
        "🎓 *ZinoProSignalAI*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🟢 Bot is online\n"
        "📊 Mode: MT4 / M1\n"
        "🎯 BASE + Recovery 1/1\n"
        "⏱️ Entry: 2 minutes\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Waiting for the strongest setup..."
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    total_win = (
        stats["base_win"]
        + stats["recovery_win"]
    )

    total_loss = (
        stats["base_loss"]
        + stats["recovery_loss"]
    )

    total = (
        total_win
        + total_loss
    )

    winrate = (
        (total_win / total) * 100
        if total
        else 0
    )

    text = (
        "📊 *ZinoProSignalAI STATS*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 BASE WIN: {stats['base_win']}\n"
        f"🔴 BASE LOSS: {stats['base_loss']}\n"
        f"♻️ RECOVERY WIN: {stats['recovery_win']}\n"
        f"❌ RECOVERY LOSS: {stats['recovery_loss']}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 TOTAL WIN: {total_win}\n"
        f"❌ TOTAL LOSS: {total_loss}\n"
        f"📈 WIN RATE: {winrate:.1f}%\n"
        f"📡 SIGNALS: {stats['signals']}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    if not history:
        await update.message.reply_text(
            "📚 History is empty."
        )

        return

    lines = [
        "📚 *ZinoProSignalAI HISTORY*",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in history[-15:][::-1]:

        result = item.get(
            "result",
            "PENDING",
        )

        emoji = {
            "WIN": "🟢",
            "LOSS": "🔴",
            "PENDING": "🟡",
        }.get(
            result,
            "⚪",
        )

        score = item.get(
            "score"
        )

        confidence = item.get(
            "confidence"
        )

        lines.append(
            f"{emoji} {item.get('symbol')} "
            f"{item.get('trade_type')} | "
            f"{item.get('direction')}"
        )

        lines.append(
            f"   {result} | "
            f"{confidence}% | "
            f"{score}/{MAX_SCORE}"
        )

        lines.append(
            f"   {item.get('time')}"
        )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
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
            "📡 *MT4 STATUS*",
            "━━━━━━━━━━━━━━━━━━",
        ]

        for symbol, data in data_store.items():

            age = (
                now_ts()
                - data.get(
                    "received_at",
                    now_ts(),
                )
            )

            lines.append(
                f"🟢 {symbol} | "
                f"{data.get('timeframe')} | "
                f"candles="
                f"{len(data.get('candles', []))} | "
                f"age={age:.0f}s"
            )

    await update.message.reply_text(
        "\n".join(lines),
        parse_mode="Markdown",
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    if not context.args:
        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURUSD"
        )

        return

    symbol = (
        context.args[0]
        .upper()
        .strip()
    )

    with data_lock:
        data = data_store.get(
            symbol
        )

    if not data:
        await update.message.reply_text(
            f"❌ لا توجد بيانات لـ {symbol}"
        )

        return

    candidate = evaluate_symbol(
        symbol,
        data,
    )

    if not candidate:
        await update.message.reply_text(
            f"❌ {symbol} لا يملك حالياً setup قوي."
        )

        return

    analysis = candidate[
        "analysis"
    ]

    text = (
        f"🔎 *{symbol} M1 ANALYSIS*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Direction: *{candidate['direction']}*\n"
        f"Confidence: *{analysis['confidence']}%*\n"
        f"UP: {analysis['up_score']}/20\n"
        f"DOWN: {analysis['down_score']}/20\n"
        f"Gap: {analysis['gap']}\n"
        f"EMA9: {analysis['ema9']:.5f}\n"
        f"EMA21: {analysis['ema21']:.5f}\n"
        f"RSI: {analysis['rsi']:.2f}\n"
        f"ADX: {analysis['adx']:.2f}\n"
        f"+DI: {analysis['plus_di']:.2f}\n"
        f"-DI: {analysis['minus_di']:.2f}\n"
        f"Structure: {analysis['structure']}\n"
        f"Breakout: {analysis['breakout']}\n"
        f"Liquidity: {analysis['liquidity']}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


# ============================================================
# WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    if not cycle["active"]:
        await update.message.reply_text(
            "ℹ️ لا توجد صفقة معلقة."
        )

        return

    trade_type = cycle[
        "trade_type"
    ]

    symbol = cycle[
        "symbol"
    ]

    direction = cycle[
        "direction"
    ]

    mark_last_pending_result(
        "WIN"
    )

    if trade_type == "BASE":
        stats["base_win"] += 1

    elif trade_type == "RECOVERY":
        stats["recovery_win"] += 1

    await update.message.reply_text(
        (
            "🟢 *WIN CONFIRMED*\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | M1\n"
            f"🎯 {trade_type}\n"
            f"📈 {direction}\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "✅ Cycle completed.\n"
            "🔎 Searching for a new BASE setup..."
        ),
        parse_mode="Markdown",
    )

    # Reset cycle
    cycle.clear()

    cycle.update(
        {
            "active": False,
            "trade_type": None,
            "symbol": None,
            "direction": None,
            "pending": False,
            "recovery_used": False,
            "recovery_requested": False,
            "recovery_requested_at": None,
            "generating": False,
            "setup_key": None,
            "last_signal_time": None,
            "base_symbol": None,
            "base_direction": None,
            "last_signal": None,
        }
    )

    # Search new BASE after result
    threading.Thread(
        target=signal_worker,
        daemon=True,
        name="NewBaseAfterWin",
    ).start()


# ============================================================
# LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    if not cycle["active"]:
        await update.message.reply_text(
            "ℹ️ لا توجد صفقة معلقة."
        )

        return

    trade_type = cycle[
        "trade_type"
    ]

    symbol = cycle[
        "symbol"
    ]

    direction = cycle[
        "direction"
    ]

    mark_last_pending_result(
        "LOSS"
    )

    if trade_type == "BASE":

        stats["base_loss"] += 1

        # ----------------------------------------------------
        # Start ONE recovery
        # ----------------------------------------------------

        cycle["pending"] = False

        cycle["trade_type"] = "RECOVERY"

        cycle["recovery_used"] = False

        cycle["recovery_requested"] = True

        cycle[
            "recovery_requested_at"
        ] = now_ts()

        await update.message.reply_text(
            (
                "🔴 *BASE LOSS*\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | M1\n"
                f"📉 {direction}\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "♻️ Recovery 1/1 allowed\n"
                f"⏳ Waiting "
                f"{RECOVERY_WAIT_SECONDS // 60} minutes...\n"
                "🔎 Recovery will search all valid pairs again.\n"
                "⚠️ No Recovery 2."
            ),
            parse_mode="Markdown",
        )

        # Do not block Telegram handler.
        threading.Thread(
            target=recovery_wait_worker,
            daemon=True,
            name="RecoveryWait",
        ).start()

        return

    # --------------------------------------------------------
    # RECOVERY LOSS
    # --------------------------------------------------------

    if trade_type == "RECOVERY":

        stats["recovery_loss"] += 1

        cycle["pending"] = False
        cycle["recovery_used"] = True

        await update.message.reply_text(
            (
                "❌ *RECOVERY LOSS*\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | M1\n"
                f"📉 {direction}\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🛑 Recovery 1/1 finished.\n"
                "🚫 No second martingale.\n"
                "🔎 Cycle stopped."
            ),
            parse_mode="Markdown",
        )

        # Stop cycle
        cycle.clear()

        cycle.update(
            {
                "active": False,
                "trade_type": None,
                "symbol": None,
                "direction": None,
                "pending": False,
                "recovery_used": False,
                "recovery_requested": False,
                "recovery_requested_at": None,
                "generating": False,
                "setup_key": None,
                "last_signal_time": None,
                "base_symbol": None,
                "base_direction": None,
                "last_signal": None,
            }
        )


def recovery_wait_worker():
    try:

        while True:

            if not cycle["active"]:
                return

            if not cycle[
                "recovery_requested"
            ]:
                return

            requested_at = cycle.get(
                "recovery_requested_at"
            )

            if not requested_at:
                return

            elapsed = (
                now_ts()
                - requested_at
            )

            remaining = (
                RECOVERY_WAIT_SECONDS
                - elapsed
            )

            if remaining <= 0:
                break

            time.sleep(
                min(
                    2,
                    max(
                        0.5,
                        remaining,
                    ),
                )
            )

        analyze_and_send_recovery()

    except Exception as exc:
        logger.exception(
            "Recovery worker error: %s",
            exc,
        )


# ============================================================
# RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    cycle.clear()

    cycle.update(
        {
            "active": False,
            "trade_type": None,
            "symbol": None,
            "direction": None,
            "pending": False,
            "recovery_used": False,
            "recovery_requested": False,
            "recovery_requested_at": None,
            "generating": False,
            "setup_key": None,
            "last_signal_time": None,
            "base_symbol": None,
            "base_direction": None,
            "last_signal": None,
        }
    )

    history.clear()

    recent_base_setups.clear()

    stats.update(
        {
            "base_win": 0,
            "base_loss": 0,
            "recovery_win": 0,
            "recovery_loss": 0,
            "signals": 0,
        }
    )

    await update.message.reply_text(
        "♻️ *ZinoProSignalAI RESET*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Cycle reset\n"
        "✅ History reset\n"
        "✅ Statistics reset\n"
        "🔎 Ready for a new BASE setup.",
        parse_mode="Markdown",
    )


# ============================================================
# TELEGRAM MAIN
# ============================================================

async def post_init(
    application
):
    global telegram_loop

    telegram_loop = asyncio.get_running_loop()

    logger.info(
        "Telegram loop initialized"
    )


def telegram_main():
    global telegram_app

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:
        raise RuntimeError(
            "OWNER_ID is missing"
        )

    telegram_app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
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
            "mt4status",
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

    logger.info(
        "Starting Telegram polling..."
    )

    telegram_app.run_polling(
        drop_pending_updates=True,
        allowed_updates=[
            "message",
        ],
    )


# ============================================================
# STARTUP
# ============================================================

def main():
    logger.info(
        "Starting ZinoProSignalAI..."
    )

    logger.info(
        "Mode: MT4"
    )

    logger.info(
        "Timeframe: %s",
        TIMEFRAME,
    )

    logger.info(
        "Entry delay: %ss",
        ENTRY_DELAY_SECONDS,
    )

    logger.info(
        "Recovery wait: %ss",
        RECOVERY_WAIT_SECONDS,
    )

    logger.info(
        "MIN SCORE: %s/%s",
        MIN_SCORE,
        MAX_SCORE,
    )

    logger.info(
        "Confidence: %s-%s%%",
        MIN_CONFIDENCE,
        MAX_CONFIDENCE,
    )

    if not BOT_TOKEN:
        logger.error(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:
        logger.error(
            "OWNER_ID is missing"
        )

    if MT4_API_KEY:
        logger.info(
            "MT4 API authentication: ENABLED"
        )
    else:
        logger.warning(
            "MT4 API authentication: DISABLED"
        )

    # HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="HTTPServer",
    )

    http_thread.start()

    # Telegram
    telegram_main()


if __name__ == "__main__":
    main()
