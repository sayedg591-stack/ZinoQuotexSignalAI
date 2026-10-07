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

try:
    PORT = int(os.getenv("PORT", "10000"))
except Exception:
    PORT = 10000

TIMEZONE = ZoneInfo("Africa/Algiers")

# Default timeframe.
# MT4 can also send "timeframe" in the JSON payload.
DEFAULT_TIMEFRAME = "M1"

MIN_CANDLES = 60
REQUIRED_TOTAL_CANDLES = 61

# Signal entry is deliberately after 2 candles for M1.
ENTRY_DELAY_SECONDS = 120

# Recovery waiting period.
RECOVERY_WAIT_SECONDS = 120

# ============================================================
# SCORE SYSTEM
#
# EXACT MAX = 20
#
# Structure       3
# Breakout        3
# Liquidity       2
# Momentum        2
# Candle          2
# RSI             1
# Oscillators     2
# Moving Average  3
# ADX / DI        2
# ----------------
# TOTAL           20
# ============================================================

MAX_SCORE = 20

MIN_SCORE = 14
MIN_SCORE_GAP = 3

MIN_CONFIDENCE = 76
MAX_CONFIDENCE = 89

MIN_ADX = 20.0
STRONG_ADX = 25.0

MAX_DATA_AGE_SECONDS = 90

RECOVERY_LIMIT = 1

RECOVERY_MIN_SCORE = 16
RECOVERY_MIN_GAP = 4
RECOVERY_MIN_ADX = 25.0

SETUP_REPEAT_BLOCK_SECONDS = 300

FRESH_SETUP_ATR_RATIO = 0.35

# Avoid chasing extreme RSI.
EXTREME_RSI_UP = 78
EXTREME_RSI_DOWN = 22

# Avoid candles that are abnormally large compared with recent candles.
MAX_CANDLE_RANGE_RATIO = 2.6

# Very small candles are unreliable.
MIN_CANDLE_BODY_RATIO = 0.25

# Breakout settings.
BREAKOUT_LOOKBACK = 20
BREAKOUT_ATR_MULTIPLIER = 0.25
BREAKOUT_RETEST_TOLERANCE = 0.20


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

    "recovery_skips": 0,
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


def normalize_timeframe(value):
    if value is None:
        return DEFAULT_TIMEFRAME

    text = str(value).strip().upper()

    aliases = {
        "1": "M1",
        "1M": "M1",
        "M1": "M1",

        "2": "M2",
        "2M": "M2",
        "M2": "M2",

        "3": "M3",
        "3M": "M3",
        "M3": "M3",

        "5": "M5",
        "5M": "M5",
        "M5": "M5",

        "15": "M15",
        "15M": "M15",
        "M15": "M15",

        "30": "M30",
        "30M": "M30",
        "M30": "M30",

        "60": "H1",
        "H1": "H1",
        "1H": "H1",
    }

    return aliases.get(text, DEFAULT_TIMEFRAME)


def timeframe_minutes(timeframe):
    tf = normalize_timeframe(timeframe)

    mapping = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
    }

    return mapping.get(tf, 1)


def entry_delay_for_timeframe(timeframe):
    """
    For M1:
        2 minutes.

    For M2:
        2 minutes.

    For M3:
        3 minutes.

    For higher timeframes:
        one candle.
    """

    minutes = timeframe_minutes(timeframe)

    if minutes <= 1:
        return 120

    if minutes <= 2:
        return 120

    if minutes <= 3:
        return 180

    return minutes * 60


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):
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

    def candle_sort_key(item):
        value = item.get("time")

        if isinstance(value, (int, float)):
            return float(value)

        return str(value or "")

    result.sort(key=candle_sort_key)

    return result


def get_closed_candles(candles):
    """
    MT4 normally sends:
        candle 0 ... candle N-2 = closed
        candle N-1             = current/forming

    Therefore remove the last candle.
    """

    if len(candles) < 3:
        return []

    return candles[:-1]


# ============================================================
# INDICATORS
# ============================================================

def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)

    first = sum(values[:period]) / period

    result = [first]

    previous = first

    for value in values[period:]:
        current = (
            (value - previous)
            * multiplier
            + previous
        )

        result.append(current)
        previous = current

    return result


def ema(values, period):
    series = ema_series(values, period)

    if not series:
        return None

    return series[-1]


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def true_ranges(candles):
    if not candles:
        return []

    result = []

    previous_close = None

    for c in candles:

        if previous_close is None:
            tr = c["high"] - c["low"]

        else:
            tr = max(
                c["high"] - c["low"],
                abs(c["high"] - previous_close),
                abs(c["low"] - previous_close),
            )

        result.append(max(tr, 0.0))

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

        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

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
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    window = candles[-period:]

    highest = max(
        c["high"]
        for c in window
    )

    lowest = min(
        c["low"]
        for c in window
    )

    if highest == lowest:
        return -50.0

    return (
        (
            highest
            - candles[-1]["close"]
        )
        / (highest - lowest)
    ) * -100.0


def stochastic(candles, period=14):
    if len(candles) < period:
        return None, None

    def calculate_k(index):
        start = max(
            0,
            index - period + 1,
        )

        window = candles[start:index + 1]

        highest = max(
            c["high"]
            for c in window
        )

        lowest = min(
            c["low"]
            for c in window
        )

        if highest == lowest:
            return 50.0

        return (
            (
                candles[index]["close"]
                - lowest
            )
            / (highest - lowest)
        ) * 100.0

    k = calculate_k(
        len(candles) - 1
    )

    values = []

    for i in range(
        max(0, len(candles) - 3),
        len(candles),
    ):
        values.append(
            calculate_k(i)
        )

    d = (
        sum(values) / len(values)
        if values
        else 50.0
    )

    return k, d


def momentum(candles, period=5):
    if len(candles) <= period:
        return None

    return (
        candles[-1]["close"]
        - candles[-1 - period]["close"]
    )


def macd(candles):
    closes = [
        c["close"]
        for c in candles
    ]

    fast_series = ema_series(
        closes,
        12,
    )

    slow_series = ema_series(
        closes,
        26,
    )

    if not fast_series or not slow_series:
        return None, None

    length = min(
        len(fast_series),
        len(slow_series),
    )

    macd_series = []

    for i in range(length):
        macd_series.append(
            fast_series[-length + i]
            - slow_series[-length + i]
        )

    if not macd_series:
        return None, None

    macd_line = macd_series[-1]

    signal = ema(
        macd_series,
        9,
    )

    return macd_line, signal


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

    window = typical[-period:]

    mean = sum(window) / period

    deviation = (
        sum(
            abs(x - mean)
            for x in window
        )
        / period
    )

    if deviation == 0:
        return 0.0

    return (
        (typical[-1] - mean)
        / (0.015 * deviation)
    )


def adx_di(candles, period=14):
    if len(candles) < period * 2 + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        up_move = (
            current["high"]
            - previous["high"]
        )

        down_move = (
            previous["low"]
            - current["low"]
        )

        plus = (
            up_move
            if (
                up_move > down_move
                and up_move > 0
            )
            else 0.0
        )

        minus = (
            down_move
            if (
                down_move > up_move
                and down_move > 0
            )
            else 0.0
        )

        tr = max(
            current["high"]
            - current["low"],

            abs(
                current["high"]
                - previous["close"]
            ),

            abs(
                current["low"]
                - previous["close"]
            ),
        )

        trs.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    dx_values = []
    plus_values = []
    minus_values = []

    for i in range(
        period,
        len(trs) + 1,
    ):

        tr_sum = sum(
            trs[i - period:i]
        )

        plus_sum = sum(
            plus_dm[i - period:i]
        )

        minus_sum = sum(
            minus_dm[i - period:i]
        )

        if tr_sum <= 0:
            plus = 0.0
            minus = 0.0

        else:
            plus = (
                100.0
                * plus_sum
                / tr_sum
            )

            minus = (
                100.0
                * minus_sum
                / tr_sum
            )

        denominator = plus + minus

        if denominator <= 0:
            dx = 0.0

        else:
            dx = (
                abs(plus - minus)
                / denominator
            ) * 100.0

        dx_values.append(dx)
        plus_values.append(plus)
        minus_values.append(minus)

    if len(dx_values) < period:
        return None, None, None

    adx = (
        sum(dx_values[-period:])
        / period
    )

    return (
        adx,
        plus_values[-1],
        minus_values[-1],
    )


# ============================================================
# PRICE ACTION
# ============================================================

def candle_metrics(candles):
    if not candles:
        return {}

    c = candles[-1]

    body = abs(
        c["close"]
        - c["open"]
    )

    total_range = (
        c["high"]
        - c["low"]
    )

    upper_wick = (
        c["high"]
        - max(
            c["open"],
            c["close"],
        )
    )

    lower_wick = (
        min(
            c["open"],
            c["close"],
        )
        - c["low"]
    )

    body_ratio = (
        body / total_range
        if total_range > 0
        else 0.0
    )

    return {
        "body": body,
        "range": total_range,
        "body_ratio": body_ratio,
        "upper_wick": max(
            upper_wick,
            0.0,
        ),
        "lower_wick": max(
            lower_wick,
            0.0,
        ),
        "bullish": (
            c["close"]
            > c["open"]
        ),
        "bearish": (
            c["close"]
            < c["open"]
        ),
    }


def structure_analysis(candles):
    """
    Uses recent swing behavior instead of only comparing
    the last two candles.
    """

    result = {
        "direction": None,
        "strength": "NONE",
    }

    if len(candles) < 12:
        return result

    recent = candles[-8:]

    highs = [
        c["high"]
        for c in recent
    ]

    lows = [
        c["low"]
        for c in recent
    ]

    # Compare two 4-candle blocks.
    first = recent[:4]
    second = recent[4:]

    first_high = max(
        c["high"]
        for c in first
    )

    second_high = max(
        c["high"]
        for c in second
    )

    first_low = min(
        c["low"]
        for c in first
    )

    second_low = min(
        c["low"]
        for c in second
    )

    if (
        second_high > first_high
        and second_low > first_low
    ):
        result["direction"] = "UP"
        result["strength"] = "STRONG"

        return result

    if (
        second_high < first_high
        and second_low < first_low
    ):
        result["direction"] = "DOWN"
        result["strength"] = "STRONG"

        return result

    # Weaker continuation structure.
    if (
        highs[-1] > highs[-2]
        and lows[-1] >= lows[-2]
    ):
        result["direction"] = "UP"
        result["strength"] = "WEAK"

        return result

    if (
        highs[-1] < highs[-2]
        and lows[-1] <= lows[-2]
    ):
        result["direction"] = "DOWN"
        result["strength"] = "WEAK"

        return result

    return result


def liquidity_analysis(candles):
    """
    Detects a liquidity sweep.

    Sweep below + close back above = bullish.
    Sweep above + close back below = bearish.
    """

    result = {
        "direction": None,
        "strength": "NONE",
    }

    if len(candles) < 15:
        return result

    current = candles[-1]

    previous = candles[-11:-1]

    highest = max(
        c["high"]
        for c in previous
    )

    lowest = min(
        c["low"]
        for c in previous
    )

    current_range = (
        current["high"]
        - current["low"]
    )

    if current_range <= 0:
        return result

    # Bullish liquidity sweep.
    if (
        current["low"] < lowest
        and current["close"] > lowest
    ):

        recovery_ratio = (
            current["close"]
            - current["low"]
        ) / current_range

        if recovery_ratio >= 0.55:
            result["direction"] = "UP"
            result["strength"] = "STRONG"
        else:
            result["direction"] = "UP"
            result["strength"] = "WEAK"

        return result

    # Bearish liquidity sweep.
    if (
        current["high"] > highest
        and current["close"] < highest
    ):

        rejection_ratio = (
            current["high"]
            - current["close"]
        ) / current_range

        if rejection_ratio >= 0.55:
            result["direction"] = "DOWN"
            result["strength"] = "STRONG"
        else:
            result["direction"] = "DOWN"
            result["strength"] = "WEAK"

        return result

    return result


# ============================================================
# BREAKOUT ENGINE
# ============================================================

def breakout_analysis(candles):
    result = {
        "direction": None,
        "strength": "NONE",
        "retest": False,
        "fake": False,
        "level": None,
        "score": 0,
    }

    if len(candles) < BREAKOUT_LOOKBACK + 5:
        return result

    atr_value = atr(
        candles,
        14,
    )

    if atr_value is None or atr_value <= 0:
        return result

    last = candles[-1]

    previous = candles[
        -BREAKOUT_LOOKBACK - 1:-1
    ]

    highest = max(
        c["high"]
        for c in previous
    )

    lowest = min(
        c["low"]
        for c in previous
    )

    candle_range = (
        last["high"]
        - last["low"]
    )

    if candle_range <= 0:
        return result

    # --------------------------------------------------------
    # TRUE UP BREAKOUT
    # --------------------------------------------------------

    if last["close"] > highest:

        distance = (
            last["close"]
            - highest
        )

        result["direction"] = "UP"
        result["level"] = highest

        if distance >= (
            atr_value
            * BREAKOUT_ATR_MULTIPLIER
        ):
            result["strength"] = "STRONG"
            result["score"] = 3
        else:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # TRUE DOWN BREAKOUT
    # --------------------------------------------------------

    elif last["close"] < lowest:

        distance = (
            lowest
            - last["close"]
        )

        result["direction"] = "DOWN"
        result["level"] = lowest

        if distance >= (
            atr_value
            * BREAKOUT_ATR_MULTIPLIER
        ):
            result["strength"] = "STRONG"
            result["score"] = 3
        else:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # FAKE UP BREAKOUT
    # --------------------------------------------------------

    elif (
        last["high"] > highest
        and last["close"] <= highest
    ):

        result["direction"] = "UP"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = highest
        result["score"] = -3

    # --------------------------------------------------------
    # FAKE DOWN BREAKOUT
    # --------------------------------------------------------

    elif (
        last["low"] < lowest
        and last["close"] >= lowest
    ):

        result["direction"] = "DOWN"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = lowest
        result["score"] = -3

    # --------------------------------------------------------
    # RETEST
    # --------------------------------------------------------

    direction = result["direction"]
    level = result["level"]

    if (
        direction in ("UP", "DOWN")
        and not result["fake"]
        and level is not None
    ):

        tolerance = (
            atr_value
            * BREAKOUT_RETEST_TOLERANCE
        )

        scan_start = max(
            1,
            len(candles) - 6,
        )

        for i in range(
            scan_start,
            len(candles) - 1,
        ):

            c = candles[i]

            if direction == "UP":

                touched = (
                    c["low"]
                    <= level + tolerance
                    and c["low"]
                    >= level - tolerance
                )

                rejected = (
                    c["close"]
                    > level
                )

                if touched and rejected:
                    result["retest"] = True
                    break

            else:

                touched = (
                    c["high"]
                    <= level + tolerance
                    and c["high"]
                    >= level - tolerance
                )

                rejected = (
                    c["close"]
                    < level
                )

                if touched and rejected:
                    result["retest"] = True
                    break

    return result


# ============================================================
# ANALYSIS
# ============================================================

def calculate_analysis(candles):
    if len(candles) < MIN_CANDLES:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    last = candles[-1]

    # --------------------------------------------------------
    # Indicators
    # --------------------------------------------------------

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    rsi14 = rsi(
        candles,
        14,
    )

    williams = williams_r(
        candles,
        14,
    )

    stoch_k, stoch_d = stochastic(
        candles,
        14,
    )

    macd_line, macd_signal = macd(
        candles
    )

    mom5 = momentum(
        candles,
        5,
    )

    adx, plus_di, minus_di = adx_di(
        candles,
        14,
    )

    atr14 = atr(
        candles,
        14,
    )

    cci20 = cci(
        candles,
        20,
    )

    structure_info = structure_analysis(
        candles
    )

    liquidity_info = liquidity_analysis(
        candles
    )

    breakout_info = breakout_analysis(
        candles
    )

    metrics = candle_metrics(
        candles
    )

    # --------------------------------------------------------
    # Directions
    # --------------------------------------------------------

    up_score = 0
    down_score = 0

    reasons_up = []
    reasons_down = []

    # ========================================================
    # 1. STRUCTURE = 3
    # ========================================================

    structure = structure_info["direction"]

    if structure == "UP":
        up_score += 3
        reasons_up.append("Structure UP")

    elif structure == "DOWN":
        down_score += 3
        reasons_down.append("Structure DOWN")

    # ========================================================
    # 2. BREAKOUT = 3
    # ========================================================

    breakout = breakout_info["direction"]
    breakout_strength = breakout_info["strength"]
    breakout_retest = breakout_info["retest"]
    breakout_fake = breakout_info["fake"]

    if (
        breakout in ("UP", "DOWN")
        and not breakout_fake
    ):

        breakout_points = 0

        if breakout_strength == "STRONG":
            breakout_points += 2

        elif breakout_strength == "WEAK":
            breakout_points += 1

        if breakout_retest:
            breakout_points += 1

        breakout_points = clamp(
            breakout_points,
            0,
            3,
        )

        if breakout == "UP":
            up_score += breakout_points

            if breakout_points:
                reasons_up.append(
                    "Breakout UP"
                )

        else:
            down_score += breakout_points

            if breakout_points:
                reasons_down.append(
                    "Breakout DOWN"
                )

    # Fake breakout is not a positive score.
    # It is treated as a warning against the breakout direction.

    # ========================================================
    # 3. LIQUIDITY = 2
    # ========================================================

    liquidity = liquidity_info["direction"]
    liquidity_strength = liquidity_info["strength"]

    if liquidity == "UP":

        points = (
            2
            if liquidity_strength == "STRONG"
            else 1
        )

        up_score += points
        reasons_up.append("Liquidity sweep UP")

    elif liquidity == "DOWN":

        points = (
            2
            if liquidity_strength == "STRONG"
            else 1
        )

        down_score += points
        reasons_down.append("Liquidity sweep DOWN")

    # ========================================================
    # 4. MOMENTUM = 2
    # ========================================================

    momentum_votes_up = 0
    momentum_votes_down = 0

    if mom5 is not None:

        if mom5 > 0:
            momentum_votes_up += 1

        elif mom5 < 0:
            momentum_votes_down += 1

    if macd_line is not None and macd_signal is not None:

        if macd_line > macd_signal:
            momentum_votes_up += 1

        elif macd_line < macd_signal:
            momentum_votes_down += 1

    if cci20 is not None:

        if cci20 > 0:
            momentum_votes_up += 1

        elif cci20 < 0:
            momentum_votes_down += 1

    if momentum_votes_up >= 2:
        up_score += 2
        reasons_up.append("Momentum UP")

    elif momentum_votes_down >= 2:
        down_score += 2
        reasons_down.append("Momentum DOWN")

    elif momentum_votes_up == 1:
        up_score += 1

    elif momentum_votes_down == 1:
        down_score += 1

    # ========================================================
    # 5. CANDLE = 2
    # ========================================================

    body_ratio = metrics.get(
        "body_ratio",
        0.0,
    )

    if body_ratio >= MIN_CANDLE_BODY_RATIO:

        if metrics.get("bullish"):

            up_score += 2
            reasons_up.append("Bullish candle")

        elif metrics.get("bearish"):

            down_score += 2
            reasons_down.append("Bearish candle")

    # ========================================================
    # 6. RSI = 1
    # ========================================================

    if rsi14 is not None:

        if 52 <= rsi14 < EXTREME_RSI_UP:

            up_score += 1
            reasons_up.append(
                f"RSI {rsi14:.1f}"
            )

        elif EXTREME_RSI_DOWN < rsi14 <= 48:

            down_score += 1
            reasons_down.append(
                f"RSI {rsi14:.1f}"
            )

    # ========================================================
    # 7. OSCILLATORS = 2
    #
    # Williams + Stochastic
    # ========================================================

    oscillator_up = 0
    oscillator_down = 0

    if williams is not None:

        if williams > -50:
            oscillator_up += 1

        elif williams < -50:
            oscillator_down += 1

    if (
        stoch_k is not None
        and stoch_d is not None
    ):

        if (
            stoch_k > stoch_d
            and stoch_k > 50
        ):
            oscillator_up += 1

        elif (
            stoch_k < stoch_d
            and stoch_k < 50
        ):
            oscillator_down += 1

    if oscillator_up >= 2:
        up_score += 2
        reasons_up.append("Oscillators UP")

    elif oscillator_down >= 2:
        down_score += 2
        reasons_down.append("Oscillators DOWN")

    elif oscillator_up == 1:
        up_score += 1

    elif oscillator_down == 1:
        down_score += 1

    # ========================================================
    # 8. MOVING AVERAGES = 3
    #
    # EMA 9/21 + price position EMA21/50
    # ========================================================

    ma_up = 0
    ma_down = 0

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:
            ma_up += 2

        elif ema9 < ema21:
            ma_down += 2

    if ema21 is not None:

        if last["close"] > ema21:
            ma_up += 1

        elif last["close"] < ema21:
            ma_down += 1

    if (
        ema50 is not None
        and ema21 is not None
    ):

        if (
            ema21 > ema50
            and last["close"] > ema50
        ):
            ma_up = min(
                3,
                ma_up + 1,
            )

        elif (
            ema21 < ema50
            and last["close"] < ema50
        ):
            ma_down = min(
                3,
                ma_down + 1,
            )

    ma_up = min(3, ma_up)
    ma_down = min(3, ma_down)

    if ma_up > ma_down:
        up_score += ma_up
        reasons_up.append("MA alignment")

    elif ma_down > ma_up:
        down_score += ma_down
        reasons_down.append("MA alignment")

    # ========================================================
    # 9. ADX / DI = 2
    # ========================================================

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
    ):

        if adx >= MIN_ADX:

            if plus_di > minus_di:
                up_score += 2
                reasons_up.append(
                    f"DI+ {plus_di:.1f}"
                )

            elif minus_di > plus_di:
                down_score += 2
                reasons_down.append(
                    f"DI- {minus_di:.1f}"
                )

    # ========================================================
    # HARD CAP
    # ========================================================

    up_score = int(
        clamp(
            up_score,
            0,
            MAX_SCORE,
        )
    )

    down_score = int(
        clamp(
            down_score,
            0,
            MAX_SCORE,
        )
    )

    # ========================================================
    # PRIMARY DIRECTION
    # ========================================================

    primary_votes = 0

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:
            primary_votes += 1

        elif ema9 < ema21:
            primary_votes -= 1

    if (
        plus_di is not None
        and minus_di is not None
    ):

        if plus_di > minus_di:
            primary_votes += 1

        elif minus_di > plus_di:
            primary_votes -= 1

    if structure == "UP":
        primary_votes += 1

    elif structure == "DOWN":
        primary_votes -= 1

    if primary_votes > 0:
        primary_direction = "UP"

    elif primary_votes < 0:
        primary_direction = "DOWN"

    else:
        primary_direction = None

    # ========================================================
    # FINAL DIRECTION
    # ========================================================

    if up_score > down_score:
        direction = "UP"

    elif down_score > up_score:
        direction = "DOWN"

    else:
        direction = None

    max_score = max(
        up_score,
        down_score,
    )

    score_gap = abs(
        up_score - down_score
    )

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    conflicts = []

    if (
        direction is not None
        and primary_direction is not None
        and direction != primary_direction
    ):
        conflicts.append(
            "Primary direction conflict"
        )

    # Strong structure against signal is a serious conflict.
    if (
        direction == "UP"
        and structure == "DOWN"
        and structure_info["strength"] == "STRONG"
    ):
        conflicts.append(
            "Strong bearish structure"
        )

    if (
        direction == "DOWN"
        and structure == "UP"
        and structure_info["strength"] == "STRONG"
    ):
        conflicts.append(
            "Strong bullish structure"
        )

    # Fake breakout against current direction.
    if breakout_fake:

        if (
            direction is not None
            and breakout is not None
            and breakout == direction
        ):
            conflicts.append(
                "Fake breakout"
            )

    # ========================================================
    # ABNORMAL CANDLE
    # ========================================================

    abnormal_candle = False

    if len(candles) >= 22:

        ranges = [
            c["high"] - c["low"]
            for c in candles[-21:-1]
        ]

        average_range = (
            sum(ranges)
            / len(ranges)
            if ranges
            else 0
        )

        current_range = (
            last["high"]
            - last["low"]
        )

        if (
            average_range > 0
            and current_range
            > average_range
            * MAX_CANDLE_RANGE_RATIO
        ):
            abnormal_candle = True

            conflicts.append(
                "Abnormal candle"
            )

    # ========================================================
    # EXTREME CHASE
    # ========================================================

    extreme_chase = False

    if rsi14 is not None:

        if (
            direction == "UP"
            and rsi14 >= EXTREME_RSI_UP
        ):
            extreme_chase = True

        if (
            direction == "DOWN"
            and rsi14 <= EXTREME_RSI_DOWN
        ):
            extreme_chase = True

    # ========================================================
    # VOLATILITY
    # ========================================================

    volatility_ok = (
        atr14 is not None
        and atr14 > 0
    )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = 74

    # Gap is important but cannot dominate.
    confidence += min(
        8,
        score_gap * 2,
    )

    # Primary alignment.
    if (
        direction is not None
        and primary_direction == direction
    ):
        confidence += 3

    # ADX.
    if adx is not None:

        if adx >= 30:
            confidence += 5

        elif adx >= STRONG_ADX:
            confidence += 3

        elif adx >= MIN_ADX:
            confidence += 1

    # Structure.
    if structure == direction:
        confidence += 3

    # Strong breakout.
    if (
        breakout == direction
        and not breakout_fake
    ):

        if breakout_strength == "STRONG":
            confidence += 4

        elif breakout_strength == "WEAK":
            confidence += 1

        if breakout_retest:
            confidence += 2

    # Liquidity confirmation.
    if liquidity == direction:
        confidence += 2

    # Candle.
    if body_ratio >= 0.55:
        confidence += 1

    # Penalties.
    if conflicts:
        confidence -= 8

    if extreme_chase:
        confidence -= 5

    if abnormal_candle:
        confidence -= 5

    confidence = int(
        clamp(
            confidence,
            MIN_CONFIDENCE,
            MAX_CONFIDENCE,
        )
    )

    # ========================================================
    # CANCELLATION LEVEL
    # ========================================================

    cancellation_level = None

    if atr14 is not None and atr14 > 0:

        if direction == "UP":
            cancellation_level = (
                last["close"]
                - atr14 * 0.35
            )

        elif direction == "DOWN":
            cancellation_level = (
                last["close"]
                + atr14 * 0.35
            )

    return {
        "direction": direction,

        "up_score": up_score,
        "down_score": down_score,

        "max_score": max_score,
        "score_gap": score_gap,

        "confidence": confidence,

        "primary_direction": primary_direction,

        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,

        "rsi": rsi14,
        "williams": williams,

        "stoch_k": stoch_k,
        "stoch_d": stoch_d,

        "macd": macd_line,
        "macd_signal": macd_signal,

        "momentum": mom5,
        "cci": cci20,

        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,

        "atr": atr14,

        "structure": structure,
        "structure_strength": structure_info["strength"],

        "breakout": breakout,
        "breakout_strength": breakout_strength,
        "breakout_retest": breakout_retest,
        "breakout_fake": breakout_fake,
        "breakout_score": breakout_info["score"],
        "breakout_level": breakout_info["level"],

        "liquidity": liquidity,
        "liquidity_strength": liquidity_strength,

        "body_ratio": body_ratio,

        "abnormal_candle": abnormal_candle,
        "volatility_ok": volatility_ok,
        "extreme_chase": extreme_chase,

        "conflict": bool(conflicts),
        "conflict_reasons": conflicts,

        "reasons_up": reasons_up,
        "reasons_down": reasons_down,

        "candle_time": last.get("time"),

        "price": last["close"],

        "cancellation_level": cancellation_level,
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
        return False, "NO_ANALYSIS"

    direction = analysis.get(
        "direction"
    )

    if direction not in (
        "UP",
        "DOWN",
    ):
        return False, "NO_CLEAR_DIRECTION"

    if (
        analysis.get(
            "primary_direction"
        )
        != direction
    ):
        return False, "PRIMARY_CONFLICT"

    if analysis.get("conflict"):
        return False, "MARKET_CONFLICT"

    if analysis.get(
        "abnormal_candle"
    ):
        return False, "ABNORMAL_CANDLE"

    if not analysis.get(
        "volatility_ok"
    ):
        return False, "NO_VOLATILITY"

    if (
        analysis.get("max_score", 0)
        < MIN_SCORE
    ):
        return False, "SCORE_TOO_LOW"

    if (
        analysis.get("score_gap", 0)
        < MIN_SCORE_GAP
    ):
        return False, "SCORE_GAP_TOO_SMALL"

    adx = analysis.get("adx")

    if (
        adx is None
        or adx < MIN_ADX
    ):
        return False, "ADX_TOO_WEAK"

    if analysis.get(
        "extreme_chase"
    ):
        return False, "EXTREME_RSI"

    if analysis.get(
        "breakout_fake"
    ):
        return False, "FAKE_BREAKOUT"

    # Do not accept a weak structure against a very strong
    # breakout unless the final direction is aligned.
    structure = analysis.get(
        "structure"
    )

    breakout = analysis.get(
        "breakout"
    )

    direction = analysis.get(
        "direction"
    )

    if (
        structure is not None
        and breakout is not None
        and structure != direction
        and breakout == direction
    ):
        # Not an automatic rejection, but require stronger score.
        if (
            analysis.get("max_score", 0)
            < 16
        ):
            return False, "STRUCTURE_BREAKOUT_MISMATCH"

    return True, "VALID"


# ============================================================
# FRESH SETUP
# ============================================================

def fresh_base_setup(
    symbol,
    direction,
    analysis,
):
    key = (
        f"{symbol}:{direction}"
    )

    previous = recent_base_setups.get(
        key
    )

    if not previous:
        return True

    elapsed = (
        time.time()
        - previous.get(
            "time",
            0,
        )
    )

    if (
        elapsed
        >= SETUP_REPEAT_BLOCK_SECONDS
    ):
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
        and atr_value is not None
        and atr_value > 0
    ):

        move = abs(
            current_price
            - previous_price
        )

        if move >= (
            atr_value
            * FRESH_SETUP_ATR_RATIO
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
            current_rsi
            - previous_rsi
        ) >= 5
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

    candles = item.get(
        "candles",
        [],
    )

    if (
        len(candles)
        < REQUIRED_TOTAL_CANDLES
    ):
        return None

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

    return {
        "symbol": symbol,
        "analysis": analysis,
        "direction": analysis[
            "direction"
        ],
        "validation": reason,
        "timeframe": normalize_timeframe(
            item.get(
                "timeframe",
                DEFAULT_TIMEFRAME,
            )
        ),
    }


# ============================================================
# BEST BASE
# ============================================================

def choose_best_base_pair():
    candidates = []

    with data_lock:
        symbols = list(
            data_store.keys()
        )

    recent_symbols = [
        x.get("symbol")
        for x in history[-10:]
        if x.get("symbol")
    ]

    for symbol in symbols:

        candidate = evaluate_symbol(
            symbol
        )

        if not candidate:
            continue

        analysis = candidate[
            "analysis"
        ]

        direction = candidate[
            "direction"
        ]

        if not fresh_base_setup(
            symbol,
            direction,
            analysis,
        ):
            continue

        score = analysis.get(
            "max_score",
            0,
        )

        gap = analysis.get(
            "score_gap",
            0,
        )

        adx = analysis.get(
            "adx"
        ) or 0

        confidence = analysis.get(
            "confidence",
            0,
        )

        # Base quality ranking.
        quality = (
            score * 10
            + gap * 7
            + adx
            + confidence * 0.25
        )

        if (
            analysis.get(
                "primary_direction"
            )
            == direction
        ):
            quality += 8

        if (
            analysis.get(
                "structure"
            )
            == direction
        ):
            quality += 8

        if (
            analysis.get(
                "structure_strength"
            )
            == "STRONG"
        ):
            quality += 4

        if (
            analysis.get(
                "breakout"
            )
            == direction
            and not analysis.get(
                "breakout_fake"
            )
        ):

            if (
                analysis.get(
                    "breakout_strength"
                )
                == "STRONG"
            ):
                quality += 8

            elif (
                analysis.get(
                    "breakout_strength"
                )
                == "WEAK"
            ):
                quality += 3

            if analysis.get(
                "breakout_retest"
            ):
                quality += 6

        if (
            analysis.get(
                "liquidity"
            )
            == direction
        ):
            quality += 4

        if symbol in recent_symbols:
            quality -= 5

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
# BEST RECOVERY
# ============================================================

def choose_best_recovery_pair():
    candidates = []

    with data_lock:
        symbols = list(
            data_store.keys()
        )

    base_symbol = cycle.get(
        "base_symbol"
    )

    base_direction = cycle.get(
        "base_direction"
    )

    for symbol in symbols:

        candidate = evaluate_symbol(
            symbol
        )

        if not candidate:
            continue

        analysis = candidate[
            "analysis"
        ]

        direction = candidate[
            "direction"
        ]

        score = analysis.get(
            "max_score",
            0,
        )

        gap = analysis.get(
            "score_gap",
            0,
        )

        adx = analysis.get(
            "adx"
        )

        structure = analysis.get(
            "structure"
        )

        breakout = analysis.get(
            "breakout"
        )

        strength = analysis.get(
            "breakout_strength"
        )

        retest = bool(
            analysis.get(
                "breakout_retest"
            )
        )

        fake = bool(
            analysis.get(
                "breakout_fake"
            )
        )

        # ====================================================
        # HARD RECOVERY FILTER
        # ====================================================

        if score < RECOVERY_MIN_SCORE:
            continue

        if gap < RECOVERY_MIN_GAP:
            continue

        if (
            adx is None
            or adx < RECOVERY_MIN_ADX
        ):
            continue

        if structure != direction:
            continue

        if breakout != direction:
            continue

        if strength != "STRONG":
            continue

        if not retest:
            continue

        if fake:
            continue

        # Recovery must not chase extreme RSI.
        if analysis.get(
            "extreme_chase"
        ):
            continue

        quality = (
            score * 12
            + gap * 8
            + adx * 1.2
        )

        quality += 15  # structure
        quality += 15  # strong breakout
        quality += 12  # retest

        if (
            analysis.get(
                "liquidity"
            )
            == direction
        ):
            quality += 5

        if (
            analysis.get(
                "primary_direction"
            )
            == direction
        ):
            quality += 8

        if (
            analysis.get(
                "body_ratio",
                0,
            )
            >= 0.55
        ):
            quality += 3

        if direction != base_direction:
            quality += 4

        if symbol == base_symbol:
            quality -= 5

        quality += (
            analysis.get(
                "confidence",
                0,
            )
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
    timeframe="M1",
):
    item = {
        "time": now_string(),

        "symbol": symbol,

        "direction": direction,

        "stage": stage,

        "result": result,

        "timeframe": timeframe,

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

        "entry_time": (
            entry_time.isoformat()
            if isinstance(
                entry_time,
                datetime,
            )
            else entry_time
        ),
    }

    history.append(item)

    if len(history) > 100:
        del history[:-100]

    return item


# ============================================================
# TELEGRAM SIGNAL CARD
# ============================================================

def build_signal_text(
    symbol,
    direction,
    stage,
    analysis,
    entry_time,
    timeframe,
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

    cancellation = analysis.get(
        "cancellation_level"
    )

    structure = analysis.get(
        "structure"
    )

    breakout = analysis.get(
        "breakout"
    )

    strength = analysis.get(
        "breakout_strength",
        "NONE",
    )

    retest = analysis.get(
        "breakout_retest",
        False,
    )

    adx = analysis.get(
        "adx"
    )

    rsi_value = analysis.get(
        "rsi"
    )

    if strength == "STRONG":
        quality_text = "🔥 STRONG"

    elif strength == "WEAK":
        quality_text = "🟡 WEAK"

    elif strength == "FAKE":
        quality_text = "⚠️ FAKE"

    else:
        quality_text = "⚪ NONE"

    stage_text = (
        "🎯 BASE TRADE"
        if stage == "BASE"
        else "♻️ RECOVERY 1/1"
    )

    delay = entry_delay_for_timeframe(
        timeframe
    )

    delay_minutes = max(
        1,
        int(round(delay / 60))
    )

    if direction == "UP":

        cancel_text = (
            f"🛑 Cancellation: "
            f"إذا أغلقت شمعة تحت "
            f"**{fmt_price(cancellation)}**"
        )

    else:

        cancel_text = (
            f"🛑 Cancellation: "
            f"إذا أغلقت شمعة فوق "
            f"**{fmt_price(cancellation)}**"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{stage_text}\n"
        f"{direction_emoji(direction)} "
        f"**{direction}**\n\n"
        f"🔥 Confidence: **{confidence}%**\n"
        f"🟢 UP Score: **{up_score}/{MAX_SCORE}**\n"
        f"🔴 DOWN Score: **{down_score}/{MAX_SCORE}**\n\n"
        f"📐 Structure: **"
        f"{structure or 'NONE'}**\n"
        f"🚀 Breakout: **"
        f"{breakout or 'NONE'}**\n"
        f"💥 Breakout Quality: **"
        f"{quality_text}**\n"
        f"🔄 Retest: **"
        f"{'YES' if retest else 'NO'}**\n\n"
        f"📈 ADX: **"
        f"{adx:.1f}**\n"
        if adx is not None
        else
        "📈 ADX: **--**\n"
    ) + (
        f"📊 RSI: **"
        f"{rsi_value:.1f}**\n\n"
        if rsi_value is not None
        else
        "📊 RSI: **--**\n\n"
    ) + (
        f"💰 Price: **"
        f"{fmt_price(price)}**\n"
        f"{cancel_text}\n\n"
        f"⏱️ Entry after: **"
        f"{delay_minutes} minutes**\n"
        f"🕐 **ENTRY TIME: "
        f"{time_string(entry_time)} 🇩🇿**\n"
        "━━━━━━━━━━━━━━━━━━"
    )


def build_result_text(
    signal,
    result,
):
    if not signal:
        return "❌ No pending signal found."

    symbol = signal.get(
        "symbol"
    )

    direction = signal.get(
        "direction"
    )

    stage = signal.get(
        "stage"
    )

    if result == "WIN":

        if stage == "BASE":

            return (
                "✅ **BASE WIN**\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "🎯 Base signal completed successfully.\n"
                "🔎 Searching for a new BASE setup..."
            )

        return (
            "✅ **RECOVERY WIN**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol}\n"
            f"{direction_emoji(direction)} "
            f"{direction}\n\n"
            "♻️ Recovery won after BASE loss.\n"
            "🔎 Searching for a new BASE setup..."
        )

    if result == "LOSS":

        if stage == "BASE":

            return (
                "❌ **BASE LOSS**\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "♻️ Recovery **1/1 ALLOWED**\n"
                "⏳ Waiting 2 minutes..."
            )

        return (
            "❌ **RECOVERY LOSS**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol}\n"
            f"{direction_emoji(direction)} "
            f"{direction}\n\n"
            "⛔ No second martingale.\n"
            "🔎 Cycle ended."
        )

    return "Unknown result."


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_telegram_message(
    text_message
):
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


def send_telegram_sync(
    text_message
):
    global telegram_loop

    if telegram_loop is None:
        return False

    try:

        future = (
            asyncio.run_coroutine_threadsafe(
                send_telegram_message(
                    text_message
                ),
                telegram_loop,
            )
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

def send_one_signal(
    candidate,
    stage,
):
    global cycle

    if not candidate:
        return False

    symbol = candidate[
        "symbol"
    ]

    direction = candidate[
        "direction"
    ]

    analysis = candidate[
        "analysis"
    ]

    timeframe = candidate.get(
        "timeframe",
        DEFAULT_TIMEFRAME,
    )

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

            if cycle.get(
                "active"
            ):
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

            if not cycle.get(
                "active"
            ):
                return False

            if (
                cycle.get(
                    "recovery_count",
                    0,
                )
                >= RECOVERY_LIMIT
            ):
                return False

            cycle["stage"] = "RECOVERY"

            cycle["recovery_count"] += 1

        else:
            return False

        delay = entry_delay_for_timeframe(
            timeframe
        )

        entry_time = (
            now_algeria()
            + timedelta(
                seconds=delay
            )
        )

        cycle["base_entry_time"] = (
            entry_time
        )

        cycle["pending_signal"] = {
            "symbol": symbol,
            "direction": direction,
            "stage": stage,
            "analysis": analysis,
            "entry_time": entry_time,
            "created_at": time.time(),
            "timeframe": timeframe,
        }

    text_message = build_signal_text(
        symbol,
        direction,
        stage,
        analysis,
        entry_time,
        timeframe,
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
        timeframe,
    )

    logger.info(
        "SIGNAL SENT | %s | %s | %s | "
        "score=%s/%s gap=%s confidence=%s "
        "structure=%s breakout=%s/%s retest=%s",
        stage,
        symbol,
        direction,
        analysis.get("max_score"),
        MAX_SCORE,
        analysis.get("score_gap"),
        analysis.get("confidence"),
        analysis.get("structure"),
        analysis.get("breakout"),
        analysis.get("breakout_strength"),
        analysis.get("breakout_retest"),
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
            item.get("symbol")
            == symbol
            and item.get("direction")
            == direction
            and item.get("stage")
            == stage
            and item.get("result")
            == "PENDING"
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

        symbol = pending[
            "symbol"
        ]

        direction = pending[
            "direction"
        ]

        stage = pending[
            "stage"
        ]

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

        symbol = pending[
            "symbol"
        ]

        direction = pending[
            "direction"
        ]

        stage = pending[
            "stage"
        ]

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
                cycle.get(
                    "recovery_count",
                    0,
                )
                < RECOVERY_LIMIT
            ):

                cycle["stage"] = (
                    "WAIT_RECOVERY"
                )

                cycle[
                    "recovery_ready_at"
                ] = (
                    time.time()
                    + RECOVERY_WAIT_SECONDS
                )

                return True, "BASE_LOSS"

        # Recovery loss or unavailable.
        cycle["active"] = False
        cycle["stage"] = "IDLE"

        cycle["base_symbol"] = None
        cycle["base_direction"] = None
        cycle["base_price"] = None

        cycle["recovery_count"] = 0
        cycle["recovery_ready_at"] = None

    return True, "RECOVERY_LOSS"


# ============================================================
# SIGNAL WORKER
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
            # WAIT RECOVERY
            # ------------------------------------------------

            if (
                current_stage
                == "WAIT_RECOVERY"
                and recovery_ready_at
                is not None
            ):

                if (
                    current
                    >= recovery_ready_at
                ):

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

                    # No strong recovery.
                    with cycle_lock:

                        cycle["active"] = False
                        cycle["stage"] = "IDLE"

                        cycle[
                            "last_result"
                        ] = "RECOVERY_SKIPPED"

                        cycle[
                            "base_symbol"
                        ] = None

                        cycle[
                            "base_direction"
                        ] = None

                        cycle[
                            "base_price"
                        ] = None

                        cycle[
                            "recovery_count"
                        ] = 0

                        cycle[
                            "recovery_ready_at"
                        ] = None

                    stats[
                        "recovery_skips"
                    ] += 1

                    send_telegram_sync(
                        "🛑 **RECOVERY SKIPPED**\n"
                        "━━━━━━━━━━━━━━━━━━\n"
                        "❌ BASE خسرت.\n\n"
                        "لم يتم العثور على Recovery قوي.\n\n"
                        "الشروط:\n"
                        "• Score ≥ 16/20\n"
                        "• Gap ≥ 4\n"
                        "• ADX ≥ 25\n"
                        "• Structure مطابق\n"
                        "• Breakout STRONG\n"
                        "• Retest YES\n"
                        "• Fake Breakout NO\n\n"
                        "⛔ لا توجد مضاعفة ضعيفة.\n"
                        "🔎 نبدأ دورة BASE جديدة."
                    )

                    continue

            # ------------------------------------------------
            # ACTIVE
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
# HTTP SERVER
# ============================================================

class RequestHandler(
    BaseHTTPRequestHandler
):

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

        self.send_response(
            status
        )

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
                        "candles": len(
                            candles
                        ),
                        "age_seconds": (
                            data_age_seconds(
                                symbol
                            )
                        ),
                        "timeframe": normalize_timeframe(
                            item.get(
                                "timeframe",
                                DEFAULT_TIMEFRAME,
                            )
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

        if content_length <= 0:
            self.send_json(
                400,
                {
                    "error": "empty body"
                },
            )

            return

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

        if not isinstance(
            payload,
            dict,
        ):

            self.send_json(
                400,
                {
                    "error": "json must be object"
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

            if (
                supplied_key
                != MT4_API_KEY
            ):

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
        # TIMEFRAME
        # ----------------------------------------------------

        timeframe = normalize_timeframe(
            payload.get(
                "timeframe"
            )
            or payload.get(
                "Timeframe"
            )
            or payload.get(
                "tf"
            )
        )

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

        if (
            len(candles)
            < REQUIRED_TOTAL_CANDLES
        ):

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

        with data_lock:

            data_store[symbol] = {
                "candles": candles,
                "received_at": time.time(),
                "updated": now_string(),
                "timeframe": timeframe,
            }

        batch_id = (
            f"{symbol}_"
            f"{int(time.time())}"
        )

        logger.info(
            "MT4 DATA | %s | %s | candles=%s",
            symbol,
            timeframe,
            len(candles),
        )

        self.send_json(
            200,
            {
                "status": "accepted",
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
                "batch_id": batch_id,
                "batch_complete": True,
            },
        )


# ============================================================
# HTTP SERVER START
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
        "📡 MT4 data feed: ACTIVE\n"
        "📊 Default timeframe: M1\n"
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

        winrate = 0.0

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
        f"**{stats['recovery_losses']}**\n"

        f"🛑 Recovery Skips: "
        f"**{stats['recovery_skips']}**\n\n"

        f"⚙️ Cycle: **{stage}**\n"
        f"Active: **{active}**\n"
        f"Recovery: "
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
            "YES"
            if item.get(
                "breakout_retest"
            )
            else "NO"
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

            f"   Structure: "
            f"{item.get('structure') or 'NONE'}\n"

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
            categories[
                category
            ]["wins"] += 1

        else:
            categories[
                category
            ]["losses"] += 1

    lines = [
        "🚀 **BREAKOUT STATISTICS**",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for category, values in categories.items():

        wins = values[
            "wins"
        ]

        losses = values[
            "losses"
        ]

        total = wins + losses

        rate = (
            (
                wins
                / total
            )
            * 100
            if total > 0
            else 0
        )

        lines.append(
            f"\n**{category}**\n"
            f"🟢 Wins: {wins}\n"
            f"🔴 Losses: {losses}\n"
            f"🎯 Winrate: {rate:.1f}%"
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
                    [],
                )
            )

            age = data_age_seconds(
                symbol
            )

            age_text = (
                "--"
                if age is None
                else f"{age:.0f}s"
            )

            timeframe = normalize_timeframe(
                item.get(
                    "timeframe",
                    DEFAULT_TIMEFRAME,
                )
            )

            lines.append(
                f"📊 {symbol} | "
                f"{timeframe} | "
                f"candles={candles} | "
                f"age={age_text}"
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
            [],
        )
    )

    analysis = calculate_analysis(
        candles
    )

    if not analysis:

        await update.message.reply_text(
            "❌ Not enough data."
        )

        return

    direction = (
        analysis.get(
            "direction"
        )
        or "NONE"
    )

    valid, reason = validate_signal(
        analysis
    )

    timeframe = normalize_timeframe(
        item.get(
            "timeframe",
            DEFAULT_TIMEFRAME,
        )
    )

    text_message = (
        "🔬 **ANALYSIS**\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"

        f"🎯 Direction: **"
        f"{direction}**\n"

        f"🟢 UP: **"
        f"{analysis['up_score']}/{MAX_SCORE}**\n"

        f"🔴 DOWN: **"
        f"{analysis['down_score']}/{MAX_SCORE}**\n"

        f"📊 Gap: **"
        f"{analysis['score_gap']}**\n"

        f"🔥 Confidence: **"
        f"{analysis['confidence']}%**\n\n"

        f"📐 Structure: **"
        f"{analysis.get('structure') or 'NONE'}**\n"

        f"🚀 Breakout: **"
        f"{analysis.get('breakout') or 'NONE'}**\n"

        f"💥 Quality: **"
        f"{analysis.get('breakout_strength', 'NONE')}**\n"

        f"🔄 Retest: **"
        f"{'YES' if analysis.get('breakout_retest') else 'NO'}**\n"

        f"⚠️ Fake: **"
        f"{'YES' if analysis.get('breakout_fake') else 'NO'}**\n\n"

        f"📈 ADX: **"
        f"{analysis.get('adx'):.1f}**\n"
        if analysis.get("adx") is not None
        else
        "📈 ADX: **--**\n"
    ) + (
        f"📊 RSI: **"
        f"{analysis.get('rsi'):.1f}**\n\n"
        if analysis.get("rsi") is not None
        else
        "📊 RSI: **--**\n\n"
    ) + (
        f"🟢 Primary: **"
        f"{analysis.get('primary_direction') or 'NONE'}**\n"
        f"💰 Price: **"
        f"{fmt_price(analysis.get('price'))}**\n\n"
        f"✅ Validation: **"
        f"{valid}**\n"
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

    success, _ = handle_win()

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

    success, _ = handle_loss()

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

            "recovery_skips": 0,
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
        "✅ Recovery skips reset\n\n"
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

    telegram_loop = (
        asyncio.get_running_loop()
    )

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
        allowed_updates=[
            "message"
        ],
    )

    logger.info(
        "Telegram polling started"
    )

    while not shutdown_event.is_set():

        await asyncio.sleep(1)

    try:
        await telegram_app.updater.stop()
    except Exception:
        pass

    try:
        await telegram_app.stop()
    except Exception:
        pass

    try:
        await telegram_app.shutdown()
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "=========================================="
    )

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "DEFAULT_TIMEFRAME=%s",
        DEFAULT_TIMEFRAME,
    )

    logger.info(
        "SCORE=%s/%s",
        MIN_SCORE,
        MAX_SCORE,
    )

    logger.info(
        "MIN_GAP=%s",
        MIN_SCORE_GAP,
    )

    logger.info(
        "MAX_DATA_AGE=%ss",
        MAX_DATA_AGE_SECONDS,
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
    # HTTP
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
