```python
import os
import json
import logging
import threading
import asyncio
import time
import uuid
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

# MT4 feed
TIMEFRAME = "M1"

# Data
MIN_CANDLES = 60
REQUIRED_TOTAL_CANDLES = 61
MAX_DATA_AGE_SECONDS = 180

# Entry
ENTRY_DELAY_SECONDS = 120

# Base quality
MIN_SCORE = 14
MAX_SCORE = 20
MIN_SCORE_GAP = 3

# Confidence
MIN_CONFIDENCE = 78
MAX_CONFIDENCE = 89

# Trend
MIN_ADX = 21.0
STRONG_ADX = 26.0

# Candle filters
MIN_CANDLE_RANGE_RATIO = 0.30
MAX_CANDLE_RANGE_RATIO = 2.30

# Price extension filter
MAX_EMA21_DISTANCE_ATR = 1.25
MAX_EMA50_DISTANCE_ATR = 2.00

# Recovery
RECOVERY_LIMIT = 1
RECOVERY_WAIT_SECONDS = 120

RECOVERY_MIN_SCORE = 17
RECOVERY_MIN_GAP = 4
RECOVERY_MIN_ADX = 26.0

# Repeated setup protection
SETUP_REPEAT_BLOCK_SECONDS = 300
FRESH_SETUP_ATR_RATIO = 0.40

# Signal quality
MIN_TREND_CANDLES = 3

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
    return "DOWN" if direction == "UP" else "UP"


def is_owner(update):
    if not update or not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


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

    result.sort(
        key=lambda x: (
            safe_float(x.get("time"), 0)
            if isinstance(x.get("time"), (int, float))
            else str(x.get("time", ""))
        )
    )

    return result


def get_closed_candles(candles):
    """
    MT4 can send the currently forming candle as the last candle.
    We intentionally remove it.
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
        current = (
            (value - previous) * multiplier
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
            (avg_gain * (period - 1))
            + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1))
            + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


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

    for i in range(
        max(0, len(candles) - 3),
        len(candles),
    ):
        sub = candles[
            max(0, i - period + 1): i + 1
        ]

        hi = max(x["high"] for x in sub)
        lo = min(x["low"] for x in sub)

        if hi == lo:
            recent_k.append(50.0)
        else:
            recent_k.append(
                (
                    (candles[i]["close"] - lo)
                    / (hi - lo)
                ) * 100.0
            )

    d = (
        sum(recent_k) / len(recent_k)
        if recent_k
        else 50.0
    )

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

    deviation = (
        sum(abs(x - mean) for x in window)
        / period
    )

    if deviation == 0:
        return 0.0

    return (
        (typical[-1] - mean)
        / (0.015 * deviation)
    )


def momentum(candles, period=10):
    if len(candles) <= period:
        return None

    return (
        candles[-1]["close"]
        - candles[-1 - period]["close"]
    )


def macd(candles):
    closes = [c["close"] for c in candles]

    fast = ema(closes, 12)
    slow = ema(closes, 26)

    if fast is None or slow is None:
        return None, None

    macd_line = fast - slow

    series_fast = ema_series(closes, 12)
    series_slow = ema_series(closes, 26)

    length = min(
        len(series_fast),
        len(series_slow),
    )

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


def adx_di(candles, period=14):
    """
    Wilder-style directional movement calculation.
    """

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

    if len(trs) < period * 2:
        return None, None, None

    # Wilder smoothing
    tr14 = sum(trs[:period])
    plus14 = sum(plus_dm[:period])
    minus14 = sum(minus_dm[:period])

    dx_values = []
    plus_di_last = 0.0
    minus_di_last = 0.0

    for i in range(period, len(trs)):

        if i > period:
            tr14 = (
                tr14
                - (tr14 / period)
                + trs[i]
            )

            plus14 = (
                plus14
                - (plus14 / period)
                + plus_dm[i]
            )

            minus14 = (
                minus14
                - (minus14 / period)
                + minus_dm[i]
            )

        if tr14 <= 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = (
                100.0 * plus14 / tr14
            )

            minus_di = (
                100.0 * minus14 / tr14
            )

        denominator = plus_di + minus_di

        if denominator <= 0:
            dx = 0.0
        else:
            dx = (
                abs(plus_di - minus_di)
                / denominator
            ) * 100.0

        dx_values.append(dx)

        plus_di_last = plus_di
        minus_di_last = minus_di

    if len(dx_values) < period:
        return None, None, None

    adx = sum(dx_values[-period:]) / period

    return (
        adx,
        plus_di_last,
        minus_di_last,
    )


# ============================================================
# PRICE ACTION
# ============================================================


def candle_metrics(candles):
    if not candles:
        return {}

    c = candles[-1]

    body = abs(
        c["close"] - c["open"]
    )

    upper_wick = (
        c["high"]
        - max(c["open"], c["close"])
    )

    lower_wick = (
        min(c["open"], c["close"])
        - c["low"]
    )

    total_range = (
        c["high"] - c["low"]
    )

    body_ratio = (
        body / total_range
        if total_range > 0
        else 0.0
    )

    close_position = (
        (
            c["close"]
            - c["low"]
        )
        / total_range
        if total_range > 0
        else 0.5
    )

    return {
        "body": body,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "range": total_range,
        "body_ratio": body_ratio,
        "bullish": c["close"] > c["open"],
        "bearish": c["close"] < c["open"],
        "close_position": close_position,
    }


def structure_direction(candles):
    """
    Stronger structure detector.

    UP:
      - recent swing highs rising
      - recent swing lows rising

    DOWN:
      - recent swing highs falling
      - recent swing lows falling
    """

    if len(candles) < 12:
        return None

    recent = candles[-8:]

    highs = [
        c["high"]
        for c in recent
    ]

    lows = [
        c["low"]
        for c in recent
    ]

    # Compare first half against second half.
    first_high = max(highs[:4])
    second_high = max(highs[4:])

    first_low = min(lows[:4])
    second_low = min(lows[4:])

    if (
        second_high > first_high
        and second_low > first_low
    ):
        return "UP"

    if (
        second_high < first_high
        and second_low < first_low
    ):
        return "DOWN"

    return None


def recent_trend_direction(candles, count=4):
    if len(candles) < count + 1:
        return None

    recent = candles[-count:]

    bullish = sum(
        1
        for c in recent
        if c["close"] > c["open"]
    )

    bearish = sum(
        1
        for c in recent
        if c["close"] < c["open"]
    )

    if bullish >= 3:
        return "UP"

    if bearish >= 3:
        return "DOWN"

    return None


def candle_confirmation(candles, direction):
    """
    Last candle must show directional acceptance.
    """

    if len(candles) < 2:
        return False

    metrics = candle_metrics(candles)

    if not metrics:
        return False

    body_ratio = metrics["body_ratio"]
    close_position = metrics["close_position"]

    if direction == "UP":
        return (
            metrics["bullish"]
            and body_ratio >= 0.50
            and close_position >= 0.65
        )

    if direction == "DOWN":
        return (
            metrics["bearish"]
            and body_ratio >= 0.50
            and close_position <= 0.35
        )

    return False


# ============================================================
# LIQUIDITY
# ============================================================


def liquidity_analysis(candles):
    """
    Detects a meaningful liquidity sweep.

    Returns:
        direction
        strength
    """

    result = {
        "direction": None,
        "strength": 0,
    }

    if len(candles) < 15:
        return result

    last = candles[-1]
    previous = candles[-11:-1]

    highest = max(
        c["high"]
        for c in previous
    )

    lowest = min(
        c["low"]
        for c in previous
    )

    atr_value = atr(candles, 14)

    if atr_value is None or atr_value <= 0:
        return result

    # Sweep above resistance -> bearish rejection
    if (
        last["high"] > highest
        and last["close"] < highest
    ):
        rejection = (
            last["high"]
            - last["close"]
        )

        if rejection >= atr_value * 0.15:
            result["direction"] = "DOWN"
            result["strength"] = 2
        else:
            result["direction"] = "DOWN"
            result["strength"] = 1

    # Sweep below support -> bullish rejection
    elif (
        last["low"] < lowest
        and last["close"] > lowest
    ):
        rejection = (
            last["close"]
            - last["low"]
        )

        if rejection >= atr_value * 0.15:
            result["direction"] = "UP"
            result["strength"] = 2
        else:
            result["direction"] = "UP"
            result["strength"] = 1

    return result


def liquidity_direction(candles):
    return liquidity_analysis(candles)["direction"]


# ============================================================
# BREAKOUT
# ============================================================


def breakout_analysis(candles):
    """
    Strict breakout detector.

    A valid breakout needs:
      - close outside a real previous range
      - meaningful distance from level
      - acceptable candle body
      - optional retest confirmation

    Fake breakout:
      - wick crosses level
      - candle closes back inside
    """

    result = {
        "direction": None,
        "strength": "NONE",
        "retest": False,
        "fake": False,
        "level": None,
        "score": 0,
    }

    if len(candles) < 30:
        return result

    last = candles[-1]

    previous = candles[-21:-1]

    if len(previous) < 15:
        return result

    highest = max(
        c["high"]
        for c in previous
    )

    lowest = min(
        c["low"]
        for c in previous
    )

    atr_value = atr(candles, 14)

    if atr_value is None or atr_value <= 0:
        return result

    candle_range = (
        last["high"] - last["low"]
    )

    if candle_range <= 0:
        return result

    body_ratio = (
        abs(last["close"] - last["open"])
        / candle_range
    )

    # --------------------------------------------------------
    # UP BREAKOUT
    # --------------------------------------------------------

    if last["close"] > highest:

        distance = (
            last["close"] - highest
        )

        result["direction"] = "UP"
        result["level"] = highest

        if (
            distance >= atr_value * 0.25
            and body_ratio >= 0.55
        ):
            result["strength"] = "STRONG"
            result["score"] = 2

        elif distance >= atr_value * 0.12:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # DOWN BREAKOUT
    # --------------------------------------------------------

    elif last["close"] < lowest:

        distance = (
            lowest - last["close"]
        )

        result["direction"] = "DOWN"
        result["level"] = lowest

        if (
            distance >= atr_value * 0.25
            and body_ratio >= 0.55
        ):
            result["strength"] = "STRONG"
            result["score"] = 2

        elif distance >= atr_value * 0.12:
            result["strength"] = "WEAK"
            result["score"] = 1

    # --------------------------------------------------------
    # FAKE UP
    # --------------------------------------------------------

    elif (
        last["high"] > highest
        and last["close"] <= highest
    ):

        result["direction"] = "UP"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = highest
        result["score"] = -2

    # --------------------------------------------------------
    # FAKE DOWN
    # --------------------------------------------------------

    elif (
        last["low"] < lowest
        and last["close"] >= lowest
    ):

        result["direction"] = "DOWN"
        result["strength"] = "FAKE"
        result["fake"] = True
        result["level"] = lowest
        result["score"] = -2

    # --------------------------------------------------------
    # RETEST
    # --------------------------------------------------------

    if (
        result["direction"] in ("UP", "DOWN")
        and not result["fake"]
        and result["level"] is not None
    ):

        direction = result["direction"]
        level = result["level"]

        tolerance = atr_value * 0.12

        # Search previous 5 closed candles for a touch.
        for i in range(
            max(0, len(candles) - 7),
            len(candles) - 1,
        ):

            c = candles[i]

            if direction == "UP":

                touched = (
                    c["low"] <= level + tolerance
                    and c["high"] >= level - tolerance
                )

                rejected = (
                    c["close"] > level
                )

                if touched and rejected:
                    result["retest"] = True
                    break

            else:

                touched = (
                    c["high"] >= level - tolerance
                    and c["low"] <= level + tolerance
                )

                rejected = (
                    c["close"] < level
                )

                if touched and rejected:
                    result["retest"] = True
                    break

    if result["retest"]:
        result["score"] += 1

    return result


def breakout_direction(candles):
    return breakout_analysis(candles)["direction"]


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

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    rsi14 = rsi(candles, 14)

    adx, plus_di, minus_di = adx_di(
        candles,
        14,
    )

    atr14 = atr(candles, 14)

    stoch_k, stoch_d = stochastic(
        candles,
        14,
    )

    cci20 = cci(
        candles,
        20,
    )

    mom10 = momentum(
        candles,
        10,
    )

    macd_line, macd_signal = macd(
        candles
    )

    williams = williams_r(
        candles,
        14,
    )

    structure = structure_direction(
        candles
    )

    recent_trend = recent_trend_direction(
        candles,
        MIN_TREND_CANDLES,
    )

    breakout_info = breakout_analysis(
        candles
    )

    breakout = breakout_info["direction"]
    breakout_strength = breakout_info["strength"]
    breakout_retest = breakout_info["retest"]
    breakout_fake = breakout_info["fake"]
    breakout_score = breakout_info["score"]
    breakout_level = breakout_info["level"]

    liquidity_info = liquidity_analysis(
        candles
    )

    liquidity = liquidity_info["direction"]
    liquidity_strength = liquidity_info["strength"]

    last = candles[-1]

    metrics = candle_metrics(
        candles
    )

    up_score = 0
    down_score = 0

    reasons_up = []
    reasons_down = []
    rejection_reasons = []

    # ========================================================
    # 1. STRUCTURE — 4 POINTS
    # ========================================================

    if structure == "UP":
        up_score += 4
        reasons_up.append("Structure UP")

    elif structure == "DOWN":
        down_score += 4
        reasons_down.append("Structure DOWN")

    # ========================================================
    # 2. BREAKOUT / RETEST — 4 POINTS
    # ========================================================

    if breakout == "UP" and not breakout_fake:

        if breakout_strength == "STRONG":
            up_score += 2
            reasons_up.append("Strong Breakout UP")

        elif breakout_strength == "WEAK":
            up_score += 1
            reasons_up.append("Weak Breakout UP")

        if breakout_retest:
            up_score += 2
            reasons_up.append("Retest UP")

    elif breakout == "DOWN" and not breakout_fake:

        if breakout_strength == "STRONG":
            down_score += 2
            reasons_down.append("Strong Breakout DOWN")

        elif breakout_strength == "WEAK":
            down_score += 1
            reasons_down.append("Weak Breakout DOWN")

        if breakout_retest:
            down_score += 2
            reasons_down.append("Retest DOWN")

    # ========================================================
    # 3. EMA TREND — 3 POINTS
    # ========================================================

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:
            up_score += 2
            reasons_up.append("EMA 9>21")

        elif ema9 < ema21:
            down_score += 2
            reasons_down.append("EMA 9<21")

    if (
        ema21 is not None
        and ema50 is not None
    ):

        if ema21 > ema50:
            up_score += 1

        elif ema21 < ema50:
            down_score += 1

    # ========================================================
    # 4. ADX / DI — 3 POINTS
    # ========================================================

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
    ):

        di_gap = abs(
            plus_di - minus_di
        )

        if plus_di > minus_di:

            if adx >= STRONG_ADX:
                up_score += 3
            else:
                up_score += 2

            reasons_up.append(
                f"DI+ {plus_di:.1f}"
            )

        elif minus_di > plus_di:

            if adx >= STRONG_ADX:
                down_score += 3
            else:
                down_score += 2

            reasons_down.append(
                f"DI- {minus_di:.1f}"
            )

    # ========================================================
    # 5. MOMENTUM / RSI — 2 POINTS
    # ========================================================

    if rsi14 is not None:

        if 52 <= rsi14 < 70:
            up_score += 2
            reasons_up.append(
                f"RSI {rsi14:.1f}"
            )

        elif 30 < rsi14 <= 48:
            down_score += 2
            reasons_down.append(
                f"RSI {rsi14:.1f}"
            )

        # Neutral 48-52 gives no point.
        # Extreme RSI does NOT add score.

    # ========================================================
    # 6. LIQUIDITY — 2 POINTS
    # ========================================================

    if liquidity == "UP":

        if liquidity_strength >= 2:
            up_score += 2
        else:
            up_score += 1

        reasons_up.append("Liquidity Sweep UP")

    elif liquidity == "DOWN":

        if liquidity_strength >= 2:
            down_score += 2
        else:
            down_score += 1

        reasons_down.append(
            "Liquidity Sweep DOWN"
        )

    # ========================================================
    # 7. CANDLE CONFIRMATION — 2 POINTS
    # ========================================================

    candle_up = candle_confirmation(
        candles,
        "UP",
    )

    candle_down = candle_confirmation(
        candles,
        "DOWN",
    )

    if candle_up:
        up_score += 2
        reasons_up.append("Bull Candle")

    elif candle_down:
        down_score += 2
        reasons_down.append("Bear Candle")

    # ========================================================
    # CAP
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
    # PRIMARY TREND
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

    if recent_trend == "UP":
        primary_votes += 1

    elif recent_trend == "DOWN":
        primary_votes -= 1

    if primary_votes > 0:
        primary_direction = "UP"

    elif primary_votes < 0:
        primary_direction = "DOWN"

    else:
        primary_direction = None

    # ========================================================
    # TREND ALIGNMENT
    # ========================================================

    trend_alignment = 0

    if direction == "UP":

        if structure == "UP":
            trend_alignment += 1

        if recent_trend == "UP":
            trend_alignment += 1

        if ema9 is not None and ema21 is not None:
            if ema9 > ema21:
                trend_alignment += 1

        if (
            plus_di is not None
            and minus_di is not None
            and plus_di > minus_di
        ):
            trend_alignment += 1

    elif direction == "DOWN":

        if structure == "DOWN":
            trend_alignment += 1

        if recent_trend == "DOWN":
            trend_alignment += 1

        if ema9 is not None and ema21 is not None:
            if ema9 < ema21:
                trend_alignment += 1

        if (
            plus_di is not None
            and minus_di is not None
            and minus_di > plus_di
        ):
            trend_alignment += 1

    # ========================================================
    # PRICE EXTENSION
    # ========================================================

    price_extended = False

    distance_ema21_atr = None
    distance_ema50_atr = None

    if (
        atr14 is not None
        and atr14 > 0
        and ema21 is not None
    ):

        distance_ema21_atr = (
            abs(last["close"] - ema21)
            / atr14
        )

        if (
            distance_ema21_atr
            > MAX_EMA21_DISTANCE_ATR
        ):
            price_extended = True
            rejection_reasons.append(
                "Price too far from EMA21"
            )

    if (
        atr14 is not None
        and atr14 > 0
        and ema50 is not None
    ):

        distance_ema50_atr = (
            abs(last["close"] - ema50)
            / atr14
        )

        if (
            distance_ema50_atr
            > MAX_EMA50_DISTANCE_ATR
        ):
            rejection_reasons.append(
                "Price extended from EMA50"
            )

    # ========================================================
    # ABNORMAL CANDLE
    # ========================================================

    abnormal_candle = False

    average_range = None
    range_ratio = None

    if len(candles) >= 21:

        recent_ranges = [
            c["high"] - c["low"]
            for c in candles[-21:-1]
        ]

        if recent_ranges:

            average_range = (
                sum(recent_ranges)
                / len(recent_ranges)
            )

            current_range = (
                last["high"] - last["low"]
            )

            if average_range > 0:

                range_ratio = (
                    current_range
                    / average_range
                )

                if (
                    range_ratio
                    > MAX_CANDLE_RANGE_RATIO
                ):
                    abnormal_candle = True
                    rejection_reasons.append(
                        "Abnormal candle"
                    )

                if (
                    range_ratio
                    < MIN_CANDLE_RANGE_RATIO
                ):
                    rejection_reasons.append(
                        "Very weak candle"
                    )

    # ========================================================
    # RSI EXTREME
    # ========================================================

    extreme_chase = False

    if rsi14 is not None:

        if (
            direction == "UP"
            and rsi14 >= 76
        ):
            extreme_chase = True
            rejection_reasons.append(
                "RSI overbought chase"
            )

        elif (
            direction == "DOWN"
            and rsi14 <= 24
        ):
            extreme_chase = True
            rejection_reasons.append(
                "RSI oversold chase"
            )

    # ========================================================
    # CONFLICT DETECTION
    # ========================================================

    conflict_reasons = []

    if (
        direction is not None
        and primary_direction is not None
        and direction != primary_direction
    ):
        conflict_reasons.append(
            "Primary direction conflict"
        )

    if (
        direction is not None
        and structure is not None
        and structure != direction
    ):
        conflict_reasons.append(
            "Structure conflict"
        )

    if (
        direction is not None
        and recent_trend is not None
        and recent_trend != direction
    ):
        conflict_reasons.append(
            "Recent candle trend conflict"
        )

    if (
        direction == "UP"
        and adx is not None
        and adx >= STRONG_ADX
        and minus_di is not None
        and plus_di is not None
        and minus_di > plus_di
    ):
        conflict_reasons.append(
            "Strong bearish DI"
        )

    if (
        direction == "DOWN"
        and adx is not None
        and adx >= STRONG_ADX
        and plus_di is not None
        and minus_di is not None
        and plus_di > minus_di
    ):
        conflict_reasons.append(
            "Strong bullish DI"
        )

    # ========================================================
    # SCORE QUALITY
    # ========================================================

    score_quality = "WEAK"

    if (
        max_score >= 17
        and score_gap >= 5
    ):
        score_quality = "VERY_STRONG"

    elif (
        max_score >= 15
        and score_gap >= 4
    ):
        score_quality = "STRONG"

    elif max_score >= 14:
        score_quality = "VALID"

    # ========================================================
    # CONFIDENCE
    # ========================================================

    confidence = 70

    confidence += min(
        10,
        score_gap * 2,
    )

    confidence += min(
        5,
        abs(primary_votes),
    )

    if adx is not None:

        if adx >= 30:
            confidence += 5

        elif adx >= 26:
            confidence += 4

        elif adx >= 21:
            confidence += 2

    if (
        direction is not None
        and structure == direction
    ):
        confidence += 4

    if (
        direction is not None
        and recent_trend == direction
    ):
        confidence += 3

    if (
        direction is not None
        and breakout == direction
        and not breakout_fake
    ):

        if breakout_strength == "STRONG":
            confidence += 4

        elif breakout_strength == "WEAK":
            confidence += 1

        if breakout_retest:
            confidence += 4

    if (
        direction is not None
        and liquidity == direction
    ):
        confidence += 2

    if trend_alignment >= 4:
        confidence += 3

    if conflict_reasons:
        confidence -= 10

    if price_extended:
        confidence -= 6

    if extreme_chase:
        confidence -= 6

    if abnormal_candle:
        confidence -= 7

    if breakout_fake:
        confidence -= 8

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

        "max_score": max_score,
        "score_gap": score_gap,

        "confidence": confidence,

        "score_quality": score_quality,

        "primary_direction": primary_direction,

        "primary_votes": primary_votes,

        "trend_alignment": trend_alignment,

        "recent_trend": recent_trend,

        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,

        "rsi": rsi14,

        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,

        "atr": atr14,

        "stoch_k": stoch_k,
        "stoch_d": stoch_d,

        "cci": cci20,
        "momentum": mom10,

        "macd": macd_line,
        "macd_signal": macd_signal,

        "williams": williams,

        "structure": structure,

        "breakout": breakout,
        "breakout_strength": breakout_strength,
        "breakout_retest": breakout_retest,
        "breakout_fake": breakout_fake,
        "breakout_score": breakout_score,
        "breakout_level": breakout_level,

        "liquidity": liquidity,
        "liquidity_strength": liquidity_strength,

        "body_ratio": metrics.get(
            "body_ratio",
            0,
        ),

        "close_position": metrics.get(
            "close_position",
            0.5,
        ),

        "abnormal_candle": abnormal_candle,

        "price_extended": price_extended,

        "distance_ema21_atr": distance_ema21_atr,
        "distance_ema50_atr": distance_ema50_atr,

        "extreme_chase": extreme_chase,

        "conflict": bool(conflict_reasons),
        "conflict_reasons": conflict_reasons,

        "rejection_reasons": rejection_reasons,

        "reasons_up": reasons_up,
        "reasons_down": reasons_down,

        "candle_time": last.get("time"),

        "price": last["close"],
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
# VALIDATION
# ============================================================


def validate_signal(analysis):
    if not analysis:
        return False, "no analysis"

    direction = analysis.get("direction")

    if direction not in ("UP", "DOWN"):
        return False, "no clear direction"

    if analysis.get("max_score", 0) < MIN_SCORE:
        return False, "score too low"

    if analysis.get("score_gap", 0) < MIN_SCORE_GAP:
        return False, "score gap too small"

    if analysis.get("primary_direction") != direction:
        return False, "primary direction conflict"

    if analysis.get("structure") != direction:
        return False, "structure conflict"

    if analysis.get("recent_trend") != direction:
        return False, "recent trend conflict"

    if analysis.get("trend_alignment", 0) < 3:
        return False, "trend alignment too weak"

    if analysis.get("conflict"):
        return False, "market conflict"

    if analysis.get("abnormal_candle"):
        return False, "abnormal candle"

    if analysis.get("price_extended"):
        return False, "price too extended"

    if analysis.get("extreme_chase"):
        return False, "extreme RSI chase"

    if analysis.get("breakout_fake"):
        return False, "fake breakout"

    if analysis.get("breakout") == direction:

        if analysis.get("breakout_strength") != "STRONG":
            return False, "breakout not strong"

        # A strong breakout without retest can still be valid
        # only if the complete trend is exceptionally strong.
        if not analysis.get("breakout_retest"):

            adx = analysis.get("adx") or 0

            if (
                adx < 30
                or analysis.get("max_score", 0) < 16
                or analysis.get("score_gap", 0) < 4
            ):
                return False, "breakout needs retest"

    adx = analysis.get("adx")

    if adx is None or adx < MIN_ADX:
        return False, "ADX too weak"

    # Avoid entering when RSI is almost neutral.
    rsi_value = analysis.get("rsi")

    if rsi_value is not None:

        if (
            direction == "UP"
            and rsi_value < 52
        ):
            return False, "RSI not bullish"

        if (
            direction == "DOWN"
            and rsi_value > 48
        ):
            return False, "RSI not bearish"

    # Candle confirmation
    body_ratio = analysis.get(
        "body_ratio",
        0,
    )

    if body_ratio < 0.40:
        return False, "weak candle body"

    return True, "valid"


# ============================================================
# FRESH SETUP CONTROL
# ============================================================


def fresh_base_setup(
    symbol,
    direction,
    analysis,
):
    key = f"{symbol}:{direction}"

    previous = recent_base_setups.get(key)

    if not previous:
        return True

    elapsed = (
        time.time()
        - previous.get("time", 0)
    )

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
        and abs(
            current_rsi - previous_rsi
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

    if len(candles) < REQUIRED_TOTAL_CANDLES:
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

    direction = analysis["direction"]

    return {
        "symbol": symbol,
        "analysis": analysis,
        "direction": direction,
        "validation": reason,
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

    current_symbol = cycle.get(
        "base_symbol"
    )

    current_direction = cycle.get(
        "base_direction"
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

        analysis = candidate["analysis"]
        direction = candidate["direction"]

        if not fresh_base_setup(
            symbol,
            direction,
            analysis,
        ):
            continue

        quality = 0.0

        # Core score
        quality += (
            analysis["max_score"]
            * 12
        )

        quality += (
            analysis["score_gap"]
            * 8
        )

        # Trend
        quality += (
            analysis.get(
                "trend_alignment",
                0
            )
            * 8
        )

        # ADX
        adx = analysis.get("adx") or 0
        quality += adx * 1.2

        # Structure
        if analysis.get("structure") == direction:
            quality += 12

        # Recent trend
        if analysis.get("recent_trend") == direction:
            quality += 10

        # Breakout
        if (
            analysis.get("breakout")
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
                quality += 15

            if analysis.get(
                "breakout_retest"
            ):
                quality += 15

        # Liquidity
        if (
            analysis.get("liquidity")
            == direction
        ):
            quality += (
                analysis.get(
                    "liquidity_strength",
                    0
                )
                * 5
            )

        # Candle
        if analysis.get(
            "body_ratio",
            0
        ) >= 0.55:
            quality += 6

        # Confidence
        quality += (
            analysis.get(
                "confidence",
                0
            )
            * 0.5
        )

        # Penalties
        if current_symbol == symbol:
            quality -= 20

        if current_direction == direction:
            quality -= 8

        if symbol in recent_symbols:
            quality -= 5

        if analysis.get(
            "price_extended"
        ):
            quality -= 30

        if analysis.get(
            "breakout_fake"
        ):
            quality -= 50

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
    """
    Recovery is intentionally much stricter.

    Requirements:
      score >= 17
      gap >= 4
      ADX >= 26
      structure aligned
      recent trend aligned
      strong breakout
      retest
      no fake breakout
      no extended price
    """

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

        analysis = candidate["analysis"]
        direction = candidate["direction"]

        score = (
            analysis.get(
                "max_score",
                0
            )
            or 0
        )

        gap = (
            analysis.get(
                "score_gap",
                0
            )
            or 0
        )

        adx = (
            analysis.get(
                "adx"
            )
        )

        if score < RECOVERY_MIN_SCORE:
            continue

        if gap < RECOVERY_MIN_GAP:
            continue

        if adx is None or adx < RECOVERY_MIN_ADX:
            continue

        if analysis.get(
            "structure"
        ) != direction:
            continue

        if analysis.get(
            "recent_trend"
        ) != direction:
            continue

        if analysis.get(
            "breakout"
        ) != direction:
            continue

        if analysis.get(
            "breakout_strength"
        ) != "STRONG":
            continue

        if not analysis.get(
            "breakout_retest"
        ):
            continue

        if analysis.get(
            "breakout_fake"
        ):
            continue

        if analysis.get(
            "price_extended"
        ):
            continue

        quality = (
            score * 15
        )

        quality += (
            gap * 10
        )

        quality += (
            adx * 2
        )

        quality += 20  # structure
        quality += 20  # trend
        quality += 20  # strong breakout
        quality += 20  # retest

        if (
            analysis.get("liquidity")
            == direction
        ):
            quality += 10

        if direction != base_direction:
            quality += 5

        if symbol == base_symbol:
            quality -= 5

        quality += (
            analysis.get(
                "confidence",
                0
            )
            * 0.5
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
    signal_id=None,
):
    item = {
        "id": signal_id or str(
            uuid.uuid4()
        ),

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

        "score_gap": analysis.get(
            "score_gap"
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

        "recent_trend": analysis.get(
            "recent_trend"
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
                datetime
            )
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

    breakout = (
        analysis.get("breakout")
        or "NONE"
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

    recent_trend = analysis.get(
        "recent_trend"
    )

    adx = analysis.get(
        "adx"
    )

    rsi_value = analysis.get(
        "rsi"
    )

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

    stage_text = (
        "🎯 BASE TRADE"
        if stage == "BASE"
        else "♻️ RECOVERY 1/1"
    )

    cancel_word = (
        "below"
        if direction == "UP"
        else "above"
    )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {TIMEFRAME}\n\n"
        f"{stage_text}\n"
        f"{direction_emoji(direction)} "
        f"**{direction}**\n\n"
        f"🔥 Confidence: **{confidence}%**\n"
        f"🟢 UP Score: **{up_score}/{MAX_SCORE}**\n"
        f"🔴 DOWN Score: **{down_score}/{MAX_SCORE}**\n\n"
        f"📐 Structure: **{structure or 'NONE'}**\n"
        f"📈 Recent Trend: **{recent_trend or 'NONE'}**\n"
        f"🚀 Breakout: **{breakout}**\n"
        f"💥 Breakout Quality: **{quality_text}**\n"
        f"🔄 Retest: **{retest_text}**\n\n"
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
        f"💰 Price: **{fmt_price(price)}**\n\n"
        f"⏱️ Entry after: **2 minutes**\n"
        f"🕐 **ENTRY TIME: "
        f"{time_string(entry_time)} 🇩🇿**\n\n"
        f"🛑 Cancellation: "
        f"إلغاء إذا أغلقت شمعة "
        f"{cancel_word} السعر المرجعي\n"
        "━━━━━━━━━━━━━━━━━━"
    )


def build_result_text(
    signal,
    result,
):
    if not signal:
        return "❌ No pending signal found."

    symbol = signal.get("symbol")
    direction = signal.get("direction")
    stage = signal.get("stage")

    if result == "WIN":

        if stage == "BASE":
            return (
                "✅ **BASE WIN**\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol} | {TIMEFRAME}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "🎯 Base signal completed successfully.\n"
                "🔎 Searching for a new BASE setup..."
            )

        return (
            "✅ **RECOVERY WIN**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {TIMEFRAME}\n"
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
                f"📊 {symbol} | {TIMEFRAME}\n"
                f"{direction_emoji(direction)} "
                f"{direction}\n\n"
                "♻️ Recovery **1/1 ALLOWED**\n"
                "⏳ Waiting 2 minutes before recovery search..."
            )

        return (
            "❌ **RECOVERY LOSS**\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {TIMEFRAME}\n"
            f"{direction_emoji(direction)} "
            f"{direction}\n\n"
            "⛔ No second martingale.\n"
            "🔎 Cycle ended. Searching for a new BASE setup..."
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

            if (
                cycle.get(
                    "recovery_count",
                    0
                )
                >= RECOVERY_LIMIT
            ):
                return False

            cycle["stage"] = "RECOVERY"

            cycle["recovery_count"] += 1

        else:
            return False

        entry_time = (
            now_algeria()
            + timedelta(
                seconds=ENTRY_DELAY_SECONDS
            )
        )

        signal_id = str(
            uuid.uuid4()
        )

        cycle["base_entry_time"] = (
            entry_time
        )

        cycle["pending_signal"] = {
            "id": signal_id,
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
        signal_id,
    )

    logger.info(
        "SIGNAL SENT | %s | %s | %s | "
        "score=%s/%s gap=%s conf=%s "
        "adx=%s structure=%s breakout=%s retest=%s",
        stage,
        symbol,
        direction,
        analysis.get("max_score"),
        MAX_SCORE,
        analysis.get("score_gap"),
        analysis.get("confidence"),
        analysis.get("adx"),
        analysis.get("structure"),
        analysis.get("breakout"),
        analysis.get("breakout_retest"),
    )

    return True


# ============================================================
# RESULT MANAGEMENT
# ============================================================


def mark_history_result(
    signal_id,
    result,
):
    for item in reversed(history):

        if (
            item.get("id") == signal_id
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

        signal_id = pending.get("id")
        stage = pending["stage"]

        mark_history_result(
            signal_id,
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

        signal_id = pending.get("id")
        stage = pending["stage"]

        mark_history_result(
            signal_id,
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
                    0
                )
                < RECOVERY_LIMIT
            ):

                cycle["stage"] = (
                    "WAIT_RECOVERY"
                )

                cycle["recovery_ready_at"] = (
                    time.time()
                    + RECOVERY_WAIT_SECONDS
                )

                return True, "BASE_LOSS"

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
            # RECOVERY
            # ------------------------------------------------

            if (
                current_stage
                == "WAIT_RECOVERY"
                and recovery_ready_at
                is not None
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

                    with cycle_lock:

                        cycle["active"] = False
                        cycle["stage"] = "IDLE"
                        cycle["last_result"] = (
                            "RECOVERY_SKIPPED"
                        )

                        cycle["base_symbol"] = None
                        cycle["base_direction"] = None
                        cycle["base_price"] = None

                        cycle["recovery_count"] = 0
                        cycle["recovery_ready_at"] = None

                    stats["recovery_skips"] += 1

                    send_telegram_sync(
                        "🛑 **RECOVERY SKIPPED**\n"
                        "━━━━━━━━━━━━━━━━━━\n"
                        "❌ BASE خسرت.\n"
                        "🔎 تم فحص الـ setups المتاحة.\n\n"
                        "⚠️ لم يتم الدخول لأن Recovery "
                        "لم يحقق الشروط الصارمة.\n\n"
                        "• Score ≥ 17/20\n"
                        "• Gap ≥ 4\n"
                        "• ADX ≥ 26\n"
                        "• Structure مطابق\n"
                        "• Recent Trend مطابق\n"
                        "• Breakout STRONG\n"
                        "• Retest YES\n"
                        "• Fake Breakout = NO\n"
                        "• Price extension = NO\n\n"
                        "⛔ لا توجد مضاعفة.\n"
                        "🔎 دورة جديدة BASE."
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
# MT4 HTTP SERVER
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
            {"error": "not found"},
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
                {"error": "not found"},
            )

            return

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
                    {"error": "unauthorized"},
                )

                return

        symbol = (
            payload.get("symbol")
            or payload.get("Symbol")
            or payload.get("pair")
            or payload.get("Pair")
        )

        if not symbol:

            self.send_json(
                400,
                {"error": "symbol missing"},
            )

            return

        symbol = str(
            symbol
        ).strip().upper()

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

        with data_lock:

            data_store[symbol] = {
                "candles": candles,
                "received_at": time.time(),
                "updated": now_string(),
                "timeframe": TIMEFRAME,
            }

        batch_id = (
            f"{symbol}_"
            f"{int(time.time())}"
        )

        logger.info(
            "MT4 data accepted | %s | candles=%s",
            symbol,
            len(candles),
        )

        self.send_json(
            200,
            {
                "status": "accepted",
                "symbol": symbol,
                "candles": len(candles),
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
        "📡 MT4 data feed: active\n"
        "📊 Timeframe: M1\n"
        "🎯 Automatic signals: ON\n"
        "♻️ Recovery: 1/1\n"
        "🛡️ Accuracy filters: ON\n\n"
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
        f"🟢 BASE Signals: **{stats['base_signals']}**\n"
        f"🟢 BASE Wins: **{stats['base_wins']}**\n"
        f"🔴 BASE Losses: **{stats['base_losses']}**\n\n"
        f"♻️ Recovery Signals: **{stats['recovery_signals']}**\n"
        f"♻️ Recovery Wins: **{stats['recovery_wins']}**\n"
        f"♻️ Recovery Losses: **{stats['recovery_losses']}**\n"
        f"🛑 Recovery Skips: **{stats['recovery_skips']}**\n\n"
        f"⚙️ Cycle: **{stage}**\n"
        f"Active: **{active}**\n"
        f"Recovery count: **{recovery_count}/1**"
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
            f"| Gap "
            f"{item.get('score_gap')}\n"
            f"   Conf "
            f"{item.get('confidence')}%\n"
            f"   Structure: "
            f"{item.get('structure') or 'NONE'} "
            f"| Trend: "
            f"{item.get('recent_trend') or 'NONE'}\n"
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

            age_text = (
                "--"
                if age is None
                else f"{age:.0f}s"
            )

            lines.append(
                f"📊 {symbol} "
                f"| candles={candles} "
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
        candles
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
        f"📊 {symbol} | {TIMEFRAME}\n\n"
        f"🎯 Direction: **{direction}**\n"
        f"🟢 UP: **{analysis['up_score']}/{MAX_SCORE}**\n"
        f"🔴 DOWN: **{analysis['down_score']}/{MAX_SCORE}**\n"
        f"📏 Gap: **{analysis['score_gap']}**\n"
        f"🔥 Confidence: **{analysis['confidence']}%**\n\n"
        f"📐 Structure: **{analysis.get('structure') or 'NONE'}**\n"
        f"📈 Recent Trend: **{analysis.get('recent_trend') or 'NONE'}**\n"
        f"🎯 Primary: **{analysis.get('primary_direction') or 'NONE'}**\n"
        f"🧭 Alignment: **{analysis.get('trend_alignment', 0)}/4**\n"
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
        f"**{analysis.get('rsi'):.1f}**\n"
        if analysis.get('rsi') is not None
        else
        "📊 RSI: **--**\n"
    ) + (
        f"💰 Price: **{fmt_price(analysis.get('price'))}**\n\n"
        f"🔎 Extended: "
        f"**{analysis.get('price_extended')}**\n"
        f"⚠️ Conflict: "
        f"**{analysis.get('conflict')}**\n\n"
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
        "ZinoProSignalAI Accuracy Edition starting..."
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
        "MIN_SCORE_GAP=%s",
        MIN_SCORE_GAP,
    )

    logger.info(
        "ENTRY_DELAY=%ss",
        ENTRY_DELAY_SECONDS,
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
```
