 
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

OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
try:
    OWNER_ID = int(OWNER_ID_RAW)
except Exception:
    OWNER_ID = 0

MT4_API_KEY = (
    os.getenv("MT4_API_KEY", "").strip()
    or os.getenv("ZINO_API_KEY", "").strip()
)

PORT = int(os.getenv("PORT", "10000"))

ALGIERS_TZ = ZoneInfo("Africa/Algiers")

BOT_NAME = "ZinoProSignalAI"

# ------------------------------------------------------------
# SIGNAL FILTERS
# ------------------------------------------------------------

MIN_SCORE = 15
MIN_SCORE_GAP = 4

MIN_ADX = 20.0
STRONG_ADX = 25.0

MAX_BASE_CONFIDENCE = 89
MAX_RECOVERY_CONFIDENCE = 89

DATA_FRESH_SECONDS = 90

# Exactly one recovery.
MAX_RECOVERY = 1

# Prevent repeated BASE signals for the same direction.
BASE_COOLDOWN_SECONDS = 300

# Minimum price movement before allowing a new same-direction signal.
MIN_PRICE_MOVE_ATR = 0.35

# Minimum RSI change before allowing another same-direction setup.
MIN_RSI_CHANGE = 5.0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(BOT_NAME)


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.RLock()

latest_market = {}

last_signal = None

signal_history = []

wins = 0
losses = 0

recovery_used = 0
recovery_skips = 0

last_base_by_symbol = {}

pending_recovery = {}


# ============================================================
# BASIC HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def format_price(price, digits=5):
    try:
        return f"{float(price):.{digits}f}"
    except Exception:
        return str(price)


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def normalize_symbol(symbol):
    if not symbol:
        return "UNKNOWN"

    return str(symbol).upper().replace("/", "").replace(" ", "")


def normalize_timeframe(tf):
    if tf is None:
        return "M1"

    text = str(tf).upper().strip()

    aliases = {
        "1": "M1",
        "2": "M2",
        "3": "M3",
        "5": "M5",
        "15": "M15",
        "30": "M30",
        "60": "H1",
        "240": "H4",
        "1440": "D1",
        "1M": "M1",
        "2M": "M2",
        "3M": "M3",
        "5M": "M5",
        "15M": "M15",
        "30M": "M30",
        "1H": "H1",
        "4H": "H4",
        "1D": "D1",
    }

    return aliases.get(text, text)


def timeframe_minutes(tf):
    tf = normalize_timeframe(tf)

    mapping = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }

    return mapping.get(tf, 1)


def entry_delay_minutes(tf):
    """
    Binary-option style entry delay.

    M1 -> 2 minutes
    M2 -> 2 minutes
    M3 -> 3 minutes
    Higher TF -> one candle
    """

    tf = normalize_timeframe(tf)

    if tf == "M1":
        return 2

    if tf == "M2":
        return 2

    if tf == "M3":
        return 3

    return timeframe_minutes(tf)


def parse_timestamp(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                float(value),
                tz=ALGIERS_TZ,
            )
        except Exception:
            return None

    text = str(value).strip()

    if not text:
        return None

    try:
        text = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ALGIERS_TZ)

        return dt.astimezone(ALGIERS_TZ)

    except Exception:
        pass

    return None


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):
    if not isinstance(c, dict):
        return None

    timestamp = (
        c.get("time")
        if c.get("time") is not None
        else c.get("timestamp")
    )

    if timestamp is None:
        timestamp = c.get("date")

    o = (
        c.get("open")
        if c.get("open") is not None
        else c.get("o")
    )

    h = (
        c.get("high")
        if c.get("high") is not None
        else c.get("h")
    )

    l = (
        c.get("low")
        if c.get("low") is not None
        else c.get("l")
    )

    close = (
        c.get("close")
        if c.get("close") is not None
        else c.get("c")
    )

    volume = (
        c.get("volume")
        if c.get("volume") is not None
        else c.get("tick_volume", 0)
    )

    o = safe_float(o)
    h = safe_float(h)
    l = safe_float(l)
    close = safe_float(close)
    volume = safe_float(volume, 0)

    if None in (o, h, l, close):
        return None

    return {
        "time": timestamp,
        "open": o,
        "high": h,
        "low": l,
        "close": close,
        "volume": volume,
    }


def extract_candles(payload):
    raw = None

    for key in ("candles", "data", "bars", "ohlc"):
        if key in payload:
            raw = payload.get(key)
            break

    if isinstance(raw, dict):
        for key in ("candles", "data", "bars"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        return []

    result = []

    for item in raw:
        candle = normalize_candle(item)

        if candle is not None:
            result.append(candle)

    return result


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    result = []

    seed = sum(values[:period]) / period
    result.append(seed)

    multiplier = 2.0 / (period + 1.0)

    for value in values[period:]:
        previous = result[-1]
        current = (
            (value - previous) * multiplier
            + previous
        )
        result.append(current)

    return result[-1]


def ema_series(values, period):
    if len(values) < period:
        return []

    seed = sum(values[:period]) / period

    result = [seed]

    multiplier = 2.0 / (period + 1.0)

    for value in values[period:]:
        previous = result[-1]

        result.append(
            (value - previous) * multiplier
            + previous
        )

    return result


def sma(values, period):
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def true_ranges(candles):
    result = []

    previous_close = None

    for candle in candles:
        high = candle["high"]
        low = candle["low"]

        if previous_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        result.append(tr)

        previous_close = candle["close"]

    return result


def atr(candles, period=14):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def rsi(candles, period=14):
    if len(candles) < period + 1:
        return None

    closes = [x["close"] for x in candles]

    gains = []
    losses_ = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]

        if change > 0:
            gains.append(change)
            losses_.append(0.0)

        else:
            gains.append(0.0)
            losses_.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses_[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1))
            + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1))
            + losses_[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    window = candles[-period:]

    highest = max(x["high"] for x in window)
    lowest = min(x["low"] for x in window)

    if highest == lowest:
        return -50.0

    return (
        (highest - candles[-1]["close"])
        / (highest - lowest)
    ) * -100.0


def stochastic(candles, period=14):
    if len(candles) < period:
        return None

    window = candles[-period:]

    highest = max(x["high"] for x in window)
    lowest = min(x["low"] for x in window)

    if highest == lowest:
        return 50.0

    return (
        (candles[-1]["close"] - lowest)
        / (highest - lowest)
    ) * 100.0


def momentum(candles, period=5):
    if len(candles) <= period:
        return None

    return (
        candles[-1]["close"]
        - candles[-1 - period]["close"]
    )


def cci(candles, period=20):
    if len(candles) < period:
        return None

    typical = [
        (x["high"] + x["low"] + x["close"]) / 3.0
        for x in candles
    ]

    window = typical[-period:]

    average = sum(window) / period

    deviation = sum(
        abs(x - average)
        for x in window
    ) / period

    if deviation == 0:
        return 0.0

    return (
        (typical[-1] - average)
        / (0.015 * deviation)
    )


def macd(candles):
    closes = [x["close"] for x in candles]

    ema12 = ema_series(closes, 12)
    ema26 = ema_series(closes, 26)

    if not ema12 or not ema26:
        return None

    # Align from the end.
    length = min(len(ema12), len(ema26))

    macd_values = []

    for i in range(length):
        macd_values.append(
            ema12[-length + i]
            - ema26[-length + i]
        )

    if len(macd_values) < 9:
        return None

    signal = ema(macd_values, 9)

    if signal is None:
        return None

    return {
        "macd": macd_values[-1],
        "signal": signal,
        "hist": macd_values[-1] - signal,
    }


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None

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
        return None

    atr_value = sum(trs[-period:]) / period

    if atr_value <= 0:
        return None

    plus_di = (
        sum(plus_dm[-period:])
        / atr_value
    ) * 100.0 / period

    minus_di = (
        sum(minus_dm[-period:])
        / atr_value
    ) * 100.0 / period

    denominator = plus_di + minus_di

    if denominator <= 0:
        adx = 0.0
    else:
        adx = (
            abs(plus_di - minus_di)
            / denominator
        ) * 100.0

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# CANDLE ANALYSIS
# ============================================================

def candle_direction(candle):
    if candle["close"] > candle["open"]:
        return "UP"

    if candle["close"] < candle["open"]:
        return "DOWN"

    return "NONE"


def candle_strength(candle):
    body = abs(candle["close"] - candle["open"])
    full_range = candle["high"] - candle["low"]

    if full_range <= 0:
        return 0.0

    return body / full_range


def candle_quality(candles):
    if len(candles) < 3:
        return {
            "direction": "NONE",
            "points": 0,
            "quality": "NONE",
        }

    c1 = candles[-1]
    c2 = candles[-2]

    direction = candle_direction(c1)

    body_ratio = candle_strength(c1)

    if direction == "NONE":
        return {
            "direction": "NONE",
            "points": 0,
            "quality": "NONE",
        }

    points = 1

    if body_ratio >= 0.55:
        points += 1

    # Avoid rewarding an abnormally large candle.
    avg_range = sum(
        x["high"] - x["low"]
        for x in candles[-10:]
    ) / min(10, len(candles))

    current_range = c1["high"] - c1["low"]

    if avg_range > 0 and current_range > avg_range * 2.5:
        points = 0

    quality = "NORMAL"

    if body_ratio >= 0.65:
        quality = "STRONG"

    return {
        "direction": direction,
        "points": min(points, 2),
        "quality": quality,
    }


# ============================================================
# STRUCTURE
# ============================================================

def analyze_structure(candles):
    if len(candles) < 12:
        return {
            "direction": "NONE",
            "points": 0,
        }

    recent = candles[-8:]

    first = recent[:4]
    second = recent[4:]

    first_high = max(x["high"] for x in first)
    first_low = min(x["low"] for x in first)

    second_high = max(x["high"] for x in second)
    second_low = min(x["low"] for x in second)

    first_close = first[-1]["close"]
    second_close = second[-1]["close"]

    up = (
        second_high > first_high
        and second_low > first_low
        and second_close > first_close
    )

    down = (
        second_high < first_high
        and second_low < first_low
        and second_close < first_close
    )

    if up:
        return {
            "direction": "UP",
            "points": 3,
        }

    if down:
        return {
            "direction": "DOWN",
            "points": 3,
        }

    # Softer continuation structure.
    if (
        second_high > first_high
        and second_close > first_close
    ):
        return {
            "direction": "UP",
            "points": 2,
        }

    if (
        second_low < first_low
        and second_close < first_close
    ):
        return {
            "direction": "DOWN",
            "points": 2,
        }

    return {
        "direction": "NONE",
        "points": 0,
    }


# ============================================================
# LIQUIDITY
# ============================================================

def analyze_liquidity(candles):
    if len(candles) < 15:
        return {
            "direction": "NONE",
            "points": 0,
            "type": "NONE",
        }

    current = candles[-1]

    previous = candles[-11:-1]

    previous_high = max(x["high"] for x in previous)
    previous_low = min(x["low"] for x in previous)

    # Sell-side liquidity sweep:
    # price goes below previous low and closes back above it.
    if (
        current["low"] < previous_low
        and current["close"] > previous_low
    ):
        return {
            "direction": "UP",
            "points": 2,
            "type": "LOW SWEEP",
        }

    # Buy-side liquidity sweep:
    # price goes above previous high and closes back below it.
    if (
        current["high"] > previous_high
        and current["close"] < previous_high
    ):
        return {
            "direction": "DOWN",
            "points": 2,
            "type": "HIGH SWEEP",
        }

    return {
        "direction": "NONE",
        "points": 0,
        "type": "NONE",
    }


# ============================================================
# BREAKOUT + RETEST
# ============================================================

def analyze_breakout(candles, atr_value):
    if len(candles) < 25 or atr_value is None or atr_value <= 0:
        return {
            "direction": "NONE",
            "points": 0,
            "quality": "NONE",
            "retest": False,
            "fake": False,
        }

    current = candles[-1]

    lookback = candles[-21:-1]

    resistance = max(x["high"] for x in lookback)
    support = min(x["low"] for x in lookback)

    tolerance = atr_value * 0.20

    # --------------------------------------------------------
    # UP BREAKOUT
    # --------------------------------------------------------

    if current["close"] > resistance:
        distance = current["close"] - resistance

        strong = distance >= atr_value * 0.15

        quality = "STRONG" if strong else "WEAK"

        retest = False

        # Search last 4 candles for a touch/retest
        # around the broken resistance.
        for candle in candles[-5:-1]:
            if (
                candle["low"] <= resistance + tolerance
                and candle["high"] >= resistance - tolerance
                and candle["close"] >= resistance
            ):
                retest = True
                break

        return {
            "direction": "UP",
            "points": 3 if strong else 2,
            "quality": quality,
            "retest": retest,
            "fake": False,
        }

    # --------------------------------------------------------
    # DOWN BREAKOUT
    # --------------------------------------------------------

    if current["close"] < support:
        distance = support - current["close"]

        strong = distance >= atr_value * 0.15

        quality = "STRONG" if strong else "WEAK"

        retest = False

        for candle in candles[-5:-1]:
            if (
                candle["high"] >= support - tolerance
                and candle["low"] <= support + tolerance
                and candle["close"] <= support
            ):
                retest = True
                break

        return {
            "direction": "DOWN",
            "points": 3 if strong else 2,
            "quality": quality,
            "retest": retest,
            "fake": False,
        }

    # --------------------------------------------------------
    # FAKE BREAKOUT DETECTION
    # --------------------------------------------------------

    previous = candles[-2]

    if (
        previous["high"] > resistance
        and current["close"] < resistance
    ):
        return {
            "direction": "DOWN",
            "points": 0,
            "quality": "FAKE",
            "retest": False,
            "fake": True,
        }

    if (
        previous["low"] < support
        and current["close"] > support
    ):
        return {
            "direction": "UP",
            "points": 0,
            "quality": "FAKE",
            "retest": False,
            "fake": True,
        }

    return {
        "direction": "NONE",
        "points": 0,
        "quality": "NONE",
        "retest": False,
        "fake": False,
    }


# ============================================================
# SWING LEVELS
# ============================================================

def recent_swing_low(candles, lookback=8):
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]

    return min(x["low"] for x in window)


def recent_swing_high(candles, lookback=8):
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]

    return max(x["high"] for x in window)


def cancellation_level(candles, direction, atr_value):
    """
    Structural cancellation.

    UP:
        cancel below a real recent swing low.

    DOWN:
        cancel above a real recent swing high.

    A small ATR buffer is added so that the level is not
    immediately triggered by tiny market noise.
    """

    if atr_value is None or atr_value <= 0:
        return None

    current = candles[-1]["close"]

    buffer = atr_value * 0.15

    if direction == "UP":
        swing = recent_swing_low(candles, 8)

        if swing is None:
            return None

        level = swing - buffer

        # Ensure the cancellation is meaningfully below price.
        minimum_distance = atr_value * 0.20

        if current - level < minimum_distance:
            level = current - minimum_distance

        return level

    if direction == "DOWN":
        swing = recent_swing_high(candles, 8)

        if swing is None:
            return None

        level = swing + buffer

        minimum_distance = atr_value * 0.20

        if level - current < minimum_distance:
            level = current + minimum_distance

        return level

    return None


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def analyze_market(candles):
    if len(candles) < 61:
        return None

    # MT4 normally sends the current forming candle last.
    # We analyze closed candles only.
    closed = candles[:-1]

    if len(closed) < 60:
        return None

    closes = [x["close"] for x in closed]

    current = closed[-1]

    price = current["close"]

    atr_value = atr(closed, 14)

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema50 = ema(closes, 50)

    rsi_value = rsi(closed, 14)
    williams = williams_r(closed, 14)
    stoch = stochastic(closed, 14)
    mom = momentum(closed, 5)
    cci_value = cci(closed, 20)
    macd_value = macd(closed)
    adx_value = adx_di(closed, 14)

    if (
        atr_value is None
        or ema9 is None
        or ema21 is None
        or ema50 is None
        or rsi_value is None
        or adx_value is None
    ):
        return None

    structure = analyze_structure(closed)
    liquidity = analyze_liquidity(closed)
    breakout = analyze_breakout(closed, atr_value)
    candle = candle_quality(closed)

    up_score = 0
    down_score = 0

    # --------------------------------------------------------
    # STRUCTURE / 3
    # --------------------------------------------------------

    if structure["direction"] == "UP":
        up_score += structure["points"]

    elif structure["direction"] == "DOWN":
        down_score += structure["points"]

    # --------------------------------------------------------
    # BREAKOUT / 3
    # --------------------------------------------------------

    if breakout["direction"] == "UP":
        up_score += breakout["points"]

    elif breakout["direction"] == "DOWN":
        down_score += breakout["points"]

    # --------------------------------------------------------
    # LIQUIDITY / 2
    # --------------------------------------------------------

    if liquidity["direction"] == "UP":
        up_score += liquidity["points"]

    elif liquidity["direction"] == "DOWN":
        down_score += liquidity["points"]

    # --------------------------------------------------------
    # MOMENTUM / 2
    # --------------------------------------------------------

    if mom is not None:

        if mom > 0:
            up_score += 2

        elif mom < 0:
            down_score += 2

    # --------------------------------------------------------
    # CANDLE / 2
    # --------------------------------------------------------

    if candle["direction"] == "UP":
        up_score += candle["points"]

    elif candle["direction"] == "DOWN":
        down_score += candle["points"]

    # --------------------------------------------------------
    # RSI / 1
    # --------------------------------------------------------

    if 52 <= rsi_value < 68:
        up_score += 1

    elif 32 < rsi_value <= 48:
        down_score += 1

    # --------------------------------------------------------
    # OSCILLATORS / 2
    # --------------------------------------------------------

    oscillator_up = 0
    oscillator_down = 0

    if williams is not None:

        if williams > -50:
            oscillator_up += 1

        elif williams < -50:
            oscillator_down += 1

    if stoch is not None:

        if stoch > 50:
            oscillator_up += 1

        elif stoch < 50:
            oscillator_down += 1

    if cci_value is not None:

        if cci_value > 0:
            oscillator_up += 1

        elif cci_value < 0:
            oscillator_down += 1

    if macd_value is not None:

        if macd_value["hist"] > 0:
            oscillator_up += 1

        elif macd_value["hist"] < 0:
            oscillator_down += 1

    if oscillator_up > oscillator_down:
        up_score += min(2, oscillator_up)

    elif oscillator_down > oscillator_up:
        down_score += min(2, oscillator_down)

    # --------------------------------------------------------
    # MOVING AVERAGES / 3
    # --------------------------------------------------------

    if ema9 > ema21 > ema50:
        up_score += 3

    elif ema9 < ema21 < ema50:
        down_score += 3

    elif ema9 > ema21:
        up_score += 2

    elif ema9 < ema21:
        down_score += 2

    # --------------------------------------------------------
    # ADX / DI / 2
    # --------------------------------------------------------

    plus_di = adx_value["plus_di"]
    minus_di = adx_value["minus_di"]

    if plus_di > minus_di:
        up_score += 2

    elif minus_di > plus_di:
        down_score += 2

    # --------------------------------------------------------
    # FINAL DIRECTION
    # --------------------------------------------------------

    if up_score > down_score:
        direction = "UP"

    elif down_score > up_score:
        direction = "DOWN"

    else:
        direction = "NONE"

    winning_score = max(up_score, down_score)
    losing_score = min(up_score, down_score)

    score_gap = winning_score - losing_score

    # --------------------------------------------------------
    # PRIMARY DIRECTION CONFLICT CHECK
    # --------------------------------------------------------

    primary_conflict = False

    if direction == "UP":

        if (
            structure["direction"] == "DOWN"
            and breakout["direction"] == "DOWN"
            and liquidity["direction"] == "DOWN"
        ):
            primary_conflict = True

    elif direction == "DOWN":

        if (
            structure["direction"] == "UP"
            and breakout["direction"] == "UP"
            and liquidity["direction"] == "UP"
        ):
            primary_conflict = True

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    ranges = [
        x["high"] - x["low"]
        for x in closed[-20:]
    ]

    average_range = (
        sum(ranges) / len(ranges)
        if ranges
        else 0
    )

    volatility_ok = (
        atr_value > 0
        and average_range > 0
        and atr_value >= average_range * 0.60
    )

    # --------------------------------------------------------
    # ABNORMAL CANDLE
    # --------------------------------------------------------

    abnormal_candle = False

    current_range = current["high"] - current["low"]

    if average_range > 0:
        if current_range > average_range * 2.8:
            abnormal_candle = True

    # --------------------------------------------------------
    # EXTREME RSI
    # --------------------------------------------------------

    extreme_rsi = (
        rsi_value >= 78
        or rsi_value <= 22
    )

    # --------------------------------------------------------
    # PRICE ACTION QUALITY
    # --------------------------------------------------------

    price_action_points = (
        structure["points"]
        + breakout["points"]
        + liquidity["points"]
        + candle["points"]
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    valid = True
    reject_reason = ""

    if direction == "NONE":
        valid = False
        reject_reason = "NO CLEAR DIRECTION"

    elif winning_score < MIN_SCORE:
        valid = False
        reject_reason = "LOW SCORE"

    elif score_gap < MIN_SCORE_GAP:
        valid = False
        reject_reason = "SMALL SCORE GAP"

    elif adx_value["adx"] < MIN_ADX:
        valid = False
        reject_reason = "WEAK ADX"

    elif primary_conflict:
        valid = False
        reject_reason = "PRIMARY CONFLICT"

    elif not volatility_ok:
        valid = False
        reject_reason = "LOW VOLATILITY"

    elif abnormal_candle:
        valid = False
        reject_reason = "ABNORMAL CANDLE"

    elif extreme_rsi:
        valid = False
        reject_reason = "EXTREME RSI"

    elif breakout["fake"]:
        valid = False
        reject_reason = "FAKE BREAKOUT"

    # --------------------------------------------------------
    # IMPORTANT PRICE-ACTION FILTER
    #
    # We do NOT want:
    #
    # Structure + EMA + ADX + RSI
    #
    # to create an 85-89% signal without actual price action.
    # --------------------------------------------------------

    if valid:

        strong_breakout = (
            breakout["direction"] == direction
            and breakout["quality"] == "STRONG"
        )

        retested_breakout = (
            strong_breakout
            and breakout["retest"]
        )

        strong_liquidity = (
            liquidity["direction"] == direction
            and liquidity["points"] >= 2
        )

        strong_structure = (
            structure["direction"] == direction
            and structure["points"] >= 3
        )

        # Require meaningful price action.
        #
        # Accepted BASE patterns:
        #
        # 1) Strong breakout + structure
        # 2) Strong breakout + retest
        # 3) Structure + liquidity sweep
        # 4) Structure + breakout
        #
        # A plain trend/EMA/ADX setup is rejected.
        price_action_ok = (
            (
                strong_breakout
                and strong_structure
            )
            or (
                retested_breakout
            )
            or (
                strong_structure
                and strong_liquidity
            )
            or (
                strong_structure
                and breakout["direction"] == direction
                and breakout["points"] >= 2
            )
        )

        if not price_action_ok:
            valid = False
            reject_reason = "WEAK PRICE ACTION"

    # --------------------------------------------------------
    # STRUCTURE / BREAKOUT MISMATCH
    # --------------------------------------------------------

    if valid:

        if (
            structure["direction"] != "NONE"
            and breakout["direction"] != "NONE"
            and structure["direction"] != breakout["direction"]
        ):
            valid = False
            reject_reason = "STRUCTURE/BREAKOUT MISMATCH"

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = 70

    if winning_score >= 15:
        confidence += 5

    if winning_score >= 16:
        confidence += 2

    if winning_score >= 17:
        confidence += 1

    if score_gap >= 5:
        confidence += 2

    if score_gap >= 7:
        confidence += 1

    if adx_value["adx"] >= STRONG_ADX:
        confidence += 2

    # Structure confirmation.
    if structure["direction"] == direction:
        confidence += 2

    # Breakout confirmation.
    if breakout["direction"] == direction:

        if breakout["quality"] == "STRONG":
            confidence += 3

        elif breakout["quality"] == "WEAK":
            confidence += 1

    # Retest is a major confirmation.
    if (
        breakout["direction"] == direction
        and breakout["retest"]
    ):
        confidence += 3

    # Liquidity confirmation.
    if liquidity["direction"] == direction:
        confidence += 2

    # Candle confirmation.
    if candle["direction"] == direction:
        confidence += 1

    # --------------------------------------------------------
    # CRITICAL CONFIDENCE CAPS
    #
    # These caps are the main correction for the AUDNZD loss.
    # --------------------------------------------------------

    if breakout["direction"] != direction:
        confidence = min(confidence, 82)

    if breakout["direction"] == "NONE":
        confidence = min(confidence, 81)

    elif (
        breakout["direction"] == direction
        and breakout["quality"] == "WEAK"
        and not breakout["retest"]
    ):
        confidence = min(confidence, 84)

    elif (
        breakout["direction"] == direction
        and breakout["quality"] == "STRONG"
        and not breakout["retest"]
    ):
        confidence = min(confidence, 86)

    elif (
        breakout["direction"] == direction
        and breakout["quality"] == "STRONG"
        and breakout["retest"]
    ):
        confidence = min(confidence, 89)

    # No 89% without actual price-action confirmation.
    if not (
        breakout["direction"] == direction
        and breakout["quality"] == "STRONG"
        and breakout["retest"]
    ):
        confidence = min(confidence, 86)

    # Liquidity can strengthen, but cannot manufacture confidence.
    if liquidity["direction"] == direction:
        confidence += 1

    confidence = max(76, min(confidence, MAX_BASE_CONFIDENCE))

    # --------------------------------------------------------
    # ADDITIONAL BASE QUALITY
    # --------------------------------------------------------

    strong_pa = (
        structure["direction"] == direction
        and structure["points"] >= 3
        and (
            (
                breakout["direction"] == direction
                and breakout["quality"] == "STRONG"
            )
            or
            liquidity["direction"] == direction
        )
    )

    # --------------------------------------------------------
    # CANCELLATION
    # --------------------------------------------------------

    cancel = cancellation_level(
        closed,
        direction,
        atr_value,
    )

    if cancel is None:
        valid = False
        reject_reason = "NO STRUCTURAL CANCELLATION"

    return {
        "valid": valid,
        "reject_reason": reject_reason,

        "direction": direction,

        "up_score": up_score,
        "down_score": down_score,

        "winning_score": winning_score,
        "losing_score": losing_score,
        "score_gap": score_gap,

        "confidence": confidence,

        "price": price,
        "atr": atr_value,

        "ema9": ema9,
        "ema21": ema21,
        "ema50": ema50,

        "rsi": rsi_value,
        "williams": williams,
        "stoch": stoch,
        "momentum": mom,
        "cci": cci_value,
        "macd": macd_value,

        "adx": adx_value["adx"],
        "plus_di": plus_di,
        "minus_di": minus_di,

        "structure": structure,
        "breakout": breakout,
        "liquidity": liquidity,
        "candle": candle,

        "price_action_points": price_action_points,
        "strong_price_action": strong_pa,

        "cancellation": cancel,
    }


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def is_fresh_data(symbol):
    with state_lock:
        item = latest_market.get(symbol)

    if not item:
        return False

    received = item.get("received_at", 0)

    return (
        time.time() - received
        <= DATA_FRESH_SECONDS
    )


def same_setup_allowed(symbol, direction, analysis):
    now = time.time()

    with state_lock:
        previous = last_base_by_symbol.get(symbol)

    if not previous:
        return True

    if previous["direction"] != direction:
        return True

    elapsed = now - previous["timestamp"]

    if elapsed >= BASE_COOLDOWN_SECONDS:
        return True

    previous_price = previous.get("price")
    previous_atr = previous.get("atr")
    previous_rsi = previous.get("rsi")

    current_price = analysis["price"]
    current_atr = analysis["atr"]
    current_rsi = analysis["rsi"]

    if (
        previous_price is not None
        and previous_atr is not None
        and previous_atr > 0
    ):
        if (
            abs(current_price - previous_price)
            >= previous_atr * MIN_PRICE_MOVE_ATR
        ):
            return True

    if (
        previous_rsi is not None
        and current_rsi is not None
    ):
        if (
            abs(current_rsi - previous_rsi)
            >= MIN_RSI_CHANGE
        ):
            return True

    return False


# ============================================================
# RECOVERY VALIDATION
# ============================================================

def recovery_allowed(symbol, analysis):
    if not analysis:
        return False

    direction = analysis["direction"]

    if direction == "NONE":
        return False

    if analysis["winning_score"] < 16:
        return False

    if analysis["score_gap"] < 4:
        return False

    if analysis["adx"] < STRONG_ADX:
        return False

    if analysis["structure"]["direction"] != direction:
        return False

    if analysis["breakout"]["direction"] != direction:
        return False

    if analysis["breakout"]["quality"] != "STRONG":
        return False

    if not analysis["breakout"]["retest"]:
        return False

    if analysis["breakout"]["fake"]:
        return False

    if analysis["rsi"] >= 75 or analysis["rsi"] <= 25:
        return False

    if analysis["cancellation"] is None:
        return False

    return True


# ============================================================
# SIGNAL CREATION
# ============================================================

def create_signal(symbol, timeframe, analysis, signal_type="BASE"):
    direction = analysis["direction"]

    delay = entry_delay_minutes(timeframe)

    created = now_algiers()

    entry_time = created + timedelta(
        minutes=delay
    )

    price = analysis["price"]

    cancellation = analysis["cancellation"]

    if direction == "UP":
        cancel_text = (
            f"إذا أغلقت شمعة تحت "
            f"{format_price(cancellation)}"
        )

    else:
        cancel_text = (
            f"إذا أغلقت شمعة فوق "
            f"{format_price(cancellation)}"
        )

    confidence = analysis["confidence"]

    if signal_type == "RECOVERY":

        # Recovery must never exceed the same 89 ceiling.
        confidence = min(
            confidence + 2,
            MAX_RECOVERY_CONFIDENCE,
        )

        # Recovery still cannot claim 89 unless breakout+retest.
        if not (
            analysis["breakout"]["direction"] == direction
            and analysis["breakout"]["quality"] == "STRONG"
            and analysis["breakout"]["retest"]
        ):
            confidence = min(confidence, 86)

    return {
        "id": int(time.time() * 1000),

        "symbol": symbol,
        "timeframe": timeframe,

        "type": signal_type,
        "direction": direction,

        "confidence": confidence,

        "up_score": analysis["up_score"],
        "down_score": analysis["down_score"],

        "structure": analysis["structure"]["direction"],

        "breakout": analysis["breakout"]["direction"],

        "breakout_quality": analysis["breakout"]["quality"],

        "retest": analysis["breakout"]["retest"],

        "liquidity": analysis["liquidity"]["direction"],

        "adx": analysis["adx"],
        "rsi": analysis["rsi"],

        "price": price,

        "cancellation": cancellation,

        "cancel_text": cancel_text,

        "delay": delay,

        "created_at": created.isoformat(),

        "entry_time": entry_time.isoformat(),

        "entry_time_text": entry_time.strftime(
            "%H:%M:%S"
        ),

        "created_timestamp": time.time(),

        "analysis": analysis,
    }


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def signal_message(signal):
    symbol = signal["symbol"]
    timeframe = signal["timeframe"]

    direction = signal["direction"]

    if direction == "UP":
        direction_line = "🟢 UP"
    else:
        direction_line = "🔴 DOWN"

    if signal["type"] == "RECOVERY":
        trade_line = "♻️ RECOVERY 1/1"
    else:
        trade_line = "🎯 BASE TRADE"

    retest_text = (
        "YES"
        if signal["retest"]
        else "NO"
    )

    breakout = signal["breakout"]

    if breakout == "NONE":
        breakout_text = "NONE"
    else:
        breakout_text = breakout

    quality = signal["breakout_quality"]

    if quality == "NONE":
        quality_text = "⚪ NONE"
    elif quality == "STRONG":
        quality_text = "🟢 STRONG"
    elif quality == "WEAK":
        quality_text = "🟡 WEAK"
    else:
        quality_text = "🔴 FAKE"

    return (
        f"🎓 {BOT_NAME}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"

        f"{trade_line}\n"
        f"{direction_line}\n\n"

        f"🔥 Confidence: {signal['confidence']}%\n"
        f"🟢 UP Score: {signal['up_score']}/20\n"
        f"🔴 DOWN Score: {signal['down_score']}/20\n\n"

        f"📐 Structure: {signal['structure']}\n"
        f"🚀 Breakout: {breakout_text}\n"
        f"💥 Breakout Quality: {quality_text}\n"
        f"🔄 Retest: {retest_text}\n\n"

        f"📈 ADX: {signal['adx']:.1f}\n"
        f"📊 RSI: {signal['rsi']:.1f}\n\n"

        f"💰 Price: {format_price(signal['price'])}\n"
        f"🛑 Cancellation: {signal['cancel_text']}\n\n"

        f"⏱️ Entry after: {signal['delay']} minutes\n"
        f"🕐 ENTRY TIME: "
        f"{signal['entry_time_text']} 🇩🇿\n"

        f"━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# HISTORY
# ============================================================

def add_history(signal):
    record = {
        "id": signal["id"],
        "symbol": signal["symbol"],
        "timeframe": signal["timeframe"],
        "type": signal["type"],
        "direction": signal["direction"],
        "confidence": signal["confidence"],
        "up_score": signal["up_score"],
        "down_score": signal["down_score"],
        "price": signal["price"],
        "entry_time": signal["entry_time_text"],
        "status": "PENDING",
        "created_at": signal["created_at"],
    }

    with state_lock:
        signal_history.insert(0, record)

        if len(signal_history) > 100:
            del signal_history[100:]


# ============================================================
# OWNER CHECK
# ============================================================

def owner_only(update):
    if not update.effective_user:
        return False

    if OWNER_ID <= 0:
        return False

    return update.effective_user.id == OWNER_ID


async def deny_if_not_owner(update):
    if not owner_only(update):
        try:
            await update.message.reply_text(
                "⛔ هذا البوت خاص بالمالك فقط."
            )
        except Exception:
            pass

        return True

    return False


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_not_owner(update):
        return

    await update.message.reply_text(
        f"🎓 {BOT_NAME}\n\n"
        "✅ Bot is online.\n"
        "📡 MT4 endpoint: /mt4\n"
        "📊 Technical analysis active.\n\n"
        "Commands:\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/status"
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_not_owner(update):
        return

    with state_lock:
        w = wins
        l = losses

    total = w + l

    if total > 0:
        rate = (w / total) * 100
    else:
        rate = 0

    await update.message.reply_text(
        f"📊 {BOT_NAME} STATS\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {w}\n"
        f"🔴 Losses: {l}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {rate:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


async def win_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global wins
    global recovery_used
    global pending_recovery

    if await deny_if_not_owner(update):
        return

    with state_lock:
        wins += 1

        recovery_used = 0

        pending_recovery = {}

        if signal_history:
            signal_history[0]["status"] = "WIN"

    await update.message.reply_text(
        "✅ WIN\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {wins} Wins / {losses} Losses\n"
        "♻️ Recovery reset."
    )


async def loss_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global losses
    global recovery_used
    global recovery_skips
    global pending_recovery

    if await deny_if_not_owner(update):
        return

    recovery_signal = None

    with state_lock:
        losses += 1

        if signal_history:
            signal_history[0]["status"] = "LOSS"

        if last_signal:
            symbol = last_signal["symbol"]

            recovery_used = 1

            analysis = last_signal.get("analysis")

            if analysis:
                pending_recovery[symbol] = {
                    "allowed": True,
                    "created": time.time(),
                    "direction": last_signal["direction"],
                }

    await update.message.reply_text(
        "❌ BASE LOSS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "♻️ Recovery 1/1 ALLOWED\n"
        "⏳ Waiting for a stronger recovery setup..."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global wins
    global losses
    global recovery_used
    global recovery_skips
    global pending_recovery
    global last_signal

    if await deny_if_not_owner(update):
        return

    with state_lock:
        wins = 0
        losses = 0

        recovery_used = 0
        recovery_skips = 0

        pending_recovery = {}

        last_signal = None

        signal_history.clear()

        last_base_by_symbol.clear()

    await update.message.reply_text(
        "♻️ RESET COMPLETE\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🟢 Wins: 0\n"
        "🔴 Losses: 0\n"
        "♻️ Recovery: 1/1 available\n"
        "📚 History cleared."
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_not_owner(update):
        return

    with state_lock:
        history = list(signal_history[:10])

    if not history:
        await update.message.reply_text(
            "📚 History empty."
        )
        return

    lines = [
        f"📚 {BOT_NAME} HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in history:
        if item["direction"] == "UP":
            direction = "🟢 UP"
        else:
            direction = "🔴 DOWN"

        lines.append(
            f"📊 {item['symbol']} | {item['timeframe']}\n"
            f"   {item['type']} | {direction}\n"
            f"   🎯 {item['confidence']}% | "
            f"📈 {item['up_score']}/20 "
            f"📉 {item['down_score']}/20\n"
            f"   💰 {format_price(item['price'])}\n"
            f"   ⏰ {item['entry_time']}\n"
            f"   📌 {item['status']}\n"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def mt4status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_not_owner(update):
        return

    with state_lock:
        items = list(latest_market.items())

    if not items:
        await update.message.reply_text(
            "📡 MT4 STATUS\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🔴 No market data received."
        )
        return

    lines = [
        "📡 MT4 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    current_time = time.time()

    for symbol, data in items[:15]:
        age = current_time - data.get(
            "received_at",
            current_time,
        )

        status = (
            "🟢 LIVE"
            if age <= DATA_FRESH_SECONDS
            else "🔴 STALE"
        )

        lines.append(
            f"📊 {symbol} | "
            f"{data.get('timeframe', 'M1')}\n"
            f"   {status} | "
            f"{age:.0f}s ago\n"
            f"   Candles: {data.get('candle_count', 0)}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if await deny_if_not_owner(update):
        return

    with state_lock:
        current = last_signal

    if not current:
        await update.message.reply_text(
            "📡 No active signal."
        )
        return

    await update.message.reply_text(
        signal_message(current)
    )


# ============================================================
# SIGNAL PROCESSOR
# ============================================================

async def process_market_data(payload):
    global last_signal
    global recovery_used
    global recovery_skips

    symbol = normalize_symbol(
        payload.get("symbol")
        or payload.get("pair")
        or payload.get("asset")
    )

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or payload.get("tf")
        or "M1"
    )

    candles = extract_candles(payload)

    if not symbol or symbol == "UNKNOWN":
        return {
            "ok": False,
            "error": "missing symbol",
        }

    if len(candles) < 61:
        return {
            "ok": False,
            "error": (
                f"not enough candles: {len(candles)}"
            ),
        }

    # --------------------------------------------------------
    # Save latest market data.
    # --------------------------------------------------------

    with state_lock:
        latest_market[symbol] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "candle_count": len(candles),
            "received_at": time.time(),
        }

    analysis = analyze_market(candles)

    if not analysis:
        return {
            "ok": False,
            "error": "analysis unavailable",
        }

    logger.info(
        "%s %s | direction=%s | UP=%s DOWN=%s "
        "gap=%s ADX=%.1f PA=%s breakout=%s retest=%s "
        "valid=%s reason=%s",
        symbol,
        timeframe,
        analysis["direction"],
        analysis["up_score"],
        analysis["down_score"],
        analysis["score_gap"],
        analysis["adx"],
        analysis["price_action_points"],
        analysis["breakout"]["direction"],
        analysis["breakout"]["retest"],
        analysis["valid"],
        analysis["reject_reason"],
    )

    # --------------------------------------------------------
    # Recovery
    # --------------------------------------------------------

    with state_lock:
        recovery_pending = pending_recovery.get(symbol)

    if recovery_pending:
        age = time.time() - recovery_pending["created"]

        if age > 900:
            with state_lock:
                pending_recovery.pop(symbol, None)

            recovery_pending = None

    if recovery_pending:

        if recovery_allowed(symbol, analysis):

            signal = create_signal(
                symbol,
                timeframe,
                analysis,
                signal_type="RECOVERY",
            )

            with state_lock:
                last_signal = signal
                recovery_used = 1

                pending_recovery.pop(
                    symbol,
                    None,
                )

            add_history(signal)

            logger.info(
                "RECOVERY SIGNAL %s %s %s",
                symbol,
                timeframe,
                signal["direction"],
            )

            return {
                "ok": True,
                "signal": signal,
                "message": signal_message(signal),
            }

        else:

            # Recovery is allowed only once.
            # If the next market setup is not strong enough,
            # do not force a recovery trade.
            with state_lock:
                recovery_skips += 1
                pending_recovery.pop(
                    symbol,
                    None,
                )

            logger.info(
                "Recovery rejected for %s: %s",
                symbol,
                analysis["reject_reason"],
            )

            return {
                "ok": True,
                "signal": None,
                "message": (
                    "Recovery setup rejected: "
                    + analysis["reject_reason"]
                ),
            }

    # --------------------------------------------------------
    # BASE SIGNAL
    # --------------------------------------------------------

    if not analysis["valid"]:
        return {
            "ok": True,
            "signal": None,
            "message": (
                "No valid BASE setup: "
                + analysis["reject_reason"]
            ),
        }

    direction = analysis["direction"]

    if not is_fresh_data(symbol):
        return {
            "ok": True,
            "signal": None,
            "message": "Market data is stale.",
        }

    if not same_setup_allowed(
        symbol,
        direction,
        analysis,
    ):
        return {
            "ok": True,
            "signal": None,
            "message": "Same setup cooldown.",
        }

    signal = create_signal(
        symbol,
        timeframe,
        analysis,
        signal_type="BASE",
    )

    with state_lock:
        last_signal = signal

        last_base_by_symbol[symbol] = {
            "timestamp": time.time(),
            "direction": direction,
            "price": analysis["price"],
            "atr": analysis["atr"],
            "rsi": analysis["rsi"],
        }

        recovery_used = 0

    add_history(signal)

    logger.info(
        "BASE SIGNAL %s %s %s %s%%",
        symbol,
        timeframe,
        direction,
        signal["confidence"],
    )

    return {
        "ok": True,
        "signal": signal,
        "message": signal_message(signal),
    }


# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def _send_json(self, status_code, data):
        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def _send_text(self, status_code, text):
        body = text.encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(
            self.path
        ).path

        if path in ("/", "/health"):
            self._send_text(
                200,
                "ZinoProSignalAI is running",
            )
            return

        if path == "/status":
            with state_lock:
                data = {
                    "status": "running",
                    "bot": BOT_NAME,
                    "symbols": list(
                        latest_market.keys()
                    ),
                    "wins": wins,
                    "losses": losses,
                    "recovery_used": recovery_used,
                    "recovery_skips": recovery_skips,
                }

            self._send_json(200, data)
            return

        if path in ("/mt4", "/api/mt4"):
            self._send_json(
                200,
                {
                    "ok": True,
                    "endpoint": path,
                    "bot": BOT_NAME,
                },
            )
            return

        self._send_text(
            404,
            "Not Found",
        )

    def do_POST(self):
        path = urlparse(
            self.path
        ).path

        if path not in (
            "/mt4",
            "/api/mt4",
        ):
            self._send_json(
                404,
                {
                    "ok": False,
                    "error": "Not Found",
                },
            )
            return

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        if MT4_API_KEY:

            provided_key = (
                self.headers.get("X-API-Key")
                or self.headers.get("Authorization")
                or ""
            )

            if provided_key.startswith("Bearer "):
                provided_key = (
                    provided_key[7:].strip()
                )

            if provided_key != MT4_API_KEY:
                self._send_json(
                    401,
                    {
                        "ok": False,
                        "error": "Unauthorized",
                    },
                )
                return

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

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
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "Empty body",
                },
            )
            return

        try:
            body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                body.decode("utf-8")
            )

        except Exception as exc:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": (
                        "Invalid JSON: "
                        + str(exc)
                    ),
                },
            )
            return

        if not isinstance(payload, dict):
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "JSON object required",
                },
            )
            return

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

        try:
            result = asyncio.run(
                process_market_data(payload)
            )

        except Exception as exc:
            logger.exception(
                "MT4 processing error"
            )

            self._send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                },
            )
            return

        # ----------------------------------------------------
        # SEND TELEGRAM SIGNAL IF CREATED
        # ----------------------------------------------------

        signal = result.get("signal")

        if signal:
            try:
                schedule_telegram_signal(
                    signal
                )
            except Exception:
                logger.exception(
                    "Failed scheduling Telegram signal"
                )

        self._send_json(
            200,
            result,
        )


# ============================================================
# TELEGRAM SIGNAL SENDER
# ============================================================

telegram_application = None


def schedule_telegram_signal(signal):
    """
    Send signal to OWNER using the already-running
    Telegram application event loop.
    """

    global telegram_application

    if telegram_application is None:
        logger.warning(
            "Telegram application not ready."
        )
        return

    async def sender():
        try:
            await telegram_application.bot.send_message(
                chat_id=OWNER_ID,
                text=signal_message(signal),
            )

        except Exception:
            logger.exception(
                "Telegram send failed"
            )

    loop = telegram_application._loop

    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(
            sender(),
            loop,
        )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# BOT STARTUP
# ============================================================

async def post_init(application):
    logger.info(
        "%s Telegram application initialized.",
        BOT_NAME,
    )


def build_application():
    global telegram_application

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable is missing."
        )

    if OWNER_ID <= 0:
        raise RuntimeError(
            "OWNER_ID environment variable is missing or invalid."
        )

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "stats",
            stats_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "win",
            win_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "loss",
            loss_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "history",
            history_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "mt4status",
            mt4status_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "status",
            status_command,
        )
    )

    telegram_application = application

    return application


def main():
    logger.info(
        "Starting %s...",
        BOT_NAME,
    )

    logger.info(
        "OWNER_ID=%s",
        OWNER_ID,
    )

    logger.info(
        "MT4 API key configured=%s",
        bool(MT4_API_KEY),
    )

    # --------------------------------------------------------
    # HTTP health server
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    application = build_application()

    logger.info(
        "Telegram polling starting..."
    )

    application.run_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
 
