import os
import json
import math
import time
import asyncio
import logging
import threading
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

OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()

try:
    OWNER_ID = int(OWNER_ID_RAW)
except Exception:
    OWNER_ID = 0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

# Backward compatibility
if not MT4_API_KEY:
    MT4_API_KEY = os.getenv("ZINO_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")

# ============================================================
# TRADING CONFIG
# ============================================================

TIMEFRAME = "M1"

MIN_CANDLES = 50

ENTRY_DELAY_SECONDS = 60

# Minimum deterministic score
MIN_SCORE = 16

MAX_SCORE = 20

# Signal cooldown after a completed cycle
NEW_SIGNAL_COOLDOWN = 30

# Data freshness
MAX_DATA_AGE_SECONDS = 180

# Only one recovery
RECOVERY_LIMIT = 1

# Don't repeatedly use exactly the same setup
SETUP_REPEAT_BLOCK_SECONDS = 240

# Maximum confidence shown
MAX_CONFIDENCE = 89

# Minimum confidence
MIN_CONFIDENCE = 76


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

# ------------------------------------------------------------
# ONE TRADE CYCLE
# ------------------------------------------------------------

cycle = {
    "active": False,

    # BASE / RECOVERY
    "trade_type": None,

    # symbol of current trade
    "symbol": None,

    # direction
    "direction": None,

    # current pending trade
    "pending": False,

    # recovery already used?
    "recovery_used": False,

    # generation in progress
    "generating": False,

    # unique setup key
    "setup_key": None,

    # last signal time
    "last_signal_time": 0,

    # base information
    "base_symbol": None,
    "base_direction": None,

    # last signal data
    "last_signal": None,
}


# ============================================================
# TELEGRAM LOOP
# ============================================================

telegram_loop = None

application = None


# ============================================================
# HELPERS
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

        if o <= 0 or h <= 0 or l <= 0 or cl <= 0:
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

    # Remove duplicate timestamps
    clean = []

    seen = set()

    for c in result:
        if c["time"] in seen:
            continue

        seen.add(c["time"])
        clean.append(c)

    return clean


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1.0)

    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (
            (value - result) * multiplier
            + result
        )

    return result


def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)

    first = sum(values[:period]) / period

    result = [first]

    prev = first

    for value in values[period:]:
        current = (
            (value - prev) * multiplier
            + prev
        )

        result.append(current)

        prev = current

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1) + gains[i])
            / period
        )

        avg_loss = (
            (avg_loss * (period - 1) + losses[i])
            / period
        )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


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


def atr(candles, period=14):
    if len(candles) < period + 1:
        return None

    trs = true_ranges(candles)

    value = sum(trs[:period]) / period

    for tr in trs[period:]:
        value = (
            (value * (period - 1) + tr)
            / period
        )

    return value


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

        if up_move > down_move and up_move > 0:
            plus = up_move
        else:
            plus = 0.0

        if down_move > up_move and down_move > 0:
            minus = down_move
        else:
            minus = 0.0

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - prev["close"]),
            abs(current["low"] - prev["close"])
        )

        tr_values.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(tr_values) < period:
        return None

    atr_val = sum(tr_values[:period]) / period
    plus_val = sum(plus_dm[:period]) / period
    minus_val = sum(minus_dm[:period]) / period

    dx_values = []

    for i in range(period, len(tr_values)):

        atr_val = (
            (atr_val * (period - 1) + tr_values[i])
            / period
        )

        plus_val = (
            (plus_val * (period - 1) + plus_dm[i])
            / period
        )

        minus_val = (
            (minus_val * (period - 1) + minus_dm[i])
            / period
        )

        if atr_val <= 0:
            continue

        plus_di = 100.0 * plus_val / atr_val
        minus_di = 100.0 * minus_val / atr_val

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

    adx_val = sum(dx_values[:period]) / period

    for value in dx_values[period:]:
        adx_val = (
            (adx_val * (period - 1) + value)
            / period
        )

    if atr_val <= 0:
        return None

    plus_di = 100.0 * plus_val / atr_val
    minus_di = 100.0 * minus_val / atr_val

    return {
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# PRICE ACTION
# ============================================================

def candle_features(c):
    body = abs(c["close"] - c["open"])

    total = c["high"] - c["low"]

    if total <= 0:
        return {
            "body": 0,
            "range": 0,
            "body_ratio": 0,
            "upper_wick": 0,
            "lower_wick": 0,
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
        return 0

    return sum(
        c["high"] - c["low"]
        for c in subset
    ) / len(subset)


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

    previous_high = max(
        c["high"] for c in previous
    )

    previous_low = min(
        c["low"] for c in previous
    )

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

    # Sweep above high then close below
    if last["high"] > high and last["close"] < high:
        if features["upper_wick"] > features["body"]:
            return "DOWN"

    # Sweep below low then close above
    if last["low"] < low and last["close"] > low:
        if features["lower_wick"] > features["body"]:
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

    # Avoid extreme M1 spikes
    if last_range > avg_range * 2.8:
        return True

    return False


def candle_direction(c):
    if c["close"] > c["open"]:
        return "UP"

    if c["close"] < c["open"]:
        return "DOWN"

    return "NEUTRAL"


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def calculate_analysis(candles):

    if len(candles) < MIN_CANDLES:
        return None

    closes = [c["close"] for c in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi14 = rsi(closes, 14)

    adx_data = adx_di(candles, 14)

    atr14 = atr(candles, 14)

    if (
        ema9 is None
        or ema21 is None
        or rsi14 is None
        or adx_data is None
        or atr14 is None
    ):
        return None

    last = candles[-1]

    structure = market_structure(candles)

    breakout = breakout_status(candles)

    liquidity = liquidity_status(candles)

    candle_dir = candle_direction(last)

    candle = candle_features(last)

    avg_range = recent_average_range(candles, 14)

    if avg_range <= 0:
        return None

    # --------------------------------------------------------
    # Determine candidate directions
    # --------------------------------------------------------

    up_score = 0
    down_score = 0

    # --------------------------------------------------------
    # EMA 9/21 = 4 points
    # --------------------------------------------------------

    if ema9 > ema21:
        up_score += 3

    elif ema9 < ema21:
        down_score += 3

    # Price alignment
    if last["close"] > ema9 and last["close"] > ema21:
        up_score += 1

    elif last["close"] < ema9 and last["close"] < ema21:
        down_score += 1

    # --------------------------------------------------------
    # RSI = 3 points
    # --------------------------------------------------------

    if 53 <= rsi14 <= 68:
        up_score += 3

    elif 32 <= rsi14 <= 47:
        down_score += 3

    elif rsi14 > 50:
        up_score += 1

    elif rsi14 < 50:
        down_score += 1

    # --------------------------------------------------------
    # ADX + DI = 4 points
    # --------------------------------------------------------

    adx_value = adx_data["adx"]
    plus_di = adx_data["plus_di"]
    minus_di = adx_data["minus_di"]

    if adx_value >= 25:

        if plus_di > minus_di:
            up_score += 4

        elif minus_di > plus_di:
            down_score += 4

    elif adx_value >= 20:

        if plus_di > minus_di:
            up_score += 2

        elif minus_di > plus_di:
            down_score += 2

    # --------------------------------------------------------
    # STRUCTURE = 3 points
    # --------------------------------------------------------

    if structure == "UP":
        up_score += 3

    elif structure == "DOWN":
        down_score += 3

    # --------------------------------------------------------
    # BREAKOUT = 2 points
    # --------------------------------------------------------

    if breakout == "UP":
        up_score += 2

    elif breakout == "DOWN":
        down_score += 2

    # --------------------------------------------------------
    # CANDLE = 2 points
    # --------------------------------------------------------

    if candle["body_ratio"] >= 0.55:

        if candle_dir == "UP":
            up_score += 2

        elif candle_dir == "DOWN":
            down_score += 2

    elif candle["body_ratio"] >= 0.35:

        if candle_dir == "UP":
            up_score += 1

        elif candle_dir == "DOWN":
            down_score += 1

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    # Liquidity sweep is a directional confirmation.
    if liquidity == "UP":
        up_score += 2

    elif liquidity == "DOWN":
        down_score += 2

    # --------------------------------------------------------
    # ABNORMAL MOVE
    # --------------------------------------------------------

    abnormal = abnormal_candle(candles)

    # --------------------------------------------------------
    # Volatility quality
    # --------------------------------------------------------

    volatility_ok = True

    if atr14 <= 0:
        volatility_ok = False

    # If candle is extremely tiny, avoid dead market.
    if (
        candle["range"] < avg_range * 0.25
    ):
        volatility_ok = False

    # If candle is huge, avoid chasing.
    if candle["range"] > avg_range * 2.8:
        volatility_ok = False

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if up_score > down_score:
        direction = "UP"
        raw_score = up_score

    elif down_score > up_score:
        direction = "DOWN"
        raw_score = down_score

    else:
        direction = None
        raw_score = 0

    # --------------------------------------------------------
    # Conflict filters
    # --------------------------------------------------------

    conflict = False

    if direction == "UP":

        if ema9 <= ema21:
            conflict = True

        if plus_di <= minus_di and adx_value >= 25:
            conflict = True

        if rsi14 < 50:
            conflict = True

        if structure == "DOWN":
            conflict = True

        if liquidity == "DOWN":
            conflict = True

    elif direction == "DOWN":

        if ema9 >= ema21:
            conflict = True

        if minus_di <= plus_di and adx_value >= 25:
            conflict = True

        if rsi14 > 50:
            conflict = True

        if structure == "UP":
            conflict = True

        if liquidity == "UP":
            conflict = True

    if abnormal:
        conflict = True

    if not volatility_ok:
        conflict = True

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = 0

    if direction:

        gap = abs(up_score - down_score)

        confidence = (
            72
            + raw_score * 0.75
            + gap * 1.0
        )

        # ADX strength bonus
        if adx_value >= 30:
            confidence += 3

        elif adx_value >= 25:
            confidence += 2

        # Strong body bonus
        if candle["body_ratio"] >= 0.65:
            confidence += 2

        confidence = int(
            clamp(
                confidence,
                MIN_CONFIDENCE,
                MAX_CONFIDENCE
            )
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

        "structure": structure,
        "breakout": breakout,
        "liquidity": liquidity,

        "candle_direction": candle_dir,
        "body_ratio": candle["body_ratio"],
        "candle_range": candle["range"],
        "average_range": avg_range,

        "abnormal": abnormal,
        "volatility_ok": volatility_ok,
        "conflict": conflict,

        "price": last["close"],
        "candle_time": last["time"],
    }


# ============================================================
# CANCELLATION PRICE
# ============================================================

def cancellation_price(candles, direction):

    recent = candles[-8:]

    if direction == "UP":
        return min(
            c["low"] for c in recent
        )

    return max(
        c["high"] for c in recent
    )


# ============================================================
# SETUP KEY
# ============================================================

def make_setup_key(symbol, analysis):

    return (
        f"{symbol}|"
        f"{analysis['direction']}|"
        f"{analysis['candle_time']}"
    )


# ============================================================
# GEMINI REFINEMENT
# ============================================================

def gemini_refine(symbol, analysis, candles):

    if not GEMINI_AVAILABLE:
        return analysis

    if not GEMINI_API_KEY:
        return analysis

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        recent = candles[-20:]

        compact = []

        for c in recent:
            compact.append({
                "t": c["time"],
                "o": round(c["open"], 6),
                "h": round(c["high"], 6),
                "l": round(c["low"], 6),
                "c": round(c["close"], 6),
            })

        prompt = f"""
You are a strict M1 forex signal validator.

Symbol: {symbol}

Deterministic technical analysis:

Direction candidate:
{analysis['direction']}

Score:
{analysis['score']}/20

UP score:
{analysis['up_score']}/20

DOWN score:
{analysis['down_score']}/20

EMA9:
{analysis['ema9']}

EMA21:
{analysis['ema21']}

RSI14:
{analysis['rsi']}

ADX14:
{analysis['adx']}

DI+:
{analysis['plus_di']}

DI-:
{analysis['minus_di']}

Structure:
{analysis['structure']}

Breakout:
{analysis['breakout']}

Liquidity:
{analysis['liquidity']}

Candle:
{analysis['candle_direction']}

Body ratio:
{analysis['body_ratio']}

Candles:
{json.dumps(compact)}

Rules:

1. Do NOT invent information.
2. Do NOT force a trade.
3. If the deterministic direction is technically weak or contradictory,
   reject it.
4. Reject if the latest candle is abnormal.
5. Reject if EMA direction, RSI and ADX/DI strongly disagree.
6. Prefer continuation after a confirmed structure/breakout.
7. Avoid chasing an extreme candle.
8. We only want ONE strongest trade.
9. Never return WAIT as a trade direction.
10. If rejected, set valid=false.

Return JSON only:

{{
  "valid": true,
  "direction": "UP",
  "confidence": 80,
  "reason": "short reason",
  "cancellation_reason": "short reason"
}}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.05,
                response_mime_type="application/json",
            )
        )

        text = response.text.strip()

        data = json.loads(text)

        valid = bool(data.get("valid", False))

        if not valid:
            analysis["gemini_valid"] = False
            analysis["gemini_reason"] = (
                data.get("reason")
                or "Gemini rejected weak setup."
            )

            return analysis

        direction = normalize_direction(
            data.get("direction")
        )

        if direction != analysis["direction"]:
            analysis["gemini_valid"] = False
            analysis["gemini_reason"] = (
                "Gemini direction disagreed with deterministic analysis."
            )

            return analysis

        gemini_conf = int(
            safe_float(
                data.get("confidence"),
                analysis["confidence"]
            )
        )

        analysis["confidence"] = int(
            clamp(
                min(
                    gemini_conf,
                    analysis["confidence"] + 4
                ),
                MIN_CONFIDENCE,
                MAX_CONFIDENCE
            )
        )

        analysis["gemini_valid"] = True

        analysis["reason"] = (
            str(
                data.get("reason")
                or "Technical conditions aligned."
            )[:300]
        )

        analysis["cancellation_reason"] = (
            str(
                data.get("cancellation_reason")
                or "Price crossed the invalidation level."
            )[:250]
        )

        return analysis

    except Exception as e:

        logger.warning(
            "Gemini refinement failed: %s",
            e
        )

        # Technical engine remains authoritative.
        return analysis


# ============================================================
# SIGNAL VALIDATION
# ============================================================

def validate_signal(symbol, candles):

    analysis = calculate_analysis(candles)

    if not analysis:
        return None

    if not analysis["direction"]:
        return None

    if analysis["conflict"]:
        return None

    if analysis["score"] < MIN_SCORE:
        return None

    # Strong disagreement is rejected
    score_gap = abs(
        analysis["up_score"]
        - analysis["down_score"]
    )

    if score_gap < 3:
        return None

    # If ADX is weak, don't force signal
    if analysis["adx"] < 20:
        return None

    # Direction must agree with RSI
    if analysis["direction"] == "UP":
        if analysis["rsi"] < 50:
            return None

    if analysis["direction"] == "DOWN":
        if analysis["rsi"] > 50:
            return None

    # Direction must agree with DI
    if analysis["direction"] == "UP":
        if analysis["plus_di"] <= analysis["minus_di"]:
            return None

    if analysis["direction"] == "DOWN":
        if analysis["minus_di"] <= analysis["plus_di"]:
            return None

    analysis = gemini_refine(
        symbol,
        analysis,
        candles
    )

    # Gemini can reject
    if analysis.get("gemini_valid") is False:
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

        received = info.get(
            "received_at",
            0
        )

        if not candles:
            continue

        if current_time - received > MAX_DATA_AGE_SECONDS:
            continue

        if len(candles) < MIN_CANDLES:
            continue

        analysis = calculate_analysis(candles)

        if not analysis:
            continue

        if not analysis["direction"]:
            continue

        if analysis["conflict"]:
            continue

        if analysis["score"] < MIN_SCORE:
            continue

        if analysis["adx"] < 20:
            continue

        # Small bonus for clean alignment
        quality = float(
            analysis["score"]
        )

        quality += abs(
            analysis["up_score"]
            - analysis["down_score"]
        ) * 0.5

        if analysis["adx"] >= 30:
            quality += 1.5

        if analysis["structure"] == analysis["direction"]:
            quality += 1

        if analysis["breakout"] == analysis["direction"]:
            quality += 1

        candidates.append({
            "symbol": symbol,
            "candles": candles,
            "analysis": analysis,
            "quality": quality,
        })

    if not candidates:
        return None

    # Highest quality first
    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_message(text):

    global application

    if application is None:
        logger.error(
            "Telegram application is not ready."
        )
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

        # Retry plain text
        try:

            await application.bot.send_message(
                chat_id=OWNER_ID,
                text=text
            )

            return True

        except Exception as e2:

            logger.error(
                "Telegram retry failed: %s",
                e2
            )

            return False


def send_message_threadsafe(text):

    global telegram_loop

    if telegram_loop is None:
        logger.error(
            "Telegram loop unavailable."
        )
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

    emoji = "🟢 UP" if direction == "UP" else "🔴 DOWN"

    confidence = analysis["confidence"]

    score = analysis["score"]

    up_score = analysis["up_score"]

    down_score = analysis["down_score"]

    price = analysis["price"]

    cancellation = analysis["cancellation"]

    entry_time = now_algiers() + timedelta(
        seconds=ENTRY_DELAY_SECONDS
    )

    reason = analysis.get(
        "reason",
        "EMA + RSI + ADX/DI + Price Action aligned."
    )

    if trade_type == "BASE":
        title = "🎯 BASE TRADE"
    else:
        title = "♻️ RECOVERY 1/1"

    text = (
        "🎓 *ZinoProSignalAI*\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 *{symbol} | M1*\n\n"
        f"{title}\n"
        f"➡️ *{emoji}*\n\n"
        f"🔥 Confidence: *{confidence}%*\n"
        f"📈 UP Score: *{up_score}/20*\n"
        f"📉 DOWN Score: *{down_score}/20*\n\n"
        f"⏱️ Entry after: *1 minute*\n"
        f"🕐 *ENTRY TIME: {fmt_time(entry_time)} 🇩🇿*\n\n"
        f"💰 Price: *{price:.6f}*\n"
        f"❌ Cancellation: *{cancellation:.6f}*\n\n"
        f"🧠 *Reason:*\n"
        f"{reason}\n\n"
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

        # ----------------------------------------------------
        # HARD ONE-TRADE LOCK
        # ----------------------------------------------------

        if cycle["pending"]:
            logger.info(
                "Signal blocked: another trade is pending."
            )
            return False

        # ----------------------------------------------------
        # BASE
        # ----------------------------------------------------

        if trade_type == "BASE":

            if cycle["active"]:
                logger.info(
                    "BASE blocked: cycle already active."
                )
                return False

        # ----------------------------------------------------
        # RECOVERY
        # ----------------------------------------------------

        if trade_type == "RECOVERY":

            if not cycle["active"]:
                logger.info(
                    "Recovery blocked: no active cycle."
                )
                return False

            if cycle["recovery_used"]:
                logger.info(
                    "Recovery blocked: already used."
                )
                return False

            if cycle["trade_type"] != "RECOVERY":
                logger.info(
                    "Recovery blocked: wrong cycle state."
                )
                return False

        # ----------------------------------------------------
        # SETUP KEY
        # ----------------------------------------------------

        setup_key = make_setup_key(
            symbol,
            analysis
        )

        # ----------------------------------------------------
        # RESERVE THE TRADE BEFORE TELEGRAM
        # ----------------------------------------------------

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

        elif trade_type == "RECOVERY":

            cycle["recovery_used"] = True

    # --------------------------------------------------------
    # Build and send outside lock
    # --------------------------------------------------------

    text = build_signal_text(
        symbol,
        analysis,
        trade_type
    )

    success = send_message_threadsafe(text)

    if not success:

        # Telegram failed, release pending state
        with cycle_lock:

            cycle["pending"] = False

            if trade_type == "BASE":
                cycle["active"] = False
                cycle["trade_type"] = None
                cycle["symbol"] = None
                cycle["direction"] = None
                cycle["setup_key"] = None

            elif trade_type == "RECOVERY":
                cycle["recovery_used"] = False

        return False

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

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
        "SIGNAL SENT | %s | %s | %s | %s/20",
        symbol,
        trade_type,
        analysis["direction"],
        analysis["score"]
    )

    return True


# ============================================================
# BASE ANALYSIS
# ============================================================

def analyze_and_send_base():

    # --------------------------------------------------------
    # HARD CYCLE CHECK
    # --------------------------------------------------------

    with cycle_lock:

        if cycle["pending"]:
            return False

        if cycle["active"]:
            return False

        if cycle["generating"]:
            return False

        # Reserve generation
        cycle["generating"] = True

    try:

        candidate = choose_best_pair()

        if not candidate:
            logger.info(
                "No high-quality setup found."
            )
            return False

        symbol = candidate["symbol"]

        candles = candidate["candles"]

        # Full validation
        analysis = validate_signal(
            symbol,
            candles
        )

        if not analysis:
            logger.info(
                "Best candidate failed final validation."
            )
            return False

        # ----------------------------------------------------
        # Send exactly ONE
        # ----------------------------------------------------

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

        symbol = cycle["symbol"]

    # --------------------------------------------------------
    # Recovery uses freshest data of current pair
    # --------------------------------------------------------

    with data_lock:

        info = data_store.get(symbol)

        if not info:
            return False

        candles = list(
            info.get("candles", [])
        )

    if len(candles) < MIN_CANDLES:
        return False

    analysis = validate_signal(
        symbol,
        candles
    )

    if not analysis:
        logger.info(
            "Recovery setup not strong enough."
        )
        return False

    return send_one_signal(
        symbol,
        candles,
        analysis,
        "RECOVERY"
    )


# ============================================================
# MT4 DATA PROCESSING
# ============================================================

def process_mt4_payload(payload):

    if not isinstance(payload, dict):
        return {
            "ok": False,
            "error": "Invalid JSON object"
        }

    symbol = str(
        payload.get("symbol")
        or ""
    ).strip().upper()

    timeframe = str(
        payload.get("timeframe")
        or payload.get("tf")
        or "M1"
    ).upper()

    candles = normalize_candles(
        payload.get("candles")
        or []
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
        "MT4 DATA | %s | candles=%d",
        symbol,
        len(candles)
    )

    # --------------------------------------------------------
    # AUTOMATIC SIGNAL ENGINE
    # --------------------------------------------------------

    # Do NOT create multiple analysis threads.
    # Only one can enter the analysis lock.

    def background_signal_check():

        if not analysis_lock.acquire(
            blocking=False
        ):
            return

        try:

            with cycle_lock:

                # ABSOLUTE ONE-TRADE RULE
                if cycle["pending"]:
                    return

                # Active BASE/RECOVERY cycle
                if cycle["active"]:
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

        if path in ("/", "/health", "/healthz"):

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

        if path in ("/mt4status", "/status"):

            with data_lock:

                symbols = {}

                for symbol, info in data_store.items():

                    symbols[symbol] = {
                        "candles": len(
                            info.get("candles", [])
                        ),
                        "age_seconds": round(
                            now_timestamp()
                            - info.get(
                                "received_at",
                                now_timestamp()
                            ),
                            1
                        )
                    }

            with cycle_lock:

                cycle_copy = {
                    "active": cycle["active"],
                    "trade_type": cycle["trade_type"],
                    "symbol": cycle["symbol"],
                    "direction": cycle["direction"],
                    "pending": cycle["pending"],
                    "recovery_used": cycle["recovery_used"],
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

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Backward compatible API key
        # ----------------------------------------------------

        json_key = str(
            payload.get("api_key")
            or ""
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

        # ----------------------------------------------------
        # PROCESS
        # ----------------------------------------------------

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
        "🎯 One strongest trade only.\n"
        "♻️ One recovery maximum.\n\n"
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
            else (
                f"{cycle['trade_type']} "
                f"{cycle['symbol']} "
                f"{cycle['direction']}"
            )
        )

        pending = cycle["pending"]

        recovery_used = cycle["recovery_used"]

    total_wins = (
        stats["base_win"]
        + stats["recovery_win"]
    )

    total_losses = (
        stats["base_loss"]
        + stats["recovery_loss"]
    )

    total = total_wins + total_losses

    if total > 0:
        winrate = (
            total_wins
            / total
            * 100
        )
    else:
        winrate = 0

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
        f"♻️ RECOVERY USED: "
        f"{'YES' if recovery_used else 'NO'}"
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

        direction = item["direction"]

        icon = (
            "🟢"
            if direction == "UP"
            else "🔴"
        )

        lines.append(
            f"{icon} {item['symbol']} "
            f"{item['trade_type']}\n"
            f"   {direction} | "
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

    global cycle

    with cycle_lock:

        if not cycle["pending"]:

            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة حالياً."
            )

            return

        trade_type = cycle["trade_type"]

        symbol = cycle["symbol"]

        direction = cycle["direction"]

        # ----------------------------------------------------
        # Record stats
        # ----------------------------------------------------

        if trade_type == "BASE":

            stats["base_win"] += 1

        elif trade_type == "RECOVERY":

            stats["recovery_win"] += 1

        # ----------------------------------------------------
        # Update history
        # ----------------------------------------------------

        with data_lock:

            for item in reversed(history):

                if (
                    item["result"] == "PENDING"
                    and item["symbol"] == symbol
                    and item["trade_type"] == trade_type
                ):

                    item["result"] = "WIN"

                    break

        # ----------------------------------------------------
        # CLOSE CYCLE
        # ----------------------------------------------------

        cycle["active"] = False

        cycle["trade_type"] = None

        cycle["symbol"] = None

        cycle["direction"] = None

        cycle["pending"] = False

        cycle["recovery_used"] = False

        cycle["setup_key"] = None

        cycle["last_signal"] = None

    await update.message.reply_text(
        f"🟢 *{trade_type} WIN*\n\n"
        f"📊 {symbol} | {direction}\n\n"
        "✅ الدورة انتهت.\n"
        "🔎 البوت الآن يبحث عن أقوى فرصة جديدة.",
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

    global cycle

    with cycle_lock:

        if not cycle["pending"]:

            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة حالياً."
            )

            return

        trade_type = cycle["trade_type"]

        symbol = cycle["symbol"]

        direction = cycle["direction"]

        # ----------------------------------------------------
        # BASE LOSS
        # ----------------------------------------------------

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

            # ------------------------------------------------
            # Convert cycle to recovery
            # ------------------------------------------------

            cycle["trade_type"] = "RECOVERY"

            cycle["pending"] = False

            cycle["recovery_used"] = False

            cycle["last_signal"] = None

            # Same pair, fresh M1 data.
            # Recovery must be independently validated.
            recovery_symbol = symbol

        # ----------------------------------------------------
        # RECOVERY LOSS
        # ----------------------------------------------------

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

            # END ENTIRE CYCLE

            cycle["active"] = False

            cycle["trade_type"] = None

            cycle["symbol"] = None

            cycle["direction"] = None

            cycle["pending"] = False

            cycle["recovery_used"] = False

            cycle["setup_key"] = None

            cycle["last_signal"] = None

            recovery_symbol = None

    # --------------------------------------------------------
    # BASE LOSS -> ONE RECOVERY
    # --------------------------------------------------------

    if trade_type == "BASE":

        await update.message.reply_text(
            "🔴 *BASE LOSS*\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {direction}\n\n"
            "♻️ Recovery 1/1 مسموحة.\n"
            "🔍 سيتم فحص الحركة الجديدة بدقة.\n"
            "⚠️ لن يتم إرسال Recovery إذا كانت الشروط ضعيفة.",
            parse_mode="Markdown"
        )

        # Give MT4 data a moment to refresh.
        await asyncio.sleep(2)

        # Run recovery outside Telegram handler state lock.
        def recovery_worker():

            if not analysis_lock.acquire(
                blocking=False
            ):
                logger.info(
                    "Recovery analysis already running."
                )
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
            "🔎 الدورة انتهت، والبحث القادم سيكون عن صفقة BASE جديدة.",
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

    global cycle

    with cycle_lock:

        cycle["active"] = False

        cycle["trade_type"] = None

        cycle["symbol"] = None

        cycle["direction"] = None

        cycle["pending"] = False

        cycle["recovery_used"] = False

        cycle["generating"] = False

        cycle["setup_key"] = None

        cycle["last_signal"] = None

    with data_lock:

        history.clear()

        stats["base_win"] = 0
        stats["base_loss"] = 0
        stats["recovery_win"] = 0
        stats["recovery_loss"] = 0

    await update.message.reply_text(
        "♻️ تم Reset بالكامل.\n"
        "🎯 البوت جاهز للبحث عن أقوى BASE جديدة."
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
                info.get("candles", [])
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
                else "🔴"
            )

            lines.append(
                f"{status} {symbol} | "
                f"M1 | candles={candles} | "
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

    # --------------------------------------------------------
    # HARD LOCK
    # --------------------------------------------------------

    with cycle_lock:

        if cycle["pending"]:

            await update.message.reply_text(
                "⛔ توجد صفقة معلقة.\n"
                "استعمل /win أو /loss أولاً."
            )

            return

        if cycle["active"]:

            await update.message.reply_text(
                "⛔ توجد دورة نشطة.\n"
                "لا يمكن إنشاء BASE جديدة الآن."
            )

            return

    args = context.args

    if not args:

        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURUSD M1"
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

    if len(candles) < MIN_CANDLES:

        await update.message.reply_text(
            f"⚠️ {symbol} عنده فقط "
            f"{len(candles)} شمعة.\n"
            f"المطلوب {MIN_CANDLES}."
        )

        return

    analysis = validate_signal(
        symbol,
        candles
    )

    if not analysis:

        await update.message.reply_text(
            f"❌ {symbol} لا يملك حالياً "
            f"إشارة بجودة {MIN_SCORE}/20 أو أكثر.\n\n"
            "البوت لن يجبر السوق على إعطاء صفقة."
        )

        return

    sent = send_one_signal(
        symbol,
        candles,
        analysis,
        "BASE"
    )

    if sent:

        await update.message.reply_text(
            "✅ تم إرسال BASE واحدة فقط."
        )

    else:

        await update.message.reply_text(
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
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CommandHandler(
            "stats",
            stats_command
        )
    )

    application.add_handler(
        CommandHandler(
            "history",
            history_command
        )
    )

    application.add_handler(
        CommandHandler(
            "win",
            win_command
        )
    )

    application.add_handler(
        CommandHandler(
            "loss",
            loss_command
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset_command
        )
    )

    application.add_handler(
        CommandHandler(
            "mt4status",
            mt4status_command
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command
        )
    )

    await application.initialize()

    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True
    )

    logger.info(
        "Telegram polling started."
    )

    # Keep loop alive
    while True:
        await asyncio.sleep(3600)


# ============================================================
# STARTUP
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
            "MT4 requests will be accepted without API authentication."
        )

    if not GEMINI_API_KEY:

        logger.warning(
            "GEMINI_API_KEY is not configured. "
            "Technical engine will work without Gemini."
        )

    if not GEMINI_AVAILABLE:

        logger.warning(
            "google-genai is not installed. "
            "Gemini refinement disabled."
        )


def main():

    validate_config()

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    logger.info(
        "ZinoProSignalAI starting..."
    )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    asyncio.run(
        telegram_main()
    )


if __name__ == "__main__":
    main()
