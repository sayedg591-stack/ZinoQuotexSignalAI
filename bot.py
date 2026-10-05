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
    MessageHandler,
    filters,
)

from google import genai
from google.genai import types


# ============================================================
# ZinoProSignalAI - MT5 VERSION
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = os.getenv("OWNER_ID", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()

# Compatibility with old EA/config
OLD_MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10

RECOVERY_LIMIT = 1

HISTORY_FILE = "signal_history.json"

# Prevent duplicate analysis on same candle
SIGNAL_COOLDOWN_SECONDS = 20

# Prevent repeating exactly the same setup
SETUP_REPEAT_BLOCK_SECONDS = 240


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
    except Exception as e:
        logger.exception("Gemini initialization failed: %s", e)
else:
    logger.warning("GEMINI_API_KEY is not configured")


# ============================================================
# MT5 DATA
# ============================================================

mt5_data = {}

mt5_lock = threading.Lock()

signal_send_lock = threading.Lock()
analysis_lock = threading.Lock()
cycle_lock = threading.Lock()


# ============================================================
# TELEGRAM GLOBALS
# ============================================================

telegram_application = None
telegram_loop = None


# ============================================================
# ANALYSIS STATE
# ============================================================

last_analysis_by_market = {}

recent_setups = {}

active_cycle = {
    "active": False,
    "symbol": None,
    "timeframe": None,
    "direction": None,
    "recovery_used": False,
    "last_trade_time": 0.0,
    "trade_number": 0,
}


session_stats = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}


trade_history = []
current_trade_id = None


# ============================================================
# AUTH
# ============================================================

def owner_id_int():
    try:
        return int(OWNER_ID)
    except Exception:
        return None


def is_owner(update: Update):
    oid = owner_id_int()

    if oid is None:
        return False

    user = update.effective_user

    if not user:
        return False

    return user.id == oid


# ============================================================
# HISTORY
# ============================================================

def load_history():
    global trade_history, session_stats, current_trade_id

    try:
        if not os.path.exists(HISTORY_FILE):
            return

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        trade_history = data.get("trade_history", [])

        saved_stats = data.get("session_stats", {})

        for key in session_stats:
            if key in saved_stats:
                try:
                    session_stats[key] = int(saved_stats[key])
                except Exception:
                    pass

        current_trade_id = data.get("current_trade_id")

        logger.info(
            "History loaded | trades=%s | wins=%s | losses=%s",
            len(trade_history),
            session_stats["wins"],
            session_stats["losses"],
        )

    except Exception as e:
        logger.exception("History load failed: %s", e)


def save_history():
    try:
        data = {
            "trade_history": trade_history[-500:],
            "session_stats": session_stats,
            "current_trade_id": current_trade_id,
        }

        temp_file = HISTORY_FILE + ".tmp"

        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_file, HISTORY_FILE)

    except Exception as e:
        logger.exception("History save failed: %s", e)


# ============================================================
# TIMEFRAME
# ============================================================

SUPPORTED_TIMEFRAMES = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
}


def normalize_timeframe(value):
    if value is None:
        return None

    tf = str(value).strip().upper()

    aliases = {
        "1": "M1",
        "1M": "M1",
        "M01": "M1",

        "2": "M2",
        "2M": "M2",
        "M02": "M2",

        "3": "M3",
        "3M": "M3",
        "M03": "M3",

        "5": "M5",
        "5M": "M5",

        "15": "M15",
        "15M": "M15",

        "30": "M30",
        "30M": "M30",

        "60": "H1",
        "1H": "H1",

        "240": "H4",
        "4H": "H4",
    }

    tf = aliases.get(tf, tf)

    if tf in SUPPORTED_TIMEFRAMES:
        return tf

    return None


def timeframe_minutes(tf):
    return SUPPORTED_TIMEFRAMES.get(normalize_timeframe(tf), 1)


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def candle_time_value(c):
    value = (
        c.get("time")
        if c.get("time") is not None
        else c.get("timestamp")
    )

    if value is None:
        value = c.get("datetime")

    if value is None:
        value = c.get("date")

    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return int(value)

    text = str(value).strip()

    try:
        return int(float(text))
    except Exception:
        pass

    try:
        dt = datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )

        return int(dt.timestamp())

    except Exception:
        return 0


def normalize_candle(c):
    if not isinstance(c, dict):
        return None

    def number(*keys):
        for key in keys:
            if key in c:
                try:
                    return float(c[key])
                except Exception:
                    return None
        return None

    timestamp = candle_time_value(c)

    o = number("open", "o")
    h = number("high", "h")
    l = number("low", "l")
    cl = number("close", "c")

    if None in (o, h, l, cl):
        return None

    volume = number("volume", "v", "tick_volume")

    result = {
        "time": timestamp,
        "open": o,
        "high": h,
        "low": l,
        "close": cl,
    }

    if volume is not None:
        result["volume"] = volume

    return result


def normalize_candles(candles):
    result = []

    if not isinstance(candles, list):
        return result

    for candle in candles:
        normalized = normalize_candle(candle)

        if normalized:
            result.append(normalized)

    result.sort(key=lambda x: x["time"])

    # Remove duplicate timestamps
    unique = {}

    for candle in result:
        unique[candle["time"]] = candle

    result = list(unique.values())
    result.sort(key=lambda x: x["time"])

    return result


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    seed = sum(values[:period]) / period

    multiplier = 2.0 / (period + 1.0)

    current = seed

    for value in values[period:]:
        current = (
            (value - current) * multiplier
        ) + current

    return current


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        change = values[i] - values[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (
            ((avg_gain * (period - 1)) + gain)
            / period
        )

        avg_loss = (
            ((avg_loss * (period - 1)) + loss)
            / period
        )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(x["high"] for x in recent)
    lowest = min(x["low"] for x in recent)

    if highest == lowest:
        return -50.0

    close = candles[-1]["close"]

    return (
        (highest - close)
        / (highest - lowest)
        * -100.0
    )


def true_ranges(candles):
    if not candles:
        return []

    tr = []

    previous_close = None

    for candle in candles:

        high = candle["high"]
        low = candle["low"]

        if previous_close is None:
            value = high - low
        else:
            value = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        tr.append(value)

        previous_close = candle["close"]

    return tr


def atr(candles, period=10):
    tr = true_ranges(candles)

    if len(tr) < period:
        return None

    value = sum(tr[:period]) / period

    for x in tr[period:]:
        value = (
            ((value * (period - 1)) + x)
            / period
        )

    return value


def adx_di(candles, period=14):
    if len(candles) < period * 2 + 1:
        return None, None, None

    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

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
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        tr_values.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(tr_values) < period:
        return None, None, None

    atr_smoothed = sum(tr_values[:period])
    plus_smoothed = sum(plus_dm[:period])
    minus_smoothed = sum(minus_dm[:period])

    dx_values = []

    last_plus_di = None
    last_minus_di = None

    for i in range(period, len(tr_values)):

        if i > period:
            atr_smoothed = (
                atr_smoothed
                - (atr_smoothed / period)
                + tr_values[i]
            )

            plus_smoothed = (
                plus_smoothed
                - (plus_smoothed / period)
                + plus_dm[i]
            )

            minus_smoothed = (
                minus_smoothed
                - (minus_smoothed / period)
                + minus_dm[i]
            )

        if atr_smoothed == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = (
                100.0
                * plus_smoothed
                / atr_smoothed
            )

            minus_di = (
                100.0
                * minus_smoothed
                / atr_smoothed
            )

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

        last_plus_di = plus_di
        last_minus_di = minus_di

    if len(dx_values) < period:
        adx = sum(dx_values) / len(dx_values)
    else:
        adx = sum(dx_values[-period:]) / period

    return adx, last_plus_di, last_minus_di


# ============================================================
# PRICE ACTION
# ============================================================

def candle_character(candles):
    if not candles:
        return {}

    c = candles[-1]

    body = abs(c["close"] - c["open"])
    total_range = c["high"] - c["low"]

    if total_range <= 0:
        return {
            "direction": "NEUTRAL",
            "body_ratio": 0,
            "upper_wick": 0,
            "lower_wick": 0,
        }

    upper_wick = (
        c["high"]
        - max(c["open"], c["close"])
    )

    lower_wick = (
        min(c["open"], c["close"])
        - c["low"]
    )

    body_ratio = body / total_range

    if c["close"] > c["open"]:
        direction = "BULLISH"
    elif c["close"] < c["open"]:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    return {
        "direction": direction,
        "body_ratio": round(body_ratio, 4),
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "range": total_range,
    }


def market_structure(candles):
    if len(candles) < 8:
        return "RANGE"

    first = candles[-8:-4]
    second = candles[-4:]

    first_high = max(x["high"] for x in first)
    first_low = min(x["low"] for x in first)

    second_high = max(x["high"] for x in second)
    second_low = min(x["low"] for x in second)

    if second_high > first_high and second_low > first_low:
        return "BULLISH"

    if second_high < first_high and second_low < first_low:
        return "BEARISH"

    return "RANGE"


def breakout_state(candles):
    if len(candles) < 9:
        return "NONE"

    last_close = candles[-1]["close"]

    previous = candles[-9:-1]

    previous_high = max(x["high"] for x in previous)
    previous_low = min(x["low"] for x in previous)

    if last_close > previous_high:
        return "UP_BREAKOUT"

    if last_close < previous_low:
        return "DOWN_BREAKOUT"

    return "NONE"


def liquidity_state(candles):
    if len(candles) < 10:
        return "UNKNOWN"

    recent = candles[-10:-1]

    previous_high = max(x["high"] for x in recent)
    previous_low = min(x["low"] for x in recent)

    last = candles[-1]

    # Sweep above liquidity then close back below
    if (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        return "HIGH_SWEEP"

    # Sweep below liquidity then close back above
    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        return "LOW_SWEEP"

    return "NONE"


def momentum_state(candles):
    if len(candles) < 6:
        return "UNKNOWN"

    recent = candles[-5:]

    bullish = 0
    bearish = 0

    for c in recent:
        if c["close"] > c["open"]:
            bullish += 1
        elif c["close"] < c["open"]:
            bearish += 1

    if bullish >= 4:
        return "BULLISH"

    if bearish >= 4:
        return "BEARISH"

    return "MIXED"


def recent_range(candles, count=8):
    recent = candles[-count:]

    if not recent:
        return None, None

    return (
        min(x["low"] for x in recent),
        max(x["high"] for x in recent),
    )


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_snapshot(candles):
    closes = [x["close"] for x in candles]

    last = candles[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema20 = ema(closes, 20)

    rsi14 = rsi(closes, 14)
    wr14 = williams_r(candles, 14)

    atr10 = atr(candles, 10)

    adx14, plus_di, minus_di = adx_di(
        candles,
        14,
    )

    structure = market_structure(candles)
    breakout = breakout_state(candles)
    liquidity = liquidity_state(candles)
    momentum = momentum_state(candles)

    candle_info = candle_character(candles)

    if ema20 is not None and atr10 is not None:
        keltner_upper = ema20 + (atr10 * 5.0)
        keltner_lower = ema20 - (atr10 * 5.0)
    else:
        keltner_upper = None
        keltner_lower = None

    recent_low, recent_high = recent_range(
        candles,
        8,
    )

    return {
        "price": last["close"],

        "open": last["open"],
        "high": last["high"],
        "low": last["low"],
        "close": last["close"],

        "ema9": ema9,
        "ema21": ema21,

        "rsi14": rsi14,
        "williams_r14": wr14,

        "atr10": atr10,

        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,

        "keltner_ema20": ema20,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,

        "structure": structure,
        "breakout": breakout,
        "liquidity": liquidity,
        "momentum": momentum,
        "candle": candle_info,

        "recent_low": recent_low,
        "recent_high": recent_high,
    }


# ============================================================
# PRE-SCORE
# ============================================================

def pre_score(snapshot):
    up = 0
    down = 0

    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]
    price = snapshot["price"]

    if ema9 is not None and ema21 is not None:

        if ema9 > ema21:
            up += 3
        elif ema9 < ema21:
            down += 3

    if ema9 is not None:

        if price > ema9:
            up += 1
        elif price < ema9:
            down += 1

    if ema21 is not None:

        if price > ema21:
            up += 1
        elif price < ema21:
            down += 1

    structure = snapshot["structure"]

    if structure == "BULLISH":
        up += 3
    elif structure == "BEARISH":
        down += 3

    breakout = snapshot["breakout"]

    if breakout == "UP_BREAKOUT":
        up += 3
    elif breakout == "DOWN_BREAKOUT":
        down += 3

    adx = snapshot["adx14"]
    plus_di = snapshot["plus_di14"]
    minus_di = snapshot["minus_di14"]

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
        and adx >= 20
    ):
        if plus_di > minus_di:
            up += 2
        elif minus_di > plus_di:
            down += 2

    rsi14 = snapshot["rsi14"]

    if rsi14 is not None:

        if 50 <= rsi14 <= 70:
            up += 1

        elif 30 <= rsi14 < 50:
            down += 1

    candle_direction = snapshot["candle"].get(
        "direction"
    )

    if candle_direction == "BULLISH":
        up += 1

    elif candle_direction == "BEARISH":
        down += 1

    if up > down:
        direction = "UP"
    elif down > up:
        direction = "DOWN"
    else:
        direction = (
            "UP"
            if (
                ema9 is not None
                and ema21 is not None
                and ema9 >= ema21
            )
            else "DOWN"
        )

    return {
        "up": up,
        "down": down,
        "direction": direction,
        "gap": abs(up - down),
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre,
    trade_type,
):
    recent = candles[-30:]

    compact_candles = []

    for c in recent:
        compact_candles.append({
            "time": c["time"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
        })

    return f"""
You are the core technical-analysis engine for ZinoProSignalAI.

MARKET:
Symbol: {symbol}
Timeframe: {timeframe}
Trade type: {trade_type}

IMPORTANT:
Analyze ONLY the supplied MT5 candle data and calculated indicators.
Do NOT invent prices, candles, indicators, news, market conditions,
support/resistance levels, or external information.

The objective is to select the stronger direction for the next trade.

You MUST return exactly one direction:
UP or DOWN.

Never return:
WAIT
NO SIGNAL
NEUTRAL

ANALYSIS PRIORITY:
1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle behavior
7. EMA 9 / EMA 21
8. RSI 14
9. Williams %R 14
10. Keltner
11. ADX / DI

INDICATOR SETTINGS:
EMA 9
EMA 21
RSI 14 with 70/30
Williams %R 14 with -20/-80
ADX 14
DI length 14
Keltner:
EMA 20
ATR 10
Multiplier 5

SCORING:
The final score MUST be exactly 18 points total.

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

Do NOT manufacture points.

A category can receive fewer points when evidence is weak.
The score must represent actual evidence from the supplied data.

The UP and DOWN scores must add to exactly 18.

Confidence must reflect evidence.
Do NOT use 90%+ confidence unless there is exceptionally strong
multi-factor confluence.

PRE-ANALYSIS FROM PYTHON:
{json.dumps(pre, ensure_ascii=False)}

TECHNICAL SNAPSHOT:
{json.dumps(snapshot, ensure_ascii=False)}

RECENT CLOSED CANDLES:
{json.dumps(compact_candles, ensure_ascii=False)}

Return ONLY valid JSON.

Required JSON:
{{
  "signal": true,
  "direction": "UP",
  "confidence": 75,
  "up_score": 13,
  "down_score": 5,
  "reason": "short technical reason",
  "cancellation_reason": "short cancellation condition"
}}

Rules:
- signal must be true
- direction must be UP or DOWN
- confidence integer 1-99
- up_score integer 0-18
- down_score integer 0-18
- up_score + down_score = 18
- reason short
- cancellation_reason short
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre,
    trade_type,
):
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client is not configured"
        )

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre,
        trade_type,
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.10,
            response_mime_type="application/json",
        ),
    )

    text_response = (
        getattr(response, "text", None)
        or ""
    ).strip()

    if not text_response:
        raise ValueError(
            "Gemini returned empty response"
        )

    try:
        data = json.loads(text_response)
    except Exception as e:
        logger.error(
            "Invalid Gemini JSON: %s",
            text_response,
        )
        raise ValueError(
            f"Invalid Gemini JSON: {e}"
        )

    return ensure_directional_signal(data)


# ============================================================
# VALIDATE GEMINI RESULT
# ============================================================

def ensure_directional_signal(data):
    if not isinstance(data, dict):
        raise ValueError("Gemini result is not an object")

    direction = str(
        data.get("direction", "")
    ).strip().upper()

    if direction not in ("UP", "DOWN"):
        raise ValueError(
            "Gemini returned invalid direction"
        )

    try:
        confidence = int(
            float(data.get("confidence", 50))
        )
    except Exception:
        confidence = 50

    confidence = max(
        1,
        min(99, confidence),
    )

    try:
        up_score = int(
            float(data.get("up_score", 0))
        )

        down_score = int(
            float(data.get("down_score", 0))
        )

    except Exception:
        raise ValueError(
            "Gemini returned invalid scores"
        )

    # IMPORTANT:
    # Never manufacture score points.
    # Reject malformed totals instead.
    if (
        up_score < 0
        or up_score > 18
        or down_score < 0
        or down_score > 18
    ):
        raise ValueError(
            "Gemini score outside 0-18"
        )

    if up_score + down_score != 18:
        raise ValueError(
            f"Gemini scores do not total 18: "
            f"{up_score}+{down_score}"
        )

    # Direction must agree with score
    if up_score > down_score:
        direction = "UP"

    elif down_score > up_score:
        direction = "DOWN"

    else:
        # Exact tie is allowed but use declared direction
        if direction not in ("UP", "DOWN"):
            direction = "UP"

    # Avoid unrealistic high confidence
    gap = abs(up_score - down_score)

    if confidence >= 90 and gap < 8:
        confidence = 89

    reason = str(
        data.get("reason", "")
    ).strip()

    cancellation_reason = str(
        data.get("cancellation_reason", "")
    ).strip()

    if not reason:
        reason = (
            "Strongest confluence comes from "
            "price action and structure."
        )

    if not cancellation_reason:
        if direction == "UP":
            cancellation_reason = (
                "إلغاء إذا أغلقت الشمعة تحت مستوى الإلغاء."
            )
        else:
            cancellation_reason = (
                "إلغاء إذا أغلقت الشمعة فوق مستوى الإلغاء."
            )

    return {
        "signal": True,
        "direction": direction,
        "confidence": confidence,
        "up_score": up_score,
        "down_score": down_score,
        "reason": reason[:350],
        "cancellation_reason": cancellation_reason[:250],
    }


# ============================================================
# ENTRY / CANCELLATION
# ============================================================

def calculate_entry(candles, timeframe, direction):
    last_closed = candles[-1]

    entry_price = last_closed["close"]

    tf_minutes = timeframe_minutes(timeframe)

    now = datetime.now(ALGIERS)

    entry_time = (
        now.replace(second=0, microsecond=0)
        + timedelta(minutes=tf_minutes)
    )

    recent = candles[-8:]

    if direction == "UP":
        cancellation_level = min(
            c["low"] for c in recent
        )

    else:
        cancellation_level = max(
            c["high"] for c in recent
        )

    return {
        "entry_price": entry_price,
        "entry_time": entry_time,
        "delay_minutes": tf_minutes,
        "cancellation_level": cancellation_level,
    }


def format_price(price):
    if price is None:
        return "N/A"

    try:
        value = float(price)

        if abs(value) >= 100:
            return f"{value:.3f}"

        if abs(value) >= 10:
            return f"{value:.4f}"

        return f"{value:.5f}"

    except Exception:
        return str(price)


# ============================================================
# TELEGRAM CARD
# ============================================================

def build_signal_card(
    symbol,
    timeframe,
    trade_type,
    result,
    entry,
):
    if trade_type == "RECOVERY":
        title_type = "🔁 RECOVERY 1/1"
    else:
        title_type = "🎯 BASE TRADE"

    direction = result["direction"]

    if direction == "UP":
        direction_text = "🟢 UP"
        cancellation = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(entry['cancellation_level'])}"
        )
    else:
        direction_text = "🔴 DOWN"
        cancellation = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(entry['cancellation_level'])}"
        )

    entry_time = entry["entry_time"].strftime(
        "%H:%M"
    )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{title_type}\n"
        f"{direction_text}\n\n"
        f"🔥 Confidence: {result['confidence']}%\n"
        f"🟢 UP Score: {result['up_score']}/18\n"
        f"🔴 DOWN Score: {result['down_score']}/18\n\n"
        f"⏱️ Entry after: {entry['delay_minutes']} minute"
        f"{'s' if entry['delay_minutes'] != 1 else ''}\n"
        f"🕐 ENTRY TIME: {entry_time} Algeria\n"
        f"💰 Entry Price: {format_price(entry['entry_price'])}\n"
        f"⚠️ {cancellation}\n\n"
        f"🧠 {result['reason']}"
    )


# ============================================================
# TRADE HISTORY
# ============================================================

def create_trade_record(
    symbol,
    timeframe,
    trade_type,
    result,
    entry,
):
    global current_trade_id

    trade_id = int(time.time() * 1000)

    record = {
        "id": trade_id,
        "created_at": datetime.now(
            ALGIERS
        ).isoformat(),

        "symbol": symbol,
        "timeframe": timeframe,
        "trade_type": trade_type,

        "direction": result["direction"],
        "confidence": result["confidence"],

        "up_score": result["up_score"],
        "down_score": result["down_score"],

        "entry_time": entry["entry_time"].isoformat(),
        "entry_price": entry["entry_price"],

        "cancellation_level": entry[
            "cancellation_level"
        ],

        "reason": result["reason"],
        "cancellation_reason": result[
            "cancellation_reason"
        ],

        "result": "PENDING",
    }

    trade_history.append(record)

    current_trade_id = trade_id

    save_history()

    return record


def find_latest_pending_trade():
    for record in reversed(trade_history):
        if record.get("result") == "PENDING":
            return record

    return None


def update_latest_trade_result(result):
    global current_trade_id

    record = None

    if current_trade_id is not None:

        for item in reversed(trade_history):
            if item.get("id") == current_trade_id:
                record = item
                break

    if record is None:
        record = find_latest_pending_trade()

    if record is None:
        return None

    record["result"] = result
    record["result_time"] = datetime.now(
        ALGIERS
    ).isoformat()

    current_trade_id = None

    save_history()

    return record


# ============================================================
# CYCLE MANAGEMENT
# ============================================================

def start_base_cycle(
    symbol,
    timeframe,
    direction,
):
    with cycle_lock:
        active_cycle["active"] = True
        active_cycle["symbol"] = symbol
        active_cycle["timeframe"] = timeframe
        active_cycle["direction"] = direction
        active_cycle["recovery_used"] = False
        active_cycle["last_trade_time"] = time.time()
        active_cycle["trade_number"] = 1


def start_recovery_cycle():
    with cycle_lock:

        if not active_cycle["active"]:
            return False

        if active_cycle["recovery_used"]:
            return False

        active_cycle["recovery_used"] = True
        active_cycle["trade_number"] = 2
        active_cycle["last_trade_time"] = time.time()

        return True


def reset_cycle():
    with cycle_lock:
        active_cycle["active"] = False
        active_cycle["symbol"] = None
        active_cycle["timeframe"] = None
        active_cycle["direction"] = None
        active_cycle["recovery_used"] = False
        active_cycle["last_trade_time"] = 0.0
        active_cycle["trade_number"] = 0


# ============================================================
# SETUP MEMORY
# ============================================================

def setup_key(
    symbol,
    timeframe,
    direction,
):
    return (
        f"{symbol}|"
        f"{timeframe}|"
        f"{direction}"
    )


def setup_recently_used(
    symbol,
    timeframe,
    direction,
):
    key = setup_key(
        symbol,
        timeframe,
        direction,
    )

    last_time = recent_setups.get(key)

    if last_time is None:
        return False

    return (
        time.time() - last_time
        < SETUP_REPEAT_BLOCK_SECONDS
    )


def remember_setup(
    symbol,
    timeframe,
    direction,
):
    key = setup_key(
        symbol,
        timeframe,
        direction,
    )

    recent_setups[key] = time.time()


# ============================================================
# NEW CLOSED CANDLE DETECTION
# ============================================================

def latest_closed_candle(candles):
    if len(candles) < 2:
        return None

    return candles[-2]


def get_closed_candles(candles):
    """
    MT5 EA is expected to send:
    oldest -> newest

    newest candle is normally the current forming candle.

    Therefore candles[:-1] are treated as closed.
    """

    if len(candles) < MIN_CLOSED_CANDLES + 1:
        return []

    return candles[:-1]


# ============================================================
# CHOOSE BEST PAIR
# ============================================================

def choose_best_pair():
    candidates = []

    with mt5_lock:
        markets = list(mt5_data.items())

    with cycle_lock:
        cycle_active = active_cycle["active"]
        cycle_symbol = active_cycle["symbol"]
        cycle_timeframe = active_cycle["timeframe"]

    for symbol, data in markets:

        timeframe = normalize_timeframe(
            data.get("timeframe")
        )

        if timeframe is None:
            continue

        candles = data.get("candles", [])

        closed = get_closed_candles(candles)

        if len(closed) < MIN_CLOSED_CANDLES:
            continue

        if cycle_active:

            if symbol != cycle_symbol:
                continue

            if timeframe != cycle_timeframe:
                continue

        try:
            snapshot = build_snapshot(closed)
            pre = pre_score(snapshot)

            candidates.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "snapshot": snapshot,
                "pre": pre,
            })

        except Exception as e:
            logger.warning(
                "Pre-analysis failed for %s: %s",
                symbol,
                e,
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            max(
                x["pre"]["up"],
                x["pre"]["down"],
            ),
            x["pre"]["gap"],
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# SEND TELEGRAM
# ============================================================

async def send_owner_message(text_message):
    if telegram_application is None:
        logger.warning(
            "Telegram application not ready"
        )
        return False

    oid = owner_id_int()

    if oid is None:
        logger.warning(
            "OWNER_ID not configured"
        )
        return False

    try:
        await telegram_application.bot.send_message(
            chat_id=oid,
            text=text_message,
        )

        return True

    except Exception as e:
        logger.exception(
            "Telegram send failed: %s",
            e,
        )

        return False


def send_owner_message_sync(text_message):
    global telegram_loop

    if telegram_loop is None:
        logger.warning(
            "Telegram loop not ready"
        )
        return False

    future = asyncio.run_coroutine_threadsafe(
        send_owner_message(text_message),
        telegram_loop,
    )

    try:
        future.result(timeout=30)
        return True
    except Exception as e:
        logger.exception(
            "Telegram future failed: %s",
            e,
        )
        return False


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(
    symbol,
    timeframe,
):
    timeframe = normalize_timeframe(timeframe)

    if timeframe is None:
        return False

    with mt5_lock:
        data = mt5_data.get(symbol)

        if not data:
            return False

        candles = list(
            data.get("candles", [])
        )

    if len(candles) < MIN_CLOSED_CANDLES + 1:
        logger.info(
            "Not enough MT5 candles | %s | %s | %s",
            symbol,
            timeframe,
            len(candles),
        )
        return False

    closed = get_closed_candles(candles)

    if len(closed) < MIN_CLOSED_CANDLES:
        return False

    # Detect new closed candle
    last_closed = closed[-1]
    candle_timestamp = last_closed["time"]

    market_key = f"{symbol}|{timeframe}"

    if (
        last_analysis_by_market.get(market_key)
        == candle_timestamp
    ):
        return False

    with cycle_lock:
        cycle_active = active_cycle["active"]
        cycle_symbol = active_cycle["symbol"]
        cycle_timeframe = active_cycle["timeframe"]
        recovery_used = active_cycle["recovery_used"]

    if cycle_active:

        if symbol != cycle_symbol:
            return False

        if timeframe != cycle_timeframe:
            return False

    # One analysis at a time
    if not analysis_lock.acquire(
        blocking=False
    ):
        return False

    try:

        snapshot = build_snapshot(closed)
        pre = pre_score(snapshot)

        if cycle_active:
            trade_type = (
                "RECOVERY"
                if recovery_used
                else "BASE"
            )
        else:
            trade_type = "BASE"

        # Avoid exact duplicate BASE setup
        if (
            trade_type == "BASE"
            and setup_recently_used(
                symbol,
                timeframe,
                pre["direction"],
            )
        ):
            last_analysis_by_market[
                market_key
            ] = candle_timestamp

            logger.info(
                "Duplicate setup blocked | %s | %s",
                symbol,
                timeframe,
            )

            return False

        logger.info(
            "Gemini analysis | %s | %s | %s",
            symbol,
            timeframe,
            trade_type,
        )

        result = analyze_with_gemini(
            symbol=symbol,
            timeframe=timeframe,
            candles=closed,
            snapshot=snapshot,
            pre=pre,
            trade_type=trade_type,
        )

        # Mark this candle as analyzed only after
        # successful Gemini validation.
        last_analysis_by_market[
            market_key
        ] = candle_timestamp

        direction = result["direction"]

        # Recovery follows the cycle and stays on same
        # symbol/timeframe. Direction is still determined
        # by the analysis.
        entry = calculate_entry(
            closed,
            timeframe,
            direction,
        )

        card = build_signal_card(
            symbol=symbol,
            timeframe=timeframe,
            trade_type=trade_type,
            result=result,
            entry=entry,
        )

        with signal_send_lock:
            sent = send_owner_message_sync(card)

        if not sent:
            logger.warning(
                "Signal could not be sent"
            )
            return False

        create_trade_record(
            symbol=symbol,
            timeframe=timeframe,
            trade_type=trade_type,
            result=result,
            entry=entry,
        )

        remember_setup(
            symbol,
            timeframe,
            direction,
        )

        if trade_type == "BASE":
            start_base_cycle(
                symbol,
                timeframe,
                direction,
            )

        else:
            with cycle_lock:
                active_cycle[
                    "last_trade_time"
                ] = time.time()

        logger.info(
            "SIGNAL SENT | %s | %s | %s | %s%% | %s/%s",
            symbol,
            timeframe,
            direction,
            result["confidence"],
            result["up_score"],
            result["down_score"],
        )

        return True

    except Exception as e:

        logger.exception(
            "Auto analysis failed | %s | %s | %s",
            symbol,
            timeframe,
            e,
        )

        return False

    finally:
        analysis_lock.release()


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def background_analysis_loop():
    logger.info(
        "Background analysis loop started"
    )

    while True:

        try:
            with cycle_lock:
                cycle_active = active_cycle[
                    "active"
                ]
                cycle_symbol = active_cycle[
                    "symbol"
                ]
                cycle_timeframe = active_cycle[
                    "timeframe"
                ]

            if cycle_active:

                auto_analyze_pair(
                    cycle_symbol,
                    cycle_timeframe,
                )

            else:

                best = choose_best_pair()

                if best:
                    auto_analyze_pair(
                        best["symbol"],
                        best["timeframe"],
                    )

        except Exception as e:
            logger.exception(
                "Background loop error: %s",
                e,
            )

        time.sleep(5)


# ============================================================
# MT5 DATA STORAGE
# ============================================================

def store_mt5_data(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Payload must be JSON object"
        )

    symbol = (
        payload.get("symbol")
        or payload.get("Symbol")
    )

    if not symbol:
        raise ValueError(
            "Missing symbol"
        )

    symbol = str(symbol).strip()

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or payload.get("TimeFrame")
        or payload.get("tf")
    )

    if timeframe is None:
        raise ValueError(
            "Invalid or missing timeframe"
        )

    candles = (
        payload.get("candles")
        or payload.get("data")
        or []
    )

    candles = normalize_candles(candles)

    if not candles:
        raise ValueError(
            "No valid candles received"
        )

    with mt5_lock:

        mt5_data[symbol] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "received_at": time.time(),
            "received_at_algiers": datetime.now(
                ALGIERS
            ).isoformat(),
        }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": len(candles),
    }


# ============================================================
# API KEY CHECK
# ============================================================

def valid_mt5_api_key(headers):
    received = (
        headers.get("X-MT5-API-Key")
        or headers.get("X-API-Key")
        or headers.get("X-MT4-API-Key")
    )

    if not received:
        return False

    # New MT5 key
    if MT5_API_KEY:
        return received.strip() == MT5_API_KEY

    # Temporary backward compatibility
    if OLD_MT4_API_KEY:
        return (
            received.strip()
            == OLD_MT4_API_KEY
        )

    # If no API key is configured,
    # reject rather than expose endpoint.
    return False


# ============================================================
# HTTP SERVER
# ============================================================

class MT5Handler(BaseHTTPRequestHandler):

    def log_message(self, format_string, *args):
        logger.info(
            "HTTP | " + format_string,
            *args,
        )

    def send_json(
        self,
        status,
        data,
    ):
        body = json.dumps(
            data,
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

        self.end_headers()

        self.wfile.write(body)

    def do_HEAD(self):

        parsed = urlparse(
            self.path
        )

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
            "/mt5",
            "/api/mt5",
            "/mt4",
            "/api/mt4",
        ):
            self.send_response(200)
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()

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
                    "service": (
                        "ZinoProSignalAI"
                    ),
                    "platform": "MT5",
                    "time": datetime.now(
                        ALGIERS
                    ).isoformat(),
                },
            )
            return

        if path in (
            "/mt5",
            "/api/mt5",
            "/mt4",
            "/api/mt4",
        ):
            with mt5_lock:
                markets = {}

                for symbol, data in mt5_data.items():

                    candles = data.get(
                        "candles",
                        [],
                    )

                    markets[symbol] = {
                        "timeframe": data.get(
                            "timeframe"
                        ),
                        "candles": len(
                            candles
                        ),
                        "received_at": data.get(
                            "received_at_algiers"
                        ),
                    }

            self.send_json(
                200,
                {
                    "ok": True,
                    "platform": "MT5",
                    "markets": markets,
                },
            )
            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "Not found",
            },
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path not in (
            "/mt5",
            "/api/mt5",
            "/mt4",
            "/api/mt4",
        ):
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found",
                },
            )
            return

        if not valid_mt5_api_key(
            self.headers
        ):
            self.send_json(
                401,
                {
                    "ok": False,
                    "error": "Unauthorized",
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

            if length <= 0:
                raise ValueError(
                    "Empty request body"
                )

            body = self.rfile.read(
                length
            )

            payload = json.loads(
                body.decode("utf-8")
            )

            stored = store_mt5_data(
                payload
            )

            logger.info(
                "MT5 DATA RECEIVED | %s | %s | %s candles",
                stored["symbol"],
                stored["timeframe"],
                stored["candles"],
            )

            # Analyze immediately in separate thread
            threading.Thread(
                target=auto_analyze_pair,
                args=(
                    stored["symbol"],
                    stored["timeframe"],
                ),
                daemon=True,
            ).start()

            self.send_json(
                200,
                {
                    "ok": True,
                    "platform": "MT5",
                    **stored,
                },
            )

        except Exception as e:

            logger.exception(
                "MT5 POST error: %s",
                e,
            )

            self.send_json(
                400,
                {
                    "ok": False,
                    "error": str(e),
                },
            )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        MT5Handler,
    )

    logger.info(
        "HTTP server started on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def cmd_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "MT5 signal engine is running.\n\n"
        "Commands:\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt5status\n"
        "/analyze"
    )


async def cmd_stats(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    total = (
        session_stats["wins"]
        + session_stats["losses"]
    )

    if total > 0:
        accuracy = (
            session_stats["wins"]
            / total
            * 100
        )
    else:
        accuracy = 0.0

    with cycle_lock:
        cycle = dict(active_cycle)

    text_message = (
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {session_stats['wins']}\n"
        f"🔴 Losses: {session_stats['losses']}\n"
        f"🎯 Accuracy: {accuracy:.1f}%\n\n"
        f"BASE Wins: {session_stats['base_wins']}\n"
        f"BASE Losses: {session_stats['base_losses']}\n"
        f"RECOVERY Wins: {session_stats['recovery_wins']}\n"
        f"RECOVERY Losses: {session_stats['recovery_losses']}\n\n"
        f"Cycle active: {'YES' if cycle['active'] else 'NO'}"
    )

    if cycle["active"]:
        text_message += (
            f"\nSymbol: {cycle['symbol']}"
            f"\nTimeframe: {cycle['timeframe']}"
            f"\nDirection: {cycle['direction']}"
            f"\nTrade: {cycle['trade_number']}/2"
        )

    await update.message.reply_text(
        text_message
    )


async def cmd_history(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    if not trade_history:
        await update.message.reply_text(
            "📭 No trade history."
        )
        return

    recent = trade_history[
        -HISTORY_DISPLAY_COUNT:
    ]

    lines = [
        "📜 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for trade in reversed(recent):

        result = trade.get(
            "result",
            "PENDING",
        )

        if result == "WIN":
            icon = "🟢"
        elif result == "LOSS":
            icon = "🔴"
        else:
            icon = "⏳"

        lines.append(
            f"{icon} "
            f"{trade.get('symbol')} "
            f"{trade.get('timeframe')} "
            f"{trade.get('direction')} "
            f"{trade.get('confidence')}% "
            f"[{trade.get('trade_type')}] "
            f"{result}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def cmd_win(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    record = update_latest_trade_result(
        "WIN"
    )

    if record is None:
        await update.message.reply_text(
            "⚠️ No pending trade."
        )
        return

    session_stats["wins"] += 1

    if record["trade_type"] == "BASE":
        session_stats["base_wins"] += 1
    else:
        session_stats["recovery_wins"] += 1

    save_history()

    reset_cycle()

    await update.message.reply_text(
        "🟢 WIN\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{record['symbol']} | "
        f"{record['timeframe']} | "
        f"{record['direction']}\n\n"
        "Cycle completed."
    )


async def cmd_loss(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    record = update_latest_trade_result(
        "LOSS"
    )

    if record is None:
        await update.message.reply_text(
            "⚠️ No pending trade."
        )
        return

    session_stats["losses"] += 1

    if record["trade_type"] == "BASE":
        session_stats["base_losses"] += 1

    else:
        session_stats["recovery_losses"] += 1

    save_history()

    if record["trade_type"] == "BASE":

        with cycle_lock:
            already_used = (
                active_cycle[
                    "recovery_used"
                ]
            )

        if not already_used:

            start_recovery_cycle()

            await update.message.reply_text(
                "🔴 BASE LOSS\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔁 Recovery 1/1 is now allowed.\n"
                f"📊 {record['symbol']} | "
                f"{record['timeframe']}\n"
                "The next analysis will remain "
                "on the same symbol/timeframe."
            )

            return

    # Recovery loss = stop cycle
    reset_cycle()

    await update.message.reply_text(
        "🔴 LOSS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Recovery limit reached.\n"
        "Cycle stopped."
    )


async def cmd_reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    session_stats.update({
        "wins": 0,
        "losses": 0,
        "base_wins": 0,
        "base_losses": 0,
        "recovery_wins": 0,
        "recovery_losses": 0,
    })

    trade_history.clear()

    recent_setups.clear()
    last_analysis_by_market.clear()

    reset_cycle()

    save_history()

    await update.message.reply_text(
        "♻️ All statistics, history and cycle "
        "state have been reset."
    )


async def cmd_mt5status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    with mt5_lock:
        markets = list(
            mt5_data.items()
        )

    if not markets:
        await update.message.reply_text(
            "🔌 MT5 STATUS\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "❌ No MT5 data received."
        )
        return

    lines = [
        "🔌 MT5 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for symbol, data in markets:

        candles = data.get(
            "candles",
            [],
        )

        tf = data.get(
            "timeframe",
            "?",
        )

        received = data.get(
            "received_at_algiers",
            "?",
        )

        lines.append(
            f"📊 {symbol} | {tf}"
        )

        lines.append(
            f"🕯️ Candles: {len(candles)}"
        )

        lines.append(
            f"🕐 Last receive: {received}"
        )

        if candles:
            last = candles[-1]

            lines.append(
                f"💰 Last: {format_price(last['close'])}"
            )

        lines.append("")

    await update.message.reply_text(
        "\n".join(lines)
    )


async def cmd_analyze(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    best = choose_best_pair()

    if not best:
        await update.message.reply_text(
            "⚠️ No suitable MT5 market found.\n"
            "Make sure the EA is sending at least "
            f"{MIN_CLOSED_CANDLES + 1} candles."
        )
        return

    await update.message.reply_text(
        "🔎 Starting analysis...\n"
        f"📊 {best['symbol']} | "
        f"{best['timeframe']}"
    )

    threading.Thread(
        target=auto_analyze_pair,
        args=(
            best["symbol"],
            best["timeframe"],
        ),
        daemon=True,
    ).start()


# ============================================================
# PHOTO HANDLER
# ============================================================

async def handle_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "📷 Image analysis is disabled in this MT5 version.\n"
        "Send market data from the MT5 EA."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_application
    global telegram_loop

    load_history()

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is not configured"
        )

    if not OWNER_ID:
        raise RuntimeError(
            "OWNER_ID is not configured"
        )

    if not MT5_API_KEY and not OLD_MT4_API_KEY:
        raise RuntimeError(
            "MT5_API_KEY is not configured"
        )

    # HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # Background analysis
    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
    )

    analysis_thread.start()

    # Telegram
    telegram_application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_application.add_handler(
        CommandHandler(
            "start",
            cmd_start,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "stats",
            cmd_stats,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "history",
            cmd_history,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "win",
            cmd_win,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "loss",
            cmd_loss,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "reset",
            cmd_reset,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "mt5status",
            cmd_mt5status,
        )
    )

    telegram_application.add_handler(
        CommandHandler(
            "analyze",
            cmd_analyze,
        )
    )

    telegram_application.add_handler(
        MessageHandler(
            filters.PHOTO,
            handle_photo,
        )
    )

    telegram_loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        telegram_loop
    )

    logger.info(
        "ZinoProSignalAI MT5 starting..."
    )

    telegram_application.run_polling(
        close_loop=False
    )


if __name__ == "__main__":
    main()
