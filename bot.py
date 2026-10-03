import os
import io
import json
import logging
import threading
import asyncio
import re
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from urllib.parse import urlparse
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from google import genai
from google.genai import types


# ============================================================
# SETTINGS
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

# تحليل تلقائي كل دقيقتين
AUTO_ANALYSIS_INTERVAL_MINUTES = 2

# أقل وقت مسموح قبل الدخول
MIN_ENTRY_LEAD_SECONDS = 40

# وقت الدخول بعد التحليل
ENTRY_DELAY_MINUTES = 2


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID_RAW:
    raise RuntimeError("OWNER_ID is missing")

if not MT4_API_KEY:
    raise RuntimeError("MT4_API_KEY is missing")

try:
    OWNER_ID = int(OWNER_ID_RAW)
except ValueError:
    raise RuntimeError("OWNER_ID must be an integer")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ZinoProSignalAI - %(levelname)s - %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GLOBAL DATA
# ============================================================

mt4_data = {}

mt4_lock = threading.Lock()

last_auto_signal = {}

last_analysis_time = {}

analysis_locks = {}

stats = {
    "wins": 0,
    "losses": 0,
}


# ============================================================
# TIMEFRAME HELPERS
# ============================================================

def timeframe_to_minutes(timeframe):
    if not timeframe:
        return 1

    tf = str(timeframe).upper().strip()

    match = re.match(r"^M(\d+)$", tf)
    if match:
        return int(match.group(1))

    match = re.match(r"^H(\d+)$", tf)
    if match:
        return int(match.group(1)) * 60

    return 1


def get_next_entry_time(
    timeframe,
    delay_minutes=None
):
    """
    Entry time is deliberately set 2 minutes after
    the moment the analysis is generated.

    This avoids signals arriving at the last second.
    """

    entry_time = (
        datetime.now(ALGIERS)
        + timedelta(minutes=ENTRY_DELAY_MINUTES)
    )

    return entry_time


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default

        return int(value)

    except Exception:
        return default


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ============================================================
# SCORE HELPERS
# ============================================================

def score_direction(result):
    direction = str(
        result.get("direction", "")
    ).upper().strip()

    if direction in ("UP", "CALL"):
        return "UP"

    if direction in ("DOWN", "PUT"):
        return "DOWN"

    return ""


def get_score(result, key):
    return safe_int(
        result.get(key, 0),
        0
    )


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(candle):
    if not isinstance(candle, dict):
        return None

    return {
        "time": candle.get(
            "time",
            candle.get("timestamp", "")
        ),
        "open": safe_float(
            candle.get("open", candle.get("o"))
        ),
        "high": safe_float(
            candle.get("high", candle.get("h"))
        ),
        "low": safe_float(
            candle.get("low", candle.get("l"))
        ),
        "close": safe_float(
            candle.get("close", candle.get("c"))
        ),
        "volume": safe_float(
            candle.get("volume", candle.get("v")),
            0
        ),
    }


def normalize_candles(candles):
    result = []

    if not isinstance(candles, list):
        return result

    for candle in candles:
        normalized = normalize_candle(candle)

        if normalized:
            result.append(normalized)

    return result


# ============================================================
# CANDLE TIME
# ============================================================

def candle_timestamp(candle):
    value = candle.get("time", "")

    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return 0

    try:
        return float(text)
    except Exception:
        pass

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(
                tzinfo=ALGIERS
            ).timestamp()

        except Exception:
            continue

    return 0


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles):
    """
    MT4 sends the newest candle first.
    The newest candle is considered the currently
    forming candle.

    Therefore we exclude it and analyze only closed candles.
    """

    normalized = normalize_candles(candles)

    if len(normalized) <= 1:
        return []

    normalized.sort(
        key=candle_timestamp
    )

    return normalized[:-1]


# ============================================================
# EMA
# ============================================================

def calculate_ema(values, period):
    if not values:
        return []

    if len(values) < period:
        return []

    multiplier = 2 / (period + 1)

    ema = []

    sma = sum(
        values[:period]
    ) / period

    ema.append(sma)

    previous = sma

    for price in values[period:]:
        current = (
            (price - previous)
            * multiplier
            + previous
        )

        ema.append(current)

        previous = current

    return ema


# ============================================================
# RSI
# ============================================================

def calculate_rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0)

        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = (
        sum(gains[:period]) / period
    )

    avg_loss = (
        sum(losses[:period]) / period
    )

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# WILLIAMS %R
# ============================================================

def calculate_williams_r(
    candles,
    period=14
):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(
        c["high"]
        for c in recent
    )

    lowest = min(
        c["low"]
        for c in recent
    )

    close = recent[-1]["close"]

    if highest == lowest:
        return -50.0

    return (
        (highest - close)
        / (highest - lowest)
        * -100
    )


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles,
    period=10
):
    if len(candles) <= period:
        return None

    true_ranges = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

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

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    return (
        sum(
            true_ranges[-period:]
        ) / period
    )


# ============================================================
# ADX / DI
# ============================================================

def calculate_adx_di(
    candles,
    period=14
):
    if len(candles) < period + 2:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

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

        if (
            up_move > down_move
            and up_move > 0
        ):
            plus = up_move
        else:
            plus = 0

        if (
            down_move > up_move
            and down_move > 0
        ):
            minus = down_move
        else:
            minus = 0

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

    if len(trs) < period:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    atr = (
        sum(trs[-period:])
        / period
    )

    if atr == 0:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    plus = (
        sum(plus_dm[-period:])
        / period
    )

    minus = (
        sum(minus_dm[-period:])
        / period
    )

    plus_di = (
        100 * plus / atr
    )

    minus_di = (
        100 * minus / atr
    )

    dx_denominator = (
        plus_di + minus_di
    )

    if dx_denominator == 0:
        adx = 0
    else:
        adx = (
            100
            * abs(
                plus_di - minus_di
            )
            / dx_denominator
        )

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# MARKET STRUCTURE
# ============================================================

def detect_structure(candles):
    if len(candles) < 10:
        return "UNKNOWN"

    recent = candles[-10:]

    highs = [
        c["high"]
        for c in recent
    ]

    lows = [
        c["low"]
        for c in recent
    ]

    if (
        highs[-1] > highs[-3]
        and lows[-1] > lows[-3]
    ):
        return "BULLISH"

    if (
        highs[-1] < highs[-3]
        and lows[-1] < lows[-3]
    ):
        return "BEARISH"

    return "RANGE"


# ============================================================
# BREAKOUT
# ============================================================

def detect_breakout(candles):
    if len(candles) < 8:
        return {
            "bullish": False,
            "bearish": False,
        }

    recent = candles[-8:-1]

    resistance = max(
        c["high"]
        for c in recent
    )

    support = min(
        c["low"]
        for c in recent
    )

    close = candles[-1]["close"]

    return {
        "bullish": close > resistance,
        "bearish": close < support,
    }


# ============================================================
# CANCELLATION LEVEL
# ============================================================

def calculate_cancellation(
    candles,
    direction
):
    if len(candles) < 5:
        return None

    recent = candles[-5:]

    if direction == "UP":
        level = min(
            c["low"]
            for c in recent
        )

        return level

    if direction == "DOWN":
        level = max(
            c["high"]
            for c in recent
        )

        return level

    return None


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def technical_snapshot(candles):
    closes = [
        c["close"]
        for c in candles
    ]

    ema9 = calculate_ema(
        closes,
        9
    )

    ema21 = calculate_ema(
        closes,
        21
    )

    rsi = calculate_rsi(
        closes,
        14
    )

    williams = calculate_williams_r(
        candles,
        14
    )

    atr = calculate_atr(
        candles,
        10
    )

    adx_data = calculate_adx_di(
        candles,
        14
    )

    current_close = (
        closes[-1]
        if closes
        else None
    )

    ema9_value = (
        ema9[-1]
        if ema9
        else None
    )

    ema21_value = (
        ema21[-1]
        if ema21
        else None
    )

    keltner_middle = calculate_ema(
        closes,
        20
    )

    keltner_middle_value = (
        keltner_middle[-1]
        if keltner_middle
        else None
    )

    keltner_upper = None
    keltner_lower = None

    if (
        keltner_middle_value is not None
        and atr is not None
    ):
        keltner_upper = (
            keltner_middle_value
            + 5 * atr
        )

        keltner_lower = (
            keltner_middle_value
            - 5 * atr
        )

    return {
        "close": current_close,

        "ema9": ema9_value,
        "ema21": ema21_value,

        "rsi14": rsi,
        "williams_r14": williams,

        "atr10": atr,

        "adx14": adx_data["adx"],
        "plus_di14": adx_data["plus_di"],
        "minus_di14": adx_data["minus_di"],

        "keltner_ema20": keltner_middle_value,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,

        "structure": detect_structure(
            candles
        ),

        "breakout": detect_breakout(
            candles
        ),
    }


# ============================================================
# SIGNAL QUALITY
# ============================================================

def evaluate_signal_quality(
    result,
    candles
):
    """
    Strict filter.

    Weak setups are rejected and
    NO Telegram signal is sent.
    """

    if not isinstance(result, dict):
        return False, "Invalid result"

    direction = score_direction(
        result
    )

    if direction not in (
        "UP",
        "DOWN",
    ):
        return False, "Invalid direction"

    total_score = safe_int(
        result.get(
            "total_score",
            result.get("score", 0)
        ),
        0
    )

    up_score = safe_int(
        result.get(
            "up_score",
            0
        ),
        0
    )

    down_score = safe_int(
        result.get(
            "down_score",
            0
        ),
        0
    )

    confidence = safe_int(
        result.get(
            "confidence",
            0
        ),
        0
    )

    confirmations = safe_int(
        result.get(
            "confirmations",
            0
        ),
        0
    )

    contradictions = safe_int(
        result.get(
            "contradictions",
            99
        ),
        99
    )

    if total_score != 18:
        return (
            False,
            "Total score must equal 18"
        )

    selected_score = (
        up_score
        if direction == "UP"
        else down_score
    )

    opposite_score = (
        down_score
        if direction == "UP"
        else up_score
    )

    if selected_score < 11:
        return (
            False,
            "Selected score below 11/18"
        )

    if selected_score - opposite_score < 5:
        return (
            False,
            "Score difference below 5"
        )

    if confidence < 70:
        return (
            False,
            "Confidence below 70%"
        )

    if confidence >= 90:
        if selected_score < 15:
            return (
                False,
                "90%+ requires very strong score"
            )

    if len(candles) < 40:
        return (
            False,
            "Not enough closed candles"
        )

    if confirmations < 4:
        return (
            False,
            "Not enough confirmations"
        )

    if contradictions >= 2:
        return (
            False,
            "Too many contradictions"
        )

    exhaustion = str(
        result.get(
            "exhaustion",
            ""
        )
    ).lower()

    if exhaustion in (
        "high",
        "extreme",
        "true",
    ):
        return (
            False,
            "Exhaustion protection"
        )

    return True, "Strong setup"


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles
):
    snapshot = technical_snapshot(
        candles
    )

    recent = candles[-40:]

    prompt = f"""
You are the technical analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied MT4 candle data.

PAIR:
{symbol}

TIMEFRAME:
{timeframe}

IMPORTANT RULES:

1. Do NOT invent market data.
2. Do NOT invent indicators.
3. Use only supplied candles and calculated values.
4. The direction must be based on evidence.
5. Do NOT force UP.
6. Do NOT force DOWN.
7. Weak setups must be rejected.
8. Prefer NO_SIGNAL internally when the setup is weak.
9. The Telegram bot will NOT send weak setups.
10. Never use martingale.
11. Never recommend doubling.
12. Never guarantee a win.
13. Do not use support/resistance lines in the final Telegram card.
14. Avoid 90%+ confidence unless confluence is exceptionally strong.

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

SCORING:

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 3
Moving Averages = 3

TOTAL = 18

REQUIRED QUALITY:

- selected direction score >= 11/18
- score difference >= 5
- confidence >= 70
- confirmations >= 4
- contradictions < 2
- at least 40 closed candles
- no extreme exhaustion

CALCULATED INDICATORS:

{json.dumps(snapshot, ensure_ascii=False, indent=2)}

RECENT CLOSED CANDLES:

{json.dumps(recent, ensure_ascii=False, indent=2)}

Return ONLY valid JSON.

JSON FORMAT:

{{
  "direction": "UP",
  "confidence": 75,
  "total_score": 18,
  "up_score": 13,
  "down_score": 5,

  "structure_score": 2,
  "breakout_score": 2,
  "liquidity_score": 1,
  "momentum_score": 2,
  "candle_score": 2,
  "rsi_score": 1,
  "summary_score": 2,
  "oscillators_score": 3,
  "moving_averages_score": 3,

  "confirmations": 5,
  "contradictions": 0,
  "exhaustion": "low",

  "reason": "Short technical reason"
}}

If setup is weak, return:

{{
  "direction": "NO_SIGNAL",
  "confidence": 0,
  "total_score": 18,
  "up_score": 0,
  "down_score": 0,
  "structure_score": 0,
  "breakout_score": 0,
  "liquidity_score": 0,
  "momentum_score": 0,
  "candle_score": 0,
  "rsi_score": 0,
  "summary_score": 0,
  "oscillators_score": 0,
  "moving_averages_score": 0,
  "confirmations": 0,
  "contradictions": 99,
  "exhaustion": "high",
  "reason": "Weak setup"
}}
"""

    return prompt


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles
):
    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

    except Exception as e:
        logger.exception(
            "Gemini API error: %s",
            e
        )

        return None

    try:
        text = response.text

        if not text:
            logger.error(
                "Gemini returned empty response"
            )

            return None

        result = json.loads(
            text.strip()
        )

        if not isinstance(result, dict):
            return None

        return result

    except Exception as e:
        logger.exception(
            "Gemini JSON parsing error: %s",
            e
        )

        return None


# ============================================================
# PRICE FORMAT
# ============================================================

def format_price(price):
    if price is None:
        return "N/A"

    price = safe_float(
        price
    )

    if price == 0:
        return "0"

    if abs(price) >= 100:
        return f"{price:.3f}"

    if abs(price) >= 10:
        return f"{price:.4f}"

    if abs(price) >= 1:
        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# MT4 SIGNAL FORMAT
# ============================================================

def format_mt4_signal(
    symbol,
    timeframe,
    result,
    current_price,
    candles
):
    direction = score_direction(
        result
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    cancellation = calculate_cancellation(
        candles,
        direction
    )

    direction_text = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    if cancellation is None:
        cancellation_text = (
            "⚠️ إلغاء حسب آخر حركة سعرية"
        )

    elif direction == "UP":
        cancellation_text = (
            "⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(cancellation)}"
        )

    else:
        cancellation_text = (
            "⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(cancellation)}"
        )

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{direction_text}\n"
        f"⏰ {entry_time.strftime('%H:%M:%S')}\n"
        f"💰 {format_price(current_price)}\n"
        f"{cancellation_text}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# ============================================================
# SCREENSHOT SIGNAL FORMAT
# ============================================================

def format_full_signal(
    symbol,
    timeframe,
    result,
    entry_price,
    candles
):
    direction = score_direction(
        result
    )

    confidence = safe_int(
        result.get(
            "confidence",
            0
        )
    )

    up_score = safe_int(
        result.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        result.get(
            "down_score",
            0
        )
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    cancellation = calculate_cancellation(
        candles,
        direction
    )

    if direction == "UP":
        direction_text = "🟢 UP"

        if cancellation is not None:
            cancel_text = (
                "⚠️ إلغاء إذا أغلقت الشمعة تحت "
                f"{format_price(cancellation)}"
            )
        else:
            cancel_text = (
                "⚠️ إلغاء حسب آخر حركة سعرية"
            )

    else:
        direction_text = "🔴 DOWN"

        if cancellation is not None:
            cancel_text = (
                "⚠️ إلغاء إذا أغلقت الشمعة فوق "
                f"{format_price(cancellation)}"
            )
        else:
            cancel_text = (
                "⚠️ إلغاء حسب آخر حركة سعرية"
            )

    reason = str(
        result.get(
            "reason",
            ""
        )
    ).strip()

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{direction_text}\n"
        f"🎯 Confidence: {confidence}%\n"
        f"📈 UP Score: {up_score}/18\n"
        f"📉 DOWN Score: {down_score}/18\n"
        f"⏳ Entry after: {ENTRY_DELAY_MINUTES} min\n"
        f"⏰ Entry Time: {entry_time.strftime('%H:%M:%S')}\n"
        f"💰 Entry Price: {format_price(entry_price)}\n"
        f"{cancel_text}\n\n"
        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# ============================================================
# MT4 ANALYSIS
# ============================================================

def analyze_mt4_payload(
    payload
):
    if not isinstance(payload, dict):
        return None, "Invalid payload"

    symbol = str(
        payload.get(
            "symbol",
            ""
        )
    ).strip().upper()

    timeframe = str(
        payload.get(
            "timeframe",
            "M1"
        )
    ).strip().upper()

    candles = payload.get(
        "candles",
        []
    )

    if not symbol:
        return None, "Missing symbol"

    if not candles:
        return None, "Missing candles"

    closed_candles = get_closed_candles(
        candles
    )

    if len(closed_candles) < 40:
        return (
            None,
            "Not enough closed candles"
        )

    result = analyze_with_gemini(
        symbol,
        timeframe,
        closed_candles
    )

    if result is None:
        return None, "Gemini analysis failed"

    good, reason = evaluate_signal_quality(
        result,
        closed_candles
    )

    if not good:
        return None, reason

    current_price = safe_float(
        payload.get(
            "price",
            closed_candles[-1]["close"]
        )
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "result": result,
        "candles": closed_candles,
        "price": current_price,
    }, "Strong setup"


# ============================================================
# AUTO ANALYSIS
# ============================================================

def get_candle_identity(payload):
    candles = payload.get(
        "candles",
        []
    )

    if not candles:
        return None

    newest = candles[0]

    if not isinstance(newest, dict):
        return None

    value = newest.get(
        "time",
        newest.get(
            "timestamp",
            ""
        )
    )

    return str(value)


async def run_auto_analysis(
    application,
    symbol,
    timeframe
):
    key = f"{symbol}:{timeframe}"

    lock = analysis_locks.setdefault(
        key,
        asyncio.Lock()
    )

    if lock.locked():
        logger.info(
            "AUTO ANALYSIS SKIPPED: %s",
            key
        )

        return

    async with lock:
        logger.info(
            "AUTO ANALYSIS START: %s",
            key
        )

        with mt4_lock:
            payload = mt4_data.get(
                key
            )

            if not payload:
                logger.warning(
                    "No MT4 payload for %s",
                    key
                )

                return

            payload = dict(payload)

        result, reason = analyze_mt4_payload(
            payload
        )

        if result is None:
            logger.info(
                "SIGNAL REJECTED: %s | %s",
                key,
                reason
            )

            return

        signal = result["result"]

        direction = score_direction(
            signal
        )

        if direction not in (
            "UP",
            "DOWN",
        ):
            logger.info(
                "SIGNAL REJECTED: %s | invalid direction",
                key
            )

            return

        message = format_mt4_signal(
            result["symbol"],
            result["timeframe"],
            signal,
            result["price"],
            result["candles"],
        )

        try:
            await application.bot.send_message(
                chat_id=OWNER_ID,
                text=message
            )

        except Exception as e:
            logger.exception(
                "Telegram send error: %s",
                e
            )

            return

        last_auto_signal[key] = {
            "direction": direction,
            "time": datetime.now(
                ALGIERS
            ).isoformat(),
            "confidence": signal.get(
                "confidence",
                0
            ),
        }

        logger.info(
            "AUTO SIGNAL SENT: %s | %s",
            key,
            direction
        )


# ============================================================
# SCHEDULE AUTO ANALYSIS
# ============================================================

def schedule_auto_analysis(
    application,
    symbol,
    timeframe
):
    key = f"{symbol}:{timeframe}"

    now = time.time()

    previous = last_analysis_time.get(
        key,
        0
    )

    elapsed = now - previous

    if elapsed < (
        AUTO_ANALYSIS_INTERVAL_MINUTES
        * 60
    ):
        logger.info(
            "AUTO ANALYSIS THROTTLED: %s | %.1fs",
            key,
            elapsed
        )

        return

    last_analysis_time[key] = now

    logger.info(
        "AUTO ANALYSIS SCHEDULED: %s",
        key
    )

    application.create_task(
        run_auto_analysis(
            application,
            symbol,
            timeframe
        )
    )


# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format,
        *args
    ):
        return

    def do_GET(self):
        parsed = urlparse(
            self.path
        )

        if parsed.path in (
            "/",
            "/health",
        ):
            body = (
                "ZinoProSignalAI is running"
            ).encode(
                "utf-8"
            )

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )

            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(
            self.path
        )

        if parsed.path != "/mt4":
            self.send_response(404)
            self.end_headers()
            return

        api_key = self.headers.get(
            "X-MT4-API-KEY",
            ""
        ).strip()

        if api_key != MT4_API_KEY:
            self.send_response(401)
            self.end_headers()

            try:
                self.wfile.write(
                    b"Unauthorized"
                )
            except Exception:
                pass

            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except Exception as e:
            logger.exception(
                "MT4 JSON error: %s",
                e
            )

            self.send_response(400)
            self.end_headers()

            try:
                self.wfile.write(
                    b"Invalid JSON"
                )
            except Exception:
                pass

            return

        symbol = str(
            payload.get(
                "symbol",
                ""
            )
        ).strip().upper()

        timeframe = str(
            payload.get(
                "timeframe",
                "M1"
            )
        ).strip().upper()

        candles = payload.get(
            "candles",
            []
        )

        if not symbol:
            self.send_response(400)
            self.end_headers()
            return

        key = f"{symbol}:{timeframe}"

        candle_identity = get_candle_identity(
            payload
        )

        with mt4_lock:
            old_payload = mt4_data.get(
                key
            )

            old_identity = (
                get_candle_identity(
                    old_payload
                )
                if old_payload
                else None
            )

            mt4_data[key] = payload

        is_new_candle = (
            candle_identity is not None
            and candle_identity != old_identity
        )

        logger.info(
            "MT4 data received: %s | %s | candles=%s | new_candle=%s",
            symbol,
            timeframe,
            len(candles)
            if isinstance(candles, list)
            else 0,
            is_new_candle
        )

        if is_new_candle:
            logger.info(
                "NEW CANDLE: %s | %s | candles=%s",
                symbol,
                timeframe,
                len(candles)
                if isinstance(candles, list)
                else 0
            )

            application = self.server.application

            schedule_auto_analysis(
                application,
                symbol,
                timeframe
            )

        response = {
            "ok": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": (
                len(candles)
                if isinstance(candles, list)
                else 0
            ),
            "new_candle": is_new_candle,
        }

        body = json.dumps(
            response
        ).encode(
            "utf-8"
        )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(
            body
        )


class ZinoHTTPServer(
    ThreadingHTTPServer
):

    daemon_threads = True


def start_http_server(
    application
):
    server = ZinoHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        HealthHandler
    )

    server.application = application

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update):
    if not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID
    )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ Bot is running.\n"
        "📡 MT4 auto-analysis enabled.\n"
        "⏱️ Analysis interval: 2 minutes.\n"
        "⏳ Entry delay: 2 minutes."
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    total = (
        stats["wins"]
        + stats["losses"]
    )

    if total:
        winrate = (
            stats["wins"]
            / total
            * 100
        )
    else:
        winrate = 0

    message = (
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"✅ Wins: {stats['wins']}\n"
        f"❌ Losses: {stats['losses']}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {winrate:.1f}%"
    )

    await update.message.reply_text(
        message
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    stats["wins"] += 1

    await update.message.reply_text(
        f"✅ WIN recorded\n"
        f"Total Wins: {stats['wins']}"
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    stats["losses"] += 1

    await update.message.reply_text(
        f"❌ LOSS recorded\n"
        f"Total Losses: {stats['losses']}"
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    stats["wins"] = 0
    stats["losses"] = 0

    await update.message.reply_text(
        "♻️ Statistics reset."
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with mt4_lock:
        count = len(
            mt4_data
        )

        keys = list(
            mt4_data.keys()
        )

    if keys:
        pairs = "\n".join(
            f"• {key}"
            for key in keys
        )
    else:
        pairs = "None"

    message = (
        "📡 MT4 Status\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Connected feeds: {count}\n\n"
        f"{pairs}"
    )

    await update.message.reply_text(
        message
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with mt4_lock:
        items = list(
            mt4_data.items()
        )

    if not items:
        await update.message.reply_text(
            "❌ No MT4 data received yet."
        )

        return

    sent = 0

    for key, payload in items:
        result, reason = analyze_mt4_payload(
            payload
        )

        if result is None:
            logger.info(
                "MANUAL SIGNAL REJECTED: %s | %s",
                key,
                reason
            )

            continue

        message = format_mt4_signal(
            result["symbol"],
            result["timeframe"],
            result["result"],
            result["price"],
            result["candles"],
        )

        await update.message.reply_text(
            message
        )

        sent += 1

    if sent == 0:
        await update.message.reply_text(
            "❌ No strong setup found.\n"
            "No signal was sent."
        )


# ============================================================
# TELEGRAM TEXT
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    if not update.message:
        return

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:
        return

    logger.info(
        "Telegram text received: %s",
        text
    )

    await update.message.reply_text(
        "📡 ZinoProSignalAI\n\n"
        "أرسل Screenshot للشارت للتحليل، "
        "أو استخدم /mt4status لمعرفة حالة MT4."
    )


# ============================================================
# SCREENSHOT ANALYSIS
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    if not update.message:
        return

    photo = update.message.photo

    if not photo:
        return

    await update.message.reply_text(
        "🔎 جاري تحليل الشارت..."
    )

    try:
        largest_photo = photo[-1]

        telegram_file = await context.bot.get_file(
            largest_photo.file_id
        )

        image_buffer = io.BytesIO()

        await telegram_file.download_to_memory(
            image_buffer
        )

        image_bytes = image_buffer.getvalue()

        prompt = """
Analyze this trading chart for ZinoProSignalAI.

Use only visible information.

Priority:
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

Rules:

- Do not invent indicators.
- Do not force UP.
- Do not force DOWN.
- Reject weak setups.
- No martingale.
- No guarantee.
- Confidence 90%+ only with exceptional confluence.
- Total score is 18.

Return ONLY JSON:

{
  "direction": "UP",
  "confidence": 75,
  "total_score": 18,
  "up_score": 13,
  "down_score": 5,
  "confirmations": 5,
  "contradictions": 0,
  "exhaustion": "low",
  "reason": "Short technical reason"
}
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(
            response.text.strip()
        )

        direction = score_direction(
            result
        )

        if direction not in (
            "UP",
            "DOWN",
        ):
            await update.message.reply_text(
                "❌ Setup ضعيف.\n"
                "لم يتم إرسال إشارة."
            )

            return

        confidence = safe_int(
            result.get(
                "confidence",
                0
            )
        )

        up_score = safe_int(
            result.get(
                "up_score",
                0
            )
        )

        down_score = safe_int(
            result.get(
                "down_score",
                0
            )
        )

        total_score = safe_int(
            result.get(
                "total_score",
                0
            )
        )

        confirmations = safe_int(
            result.get(
                "confirmations",
                0
            )
        )

        contradictions = safe_int(
            result.get(
                "contradictions",
                99
            )
        )

        selected_score = (
            up_score
            if direction == "UP"
            else down_score
        )

        opposite_score = (
            down_score
            if direction == "UP"
            else up_score
        )

        if (
            total_score != 18
            or selected_score < 11
            or selected_score - opposite_score < 5
            or confidence < 70
            or confirmations < 4
            or contradictions >= 2
        ):
            await update.message.reply_text(
                "❌ Setup ضعيف.\n"
                "لم يتم إرسال إشارة."
            )

            return

        # محاولة استخراج السعر من caption
        caption = (
            update.message.caption
            or ""
        )

        symbol = "UNKNOWN"
        timeframe = "M1"

        symbol_match = re.search(
            r"([A-Z]{3,6}[/_-]?[A-Z]{3,6})",
            caption.upper()
        )

        if symbol_match:
            symbol = (
                symbol_match
                .group(1)
                .replace("_", "")
                .replace("-", "")
            )

        tf_match = re.search(
            r"\b(M\d+|H\d+)\b",
            caption.upper()
        )

        if tf_match:
            timeframe = tf_match.group(1)

        entry_time = get_next_entry_time(
            timeframe
        )

        reason = str(
            result.get(
                "reason",
                ""
            )
        ).strip()

        message = (
            "🎓 ZinoProSignalAI\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol} | {timeframe}\n\n"
            f"{'🟢 UP' if direction == 'UP' else '🔴 DOWN'}\n"
            f"🎯 Confidence: {confidence}%\n"
            f"📈 UP Score: {up_score}/18\n"
            f"📉 DOWN Score: {down_score}/18\n"
            f"⏳ Entry after: {ENTRY_DELAY_MINUTES} min\n"
            f"⏰ Entry Time: {entry_time.strftime('%H:%M:%S')}\n"
        )

        if reason:
            message += (
                f"\n📝 {reason}\n"
            )

        message += (
            "━━━━━━━━━━━━━━━━━━"
        )

        await update.message.reply_text(
            message
        )

    except Exception as e:
        logger.exception(
            "Screenshot analysis error: %s",
            e
        )

        await update.message.reply_text(
            "❌ حدث خطأ أثناء تحليل الصورة."
        )


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(
        "Starting ZinoProSignalAI..."
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Auto analysis interval: %s minutes",
        AUTO_ANALYSIS_INTERVAL_MINUTES
    )

    logger.info(
        "Entry delay: %s minutes",
        ENTRY_DELAY_MINUTES
    )

    logger.info(
        "Minimum entry lead: %s seconds",
        MIN_ENTRY_LEAD_SECONDS
    )

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

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler
        )
    )

    http_thread = threading.Thread(
        target=start_http_server,
        args=(application,),
        daemon=True,
    )

    http_thread.start()

    logger.info(
        "HTTP health server started"
    )

    logger.info(
        "Starting Telegram polling..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
