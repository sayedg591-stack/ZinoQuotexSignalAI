````python
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
OWNER_ID = os.getenv("OWNER_ID", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

# تحليل تلقائي كل 3 دقائق
AUTO_ANALYSIS_INTERVAL_MINUTES = 3

# وقت الدخول بعد إنشاء التحليل
ENTRY_DELAY_MINUTES = 2

# يجب أن يبقى هناك وقت كافٍ للدخول
MIN_ENTRY_LEAD_SECONDS = 40


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID:
    raise RuntimeError("OWNER_ID is missing")

if not MT4_API_KEY:
    raise RuntimeError("MT4_API_KEY is missing")

try:
    OWNER_ID_INT = int(OWNER_ID)
except Exception:
    raise RuntimeError("OWNER_ID must be an integer")


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GLOBAL MT4 DATA
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

# آخر شمعة مغلقة تمت معالجتها لكل رمز/فريم
last_processed_closed_candle = {}

# آخر وقت تحليل لكل رمز/فريم
last_auto_analysis = {}

# لمنع تحليلين في نفس الوقت
analysis_lock = threading.Lock()

# إحصائيات البوت
stats_data = {
    "wins": 0,
    "losses": 0,
}


# ============================================================
# TIME HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def format_algiers(dt):
    if dt is None:
        return "--:--:--"

    return dt.astimezone(ALGIERS).strftime("%H:%M:%S")


def parse_mt4_time(value):
    try:
        if value is None:
            return None

        value = int(value)

        # Unix timestamp
        return datetime.fromtimestamp(
            value,
            tz=ZoneInfo("UTC")
        ).astimezone(ALGIERS)

    except Exception:
        return None


# ============================================================
# TIMEFRAME
# ============================================================

def timeframe_to_minutes(timeframe):
    if not timeframe:
        return 1

    text = str(timeframe).upper().strip()

    match = re.search(r"(\d+)", text)

    if not match:
        return 1

    number = int(match.group(1))

    if "MN" in text:
        return number * 43200

    if "W" in text:
        return number * 10080

    if "D" in text:
        return number * 1440

    if "H" in text:
        return number * 60

    return number


# ============================================================
# ENTRY TIME
# ============================================================

def get_next_entry_time(timeframe, delay_minutes=None):

    if delay_minutes is None:
        delay_minutes = ENTRY_DELAY_MINUTES

    return now_algiers() + timedelta(
        minutes=delay_minutes
    )


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(candle):

    if not isinstance(candle, dict):
        return None

    return {
        "time": safe_int(candle.get("time")),
        "open": safe_float(candle.get("open")),
        "high": safe_float(candle.get("high")),
        "low": safe_float(candle.get("low")),
        "close": safe_float(candle.get("close")),
        "volume": safe_int(candle.get("volume")),
    }


def normalize_candles(candles):

    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:
        normalized = normalize_candle(candle)

        if normalized is not None:
            result.append(normalized)

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles):

    """
    MT4 sends candles from oldest -> newest.

    The newest candle can be the currently forming candle,
    therefore we exclude it.

    The remaining candles are closed candles.
    """

    candles = normalize_candles(candles)

    if len(candles) < 3:
        return []

    return candles[:-1]


# ============================================================
# EMA
# ============================================================

def calculate_ema(values, period):

    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1)

    ema = sum(values[:period]) / period

    for price in values[period:]:
        ema = (
            (price - ema) * multiplier
        ) + ema

    return ema


# ============================================================
# RSI
# ============================================================

def calculate_rsi(closes, period=14):

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(closes)):

        change = closes[i] - closes[i - 1]

        gain = max(change, 0)
        loss = max(-change, 0)

        avg_gain = (
            (avg_gain * (period - 1)) + gain
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + loss
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ============================================================
# WILLIAMS %R
# ============================================================

def calculate_williams_r(candles, period=14):

    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(
        c["high"] for c in recent
    )

    lowest = min(
        c["low"] for c in recent
    )

    close = recent[-1]["close"]

    if highest == lowest:
        return -50.0

    return (
        (highest - close)
        / (highest - lowest)
    ) * -100


# ============================================================
# ATR
# ============================================================

def calculate_atr(candles, period=10):

    if len(candles) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(candles)):

        current = candles[i]
        previous = candles[i - 1]

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

        true_ranges.append(tr)

    if len(true_ranges) < period:
        return None

    return (
        sum(true_ranges[-period:])
        / period
    )


# ============================================================
# ADX / DI
# ============================================================

def calculate_adx_di(candles, period=14):

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

        plus = (
            up_move
            if up_move > down_move
            and up_move > 0
            else 0
        )

        minus = (
            down_move
            if down_move > up_move
            and down_move > 0
            else 0
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

    if len(trs) < period:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    atr = sum(trs[-period:]) / period

    if atr == 0:
        return {
            "adx": 0,
            "plus_di": 0,
            "minus_di": 0,
        }

    plus = (
        sum(plus_dm[-period:])
        / period
    )

    minus = (
        sum(minus_dm[-period:])
        / period
    )

    plus_di = 100 * plus / atr
    minus_di = 100 * minus / atr

    dx_denominator = (
        plus_di + minus_di
    )

    if dx_denominator == 0:
        adx = 0
    else:
        adx = (
            abs(plus_di - minus_di)
            / dx_denominator
        ) * 100

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# KELTNER
# ============================================================

def calculate_keltner(candles):

    if len(candles) < 20:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    ema20 = calculate_ema(
        closes,
        20
    )

    atr10 = calculate_atr(
        candles,
        10
    )

    if ema20 is None or atr10 is None:
        return None

    multiplier = 5.0

    return {
        "middle": ema20,
        "upper": ema20 + (
            atr10 * multiplier
        ),
        "lower": ema20 - (
            atr10 * multiplier
        ),
    }


# ============================================================
# MARKET STRUCTURE
# ============================================================

def analyze_structure(candles):

    if len(candles) < 10:
        return {
            "trend": "UNKNOWN",
            "higher_high": False,
            "higher_low": False,
            "lower_high": False,
            "lower_low": False,
        }

    recent = candles[-10:]

    highs = [
        c["high"]
        for c in recent
    ]

    lows = [
        c["low"]
        for c in recent
    ]

    first_half_high = max(
        highs[:5]
    )

    second_half_high = max(
        highs[5:]
    )

    first_half_low = min(
        lows[:5]
    )

    second_half_low = min(
        lows[5:]
    )

    higher_high = (
        second_half_high
        > first_half_high
    )

    higher_low = (
        second_half_low
        > first_half_low
    )

    lower_high = (
        second_half_high
        < first_half_high
    )

    lower_low = (
        second_half_low
        < first_half_low
    )

    if higher_high and higher_low:
        trend = "UP"

    elif lower_high and lower_low:
        trend = "DOWN"

    else:
        trend = "RANGE"

    return {
        "trend": trend,
        "higher_high": higher_high,
        "higher_low": higher_low,
        "lower_high": lower_high,
        "lower_low": lower_low,
    }


# ============================================================
# BREAKOUT
# ============================================================

def analyze_breakout(candles):

    if len(candles) < 12:
        return {
            "up": False,
            "down": False,
        }

    previous = candles[-11:-1]
    last = candles[-1]

    resistance = max(
        c["high"]
        for c in previous
    )

    support = min(
        c["low"]
        for c in previous
    )

    return {
        "up": last["close"] > resistance,
        "down": last["close"] < support,
    }


# ============================================================
# CANCELLATION LEVEL
# ============================================================

def get_cancellation_level(
    candles,
    direction
):

    if len(candles) < 5:
        return None

    recent = candles[-8:]

    if direction == "UP":

        return min(
            c["low"]
            for c in recent
        )

    return max(
        c["high"]
        for c in recent
    )


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_technical_snapshot(candles):

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

    adx = calculate_adx_di(
        candles,
        14
    )

    keltner = calculate_keltner(
        candles
    )

    structure = analyze_structure(
        candles
    )

    breakout = analyze_breakout(
        candles
    )

    last_close = (
        candles[-1]["close"]
        if candles
        else 0
    )

    return {
        "last_close": last_close,
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi,
        "williams_r14": williams,
        "atr10": atr,
        "adx14": adx,
        "keltner": keltner,
        "structure": structure,
        "breakout": breakout,
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles,
    snapshot
):

    candle_text = json.dumps(
        candles[-60:],
        ensure_ascii=False
    )

    snapshot_text = json.dumps(
        snapshot,
        ensure_ascii=False
    )

    return f"""
You are the technical-analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied MT4 closed-candle data.

Symbol:
{symbol}

Timeframe:
{timeframe}

Technical snapshot:
{snapshot_text}

Recent closed candles:
{candle_text}

============================================================
PRIMARY OBJECTIVE
============================================================

Quality is more important than frequency.

DO NOT force a signal.

If the setup is weak, contradictory, unclear, exhausted,
or does not have enough confluence, return:

"signal": false

The bot must NOT send a Telegram signal in that case.

A signal is allowed only when there is strong evidence for
one direction.

Do not create an UP signal simply because the market has
recently moved down.

Do not create a DOWN signal simply because the market has
recently moved up.

============================================================
ANALYSIS PRIORITY
============================================================

Use this priority:

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

Do not invent indicators.

============================================================
SCORING
============================================================

Total score must be exactly 18.

Structure: 2
Breakout: 2
Liquidity: 1
Momentum: 2
Candle: 2
RSI: 1
Summary: 2
Oscillators: 3
Moving Averages: 3

UP score + DOWN score does not need to equal 18.

However, each selected direction must have a coherent
score based on actual evidence.

============================================================
SIGNAL REQUIREMENTS
============================================================

A signal should normally require:

- selected score >= 11/18
- difference between selected direction and opposite >= 5
- confidence >= 70
- at least 4 independent confirmations
- fewer than 2 major contradictions
- no obvious exhaustion
- enough closed candles
- clear directional structure

Do NOT use 90%+ confidence unless the confluence is
exceptionally strong.

============================================================
IMPORTANT
============================================================

The bot prefers NO SIGNAL over a weak signal.

Do not use WAIT, NEUTRAL, or ambiguous direction.

Instead use:

signal = false

when the setup is not strong enough.

============================================================
JSON ONLY
============================================================

Return ONLY valid JSON.

Required structure:

{{
  "signal": true,
  "direction": "UP",
  "confidence": 82,
  "up_score": 15,
  "down_score": 4,
  "structure_score": 2,
  "breakout_score": 2,
  "liquidity_score": 1,
  "momentum_score": 2,
  "candle_score": 2,
  "rsi_score": 1,
  "summary_score": 2,
  "oscillators_score": 2,
  "moving_averages_score": 1,
  "reason": "Short evidence-based explanation",
  "cancellation_reason": "Invalidation below/above recent structure"
}}

If the setup is weak:

{{
  "signal": false,
  "direction": "UP",
  "confidence": 58,
  "up_score": 8,
  "down_score": 7,
  "structure_score": 1,
  "breakout_score": 0,
  "liquidity_score": 1,
  "momentum_score": 1,
  "candle_score": 1,
  "rsi_score": 1,
  "summary_score": 1,
  "oscillators_score": 1,
  "moving_averages_score": 1,
  "reason": "Weak and conflicting setup",
  "cancellation_reason": ""
}}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles
):

    snapshot = build_technical_snapshot(
        candles
    )

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        snapshot
    )

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

        text = (
            response.text
            if response
            else ""
        )

        if not text:
            return None

        text = text.strip()

        # حماية إضافية إذا رجع Gemini markdown
        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?",
                "",
                text,
                flags=re.IGNORECASE
            )

            text = re.sub(
                r"```$",
                "",
                text
            )

            text = text.strip()

        data = json.loads(text)

        if not isinstance(data, dict):
            return None

        return data

    except Exception as exc:

        logger.exception(
            "Gemini analysis error: %s",
            exc
        )

        return None


# ============================================================
# SIGNAL QUALITY
# ============================================================

def evaluate_signal_quality(
    analysis,
    candles
):

    if not isinstance(analysis, dict):
        return False

    if analysis.get("signal") is not True:
        return False

    direction = str(
        analysis.get(
            "direction",
            ""
        )
    ).upper()

    if direction not in (
        "UP",
        "DOWN"
    ):
        return False

    confidence = safe_float(
        analysis.get(
            "confidence",
            0
        )
    )

    up_score = safe_int(
        analysis.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        analysis.get(
            "down_score",
            0
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

    score_difference = (
        selected_score
        - opposite_score
    )

    confirmations = 0

    for key in (
        "structure_score",
        "breakout_score",
        "liquidity_score",
        "momentum_score",
        "candle_score",
        "rsi_score",
        "summary_score",
        "oscillators_score",
        "moving_averages_score",
    ):

        value = safe_int(
            analysis.get(key, 0)
        )

        if value > 0:
            confirmations += 1

    contradictions = safe_int(
        analysis.get(
            "contradictions",
            0
        )
    )

    if len(candles) < 40:
        return False

    if selected_score < 11:
        return False

    if score_difference < 5:
        return False

    if confidence < 70:
        return False

    if confirmations < 4:
        return False

    if contradictions >= 2:
        return False

    # حماية من الإشارات غير المنطقية
    reason = str(
        analysis.get(
            "reason",
            ""
        )
    ).strip()

    if len(reason) < 10:
        return False

    return True


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(
    symbol,
    timeframe,
    analysis,
    candles
):

    direction = str(
        analysis.get(
            "direction",
            "UP"
        )
    ).upper()

    confidence = safe_float(
        analysis.get(
            "confidence",
            0
        )
    )

    up_score = safe_int(
        analysis.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        analysis.get(
            "down_score",
            0
        )
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    entry_price = safe_float(
        candles[-1]["close"]
    )

    cancellation = get_cancellation_level(
        candles,
        direction
    )

    if cancellation is None:
        cancellation = entry_price

    reason = str(
        analysis.get(
            "reason",
            ""
        )
    ).strip()

    if direction == "UP":
        decision = "🟢 UP"
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{cancellation}"
        )
    else:
        decision = "🔴 DOWN"
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{cancellation}"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{decision}\n"
        f"🎯 Confidence: {confidence:.0f}%\n"
        f"📈 UP Score: {up_score}/18\n"
        f"📉 DOWN Score: {down_score}/18\n\n"
        f"⏰ Entry Time: "
        f"{format_algiers(entry_time)}\n"
        f"💰 Entry Price: {entry_price}\n"
        f"⚠️ {cancel_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# SEND TELEGRAM
# ============================================================

async def send_owner_message(
    application,
    message
):

    try:

        await application.bot.send_message(
            chat_id=OWNER_ID_INT,
            text=message
        )

        return True

    except Exception as exc:

        logger.exception(
            "Telegram send error: %s",
            exc
        )

        return False


# ============================================================
# NEW CANDLE DETECTION
# ============================================================

def detect_new_closed_candle(
    symbol,
    timeframe,
    candles
):
    """
    IMPORTANT FIX

    MT4 sends the newest candle as the currently forming candle.

    Therefore the candle that should trigger analysis is:

        candles[-2]

    and NOT:

        candles[-1]

    We remember the timestamp of candles[-2].
    """

    closed = get_closed_candles(
        candles
    )

    if not closed:
        return False, None

    latest_closed = closed[-1]

    candle_time = latest_closed.get(
        "time"
    )

    if not candle_time:
        return False, None

    key = (
        symbol.upper(),
        timeframe.upper()
    )

    previous = last_processed_closed_candle.get(
        key
    )

    if previous is None:

        last_processed_closed_candle[
            key
        ] = candle_time

        return False, latest_closed

    if candle_time > previous:

        last_processed_closed_candle[
            key
        ] = candle_time

        return True, latest_closed

    return False, latest_closed


# ============================================================
# AUTO ANALYSIS
# ============================================================

async def auto_analyze_pair(
    application,
    symbol,
    timeframe,
    candles
):

    key = (
        symbol.upper(),
        timeframe.upper()
    )

    current_time = time.time()

    last_time = last_auto_analysis.get(
        key,
        0
    )

    # لا نحلل نفس الزوج/الفريم أكثر من مرة
    # خلال الفترة المحددة
    if (
        current_time - last_time
        < AUTO_ANALYSIS_INTERVAL_MINUTES * 60
    ):
        return

    closed = get_closed_candles(
        candles
    )

    if len(closed) < 40:
        logger.info(
            "Not enough closed candles for %s %s: %s",
            symbol,
            timeframe,
            len(closed)
        )
        return

    last_auto_analysis[key] = current_time

    logger.info(
        "Starting automatic analysis: %s %s",
        symbol,
        timeframe
    )

    analysis = await asyncio.to_thread(
        analyze_with_gemini,
        symbol,
        timeframe,
        closed
    )

    if not analysis:

        logger.warning(
            "Gemini returned no analysis for %s %s",
            symbol,
            timeframe
        )

        return

    direction = str(
        analysis.get(
            "direction",
            ""
        )
    ).upper()

    confidence = safe_float(
        analysis.get(
            "confidence",
            0
        )
    )

    up_score = safe_int(
        analysis.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        analysis.get(
            "down_score",
            0
        )
    )

    signal = (
        analysis.get("signal")
        is True
    )

    logger.info(
        "Analysis %s %s | signal=%s | direction=%s | confidence=%s | UP=%s | DOWN=%s",
        symbol,
        timeframe,
        signal,
        direction,
        confidence,
        up_score,
        down_score
    )

    # ========================================================
    # WEAK SETUP = NO TELEGRAM MESSAGE
    # ========================================================

    if not evaluate_signal_quality(
        analysis,
        closed
    ):

        logger.info(
            "Weak setup rejected: %s %s",
            symbol,
            timeframe
        )

        return

    # ========================================================
    # ENTRY TIME
    # ========================================================

    entry_time = get_next_entry_time(
        timeframe
    )

    seconds_until_entry = (
        entry_time
        - now_algiers()
    ).total_seconds()

    if seconds_until_entry < MIN_ENTRY_LEAD_SECONDS:

        logger.info(
            "Signal rejected because entry lead is too short."
        )

        return

    message = format_signal(
        symbol,
        timeframe,
        analysis,
        closed
    )

    sent = await send_owner_message(
        application,
        message
    )

    if sent:

        logger.info(
            "SIGNAL SENT: %s %s %s %s%%",
            symbol,
            timeframe,
            direction,
            confidence
        )


# ============================================================
# MT4 HTTP SERVER
# ============================================================

class MT4Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format,
        *args
    ):
        return

    def _send_json(
        self,
        status,
        data
    ):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)

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
            "/health"
        ):

            self._send_json(
                200,
                {
                    "ok": True,
                    "service":
                        "ZinoProSignalAI",
                    "status":
                        "running",
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

        if path != "/mt4":

            self._send_json(
                404,
                {
                    "ok": False,
                    "error": "Not found"
                }
            )

            return

        header_key = (
            self.headers.get(
                "X-MT4-API-Key",
                ""
            ).strip()
        )

        content_length = safe_int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        try:

            raw_body = self.rfile.read(
                content_length
            )

            body = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:

            logger.warning(
                "Invalid MT4 JSON: %s",
                exc
            )

            self._send_json(
                400,
                {
                    "ok": False,
                    "error":
                        "Invalid JSON"
                }
            )

            return

        body_key = str(
            body.get(
                "api_key",
                ""
            )
        ).strip()

        # Accept the API key from either
        # header or JSON body.
        supplied_key = (
            header_key
            or body_key
        )

        if supplied_key != MT4_API_KEY:

            logger.warning(
                "MT4 unauthorized request"
            )

            self._send_json(
                401,
                {
                    "ok": False,
                    "error":
                        "Unauthorized"
                }
            )

            return

        symbol = str(
            body.get(
                "symbol",
                ""
            )
        ).strip()

        timeframe = str(
            body.get(
                "timeframe",
                ""
            )
        ).strip()

        candles = normalize_candles(
            body.get(
                "candles",
                []
            )
        )

        if not symbol or not timeframe:

            self._send_json(
                400,
                {
                    "ok": False,
                    "error":
                        "Missing symbol/timeframe"
                }
            )

            return

        if len(candles) < 3:

            self._send_json(
                400,
                {
                    "ok": False,
                    "error":
                        "Not enough candles"
                }
            )

            return

        # Store MT4 data
        key = (
            symbol.upper(),
            timeframe.upper()
        )

        with mt4_lock:

            mt4_data[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "received_at":
                    time.time(),
            }

        # ====================================================
        # FIXED NEW CANDLE DETECTION
        # ====================================================

        is_new_candle, latest_closed = (
            detect_new_closed_candle(
                symbol,
                timeframe,
                candles
            )
        )

        logger.info(
            "MT4 received: %s %s | candles=%s | new_closed_candle=%s | closed_time=%s",
            symbol,
            timeframe,
            len(candles),
            is_new_candle,
            (
                latest_closed.get("time")
                if latest_closed
                else None
            )
        )

        # Respond immediately to MT4
        self._send_json(
            200,
            {
                "ok": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
                "new_candle":
                    is_new_candle,
            }
        )

        # ====================================================
        # AUTO ANALYSIS
        # ====================================================

        if is_new_candle:

            application = telegram_application

            if application is not None:

                try:

                    loop = asyncio.get_running_loop()

                    asyncio.create_task(
                        auto_analyze_pair(
                            application,
                            symbol,
                            timeframe,
                            candles
                        )
                    )

                except RuntimeError:

                    logger.warning(
                        "No running asyncio loop"
                    )


# ============================================================
# HTTP SERVER
# ============================================================

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
# TELEGRAM
# ============================================================

telegram_application = None


async def owner_only(
    update: Update
):

    if not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID_INT
    )


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "✅ البوت يعمل\n"
        "📡 MT4 → Render → Gemini → Telegram"
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n\n"
        f"🟢 WIN: {stats_data['wins']}\n"
        f"🔴 LOSS: {stats_data['losses']}"
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    stats_data["wins"] += 1

    await update.message.reply_text(
        f"🟢 WIN +1\n"
        f"Total WIN: {stats_data['wins']}"
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    stats_data["losses"] += 1

    await update.message.reply_text(
        f"🔴 LOSS +1\n"
        f"Total LOSS: {stats_data['losses']}"
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    stats_data["wins"] = 0
    stats_data["losses"] = 0

    await update.message.reply_text(
        "♻️ Stats reset."
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with mt4_lock:

        if not mt4_data:

            await update.message.reply_text(
                "❌ لا توجد بيانات MT4."
            )

            return

        lines = [
            "📡 MT4 STATUS",
            "━━━━━━━━━━━━━━━━━━",
        ]

        for key, data in mt4_data.items():

            symbol, timeframe = key

            received = data.get(
                "received_at",
                0
            )

            age = (
                time.time()
                - received
            )

            lines.append(
                f"📊 {symbol} | {timeframe}"
            )

            lines.append(
                f"🕐 آخر استقبال منذ {age:.0f}s"
            )

            lines.append(
                f"🕯️ شموع: "
                f"{len(data.get('candles', []))}"
            )

        await update.message.reply_text(
            "\n".join(lines)
        )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    with mt4_lock:

        if not mt4_data:

            await update.message.reply_text(
                "❌ لا توجد بيانات MT4."
            )

            return

        items = list(
            mt4_data.values()
        )

    sent_any = False

    for data in items:

        symbol = data["symbol"]
        timeframe = data["timeframe"]
        candles = data["candles"]

        closed = get_closed_candles(
            candles
        )

        if len(closed) < 40:
            continue

        analysis = await asyncio.to_thread(
            analyze_with_gemini,
            symbol,
            timeframe,
            closed
        )

        if not evaluate_signal_quality(
            analysis,
            closed
        ):
            continue

        message = format_signal(
            symbol,
            timeframe,
            analysis,
            closed
        )

        await send_owner_message(
            telegram_application,
            message
        )

        sent_any = True

    if not sent_any:

        await update.message.reply_text(
            "🔎 تم التحليل.\n"
            "❌ لا يوجد Setup قوي حاليًا."
        )


# ============================================================
# TEXT ANALYSIS
# ============================================================

async def text_message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:
        return

    # لا نحلل أي نص عشوائي
    # إلا إذا كان فيه زوج/فريم واضح
    pair_match = re.search(
        r"\b([A-Z]{3,6})\b",
        text.upper()
    )

    if not pair_match:
        return

    await update.message.reply_text(
        "📡 التحليل النصي يحتاج بيانات MT4 الحالية.\n"
        "استخدم /analyze لتحليل آخر بيانات مستلمة."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    try:

        photo = update.message.photo[-1]

        file = await context.bot.get_file(
            photo.file_id
        )

        buffer = io.BytesIO()

        await file.download_to_memory(
            buffer
        )

        image_bytes = buffer.getvalue()

        if not image_bytes:
            await update.message.reply_text(
                "❌ لم أستطع قراءة الصورة."
            )
            return

        prompt = """
Analyze this trading chart image.

Return a concise technical analysis.

Focus on:

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

Do not invent values that cannot be seen.

If the setup is weak, clearly say:
Setup ضعيف.

If the setup is strong, identify UP or DOWN
and explain why.

Do not guarantee a winning trade.
"""

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt,
            ]
        )

        text = (
            response.text
            if response
            else
            "لم يتم الحصول على تحليل."
        )

        await update.message.reply_text(
            "🎓 ZinoProSignalAI\n"
            "━━━━━━━━━━━━━━━━━━\n"
            + text
        )

    except Exception as exc:

        logger.exception(
            "Photo analysis error: %s",
            exc
        )

        await update.message.reply_text(
            "❌ حدث خطأ أثناء تحليل الصورة."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_application

    logger.info(
        "Starting ZinoProSignalAI"
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

    # HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    # Telegram
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_application = application

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
            text_message_handler
        )
    )

    logger.info(
        "Telegram bot starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
````
