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

# الحد الأدنى بين تحليلات تلقائية لنفس الزوج والفريم
AUTO_ANALYSIS_INTERVAL_MINUTES = 3

# الحد الأدنى للوقت المتبقي قبل الدخول
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
# GLOBAL DATA
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

last_processed_closed_candle = {}

last_auto_analysis = {}

stats_data = {
    "wins": 0,
    "losses": 0,
}

telegram_application = None
telegram_loop = None


# ============================================================
# TIME HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def format_algiers(dt):
    if dt is None:
        return "--:--:--"

    return dt.astimezone(ALGIERS).strftime(
        "%H:%M:%S"
    )


def parse_mt4_time(value):
    try:
        if value is None:
            return None

        value = int(value)

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

    match = re.search(
        r"(\d+)",
        text
    )

    if not match:
        return 1

    number = int(
        match.group(1)
    )

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

def get_next_entry_time(timeframe):

    minutes = timeframe_to_minutes(
        timeframe
    )

    if minutes < 1:
        minutes = 1

    return now_algiers() + timedelta(
        minutes=minutes
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
        "time": safe_int(
            candle.get("time")
        ),
        "open": safe_float(
            candle.get("open")
        ),
        "high": safe_float(
            candle.get("high")
        ),
        "low": safe_float(
            candle.get("low")
        ),
        "close": safe_float(
            candle.get("close")
        ),
        "volume": safe_int(
            candle.get("volume")
        ),
    }


def normalize_candles(candles):

    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:

        normalized = normalize_candle(
            candle
        )

        if normalized is not None:
            result.append(
                normalized
            )

    result.sort(
        key=lambda x: x["time"]
    )

    return result


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles):

    candles = normalize_candles(
        candles
    )

    if len(candles) < 3:
        return []

    # MT4 يرسل آخر شمعة باعتبارها الحالية.
    # لذلك نستبعد الأخيرة لأنها غير مغلقة.
    return candles[:-1]


# ============================================================
# EMA
# ============================================================

def calculate_ema(
    values,
    period
):

    if len(values) < period:
        return None

    multiplier = 2.0 / (
        period + 1
    )

    ema = (
        sum(values[:period])
        / period
    )

    for price in values[period:]:

        ema = (
            (price - ema)
            * multiplier
        ) + ema

    return ema


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    closes,
    period=14
):

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(
        1,
        period + 1
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(
                abs(change)
            )

    avg_gain = (
        sum(gains) / period
    )

    avg_loss = (
        sum(losses) / period
    )

    for i in range(
        period + 1,
        len(closes)
    ):

        change = (
            closes[i]
            - closes[i - 1]
        )

        gain = max(
            change,
            0
        )

        loss = max(
            -change,
            0
        )

        avg_gain = (
            (
                avg_gain
                * (period - 1)
            )
            + gain
        ) / period

        avg_loss = (
            (
                avg_loss
                * (period - 1)
            )
            + loss
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = (
        avg_gain
        / avg_loss
    )

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
        (
            highest - close
        )
        / (
            highest - lowest
        )
    ) * -100


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles,
    period=10
):

    if len(candles) < period + 1:
        return None

    true_ranges = []

    for i in range(
        1,
        len(candles)
    ):

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
        )
        / period
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

    for i in range(
        1,
        len(candles)
    ):

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
            else 0
        )

        minus = (
            down_move
            if (
                down_move > up_move
                and down_move > 0
            )
            else 0
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
            "adx": 0,
            "plus_di": 0,
            "minus_di": 0,
        }

    plus = (
        sum(
            plus_dm[-period:]
        )
        / period
    )

    minus = (
        sum(
            minus_dm[-period:]
        )
        / period
    )

    plus_di = (
        100 * plus / atr
    )

    minus_di = (
        100 * minus / atr
    )

    denominator = (
        plus_di
        + minus_di
    )

    if denominator == 0:

        adx = 0

    else:

        adx = (
            abs(
                plus_di
                - minus_di
            )
            / denominator
        ) * 100

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# KELTNER
# ============================================================

def calculate_keltner(
    candles
):

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

    if (
        ema20 is None
        or atr10 is None
    ):
        return None

    multiplier = 5.0

    return {
        "middle": ema20,
        "upper": (
            ema20
            + atr10 * multiplier
        ),
        "lower": (
            ema20
            - atr10 * multiplier
        ),
    }


# ============================================================
# MARKET STRUCTURE
# ============================================================

def analyze_structure(
    candles
):

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

    if (
        higher_high
        and higher_low
    ):
        trend = "UP"

    elif (
        lower_high
        and lower_low
    ):
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

def analyze_breakout(
    candles
):

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
        "up": (
            last["close"]
            > resistance
        ),
        "down": (
            last["close"]
            < support
        ),
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

def build_technical_snapshot(
    candles
):

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

Analyze ONLY the supplied MT4 CLOSED candles.

Symbol:
{symbol}

Timeframe:
{timeframe}

Technical snapshot:
{snapshot_text}

Recent closed candles:
{candle_text}

============================================================
CORE RULE
============================================================

Accuracy and setup quality are more important than frequency.

DO NOT force a trade.

If the setup is weak, mixed, ranging, exhausted,
or contradictory:

"signal": false

Never create a signal merely because the user wants one.

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

Price Action and Structure have more importance than
late indicator signals.

Do not invent data.

Do not claim that an indicator confirms a direction
if the supplied values do not support it.

============================================================
UP / DOWN SCORING
============================================================

Each direction has a maximum score of 18.

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

Calculate UP and DOWN independently.

============================================================
SIGNAL FILTER
============================================================

A valid signal normally requires ALL of these:

- signal = true
- direction = UP or DOWN
- selected score >= 11/18
- score difference >= 5
- confidence >= 70
- at least 4 independent confirmations
- contradictions < 2
- at least 40 closed candles
- clear directional structure
- no obvious exhaustion

If these conditions are not satisfied:

signal = false

============================================================
CONFIDENCE
============================================================

Do NOT use 90% or higher unless the setup is exceptionally
strong and several independent factors align.

Avoid artificially high confidence.

============================================================
IMPORTANT REVERSAL RULE
============================================================

Do not assume:

recent green candles = DOWN

or:

recent red candles = UP.

A reversal requires actual evidence such as:

- structure shift
- rejection
- failed breakout
- liquidity sweep
- momentum change
- candle confirmation

============================================================
RANGE RULE
============================================================

If structure is RANGE and there is no clear breakout,
retest, or strong rejection:

signal = false

============================================================
JSON ONLY
============================================================

Return ONLY valid JSON.

Use exactly these fields:

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
  "contradictions": 0,
  "reason": "Short evidence-based explanation"
}}

For a weak setup:

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
  "contradictions": 2,
  "reason": "Weak and conflicting setup"
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

        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.10,
                    response_mime_type="application/json",
                ),
            )
        )

        text = (
            response.text
            if response
            else ""
        )

        if not text:
            logger.warning(
                "Gemini returned empty response."
            )
            return None

        text = text.strip()

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

        data = json.loads(
            text
        )

        if not isinstance(
            data,
            dict
        ):
            return None

        return data

    except Exception as exc:

        logger.exception(
            "Gemini analysis error: %s",
            exc
        )

        return None


# ============================================================
# SIGNAL REJECTION REASONS
# ============================================================

def get_signal_rejection_reasons(
    analysis,
    candles
):

    reasons = []

    if not isinstance(
        analysis,
        dict
    ):
        return [
            "analysis_invalid"
        ]

    signal = (
        analysis.get("signal")
        is True
    )

    direction = str(
        analysis.get(
            "direction",
            ""
        )
    ).upper().strip()

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
        if direction == "DOWN"
        else 0
    )

    opposite_score = (
        down_score
        if direction == "UP"
        else up_score
        if direction == "DOWN"
        else 0
    )

    score_difference = (
        selected_score
        - opposite_score
    )

    contradictions = safe_int(
        analysis.get(
            "contradictions",
            0
        )
    )

    confirmation_keys = (
        "structure_score",
        "breakout_score",
        "liquidity_score",
        "momentum_score",
        "candle_score",
        "rsi_score",
        "summary_score",
        "oscillators_score",
        "moving_averages_score",
    )

    confirmations = 0

    for key in confirmation_keys:

        if safe_int(
            analysis.get(key, 0)
        ) > 0:

            confirmations += 1

    reason = str(
        analysis.get(
            "reason",
            ""
        )
    ).strip()

    if not signal:
        reasons.append(
            f"signal={analysis.get('signal')}"
        )

    if direction not in (
        "UP",
        "DOWN"
    ):
        reasons.append(
            f"direction={direction or 'EMPTY'}"
        )

    if selected_score < 11:
        reasons.append(
            f"score={selected_score}/18"
        )

    if score_difference < 5:
        reasons.append(
            f"difference={score_difference}"
        )

    if confidence < 70:
        reasons.append(
            f"confidence={confidence:.0f}%"
        )

    if confirmations < 4:
        reasons.append(
            f"confirmations={confirmations}"
        )

    if contradictions >= 2:
        reasons.append(
            f"contradictions={contradictions}"
        )

    if len(candles) < 40:
        reasons.append(
            f"closed_candles={len(candles)}"
        )

    if len(reason) < 10:
        reasons.append(
            "reason_too_short"
        )

    return reasons


# ============================================================
# SIGNAL QUALITY
# ============================================================

def evaluate_signal_quality(
    analysis,
    candles
):

    reasons = get_signal_rejection_reasons(
        analysis,
        candles
    )

    return len(reasons) == 0


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

    cancellation = (
        get_cancellation_level(
            candles,
            direction
        )
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
            "إلغاء إذا أغلقت الشمعة تحت "
            f"{cancellation}"
        )

    else:

        decision = "🔴 DOWN"

        cancel_text = (
            "إلغاء إذا أغلقت الشمعة فوق "
            f"{cancellation}"
        )

    timeframe_minutes = (
        timeframe_to_minutes(
            timeframe
        )
    )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{decision}\n"
        f"🎯 Confidence: {confidence:.0f}%\n"
        f"📈 UP Score: {up_score}/18\n"
        f"📉 DOWN Score: {down_score}/18\n\n"
        f"⏳ Entry after: "
        f"{timeframe_minutes} min\n"
        f"⏰ Entry Time: "
        f"{format_algiers(entry_time)}\n"
        f"💰 Entry Price: "
        f"{entry_price}\n"
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

    if application is None:
        return False

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
# NEW CLOSED CANDLE DETECTION
# ============================================================

def detect_new_closed_candle(
    symbol,
    timeframe,
    candles
):

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

    previous = (
        last_processed_closed_candle
        .get(key)
    )

    # أول استقبال:
    # نحفظ آخر شمعة كـ baseline
    if previous is None:

        last_processed_closed_candle[
            key
        ] = candle_time

        logger.info(
            "Initialized closed candle baseline: "
            "%s %s | %s",
            symbol,
            timeframe,
            candle_time
        )

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

    last_time = (
        last_auto_analysis.get(
            key,
            0
        )
    )

    if (
        current_time - last_time
        <
        AUTO_ANALYSIS_INTERVAL_MINUTES * 60
    ):

        logger.info(
            "Auto analysis throttled: "
            "%s %s",
            symbol,
            timeframe
        )

        return

    closed = get_closed_candles(
        candles
    )

    if len(closed) < 40:

        logger.info(
            "Not enough closed candles: "
            "%s %s | %s",
            symbol,
            timeframe,
            len(closed)
        )

        return

    last_auto_analysis[key] = (
        current_time
    )

    logger.info(
        "STARTING AUTOMATIC ANALYSIS: "
        "%s %s",
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
            "No Gemini analysis: "
            "%s %s",
            symbol,
            timeframe
        )

        return

    try:

        logger.info(
            "AUTO ANALYSIS RAW | "
            "%s %s | %s",
            symbol,
            timeframe,
            json.dumps(
                analysis,
                ensure_ascii=False
            )
        )

    except Exception:
        pass

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

    logger.info(
        "Analysis %s %s | "
        "signal=%s | direction=%s | "
        "confidence=%s | UP=%s | DOWN=%s",
        symbol,
        timeframe,
        analysis.get("signal"),
        direction,
        confidence,
        up_score,
        down_score
    )

    if not evaluate_signal_quality(
        analysis,
        closed
    ):

        reasons = (
            get_signal_rejection_reasons(
                analysis,
                closed
            )
        )

        logger.warning(
            "WEAK SETUP REJECTED | "
            "%s %s | %s",
            symbol,
            timeframe,
            " | ".join(reasons)
        )

        return

    entry_time = (
        get_next_entry_time(
            timeframe
        )
    )

    seconds_until_entry = (
        entry_time
        - now_algiers()
    ).total_seconds()

    if (
        seconds_until_entry
        < MIN_ENTRY_LEAD_SECONDS
    ):

        logger.warning(
            "Signal rejected: "
            "entry lead too short | "
            "%s %s | %.1fs",
            symbol,
            timeframe,
            seconds_until_entry
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
            "SIGNAL SENT | %s %s | "
            "%s | %.0f%%",
            symbol,
            timeframe,
            direction,
            confidence
        )


# ============================================================
# MT4 HTTP SERVER
# ============================================================

class MT4Handler(
    BaseHTTPRequestHandler
):

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

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; "
            "charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(
            body
        )

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

            raw_body = (
                self.rfile.read(
                    content_length
                )
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

        supplied_key = (
            header_key
            or body_key
        )

        if (
            supplied_key
            != MT4_API_KEY
        ):

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

        if (
            not symbol
            or not timeframe
        ):

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

        is_new_candle, latest_closed = (
            detect_new_closed_candle(
                symbol,
                timeframe,
                candles
            )
        )

        closed_time = (
            latest_closed.get(
                "time"
            )
            if latest_closed
            else None
        )

        logger.info(
            "MT4 data received: "
            "%s | %s | candles=%s | "
            "new_candle=%s | "
            "closed_time=%s",
            symbol,
            timeframe,
            len(candles),
            is_new_candle,
            closed_time
        )

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

        if not is_new_candle:
            return

        application = (
            telegram_application
        )

        if application is None:
            logger.warning(
                "Telegram application "
                "is not ready."
            )
            return

        if telegram_loop is None:
            logger.warning(
                "Telegram loop "
                "is not ready."
            )
            return

        try:

            asyncio.run_coroutine_threadsafe(
                auto_analyze_pair(
                    application,
                    symbol,
                    timeframe,
                    candles
                ),
                telegram_loop
            )

            logger.info(
                "Automatic analysis "
                "scheduled: %s %s",
                symbol,
                timeframe
            )

        except Exception as exc:

            logger.exception(
                "Failed to schedule "
                "automatic analysis: %s",
                exc
            )


# ============================================================
# HTTP SERVER
# ============================================================

def start_http_server():

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        MT4Handler
    )

    logger.info(
        "HTTP server listening "
        "on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# OWNER CHECK
# ============================================================

async def owner_only(
    update: Update
):

    if not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID_INT
    )


# ============================================================
# START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ البوت يعمل\n"
        "📡 MT4 → Render → Gemini → Telegram\n\n"
        "الأوامر:\n"
        "/analyze\n"
        "/mt4status\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


# ============================================================
# STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not await owner_only(update):
        return

    wins = stats_data["wins"]
    losses = stats_data["losses"]
    total = wins + losses

    if total > 0:
        winrate = (
            wins / total
        ) * 100
    else:
        winrate = 0

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 WIN: {wins}\n"
        f"🔴 LOSS: {losses}\n"
        f"📌 TOTAL: {total}\n"
        f"📈 WIN RATE: {winrate:.1f}%"
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
        f"Total WIN: "
        f"{stats_data['wins']}"
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
        f"Total LOSS: "
        f"{stats_data['losses']}"
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


# ============================================================
# MT4 STATUS
# ============================================================

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

            candles_count = len(
                data.get(
                    "candles",
                    []
                )
            )

            lines.append(
                f"📊 {symbol} | "
                f"{timeframe}"
            )

            lines.append(
                f"🕐 آخر استقبال: "
                f"{age:.0f}s"
            )

            lines.append(
                f"🕯️ Candles: "
                f"{candles_count}"
            )

        await update.message.reply_text(
            "\n".join(lines)
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
    analyzed_count = 0
    rejected_count = 0

    for data in items:

        symbol = data.get(
            "symbol",
            "UNKNOWN"
        )

        timeframe = data.get(
            "timeframe",
            "M1"
        )

        candles = data.get(
            "candles",
            []
        )

        closed = get_closed_candles(
            candles
        )

        logger.info(
            "MANUAL ANALYZE START | "
            "%s %s | total=%s | closed=%s",
            symbol,
            timeframe,
            len(candles),
            len(closed)
        )

        if len(closed) < 40:

            logger.warning(
                "MANUAL REJECT | "
                "%s %s | "
                "closed candles=%s",
                symbol,
                timeframe,
                len(closed)
            )

            continue

        analyzed_count += 1

        analysis = await asyncio.to_thread(
            analyze_with_gemini,
            symbol,
            timeframe,
            closed
        )

        if analysis is None:

            logger.warning(
                "MANUAL ANALYSIS FAILED | "
                "%s %s",
                symbol,
                timeframe
            )

            rejected_count += 1

            continue

        try:

            logger.info(
                "MANUAL ANALYSIS RAW | "
                "%s %s | %s",
                symbol,
                timeframe,
                json.dumps(
                    analysis,
                    ensure_ascii=False
                )
            )

        except Exception:
            pass

        reasons = (
            get_signal_rejection_reasons(
                analysis,
                closed
            )
        )

        if reasons:

            logger.warning(
                "MANUAL ANALYSIS REJECTED | "
                "%s %s | %s",
                symbol,
                timeframe,
                " | ".join(reasons)
            )

            rejected_count += 1

            continue

        message = format_signal(
            symbol,
            timeframe,
            analysis,
            closed
        )

        sent = await send_owner_message(
            telegram_application,
            message
        )

        if sent:

            sent_any = True

            logger.info(
                "MANUAL SIGNAL SENT | "
                "%s %s",
                symbol,
                timeframe
            )

    if sent_any:

        await update.message.reply_text(
            "✅ تم إرسال الإشارة."
        )

    else:

        await update.message.reply_text(
            "🔎 تم التحليل.\n"
            "❌ لا يوجد Setup قوي حاليًا.\n\n"
            f"📊 أزواج تم تحليلها: "
            f"{analyzed_count}\n"
            f"🚫 مرفوضة: "
            f"{rejected_count}\n\n"
            "📋 السبب التفصيلي موجود في "
            "Render Logs."
        )


# ============================================================
# TEXT HANDLER
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

    await update.message.reply_text(
        "📡 بيانات MT4 موجودة.\n"
        "استخدم /analyze لتحليل آخر البيانات."
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

        photo = (
            update.message.photo[-1]
        )

        file = await context.bot.get_file(
            photo.file_id
        )

        buffer = io.BytesIO()

        await file.download_to_memory(
            buffer
        )

        image_bytes = (
            buffer.getvalue()
        )

        if not image_bytes:

            await update.message.reply_text(
                "❌ لم أستطع قراءة الصورة."
            )

            return

        prompt = """
Analyze this trading chart image.

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

If the setup is weak, say:

Setup ضعيف.

If the setup is strong, identify UP or DOWN
and explain the evidence.

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
            else "لم يتم الحصول على تحليل."
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
# POST INIT
# ============================================================

async def post_init(
    application
):

    global telegram_loop

    telegram_loop = (
        asyncio.get_running_loop()
    )

    logger.info(
        "Telegram event loop "
        "captured successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_application

    logger.info(
        "================================="
    )

    logger.info(
        "Starting ZinoProSignalAI"
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "Auto analysis interval: "
        "%s minutes",
        AUTO_ANALYSIS_INTERVAL_MINUTES
    )

    logger.info(
        "Entry delay follows timeframe."
    )

    logger.info(
        "================================="
    )

    # ========================================================
    # HTTP SERVER
    # ========================================================

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    # ========================================================
    # TELEGRAM
    # ========================================================

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    telegram_application = (
        application
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
            text_message_handler
        )
    )

    logger.info(
        "Telegram bot starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
