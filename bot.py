import os
import io
import json
import logging
import threading
import asyncio
import re
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

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_ID_RAW = os.getenv("OWNER_ID")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MT4_API_KEY = os.getenv("MT4_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID_RAW:
    raise RuntimeError("OWNER_ID is missing")

if not MT4_API_KEY:
    raise RuntimeError("MT4_API_KEY is missing")

OWNER_ID = int(OWNER_ID_RAW)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# GLOBAL STATE
# ============================================================

mt4_data = {}
mt4_data_lock = threading.Lock()

telegram_loop = None
telegram_bot = None

auto_analysis_running = set()
auto_analysis_lock = threading.Lock()

last_auto_candle = {}
last_auto_signal = {}
auto_signal_lock = threading.Lock()

stats = {
    "wins": 0,
    "losses": 0,
}


# ============================================================
# TIMEFRAME HELPERS
# ============================================================

def timeframe_to_minutes(timeframe):
    tf = str(timeframe or "").upper().strip()

    mapping = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M10": 10,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H2": 120,
        "H4": 240,
    }

    return mapping.get(tf, 1)


def get_next_entry_time(timeframe, delay_minutes=None):
    """
    Entry time is aligned to the next candle boundary.
    Example:
    M1 signal at 16:20 -> entry around 16:21.
    """

    minutes = timeframe_to_minutes(timeframe)

    if delay_minutes is None:
        delay_minutes = minutes

    now = datetime.now(ALGIERS)

    base = now.replace(
        second=0,
        microsecond=0,
    )

    next_boundary = base + timedelta(minutes=minutes)

    entry_time = next_boundary + timedelta(
        minutes=max(0, int(delay_minutes) - minutes)
    )

    return entry_time


# ============================================================
# NUMBER HELPERS
# ============================================================

def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        number = float(value)

        if number != number:
            return default

        return number

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def safe_score(value):
    try:
        number = int(value)
    except Exception:
        return 0

    return max(0, min(18, number))


def determine_direction(up_score, down_score):
    up_score = safe_score(up_score)
    down_score = safe_score(down_score)

    if up_score > down_score:
        return "UP"

    if down_score > up_score:
        return "DOWN"

    raise RuntimeError(
        f"Direction unclear: UP={up_score} DOWN={down_score}"
    )


def infer_digits(price):
    if price is None:
        return 5

    text = f"{price:.10f}".rstrip("0")

    if "." in text:
        return min(8, max(2, len(text.split(".")[1])))

    return 5


def format_price(price, digits=5):
    value = safe_float(price)

    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


# ============================================================
# CANDLE HELPERS
# ============================================================

def candle_time_value(candle):
    """
    Converts common MT4 candle time formats into a sortable value.
    """

    value = candle.get("time")

    if value is None:
        return 0

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if not text:
        return 0

    # Unix timestamp
    try:
        return float(text)
    except Exception:
        pass

    cleaned = text.replace("Z", "+00:00")

    formats = [
        "%Y.%m.%d %H:%M:%S",
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.timestamp()
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(cleaned)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ALGIERS)

        return dt.timestamp()

    except Exception:
        return 0


def normalize_candles(candles):
    clean = []

    if not isinstance(candles, list):
        return clean

    for candle in candles:
        if not isinstance(candle, dict):
            continue

        open_price = safe_float(candle.get("open"))
        high_price = safe_float(candle.get("high"))
        low_price = safe_float(candle.get("low"))
        close_price = safe_float(candle.get("close"))

        if None in (
            open_price,
            high_price,
            low_price,
            close_price,
        ):
            continue

        item = dict(candle)

        item["open"] = open_price
        item["high"] = high_price
        item["low"] = low_price
        item["close"] = close_price

        if candle.get("volume") is not None:
            item["volume"] = safe_float(
                candle.get("volume"),
                0,
            )

        clean.append(item)

    clean.sort(key=candle_time_value)

    return clean


def get_closed_candles(candles):
    """
    MT4 normally sends candles oldest -> newest.
    The newest candle is treated as the currently forming candle.

    Therefore:
        candles[:-1] = closed candles
    """

    clean = normalize_candles(candles)

    if len(clean) < 2:
        return []

    return clean[:-1]


# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1)

    current = sum(values[:period]) / period

    for value in values[period:]:
        current = (
            (value - current) * multiplier
        ) + current

    return current


def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]

        gains.append(max(change, 0))
        losses.append(max(-change, 0))

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

    return 100 - (100 / (1 + rs))


def calculate_williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(c["high"] for c in recent)
    lowest = min(c["low"] for c in recent)

    close = recent[-1]["close"]

    if highest == lowest:
        return -50.0

    return (
        (highest - close)
        / (highest - lowest)
        * -100
    )


def calculate_atr(candles, period=10):
    if len(candles) < period + 1:
        return None

    trs = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def calculate_adx(candles, period=14):
    if len(candles) < period * 2 + 2:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    true_ranges = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        plus = up_move if up_move > down_move and up_move > 0 else 0
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

        true_ranges.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(true_ranges) < period:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    atr = sum(true_ranges[:period]) / period
    plus_smoothed = sum(plus_dm[:period]) / period
    minus_smoothed = sum(minus_dm[:period]) / period

    dx_values = []

    for i in range(period, len(true_ranges)):
        atr = (
            (atr * (period - 1))
            + true_ranges[i]
        ) / period

        plus_smoothed = (
            (plus_smoothed * (period - 1))
            + plus_dm[i]
        ) / period

        minus_smoothed = (
            (minus_smoothed * (period - 1))
            + minus_dm[i]
        ) / period

        if atr == 0:
            continue

        plus_di = 100 * plus_smoothed / atr
        minus_di = 100 * minus_smoothed / atr

        denominator = plus_di + minus_di

        if denominator == 0:
            dx = 0
        else:
            dx = (
                abs(plus_di - minus_di)
                / denominator
            ) * 100

        dx_values.append(
            (dx, plus_di, minus_di)
        )

    if len(dx_values) < period:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    adx = sum(
        item[0] for item in dx_values[-period:]
    ) / period

    plus_di = dx_values[-1][1]
    minus_di = dx_values[-1][2]

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def calculate_keltner(candles):
    closes = [c["close"] for c in candles]

    middle = ema(closes, 20)
    atr = calculate_atr(candles, 10)

    if middle is None or atr is None:
        return {
            "middle": None,
            "upper": None,
            "lower": None,
        }

    multiplier = 5

    return {
        "middle": middle,
        "upper": middle + (atr * multiplier),
        "lower": middle - (atr * multiplier),
    }


def calculate_indicators(candles):
    closes = [c["close"] for c in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi = calculate_rsi(closes, 14)
    williams = calculate_williams_r(candles, 14)

    adx = calculate_adx(candles, 14)

    keltner = calculate_keltner(candles)

    return {
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi,
        "williams_r": williams,
        "adx": adx,
        "keltner": keltner,
    }


# ============================================================
# PRICE ACTION / STRUCTURE
# ============================================================

def candle_description(candle):
    open_price = candle["open"]
    high = candle["high"]
    low = candle["low"]
    close = candle["close"]

    body = abs(close - open_price)
    full_range = high - low

    if full_range <= 0:
        return "flat"

    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low

    body_ratio = body / full_range

    if close > open_price:
        direction = "bullish"
    elif close < open_price:
        direction = "bearish"
    else:
        direction = "neutral"

    if body_ratio >= 0.65:
        strength = "strong"
    elif body_ratio >= 0.35:
        strength = "moderate"
    else:
        strength = "weak"

    return (
        f"{direction} {strength}; "
        f"body_ratio={body_ratio:.2f}; "
        f"upper_wick={upper_wick:.6f}; "
        f"lower_wick={lower_wick:.6f}"
    )


def structure_analysis(candles):
    if len(candles) < 8:
        return "insufficient"

    recent = candles[-8:]

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    first_half_high = max(highs[:4])
    second_half_high = max(highs[4:])

    first_half_low = min(lows[:4])
    second_half_low = min(lows[4:])

    if (
        second_half_high > first_half_high
        and second_half_low > first_half_low
    ):
        return "HH + HL"

    if (
        second_half_high < first_half_high
        and second_half_low < first_half_low
    ):
        return "LH + LL"

    return "mixed/range"


def detect_recent_breakout(candles):
    if len(candles) < 8:
        return "none"

    last = candles[-1]

    previous = candles[-7:-1]

    previous_high = max(c["high"] for c in previous)
    previous_low = min(c["low"] for c in previous)

    if last["close"] > previous_high:
        return "bullish breakout"

    if last["close"] < previous_low:
        return "bearish breakout"

    return "no clear breakout"


# ============================================================
# DETERMINISTIC CANCELLATION LEVEL
# ============================================================

def find_swing_low(candles):
    if len(candles) < 5:
        return None

    # Search from most recent closed candle backwards.
    for i in range(len(candles) - 2, 1, -1):
        current = candles[i]

        if (
            current["low"] <= candles[i - 1]["low"]
            and current["low"] <= candles[i + 1]["low"]
        ):
            return current["low"]

    recent = candles[-6:]

    return min(c["low"] for c in recent)


def find_swing_high(candles):
    if len(candles) < 5:
        return None

    for i in range(len(candles) - 2, 1, -1):
        current = candles[i]

        if (
            current["high"] >= candles[i - 1]["high"]
            and current["high"] >= candles[i + 1]["high"]
        ):
            return current["high"]

    recent = candles[-6:]

    return max(c["high"] for c in recent)


def calculate_cancellation(direction, candles, entry_price):
    if not candles or entry_price is None:
        return None

    recent = candles[-12:]

    if direction == "UP":
        level = find_swing_low(recent)

        if level is None:
            level = min(c["low"] for c in recent)

        # Must actually be below entry.
        if level >= entry_price:
            level = min(
                c["low"] for c in recent
            )

        return level

    level = find_swing_high(recent)

    if level is None:
        level = max(c["high"] for c in recent)

    if level <= entry_price:
        level = max(
            c["high"] for c in recent
        )

    return level


# ============================================================
# GEMINI PROMPT
# ============================================================

MT4_ANALYSIS_PROMPT = """
You are the technical-analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied MT4 OHLC candle data and calculated indicators.

IMPORTANT:
- Do NOT use internet data.
- Do NOT invent prices.
- Do NOT invent indicator values.
- Do NOT use future candles.
- The supplied candles are CLOSED candles only.
- The last supplied candle is the most recently CLOSED candle.
- Analyze the current market structure from those closed candles.

PRIMARY PRIORITY:
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

SCORING MUST TOTAL 18:

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

For every category, assign points toward UP or DOWN.

Do NOT force a strong score when evidence is weak.

A score difference of 1-2 points means the market is relatively close.
A score difference of 3-5 points means moderate directional evidence.
A score difference of 6+ points means strong directional evidence.

Confidence must reflect evidence quality.

Do NOT give 90%+ confidence unless there is unusually strong multi-factor confluence.

The final direction must be determined strictly from the larger score:
UP if UP score > DOWN score.
DOWN if DOWN score > UP score.

Never use WAIT, NEUTRAL or NO SIGNAL.
If the scores are equal, re-evaluate the evidence and choose the side with the stronger concrete price-action evidence.

Return ONLY valid JSON.

JSON schema:

{
  "asset": "EURUSD",
  "timeframe": "M1",
  "direction": "UP",
  "confidence": 75,
  "up_score": 11,
  "down_score": 7,
  "structure": "...",
  "breakout": "...",
  "liquidity": "...",
  "momentum": "...",
  "candle": "...",
  "rsi": "...",
  "williams": "...",
  "ema": "...",
  "keltner": "...",
  "adx": "...",
  "reason": "short concise reason"
}
"""


# ============================================================
# GEMINI JSON CLEANER
# ============================================================

def clean_json_text(text):
    if not text:
        raise ValueError("Gemini returned empty response")

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


# ============================================================
# GEMINI ANALYSIS
# ============================================================

async def analyze_mt4_data(market_data):
    symbol = str(
        market_data.get("symbol", "UNKNOWN")
    ).upper()

    timeframe = str(
        market_data.get("timeframe", "M1")
    ).upper()

    candles = normalize_candles(
        market_data.get("candles", [])
    )

    closed_candles = get_closed_candles(candles)

    if len(closed_candles) < 30:
        raise RuntimeError(
            f"Not enough closed candles: {len(closed_candles)}"
        )

    # IMPORTANT:
    # Gemini receives ONLY closed candles.
    analysis_candles = closed_candles[-100:]

    indicators = calculate_indicators(
        analysis_candles
    )

    structure = structure_analysis(
        analysis_candles
    )

    breakout = detect_recent_breakout(
        analysis_candles
    )

    latest_closed = analysis_candles[-1]

    candle_info = candle_description(
        latest_closed
    )

    compact_candles = []

    for candle in analysis_candles:
        compact_candles.append({
            "time": candle.get("time"),
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle.get("volume", 0),
        })

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "latest_closed_candle": latest_closed,
        "structure": structure,
        "breakout": breakout,
        "latest_candle_description": candle_info,
        "indicators": indicators,
        "candles": compact_candles,
    }

    prompt = (
        MT4_ANALYSIS_PROMPT
        + "\n\nMARKET DATA:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
    )

    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.15,
        ),
    )

    raw_text = getattr(
        response,
        "text",
        None,
    )

    cleaned = clean_json_text(raw_text)

    result = json.loads(cleaned)

    up_score = safe_score(
        result.get("up_score")
    )

    down_score = safe_score(
        result.get("down_score")
    )

    # Direction is NOT trusted from Gemini text.
    direction = determine_direction(
        up_score,
        down_score,
    )

    result["asset"] = symbol
    result["timeframe"] = timeframe
    result["up_score"] = up_score
    result["down_score"] = down_score
    result["direction"] = direction

    confidence = safe_int(
        result.get("confidence"),
        50,
    )

    confidence = max(
        50,
        min(89, confidence),
    )

    # Prevent exaggerated confidence when score difference is small.
    difference = abs(
        up_score - down_score
    )

    if difference <= 1:
        confidence = min(
            confidence,
            58,
        )
    elif difference == 2:
        confidence = min(
            confidence,
            64,
        )
    elif difference == 3:
        confidence = min(
            confidence,
            70,
        )
    elif difference == 4:
        confidence = min(
            confidence,
            76,
        )

    result["confidence"] = confidence

    # Keep actual MT4 price.
    actual_price = safe_float(
        market_data.get("price")
    )

    if actual_price is None:
        actual_price = latest_closed["close"]

    result["entry_price"] = actual_price

    result["_closed_candles"] = analysis_candles

    return result


# ============================================================
# FORMAT MT4 SIGNAL
# ============================================================

def format_mt4_signal(result, market_data):
    symbol = str(
        result.get("asset", market_data.get("symbol", "UNKNOWN"))
    ).upper()

    timeframe = str(
        result.get(
            "timeframe",
            market_data.get("timeframe", "M1"),
        )
    ).upper()

    up_score = safe_score(
        result.get("up_score")
    )

    down_score = safe_score(
        result.get("down_score")
    )

    direction = determine_direction(
        up_score,
        down_score,
    )

    confidence = safe_int(
        result.get("confidence"),
        50,
    )

    confidence = max(
        50,
        min(89, confidence),
    )

    entry_price = safe_float(
        market_data.get("price")
    )

    if entry_price is None:
        entry_price = safe_float(
            result.get("entry_price")
        )

    if entry_price is None:
        closed = result.get("_closed_candles", [])

        if closed:
            entry_price = closed[-1]["close"]

    digits = safe_int(
        market_data.get("digits"),
        infer_digits(entry_price),
    )

    delay = timeframe_to_minutes(
        timeframe
    )

    entry_time = get_next_entry_time(
        timeframe,
        delay,
    )

    cancellation = calculate_cancellation(
        direction,
        result.get("_closed_candles", []),
        entry_price,
    )

    price_text = format_price(
        entry_price,
        digits,
    )

    cancellation_text = format_price(
        cancellation,
        digits,
    )

    if direction == "UP":
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{cancellation_text}"
        )
    else:
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{cancellation_text}"
        )

    reason = str(
        result.get(
            "reason",
            "Price action and structure analysis.",
        )
    ).strip()

    if len(reason) > 240:
        reason = reason[:237] + "..."

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 الأصل: {symbol}\n"
        f"⏱ الفريم: {timeframe}\n\n"
        f"🎯 Confidence: {confidence}%\n"
        f"📌 القرار: {direction}\n\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏳ الدخول بعد: {delay} دقيقة\n"
        f"🕒 وقت الدخول: "
        f"{entry_time.strftime('%H:%M:%S')}\n"
        f"💰 سعر الدخول: {price_text}\n"
        f"⚠️ {cancel_text}\n\n"
        f"📝 السبب:\n{reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# AUTO ANALYSIS
# ============================================================

async def auto_analyze_and_send(symbol, timeframe):
    key = (
        f"{str(symbol).upper()}:"
        f"{str(timeframe).upper()}"
    )

    with auto_analysis_lock:
        if key in auto_analysis_running:
            logger.info(
                "AUTO ANALYSIS ALREADY RUNNING: %s",
                key,
            )
            return

        auto_analysis_running.add(key)

    try:
        market_data = get_mt4_data(
            symbol,
            timeframe,
        )

        if market_data is None:
            logger.warning(
                "AUTO ANALYSIS: no MT4 data for %s",
                key,
            )
            return

        candles = normalize_candles(
            market_data.get("candles", [])
        )

        if len(candles) < 31:
            logger.warning(
                "AUTO ANALYSIS: not enough candles for %s: %s",
                key,
                len(candles),
            )
            return

        latest_candle_time = str(
            candles[-1].get("time", "")
        )

        signal_key = (
            f"{key}:{latest_candle_time}"
        )

        with auto_signal_lock:
            if (
                last_auto_signal.get(key)
                == signal_key
            ):
                logger.info(
                    "AUTO SIGNAL DUPLICATE BLOCKED: %s",
                    signal_key,
                )
                return

        logger.info(
            "AUTO ANALYSIS START: %s",
            key,
        )

        result = await analyze_mt4_data(
            market_data
        )

        signal = format_mt4_signal(
            result,
            market_data,
        )

        with auto_signal_lock:
            if (
                last_auto_signal.get(key)
                == signal_key
            ):
                return

            last_auto_signal[key] = signal_key

        if telegram_bot is None:
            logger.warning(
                "Telegram bot is not ready"
            )
            return

        await telegram_bot.send_message(
            chat_id=OWNER_ID,
            text=signal,
        )

        logger.info(
            "AUTO SIGNAL SENT: %s",
            key,
        )

    except Exception:
        logger.exception(
            "AUTO ANALYSIS ERROR: %s",
            key,
        )

    finally:
        with auto_analysis_lock:
            auto_analysis_running.discard(key)


def schedule_auto_analysis(symbol, timeframe):
    global telegram_loop

    if telegram_loop is None:
        logger.warning(
            "Cannot schedule auto analysis: Telegram loop unavailable"
        )
        return

    future = asyncio.run_coroutine_threadsafe(
        auto_analyze_and_send(
            symbol,
            timeframe,
        ),
        telegram_loop,
    )

    def done_callback(f):
        try:
            f.result()
        except Exception:
            logger.exception(
                "Scheduled auto analysis failed"
            )

    future.add_done_callback(done_callback)

    logger.info(
        "AUTO ANALYSIS SCHEDULED: %s %s",
        symbol,
        timeframe,
    )


# ============================================================
# MT4 DATA ACCESS
# ============================================================

def get_mt4_data(symbol, timeframe):
    key = (
        f"{str(symbol).upper()}:"
        f"{str(timeframe).upper()}"
    )

    with mt4_data_lock:
        data = mt4_data.get(key)

        if data is None:
            return None

        return dict(data)


# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def send_json(self, status_code, payload):
        body = json.dumps(
            payload,
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

    def do_GET(self):
        path = urlparse(
            self.path
        ).path

        if path == "/":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                },
            )
            return

        if path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                },
            )
            return

        self.send_json(
            404,
            {
                "status": "error",
                "message": "Not found",
            },
        )

    def do_POST(self):
        path = urlparse(
            self.path
        ).path

        if path != "/mt4":
            self.send_json(
                404,
                {
                    "status": "error",
                    "message": "Not found",
                },
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

        except Exception:
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "Invalid Content-Length",
                },
            )
            return

        if content_length <= 0:
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "Empty request",
                },
            )
            return

        if content_length > 2_000_000:
            self.send_json(
                413,
                {
                    "status": "error",
                    "message": "Request too large",
                },
            )
            return

        try:
            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

        except Exception:
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "Invalid JSON",
                },
            )
            return

        if payload.get("api_key") != MT4_API_KEY:
            self.send_json(
                401,
                {
                    "status": "error",
                    "message": "Unauthorized",
                },
            )
            return

        symbol = str(
            payload.get("symbol", "")
        ).strip().upper()

        timeframe = str(
            payload.get("timeframe", "M1")
        ).strip().upper()

        candles = payload.get(
            "candles",
            [],
        )

        if not symbol:
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "Missing symbol",
                },
            )
            return

        if not isinstance(candles, list):
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "candles must be a list",
                },
            )
            return

        clean_candles = normalize_candles(
            candles[:200]
        )

        if not clean_candles:
            self.send_json(
                400,
                {
                    "status": "error",
                    "message": "No valid candles",
                },
            )
            return

        key = (
            f"{symbol}:{timeframe}"
        )

        latest_candle_time = str(
            clean_candles[-1].get(
                "time",
                "",
            )
        )

        previous_candle_time = (
            last_auto_candle.get(key)
        )

        new_candle = (
            bool(latest_candle_time)
            and latest_candle_time
            != previous_candle_time
        )

        if latest_candle_time:
            last_auto_candle[key] = (
                latest_candle_time
            )

        price = safe_float(
            payload.get("price")
        )

        digits = safe_int(
            payload.get("digits"),
            infer_digits(price),
        )

        server_time = payload.get(
            "server_time"
        )

        data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": price,
            "digits": digits,
            "server_time": server_time,
            "received_at": datetime.now(
                ALGIERS
            ).isoformat(),
            "candles": clean_candles,
        }

        with mt4_data_lock:
            mt4_data[key] = data

        logger.info(
            "MT4 data received: %s | %s | candles=%s | new_candle=%s",
            symbol,
            timeframe,
            len(clean_candles),
            new_candle,
        )

        if new_candle:
            schedule_auto_analysis(
                symbol,
                timeframe,
            )

        self.send_json(
            200,
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(clean_candles),
                "new_candle": new_candle,
                "auto_analysis": True,
            },
        )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    logger.info(
        "HTTP server running on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# TELEGRAM OWNER CHECK
# ============================================================

def is_owner(update):
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


async def reject_non_owner(update):
    if update.message:
        await update.message.reply_text(
            "⛔ غير مصرح."
        )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ MT4 Auto Analysis فعال\n"
        "📊 التحليل يتم تلقائياً عند بداية كل شمعة جديدة.\n\n"
        "الأوامر:\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/analyze EURUSD M1"
    )


async def stats_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    wins = stats["wins"]
    losses = stats["losses"]
    total = wins + losses

    if total:
        winrate = (wins / total) * 100
    else:
        winrate = 0

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n\n"
        f"✅ WIN: {wins}\n"
        f"❌ LOSS: {losses}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {winrate:.1f}%"
    )


async def win_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    stats["wins"] += 1

    await update.message.reply_text(
        f"✅ WIN registered\n"
        f"Wins: {stats['wins']}\n"
        f"Losses: {stats['losses']}"
    )


async def loss_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    stats["losses"] += 1

    await update.message.reply_text(
        f"❌ LOSS registered\n"
        f"Wins: {stats['wins']}\n"
        f"Losses: {stats['losses']}"
    )


async def reset_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    stats["wins"] = 0
    stats["losses"] = 0

    await update.message.reply_text(
        "♻️ Stats reset."
    )


async def mt4status_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    with mt4_data_lock:
        items = list(
            mt4_data.items()
        )

    if not items:
        await update.message.reply_text(
            "❌ لا توجد بيانات MT4 حالياً."
        )
        return

    lines = [
        "📡 MT4 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for key, data in items:
        candles = data.get(
            "candles",
            [],
        )

        lines.append(
            f"📊 {key} | "
            f"Candles: {len(candles)} | "
            f"Price: {data.get('price')}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# MANUAL MT4 ANALYSIS
# ============================================================

async def manual_analyze(update, context, symbol, timeframe):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    await update.message.reply_text(
        f"🔎 جاري تحليل {symbol.upper()} {timeframe.upper()}..."
    )

    market_data = get_mt4_data(
        symbol,
        timeframe,
    )

    if market_data is None:
        await update.message.reply_text(
            "❌ لا توجد بيانات MT4 لهذا الزوج والفريم."
        )
        return

    try:
        result = await analyze_mt4_data(
            market_data
        )

        signal = format_mt4_signal(
            result,
            market_data,
        )

        await update.message.reply_text(
            signal
        )

    except Exception as exc:
        logger.exception(
            "Manual MT4 analysis failed"
        )

        await update.message.reply_text(
            f"❌ فشل التحليل:\n{exc}"
        )


async def analyze_command(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    if len(context.args) < 1:
        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURUSD M1"
        )
        return

    symbol = context.args[0].upper()

    timeframe = (
        context.args[1].upper()
        if len(context.args) >= 2
        else "M1"
    )

    await manual_analyze(
        update,
        context,
        symbol,
        timeframe,
    )


# ============================================================
# SCREENSHOT ANALYSIS
# ============================================================

SCREENSHOT_PROMPT = """
Analyze this trading chart for short-term binary-options technical analysis.

Use ONLY visible chart information.

Indicators expected when visible:
- EMA 9
- EMA 21
- RSI 14
- Williams %R 14
- ADX 14 + DI 14
- Keltner EMA20 / ATR10 / multiplier5

Priority:
Price Action > Structure > Breakout/Retest > Liquidity >
Momentum > Candle > EMA > RSI > Williams > Keltner > ADX/DI

Scoring:
Structure 2
Breakout 2
Liquidity 1
Momentum 2
Candle 2
RSI 1
Summary 2
Oscillators 3
Moving Averages 3

TOTAL = 18.

Return JSON only:

{
  "asset": "...",
  "timeframe": "...",
  "direction": "UP",
  "confidence": 70,
  "up_score": 11,
  "down_score": 7,
  "reason": "..."
}

Direction must be UP or DOWN.
Do not return WAIT or NO SIGNAL.
Do not invent unseen values.
Do not give 90%+ confidence without very strong confluence.
"""


async def analyze_chart(image_bytes):
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )

    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=GEMINI_MODEL,
        contents=[
            SCREENSHOT_PROMPT,
            image_part,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.15,
        ),
    )

    raw_text = getattr(
        response,
        "text",
        None,
    )

    cleaned = clean_json_text(
        raw_text
    )

    result = json.loads(cleaned)

    up_score = safe_score(
        result.get("up_score")
    )

    down_score = safe_score(
        result.get("down_score")
    )

    direction = determine_direction(
        up_score,
        down_score,
    )

    confidence = safe_int(
        result.get("confidence"),
        50,
    )

    confidence = max(
        50,
        min(89, confidence),
    )

    difference = abs(
        up_score - down_score
    )

    if difference <= 1:
        confidence = min(
            confidence,
            58,
        )
    elif difference == 2:
        confidence = min(
            confidence,
            64,
        )
    elif difference == 3:
        confidence = min(
            confidence,
            70,
        )
    elif difference == 4:
        confidence = min(
            confidence,
            76,
        )

    result["direction"] = direction
    result["up_score"] = up_score
    result["down_score"] = down_score
    result["confidence"] = confidence

    return result


def format_screenshot_signal(result):
    asset = str(
        result.get(
            "asset",
            "UNKNOWN",
        )
    ).upper()

    timeframe = str(
        result.get(
            "timeframe",
            "M1",
        )
    ).upper()

    direction = determine_direction(
        result.get("up_score"),
        result.get("down_score"),
    )

    confidence = safe_int(
        result.get("confidence"),
        50,
    )

    up_score = safe_score(
        result.get("up_score")
    )

    down_score = safe_score(
        result.get("down_score")
    )

    delay = timeframe_to_minutes(
        timeframe
    )

    entry_time = get_next_entry_time(
        timeframe,
        delay,
    )

    reason = str(
        result.get(
            "reason",
            "Chart price-action analysis.",
        )
    ).strip()

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 الأصل: {asset}\n"
        f"⏱ الفريم: {timeframe}\n\n"
        f"🎯 Confidence: {confidence}%\n"
        f"📌 القرار: {direction}\n\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏳ الدخول بعد: {delay} دقيقة\n"
        f"🕒 وقت الدخول: "
        f"{entry_time.strftime('%H:%M:%S')}\n\n"
        f"📝 السبب:\n{reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


async def photo_handler(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    if not update.message or not update.message.photo:
        return

    try:
        photo = update.message.photo[-1]

        telegram_file = await photo.get_file()

        buffer = io.BytesIO()

        await telegram_file.download_to_memory(
            buffer
        )

        image_bytes = buffer.getvalue()

        if not image_bytes:
            await update.message.reply_text(
                "❌ الصورة فارغة."
            )
            return

        await update.message.reply_text(
            "🔎 جاري تحليل الصورة..."
        )

        result = await analyze_chart(
            image_bytes
        )

        signal = format_screenshot_signal(
            result
        )

        await update.message.reply_text(
            signal
        )

    except Exception as exc:
        logger.exception(
            "Screenshot analysis failed"
        )

        await update.message.reply_text(
            f"❌ فشل تحليل الصورة:\n{exc}"
        )


# ============================================================
# TEXT SHORTCUT
# ============================================================

async def text_handler(update, context):
    if not is_owner(update):
        await reject_non_owner(update)
        return

    if not update.message:
        return

    text = update.message.text.strip()

    parts = text.split()

    if len(parts) >= 1:
        symbol = parts[0].upper()

        timeframe = (
            parts[1].upper()
            if len(parts) >= 2
            else "M1"
        )

        known_timeframes = {
            "M1",
            "M2",
            "M3",
            "M5",
            "M10",
            "M15",
            "M30",
            "H1",
            "H2",
            "H4",
        }

        if timeframe in known_timeframes:
            await manual_analyze(
                update,
                context,
                symbol,
                timeframe,
            )


# ============================================================
# TELEGRAM POST INIT
# ============================================================

async def post_init(application):
    global telegram_loop
    global telegram_bot

    telegram_loop = asyncio.get_running_loop()
    telegram_bot = application.bot

    logger.info(
        "Telegram loop initialized"
    )

    logger.info(
        "ZinoProSignalAI Telegram bot ready"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(
        "================================="
    )

    logger.info(
        "ZinoProSignalAI starting"
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "Automatic MT4 analysis: ENABLED"
    )

    logger.info(
        "Analysis frequency: NEW CANDLE"
    )

    logger.info(
        "================================="
    )

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

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
            "mt4status",
            mt4status_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    logger.info(
        "Starting Telegram polling..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
