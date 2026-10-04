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

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_ID_RAW = os.getenv("OWNER_ID")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MT4_API_KEY = os.getenv("MT4_API_KEY")

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

# ============================================================
# AUTO CYCLE SETTINGS
# ============================================================

# دورة جديدة كل 3 دقائق
AUTO_CYCLE_MINUTES = 3
AUTO_CYCLE_SECONDS = AUTO_CYCLE_MINUTES * 60

# الإشارة الثانية بعد دقيقتين
SECOND_SIGNAL_DELAY_SECONDS = 120

# أقصى عدد إشارات في الدورة
MAX_SIGNALS_PER_CYCLE = 2

# أقل وقت متبقٍ قبل الدخول
MIN_ENTRY_LEAD_SECONDS = 20

# الفريمات التي تدخل في الاختيار التلقائي
AUTO_TIMEFRAMES = {"M1", "M3"}

# عدد المرشحين الذين نرسلهم إلى Gemini
AUTO_MAX_CANDIDATES = 3

# عمر بيانات MT4 الأقصى
AUTO_DATA_MAX_AGE_SECONDS = 180


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
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
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

last_auto_candle = {}
last_auto_signal = {}

auto_cycle_running = False
auto_cycle_lock = None

stats = {
    "wins": 0,
    "losses": 0,
}


# ============================================================
# BASIC HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def is_owner(update: Update):
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


def normalize_symbol(symbol):
    return str(symbol or "").strip().upper()


def normalize_timeframe(timeframe):
    return str(timeframe or "M1").strip().upper()


def timeframe_to_minutes(timeframe):
    tf = normalize_timeframe(timeframe)

    mapping = {
        "M1": 1,
        "1M": 1,
        "M2": 2,
        "2M": 2,
        "M3": 3,
        "3M": 3,
        "M5": 5,
        "5M": 5,
        "M10": 10,
        "10M": 10,
        "M15": 15,
        "15M": 15,
        "M30": 30,
        "30M": 30,
        "H1": 60,
        "1H": 60,
        "H2": 120,
        "2H": 120,
        "H4": 240,
        "4H": 240,
    }

    return mapping.get(tf, 1)


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        text = str(value).strip()

        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ALGIERS)

        return dt.astimezone(ALGIERS)

    except Exception:
        return None


def get_data_age_seconds(data):
    received = parse_iso_datetime(data.get("received_at"))

    if received is None:
        received = parse_iso_datetime(data.get("server_time"))

    if received is None:
        return 999999

    age = (now_algiers() - received).total_seconds()

    return max(0, age)


# ============================================================
# ENTRY TIME
# ============================================================

def get_next_entry_time(timeframe, delay_minutes=None):
    """
    Calculates the next valid candle boundary in Algiers time.

    Hard rule:
    The entry must always be at least MIN_ENTRY_LEAD_SECONDS
    in the future.
    """

    minutes = delay_minutes or timeframe_to_minutes(timeframe)

    now = now_algiers()

    total_minutes = now.hour * 60 + now.minute

    next_total_minutes = (
        (total_minutes // minutes) + 1
    ) * minutes

    day_offset = 0

    if next_total_minutes >= 24 * 60:
        next_total_minutes -= 24 * 60
        day_offset = 1

    hour = next_total_minutes // 60
    minute = next_total_minutes % 60

    entry_time = now.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )

    if day_offset:
        entry_time += timedelta(days=1)

    # Hard minimum lead-time protection
    while (entry_time - now).total_seconds() < MIN_ENTRY_LEAD_SECONDS:
        entry_time += timedelta(minutes=minutes)

    return entry_time


# ============================================================
# EMA
# ============================================================

def ema(values, period):
    if not values:
        return []

    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)

    sma = sum(values[:period]) / period
    result[period - 1] = sma

    multiplier = 2 / (period + 1)

    previous = sma

    for i in range(period, len(values)):
        previous = (
            (values[i] - previous) * multiplier
        ) + previous

        result[i] = previous

    return result


# ============================================================
# RSI
# ============================================================

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

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1)) + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + losses[i]
        ) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


# ============================================================
# WILLIAMS %R
# ============================================================

def calculate_williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(c["high"] for c in recent)
    lowest = min(c["low"] for c in recent)

    close = recent[-1]["close"]

    if highest == lowest:
        return -50.0

    value = (
        (highest - close)
        / (highest - lowest)
    ) * -100

    return round(value, 2)


# ============================================================
# ATR
# ============================================================

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


# ============================================================
# ADX + DI
# ============================================================

def calculate_adx(candles, period=14):
    if len(candles) < period * 2:
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

        high_diff = current["high"] - previous["high"]
        low_diff = previous["low"] - current["low"]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)

        if high_diff > low_diff and high_diff > 0:
            plus_dm.append(high_diff)
        else:
            plus_dm.append(0)

        if low_diff > high_diff and low_diff > 0:
            minus_dm.append(low_diff)
        else:
            minus_dm.append(0)

    if len(trs) < period:
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    atr = sum(trs[:period]) / period
    plus = sum(plus_dm[:period]) / period
    minus = sum(minus_dm[:period]) / period

    dx_values = []

    for i in range(period, len(trs)):
        atr = ((atr * (period - 1)) + trs[i]) / period
        plus = ((plus * (period - 1)) + plus_dm[i]) / period
        minus = ((minus * (period - 1)) + minus_dm[i]) / period

        if atr == 0:
            continue

        plus_di = (plus / atr) * 100
        minus_di = (minus / atr) * 100

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
        item[0] for item in dx_values[:period]
    ) / period

    last_plus = dx_values[period - 1][1]
    last_minus = dx_values[period - 1][2]

    for i in range(period, len(dx_values)):
        adx = (
            (adx * (period - 1))
            + dx_values[i][0]
        ) / period

        last_plus = dx_values[i][1]
        last_minus = dx_values[i][2]

    return {
        "adx": round(adx, 2),
        "plus_di": round(last_plus, 2),
        "minus_di": round(last_minus, 2),
    }


# ============================================================
# KELTNER
# ============================================================

def calculate_keltner(candles):
    if len(candles) < 20:
        return {
            "middle": None,
            "upper": None,
            "lower": None,
        }

    closes = [c["close"] for c in candles]

    middle_values = ema(closes, 20)
    middle = middle_values[-1]

    atr = calculate_atr(candles, 10)

    if middle is None or atr is None:
        return {
            "middle": None,
            "upper": None,
            "lower": None,
        }

    multiplier = 5

    upper = middle + (
        atr * multiplier
    )

    lower = middle - (
        atr * multiplier
    )

    return {
        "middle": round(middle, 8),
        "upper": round(upper, 8),
        "lower": round(lower, 8),
    }


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(candles):
    closes = [c["close"] for c in candles]

    ema9_values = ema(closes, 9)
    ema21_values = ema(closes, 21)

    adx_data = calculate_adx(
        candles,
        14,
    )

    keltner = calculate_keltner(
        candles
    )

    rsi = calculate_rsi(
        closes,
        14,
    )

    williams = calculate_williams_r(
        candles,
        14,
    )

    return {
        "ema9": (
            round(ema9_values[-1], 8)
            if ema9_values[-1] is not None
            else None
        ),
        "ema21": (
            round(ema21_values[-1], 8)
            if ema21_values[-1] is not None
            else None
        ),
        "rsi14": rsi,
        "williams_r14": williams,
        "adx14": adx_data["adx"],
        "plus_di14": adx_data["plus_di"],
        "minus_di14": adx_data["minus_di"],
        "keltner": keltner,
    }


# ============================================================
# STRUCTURE
# ============================================================

def structure_analysis(candles):
    if len(candles) < 8:
        return "mixed"

    recent = candles[-8:]

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    first_half = recent[:4]
    second_half = recent[4:]

    first_high = max(
        c["high"] for c in first_half
    )

    second_high = max(
        c["high"] for c in second_half
    )

    first_low = min(
        c["low"] for c in first_half
    )

    second_low = min(
        c["low"] for c in second_half
    )

    if (
        second_high > first_high
        and second_low > first_low
    ):
        return "HH + HL"

    if (
        second_high < first_high
        and second_low < first_low
    ):
        return "LH + LL"

    return "mixed"


# ============================================================
# BREAKOUT
# ============================================================

def detect_recent_breakout(candles):
    if len(candles) < 12:
        return "no clear breakout"

    previous = candles[-12:-2]
    last_two = candles[-2:]

    resistance = max(
        c["high"] for c in previous
    )

    support = min(
        c["low"] for c in previous
    )

    last_close = last_two[-1]["close"]

    if last_close > resistance:
        return "bullish breakout"

    if last_close < support:
        return "bearish breakout"

    return "no clear breakout"


# ============================================================
# CANDLE DESCRIPTION
# ============================================================

def candle_description(candle):
    open_price = candle["open"]
    close_price = candle["close"]
    high = candle["high"]
    low = candle["low"]

    body = abs(close_price - open_price)
    full_range = max(high - low, 0.00000001)

    body_ratio = body / full_range

    if close_price > open_price:
        direction = "bullish"
    elif close_price < open_price:
        direction = "bearish"
    else:
        direction = "doji"

    return {
        "direction": direction,
        "body_ratio": round(body_ratio, 3),
        "body": body,
        "range": full_range,
    }


# ============================================================
# SWING LEVELS / CANCELLATION
# ============================================================

def find_swing_low(candles, lookback=12):
    recent = candles[-lookback:]

    return min(
        c["low"] for c in recent
    )


def find_swing_high(candles, lookback=12):
    recent = candles[-lookback:]

    return max(
        c["high"] for c in recent
    )


def calculate_cancellation(
    direction,
    candles,
    entry_price,
):
    direction = str(direction).upper()

    if direction == "UP":
        swing = find_swing_low(
            candles,
            min(12, len(candles)),
        )

        if swing >= entry_price:
            return round(
                entry_price * 0.9995,
                8,
            )

        return round(swing, 8)

    swing = find_swing_high(
        candles,
        min(12, len(candles)),
    )

    if swing <= entry_price:
        return round(
            entry_price * 1.0005,
            8,
        )

    return round(swing, 8)


# ============================================================
# GET CLOSED CANDLES
# ============================================================

def get_closed_candles(raw_candles):
    candles = []

    if not isinstance(raw_candles, list):
        return candles

    for item in raw_candles:
        if not isinstance(item, dict):
            continue

        try:
            candle = {
                "time": item.get("time"),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
            }

            candles.append(candle)

        except Exception:
            continue

    # MT4 sends the newest candle as forming candle.
    # We remove it and analyze closed candles only.
    if len(candles) >= 2:
        return candles[:-1]

    return []


# ============================================================
# QUALITY FILTER
# ============================================================

def evaluate_signal_quality(
    result,
    candles,
    indicators,
):
    try:
        up_score = int(result.get("up_score", 0))
        down_score = int(result.get("down_score", 0))
        confidence = float(
            result.get("confidence", 0)
        )
    except Exception:
        return {
            "approved": False,
            "reason": "invalid score data",
        }

    total = up_score + down_score

    if total != 18:
        return {
            "approved": False,
            "reason": f"score total is {total}, expected 18",
        }

    if up_score == down_score:
        return {
            "approved": False,
            "reason": "scores are equal",
        }

    direction = (
        "UP"
        if up_score > down_score
        else "DOWN"
    )

    selected_score = max(
        up_score,
        down_score,
    )

    score_difference = abs(
        up_score - down_score
    )

    if score_difference < 5:
        return {
            "approved": False,
            "reason": "score difference too small",
        }

    if selected_score < 11:
        return {
            "approved": False,
            "reason": "selected score below 11/18",
        }

    if confidence < 70:
        return {
            "approved": False,
            "reason": "confidence below 70%",
        }

    # Never allow fake ultra-high confidence.
    if score_difference <= 1:
        confidence = min(confidence, 58)
    elif score_difference == 2:
        confidence = min(confidence, 64)
    elif score_difference == 3:
        confidence = min(confidence, 70)
    elif score_difference == 4:
        confidence = min(confidence, 76)
    else:
        confidence = min(confidence, 89)

    if len(candles) < 40:
        return {
            "approved": False,
            "reason": "not enough closed candles",
        }

    structure = structure_analysis(candles)
    breakout = detect_recent_breakout(candles)

    ema9 = indicators.get("ema9")
    ema21 = indicators.get("ema21")
    rsi = indicators.get("rsi14")
    williams = indicators.get("williams_r14")
    adx = indicators.get("adx14")
    plus_di = indicators.get("plus_di14")
    minus_di = indicators.get("minus_di14")

    last = candles[-1]

    candle_info = candle_description(last)
    body_ratio = candle_info["body_ratio"]

    confirmations = 0
    contradictions = 0
    evidence = []

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    if direction == "UP":

        if structure == "HH + HL":
            confirmations += 1
            evidence.append("bullish structure")

        elif structure == "LH + LL":
            contradictions += 1
            evidence.append("bearish structure")

    else:

        if structure == "LH + LL":
            confirmations += 1
            evidence.append("bearish structure")

        elif structure == "HH + HL":
            contradictions += 1
            evidence.append("bullish structure")

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if direction == "UP":

        if breakout == "bullish breakout":
            confirmations += 1
            evidence.append("bullish breakout")

        elif breakout == "bearish breakout":
            contradictions += 1

    else:

        if breakout == "bearish breakout":
            confirmations += 1
            evidence.append("bearish breakout")

        elif breakout == "bullish breakout":
            contradictions += 1

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if direction == "UP":

            if (
                last["close"] > ema9 > ema21
            ):
                confirmations += 1
                evidence.append("EMA bullish alignment")

            elif (
                last["close"] < ema9 < ema21
            ):
                contradictions += 1

        else:

            if (
                last["close"] < ema9 < ema21
            ):
                confirmations += 1
                evidence.append("EMA bearish alignment")

            elif (
                last["close"] > ema9 > ema21
            ):
                contradictions += 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi is not None:

        if direction == "UP":

            if rsi >= 52:
                confirmations += 1
                evidence.append("RSI bullish")

            elif rsi <= 45:
                contradictions += 1

        else:

            if rsi <= 48:
                confirmations += 1
                evidence.append("RSI bearish")

            elif rsi >= 55:
                contradictions += 1

    # --------------------------------------------------------
    # WILLIAMS
    # --------------------------------------------------------

    if williams is not None:

        if direction == "UP":

            if williams > -50:
                confirmations += 1
                evidence.append("Williams bullish")

            elif williams < -70:
                contradictions += 1

        else:

            if williams < -50:
                confirmations += 1
                evidence.append("Williams bearish")

            elif williams > -30:
                contradictions += 1

    # --------------------------------------------------------
    # DI
    # --------------------------------------------------------

    if (
        plus_di is not None
        and minus_di is not None
    ):

        if direction == "UP":

            if plus_di > minus_di:
                confirmations += 1
                evidence.append("DI bullish")

            else:
                contradictions += 1

        else:

            if minus_di > plus_di:
                confirmations += 1
                evidence.append("DI bearish")

            else:
                contradictions += 1

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx is not None:

        if adx >= 18:
            confirmations += 1
            evidence.append("ADX active")

        elif adx < 14:
            contradictions += 1

    # --------------------------------------------------------
    # CANDLE
    # --------------------------------------------------------

    if direction == "UP":

        if (
            candle_info["direction"] == "bullish"
            and body_ratio >= 0.35
        ):
            confirmations += 1
            evidence.append("bullish candle")

        elif (
            candle_info["direction"] == "bearish"
            and body_ratio >= 0.50
        ):
            contradictions += 1

    else:

        if (
            candle_info["direction"] == "bearish"
            and body_ratio >= 0.35
        ):
            confirmations += 1
            evidence.append("bearish candle")

        elif (
            candle_info["direction"] == "bullish"
            and body_ratio >= 0.50
        ):
            contradictions += 1

    # --------------------------------------------------------
    # EXHAUSTION PROTECTION
    # --------------------------------------------------------

    if len(candles) >= 5:

        last5 = candles[-5:]

        bullish_count = sum(
            1
            for c in last5
            if c["close"] > c["open"]
        )

        bearish_count = sum(
            1
            for c in last5
            if c["close"] < c["open"]
        )

        if (
            direction == "DOWN"
            and bearish_count >= 5
            and rsi is not None
            and rsi < 25
            and williams is not None
            and williams < -90
        ):
            return {
                "approved": False,
                "reason": "bearish exhaustion detected",
            }

        if (
            direction == "UP"
            and bullish_count >= 5
            and rsi is not None
            and rsi > 75
            and williams is not None
            and williams > -10
        ):
            return {
                "approved": False,
                "reason": "bullish exhaustion detected",
            }

    # --------------------------------------------------------
    # FINAL FILTER
    # --------------------------------------------------------

    if confirmations < 4:
        return {
            "approved": False,
            "reason": f"only {confirmations} confirmations",
        }

    if contradictions >= 2:
        return {
            "approved": False,
            "reason": f"{contradictions} contradictions",
        }

    return {
        "approved": True,
        "direction": direction,
        "selected_score": selected_score,
        "score_difference": score_difference,
        "confidence": round(confidence),
        "confirmations": confirmations,
        "contradictions": contradictions,
        "evidence": evidence,
        "reason": " / ".join(evidence[:5]),
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

GEMINI_PROMPT = """
You are the main technical-analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied MT4 OHLC candles and calculated indicators.

Do NOT use internet data.
Do NOT invent candles.
Do NOT invent indicators.
Do NOT predict from future candles.
The last supplied candle is the MOST RECENT CLOSED candle.

The trading direction MUST be UP or DOWN.
Never return WAIT, HOLD, NEUTRAL or NO SIGNAL.

However, if the setup is weak, inconsistent, exhausted,
or lacks confluence, score it conservatively so the external
quality filter can reject it.

IMPORTANT PRIORITY:

1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle
7. EMA 9/21
8. RSI 14
9. Williams %R 14
10. Keltner
11. ADX / DI

SCORING MUST TOTAL EXACTLY 18:

Structure: 2
Breakout: 2
Liquidity: 1
Momentum: 2
Candle: 2
RSI: 1
Summary: 2
Oscillators: 3
Moving Averages: 3

UP_SCORE + DOWN_SCORE MUST EQUAL 18.

The final direction must correspond to the larger score.

Do NOT give 90%+ confidence unless there is exceptionally
strong multi-factor confluence.

Normal strong signals should generally stay below 90%.

Be conservative.

Return ONLY valid JSON.

Required JSON schema:

{
  "asset": "string",
  "timeframe": "M1",
  "direction": "UP",
  "confidence": 0,
  "up_score": 0,
  "down_score": 0,
  "structure": "string",
  "breakout": "string",
  "liquidity": "string",
  "momentum": "string",
  "candle": "string",
  "rsi": "string",
  "williams": "string",
  "ema": "string",
  "keltner": "string",
  "adx": "string",
  "reason": "short explanation"
}
"""


# ============================================================
# ANALYZE MT4 DATA
# ============================================================

async def analyze_mt4_data(market_data):
    symbol = normalize_symbol(
        market_data.get("symbol")
    )

    timeframe = normalize_timeframe(
        market_data.get("timeframe")
    )

    raw_candles = market_data.get(
        "candles",
        [],
    )

    closed_candles = get_closed_candles(
        raw_candles
    )

    if len(closed_candles) < 30:
        return None

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

    latest_candle = candle_description(
        analysis_candles[-1]
    )

    payload = {
        "asset": symbol,
        "timeframe": timeframe,
        "structure": structure,
        "breakout": breakout,
        "latest_candle": latest_candle,
        "indicators": indicators,
        "candles": analysis_candles,
    }

    prompt = (
        GEMINI_PROMPT
        + "\n\nMARKET DATA:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    try:

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""

        text = text.strip()

        if text.startswith("```"):
            text = re.sub(
                r"^```(?:json)?",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"```$",
                "",
                text,
            )

            text = text.strip()

        result = json.loads(text)

    except Exception as e:

        logger.exception(
            "Gemini analysis failed for %s %s",
            symbol,
            timeframe,
        )

        return None

    try:

        up_score = int(
            result.get("up_score", 0)
        )

        down_score = int(
            result.get("down_score", 0)
        )

        confidence = float(
            result.get("confidence", 0)
        )

    except Exception:

        return None

    total = up_score + down_score

    if total != 18:

        logger.warning(
            "%s %s invalid score total: %s",
            symbol,
            timeframe,
            total,
        )

        return None

    direction = (
        "UP"
        if up_score > down_score
        else "DOWN"
    )

    result["asset"] = symbol
    result["timeframe"] = timeframe
    result["direction"] = direction
    result["confidence"] = min(
        max(confidence, 0),
        89,
    )
    result["up_score"] = up_score
    result["down_score"] = down_score

    result["_closed_candles"] = analysis_candles
    result["_indicators"] = indicators

    return result


# ============================================================
# QUICK CANDIDATE SCORE
# ============================================================

def quick_candidate_score(market_data):
    """
    Cheap local filter.
    This is NOT the final signal.
    It only decides which pairs deserve Gemini analysis.
    """

    try:

        candles = get_closed_candles(
            market_data.get("candles", [])
        )

        if len(candles) < 40:
            return -999

        candles = candles[-100:]

        indicators = calculate_indicators(
            candles
        )

        score_up = 0
        score_down = 0

        structure = structure_analysis(
            candles
        )

        breakout = detect_recent_breakout(
            candles
        )

        ema9 = indicators.get("ema9")
        ema21 = indicators.get("ema21")
        rsi = indicators.get("rsi14")
        williams = indicators.get("williams_r14")
        adx = indicators.get("adx14")
        plus_di = indicators.get("plus_di14")
        minus_di = indicators.get("minus_di14")

        close = candles[-1]["close"]

        # Structure
        if structure == "HH + HL":
            score_up += 2

        elif structure == "LH + LL":
            score_down += 2

        # Breakout
        if breakout == "bullish breakout":
            score_up += 2

        elif breakout == "bearish breakout":
            score_down += 2

        # EMA
        if (
            ema9 is not None
            and ema21 is not None
        ):

            if close > ema9 > ema21:
                score_up += 2

            elif close < ema9 < ema21:
                score_down += 2

        # RSI
        if rsi is not None:

            if rsi >= 52:
                score_up += 1

            elif rsi <= 48:
                score_down += 1

        # Williams
        if williams is not None:

            if williams > -50:
                score_up += 1

            elif williams < -50:
                score_down += 1

        # DI
        if (
            plus_di is not None
            and minus_di is not None
        ):

            if plus_di > minus_di:
                score_up += 1

            elif minus_di > plus_di:
                score_down += 1

        # ADX
        if adx is not None and adx >= 18:
            if score_up >= score_down:
                score_up += 1
            else:
                score_down += 1

        directional_score = max(
            score_up,
            score_down,
        )

        difference = abs(
            score_up - score_down
        )

        freshness_bonus = max(
            0,
            3 - (
                get_data_age_seconds(
                    market_data
                ) / 60
            ),
        )

        return (
            directional_score * 10
            + difference * 3
            + freshness_bonus
        )

    except Exception:
        return -999


# ============================================================
# AUTO CANDIDATES
# ============================================================

def get_auto_candidates(
    excluded_symbols=None,
):
    excluded_symbols = {
        normalize_symbol(x)
        for x in (excluded_symbols or set())
    }

    candidates = []

    with mt4_data_lock:

        items = list(
            mt4_data.items()
        )

    for key, data in items:

        symbol = normalize_symbol(
            data.get("symbol")
        )

        timeframe = normalize_timeframe(
            data.get("timeframe")
        )

        if not symbol:
            continue

        if symbol in excluded_symbols:
            continue

        if timeframe not in AUTO_TIMEFRAMES:
            continue

        age = get_data_age_seconds(data)

        if age > AUTO_DATA_MAX_AGE_SECONDS:
            continue

        candles = data.get(
            "candles",
            [],
        )

        if not isinstance(candles, list):
            continue

        # 40 closed + 1 forming
        if len(candles) < 41:
            continue

        signal_key = (
            f"{symbol}:{timeframe}"
        )

        latest_candle_time = None

        try:
            latest_candle_time = str(
                candles[-1].get("time")
            )
        except Exception:
            pass

        # Don't analyze the exact same forming candle again.
        if (
            latest_candle_time
            and last_auto_signal.get(
                signal_key
            ) == latest_candle_time
        ):
            continue

        quick_score = quick_candidate_score(
            data
        )

        if quick_score < 0:
            continue

        candidates.append(
            {
                "key": signal_key,
                "symbol": symbol,
                "timeframe": timeframe,
                "data": data,
                "quick_score": quick_score,
                "latest_candle_time": latest_candle_time,
            }
        )

    candidates.sort(
        key=lambda x: (
            x["quick_score"],
            -get_data_age_seconds(
                x["data"]
            ),
        ),
        reverse=True,
    )

    return candidates[
        :AUTO_MAX_CANDIDATES
    ]


# ============================================================
# SIGNAL RANK
# ============================================================

def signal_rank(approved):
    quality = approved["quality"]

    return (
        quality.get("selected_score", 0),
        quality.get("score_difference", 0),
        quality.get("confidence", 0),
        quality.get("confirmations", 0),
        -quality.get("contradictions", 0),
    )


# ============================================================
# ANALYZE ONE AUTO CANDIDATE
# ============================================================

async def analyze_candidate(candidate):

    data = candidate["data"]

    result = await analyze_mt4_data(
        data
    )

    if not result:
        return None

    candles = result.get(
        "_closed_candles",
        [],
    )

    indicators = result.get(
        "_indicators",
        {},
    )

    quality = evaluate_signal_quality(
        result,
        candles,
        indicators,
    )

    if not quality.get("approved"):
        logger.info(
            "Rejected %s %s: %s",
            candidate["symbol"],
            candidate["timeframe"],
            quality.get("reason"),
        )

        return None

    entry_time = get_next_entry_time(
        candidate["timeframe"]
    )

    lead_seconds = (
        entry_time - now_algiers()
    ).total_seconds()

    if lead_seconds < MIN_ENTRY_LEAD_SECONDS:
        logger.info(
            "Rejected %s %s: entry lead %.1fs",
            candidate["symbol"],
            candidate["timeframe"],
            lead_seconds,
        )

        return None

    return {
        "candidate": candidate,
        "result": result,
        "quality": quality,
        "entry_time": entry_time,
        "lead_seconds": lead_seconds,
    }


# ============================================================
# FORMAT AUTO SIGNAL
# ============================================================

def format_mt4_signal(
    result,
    market_data,
    entry_time=None,
):
    symbol = normalize_symbol(
        result.get(
            "asset",
            market_data.get("symbol"),
        )
    )

    timeframe = normalize_timeframe(
        result.get(
            "timeframe",
            market_data.get("timeframe"),
        )
    )

    direction = str(
        result.get(
            "direction",
            "UP",
        )
    ).upper()

    closed_candles = result.get(
        "_closed_candles",
        [],
    )

    try:
        entry_price = float(
            market_data.get(
                "price",
                market_data.get(
                    "current_price",
                    closed_candles[-1]["close"]
                    if closed_candles
                    else 0,
                ),
            )
        )
    except Exception:
        entry_price = (
            closed_candles[-1]["close"]
            if closed_candles
            else 0
        )

    cancellation = calculate_cancellation(
        direction,
        closed_candles,
        entry_price,
    )

    if entry_time is None:
        entry_time = get_next_entry_time(
            timeframe
        )

    entry_clock = entry_time.strftime(
        "%H:%M:%S"
    )

    if direction == "UP":
        arrow = "🟢 UP"
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{cancellation}"
        )
    else:
        arrow = "🔴 DOWN"
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{cancellation}"
        )

    confidence = result.get(
        "confidence",
        0,
    )

    up_score = result.get(
        "up_score",
        0,
    )

    down_score = result.get(
        "down_score",
        0,
    )

    reason = str(
        result.get(
            "reason",
            "",
        )
    ).strip()

    if len(reason) > 180:
        reason = reason[:177] + "..."

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{arrow}\n"
        f"📈 Confidence: {round(float(confidence))}%\n"
        f"🎯 Score: {up_score}/18 UP | "
        f"{down_score}/18 DOWN\n"
        f"⏰ Entry: {entry_clock}\n"
        f"💰 Price: {entry_price}\n"
        f"{cancel_text}\n"
    )

    if reason:
        message += (
            f"\n🧠 {reason}\n"
        )

    message += (
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# ============================================================
# SEND AUTO SIGNAL
# ============================================================

async def send_auto_signal(approved):
    candidate = approved["candidate"]
    result = approved["result"]

    symbol = candidate["symbol"]
    timeframe = candidate["timeframe"]

    signal_key = (
        f"{symbol}:{timeframe}"
    )

    candles = candidate["data"].get(
        "candles",
        [],
    )

    latest_candle_time = None

    if candles:
        try:
            latest_candle_time = str(
                candles[-1].get("time")
            )
        except Exception:
            pass

    # Recalculate entry time immediately before send.
    entry_time = get_next_entry_time(
        timeframe
    )

    lead_seconds = (
        entry_time - now_algiers()
    ).total_seconds()

    if lead_seconds < MIN_ENTRY_LEAD_SECONDS:
        logger.info(
            "Send cancelled: %s %s has only %.1fs lead",
            symbol,
            timeframe,
            lead_seconds,
        )

        return False, None

    # Duplicate protection again.
    if (
        latest_candle_time
        and last_auto_signal.get(
            signal_key
        ) == latest_candle_time
    ):
        logger.info(
            "Duplicate blocked: %s",
            signal_key,
        )

        return False, None

    message = format_mt4_signal(
        result,
        candidate["data"],
        entry_time=entry_time,
    )

    if telegram_bot is None:
        logger.error(
            "Telegram bot is not ready"
        )

        return False, None

    try:

        await telegram_bot.send_message(
            chat_id=OWNER_ID,
            text=message,
        )

        # IMPORTANT:
        # Mark as sent ONLY after successful Telegram send.
        if latest_candle_time:
            last_auto_signal[
                signal_key
            ] = latest_candle_time

        sent_at = now_algiers()

        logger.info(
            "AUTO SIGNAL SENT: %s %s | %s | entry=%s",
            symbol,
            timeframe,
            result.get("direction"),
            entry_time.strftime("%H:%M:%S"),
        )

        return True, sent_at

    except Exception:

        logger.exception(
            "Failed to send auto signal"
        )

        return False, None


# ============================================================
# FIRST SIGNAL CYCLE
# ============================================================

async def run_first_signal_cycle():

    logger.info(
        "========== AUTO CYCLE START =========="
    )

    candidates = get_auto_candidates()

    if not candidates:

        logger.info(
            "No valid MT4 candidates"
        )

        return None, None

    logger.info(
        "Candidates: %s",
        ", ".join(
            f"{c['symbol']}:{c['timeframe']}"
            for c in candidates
        ),
    )

    approved_signals = []

    for candidate in candidates:

        try:

            approved = await analyze_candidate(
                candidate
            )

            if approved:
                approved_signals.append(
                    approved
                )

        except Exception:

            logger.exception(
                "Candidate analysis failed: %s %s",
                candidate["symbol"],
                candidate["timeframe"],
            )

    if not approved_signals:

        logger.info(
            "No approved signal in first cycle"
        )

        return None, None

    approved_signals.sort(
        key=signal_rank,
        reverse=True,
    )

    best = approved_signals[0]

    sent, sent_at = await send_auto_signal(
        best
    )

    if not sent:
        return None, None

    return (
        best["candidate"]["symbol"],
        sent_at,
    )


# ============================================================
# SECOND SIGNAL
# ============================================================

async def run_second_signal(
    first_symbol,
):
    """
    Second signal must be a DIFFERENT pair.
    """

    logger.info(
        "========== SECOND SIGNAL SEARCH =========="
    )

    candidates = get_auto_candidates(
        excluded_symbols={
            first_symbol
        }
    )

    if not candidates:

        logger.info(
            "No different pair available "
            "for second signal"
        )

        return False

    approved_signals = []

    for candidate in candidates:

        try:

            approved = await analyze_candidate(
                candidate
            )

            if approved:
                approved_signals.append(
                    approved
                )

        except Exception:

            logger.exception(
                "Second candidate failed: %s %s",
                candidate["symbol"],
                candidate["timeframe"],
            )

    if not approved_signals:

        logger.info(
            "No strong second signal"
        )

        return False

    approved_signals.sort(
        key=signal_rank,
        reverse=True,
    )

    for approved in approved_signals:

        sent, _ = await send_auto_signal(
            approved
        )

        if sent:
            return True

    return False


# ============================================================
# WAIT FOR NEXT 3-MINUTE BOUNDARY
# ============================================================

async def wait_until_next_cycle():
    while True:

        now = now_algiers()

        seconds_into_minute = (
            now.second
            + now.microsecond / 1_000_000
        )

        current_minute = now.minute

        next_block = (
            (
                current_minute
                // AUTO_CYCLE_MINUTES
            )
            + 1
        ) * AUTO_CYCLE_MINUTES

        if next_block >= 60:

            next_time = (
                now.replace(
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                + timedelta(hours=1)
            )

        else:

            next_time = now.replace(
                minute=next_block,
                second=0,
                microsecond=0,
            )

        wait_seconds = (
            next_time - now
        ).total_seconds()

        if wait_seconds > 0:
            await asyncio.sleep(
                wait_seconds
            )

        return


# ============================================================
# AUTO CYCLE LOOP
# ============================================================

async def auto_signal_cycle_loop():

    global auto_cycle_running

    logger.info(
        "ZinoProSignalAI auto cycle started"
    )

    while True:

        try:

            await wait_until_next_cycle()

            if auto_cycle_running:
                logger.warning(
                    "Previous auto cycle still running"
                )
                continue

            auto_cycle_running = True

            try:

                first_symbol, sent_at = (
                    await run_first_signal_cycle()
                )

                if first_symbol and sent_at:

                    # Wait exactly 2 minutes
                    # from successful first send.
                    target_time = (
                        sent_at
                        + timedelta(
                            seconds=SECOND_SIGNAL_DELAY_SECONDS
                        )
                    )

                    wait_seconds = (
                        target_time
                        - now_algiers()
                    ).total_seconds()

                    if wait_seconds > 0:
                        logger.info(
                            "Second signal search in %.1f seconds",
                            wait_seconds,
                        )

                        await asyncio.sleep(
                            wait_seconds
                        )

                    if (
                        MAX_SIGNALS_PER_CYCLE >= 2
                    ):

                        await run_second_signal(
                            first_symbol
                        )

            finally:

                auto_cycle_running = False

                logger.info(
                    "========== AUTO CYCLE END =========="
                )

        except asyncio.CancelledError:

            logger.info(
                "Auto cycle task cancelled"
            )

            break

        except Exception:

            logger.exception(
                "AUTO CYCLE ERROR"
            )

            auto_cycle_running = False

            await asyncio.sleep(5)


# ============================================================
# HTTP HEALTH + MT4 SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format,
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
                    "auto_cycle": True,
                    "cycle_minutes": 3,
                },
            )

            return

        if path == "/health":

            self.send_json(
                200,
                {
                    "status": "healthy",
                    "service": "ZinoProSignalAI",
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

        path = urlparse(
            self.path
        ).path

        if path != "/mt4":

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

        received_key = (
            self.headers.get(
                "X-API-Key",
                ""
            )
        )

        if received_key != MT4_API_KEY:

            self.send_json(
                401,
                {
                    "error": "unauthorized"
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

            self.send_json(
                400,
                {
                    "error": "invalid content length"
                },
            )

            return

        if content_length <= 0:

            self.send_json(
                400,
                {
                    "error": "empty body"
                },
            )

            return

        if content_length > 2_000_000:

            self.send_json(
                413,
                {
                    "error": "payload too large"
                },
            )

            return

        try:

            body = self.rfile.read(
                content_length
            )

            data = json.loads(
                body.decode("utf-8")
            )

        except Exception:

            self.send_json(
                400,
                {
                    "error": "invalid JSON"
                },
            )

            return

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        symbol = normalize_symbol(
            data.get("symbol")
        )

        timeframe = normalize_timeframe(
            data.get("timeframe")
        )

        candles = data.get(
            "candles",
            [],
        )

        if not symbol:

            self.send_json(
                400,
                {
                    "error": "symbol missing"
                },
            )

            return

        if not isinstance(
            candles,
            list,
        ):

            self.send_json(
                400,
                {
                    "error": "candles must be a list"
                },
            )

            return

        # Keep latest 200 candles
        candles = candles[-200:]

        data["symbol"] = symbol
        data["timeframe"] = timeframe
        data["candles"] = candles
        data["received_at"] = (
            now_algiers().isoformat()
        )

        key = (
            f"{symbol}:{timeframe}"
        )

        # ----------------------------------------------------
        # NEW CANDLE DETECTION
        # ----------------------------------------------------

        latest_candle = None

        if candles:

            try:
                latest_candle = str(
                    candles[-1].get("time")
                )
            except Exception:
                latest_candle = None

        previous_candle = last_auto_candle.get(
            key
        )

        new_candle = (
            latest_candle is not None
            and latest_candle != previous_candle
        )

        if latest_candle is not None:

            last_auto_candle[
                key
            ] = latest_candle

        # ----------------------------------------------------
        # STORE ONLY
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # We DO NOT trigger Gemini here.
        #
        # The centralized 3-minute cycle does that.
        #

        with mt4_data_lock:

            mt4_data[key] = data

        logger.info(
            "MT4 data received: %s | candles=%s | new_candle=%s",
            key,
            len(candles),
            new_candle,
        )

        self.send_json(
            200,
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
                "new_candle": new_candle,
                "auto_cycle": True,
                "cycle_minutes": AUTO_CYCLE_MINUTES,
            },
        )


# ============================================================
# HTTP SERVER
# ============================================================

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
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    text = (
        "🎓 ZinoProSignalAI\n\n"
        "✅ MT4 Auto Cycle فعال\n"
        "⏱️ دورة جديدة كل 3 دقائق\n"
        "🎯 أقصى حد: إشارتان في الدورة\n"
        "🔄 الإشارة الثانية بعد دقيقتين\n"
        "📊 الزوج الثاني مختلف عن الأول\n"
        "🧠 Quality Filter فعال\n"
        "🚫 الإشارات الضعيفة لا يتم إرسالها\n"
        "⏳ أقل وقت قبل الدخول: 20 ثانية\n"
        "🕒 التوقيت: Africa/Algiers\n\n"
        "الأوامر:\n"
        "/stats - الإحصائيات\n"
        "/win - تسجيل ربح\n"
        "/loss - تسجيل خسارة\n"
        "/reset - تصفير الإحصائيات\n"
        "/mt4status - حالة بيانات MT4\n"
        "/analyze SYMBOL M1 - تحليل يدوي"
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    wins = stats["wins"]
    losses = stats["losses"]

    total = wins + losses

    if total:
        winrate = (
            wins / total
        ) * 100
    else:
        winrate = 0

    text = (
        "📊 ZinoProSignalAI Stats\n\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {winrate:.1f}%"
    )

    await update.message.reply_text(
        text
    )


# ============================================================
# WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    stats["wins"] += 1

    await update.message.reply_text(
        f"🟢 WIN recorded\n"
        f"Total Wins: {stats['wins']}"
    )


# ============================================================
# LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    stats["losses"] += 1

    await update.message.reply_text(
        f"🔴 LOSS recorded\n"
        f"Total Losses: {stats['losses']}"
    )


# ============================================================
# RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    stats["wins"] = 0
    stats["losses"] = 0

    await update.message.reply_text(
        "♻️ Statistics reset."
    )


# ============================================================
# MT4 STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    with mt4_data_lock:

        items = list(
            mt4_data.items()
        )

    if not items:

        await update.message.reply_text(
            "❌ لا توجد بيانات MT4 حتى الآن."
        )

        return

    lines = [
        "📡 MT4 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for key, data in sorted(items):

        candles = data.get(
            "candles",
            [],
        )

        age = get_data_age_seconds(
            data
        )

        status = (
            "🟢"
            if age <= AUTO_DATA_MAX_AGE_SECONDS
            else "🔴"
        )

        lines.append(
            f"{status} {key} | "
            f"{len(candles)} candles | "
            f"{age:.0f}s"
        )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"Auto cycle: every "
        f"{AUTO_CYCLE_MINUTES} minutes"
    )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# MANUAL ANALYZE
# ============================================================

async def manual_analyze(
    update,
    symbol,
    timeframe,
):

    if not is_owner(update):
        return

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)

    key = (
        f"{symbol}:{timeframe}"
    )

    with mt4_data_lock:

        market_data = mt4_data.get(
            key
        )

    if not market_data:

        await update.message.reply_text(
            f"❌ لا توجد بيانات MT4 لـ "
            f"{symbol} {timeframe}"
        )

        return

    await update.message.reply_text(
        f"🧠 جاري تحليل {symbol} {timeframe}..."
    )

    result = await analyze_mt4_data(
        market_data
    )

    if not result:

        await update.message.reply_text(
            "❌ فشل التحليل."
        )

        return

    candles = result.get(
        "_closed_candles",
        [],
    )

    indicators = result.get(
        "_indicators",
        {},
    )

    quality = evaluate_signal_quality(
        result,
        candles,
        indicators,
    )

    if not quality.get("approved"):

        await update.message.reply_text(
            "🚫 الإشارة مرفوضة من Quality Filter\n\n"
            f"السبب: {quality.get('reason')}"
        )

        return

    entry_time = get_next_entry_time(
        timeframe
    )

    message = format_mt4_signal(
        result,
        market_data,
        entry_time=entry_time,
    )

    await update.message.reply_text(
        message
    )


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not context.args:

        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURUSD M1"
        )

        return

    symbol = context.args[0]

    timeframe = (
        context.args[1]
        if len(context.args) > 1
        else "M1"
    )

    await manual_analyze(
        update,
        symbol,
        timeframe,
    )


# ============================================================
# SCREENSHOT PROMPT
# ============================================================

SCREENSHOT_PROMPT = """
You are ZinoProSignalAI chart-analysis engine.

Analyze the supplied trading chart image.

Return ONLY valid JSON.

The final direction MUST be UP or DOWN.
Never return WAIT, NO SIGNAL or NEUTRAL.

Use this scoring system exactly:

Structure: 2
Breakout: 2
Liquidity: 1
Momentum: 2
Candle: 2
RSI: 1
Summary: 2
Oscillators: 3
Moving Averages: 3

UP_SCORE + DOWN_SCORE = 18.

Prioritize:

Price Action
Structure
Breakout / Retest
Liquidity
Momentum
Candle
EMA 9/21
RSI 14
Williams %R 14
Keltner
ADX/DI

Do not invent indicators that are not visible.

Do not give 90%+ confidence without exceptional confluence.

Return:

{
 "asset": "string",
 "timeframe": "M1",
 "direction": "UP",
 "confidence": 0,
 "up_score": 0,
 "down_score": 0,
 "reason": "short reason"
}
"""


# ============================================================
# SCREENSHOT ANALYSIS
# ============================================================

async def analyze_chart(
    image_bytes,
):

    try:

        response = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                SCREENSHOT_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""

        text = text.strip()

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"```$",
                "",
                text,
            )

            text = text.strip()

        result = json.loads(text)

        return result

    except Exception:

        logger.exception(
            "Screenshot analysis failed"
        )

        return None


# ============================================================
# SCREENSHOT FORMAT
# ============================================================

def format_screenshot_signal(
    result
):

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

    direction = str(
        result.get(
            "direction",
            "UP",
        )
    ).upper()

    confidence = result.get(
        "confidence",
        0,
    )

    up_score = result.get(
        "up_score",
        0,
    )

    down_score = result.get(
        "down_score",
        0,
    )

    reason = str(
        result.get(
            "reason",
            "",
        )
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    if direction == "UP":
        arrow = "🟢 UP"
    else:
        arrow = "🔴 DOWN"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {asset} | {timeframe}\n\n"
        f"{arrow}\n"
        f"📈 Confidence: {round(float(confidence))}%\n"
        f"🎯 Score: {up_score}/18 UP | "
        f"{down_score}/18 DOWN\n"
        f"⏰ Entry: "
        f"{entry_time.strftime('%H:%M:%S')}\n"
        f"🧠 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not update.message:
        return

    if not update.message.photo:
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

    except Exception:

        logger.exception(
            "Failed downloading Telegram image"
        )

        await update.message.reply_text(
            "❌ فشل تحميل الصورة."
        )

        return

    await update.message.reply_text(
        "🧠 جاري تحليل الشارت..."
    )

    result = await analyze_chart(
        image_bytes
    )

    if not result:

        await update.message.reply_text(
            "❌ فشل تحليل الصورة."
        )

        return

    message = format_screenshot_signal(
        result
    )

    await update.message.reply_text(
        message
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not update.message:
        return

    text = (
        update.message.text or ""
    ).strip()

    if not text:
        return

    # Example:
    # EURUSD M1
    # GBPCHF M3
    match = re.match(
        r"^([A-Za-z0-9._/-]+)\s+([A-Za-z0-9]+)$",
        text,
    )

    if match:

        symbol = match.group(1)
        timeframe = match.group(2)

        await manual_analyze(
            update,
            symbol,
            timeframe,
        )

        return


# ============================================================
# POST INIT
# ============================================================

async def post_init(
    application: Application,
):

    global telegram_loop
    global telegram_bot
    global auto_cycle_lock

    telegram_loop = asyncio.get_running_loop()

    telegram_bot = application.bot

    auto_cycle_lock = asyncio.Lock()

    logger.info(
        "Telegram loop initialized"
    )

    # Start centralized auto cycle.
    application.create_task(
        auto_signal_cycle_loop(),
        name="zino-auto-signal-cycle",
    )

    logger.info(
        "3-minute auto cycle task started"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "======================================"
    )

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Auto cycle: every %s minutes",
        AUTO_CYCLE_MINUTES,
    )

    logger.info(
        "Max signals per cycle: %s",
        MAX_SIGNALS_PER_CYCLE,
    )

    logger.info(
        "Second signal delay: %s seconds",
        SECOND_SIGNAL_DELAY_SECONDS,
    )

    logger.info(
        "Minimum entry lead: %s seconds",
        MIN_ENTRY_LEAD_SECONDS,
    )

    logger.info(
        "Auto timeframes: %s",
        ", ".join(
            sorted(AUTO_TIMEFRAMES)
        ),
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "======================================"
    )

    # --------------------------------------------------------
    # HTTP SERVER
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="http-health-server",
    )

    http_thread.start()

    # --------------------------------------------------------
    # TELEGRAM APPLICATION
    # --------------------------------------------------------

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
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

    # Photos
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    # Text
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
