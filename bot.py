import os
import json
import logging
import threading
import asyncio
import time
import urllib.request
import urllib.parse
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
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = os.getenv("OWNER_ID", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

# يدعم MT4_API_KEY وكذلك ZINO_API_KEY
MT4_API_KEY = (
    os.getenv("MT4_API_KEY")
    or os.getenv("ZINO_API_KEY")
    or ""
).strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")


# ============================================================
# M1 CONFIG
# ============================================================

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40

HISTORY_DISPLAY_COUNT = 10

# حماية إضافية: لا توجد إشارة أخرى خلال هذه المدة
SIGNAL_COOLDOWN_SECONDS = 120

# نفس setup لا يعاد مباشرة
SETUP_REPEAT_BLOCK_SECONDS = 360

# Recovery واحدة فقط
RECOVERY_LIMIT = 1

HISTORY_FILE = "signal_history.json"

# فحص البيانات كل دقيقة
AUTO_ANALYSIS_INTERVAL_SECONDS = 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        logger.info("Gemini client initialized")

    except Exception as e:
        logger.exception(
            "Gemini initialization failed: %s",
            e
        )

else:
    logger.warning("GEMINI_API_KEY is missing")


# ============================================================
# GLOBAL STATE
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

telegram_application = None

last_signal_sent_at = 0.0

signal_send_lock = threading.Lock()

cycle_lock = threading.Lock()

analysis_lock = threading.Lock()

recent_setups = {}

setup_memory_lock = threading.Lock()


# ============================================================
# ACTIVE CYCLE
#
# أهم تغيير:
#
# pending_result = True
# معناها توجد صفقة مرسلة وننتظر /win أو /loss
#
# recovery_ready = True
# معناها BASE خسرت ونسمح بإرسال Recovery واحدة فقط
# ============================================================

active_cycle = {

    "active": False,

    "symbol": None,

    "timeframe": None,

    "trade_type": None,

    "direction": None,

    "recovery_used": False,

    "recovery_ready": False,

    "pending_result": False,

    "last_trade_time": 0.0,

    "trade_number": 0,
}


# ============================================================
# STATS
# ============================================================

stats_data = {

    "wins": 0,
    "losses": 0,

    "base_wins": 0,
    "base_losses": 0,

    "recovery_wins": 0,
    "recovery_losses": 0,
}


# ============================================================
# TRADE HISTORY
# ============================================================

history_lock = threading.Lock()

trade_history = []

current_trade_id = None


# ============================================================
# HISTORY
# ============================================================

def load_history():

    global trade_history

    try:

        if not os.path.exists(HISTORY_FILE):

            trade_history = []

            return

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            trade_history = data

        else:

            trade_history = []

        logger.info(
            "Trade history loaded: %s records",
            len(trade_history)
        )

    except Exception as e:

        logger.exception(
            "Failed to load trade history: %s",
            e
        )

        trade_history = []


def save_history():

    try:

        temp_file = HISTORY_FILE + ".tmp"

        with history_lock:

            data = list(trade_history)

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        os.replace(
            temp_file,
            HISTORY_FILE
        )

    except Exception as e:

        logger.exception(
            "Failed to save trade history: %s",
            e
        )


# ============================================================
# HELPERS
# ============================================================

def owner_id_int():

    try:
        return int(OWNER_ID)

    except Exception:
        return None


def is_owner(update: Update) -> bool:

    oid = owner_id_int()

    if oid is None:
        return False

    user = update.effective_user

    if user is None:
        return False

    return user.id == oid


def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        return float(value)

    except Exception:

        return default


def normalize_timeframe(value):

    if value is None:
        return ""

    value = str(value).upper().strip()

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
        "1H": "H1",
        "H1": "H1",

        "240": "H4",
        "4H": "H4",
        "H4": "H4",
    }

    return aliases.get(value, value)


def timeframe_minutes(timeframe):

    values = {

        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
    }

    return values.get(
        normalize_timeframe(timeframe),
        1
    )


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(candle):

    if not isinstance(candle, dict):
        return None

    timestamp = (
        candle.get("time")
        or candle.get("timestamp")
        or candle.get("datetime")
        or candle.get("date")
    )

    open_price = (
        candle.get("open")
        if candle.get("open") is not None
        else candle.get("o")
    )

    high_price = (
        candle.get("high")
        if candle.get("high") is not None
        else candle.get("h")
    )

    low_price = (
        candle.get("low")
        if candle.get("low") is not None
        else candle.get("l")
    )

    close_price = (
        candle.get("close")
        if candle.get("close") is not None
        else candle.get("c")
    )

    volume = (
        candle.get("volume")
        if candle.get("volume") is not None
        else candle.get("v", 0)
    )

    o = safe_float(open_price)
    h = safe_float(high_price)
    l = safe_float(low_price)
    c = safe_float(close_price)
    v = safe_float(volume, 0)

    if (
        o is None
        or h is None
        or l is None
        or c is None
    ):
        return None

    if (
        o <= 0
        or h <= 0
        or l <= 0
        or c <= 0
    ):
        return None

    return {

        "time": timestamp,

        "open": o,
        "high": h,
        "low": l,
        "close": c,

        "volume": v,
    }


def normalize_candles(candles):

    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:

        normalized = normalize_candle(candle)

        if normalized:
            result.append(normalized)

    result.sort(
        key=lambda x: (
            safe_float(
                x.get("time"),
                0
            ) or 0
        )
    )

    return result


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):

    if not values or len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    current = (
        sum(values[:period])
        / period
    )

    for price in values[period:]:

        current = (
            (price - current)
            * multiplier
        ) + current

    return current


def rsi(values, period=14):

    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):

        diff = (
            values[i]
            - values[i - 1]
        )

        if diff >= 0:

            gains.append(diff)
            losses.append(0)

        else:

            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(
        period + 1,
        len(values)
    ):

        diff = (
            values[i]
            - values[i - 1]
        )

        gain = max(diff, 0)
        loss = max(-diff, 0)

        avg_gain = (
            (
                avg_gain * (period - 1)
            ) + gain
        ) / period

        avg_loss = (
            (
                avg_loss * (period - 1)
            ) + loss
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def williams_r(candles, period=14):

    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(
        x["high"]
        for x in recent
    )

    lowest = min(
        x["low"]
        for x in recent
    )

    if highest == lowest:
        return -50.0

    close = recent[-1]["close"]

    return (
        (
            highest - close
        )
        /
        (
            highest - lowest
        )
    ) * -100


def true_ranges(candles):

    if len(candles) < 2:
        return []

    result = []

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

        result.append(tr)

    return result


def atr(candles, period=10):

    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(
        trs[-period:]
    ) / period


def adx_di(candles, period=14):

    if len(candles) < period + 2:

        return (
            None,
            None,
            None
        )

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

        return (
            None,
            None,
            None
        )

    tr_avg = (
        sum(trs[-period:])
        / period
    )

    plus_avg = (
        sum(plus_dm[-period:])
        / period
    )

    minus_avg = (
        sum(minus_dm[-period:])
        / period
    )

    if tr_avg == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    plus_di = (
        100
        * plus_avg
        / tr_avg
    )

    minus_di = (
        100
        * minus_avg
        / tr_avg
    )

    denominator = (
        plus_di
        + minus_di
    )

    if denominator == 0:
        dx = 0

    else:

        dx = (
            100
            * abs(
                plus_di
                - minus_di
            )
            / denominator
        )

    return (
        dx,
        plus_di,
        minus_di
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(candles):

    if len(candles) < 8:
        return "UNKNOWN"

    recent = candles[-8:]

    first_high = max(
        x["high"]
        for x in recent[:4]
    )

    second_high = max(
        x["high"]
        for x in recent[4:]
    )

    first_low = min(
        x["low"]
        for x in recent[:4]
    )

    second_low = min(
        x["low"]
        for x in recent[4:]
    )

    if (
        second_high > first_high
        and second_low > first_low
    ):
        return "BULLISH"

    if (
        second_high < first_high
        and second_low < first_low
    ):
        return "BEARISH"

    return "RANGE"


def breakout_state(candles):

    if len(candles) < 10:
        return "NONE"

    previous = candles[-9:-1]

    previous_high = max(
        x["high"]
        for x in previous
    )

    previous_low = min(
        x["low"]
        for x in previous
    )

    last_close = candles[-1]["close"]

    if last_close > previous_high:
        return "UP_BREAKOUT"

    if last_close < previous_low:
        return "DOWN_BREAKOUT"

    return "NONE"


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_technical_snapshot(candles):

    closes = [
        x["close"]
        for x in candles
    ]

    current = candles[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    current_rsi = rsi(closes, 14)

    current_wr = williams_r(
        candles,
        14
    )

    current_atr = atr(
        candles,
        10
    )

    adx, plus_di, minus_di = adx_di(
        candles,
        14
    )

    keltner_mid = ema(
        closes,
        20
    )

    keltner_atr = atr(
        candles,
        10
    )

    keltner_upper = None
    keltner_lower = None

    if (
        keltner_mid is not None
        and keltner_atr is not None
    ):

        keltner_upper = (
            keltner_mid
            + keltner_atr * 5
        )

        keltner_lower = (
            keltner_mid
            - keltner_atr * 5
        )

    recent8 = candles[-8:]

    return {

        "price": current["close"],
        "open": current["open"],
        "high": current["high"],
        "low": current["low"],

        "ema9": ema9,
        "ema21": ema21,

        "rsi14": current_rsi,
        "williams_r14": current_wr,

        "atr10": current_atr,

        "adx14": adx,
        "plus_di14": plus_di,
        "minus_di14": minus_di,

        "keltner_mid": keltner_mid,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,

        "structure": market_structure(candles),
        "breakout": breakout_state(candles),

        "recent_low": min(
            x["low"]
            for x in recent8
        ),

        "recent_high": max(
            x["high"]
            for x in recent8
        ),
    }


# ============================================================
# PRE SCORE
# ============================================================

def directional_pre_score(candles):

    if len(candles) < MIN_CLOSED_CANDLES:

        return {
            "score": -999,
            "direction": "DOWN",
            "up": 0,
            "down": 0,
        }

    snapshot = build_technical_snapshot(
        candles
    )

    up = 0
    down = 0

    price = snapshot["price"]

    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    if (
        ema9 is not None
        and ema21 is not None
    ):

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

    adx_value = snapshot["adx14"]
    plus_di = snapshot["plus_di14"]
    minus_di = snapshot["minus_di14"]

    if (
        adx_value is not None
        and adx_value >= 20
        and plus_di is not None
        and minus_di is not None
    ):

        if plus_di > minus_di:
            up += 2

        elif minus_di > plus_di:
            down += 2

    rsi_value = snapshot["rsi14"]

    if rsi_value is not None:

        if 50 < rsi_value < 70:
            up += 1

        elif 30 < rsi_value < 50:
            down += 1

    last = candles[-1]

    if last["close"] > last["open"]:
        up += 1

    elif last["close"] < last["open"]:
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

        "score": max(up, down),

        "direction": direction,

        "up": up,

        "down": down,
    }


# ============================================================
# BEST PAIR
# ============================================================

def choose_best_pair():

    with mt4_lock:

        candidates = []

        for symbol, timeframes in mt4_data.items():

            if not isinstance(
                timeframes,
                dict
            ):
                continue

            candles = timeframes.get(
                ANALYSIS_TIMEFRAME
            )

            if not candles:
                continue

            closed = candles[:-1]

            if len(closed) < MIN_CLOSED_CANDLES:
                continue

            pre = directional_pre_score(
                closed
            )

            candidates.append({

                "symbol": symbol,

                "timeframe": ANALYSIS_TIMEFRAME,

                "score": pre["score"],

                "direction": pre["direction"],

                "up": pre["up"],

                "down": pre["down"],
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["score"],
            abs(
                x["up"]
                - x["down"]
            ),
        ),
        reverse=True
    )

    cycle = get_active_cycle()

    if cycle["active"]:

        for candidate in candidates:

            if candidate["symbol"] == cycle["symbol"]:
                return candidate

        return None

    return candidates[0]


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles
):

    snapshot = build_technical_snapshot(
        candles
    )

    pre_score = directional_pre_score(
        candles
    )

    recent = candles[-30:]

    payload = {

        "symbol": symbol,

        "timeframe": timeframe,

        "candles_count": len(candles),

        "technical_snapshot": snapshot,

        "pre_analysis": pre_score,

        "recent_candles": recent,
    }

    return f"""
You are the technical analysis engine for
ZinoProSignalAI.

Analyze ONLY the supplied closed-candle data.

TIMEFRAME:
{timeframe}

SYMBOL:
{symbol}

IMPORTANT RULES:

1. Never invent market data.
2. Never use external prices.
3. Never assume indicators that are not supplied.
4. Direction MUST be exactly UP or DOWN.
5. Never return WAIT.
6. Never return NO SIGNAL.
7. Never return NEUTRAL.
8. Confidence must reflect actual evidence.
9. Do not use 90%+ confidence unless the evidence is exceptionally strong.
10. Confidence is NOT a guarantee.
11. Price Action has the highest priority.
12. Market Structure comes next.
13. Breakout / Retest comes next.
14. Liquidity comes next.
15. Momentum comes next.
16. Candle behavior comes next.
17. EMA 9/21 comes next.
18. RSI comes next.
19. Williams %R comes next.
20. Keltner comes next.
21. ADX/DI is supporting evidence only.

Look for multiple independent pieces of confluence.

Check contradictions carefully.

SCORING:

The total score must be exactly 18.

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

The larger score should normally match the selected direction.

Do not manufacture scores from unavailable information.

Return JSON ONLY.

Required JSON:

{{
  "signal": true,
  "direction": "UP",
  "confidence": 75,
  "up_score": 13,
  "down_score": 5,
  "reason": "Short factual reason based only on supplied data.",
  "cancellation_reason": "Cancel if the next closed candle invalidates the directional structure."
}}

MARKET DATA:

{json.dumps(
    payload,
    ensure_ascii=False
)}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles
):

    if gemini_client is None:

        raise RuntimeError(
            "Gemini client is not initialized"
        )

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles
    )

    response = (
        gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.10,
                response_mime_type="application/json",
            ),
        )
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:

        raise RuntimeError(
            "Gemini returned empty response"
        )

    text = text.strip()

    if text.startswith("```"):

        text = text.replace(
            "```json",
            ""
        )

        text = text.replace(
            "```",
            ""
        )

        text = text.strip()

    return json.loads(text)


# ============================================================
# VALIDATE RESULT
# ============================================================

def ensure_directional_signal(
    result,
    candles
):

    if not isinstance(result, dict):
        result = {}

    direction = str(
        result.get(
            "direction",
            ""
        )
    ).upper().strip()

    up_score = int(
        safe_float(
            result.get("up_score"),
            0
        ) or 0
    )

    down_score = int(
        safe_float(
            result.get("down_score"),
            0
        ) or 0
    )

    up_score = max(
        0,
        min(18, up_score)
    )

    down_score = max(
        0,
        min(18, down_score)
    )

    total = up_score + down_score

    if total != 18:

        if total == 0:

            pre = directional_pre_score(
                candles
            )

            if pre["up"] >= pre["down"]:

                up_score = 18
                down_score = 0

            else:

                up_score = 0
                down_score = 18

        elif total < 18:

            missing = 18 - total

            if up_score >= down_score:
                up_score += missing
            else:
                down_score += missing

        else:

            excess = total - 18

            if up_score >= down_score:

                up_score = max(
                    0,
                    up_score - excess
                )

            else:

                down_score = max(
                    0,
                    down_score - excess
                )

    if up_score > down_score:

        direction = "UP"

    elif down_score > up_score:

        direction = "DOWN"

    else:

        pre = directional_pre_score(
            candles
        )

        direction = pre["direction"]

    confidence = safe_float(
        result.get("confidence"),
        50
    )

    confidence = max(
        1,
        min(
            99,
            int(confidence)
        )
    )

    if confidence >= 90:

        score_gap = abs(
            up_score
            - down_score
        )

        if score_gap < 8:
            confidence = 89

    reason = str(
        result.get("reason")
        or
        "Price action and technical confluence indicate the selected direction."
    ).strip()

    cancellation_reason = str(
        result.get("cancellation_reason")
        or
        "Cancel if the next closed candle invalidates the current structure."
    ).strip()

    return {

        "signal": True,

        "direction": direction,

        "confidence": confidence,

        "up_score": up_score,

        "down_score": down_score,

        "reason": reason,

        "cancellation_reason":
            cancellation_reason,
    }


# ============================================================
# ENTRY
# ============================================================

def get_entry_delay_minutes(timeframe):

    return timeframe_minutes(timeframe)


def get_entry_price(candles):

    return candles[-1]["close"]


def get_cancellation_level(
    candles,
    direction
):

    recent = candles[-8:]

    if direction == "UP":

        return min(
            x["low"]
            for x in recent
        )

    return max(
        x["high"]
        for x in recent
    )


def format_price(price):

    if price is None:
        return "N/A"

    if abs(price) >= 1:
        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# SETUP
# ============================================================

def candle_identity(candle):

    if not candle:
        return "unknown"

    timestamp = candle.get("time")

    if timestamp is not None:
        return str(timestamp)

    return (
        f"{candle.get('open')}_"
        f"{candle.get('high')}_"
        f"{candle.get('low')}_"
        f"{candle.get('close')}"
    )


def setup_key(
    symbol,
    timeframe,
    direction,
    candles
):

    return (
        f"{symbol}|"
        f"{timeframe}|"
        f"{direction}|"
        f"{candle_identity(candles[-1])}"
    )


def cleanup_setup_memory():

    now = time.time()

    with setup_memory_lock:

        expired = [

            key

            for key, timestamp
            in recent_setups.items()

            if (
                now - timestamp
                >= SETUP_REPEAT_BLOCK_SECONDS
            )
        ]

        for key in expired:

            recent_setups.pop(
                key,
                None
            )


def setup_repeat_blocked(key):

    cleanup_setup_memory()

    with setup_memory_lock:

        timestamp = recent_setups.get(key)

        if timestamp is None:
            return False, 0

        elapsed = time.time() - timestamp

        if elapsed >= SETUP_REPEAT_BLOCK_SECONDS:

            recent_setups.pop(
                key,
                None
            )

            return False, 0

        remaining = int(
            SETUP_REPEAT_BLOCK_SECONDS
            - elapsed
        )

        return True, remaining


def remember_setup(key):

    with setup_memory_lock:

        recent_setups[key] = time.time()


# ============================================================
# SIGNAL FORMAT
# ============================================================

def format_signal(
    symbol,
    timeframe,
    analysis,
    candles,
    trade_type
):

    direction = analysis["direction"]

    now = datetime.now(ALGIERS)

    delay_minutes = get_entry_delay_minutes(
        timeframe
    )

    entry_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        + timedelta(
            minutes=delay_minutes
        )
    )

    entry_price = get_entry_price(candles)

    cancellation_level = (
        get_cancellation_level(
            candles,
            direction
        )
    )

    if direction == "UP":

        cancel_text = (
            "إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(cancellation_level)}"
        )

    else:

        cancel_text = (
            "إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(cancellation_level)}"
        )

    if trade_type == "RECOVERY":

        trade_label = "🔁 RECOVERY 1/1"

    else:

        trade_label = "🎯 BASE TRADE"

    direction_icon = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    message = (

        "🎓 ZinoProSignalAI\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"📊 {symbol} | {timeframe}\n"

        f"{trade_label}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"{direction_icon}\n"

        f"🎯 Confidence: "
        f"{analysis['confidence']}%\n"

        f"📈 UP Score: "
        f"{analysis['up_score']}/18\n"

        f"📉 DOWN Score: "
        f"{analysis['down_score']}/18\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"⏳ Entry after: "
        f"{delay_minutes} min\n"

        f"⏰ ENTRY TIME: "
        f"{entry_time.strftime('%H:%M:%S')} 🇩🇿\n"

        f"💰 Entry Price: "
        f"{format_price(entry_price)}\n"

        f"⚠️ {cancel_text}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"🧠 {analysis['reason']}\n"
    )

    return (
        message,
        entry_time,
        entry_price,
        cancellation_level
    )


# ============================================================
# HISTORY
# ============================================================

def create_trade_record(
    symbol,
    timeframe,
    analysis,
    candles,
    trade_type,
    entry_time,
    entry_price,
    cancellation_level
):

    now = datetime.now(ALGIERS)

    trade_id = (
        f"{now.strftime('%Y%m%d%H%M%S')}"
        f"-{int(time.time() * 1000) % 1000:03d}"
    )

    return {

        "id": trade_id,

        "created_at": now.isoformat(),

        "symbol": symbol,

        "timeframe": timeframe,

        "trade_type": trade_type,

        "direction": analysis["direction"],

        "confidence": analysis["confidence"],

        "up_score": analysis["up_score"],

        "down_score": analysis["down_score"],

        "entry_time": entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "entry_price": entry_price,

        "cancellation_level":
            cancellation_level,

        "reason": analysis["reason"],

        "cancellation_reason":
            analysis["cancellation_reason"],

        "result": "PENDING",

        "result_time": None,
    }


def add_trade_record(record):

    global current_trade_id

    with history_lock:

        trade_history.append(record)

        current_trade_id = record["id"]

    save_history()

    logger.info(
        "TRADE RECORDED | %s | %s | %s",
        record["symbol"],
        record["trade_type"],
        record["direction"]
    )


def update_current_trade_result(result):

    global current_trade_id

    if result not in ("WIN", "LOSS"):
        return False

    updated = False

    now = datetime.now(ALGIERS)

    with history_lock:

        target_id = current_trade_id

        if target_id is not None:

            for record in reversed(
                trade_history
            ):

                if (
                    record.get("id") == target_id
                    and record.get("result") == "PENDING"
                ):

                    record["result"] = result

                    record["result_time"] = (
                        now.isoformat()
                    )

                    updated = True

                    break

        if not updated:

            for record in reversed(
                trade_history
            ):

                if record.get("result") == "PENDING":

                    record["result"] = result

                    record["result_time"] = (
                        now.isoformat()
                    )

                    current_trade_id = (
                        record.get("id")
                    )

                    updated = True

                    break

    if updated:

        save_history()

    return updated


def history_summary():

    with history_lock:

        records = list(trade_history)

    total = len(records)

    wins = sum(
        1
        for x in records
        if x.get("result") == "WIN"
    )

    losses = sum(
        1
        for x in records
        if x.get("result") == "LOSS"
    )

    pending = sum(
        1
        for x in records
        if x.get("result") == "PENDING"
    )

    return (
        records,
        total,
        wins,
        losses,
        pending
    )


# ============================================================
# CYCLE
# ============================================================

def get_active_cycle():

    with cycle_lock:
        return dict(active_cycle)


def start_base_cycle(
    symbol,
    timeframe,
    direction
):

    with cycle_lock:

        active_cycle["active"] = True

        active_cycle["symbol"] = symbol

        active_cycle["timeframe"] = timeframe

        active_cycle["trade_type"] = "BASE"

        active_cycle["direction"] = direction

        active_cycle["recovery_used"] = False

        active_cycle["recovery_ready"] = False

        # الصفقة BASE مرسلة وننتظر نتيجتها
        active_cycle["pending_result"] = True

        active_cycle["last_trade_time"] = time.time()

        active_cycle["trade_number"] = 1

    logger.info(
        "BASE CYCLE STARTED | %s | %s",
        symbol,
        direction
    )


def start_recovery_cycle():

    with cycle_lock:

        active_cycle["active"] = True

        active_cycle["trade_type"] = "RECOVERY"

        active_cycle["recovery_used"] = True

        # الآن مسموح بإرسال Recovery واحدة
        active_cycle["recovery_ready"] = True

        # لا توجد صفقة Recovery مرسلة بعد
        active_cycle["pending_result"] = False

        active_cycle["trade_number"] = 2

        active_cycle["last_trade_time"] = time.time()

    logger.info("RECOVERY IS READY")


def mark_recovery_sent(direction):

    with cycle_lock:

        active_cycle["trade_type"] = "RECOVERY"

        active_cycle["recovery_ready"] = False

        # Recovery مرسلة، الآن ننتظر النتيجة
        active_cycle["pending_result"] = True

        active_cycle["direction"] = direction

        active_cycle["last_trade_time"] = time.time()

        active_cycle["trade_number"] = 2

    logger.info(
        "RECOVERY SENT | %s",
        direction
    )


def reset_cycle():

    with cycle_lock:

        active_cycle["active"] = False

        active_cycle["symbol"] = None

        active_cycle["timeframe"] = None

        active_cycle["trade_type"] = None

        active_cycle["direction"] = None

        active_cycle["recovery_used"] = False

        active_cycle["recovery_ready"] = False

        active_cycle["pending_result"] = False

        active_cycle["last_trade_time"] = 0.0

        active_cycle["trade_number"] = 0

    logger.info(
        "CYCLE RESET - SEARCH FOR NEW PAIR"
    )


# ============================================================
# SIGNAL COOLDOWN
# ============================================================

def signal_cooldown_active():

    elapsed = (
        time.time()
        - last_signal_sent_at
    )

    if elapsed < SIGNAL_COOLDOWN_SECONDS:

        remaining = int(
            SIGNAL_COOLDOWN_SECONDS
            - elapsed
        )

        return True, remaining

    return False, 0


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram_direct(message):

    if not BOT_TOKEN:

        logger.error(
            "BOT_TOKEN is missing"
        )

        return False

    oid = owner_id_int()

    if oid is None:

        logger.error(
            "OWNER_ID is invalid"
        )

        return False

    try:

        url = (
            "https://api.telegram.org/bot"
            f"{BOT_TOKEN}/sendMessage"
        )

        data = urllib.parse.urlencode({

            "chat_id": str(oid),

            "text": message,

        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=data,
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=30
        ) as response:

            body = response.read().decode(
                "utf-8"
            )

        result = json.loads(body)

        if result.get("ok"):

            return True

        logger.error(
            "Telegram API error: %s",
            result
        )

        return False

    except Exception as e:

        logger.exception(
            "Telegram direct send failed: %s",
            e
        )

        return False


def send_signal_safely(message):

    global last_signal_sent_at

    with signal_send_lock:

        active, remaining = (
            signal_cooldown_active()
        )

        if active:

            logger.info(
                "Signal cooldown active: %ss",
                remaining
            )

            return False

        success = send_telegram_direct(
            message
        )

        if success:

            last_signal_sent_at = time.time()

            return True

        return False


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(
    symbol,
    timeframe
):

    if not analysis_lock.acquire(
        blocking=False
    ):

        logger.info(
            "Analysis already running"
        )

        return

    try:

        timeframe = normalize_timeframe(
            timeframe
        )

        if timeframe != ANALYSIS_TIMEFRAME:
            return

        cycle = get_active_cycle()

        # ====================================================
        # أهم حماية:
        #
        # إذا توجد صفقة مرسلة وننتظر /win أو /loss
        # ممنوع إرسال أي صفقة أخرى.
        # ====================================================

        if cycle["active"]:

            if cycle["pending_result"]:

                logger.info(
                    "BLOCKED: active trade waiting for result"
                )

                return

            # Recovery ليست جاهزة
            if (
                cycle["trade_type"] == "RECOVERY"
                and not cycle["recovery_ready"]
            ):

                logger.info(
                    "BLOCKED: recovery already used/sent"
                )

                return

            # نفس الزوج فقط داخل الدورة
            if cycle["symbol"] != symbol:
                return

            if cycle["timeframe"] != timeframe:
                return

        else:

            best = choose_best_pair()

            if not best:
                return

            if best["symbol"] != symbol:
                return

            if best["timeframe"] != timeframe:
                return

        # ====================================================
        # GLOBAL COOLDOWN
        # ====================================================

        active, remaining = (
            signal_cooldown_active()
        )

        if active:

            logger.info(
                "Cooldown active: %ss",
                remaining
            )

            return

        # ====================================================
        # GET MT4 DATA
        # ====================================================

        with mt4_lock:

            timeframes = mt4_data.get(symbol)

            if not timeframes:
                return

            candles = timeframes.get(timeframe)

            if not candles:
                return

            candles = list(candles)

        if len(candles) < (
            MIN_CLOSED_CANDLES + 1
        ):

            logger.info(
                "%s: not enough candles: %s",
                symbol,
                len(candles)
            )

            return

        closed = candles[:-1]

        if len(closed) < MIN_CLOSED_CANDLES:
            return

        # ====================================================
        # GEMINI
        # ====================================================

        logger.info(
            "Analyzing ONE candidate: %s %s",
            symbol,
            timeframe
        )

        analysis = analyze_with_gemini(
            symbol,
            timeframe,
            closed
        )

        analysis = ensure_directional_signal(
            analysis,
            closed
        )

        logger.info(
            "GEMINI RESULT | %s | %s | %s%% | %s/%s",
            symbol,
            analysis["direction"],
            analysis["confidence"],
            analysis["up_score"],
            analysis["down_score"]
        )

        # ====================================================
        # TRADE TYPE
        # ====================================================

        cycle = get_active_cycle()

        if cycle["active"]:

            trade_type = cycle["trade_type"]

        else:

            trade_type = "BASE"

        # ====================================================
        # SETUP PROTECTION
        # ====================================================

        key = setup_key(
            symbol,
            timeframe,
            analysis["direction"],
            closed
        )

        if trade_type == "BASE":

            blocked, remaining = (
                setup_repeat_blocked(key)
            )

            if blocked:

                logger.info(
                    "Same BASE setup blocked: %ss",
                    remaining
                )

                return

        # ====================================================
        # BUILD ONE SIGNAL
        # ====================================================

        (
            message,
            entry_time,
            entry_price,
            cancellation_level
        ) = format_signal(
            symbol,
            timeframe,
            analysis,
            closed,
            trade_type
        )

        # ====================================================
        # SEND ONLY ONE
        # ====================================================

        sent = send_signal_safely(
            message
        )

        if not sent:

            logger.info(
                "Signal not sent"
            )

            return

        # ====================================================
        # RECORD
        # ====================================================

        record = create_trade_record(
            symbol=symbol,
            timeframe=timeframe,
            analysis=analysis,
            candles=closed,
            trade_type=trade_type,
            entry_time=entry_time,
            entry_price=entry_price,
            cancellation_level=cancellation_level,
        )

        add_trade_record(record)

        remember_setup(key)

        # ====================================================
        # CYCLE STATE
        # ====================================================

        if not cycle["active"]:

            # BASE
            start_base_cycle(
                symbol,
                timeframe,
                analysis["direction"]
            )

        else:

            # RECOVERY
            if trade_type == "RECOVERY":

                mark_recovery_sent(
                    analysis["direction"]
                )

        logger.info(
            "ONE SIGNAL SENT | %s | %s | %s",
            symbol,
            trade_type,
            analysis["direction"]
        )

    except Exception as e:

        logger.exception(
            "Auto analysis failed: %s",
            e
        )

    finally:

        analysis_lock.release()


# ============================================================
# BACKGROUND LOOP
# ============================================================

def background_analysis_loop():

    logger.info(
        "Background analysis loop started"
    )

    while True:

        try:

            cleanup_setup_memory()

            cycle = get_active_cycle()

            # =================================================
            # توجد صفقة وننتظر نتيجتها
            # لا تحليل ولا صفقة جديدة
            # =================================================

            if (
                cycle["active"]
                and cycle["pending_result"]
            ):

                logger.info(
                    "Waiting for /win or /loss | %s",
                    cycle["trade_type"]
                )

            elif cycle["active"]:

                if (
                    cycle["symbol"]
                    and cycle["recovery_ready"]
                ):

                    auto_analyze_pair(
                        cycle["symbol"],
                        ANALYSIS_TIMEFRAME
                    )

            else:

                best = choose_best_pair()

                if best:

                    auto_analyze_pair(
                        best["symbol"],
                        ANALYSIS_TIMEFRAME
                    )

                else:

                    logger.info(
                        "No suitable M1 candidate yet"
                    )

        except Exception as e:

            logger.exception(
                "Background loop error: %s",
                e
            )

        time.sleep(
            AUTO_ANALYSIS_INTERVAL_SECONDS
        )


# ============================================================
# MT4 DATA
# ============================================================

def store_mt4_data(payload):

    symbol = str(
        payload.get("symbol")
        or payload.get("Symbol")
        or ""
    ).upper().strip()

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or payload.get("Timeframe")
        or payload.get("tf")
    )

    candles = (
        payload.get("candles")
        or payload.get("data")
        or []
    )

    if not symbol:
        raise ValueError("Missing symbol")

    if not timeframe:
        raise ValueError("Missing timeframe")

    normalized = normalize_candles(candles)

    if not normalized:
        raise ValueError("No valid candles")

    with mt4_lock:

        if symbol not in mt4_data:
            mt4_data[symbol] = {}

        mt4_data[symbol][timeframe] = normalized

    logger.info(
        "MT4 data stored: %s %s candles=%s",
        symbol,
        timeframe,
        len(normalized)
    )

    return (
        symbol,
        timeframe,
        normalized
    )


# ============================================================
# HTTP SERVER
# ============================================================

class MT4Handler(BaseHTTPRequestHandler):

    def do_HEAD(self):

        parsed = urlparse(self.path)

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
            "/mt4",
            "/api/mt4",
        ):

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.end_headers()

            return

        self.send_response(404)

        self.end_headers()


    def do_GET(self):

        parsed = urlparse(self.path)

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
        ):

            body = (
                "ZinoProSignalAI is running"
            ).encode("utf-8")

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

            self.wfile.write(body)

            return

        if parsed.path in (
            "/mt4",
            "/api/mt4",
        ):

            body = (
                "ZinoProSignalAI MT4 endpoint"
            ).encode("utf-8")

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

            self.wfile.write(body)

            return

        self.send_response(404)

        self.end_headers()


    def do_POST(self):

        parsed = urlparse(self.path)

        if parsed.path not in (
            "/mt4",
            "/api/mt4"
        ):

            self.send_response(404)

            self.end_headers()

            return

        try:

            # =================================================
            # API KEY
            # يدعم Header
            # ويدعم api_key داخل JSON
            # =================================================

            received_key = (
                self.headers.get(
                    "X-MT4-API-Key"
                )
                or
                self.headers.get(
                    "X-API-Key"
                )
                or
                ""
            ).strip()

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            if content_length <= 0:

                body = b"Empty request"

                self.send_response(400)

                self.send_header(
                    "Content-Type",
                    "text/plain"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(body)

                return

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

            # توافق مع EA القديم
            if not received_key:

                received_key = str(
                    payload.get(
                        "api_key",
                        ""
                    )
                ).strip()

            if MT4_API_KEY:

                if received_key != MT4_API_KEY:

                    logger.warning(
                        "Invalid MT4 API key"
                    )

                    body = b"Invalid API key"

                    self.send_response(401)

                    self.send_header(
                        "Content-Type",
                        "text/plain"
                    )

                    self.send_header(
                        "Content-Length",
                        str(len(body))
                    )

                    self.end_headers()

                    self.wfile.write(body)

                    return

            (
                symbol,
                timeframe,
                candles
            ) = store_mt4_data(payload)

            # =================================================
            # فقط M1
            # =================================================

            if timeframe == ANALYSIS_TIMEFRAME:

                analysis_thread = threading.Thread(

                    target=auto_analyze_pair,

                    args=(
                        symbol,
                        timeframe,
                    ),

                    daemon=True,
                )

                analysis_thread.start()

            response = {

                "ok": True,

                "symbol": symbol,

                "timeframe": timeframe,

                "candles": len(candles),

                "analysis_timeframe":
                    ANALYSIS_TIMEFRAME,
            }

            body = json.dumps(
                response
            ).encode("utf-8")

            self.send_response(200)

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

        except json.JSONDecodeError:

            body = b"Invalid JSON"

            self.send_response(400)

            self.send_header(
                "Content-Type",
                "text/plain"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(body)

        except Exception as e:

            logger.exception(
                "MT4 POST error: %s",
                e
            )

            body = (
                f"Server error: {e}"
            ).encode("utf-8")

            self.send_response(500)

            self.send_header(
                "Content-Type",
                "text/plain"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(body)


    def log_message(
        self,
        format_string,
        *args
    ):

        logger.info(
            "HTTP %s - %s",
            self.address_string(),
            format_string % args
        )


def start_http_server():

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        MT4Handler,
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

    if not is_owner(update):
        return

    await update.message.reply_text(

        "🎓 ZinoProSignalAI\n\n"

        "✅ Bot is running\n\n"

        "📡 MT4 M1 → Render → Gemini → Telegram\n\n"

        "🎯 ONE BEST SIGNAL ONLY\n"

        "⏱️ Entry delay: 1 min\n"

        "🔁 Recovery: 1/1 فقط\n"

        "🛑 انتظار /win أو /loss بعد كل صفقة\n\n"

        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/analyze"
    )


# ============================================================
# STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    total = (
        stats_data["wins"]
        + stats_data["losses"]
    )

    winrate = (
        (
            stats_data["wins"]
            / total
        ) * 100
        if total > 0
        else 0
    )

    cycle = get_active_cycle()

    if cycle["active"]:

        if cycle["pending_result"]:

            state = "⏳ WAITING RESULT"

        elif cycle["recovery_ready"]:

            state = "🔁 RECOVERY READY"

        else:

            state = "ACTIVE"

        cycle_text = (
            f"{state} | "
            f"{cycle['trade_type']} | "
            f"{cycle['symbol']} | "
            f"{cycle['timeframe']}"
        )

    else:

        cycle_text = "🔎 SEARCHING NEW PAIR"

    (
        records,
        history_total,
        history_wins,
        history_losses,
        history_pending
    ) = history_summary()

    history_winrate = (
        (
            history_wins
            /
            (
                history_wins
                + history_losses
            )
        ) * 100
        if (
            history_wins
            + history_losses
        ) > 0
        else 0
    )

    text = (

        "📊 ZinoProSignalAI Stats\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"Session trades: {total}\n"

        f"🟢 Wins: {stats_data['wins']}\n"

        f"🔴 Losses: {stats_data['losses']}\n"

        f"📈 Win rate: {winrate:.1f}%\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"Base wins: {stats_data['base_wins']}\n"

        f"Base losses: {stats_data['base_losses']}\n"

        f"Recovery wins: {stats_data['recovery_wins']}\n"

        f"Recovery losses: {stats_data['recovery_losses']}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"History: {history_total}\n"

        f"🟢 WIN: {history_wins}\n"

        f"🔴 LOSS: {history_losses}\n"

        f"⏳ Pending: {history_pending}\n"

        f"📈 History win rate: "
        f"{history_winrate:.1f}%\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"Cycle: {cycle_text}"
    )

    await update.message.reply_text(text)


# ============================================================
# HISTORY
# ============================================================

async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    with history_lock:

        records = list(
            trade_history[
                -HISTORY_DISPLAY_COUNT:
            ]
        )

    if not records:

        await update.message.reply_text(
            "📚 لا توجد صفقات مسجلة حتى الآن."
        )

        return

    lines = [

        "📚 ZinoProSignalAI HISTORY",

        "━━━━━━━━━━━━━━━━━━"
    ]

    for record in reversed(records):

        result = record.get(
            "result",
            "PENDING"
        )

        if result == "WIN":
            result_icon = "🟢"

        elif result == "LOSS":
            result_icon = "🔴"

        else:
            result_icon = "⏳"

        direction = record.get(
            "direction",
            "?"
        )

        direction_icon = (
            "🟢"
            if direction == "UP"
            else "🔴"
        )

        lines.append(

            f"{result_icon} "
            f"{record.get('symbol', '?')} "
            f"{record.get('timeframe', '?')}\n"

            f"   {record.get('trade_type', '?')} | "
            f"{direction_icon} {direction}\n"

            f"   🎯 {record.get('confidence', '?')}% | "
            f"📈 {record.get('up_score', '?')}/18 "
            f"📉 {record.get('down_score', '?')}/18\n"

            f"   💰 "
            f"{format_price(safe_float(record.get('entry_price')))}\n"

            f"   ⏰ "
            f"{record.get('entry_time', '?')}"
        )

        lines.append(
            "━━━━━━━━━━━━━━━━━━"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    cycle = get_active_cycle()

    if not cycle["active"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة نشطة."
        )

        return

    if not cycle["pending_result"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة تنتظر النتيجة."
        )

        return

    trade_type = cycle["trade_type"]

    update_current_trade_result("WIN")

    stats_data["wins"] += 1

    if trade_type == "BASE":

        stats_data["base_wins"] += 1

        text = (

            "🟢 BASE WIN\n\n"

            "📚 تم تسجيل الصفقة: WIN\n\n"

            "🏁 انتهت الدورة بنجاح.\n\n"

            "🔎 البحث عن أفضل صفقة جديدة."
        )

    else:

        stats_data["recovery_wins"] += 1

        text = (

            "🟢 RECOVERY WIN\n\n"

            "📚 تم تسجيل Recovery: WIN\n\n"

            "💰 تم تعويض خسارة BASE.\n\n"

            "🏁 انتهت الدورة.\n"

            "🔎 البحث عن أفضل صفقة جديدة."
        )

    reset_cycle()

    await update.message.reply_text(text)


# ============================================================
# LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    cycle = get_active_cycle()

    if not cycle["active"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة نشطة."
        )

        return

    if not cycle["pending_result"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة تنتظر النتيجة."
        )

        return

    trade_type = cycle["trade_type"]

    update_current_trade_result("LOSS")

    stats_data["losses"] += 1

    # ========================================================
    # BASE LOSS
    # ========================================================

    if trade_type == "BASE":

        stats_data["base_losses"] += 1

        start_recovery_cycle()

        await update.message.reply_text(

            "🔴 BASE LOSS\n\n"

            "📚 تم تسجيل الصفقة: LOSS\n\n"

            "🔁 Recovery 1/1 مسموح.\n\n"

            "🎯 سيتم البحث عن Recovery واحدة فقط.\n"

            "⏳ بعدها سأتوقف وأنتظر /win أو /loss.\n\n"

            "🚫 لا توجد Recovery ثانية."
        )

        return

    # ========================================================
    # RECOVERY LOSS
    # ========================================================

    stats_data["recovery_losses"] += 1

    reset_cycle()

    await update.message.reply_text(

        "🔴 RECOVERY LOSS\n\n"

        "📚 تم تسجيل Recovery: LOSS\n\n"

        "⛔ انتهت الدورة.\n\n"

        "🚫 لا توجد Recovery ثانية.\n"

        "🔎 سيتم البحث عن أفضل صفقة جديدة."
    )


# ============================================================
# RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    for key in stats_data:
        stats_data[key] = 0

    reset_cycle()

    global last_signal_sent_at
    global current_trade_id

    last_signal_sent_at = 0.0

    current_trade_id = None

    with setup_memory_lock:
        recent_setups.clear()

    with history_lock:
        trade_history.clear()

    save_history()

    await update.message.reply_text(

        "♻️ تم التصفير:\n\n"

        "• الإحصائيات\n"
        "• الدورة\n"
        "• Recovery\n"
        "• ذاكرة الإشارات\n"
        "• سجل الصفقات"
    )


# ============================================================
# MT4 STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    with mt4_lock:

        data_copy = {

            symbol: dict(timeframes)

            for symbol, timeframes
            in mt4_data.items()
        }

    if not data_copy:

        await update.message.reply_text(
            "📡 لا توجد بيانات MT4."
        )

        return

    lines = [

        "📡 MT4 STATUS",

        "━━━━━━━━━━━━━━━━━━"
    ]

    for symbol, timeframes in data_copy.items():

        for timeframe, candles in timeframes.items():

            lines.append(

                f"📊 {symbol} {timeframe}: "
                f"{len(candles)} candles"
            )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    cycle = get_active_cycle()

    if cycle["active"]:

        if cycle["pending_result"]:

            await update.message.reply_text(

                "⏳ توجد صفقة نشطة.\n\n"

                "أرسل /win أو /loss أولًا."
            )

            return

        if cycle["recovery_ready"]:

            await update.message.reply_text(

                "🔁 Recovery 1/1 جاهزة.\n\n"

                "لا حاجة لإرسال /analyze."
            )

            return

    best = choose_best_pair()

    if not best:

        await update.message.reply_text(
            "⏳ لا توجد بيانات M1 كافية."
        )

        return

    await update.message.reply_text(

        "🔎 أفضل مرشح حاليًا:\n\n"

        f"📊 {best['symbol']} | M1\n"

        f"📈 Pre-score: {best['score']}\n"

        f"🧭 Bias: {best['direction']}\n\n"

        "🧠 سيتم إرسال صفقة واحدة فقط."
    )

    thread = threading.Thread(

        target=auto_analyze_pair,

        args=(
            best["symbol"],
            ANALYSIS_TIMEFRAME,
        ),

        daemon=True,
    )

    thread.start()


# ============================================================
# TEXT / PHOTO
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    text = update.message.text or ""

    if text.startswith("/"):
        return

    await update.message.reply_text(

        "📡 النظام يعمل من بيانات MT4 المباشرة.\n\n"

        "MT4 M1 → Render → Gemini → Telegram\n\n"

        "🎯 صفقة واحدة فقط في كل دورة."
    )


async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):
        return

    await update.message.reply_text(

        "📸 التحليل الحالي لا يعتمد على الصور.\n\n"

        "📡 MT4 M1 → Render → Gemini"
    )


# ============================================================
# TELEGRAM ERROR
# ============================================================

async def telegram_error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.error(
        "Telegram error: %s",
        context.error
    )


# ============================================================
# TELEGRAM APPLICATION
# ============================================================

def build_telegram_application():

    global telegram_application

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN is missing"
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

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            text_handler
        )
    )

    application.add_error_handler(
        telegram_error_handler
    )

    telegram_application = application

    return application


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "========================================"
    )

    logger.info(
        "Starting ZinoProSignalAI"
    )

    logger.info(
        "Model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Port: %s",
        PORT
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "Analysis timeframe: M1"
    )

    logger.info(
        "ONE SIGNAL PER CYCLE"
    )

    logger.info(
        "Recovery limit: 1"
    )

    logger.info(
        "========================================"
    )

    load_history()

    # ========================================================
    # HTTP
    # ========================================================

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # ========================================================
    # BACKGROUND ANALYSIS
    # ========================================================

    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
    )

    analysis_thread.start()

    # ========================================================
    # TELEGRAM
    # ========================================================

    application = (
        build_telegram_application()
    )

    logger.info(
        "Telegram bot starting"
    )

    application.run_polling(
        close_loop=False
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
