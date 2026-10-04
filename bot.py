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
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")


# ============================================================
# AUTO SETTINGS
# ============================================================

AUTO_CYCLE_MINUTES = 3
AUTO_CYCLE_SECONDS = 180

# IMPORTANT:
# Only ONE signal per cycle.
MAX_SIGNALS_PER_CYCLE = 1

# Only M1 / M3 are eligible for automatic signals.
AUTO_TIMEFRAMES = {"M1", "M3"}

# Consider many pairs so the bot can really choose
# the strongest available pair.
AUTO_MAX_CANDIDATES = 50

# MT4 data must be fresh.
AUTO_DATA_MAX_AGE_SECONDS = 180

# Do not send a signal when there is not enough time
# before the calculated entry.
MIN_ENTRY_LEAD_SECONDS = 20

# Recovery remains one attempt.
MAX_RECOVERY = 1


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# OWNER
# ============================================================

try:
    OWNER_ID = int(OWNER_ID_RAW)
except Exception:
    OWNER_ID = 0


def is_owner(update: Update) -> bool:
    if OWNER_ID <= 0:
        return False

    user = update.effective_user

    if not user:
        return False

    return user.id == OWNER_ID


async def owner_only(update: Update) -> bool:
    if is_owner(update):
        return True

    try:
        await update.message.reply_text(
            "⛔ هذا البوت خاص بالمالك فقط."
        )
    except Exception:
        pass

    return False


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


# ============================================================
# GLOBAL DATA
# ============================================================

# key = SYMBOL|TIMEFRAME
# value = {
#   "symbol": ...,
#   "timeframe": ...,
#   "candles": [...],
#   "updated_at": unix_time
# }
mt4_data = {}

data_lock = threading.Lock()


# Last automatic candle processed.
last_auto_candle = {}

# Last automatic signal information.
last_auto_signal = {}

# Direction history used only for diversity.
last_signal_direction = None
direction_streak = 0

# Current active trade.
active_trade = None

# Recovery state.
recovery_state = {
    "active": False,
    "attempt": 0,
    "base_trade": None,
}

# Statistics.
stats = {
    "wins": 0,
    "losses": 0,
}

# Signal history.
signal_history = []

history_lock = threading.Lock()


# ============================================================
# EXCLUDED SYMBOLS
# ============================================================

EXCLUDED_SYMBOLS = {
    "",
    "UNKNOWN",
    "NONE",
}


# ============================================================
# HELPERS
# ============================================================

def now_algiers() -> datetime:
    return datetime.now(ALGIERS)


def unix_now() -> float:
    return time.time()


def normalize_symbol(symbol: str) -> str:
    symbol = str(symbol or "").strip().upper()

    symbol = symbol.replace("/", "")
    symbol = symbol.replace(" ", "")

    return symbol


def normalize_timeframe(timeframe) -> str:
    if timeframe is None:
        return ""

    value = str(timeframe).strip().upper()

    value = value.replace("MINUTE", "M")
    value = value.replace("MIN", "M")

    if value.isdigit():
        return f"M{value}"

    if value.startswith("M"):
        digits = value[1:]

        if digits.isdigit():
            return f"M{int(digits)}"

    return value


def parse_owner_id(value):
    try:
        return int(value)
    except Exception:
        return 0


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def format_price(price):
    price = safe_float(price)

    if price == 0:
        return "N/A"

    if abs(price) >= 100:
        return f"{price:.3f}"

    if abs(price) >= 10:
        return f"{price:.4f}"

    if abs(price) >= 1:
        return f"{price:.5f}"

    return f"{price:.6f}"


def timeframe_seconds(timeframe):
    tf = normalize_timeframe(timeframe)

    if tf == "M1":
        return 60

    if tf == "M2":
        return 120

    if tf == "M3":
        return 180

    if tf == "M5":
        return 300

    if tf == "M15":
        return 900

    if tf == "H1":
        return 3600

    return 60


def candle_timestamp(candle):
    if not isinstance(candle, dict):
        return None

    for key in (
        "time",
        "timestamp",
        "datetime",
        "date",
        "open_time",
    ):
        if key in candle:
            value = candle[key]

            if isinstance(value, (int, float)):
                return float(value)

            if isinstance(value, str):
                try:
                    return float(value)
                except Exception:
                    pass

                try:
                    dt = datetime.fromisoformat(
                        value.replace("Z", "+00:00")
                    )

                    if dt.tzinfo is None:
                        dt = dt.replace(
                            tzinfo=ALGIERS
                        )

                    return dt.timestamp()

                except Exception:
                    pass

    return None


def candle_ohlc(candle):
    if not isinstance(candle, dict):
        return None

    o = safe_float(
        candle.get(
            "open",
            candle.get("o", 0)
        )
    )

    h = safe_float(
        candle.get(
            "high",
            candle.get("h", 0)
        )
    )

    l = safe_float(
        candle.get(
            "low",
            candle.get("l", 0)
        )
    )

    c = safe_float(
        candle.get(
            "close",
            candle.get("c", 0)
        )
    )

    if not all([o, h, l, c]):
        return None

    return o, h, l, c


def get_next_entry_time(timeframe):
    """
    Entry is calculated from the next candle boundary
    in Africa/Algiers.
    """

    now = now_algiers()

    seconds = timeframe_seconds(timeframe)

    epoch = int(now.timestamp())

    next_epoch = (
        (epoch // seconds) + 1
    ) * seconds

    entry = datetime.fromtimestamp(
        next_epoch,
        ALGIERS
    )

    lead = (
        entry - now
    ).total_seconds()

    if lead < MIN_ENTRY_LEAD_SECONDS:
        entry += timedelta(
            seconds=seconds
        )

    return entry


def get_entry_price(candles):
    if not candles:
        return 0.0

    # Current/forming candle if available.
    candle = candles[-1]

    if isinstance(candle, dict):
        for key in (
            "close",
            "c",
            "price",
        ):
            if key in candle:
                price = safe_float(
                    candle[key]
                )

                if price:
                    return price

    ohlc = candle_ohlc(candle)

    if ohlc:
        return ohlc[3]

    return 0.0


# ============================================================
# CANDLE DATA
# ============================================================

def get_closed_candles(candles):
    if not isinstance(candles, list):
        return []

    if len(candles) < 2:
        return []

    # MT4 sends newest candle as forming candle.
    return candles[:-1]


def get_data_age(data):
    if not data:
        return 999999

    updated_at = safe_float(
        data.get("updated_at", 0)
    )

    if updated_at <= 0:
        return 999999

    return max(
        0,
        unix_now() - updated_at
    )


def get_auto_candidates():
    candidates = []

    with data_lock:
        snapshot = dict(mt4_data)

    for key, data in snapshot.items():

        symbol = normalize_symbol(
            data.get("symbol", "")
        )

        timeframe = normalize_timeframe(
            data.get("timeframe", "")
        )

        if symbol in EXCLUDED_SYMBOLS:
            continue

        if timeframe not in AUTO_TIMEFRAMES:
            continue

        age = get_data_age(data)

        if age > AUTO_DATA_MAX_AGE_SECONDS:
            continue

        candles = data.get(
            "candles",
            []
        )

        if not isinstance(candles, list):
            continue

        # Need at least 41 raw candles:
        # 40 closed + 1 forming.
        if len(candles) < 41:
            continue

        candidates.append({
            "key": key,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": candles,
            "age": age,
        })

    return candidates[:AUTO_MAX_CANDIDATES]


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    result = sum(
        values[:period]
    ) / period

    for price in values[period:]:
        result = (
            (price - result)
            * multiplier
            + result
        )

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        gains.append(
            max(diff, 0)
        )

        losses.append(
            max(-diff, 0)
        )

    avg_gain = (
        sum(gains[:period])
        / period
    )

    avg_loss = (
        sum(losses[:period])
        / period
    )

    for i in range(
        period,
        len(gains)
    ):
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

    return 100 - (
        100 / (1 + rs)
    )


def true_ranges(candles):
    result = []

    for i, candle in enumerate(candles):
        values = candle_ohlc(candle)

        if not values:
            continue

        o, h, l, c = values

        if i == 0:
            tr = h - l
        else:
            prev = candle_ohlc(
                candles[i - 1]
            )

            if not prev:
                tr = h - l
            else:
                pc = prev[3]

                tr = max(
                    h - l,
                    abs(h - pc),
                    abs(l - pc),
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


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highs = []
    lows = []

    for candle in recent:
        values = candle_ohlc(candle)

        if not values:
            continue

        highs.append(values[1])
        lows.append(values[2])

    if not highs or not lows:
        return None

    highest = max(highs)
    lowest = min(lows)

    close = candle_ohlc(
        candles[-1]
    )[3]

    if highest == lowest:
        return -50.0

    return (
        (highest - close)
        / (highest - lowest)
    ) * -100


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(
        1,
        len(candles)
    ):
        current = candle_ohlc(
            candles[i]
        )

        previous = candle_ohlc(
            candles[i - 1]
        )

        if not current or not previous:
            continue

        _, high, low, close = current
        _, prev_high, prev_low, _ = previous

        tr = max(
            high - low,
            abs(high - close),
            abs(low - close),
        )

        up_move = (
            high - prev_high
        )

        down_move = (
            prev_low - low
        )

        pdm = (
            up_move
            if up_move > down_move
            and up_move > 0
            else 0
        )

        mdm = (
            down_move
            if down_move > up_move
            and down_move > 0
            else 0
        )

        trs.append(tr)
        plus_dm.append(pdm)
        minus_dm.append(mdm)

    if len(trs) < period:
        return None, None, None

    tr_sum = sum(
        trs[-period:]
    )

    plus_sum = sum(
        plus_dm[-period:]
    )

    minus_sum = sum(
        minus_dm[-period:]
    )

    if tr_sum == 0:
        return 0.0, 0.0, 0.0

    plus_di = (
        100 * plus_sum / tr_sum
    )

    minus_di = (
        100 * minus_sum / tr_sum
    )

    denominator = (
        plus_di + minus_di
    )

    if denominator == 0:
        adx = 0.0
    else:
        adx = (
            100
            * abs(
                plus_di - minus_di
            )
            / denominator
        )

    return adx, plus_di, minus_di


def keltner(candles):
    closes = []

    for candle in candles:
        values = candle_ohlc(candle)

        if values:
            closes.append(values[3])

    if len(closes) < 20:
        return None, None, None

    middle = ema(
        closes,
        20
    )

    current_atr = atr(
        candles,
        10
    )

    if middle is None or current_atr is None:
        return None, None, None

    multiplier = 5

    upper = (
        middle
        + current_atr * multiplier
    )

    lower = (
        middle
        - current_atr * multiplier
    )

    return middle, upper, lower


# ============================================================
# LOCAL QUICK ANALYSIS
# ============================================================

def quick_candidate_score(candles):
    """
    Cheap local ranking before Gemini.

    This does NOT decide the final trade.
    It only helps identify strong candidates.
    """

    closed = get_closed_candles(
        candles
    )

    if len(closed) < 40:
        return 0, "UP", 0

    closes = []

    highs = []
    lows = []

    for candle in closed:
        values = candle_ohlc(candle)

        if not values:
            continue

        _, h, l, c = values

        closes.append(c)
        highs.append(h)
        lows.append(l)

    if len(closes) < 40:
        return 0, "UP", 0

    score_up = 0
    score_down = 0

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    recent_closes = closes[-10:]

    if recent_closes[-1] > recent_closes[0]:
        score_up += 2
    elif recent_closes[-1] < recent_closes[0]:
        score_down += 2

    # --------------------------------------------------------
    # EMA 9 / 21
    # --------------------------------------------------------

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    if ema9 and ema21:
        if ema9 > ema21:
            score_up += 2
        elif ema9 < ema21:
            score_down += 2

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    r = rsi(
        closes,
        14
    )

    if r is not None:

        if 52 <= r <= 68:
            score_up += 1

        elif 32 <= r <= 48:
            score_down += 1

    # --------------------------------------------------------
    # Williams %R
    # --------------------------------------------------------

    wr = williams_r(
        closed,
        14
    )

    if wr is not None:

        if -80 < wr < -20:

            if wr > -50:
                score_up += 1
            else:
                score_down += 1

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

    adx, plus_di, minus_di = adx_di(
        closed,
        14
    )

    if adx is not None:

        if adx >= 18:

            if plus_di > minus_di:
                score_up += 2

            elif minus_di > plus_di:
                score_down += 2

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if len(closes) >= 6:

        momentum = (
            closes[-1]
            - closes[-6]
        )

        if momentum > 0:
            score_up += 2

        elif momentum < 0:
            score_down += 2

    # --------------------------------------------------------
    # Candle direction
    # --------------------------------------------------------

    last_values = candle_ohlc(
        closed[-1]
    )

    if last_values:
        o, h, l, c = last_values

        body = c - o

        if body > 0:
            score_up += 1

        elif body < 0:
            score_down += 1

    total = (
        score_up
        + score_down
    )

    if score_up >= score_down:
        direction = "UP"
        local_score = score_up
    else:
        direction = "DOWN"
        local_score = score_down

    # Freshness / completeness bonus
    if len(candles) >= 100:
        local_score += 1

    return (
        local_score,
        direction,
        total,
    )


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles
):

    closed = get_closed_candles(
        candles
    )

    if len(closed) > 120:
        closed = closed[-120:]

    candle_text = []

    for i, candle in enumerate(
        closed
    ):

        values = candle_ohlc(
            candle
        )

        if not values:
            continue

        o, h, l, c = values

        ts = candle_timestamp(
            candle
        )

        if ts:
            dt = datetime.fromtimestamp(
                ts,
                ALGIERS
            )

            stamp = dt.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            stamp = str(i)

        candle_text.append(
            {
                "time": stamp,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
            }
        )

    ema9 = None
    ema21 = None
    r = None
    wr = None
    current_atr = None
    adx = None
    plus_di = None
    minus_di = None
    kc_middle = None
    kc_upper = None
    kc_lower = None

    closes = []

    for candle in closed:
        values = candle_ohlc(
            candle
        )

        if values:
            closes.append(
                values[3]
            )

    if len(closes) >= 21:
        ema9 = ema(
            closes,
            9
        )

        ema21 = ema(
            closes,
            21
        )

    r = rsi(
        closes,
        14
    )

    wr = williams_r(
        closed,
        14
    )

    current_atr = atr(
        closed,
        10
    )

    adx, plus_di, minus_di = adx_di(
        closed,
        14
    )

    (
        kc_middle,
        kc_upper,
        kc_lower
    ) = keltner(
        closed
    )

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timezone": "Africa/Algiers",
        "indicators": {
            "EMA9": ema9,
            "EMA21": ema21,
            "RSI14": r,
            "WilliamsR14": wr,
            "ATR10": current_atr,
            "ADX14": adx,
            "DI_plus14": plus_di,
            "DI_minus14": minus_di,
            "Keltner_middle_EMA20": kc_middle,
            "Keltner_upper": kc_upper,
            "Keltner_lower": kc_lower,
            "Keltner_multiplier": 5,
        },
        "candles": candle_text,
    }

    prompt = f"""
You are the technical-analysis engine of ZinoProSignalAI.

Analyze ONLY the supplied real MT4 candle data.

Symbol:
{symbol}

Timeframe:
{timeframe}

IMPORTANT:
- Do not invent candles.
- Do not invent indicators.
- Do not use unavailable market data.
- Use the supplied candles only.
- Final direction MUST be UP or DOWN.
- Never return WAIT.
- Never return NO SIGNAL.
- Never return NEUTRAL.
- Do not force UP.
- Do not force DOWN.
- Choose the direction with stronger technical evidence.

SCORING:
Total = 18 points.

1. Structure = 2
2. Breakout / Retest = 2
3. Liquidity = 1
4. Momentum = 2
5. Candle = 2
6. RSI = 1
7. Summary = 2
8. Oscillators = 3
9. Moving Averages = 3

TOTAL = 18.

PRIORITY:
Price Action
>
Structure
>
Breakout / Retest
>
Liquidity
>
Momentum
>
Candle
>
EMA 9 / EMA 21
>
RSI 14
>
Williams %R 14
>
Keltner
>
ADX / DI

IMPORTANT QUALITY RULES:
- Strong confluence is required.
- Do not give high confidence just because one indicator agrees.
- Confidence must be between 70 and 89.
- Do not use 90 or higher.
- selected_score must be at least 11.
- score_difference must be at least 5.
- up_score + down_score MUST equal exactly 18.
- confirmations should be at least 4.
- contradictions should be less than 2.
- Avoid entries after obvious exhaustion.
- Prefer continuation/retest setups with structure confirmation.
- Do not blindly trade overbought/oversold.
- Do not treat RSI alone as a reversal signal.
- EMA direction must agree when used as confirmation.
- ADX/DI is confirmation, not a standalone reason.
- Williams %R is confirmation, not a standalone reason.
- Keltner is confirmation only.

Return ONLY valid JSON.

Required JSON:

{{
  "direction": "UP or DOWN",
  "confidence": 0,
  "up_score": 0,
  "down_score": 0,
  "selected_score": 0,
  "score_difference": 0,
  "confirmations": 0,
  "contradictions": 0,
  "structure": "",
  "breakout": "",
  "liquidity": "",
  "momentum": "",
  "candle": "",
  "rsi": "",
  "oscillators": "",
  "moving_averages": "",
  "summary": "",
  "reason": ""
}}

DATA:
{json.dumps(payload, ensure_ascii=False)}
"""

    return prompt


# ============================================================
# GEMINI CALL
# ============================================================

def clean_json_text(text):
    if not text:
        return ""

    text = str(text).strip()

    # Remove markdown fences.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Find first JSON object if extra text exists.
    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return text.strip()


def call_gemini_json(prompt):
    if gemini_client is None:
        return None

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        text = getattr(
            response,
            "text",
            None
        )

        if not text:
            return None

        cleaned = clean_json_text(
            text
        )

        if not cleaned:
            return None

        return json.loads(
            cleaned
        )

    except Exception as e:
        logger.exception(
            "Gemini error: %s",
            e
        )

        return None


# ============================================================
# QUALITY FILTER
# ============================================================

def validate_signal(result, candles):
    if not isinstance(result, dict):
        return False, None

    direction = str(
        result.get(
            "direction",
            ""
        )
    ).upper().strip()

    if direction not in {
        "UP",
        "DOWN",
    }:
        return False, None

    confidence = int(
        safe_float(
            result.get(
                "confidence",
                0
            )
        )
    )

    confidence = int(
        clamp(
            confidence,
            0,
            89
        )
    )

    up_score = int(
        safe_float(
            result.get(
                "up_score",
                0
            )
        )
    )

    down_score = int(
        safe_float(
            result.get(
                "down_score",
                0
            )
        )
    )

    selected_score = int(
        safe_float(
            result.get(
                "selected_score",
                0
            )
        )
    )

    score_difference = int(
        safe_float(
            result.get(
                "score_difference",
                0
            )
        )
    )

    confirmations = int(
        safe_float(
            result.get(
                "confirmations",
                0
            )
        )
    )

    contradictions = int(
        safe_float(
            result.get(
                "contradictions",
                0
            )
        )
    )

    # Hard score integrity.
    if up_score + down_score != 18:
        logger.info(
            "Rejected: score total != 18"
        )
        return False, None

    if up_score == down_score:
        logger.info(
            "Rejected: equal scores"
        )
        return False, None

    if direction == "UP":
        if up_score <= down_score:
            return False, None

        selected_score = up_score

    else:
        if down_score <= up_score:
            return False, None

        selected_score = down_score

    score_difference = abs(
        up_score - down_score
    )

    if selected_score < 11:
        logger.info(
            "Rejected: selected score < 11"
        )
        return False, None

    if score_difference < 5:
        logger.info(
            "Rejected: score difference < 5"
        )
        return False, None

    if confidence < 70:
        logger.info(
            "Rejected: confidence < 70"
        )
        return False, None

    if confirmations < 4:
        logger.info(
            "Rejected: confirmations < 4"
        )
        return False, None

    if contradictions >= 2:
        logger.info(
            "Rejected: contradictions >= 2"
        )
        return False, None

    closed = get_closed_candles(
        candles
    )

    if len(closed) < 40:
        logger.info(
            "Rejected: not enough closed candles"
        )
        return False, None

    # Exhaustion protection.
    closes = []

    for candle in closed:
        values = candle_ohlc(
            candle
        )

        if values:
            closes.append(
                values[3]
            )

    if len(closes) >= 8:

        recent_move = (
            closes[-1]
            - closes[-8]
        )

        current_atr = atr(
            closed,
            10
        )

        if current_atr and current_atr > 0:

            if abs(recent_move) > (
                current_atr * 3.0
            ):
                logger.info(
                    "Rejected: possible exhaustion"
                )
                return False, None

    result["direction"] = direction
    result["confidence"] = confidence
    result["up_score"] = up_score
    result["down_score"] = down_score
    result["selected_score"] = selected_score
    result["score_difference"] = score_difference
    result["confirmations"] = confirmations
    result["contradictions"] = contradictions

    return True, result


# ============================================================
# ANALYZE CANDIDATE
# ============================================================

def analyze_candidate(candidate):
    symbol = candidate["symbol"]
    timeframe = candidate["timeframe"]
    candles = candidate["candles"]

    local_score, local_direction, _ = (
        quick_candidate_score(
            candles
        )
    )

    logger.info(
        "Candidate | %s | %s | local=%s | %s",
        symbol,
        timeframe,
        local_score,
        local_direction,
    )

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles
    )

    result = call_gemini_json(
        prompt
    )

    if result is None:
        return None

    approved, quality = validate_signal(
        result,
        candles
    )

    if not approved:
        return None

    return {
        "candidate": candidate,
        "quality": quality,
        "local_score": local_score,
        "local_direction": local_direction,
    }


# ============================================================
# SIGNAL RANKING
# ============================================================

def signal_rank(approved):
    """
    Strongest technical setup remains the priority.

    Direction diversity is only a SMALL tie-breaker.
    It must NEVER make a clearly weaker setup beat
    a clearly stronger setup.
    """

    quality = approved["quality"]

    selected = quality.get(
        "selected_score",
        0
    )

    difference = quality.get(
        "score_difference",
        0
    )

    confidence = quality.get(
        "confidence",
        0
    )

    confirmations = quality.get(
        "confirmations",
        0
    )

    contradictions = quality.get(
        "contradictions",
        0
    )

    direction = quality.get(
        "direction",
        ""
    )

    global last_signal_direction
    global direction_streak

    # Base technical quality.
    base = (
        selected * 100
        + difference * 10
        + confidence * 0.5
        + confirmations * 2
        - contradictions * 5
    )

    diversity_bonus = 0

    if last_signal_direction:
        if direction != last_signal_direction:
            # Small bonus for changing direction.
            diversity_bonus = 8

        else:
            # Small penalty for repeating.
            diversity_bonus = -4

            # If same direction has repeated,
            # increase the penalty slightly.
            if direction_streak >= 2:
                diversity_bonus = -8

    return (
        base + diversity_bonus,
        selected,
        difference,
        confidence,
        confirmations,
        -contradictions,
    )


# ============================================================
# ENTRY / CARD
# ============================================================

def create_signal_card(
    symbol,
    timeframe,
    quality,
    entry_time,
    entry_price,
    recovery=False,
):
    direction = quality.get(
        "direction",
        "UP"
    )

    confidence = int(
        quality.get(
            "confidence",
            70
        )
    )

    up_score = int(
        quality.get(
            "up_score",
            0
        )
    )

    down_score = int(
        quality.get(
            "down_score",
            0
        )
    )

    reason = str(
        quality.get(
            "reason",
            quality.get(
                "summary",
                "Technical confluence."
            )
        )
    ).strip()

    if len(reason) > 180:
        reason = reason[:177] + "..."

    price = safe_float(
        entry_price
    )

    # Cancellation level.
    closed = quality.get(
        "_candles",
        []
    )

    cancel_price = 0.0

    if isinstance(
        closed,
        list
    ) and closed:

        candle = closed[-1]

        values = candle_ohlc(
            candle
        )

        if values:
            o, h, l, c = values

            if direction == "UP":
                cancel_price = l
            else:
                cancel_price = h

    # Fallback.
    if cancel_price <= 0:
        cancel_price = price

    if direction == "UP":
        emoji = "🟢"
        decision = "UP"
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(cancel_price)}"
        )

    else:
        emoji = "🔴"
        decision = "DOWN"
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(cancel_price)}"
        )

    if recovery:
        recovery_line = (
            "🔁 RECOVERY 1/1\n"
        )
    else:
        recovery_line = ""

    card = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{recovery_line}"
        f"{emoji} {decision}\n"
        f"🎯 Confidence: {confidence}%\n"
        f"📈 UP Score: {up_score}/18\n"
        f"📉 DOWN Score: {down_score}/18\n\n"
        f"⏰ ENTRY TIME\n"
        f"{entry_time.strftime('%H:%M:%S')}\n\n"
        f"💰 Entry Price\n"
        f"{format_price(price)}\n"
        f"{cancel_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return card


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_message_safe(
    application,
    text_message,
):
    if not application:
        return False

    try:
        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=text_message,
        )

        return True

    except Exception as e:
        logger.exception(
            "Telegram send error: %s",
            e
        )

        return False


# ============================================================
# HISTORY
# ============================================================

def add_history(record):
    with history_lock:
        signal_history.append(
            record
        )

        # Keep latest 100.
        if len(signal_history) > 100:
            del signal_history[:-100]


def update_last_trade_result(result):
    global active_trade

    with history_lock:

        if active_trade:
            active_trade["result"] = result
            active_trade["result_time"] = (
                now_algiers().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

            signal_id = active_trade.get(
                "id"
            )

            for item in reversed(
                signal_history
            ):
                if item.get("id") == signal_id:
                    item["result"] = result
                    item["result_time"] = (
                        active_trade[
                            "result_time"
                        ]
                    )
                    break


def get_current_trade_record():
    global active_trade

    return active_trade


# ============================================================
# SEND AUTO SIGNAL
# ============================================================

async def send_auto_signal(
    application,
    approved,
    recovery=False,
):
    global active_trade
    global last_signal_direction
    global direction_streak
    global last_auto_signal

    candidate = approved["candidate"]
    quality = approved["quality"]

    symbol = candidate["symbol"]
    timeframe = candidate["timeframe"]
    candles = candidate["candles"]

    # Attach candles for cancellation calculation.
    quality["_candles"] = (
        get_closed_candles(
            candles
        )
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    now = now_algiers()

    lead = (
        entry_time - now
    ).total_seconds()

    if lead < MIN_ENTRY_LEAD_SECONDS:
        logger.info(
            "Signal skipped: entry lead only %.1f sec",
            lead
        )
        return False

    entry_price = get_entry_price(
        candles
    )

    card = create_signal_card(
        symbol=symbol,
        timeframe=timeframe,
        quality=quality,
        entry_time=entry_time,
        entry_price=entry_price,
        recovery=recovery,
    )

    sent = await send_message_safe(
        application,
        card
    )

    if not sent:
        return False

    trade_id = (
        f"{symbol}_{int(time.time())}"
    )

    active_trade = {
        "id": trade_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": quality["direction"],
        "confidence": quality["confidence"],
        "up_score": quality["up_score"],
        "down_score": quality["down_score"],
        "selected_score": quality["selected_score"],
        "entry_time": entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "entry_price": entry_price,
        "recovery": recovery,
        "result": None,
        "created_at": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    add_history(
        dict(active_trade)
    )

    last_signal_direction = (
        quality["direction"]
    )

    if (
        last_signal_direction
        == quality["direction"]
    ):
        direction_streak += 1
    else:
        direction_streak = 1

    last_auto_signal = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": quality["direction"],
        "sent_at": unix_now(),
        "entry_time": entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    }

    logger.info(
        "AUTO SIGNAL SENT | %s | %s | %s | %s%%",
        symbol,
        timeframe,
        quality["direction"],
        quality["confidence"],
    )

    return True


# ============================================================
# FIRST SIGNAL CYCLE
# ============================================================

async def run_first_signal_cycle(
    application
):
    candidates = get_auto_candidates()

    if not candidates:
        logger.warning(
            "No eligible M1/M3 candidates."
        )
        return None

    logger.info(
        "Auto candidates available: %s",
        len(candidates)
    )

    approved = []

    for candidate in candidates:

        try:
            result = analyze_candidate(
                candidate
            )

            if result:
                approved.append(
                    result
                )

        except Exception as e:
            logger.exception(
                "Candidate analysis error %s: %s",
                candidate.get("symbol"),
                e
            )

    if not approved:
        logger.warning(
            "No approved signal this cycle."
        )
        return None

    approved.sort(
        key=signal_rank,
        reverse=True
    )

    # ONE SIGNAL ONLY.
    best = approved[0]

    logger.info(
        "BEST PAIR | %s | %s | %s | %s/18 | conf=%s",
        best["candidate"]["symbol"],
        best["candidate"]["timeframe"],
        best["quality"]["direction"],
        best["quality"]["selected_score"],
        best["quality"]["confidence"],
    )

    sent = await send_auto_signal(
        application,
        best,
        recovery=False,
    )

    if sent:
        return {
            "symbol": best["candidate"]["symbol"],
            "sent_at": unix_now(),
        }

    return None


# ============================================================
# RECOVERY
# ============================================================

async def run_recovery_cycle(
    application
):
    """
    Recovery is deliberately kept as ONE attempt.

    It does not send multiple unrelated signals.
    """

    global recovery_state

    if not recovery_state.get(
        "active",
        False
    ):
        return False

    if recovery_state.get(
        "attempt",
        0
    ) >= MAX_RECOVERY:
        recovery_state["active"] = False
        return False

    candidates = get_auto_candidates()

    if not candidates:
        return False

    approved = []

    for candidate in candidates:

        result = analyze_candidate(
            candidate
        )

        if result:
            approved.append(
                result
            )

    if not approved:
        return False

    approved.sort(
        key=signal_rank,
        reverse=True
    )

    best = approved[0]

    recovery_state["attempt"] += 1

    sent = await send_auto_signal(
        application,
        best,
        recovery=True,
    )

    if sent:
        recovery_state["active"] = False
        return True

    recovery_state["attempt"] -= 1

    return False


# ============================================================
# AUTO LOOP
# ============================================================

async def auto_loop(application):
    logger.info(
        "Automatic signal loop started."
    )

    while True:

        try:
            now = now_algiers()

            # Wait for next 3-minute boundary.
            epoch = int(
                now.timestamp()
            )

            next_boundary = (
                (epoch // AUTO_CYCLE_SECONDS)
                + 1
            ) * AUTO_CYCLE_SECONDS

            wait_seconds = (
                next_boundary
                - time.time()
            )

            if wait_seconds < 1:
                wait_seconds = 1

            logger.info(
                "Next auto cycle in %.1f sec",
                wait_seconds
            )

            await asyncio.sleep(
                wait_seconds
            )

            # ------------------------------------------------
            # If previous BASE signal lost,
            # use Recovery 1/1.
            # ------------------------------------------------

            if recovery_state.get(
                "active",
                False
            ):

                logger.info(
                    "Recovery cycle active."
                )

                await run_recovery_cycle(
                    application
                )

            else:

                # ------------------------------------------------
                # ONE SIGNAL ONLY.
                # ------------------------------------------------

                await run_first_signal_cycle(
                    application
                )

            # No second signal.
            await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info(
                "Auto loop cancelled."
            )
            break

        except Exception as e:
            logger.exception(
                "Auto loop error: %s",
                e
            )

            await asyncio.sleep(
                5
            )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    text = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Bot is running.\n\n"
        "📡 MT4 → Render → Gemini\n"
        "🎯 One strongest pair every 3 minutes\n"
        "📊 M1 / M3 only\n"
        "🔁 Recovery 1/1\n\n"
        "Commands:\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/analyze SYMBOL M1"
    )

    await update.message.reply_text(
        text
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    active = get_current_trade_record()

    active_text = "None"

    if active:
        active_text = (
            f"{active.get('symbol')} "
            f"{active.get('timeframe')} "
            f"{active.get('direction')} "
            f"| {active.get('result') or 'PENDING'}"
        )

    total = (
        stats["wins"]
        + stats["losses"]
    )

    if total:
        rate = (
            stats["wins"]
            / total
        ) * 100
    else:
        rate = 0

    text = (
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"✅ WIN: {stats['wins']}\n"
        f"❌ LOSS: {stats['losses']}\n"
        f"📈 TOTAL: {total}\n"
        f"🎯 RATE: {rate:.1f}%\n\n"
        f"🔔 Active:\n{active_text}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    await update.message.reply_text(
        text
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    global stats
    global active_trade
    global recovery_state

    trade = get_current_trade_record()

    if not trade:
        await update.message.reply_text(
            "⚠️ لا توجد صفقة نشطة."
        )
        return

    if trade.get("result"):
        await update.message.reply_text(
            "⚠️ نتيجة هذه الصفقة مسجلة مسبقًا."
        )
        return

    stats["wins"] += 1

    update_last_trade_result(
        "WIN"
    )

    # A WIN closes recovery chain.
    recovery_state = {
        "active": False,
        "attempt": 0,
        "base_trade": None,
    }

    await update.message.reply_text(
        "✅ WIN\n"
        "تم تسجيل الربح للصفقة الحالية."
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    global stats
    global recovery_state

    trade = get_current_trade_record()

    if not trade:
        await update.message.reply_text(
            "⚠️ لا توجد صفقة نشطة."
        )
        return

    if trade.get("result"):
        await update.message.reply_text(
            "⚠️ نتيجة هذه الصفقة مسجلة مسبقًا."
        )
        return

    stats["losses"] += 1

    update_last_trade_result(
        "LOSS"
    )

    # Start ONE recovery.
    recovery_state = {
        "active": True,
        "attempt": 0,
        "base_trade": dict(trade),
    }

    await update.message.reply_text(
        "❌ LOSS\n"
        "🔁 Recovery 1/1 جاهز للدورة القادمة."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    global stats
    global signal_history
    global active_trade
    global recovery_state
    global last_signal_direction
    global direction_streak

    stats = {
        "wins": 0,
        "losses": 0,
    }

    signal_history = []

    active_trade = None

    recovery_state = {
        "active": False,
        "attempt": 0,
        "base_trade": None,
    }

    last_signal_direction = None
    direction_streak = 0

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات والسجل."
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    with history_lock:
        items = list(
            signal_history[-20:]
        )

    if not items:
        await update.message.reply_text(
            "📭 لا توجد صفقات مسجلة."
        )
        return

    lines = [
        "📚 ZinoProSignalAI History",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in reversed(items):

        result = item.get(
            "result",
            "PENDING"
        )

        if result == "WIN":
            icon = "✅"
        elif result == "LOSS":
            icon = "❌"
        else:
            icon = "⏳"

        lines.append(
            f"{icon} "
            f"{item.get('symbol')} | "
            f"{item.get('timeframe')} | "
            f"{item.get('direction')} | "
            f"{item.get('confidence')}% | "
            f"{result}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    with data_lock:
        snapshot = dict(mt4_data)

    if not snapshot:
        await update.message.reply_text(
            "📡 لا توجد بيانات MT4 حاليًا."
        )
        return

    lines = [
        "📡 MT4 DATA STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for key, data in sorted(
        snapshot.items()
    ):

        symbol = data.get(
            "symbol",
            "?"
        )

        timeframe = data.get(
            "timeframe",
            "?"
        )

        candles = data.get(
            "candles",
            []
        )

        age = get_data_age(
            data
        )

        lines.append(
            f"• {symbol} | {timeframe} | "
            f"{len(candles)} candles | "
            f"age {age:.0f}s"
        )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    eligible = get_auto_candidates()

    lines.append(
        f"🎯 Eligible M1/M3: {len(eligible)}"
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

    args = context.args

    if len(args) < 2:
        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURCAD.wt M1"
        )
        return

    symbol = normalize_symbol(
        args[0]
    )

    timeframe = normalize_timeframe(
        args[1]
    )

    key = (
        f"{symbol}|{timeframe}"
    )

    with data_lock:
        data = mt4_data.get(
            key
        )

    if not data:
        await update.message.reply_text(
            f"❌ لا توجد بيانات "
            f"{symbol} {timeframe}"
        )
        return

    candidate = {
        "key": key,
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": data.get(
            "candles",
            []
        ),
        "age": get_data_age(
            data
        ),
    }

    await update.message.reply_text(
        f"🔎 جاري تحليل {symbol} {timeframe}..."
    )

    result = analyze_candidate(
        candidate
    )

    if not result:
        await update.message.reply_text(
            "❌ لم يتم اعتماد الصفقة "
            "لأن شروط الجودة لم تتحقق."
        )
        return

    quality = result["quality"]

    quality["_candles"] = (
        get_closed_candles(
            candidate["candles"]
        )
    )

    entry_time = get_next_entry_time(
        timeframe
    )

    entry_price = get_entry_price(
        candidate["candles"]
    )

    card = create_signal_card(
        symbol=symbol,
        timeframe=timeframe,
        quality=quality,
        entry_time=entry_time,
        entry_price=entry_price,
        recovery=False,
    )

    await update.message.reply_text(
        card
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

    def send_json(
        self,
        status,
        payload
    ):
        body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
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

        parsed = urlparse(
            self.path
        )

        if parsed.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                }
            )
            return

        if parsed.path == "/":
            self.send_json(
                200,
                {
                    "status": "running",
                    "service": "ZinoProSignalAI",
                }
            )
            return

        if parsed.path == "/mt4status":

            with data_lock:
                snapshot = dict(
                    mt4_data
                )

            result = []

            for key, data in snapshot.items():

                result.append({
                    "key": key,
                    "symbol": data.get(
                        "symbol"
                    ),
                    "timeframe": data.get(
                        "timeframe"
                    ),
                    "candles": len(
                        data.get(
                            "candles",
                            []
                        )
                    ),
                    "age": get_data_age(
                        data
                    ),
                })

            self.send_json(
                200,
                {
                    "status": "ok",
                    "data": result,
                }
            )

            return

        self.send_json(
            404,
            {
                "error": "not found"
            }
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        if parsed.path != "/mt4":
            self.send_json(
                404,
                {
                    "error": "not found"
                }
            )
            return

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        received_key = (
            self.headers.get(
                "X-API-Key",
                ""
            ).strip()
        )

        if MT4_API_KEY:

            if received_key != MT4_API_KEY:
                self.send_json(
                    401,
                    {
                        "error": "unauthorized"
                    }
                )
                return

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

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

            self.send_json(
                400,
                {
                    "error": "invalid_json",
                    "detail": str(e),
                }
            )

            return

        # ----------------------------------------------------
        # SYMBOL
        # ----------------------------------------------------

        symbol = normalize_symbol(
            payload.get(
                "symbol",
                ""
            )
        )

        timeframe = normalize_timeframe(
            payload.get(
                "timeframe",
                payload.get(
                    "tf",
                    ""
                )
            )
        )

        candles = payload.get(
            "candles",
            []
        )

        if not symbol:
            self.send_json(
                400,
                {
                    "error": "missing_symbol"
                }
            )
            return

        if not timeframe:
            self.send_json(
                400,
                {
                    "error": "missing_timeframe"
                }
            )
            return

        if not isinstance(
            candles,
            list
        ):
            self.send_json(
                400,
                {
                    "error": "candles_must_be_array"
                }
            )
            return

        # Keep latest 200.
        candles = candles[-200:]

        key = (
            f"{symbol}|{timeframe}"
        )

        with data_lock:

            mt4_data[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "updated_at": unix_now(),
            }

        logger.info(
            "MT4 DATA | %s | %s | %s candles",
            symbol,
            timeframe,
            len(candles),
        )

        self.send_json(
            200,
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
            }
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
# PHOTO ANALYSIS
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    if gemini_client is None:
        await update.message.reply_text(
            "❌ Gemini غير متصل."
        )
        return

    photo = update.message.photo

    if not photo:
        return

    await update.message.reply_text(
        "🔎 جاري تحليل الصورة..."
    )

    try:

        file = await context.bot.get_file(
            photo[-1].file_id
        )

        image_bytes = (
            await file.download_as_bytearray()
        )

        prompt = """
You are ZinoProSignalAI.

Analyze this Quotex/TradingView chart screenshot.

Return ONLY valid JSON.

Required:

{
  "symbol": "",
  "timeframe": "M1 or M3",
  "direction": "UP or DOWN",
  "confidence": 0,
  "up_score": 0,
  "down_score": 0,
  "selected_score": 0,
  "score_difference": 0,
  "confirmations": 0,
  "contradictions": 0,
  "reason": ""
}

Rules:
- Never WAIT.
- Never NO SIGNAL.
- Never NEUTRAL.
- Direction must be UP or DOWN.
- Total score exactly 18.
- Selected score >= 11.
- Difference >= 5.
- Confidence 70-89.
- Do not invent indicators.
- Prefer Price Action and Structure.
- EMA 9/21.
- RSI 14.
- Williams %R 14.
- ADX/DI 14.
- Keltner EMA20 / ATR10 / multiplier 5.
- Strong confluence required.
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=bytes(
                        image_bytes
                    ),
                    mime_type="image/jpeg",
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        text_response = getattr(
            response,
            "text",
            ""
        )

        result = json.loads(
            clean_json_text(
                text_response
            )
        )

        direction = str(
            result.get(
                "direction",
                ""
            )
        ).upper()

        if direction not in {
            "UP",
            "DOWN",
        }:
            await update.message.reply_text(
                "❌ لم يتم الحصول على اتجاه صالح."
            )
            return

        confidence = int(
            clamp(
                safe_float(
                    result.get(
                        "confidence",
                        0
                    )
                ),
                0,
                89
            )
        )

        up_score = int(
            safe_float(
                result.get(
                    "up_score",
                    0
                )
            )
        )

        down_score = int(
            safe_float(
                result.get(
                    "down_score",
                    0
                )
            )
        )

        symbol = str(
            result.get(
                "symbol",
                "UNKNOWN"
            )
        ).upper()

        timeframe = normalize_timeframe(
            result.get(
                "timeframe",
                "M1"
            )
        )

        if timeframe not in {
            "M1",
            "M3",
        }:
            timeframe = "M1"

        if (
            up_score
            + down_score
            != 18
        ):
            await update.message.reply_text(
                "❌ التحليل مرفوض: "
                "مجموع النقاط ليس 18."
            )
            return

        if (
            confidence < 70
            or abs(
                up_score
                - down_score
            ) < 5
            or max(
                up_score,
                down_score
            ) < 11
        ):
            await update.message.reply_text(
                "❌ الإشارة ضعيفة ولم "
                "تتجاوز شروط الجودة."
            )
            return

        entry_time = get_next_entry_time(
            timeframe
        )

        reason = str(
            result.get(
                "reason",
                "Strong technical confluence."
            )
        )

        if len(reason) > 180:
            reason = reason[:177] + "..."

        quality = {
            "direction": direction,
            "confidence": confidence,
            "up_score": up_score,
            "down_score": down_score,
            "selected_score": max(
                up_score,
                down_score
            ),
            "score_difference": abs(
                up_score
                - down_score
            ),
            "confirmations": int(
                safe_float(
                    result.get(
                        "confirmations",
                        4
                    )
                )
            ),
            "contradictions": int(
                safe_float(
                    result.get(
                        "contradictions",
                        0
                    )
                )
            ),
            "reason": reason,
            "_candles": [],
        }

        # Screenshot does not contain exact MT4 price data.
        entry_price = 0.0

        card = create_signal_card(
            symbol=symbol,
            timeframe=timeframe,
            quality=quality,
            entry_time=entry_time,
            entry_price=entry_price,
            recovery=False,
        )

        await update.message.reply_text(
            card
        )

    except Exception as e:

        logger.exception(
            "Photo analysis error: %s",
            e
        )

        await update.message.reply_text(
            "❌ حدث خطأ أثناء تحليل الصورة."
        )


# ============================================================
# APPLICATION
# ============================================================

async def post_init(
    application
):
    asyncio.create_task(
        auto_loop(
            application
        )
    )

    logger.info(
        "ZinoProSignalAI automatic loop initialized."
    )


def main():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing."
        )

    if OWNER_ID <= 0:
        raise RuntimeError(
            "OWNER_ID is missing or invalid."
        )

    # Start Render HTTP server.
    server_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    server_thread.start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands.
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
            "history",
            history_command
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

    # Screenshot handler.
    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    logger.info(
        "ZinoProSignalAI is running"
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
