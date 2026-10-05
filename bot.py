import os
import json
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

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
# ZinoProSignalAI
# MT5 -> Render -> Gemini -> Telegram
# ============================================================

BOT_NAME = "ZinoProSignalAI"

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

PORT = int(os.getenv("PORT", "10000"))

# ------------------------------------------------------------
# GENERAL SETTINGS
# ------------------------------------------------------------

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10

# IMPORTANT:
# 0 = allow a new signal every new MT5 candle.
# 60 = one signal per minute.
# We use 30 to avoid blocking direction changes on M1.
SIGNAL_COOLDOWN_SECONDS = 30

SETUP_REPEAT_BLOCK_SECONDS = 360

RECOVERY_LIMIT = 1

AUTO_ANALYSIS_INTERVAL_SECONDS = 30

ALGIERS_TZ = ZoneInfo("Africa/Algiers")

# ------------------------------------------------------------
# OWNER
# ------------------------------------------------------------

try:
    OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW else 0
except Exception:
    OWNER_ID = 0


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(BOT_NAME)


# ============================================================
# GEMINI
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
    except Exception as exc:
        logger.exception("Gemini initialization failed: %s", exc)
else:
    logger.warning("GEMINI_API_KEY is missing")


# ============================================================
# GLOBAL STATE
# ============================================================

mt5_data = {}
mt5_lock = threading.Lock()

stats_lock = threading.Lock()

stats = {
    "wins": 0,
    "losses": 0,
}

current_cycle = None

history = []

last_signal_time = 0.0
last_setup_fingerprint = ""

state_lock = threading.Lock()


# ============================================================
# TIME
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def format_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def timeframe_minutes(timeframe):
    tf = str(timeframe or "M1").upper().strip()

    if tf.startswith("M"):
        try:
            value = int(tf[1:])
            return max(1, value)
        except Exception:
            return 1

    if tf.startswith("H"):
        try:
            value = int(tf[1:])
            return max(1, value * 60)
        except Exception:
            return 60

    return 1


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update: Update):
    if not update or not update.effective_user:
        return False

    if OWNER_ID <= 0:
        return False

    return update.effective_user.id == OWNER_ID


async def owner_only(update: Update):
    if not is_owner(update):
        try:
            await update.effective_message.reply_text(
                "⛔ Unauthorized."
            )
        except Exception:
            pass
        return False

    return True


# ============================================================
# SAFE NUMBERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        result = float(value)

        if result != result:
            return default

        return result

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):
    if not isinstance(c, dict):
        return None

    try:
        t = safe_int(c.get("time"))

        o = safe_float(c.get("open"))
        h = safe_float(c.get("high"))
        l = safe_float(c.get("low"))
        close = safe_float(c.get("close"))

        if t <= 0:
            return None

        if h <= 0 or l <= 0:
            return None

        if o <= 0 or close <= 0:
            return None

        if h < max(o, close):
            return None

        if l > min(o, close):
            return None

        return {
            "time": t,
            "open": o,
            "high": h,
            "low": l,
            "close": close,
            "tick_volume": safe_int(c.get("tick_volume")),
            "real_volume": safe_int(c.get("real_volume")),
            "spread": safe_int(c.get("spread")),
        }

    except Exception:
        return None


def normalize_candles(candles):
    result = []

    if not isinstance(candles, list):
        return result

    for item in candles:
        candle = normalize_candle(item)

        if candle:
            result.append(candle)

    result.sort(key=lambda x: x["time"])

    # Remove duplicate timestamps
    unique = {}

    for candle in result:
        unique[candle["time"]] = candle

    return list(sorted(unique.values(), key=lambda x: x["time"]))


# ============================================================
# MT5 PAYLOAD
# ============================================================

def validate_mt5_payload(payload):
    if not isinstance(payload, dict):
        return False, "Invalid JSON object"

    if MT5_API_KEY:
        supplied_key = str(payload.get("api_key", "")).strip()

        if not supplied_key:
            supplied_key = str(
                payload.get("_api_key", "")
            ).strip()

        if supplied_key != MT5_API_KEY:
            return False, "Invalid API key"

    symbol = str(payload.get("symbol", "")).strip()

    if not symbol:
        return False, "Missing symbol"

    candles = normalize_candles(payload.get("candles"))

    if len(candles) < MIN_CLOSED_CANDLES:
        return False, (
            f"Not enough candles: {len(candles)}"
        )

    return True, ""


def store_mt5_payload(payload):
    symbol = str(
        payload.get("symbol", "")
    ).strip().upper()

    candles = normalize_candles(
        payload.get("candles", [])
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return False

    timeframe = str(
        payload.get("timeframe", ANALYSIS_TIMEFRAME)
    ).upper().strip()

    if not timeframe:
        timeframe = ANALYSIS_TIMEFRAME

    item = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "current_bid": safe_float(
            payload.get("current_bid")
        ),
        "current_ask": safe_float(
            payload.get("current_ask")
        ),
        "digits": safe_int(
            payload.get("digits")
        ),
        "closed_candle_time": safe_int(
            payload.get("closed_candle_time")
        ),
        "batch_id": str(
            payload.get("batch_id", "")
        ),
        "received_at": time.time(),
    }

    with mt5_lock:
        mt5_data[symbol] = item

    return True


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(data):
    if not data:
        return []

    candles = list(
        data.get("candles", [])
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return []

    # MT5 normally sends the currently forming candle as the
    # last candle. Remove it so analysis uses closed candles.
    closed = candles[:-1]

    if len(closed) < MIN_CLOSED_CANDLES:
        closed = candles

    return closed


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    values = [
        safe_float(x)
        for x in values
    ]

    seed = sum(values[:period]) / period

    result = seed

    multiplier = 2.0 / (period + 1.0)

    for value in values[period:]:
        result = (
            (value - result) * multiplier
            + result
        )

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    if len(gains) < period:
        return None

    avg_gain = (
        sum(gains[:period]) / period
    )

    avg_loss = (
        sum(losses[:period]) / period
    )

    for i in range(period, len(gains)):
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

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def williams_r(candles, period=14):
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
        * -100.0
    )


def true_ranges(candles):
    if not candles:
        return []

    trs = []

    previous_close = candles[0]["close"]

    for candle in candles:
        high = candle["high"]
        low = candle["low"]

        tr = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )

        trs.append(tr)

        previous_close = candle["close"]

    return trs


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return (
        sum(trs[-period:])
        / period
    )


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

    plus_dm = []
    minus_dm = []
    trs = []

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
            abs(
                current["high"]
                - previous["close"]
            ),
            abs(
                current["low"]
                - previous["close"]
            ),
        )

        plus_dm.append(pdm)
        minus_dm.append(mdm)
        trs.append(tr)

    if len(trs) < period:
        return None, None, None

    # Custom/simple average matching the old logic.
    tr_avg = sum(trs[-period:]) / period

    if tr_avg <= 0:
        return 0.0, 0.0, 0.0

    plus_avg = (
        sum(plus_dm[-period:])
        / period
    )

    minus_avg = (
        sum(minus_dm[-period:])
        / period
    )

    plus_di = (
        100.0
        * plus_avg
        / tr_avg
    )

    minus_di = (
        100.0
        * minus_avg
        / tr_avg
    )

    denominator = plus_di + minus_di

    if denominator <= 0:
        dx = 0.0
    else:
        dx = (
            100.0
            * abs(plus_di - minus_di)
            / denominator
        )

    return dx, plus_di, minus_di


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(candles):
    if len(candles) < 8:
        return "NEUTRAL"

    recent = candles[-8:]

    first = recent[:4]
    last = recent[4:]

    first_high = max(
        c["high"] for c in first
    )

    last_high = max(
        c["high"] for c in last
    )

    first_low = min(
        c["low"] for c in first
    )

    last_low = min(
        c["low"] for c in last
    )

    if (
        last_high > first_high
        and last_low > first_low
    ):
        return "BULLISH"

    if (
        last_high < first_high
        and last_low < first_low
    ):
        return "BEARISH"

    # Additional close-based structure check.
    first_close = first[-1]["close"]
    last_close = last[-1]["close"]

    if last_close > first_close:
        return "BULLISH"

    if last_close < first_close:
        return "BEARISH"

    return "NEUTRAL"


def breakout_state(candles):
    if len(candles) < 10:
        return "NONE"

    previous = candles[-9:-1]
    last = candles[-1]

    previous_high = max(
        c["high"] for c in previous
    )

    previous_low = min(
        c["low"] for c in previous
    )

    if last["close"] > previous_high:
        return "BULLISH_BREAKOUT"

    if last["close"] < previous_low:
        return "BEARISH_BREAKOUT"

    # Retest/failed breakout behavior.
    if (
        last["high"] > previous_high
        and last["close"] < previous_high
    ):
        return "BEARISH_REJECTION"

    if (
        last["low"] < previous_low
        and last["close"] > previous_low
    ):
        return "BULLISH_REJECTION"

    return "NONE"


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def technical_snapshot(candles):
    closes = [
        c["close"]
        for c in candles
    ]

    if len(closes) < MIN_CLOSED_CANDLES:
        return None

    price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema20 = ema(closes, 20)

    rsi14 = rsi(closes, 14)
    wr14 = williams_r(candles, 14)

    atr10 = atr(candles, 10)

    adx14, plus_di, minus_di = adx_di(
        candles,
        14
    )

    structure = market_structure(
        candles
    )

    breakout = breakout_state(
        candles
    )

    recent8 = candles[-8:]

    recent_low = min(
        c["low"]
        for c in recent8
    )

    recent_high = max(
        c["high"]
        for c in recent8
    )

    if ema20 is not None and atr10 is not None:
        keltner_upper = (
            ema20 + atr10 * 5.0
        )

        keltner_lower = (
            ema20 - atr10 * 5.0
        )
    else:
        keltner_upper = None
        keltner_lower = None

    last = candles[-1]

    body = abs(
        last["close"]
        - last["open"]
    )

    candle_range = (
        last["high"]
        - last["low"]
    )

    if candle_range > 0:
        body_ratio = body / candle_range
    else:
        body_ratio = 0.0

    return {
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "ema20": ema20,
        "rsi14": rsi14,
        "williams_r14": wr14,
        "atr10": atr10,
        "adx14": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,
        "structure": structure,
        "breakout": breakout,
        "recent_low": recent_low,
        "recent_high": recent_high,
        "last_open": last["open"],
        "last_high": last["high"],
        "last_low": last["low"],
        "last_close": last["close"],
        "body_ratio": body_ratio,
    }


# ============================================================
# DETERMINISTIC DIRECTION ENGINE
# ============================================================

def directional_pre_score(snapshot):
    """
    Strong deterministic arbitration.

    This is intentionally separate from Gemini.

    UP and DOWN are calculated independently.
    Gemini cannot force a permanent DOWN bias.
    """

    up = 0
    down = 0

    reasons_up = []
    reasons_down = []

    price = snapshot["price"]

    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    structure = snapshot["structure"]
    breakout = snapshot["breakout"]

    adx14 = snapshot["adx14"]
    plus_di = snapshot["plus_di"]
    minus_di = snapshot["minus_di"]

    rsi14 = snapshot["rsi14"]
    wr14 = snapshot["williams_r14"]

    last_open = snapshot["last_open"]
    last_close = snapshot["last_close"]

    body_ratio = snapshot["body_ratio"]

    # --------------------------------------------------------
    # EMA 9 / 21
    # --------------------------------------------------------

    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            up += 3
            reasons_up.append("EMA9>EMA21")
        elif ema9 < ema21:
            down += 3
            reasons_down.append("EMA9<EMA21")

    # --------------------------------------------------------
    # Price vs EMA9
    # --------------------------------------------------------

    if ema9 is not None:
        if price > ema9:
            up += 1
            reasons_up.append("Price>EMA9")
        elif price < ema9:
            down += 1
            reasons_down.append("Price<EMA9")

    # --------------------------------------------------------
    # Price vs EMA21
    # --------------------------------------------------------

    if ema21 is not None:
        if price > ema21:
            up += 1
            reasons_up.append("Price>EMA21")
        elif price < ema21:
            down += 1
            reasons_down.append("Price<EMA21")

    # --------------------------------------------------------
    # Market Structure
    # --------------------------------------------------------

    if structure == "BULLISH":
        up += 3
        reasons_up.append("Bullish structure")

    elif structure == "BEARISH":
        down += 3
        reasons_down.append("Bearish structure")

    # --------------------------------------------------------
    # Breakout / Retest
    # --------------------------------------------------------

    if breakout == "BULLISH_BREAKOUT":
        up += 3
        reasons_up.append("Bullish breakout")

    elif breakout == "BEARISH_BREAKOUT":
        down += 3
        reasons_down.append("Bearish breakout")

    elif breakout == "BULLISH_REJECTION":
        up += 2
        reasons_up.append("Bullish rejection")

    elif breakout == "BEARISH_REJECTION":
        down += 2
        reasons_down.append("Bearish rejection")

    # --------------------------------------------------------
    # ADX + DI
    # --------------------------------------------------------

    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
    ):
        if adx14 >= 20:
            if plus_di > minus_di:
                up += 2
                reasons_up.append("ADX/DI bullish")
            elif minus_di > plus_di:
                down += 2
                reasons_down.append("ADX/DI bearish")

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi14 is not None:

        if 50 < rsi14 < 70:
            up += 1
            reasons_up.append("RSI bullish zone")

        elif 30 < rsi14 < 50:
            down += 1
            reasons_down.append("RSI bearish zone")

        elif rsi14 >= 70:
            # Do NOT automatically short just because RSI is high.
            # Strong trend can remain overbought.
            if (
                ema9 is not None
                and ema21 is not None
                and ema9 < ema21
            ):
                down += 1
                reasons_down.append(
                    "RSI high + bearish EMA"
                )

        elif rsi14 <= 30:
            # Do NOT automatically buy simply because RSI is low.
            if (
                ema9 is not None
                and ema21 is not None
                and ema9 > ema21
            ):
                up += 1
                reasons_up.append(
                    "RSI low + bullish EMA"
                )

    # --------------------------------------------------------
    # Williams %R
    # --------------------------------------------------------

    if wr14 is not None:

        if wr14 > -50:
            up += 1
            reasons_up.append("Williams bullish")

        elif wr14 < -50:
            down += 1
            reasons_down.append("Williams bearish")

    # --------------------------------------------------------
    # Last candle
    # --------------------------------------------------------

    if last_close > last_open:
        if body_ratio >= 0.30:
            up += 1
            reasons_up.append("Bullish candle")

    elif last_close < last_open:
        if body_ratio >= 0.30:
            down += 1
            reasons_down.append("Bearish candle")

    # --------------------------------------------------------
    # FINAL DIRECTION
    # --------------------------------------------------------

    if up > down:
        direction = "UP"

    elif down > up:
        direction = "DOWN"

    else:
        # Tie breaker:
        # EMA first, then latest candle.
        if (
            ema9 is not None
            and ema21 is not None
        ):
            if ema9 >= ema21:
                direction = "UP"
            else:
                direction = "DOWN"

        elif last_close >= last_open:
            direction = "UP"

        else:
            direction = "DOWN"

    return {
        "direction": direction,
        "up": up,
        "down": down,
        "reasons_up": reasons_up,
        "reasons_down": reasons_down,
    }


# ============================================================
# SCORE CONVERSION
# ============================================================

def convert_pre_score_to_18(pre):
    """
    Convert deterministic direction strength to an /18 display.

    We preserve the direction and keep the total exactly 18.
    """

    up = int(pre["up"])
    down = int(pre["down"])

    total_raw = up + down

    if total_raw <= 0:
        if pre["direction"] == "UP":
            return 10, 8
        return 8, 10

    if pre["direction"] == "UP":
        base = 9 + min(8, max(1, up - down))
        up_score = clamp(base, 9, 18)
        down_score = 18 - up_score
    else:
        base = 9 + min(8, max(1, down - up))
        down_score = clamp(base, 9, 18)
        up_score = 18 - down_score

    return int(up_score), int(down_score)


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score,
):
    recent = candles[-40:]

    compact = []

    for c in recent:
        compact.append({
            "time": c["time"],
            "open": round(c["open"], 8),
            "high": round(c["high"], 8),
            "low": round(c["low"], 8),
            "close": round(c["close"], 8),
            "volume": c.get("tick_volume", 0),
        })

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": compact,
        "technical_snapshot": {
            "price": snapshot["price"],
            "ema9": snapshot["ema9"],
            "ema21": snapshot["ema21"],
            "rsi14": snapshot["rsi14"],
            "williams_r14": snapshot["williams_r14"],
            "atr10": snapshot["atr10"],
            "adx14": snapshot["adx14"],
            "plus_di": snapshot["plus_di"],
            "minus_di": snapshot["minus_di"],
            "keltner_upper": snapshot["keltner_upper"],
            "keltner_lower": snapshot["keltner_lower"],
            "structure": snapshot["structure"],
            "breakout": snapshot["breakout"],
            "body_ratio": snapshot["body_ratio"],
        },
        "deterministic_engine": {
            "direction": pre_score["direction"],
            "up_score_raw": pre_score["up"],
            "down_score_raw": pre_score["down"],
            "up_reasons": pre_score["reasons_up"],
            "down_reasons": pre_score["reasons_down"],
        },
    }

    prompt = f"""
You are the technical validation engine for {BOT_NAME}.

IMPORTANT:
This is a short-term binary-options style directional analysis.
It is NOT guaranteed and must not claim certainty.

You MUST choose exactly one:
UP
DOWN

Never output WAIT.
Never output NEUTRAL.
Never invent missing data.

The deterministic engine has already calculated both directions.
Do NOT automatically favor DOWN.
Do NOT automatically favor UP.

The final direction must be based on the actual market evidence.

Priority:
1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity behavior
5. Momentum
6. Candle behavior
7. EMA 9/21
8. RSI 14
9. Williams %R 14
10. Keltner
11. ADX/DI

Indicators:
EMA 9
EMA 21
RSI 14
Williams %R 14
ADX 14
DI 14
Keltner EMA20 / ATR10 / multiplier 5

Do not invent support/resistance levels.

Scoring:
The returned UP score + DOWN score MUST equal exactly 18.

Confidence:
- Must be between 50 and 89 for normal signals.
- Do not give 90%+ unless there is exceptionally strong multi-factor confluence.
- Confidence is NOT probability and NOT a guarantee.

Direction arbitration:
The deterministic engine is important.
If its direction is supported by the candles and indicators, keep it.
If Gemini believes the opposite direction is clearly stronger, it may switch,
but it must explain why using actual supplied data.

Cancellation:
For UP, cancellation should be related to a meaningful recent closed-candle low.
For DOWN, cancellation should be related to a meaningful recent closed-candle high.

Return ONLY valid JSON.

JSON:
{{
  "signal": "UP or DOWN",
  "direction": "UP or DOWN",
  "confidence": 50,
  "up_score": 9,
  "down_score": 9,
  "reason": "short technical reason",
  "cancellation_reason": "short cancellation condition"
}}

DATA:
{json.dumps(payload, ensure_ascii=False)}
"""

    return prompt


# ============================================================
# GEMINI CALL
# ============================================================

def call_gemini(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score,
):
    if gemini_client is None:
        return None

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre_score,
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

        text = getattr(
            response,
            "text",
            ""
        )

        if not text:
            return None

        text = text.strip()

        if text.startswith("```"):
            text = text.replace(
                "```json",
                "",
                1
            ).replace(
                "```",
                ""
            ).strip()

        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        return result

    except Exception as exc:
        logger.exception(
            "Gemini analysis failed: %s",
            exc
        )
        return None


# ============================================================
# VALIDATE GEMINI RESULT
# ============================================================

def validate_gemini_result(
    result,
    pre_score,
):
    deterministic_direction = (
        pre_score["direction"]
    )

    deterministic_up = (
        pre_score["up"]
    )

    deterministic_down = (
        pre_score["down"]
    )

    fallback_up, fallback_down = (
        convert_pre_score_to_18(pre_score)
    )

    if not isinstance(result, dict):
        result = {}

    direction = str(
        result.get(
            "direction",
            result.get("signal", "")
        )
    ).upper().strip()

    if direction not in ("UP", "DOWN"):
        direction = deterministic_direction

    confidence = safe_int(
        result.get("confidence"),
        65
    )

    confidence = clamp(
        confidence,
        50,
        89
    )

    up_score = safe_int(
        result.get("up_score"),
        fallback_up
    )

    down_score = safe_int(
        result.get("down_score"),
        fallback_down
    )

    up_score = clamp(
        up_score,
        0,
        18
    )

    down_score = clamp(
        down_score,
        0,
        18
    )

    # --------------------------------------------------------
    # Deterministic direction protection
    # --------------------------------------------------------

    raw_gap = abs(
        deterministic_up
        - deterministic_down
    )

    # If deterministic engine has a clear advantage,
    # Gemini cannot flip the signal just from a weak opinion.
    if raw_gap >= 3:
        direction = deterministic_direction

    # --------------------------------------------------------
    # Force scores to sum to 18
    # --------------------------------------------------------

    if up_score + down_score != 18:

        if direction == "UP":
            up_score = clamp(
                max(
                    fallback_up,
                    up_score
                ),
                0,
                18
            )

            down_score = 18 - up_score

        else:
            down_score = clamp(
                max(
                    fallback_down,
                    down_score
                ),
                0,
                18
            )

            up_score = 18 - down_score

    # --------------------------------------------------------
    # Score direction must agree
    # --------------------------------------------------------

    if direction == "UP":
        if up_score <= down_score:
            up_score = max(
                up_score,
                down_score + 1
            )

            up_score = clamp(
                up_score,
                1,
                18
            )

            down_score = 18 - up_score

    else:
        if down_score <= up_score:
            down_score = max(
                down_score,
                up_score + 1
            )

            down_score = clamp(
                down_score,
                1,
                18
            )

            up_score = 18 - down_score

    # --------------------------------------------------------
    # Confidence based on actual score gap
    # --------------------------------------------------------

    gap = abs(
        up_score - down_score
    )

    if gap <= 1:
        confidence = min(
            confidence,
            60
        )

    elif gap <= 3:
        confidence = min(
            confidence,
            68
        )

    elif gap <= 5:
        confidence = min(
            confidence,
            75
        )

    elif gap <= 7:
        confidence = min(
            confidence,
            82
        )

    else:
        confidence = min(
            confidence,
            88
        )

    if confidence >= 90:
        confidence = 89

    reason = str(
        result.get(
            "reason",
            ""
        )
    ).strip()

    cancellation_reason = str(
        result.get(
            "cancellation_reason",
            ""
        )
    ).strip()

    if not reason:
        if direction == "UP":
            reason = (
                "Bullish price action and "
                "directional confluence."
            )
        else:
            reason = (
                "Bearish price action and "
                "directional confluence."
            )

    if not cancellation_reason:
        if direction == "UP":
            cancellation_reason = (
                "إلغاء إذا أغلقت شمعة تحت مستوى الإلغاء."
            )
        else:
            cancellation_reason = (
                "إلغاء إذا أغلقت شمعة فوق مستوى الإلغاء."
            )

    return {
        "signal": direction,
        "direction": direction,
        "confidence": int(confidence),
        "up_score": int(up_score),
        "down_score": int(down_score),
        "reason": reason,
        "cancellation_reason": cancellation_reason,
    }


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_market(data):
    if not data:
        return None

    symbol = str(
        data.get("symbol", "")
    ).upper().strip()

    timeframe = str(
        data.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
    ).upper().strip()

    candles = get_closed_candles(data)

    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    snapshot = technical_snapshot(
        candles
    )

    if not snapshot:
        return None

    pre_score = directional_pre_score(
        snapshot
    )

    gemini_result = call_gemini(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre_score,
    )

    validated = validate_gemini_result(
        gemini_result,
        pre_score,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Final direction remains data-driven.
    #
    # If the deterministic engine has strong evidence,
    # preserve it.
    # --------------------------------------------------------

    if abs(
        pre_score["up"]
        - pre_score["down"]
    ) >= 3:

        final_direction = (
            pre_score["direction"]
        )

        up_score, down_score = (
            convert_pre_score_to_18(
                pre_score
            )
        )

        validated["direction"] = (
            final_direction
        )

        validated["signal"] = (
            final_direction
        )

        validated["up_score"] = up_score
        validated["down_score"] = down_score

        gap = abs(
            up_score
            - down_score
        )

        if gap <= 1:
            validated["confidence"] = min(
                validated["confidence"],
                60
            )
        elif gap <= 3:
            validated["confidence"] = min(
                validated["confidence"],
                68
            )
        elif gap <= 5:
            validated["confidence"] = min(
                validated["confidence"],
                75
            )
        elif gap <= 7:
            validated["confidence"] = min(
                validated["confidence"],
                82
            )
        else:
            validated["confidence"] = min(
                validated["confidence"],
                88
            )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "snapshot": snapshot,
        "pre_score": pre_score,
        "analysis": validated,
        "received_at": data.get(
            "received_at",
            time.time()
        ),
    }


# ============================================================
# ENTRY CALCULATION
# ============================================================

def calculate_entry(analysis):
    candles = analysis["candles"]

    snapshot = analysis["snapshot"]

    direction = analysis[
        "analysis"
    ]["direction"]

    timeframe = analysis[
        "timeframe"
    ]

    delay_minutes = timeframe_minutes(
        timeframe
    )

    # Entry price = latest CLOSED candle close.
    entry_price = candles[-1]["close"]

    recent = candles[-8:]

    if direction == "UP":
        cancellation_level = min(
            c["low"]
            for c in recent
        )

        cancellation_text = (
            f"إلغاء إذا أغلقت شمعة تحت "
            f"{cancellation_level:.8f}"
        )

    else:
        cancellation_level = max(
            c["high"]
            for c in recent
        )

        cancellation_text = (
            f"إلغاء إذا أغلقت شمعة فوق "
            f"{cancellation_level:.8f}"
        )

    now = now_algiers()

    # Round to current minute.
    base_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        base_time
        + timedelta(minutes=delay_minutes)
    )

    return {
        "entry_price": entry_price,
        "cancellation_level": cancellation_level,
        "cancellation_text": cancellation_text,
        "entry_time": entry_time,
        "delay_minutes": delay_minutes,
    }


# ============================================================
# SETUP FINGERPRINT
# ============================================================

def make_setup_fingerprint(
    analysis
):
    symbol = analysis["symbol"]

    timeframe = analysis["timeframe"]

    direction = analysis[
        "analysis"
    ]["direction"]

    candle_time = analysis[
        "candles"
    ][-1]["time"]

    return (
        f"{symbol}|"
        f"{timeframe}|"
        f"{candle_time}|"
        f"{direction}"
    )


# ============================================================
# SIGNAL CREATION
# ============================================================

def create_signal(
    analysis,
    trade_type="BASE",
    recovery_number=0,
):
    global last_signal_time
    global last_setup_fingerprint
    global current_cycle

    signal = analysis["analysis"]

    entry = calculate_entry(
        analysis
    )

    fingerprint = (
        make_setup_fingerprint(
            analysis
        )
    )

    current_time = time.time()

    with state_lock:

        # Duplicate exact setup.
        if (
            fingerprint
            == last_setup_fingerprint
        ):
            return None

        # Cooldown only applies to BASE signals.
        # Recovery has its own fresh analysis.
        if (
            trade_type == "BASE"
            and SIGNAL_COOLDOWN_SECONDS > 0
            and current_time
            - last_signal_time
            < SIGNAL_COOLDOWN_SECONDS
        ):
            return None

        last_setup_fingerprint = (
            fingerprint
        )

        last_signal_time = (
            current_time
        )

        cycle_id = (
            f"{analysis['symbol']}_"
            f"{analysis['candles'][-1]['time']}_"
            f"{int(current_time)}"
        )

        current_cycle = {
            "cycle_id": cycle_id,
            "symbol": analysis["symbol"],
            "timeframe": analysis["timeframe"],
            "direction": signal["direction"],
            "confidence": signal["confidence"],
            "up_score": signal["up_score"],
            "down_score": signal["down_score"],
            "entry_price": entry["entry_price"],
            "cancellation_level": entry[
                "cancellation_level"
            ],
            "entry_time": entry["entry_time"].isoformat(),
            "trade_type": trade_type,
            "recovery_number": recovery_number,
            "trade_number": (
                1
                if trade_type == "BASE"
                else recovery_number + 1
            ),
            "status": "PENDING",
            "reason": signal["reason"],
            "cancellation_text": entry[
                "cancellation_text"
            ],
            "created_at": now_algiers().isoformat(),
            "analysis": analysis,
        }

    return current_cycle.copy()


# ============================================================
# HISTORY
# ============================================================

def add_history(cycle, result):
    item = {
        "cycle_id": cycle.get(
            "cycle_id",
            ""
        ),
        "symbol": cycle.get(
            "symbol",
            ""
        ),
        "timeframe": cycle.get(
            "timeframe",
            ""
        ),
        "trade_type": cycle.get(
            "trade_type",
            "BASE"
        ),
        "recovery_number": cycle.get(
            "recovery_number",
            0
        ),
        "direction": cycle.get(
            "direction",
            ""
        ),
        "confidence": cycle.get(
            "confidence",
            0
        ),
        "up_score": cycle.get(
            "up_score",
            0
        ),
        "down_score": cycle.get(
            "down_score",
            0
        ),
        "entry_price": cycle.get(
            "entry_price",
            0
        ),
        "status": result,
        "created_at": cycle.get(
            "created_at",
            ""
        ),
        "closed_at": now_algiers().isoformat(),
    }

    with state_lock:
        history.append(item)

        if len(history) > 100:
            del history[:-100]


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal_message(cycle):
    direction = cycle["direction"]

    if direction == "UP":
        emoji = "🟢"
        word = "UP"
    else:
        emoji = "🔴"
        word = "DOWN"

    trade_type = cycle.get(
        "trade_type",
        "BASE"
    )

    recovery_number = cycle.get(
        "recovery_number",
        0
    )

    if trade_type == "RECOVERY":
        trade_label = (
            f"🔁 RECOVERY "
            f"{recovery_number}/1"
        )
    else:
        trade_label = "🎯 BASE TRADE"

    entry_dt = datetime.fromisoformat(
        cycle["entry_time"]
    )

    entry_dt = entry_dt.astimezone(
        ALGIERS_TZ
    )

    price = safe_float(
        cycle["entry_price"]
    )

    cancel = safe_float(
        cycle["cancellation_level"]
    )

    return (
        f"🎓 {BOT_NAME}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 {cycle['symbol']} | "
        f"{cycle['timeframe']}\n\n"
        f"{trade_label}\n"
        f"{emoji} {word}\n\n"
        f"🔥 Confidence: "
        f"{cycle['confidence']}%\n"
        f"🟢 UP Score: "
        f"{cycle['up_score']}/18\n"
        f"🔴 DOWN Score: "
        f"{cycle['down_score']}/18\n\n"
        f"⏱️ Entry after: "
        f"{cycle.get('analysis', {}).get('timeframe', 'M1')} "
        f"delay\n"
        f"🕐 ENTRY TIME: "
        f"{entry_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"💰 ENTRY PRICE: "
        f"{price:.8f}\n"
        f"⚠️ CANCEL LEVEL: "
        f"{cancel:.8f}\n"
        f"   {cycle['cancellation_text']}\n\n"
        f"🧠 Reason:\n"
        f"{cycle['reason']}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# AUTO ANALYSIS
# ============================================================

def get_best_symbol():
    with mt5_lock:
        items = list(
            mt5_data.values()
        )

    best = None
    best_score = -999

    for data in items:
        try:
            candles = get_closed_candles(
                data
            )

            if len(candles) < MIN_CLOSED_CANDLES:
                continue

            snapshot = technical_snapshot(
                candles
            )

            if not snapshot:
                continue

            pre = directional_pre_score(
                snapshot
            )

            strength = (
                abs(
                    pre["up"]
                    - pre["down"]
                )
                + max(
                    pre["up"],
                    pre["down"]
                )
            )

            if strength > best_score:
                best_score = strength
                best = data

        except Exception:
            logger.exception(
                "get_best_symbol failed"
            )

    return best


# ============================================================
# ACTIVE CYCLE PROTECTION
# ============================================================

def has_pending_cycle():
    with state_lock:
        if not current_cycle:
            return False

        return (
            current_cycle.get(
                "status"
            )
            == "PENDING"
        )


# ============================================================
# AUTO ANALYSIS THREAD
# ============================================================

telegram_application = None


async def send_cycle_to_owner(cycle):
    global telegram_application

    if not telegram_application:
        return

    if OWNER_ID <= 0:
        return

    try:
        await telegram_application.bot.send_message(
            chat_id=OWNER_ID,
            text=format_signal_message(
                cycle
            ),
        )

    except Exception as exc:
        logger.exception(
            "Telegram send failed: %s",
            exc
        )


def run_async(coro):
    try:
        return asyncio.run(coro)
    except Exception:
        logger.exception(
            "Async execution failed"
        )
        return None


def auto_analysis_once():
    try:
        # Do not overwrite an active trade cycle.
        if has_pending_cycle():
            return

        data = get_best_symbol()

        if not data:
            return

        analysis = analyze_market(
            data
        )

        if not analysis:
            return

        cycle = create_signal(
            analysis,
            trade_type="BASE",
            recovery_number=0,
        )

        if cycle:
            logger.info(
                "NEW SIGNAL: %s %s %s",
                cycle["symbol"],
                cycle["timeframe"],
                cycle["direction"],
            )

            run_async(
                send_cycle_to_owner(
                    cycle
                )
            )

    except Exception:
        logger.exception(
            "Auto analysis failed"
        )


def auto_analysis_loop():
    logger.info(
        "Auto analysis loop started"
    )

    while True:
        try:
            auto_analysis_once()
        except Exception:
            logger.exception(
                "Auto loop exception"
            )

        time.sleep(
            AUTO_ANALYSIS_INTERVAL_SECONDS
        )


# ============================================================
# WIN / LOSS
# ============================================================

def mark_win():
    global current_cycle

    with state_lock:
        if not current_cycle:
            return None

        if current_cycle.get(
            "status"
        ) != "PENDING":
            return None

        current_cycle["status"] = "WIN"

        finished = current_cycle.copy()

    with stats_lock:
        stats["wins"] += 1

    add_history(
        finished,
        "WIN"
    )

    return finished


def mark_loss():
    global current_cycle

    with state_lock:
        if not current_cycle:
            return None

        if current_cycle.get(
            "status"
        ) != "PENDING":
            return None

        losing_cycle = (
            current_cycle.copy()
        )

    with stats_lock:
        stats["losses"] += 1

    add_history(
        losing_cycle,
        "LOSS"
    )

    # --------------------------------------------------------
    # BASE LOSS:
    # Fresh recovery analysis.
    # --------------------------------------------------------

    if (
        losing_cycle.get(
            "trade_type"
        ) == "BASE"
        and RECOVERY_LIMIT >= 1
    ):
        symbol = losing_cycle[
            "symbol"
        ]

        with mt5_lock:
            latest_data = mt5_data.get(
                symbol
            )

        if latest_data:
            recovery_analysis = (
                analyze_market(
                    latest_data
                )
            )

            if recovery_analysis:

                recovery_cycle = (
                    create_signal(
                        recovery_analysis,
                        trade_type="RECOVERY",
                        recovery_number=1,
                    )
                )

                if recovery_cycle:

                    # Important:
                    # Recovery direction is allowed to
                    # change from DOWN -> UP or UP -> DOWN.
                    return recovery_cycle

        # If fresh recovery analysis unavailable,
        # close cycle.
        with state_lock:
            current_cycle = None

        return None

    # --------------------------------------------------------
    # Recovery loss:
    # Stop cycle.
    # --------------------------------------------------------

    with state_lock:
        current_cycle = None

    return None


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    text = (
        f"🎓 {BOT_NAME}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🟢 MT5 bridge: ACTIVE\n"
        f"🧠 Gemini: "
        f"{'ACTIVE' if gemini_client else 'OFF'}\n"
        f"📊 Timeframe: "
        f"{ANALYSIS_TIMEFRAME}\n\n"
        f"Commands:\n"
        f"/mt5status\n"
        f"/analyze\n"
        f"/stats\n"
        f"/history\n"
        f"/win\n"
        f"/loss\n"
        f"/reset\n"
    )

    await update.message.reply_text(
        text
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    with stats_lock:
        wins = stats["wins"]
        losses = stats["losses"]

    total = wins + losses

    if total > 0:
        winrate = (
            wins / total * 100
        )
    else:
        winrate = 0.0

    with state_lock:
        cycle = (
            current_cycle.copy()
            if current_cycle
            else None
        )

    active_text = "NONE"

    if cycle:
        active_text = (
            f"{cycle['symbol']} "
            f"{cycle['direction']} "
            f"{cycle['status']}"
        )

    text = (
        f"📊 {BOT_NAME} STATS\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {winrate:.1f}%\n\n"
        f"🔄 Active cycle:\n"
        f"{active_text}"
    )

    await update.message.reply_text(
        text
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    with state_lock:
        items = list(
            history[-HISTORY_DISPLAY_COUNT:]
        )

    if not items:
        await update.message.reply_text(
            "📚 History empty."
        )
        return

    lines = [
        f"📚 {BOT_NAME} HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in reversed(items):

        status = item.get(
            "status",
            ""
        )

        if status == "WIN":
            icon = "🟢"
        elif status == "LOSS":
            icon = "🔴"
        else:
            icon = "🟡"

        trade = item.get(
            "trade_type",
            "BASE"
        )

        recovery = item.get(
            "recovery_number",
            0
        )

        if trade == "RECOVERY":
            label = (
                f"RECOVERY {recovery}/1"
            )
        else:
            label = "BASE"

        lines.append(
            f"{icon} "
            f"{item.get('symbol', '')} "
            f"{item.get('timeframe', '')}\n"
            f"   {label} | "
            f"{item.get('direction', '')}\n"
            f"   🎯 {item.get('confidence', 0)}% | "
            f"📈 {item.get('up_score', 0)}/18 "
            f"📉 {item.get('down_score', 0)}/18\n"
            f"   💰 "
            f"{safe_float(item.get('entry_price')):.8f}\n"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    cycle = mark_win()

    if not cycle:
        await update.message.reply_text(
            "⚠️ No pending trade."
        )
        return

    await update.message.reply_text(
        "🟢 WIN recorded.\n"
        f"{cycle['symbol']} "
        f"{cycle['direction']}"
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    result = mark_loss()

    if not result:
        await update.message.reply_text(
            "🔴 LOSS recorded.\n"
            "No recovery signal was generated."
        )
        return

    # If result contains a new cycle,
    # it is the fresh Recovery.
    if (
        isinstance(result, dict)
        and result.get("trade_type")
        == "RECOVERY"
    ):
        await update.message.reply_text(
            "🔴 BASE LOSS recorded.\n\n"
            "🔁 Fresh Recovery analysis:\n\n"
            + format_signal_message(
                result
            )
        )

        return

    await update.message.reply_text(
        "🔴 LOSS recorded."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global current_cycle
    global last_signal_time
    global last_setup_fingerprint

    if not await owner_only(update):
        return

    with stats_lock:
        stats["wins"] = 0
        stats["losses"] = 0

    with state_lock:
        history.clear()
        current_cycle = None
        last_signal_time = 0.0
        last_setup_fingerprint = ""

    await update.message.reply_text(
        "♻️ All statistics, history and "
        "active cycle have been reset."
    )


# ============================================================
# MT5 STATUS
# ============================================================

def build_mt5_status():
    with mt5_lock:
        items = list(
            mt5_data.values()
        )

    lines = [
        f"🟢 {BOT_NAME} MT5 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    if not items:
        lines.append(
            "❌ No MT5 data received."
        )
        return "\n".join(lines)

    for data in items:
        symbol = data.get(
            "symbol",
            ""
        )

        timeframe = data.get(
            "timeframe",
            ""
        )

        candles = get_closed_candles(
            data
        )

        received = data.get(
            "received_at",
            0
        )

        age = (
            time.time() - received
            if received
            else 999999
        )

        if age < 90:
            status = "🟢 LIVE"
        elif age < 300:
            status = "🟡 OLD"
        else:
            status = "🔴 STALE"

        lines.append(
            f"{status} {symbol} | "
            f"{timeframe} | "
            f"{len(candles)} closed candles | "
            f"{int(age)}s"
        )

    return "\n".join(lines)


async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        build_mt5_status()
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        build_mt5_status()
    )


# ============================================================
# MANUAL ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    # If a trade is pending, don't overwrite it.
    if has_pending_cycle():
        await update.message.reply_text(
            "⚠️ There is already a pending trade.\n"
            "Use /win or /loss first."
        )
        return

    data = get_best_symbol()

    if not data:
        await update.message.reply_text(
            "❌ No valid MT5 data available."
        )
        return

    analysis = analyze_market(
        data
    )

    if not analysis:
        await update.message.reply_text(
            "❌ Analysis failed."
        )
        return

    cycle = create_signal(
        analysis,
        trade_type="BASE",
        recovery_number=0,
    )

    if not cycle:
        await update.message.reply_text(
            "⚠️ Signal blocked by cooldown "
            "or duplicate setup."
        )
        return

    await update.message.reply_text(
        format_signal_message(
            cycle
        )
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🟢 ZinoProSignalAI is running.\n\n"
        "MT5 sends market data automatically.\n"
        "Use /analyze for a manual analysis."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        "📷 Image analysis is disabled.\n"
        "This version uses MT5 candle data directly."
    )


# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format_string,
        *args
    ):
        return

    def send_json(
        self,
        code,
        payload
    ):
        body = json.dumps(
            payload,
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

    def send_text(
        self,
        code,
        text
    ):
        body = text.encode("utf-8")

        self.send_response(code)

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

    def do_GET(self):

        path = urlparse(
            self.path
        ).path

        if path in (
            "/",
            "/health",
            "/healthz",
        ):
            self.send_text(
                200,
                f"{BOT_NAME} is running"
            )
            return

        if path in (
            "/mt5",
            "/api/mt5",
            "/mt4",
            "/api/mt4",
        ):
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": BOT_NAME,
                    "endpoint": path,
                    "symbols": list(
                        mt5_data.keys()
                    ),
                }
            )
            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "Not found",
            }
        )

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

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
                }
            )
            return

        try:
            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            if content_length <= 0:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": "Empty body",
                    }
                )
                return

            if content_length > 5_000_000:
                self.send_json(
                    413,
                    {
                        "ok": False,
                        "error": "Payload too large",
                    }
                )
                return

            raw = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw.decode("utf-8")
            )

            # Header API key support.
            if MT5_API_KEY:
                header_key = self.headers.get(
                    "X-MT5-API-Key",
                    ""
                ).strip()

                if header_key:
                    payload["api_key"] = (
                        header_key
                    )

            valid, error = (
                validate_mt5_payload(
                    payload
                )
            )

            if not valid:
                status = (
                    401
                    if "API key" in error
                    else 400
                )

                self.send_json(
                    status,
                    {
                        "ok": False,
                        "error": error,
                    }
                )

                return

            stored = store_mt5_payload(
                payload
            )

            if not stored:
                self.send_json(
                    400,
                    {
                        "ok": False,
                        "error": (
                            "Could not store MT5 data"
                        ),
                    }
                )
                return

            symbol = str(
                payload.get(
                    "symbol",
                    ""
                )
            ).upper()

            timeframe = str(
                payload.get(
                    "timeframe",
                    "M1"
                )
            ).upper()

            candles_count = len(
                normalize_candles(
                    payload.get(
                        "candles",
                        []
                    )
                )
            )

            logger.info(
                "MT5 data received | "
                "%s | %s | %s candles",
                symbol,
                timeframe,
                candles_count,
            )

            self.send_json(
                200,
                {
                    "ok": True,
                    "service": BOT_NAME,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": candles_count,
                    "message": "MT5 data accepted",
                }
            )

        except json.JSONDecodeError:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": "Invalid JSON",
                }
            )

        except Exception as exc:
            logger.exception(
                "POST handler failed"
            )

            self.send_json(
                500,
                {
                    "ok": False,
                    "error": str(exc),
                }
            )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# TELEGRAM APPLICATION
# ============================================================

def build_application():
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
            "mt5status",
            mt5status_command
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

    return application


# ============================================================
# MAIN
# ============================================================

def main():
    global telegram_application

    logger.info(
        "Starting %s",
        BOT_NAME
    )

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN environment variable is missing"
        )

    if not OWNER_ID:
        logger.warning(
            "OWNER_ID is missing or invalid"
        )

    if not MT5_API_KEY:
        logger.warning(
            "MT5_API_KEY is empty. "
            "MT5 authentication is disabled."
        )

    # HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="HTTPServer",
    )

    http_thread.start()

    # Auto analysis
    analysis_thread = threading.Thread(
        target=auto_analysis_loop,
        daemon=True,
        name="AutoAnalysis",
    )

    analysis_thread.start()

    # Telegram
    telegram_application = (
        build_application()
    )

    logger.info(
        "Telegram polling starting"
    )

    telegram_application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
