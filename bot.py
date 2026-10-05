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


# ============================================================
# ENV
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

PORT = int(os.getenv("PORT", "10000"))


# ============================================================
# GENERAL SETTINGS
# ============================================================

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10

# One analysis per new setup.
SIGNAL_COOLDOWN_SECONDS = 30

SETUP_REPEAT_BLOCK_SECONDS = 360

RECOVERY_LIMIT = 1

AUTO_ANALYSIS_INTERVAL_SECONDS = 30

ALGIERS_TZ = ZoneInfo("Africa/Algiers")


# ============================================================
# OWNER
# ============================================================

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
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        logger.info(
            "Gemini client initialized"
        )

    except Exception as exc:
        logger.exception(
            "Gemini initialization failed: %s",
            exc
        )
else:
    logger.warning(
        "GEMINI_API_KEY is missing"
    )


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
# TELEGRAM EVENT LOOP
# ============================================================

telegram_application = None
telegram_loop = None


# ============================================================
# TIME
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def format_dt(dt):
    return dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def timeframe_minutes(timeframe):
    tf = str(
        timeframe or "M1"
    ).upper().strip()

    if tf.startswith("M"):
        try:
            value = int(tf[1:])
            return max(1, value)
        except Exception:
            return 1

    if tf.startswith("H"):
        try:
            value = int(tf[1:])
            return max(
                1,
                value * 60
            )
        except Exception:
            return 60

    return 1


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update: Update):
    if not update:
        return False

    if not update.effective_user:
        return False

    if OWNER_ID <= 0:
        return False

    return (
        update.effective_user.id
        == OWNER_ID
    )


async def owner_only(update: Update):
    if not is_owner(update):
        try:
            if update.effective_message:
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
    return max(
        minimum,
        min(maximum, value)
    )


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(c):
    if not isinstance(c, dict):
        return None

    try:
        t = safe_int(
            c.get("time")
        )

        o = safe_float(
            c.get("open")
        )

        h = safe_float(
            c.get("high")
        )

        l = safe_float(
            c.get("low")
        )

        close = safe_float(
            c.get("close")
        )

        if t <= 0:
            return None

        if (
            h <= 0
            or l <= 0
            or o <= 0
            or close <= 0
        ):
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
            "tick_volume": safe_int(
                c.get("tick_volume")
            ),
            "real_volume": safe_int(
                c.get("real_volume")
            ),
            "spread": safe_int(
                c.get("spread")
            ),
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

    result.sort(
        key=lambda x: x["time"]
    )

    unique = {}

    for candle in result:
        unique[candle["time"]] = candle

    return list(
        sorted(
            unique.values(),
            key=lambda x: x["time"]
        )
    )


# ============================================================
# MT5 PAYLOAD
# ============================================================

def validate_mt5_payload(payload):
    if not isinstance(payload, dict):
        return False, "Invalid JSON object"

    if MT5_API_KEY:
        supplied_key = str(
            payload.get(
                "api_key",
                ""
            )
        ).strip()

        if not supplied_key:
            supplied_key = str(
                payload.get(
                    "_api_key",
                    ""
                )
            ).strip()

        if supplied_key != MT5_API_KEY:
            return False, "Invalid API key"

    symbol = str(
        payload.get(
            "symbol",
            ""
        )
    ).strip()

    if not symbol:
        return False, "Missing symbol"

    candles = normalize_candles(
        payload.get("candles")
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return False, (
            f"Not enough candles: "
            f"{len(candles)}"
        )

    return True, ""


def store_mt5_payload(payload):
    symbol = str(
        payload.get(
            "symbol",
            ""
        )
    ).strip().upper()

    candles = normalize_candles(
        payload.get(
            "candles",
            []
        )
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return False

    timeframe = str(
        payload.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
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
            payload.get(
                "closed_candle_time"
            )
        ),
        "batch_id": str(
            payload.get(
                "batch_id",
                ""
            )
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
        data.get(
            "candles",
            []
        )
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return []

    # MT5 sends current forming candle last.
    # Analysis uses CLOSED candles only.
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

    seed = (
        sum(values[:period])
        / period
    )

    result = seed

    multiplier = (
        2.0
        / (period + 1.0)
    )

    for value in values[period:]:
        result = (
            (value - result)
            * multiplier
            + result
        )

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(
        1,
        len(values)
    ):
        diff = (
            values[i]
            - values[i - 1]
        )

        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(
                abs(diff)
            )

    if len(gains) < period:
        return None

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
            (
                avg_gain
                * (period - 1)
            )
            + gains[i]
        ) / period

        avg_loss = (
            (
                avg_loss
                * (period - 1)
            )
            + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = (
        avg_gain
        / avg_loss
    )

    return 100.0 - (
        100.0
        / (1.0 + rs)
    )


def williams_r(
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
        * -100.0
    )


def true_ranges(candles):
    if not candles:
        return []

    trs = []

    previous_close = (
        candles[0]["close"]
    )

    for candle in candles:
        high = candle["high"]
        low = candle["low"]

        tr = max(
            high - low,
            abs(
                high
                - previous_close
            ),
            abs(
                low
                - previous_close
            ),
        )

        trs.append(tr)

        previous_close = (
            candle["close"]
        )

    return trs


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return (
        sum(trs[-period:])
        / period
    )


def adx_di(
    candles,
    period=14
):
    if len(candles) < period + 2:
        return None, None, None

    plus_dm = []
    minus_dm = []
    trs = []

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

        if (
            up_move > down_move
            and up_move > 0
        ):
            pdm = up_move
        else:
            pdm = 0.0

        if (
            down_move > up_move
            and down_move > 0
        ):
            mdm = down_move
        else:
            mdm = 0.0

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

        plus_dm.append(pdm)
        minus_dm.append(mdm)
        trs.append(tr)

    if len(trs) < period:
        return None, None, None

    tr_avg = (
        sum(trs[-period:])
        / period
    )

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

    denominator = (
        plus_di
        + minus_di
    )

    if denominator <= 0:
        dx = 0.0
    else:
        dx = (
            100.0
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
        return "NEUTRAL"

    recent = candles[-8:]

    first = recent[:4]
    last = recent[4:]

    first_high = max(
        c["high"]
        for c in first
    )

    last_high = max(
        c["high"]
        for c in last
    )

    first_low = min(
        c["low"]
        for c in first
    )

    last_low = min(
        c["low"]
        for c in last
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

    first_close = (
        first[-1]["close"]
    )

    last_close = (
        last[-1]["close"]
    )

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
        c["high"]
        for c in previous
    )

    previous_low = min(
        c["low"]
        for c in previous
    )

    if last["close"] > previous_high:
        return "BULLISH_BREAKOUT"

    if last["close"] < previous_low:
        return "BEARISH_BREAKOUT"

    if (
        last["high"] > previous_high
        and last["close"]
        < previous_high
    ):
        return "BEARISH_REJECTION"

    if (
        last["low"] < previous_low
        and last["close"]
        > previous_low
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

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    ema20 = ema(
        closes,
        20
    )

    rsi14 = rsi(
        closes,
        14
    )

    wr14 = williams_r(
        candles,
        14
    )

    atr10 = atr(
        candles,
        10
    )

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

    if (
        ema20 is not None
        and atr10 is not None
    ):
        keltner_upper = (
            ema20
            + atr10 * 5.0
        )

        keltner_lower = (
            ema20
            - atr10 * 5.0
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
        body_ratio = (
            body
            / candle_range
        )
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
# DIRECTION ENGINE
# ============================================================

def directional_pre_score(snapshot):
    """
    UP and DOWN are calculated independently.

    Important:
    There is NO artificial DOWN default.
    There is NO artificial UP default.

    The final side follows the evidence.
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

    if (
        ema9 is not None
        and ema21 is not None
    ):
        if ema9 > ema21:
            up += 3
            reasons_up.append(
                "EMA9>EMA21"
            )

        elif ema9 < ema21:
            down += 3
            reasons_down.append(
                "EMA9<EMA21"
            )

    # --------------------------------------------------------
    # Price vs EMA9
    # --------------------------------------------------------

    if ema9 is not None:
        if price > ema9:
            up += 1
            reasons_up.append(
                "Price>EMA9"
            )

        elif price < ema9:
            down += 1
            reasons_down.append(
                "Price<EMA9"
            )

    # --------------------------------------------------------
    # Price vs EMA21
    # --------------------------------------------------------

    if ema21 is not None:
        if price > ema21:
            up += 1
            reasons_up.append(
                "Price>EMA21"
            )

        elif price < ema21:
            down += 1
            reasons_down.append(
                "Price<EMA21"
            )

    # --------------------------------------------------------
    # Market Structure
    # --------------------------------------------------------

    if structure == "BULLISH":
        up += 3
        reasons_up.append(
            "Bullish structure"
        )

    elif structure == "BEARISH":
        down += 3
        reasons_down.append(
            "Bearish structure"
        )

    # --------------------------------------------------------
    # Breakout / Retest
    # --------------------------------------------------------

    if breakout == "BULLISH_BREAKOUT":
        up += 3
        reasons_up.append(
            "Bullish breakout"
        )

    elif breakout == "BEARISH_BREAKOUT":
        down += 3
        reasons_down.append(
            "Bearish breakout"
        )

    elif breakout == "BULLISH_REJECTION":
        up += 2
        reasons_up.append(
            "Bullish rejection"
        )

    elif breakout == "BEARISH_REJECTION":
        down += 2
        reasons_down.append(
            "Bearish rejection"
        )

    # --------------------------------------------------------
    # ADX + DI
    # --------------------------------------------------------

    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
        and adx14 >= 20
    ):
        di_gap = abs(
            plus_di
            - minus_di
        )

        if di_gap >= 2:
            if plus_di > minus_di:
                up += 2
                reasons_up.append(
                    "ADX/DI bullish"
                )

            elif minus_di > plus_di:
                down += 2
                reasons_down.append(
                    "ADX/DI bearish"
                )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi14 is not None:

        if 50 < rsi14 < 70:
            up += 1
            reasons_up.append(
                "RSI bullish zone"
            )

        elif 30 < rsi14 < 50:
            down += 1
            reasons_down.append(
                "RSI bearish zone"
            )

        elif rsi14 >= 70:
            # Overbought is NOT automatically DOWN.
            # Only adds bearish weight if EMA trend agrees.
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
            # Oversold is NOT automatically UP.
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

        # Keep this symmetric around -50.
        if wr14 > -50:
            up += 1
            reasons_up.append(
                "Williams bullish"
            )

        elif wr14 < -50:
            down += 1
            reasons_down.append(
                "Williams bearish"
            )

    # --------------------------------------------------------
    # Last Candle
    # --------------------------------------------------------

    if last_close > last_open:
        if body_ratio >= 0.30:
            up += 1
            reasons_up.append(
                "Bullish candle"
            )

    elif last_close < last_open:
        if body_ratio >= 0.30:
            down += 1
            reasons_down.append(
                "Bearish candle"
            )

    # --------------------------------------------------------
    # FINAL DIRECTION
    # --------------------------------------------------------

    if up > down:
        direction = "UP"

    elif down > up:
        direction = "DOWN"

    else:
        # Equal evidence:
        # use EMA as first tie-breaker.
        if (
            ema9 is not None
            and ema21 is not None
        ):
            if ema9 > ema21:
                direction = "UP"

            elif ema9 < ema21:
                direction = "DOWN"

            elif last_close >= last_open:
                direction = "UP"

            else:
                direction = "DOWN"

        elif last_close >= last_open:
            direction = "UP"

        else:
            direction = "DOWN"

    result = {
        "direction": direction,
        "up": up,
        "down": down,
        "gap": abs(up - down),
        "reasons_up": reasons_up,
        "reasons_down": reasons_down,
    }

    logger.info(
        "DIRECTION ENGINE | "
        "UP=%s DOWN=%s GAP=%s -> %s",
        up,
        down,
        abs(up - down),
        direction,
    )

    return result


# ============================================================
# SCORE CONVERSION
# ============================================================

def convert_pre_score_to_18(pre):
    """
    Convert raw directional evidence to exactly /18.

    The stronger direction receives the larger score.
    """

    up = int(
        pre.get("up", 0)
    )

    down = int(
        pre.get("down", 0)
    )

    if up <= 0 and down <= 0:
        if pre["direction"] == "UP":
            return 10, 8

        return 8, 10

    if up == down:
        if pre["direction"] == "UP":
            return 10, 8

        return 8, 10

    total = up + down

    up_score = round(
        (up / total) * 18
    )

    down_score = (
        18 - up_score
    )

    if pre["direction"] == "UP":

        if up_score <= down_score:
            up_score = (
                down_score + 1
            )

            if up_score > 18:
                up_score = 18

            down_score = (
                18 - up_score
            )

    else:

        if down_score <= up_score:
            down_score = (
                up_score + 1
            )

            if down_score > 18:
                down_score = 18

            up_score = (
                18 - down_score
            )

    return (
        int(clamp(up_score, 0, 18)),
        int(clamp(down_score, 0, 18))
    )


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
            "open": round(
                c["open"],
                8
            ),
            "high": round(
                c["high"],
                8
            ),
            "low": round(
                c["low"],
                8
            ),
            "close": round(
                c["close"],
                8
            ),
            "volume": c.get(
                "tick_volume",
                0
            ),
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
            "williams_r14": snapshot[
                "williams_r14"
            ],
            "atr10": snapshot["atr10"],
            "adx14": snapshot["adx14"],
            "plus_di": snapshot["plus_di"],
            "minus_di": snapshot["minus_di"],
            "keltner_upper": snapshot[
                "keltner_upper"
            ],
            "keltner_lower": snapshot[
                "keltner_lower"
            ],
            "structure": snapshot[
                "structure"
            ],
            "breakout": snapshot[
                "breakout"
            ],
            "body_ratio": snapshot[
                "body_ratio"
            ],
        },
        "deterministic_engine": {
            "direction": pre_score[
                "direction"
            ],
            "up_score_raw": pre_score[
                "up"
            ],
            "down_score_raw": pre_score[
                "down"
            ],
            "gap": pre_score[
                "gap"
            ],
            "up_reasons": pre_score[
                "reasons_up"
            ],
            "down_reasons": pre_score[
                "reasons_down"
            ],
        },
    }

    prompt = f"""
You are the technical validation engine for {BOT_NAME}.

This is short-term directional analysis.
It is NOT guaranteed.

You MUST choose exactly one:
UP
DOWN

Never output WAIT.
Never output NEUTRAL.
Never invent missing data.

IMPORTANT DIRECTION RULE:

The deterministic engine calculates UP and DOWN independently.

Do NOT automatically favor DOWN.
Do NOT automatically favor UP.

The supplied candle data is authoritative.

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

Do not invent support/resistance.

The final signal must agree with the actual supplied market evidence.

If deterministic direction has a clear advantage of 3 or more raw points,
keep that direction unless the candle data clearly contradicts it.

If the deterministic gap is small, independently verify both sides.

Scores:
UP score + DOWN score MUST equal exactly 18.

Confidence:
50 to 89 only.
90+ is forbidden.

Confidence is not probability.
Confidence is not a guarantee.

Cancellation:
UP -> meaningful recent closed-candle low.
DOWN -> meaningful recent closed-candle high.

Return ONLY valid JSON.

JSON:
{{
  "signal": "UP",
  "direction": "UP",
  "confidence": 65,
  "up_score": 10,
  "down_score": 8,
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
        logger.warning(
            "Gemini unavailable - "
            "using deterministic engine"
        )
        return None

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre_score,
    )

    try:
        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
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
            )

            text = text.replace(
                "```",
                ""
            ).strip()

        result = json.loads(text)

        if not isinstance(
            result,
            dict
        ):
            return None

        return result

    except Exception as exc:
        logger.exception(
            "Gemini analysis failed: %s",
            exc
        )
        return None


# ============================================================
# GEMINI VALIDATION
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
        convert_pre_score_to_18(
            pre_score
        )
    )

    if not isinstance(
        result,
        dict
    ):
        result = {}

    direction = str(
        result.get(
            "direction",
            result.get(
                "signal",
                ""
            )
        )
    ).upper().strip()

    if direction not in (
        "UP",
        "DOWN"
    ):
        direction = (
            deterministic_direction
        )

    confidence = safe_int(
        result.get(
            "confidence"
        ),
        65
    )

    confidence = clamp(
        confidence,
        50,
        89
    )

    up_score = safe_int(
        result.get(
            "up_score"
        ),
        fallback_up
    )

    down_score = safe_int(
        result.get(
            "down_score"
        ),
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

    raw_gap = abs(
        deterministic_up
        - deterministic_down
    )

    # Strong deterministic evidence
    # protects the direction.
    if raw_gap >= 3:
        direction = (
            deterministic_direction
        )

        up_score, down_score = (
            fallback_up,
            fallback_down
        )

    else:
        # Weak deterministic edge:
        # Gemini may validate either direction,
        # but scores must agree with final direction.
        if (
            up_score + down_score
            != 18
        ):
            up_score, down_score = (
                fallback_up,
                fallback_down
            )

    # --------------------------------------------------------
    # Force exact total = 18
    # --------------------------------------------------------

    if (
        up_score
        + down_score
        != 18
    ):

        if direction == "UP":

            up_score = max(
                up_score,
                fallback_up
            )

            up_score = clamp(
                up_score,
                1,
                18
            )

            down_score = (
                18 - up_score
            )

        else:

            down_score = max(
                down_score,
                fallback_down
            )

            down_score = clamp(
                down_score,
                1,
                18
            )

            up_score = (
                18 - down_score
            )

    # --------------------------------------------------------
    # Score must agree with direction
    # --------------------------------------------------------

    if direction == "UP":

        if up_score <= down_score:

            up_score = (
                down_score + 1
            )

            if up_score > 18:
                up_score = 18

            down_score = (
                18 - up_score
            )

    else:

        if down_score <= up_score:

            down_score = (
                up_score + 1
            )

            if down_score > 18:
                down_score = 18

            up_score = (
                18 - down_score
            )

    # --------------------------------------------------------
    # Confidence based on final gap
    # --------------------------------------------------------

    gap = abs(
        up_score
        - down_score
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

    confidence = clamp(
        confidence,
        50,
        89
    )

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
                "Bullish price action "
                "with directional confluence."
            )

        else:
            reason = (
                "Bearish price action "
                "with directional confluence."
            )

    if not cancellation_reason:

        if direction == "UP":
            cancellation_reason = (
                "إلغاء إذا أغلقت شمعة "
                "تحت مستوى الإلغاء."
            )

        else:
            cancellation_reason = (
                "إلغاء إذا أغلقت شمعة "
                "فوق مستوى الإلغاء."
            )

    return {
        "signal": direction,
        "direction": direction,
        "confidence": int(
            confidence
        ),
        "up_score": int(
            up_score
        ),
        "down_score": int(
            down_score
        ),
        "reason": reason,
        "cancellation_reason": (
            cancellation_reason
        ),
    }


# ============================================================
# FULL ANALYSIS
# ============================================================

def analyze_market(data):
    if not data:
        return None

    symbol = str(
        data.get(
            "symbol",
            ""
        )
    ).upper().strip()

    timeframe = str(
        data.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
    ).upper().strip()

    candles = get_closed_candles(
        data
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    snapshot = technical_snapshot(
        candles
    )

    if not snapshot:
        return None

    pre_score = (
        directional_pre_score(
            snapshot
        )
    )

    gemini_result = call_gemini(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre_score,
    )

    validated = (
        validate_gemini_result(
            gemini_result,
            pre_score
        )
    )

    # Strong deterministic evidence
    # is authoritative.
    if (
        abs(
            pre_score["up"]
            - pre_score["down"]
        ) >= 3
    ):

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

        validated["up_score"] = (
            up_score
        )

        validated["down_score"] = (
            down_score
        )

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

    logger.info(
        "FINAL ANALYSIS | "
        "%s %s | "
        "UP=%s/18 DOWN=%s/18 | "
        "CONF=%s%% | %s",
        symbol,
        timeframe,
        validated["up_score"],
        validated["down_score"],
        validated["confidence"],
        validated["direction"],
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
    candles = analysis[
        "candles"
    ]

    direction = analysis[
        "analysis"
    ]["direction"]

    timeframe = analysis[
        "timeframe"
    ]

    delay_minutes = (
        timeframe_minutes(
            timeframe
        )
    )

    entry_price = (
        candles[-1]["close"]
    )

    recent = candles[-8:]

    if direction == "UP":

        cancellation_level = min(
            c["low"]
            for c in recent
        )

        cancellation_text = (
            f"إلغاء إذا أغلقت شمعة "
            f"تحت "
            f"{cancellation_level:.8f}"
        )

    else:

        cancellation_level = max(
            c["high"]
            for c in recent
        )

        cancellation_text = (
            f"إلغاء إذا أغلقت شمعة "
            f"فوق "
            f"{cancellation_level:.8f}"
        )

    now = now_algiers()

    base_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        base_time
        + timedelta(
            minutes=delay_minutes
        )
    )

    return {
        "entry_price": entry_price,
        "cancellation_level": (
            cancellation_level
        ),
        "cancellation_text": (
            cancellation_text
        ),
        "entry_time": entry_time,
        "delay_minutes": delay_minutes,
    }


# ============================================================
# SETUP FINGERPRINT
# ============================================================

def make_setup_fingerprint(
    analysis,
    trade_type="BASE",
    recovery_number=0,
):
    symbol = analysis[
        "symbol"
    ]

    timeframe = analysis[
        "timeframe"
    ]

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
        f"{direction}|"
        f"{trade_type}|"
        f"{recovery_number}"
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

    signal = analysis[
        "analysis"
    ]

    entry = calculate_entry(
        analysis
    )

    fingerprint = (
        make_setup_fingerprint(
            analysis,
            trade_type,
            recovery_number
        )
    )

    current_time = time.time()

    with state_lock:

        if (
            fingerprint
            == last_setup_fingerprint
        ):
            logger.info(
                "Signal blocked: "
                "duplicate setup"
            )
            return None

        if (
            trade_type == "BASE"
            and SIGNAL_COOLDOWN_SECONDS > 0
            and (
                current_time
                - last_signal_time
                < SIGNAL_COOLDOWN_SECONDS
            )
        ):
            logger.info(
                "Signal blocked: "
                "cooldown"
            )
            return None

        last_setup_fingerprint = (
            fingerprint
        )

        if trade_type == "BASE":
            last_signal_time = (
                current_time
            )

        cycle_id = (
            f"{analysis['symbol']}_"
            f"{analysis['candles'][-1]['time']}_"
            f"{trade_type}_"
            f"{recovery_number}_"
            f"{int(current_time)}"
        )

        current_cycle = {
            "cycle_id": cycle_id,
            "symbol": analysis[
                "symbol"
            ],
            "timeframe": analysis[
                "timeframe"
            ],
            "direction": signal[
                "direction"
            ],
            "confidence": signal[
                "confidence"
            ],
            "up_score": signal[
                "up_score"
            ],
            "down_score": signal[
                "down_score"
            ],
            "entry_price": entry[
                "entry_price"
            ],
            "cancellation_level": entry[
                "cancellation_level"
            ],
            "entry_time": entry[
                "entry_time"
            ].isoformat(),
            "entry_delay_minutes": entry[
                "delay_minutes"
            ],
            "trade_type": trade_type,
            "recovery_number": recovery_number,
            "trade_number": (
                1
                if trade_type == "BASE"
                else recovery_number + 1
            ),
            "status": "PENDING",
            "reason": signal[
                "reason"
            ],
            "cancellation_text": entry[
                "cancellation_text"
            ],
            "created_at": (
                now_algiers()
                .isoformat()
            ),
            "analysis": analysis,
        }

    return current_cycle.copy()


# ============================================================
# HISTORY
# ============================================================

def add_history(
    cycle,
    result
):
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
        "closed_at": (
            now_algiers()
            .isoformat()
        ),
    }

    with state_lock:

        history.append(item)

        if len(history) > 100:
            del history[:-100]


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal_message(
    cycle
):
    direction = cycle[
        "direction"
    ]

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
        trade_label = (
            "🎯 BASE TRADE"
        )

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
        cycle[
            "cancellation_level"
        ]
    )

    delay = safe_int(
        cycle.get(
            "entry_delay_minutes",
            timeframe_minutes(
                cycle.get(
                    "timeframe",
                    "M1"
                )
            )
        ),
        1
    )

    delay_text = (
        f"{delay} minute"
        if delay == 1
        else f"{delay} minutes"
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
        f"{delay_text}\n"

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
# BEST SYMBOL
# ============================================================

def get_best_symbol():
    with mt5_lock:
        items = list(
            mt5_data.values()
        )

    best = None
    best_score = -999999

    for data in items:

        try:
            candles = (
                get_closed_candles(
                    data
                )
            )

            if len(candles) < (
                MIN_CLOSED_CANDLES
            ):
                continue

            snapshot = (
                technical_snapshot(
                    candles
                )
            )

            if not snapshot:
                continue

            pre = (
                directional_pre_score(
                    snapshot
                )
            )

            # Freshness matters.
            received_at = safe_float(
                data.get(
                    "received_at"
                )
            )

            age = (
                time.time()
                - received_at
                if received_at
                else 999999
            )

            if age > 180:
                freshness_penalty = 10
            elif age > 90:
                freshness_penalty = 4
            else:
                freshness_penalty = 0

            strength = (
                max(
                    pre["up"],
                    pre["down"]
                )
                * 2
                + abs(
                    pre["up"]
                    - pre["down"]
                )
                - freshness_penalty
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
# TELEGRAM SAFE SCHEDULER
# ============================================================

async def send_cycle_to_owner(
    cycle
):
    global telegram_application

    if not telegram_application:
        logger.warning(
            "Telegram application not ready"
        )
        return

    if OWNER_ID <= 0:
        logger.warning(
            "OWNER_ID invalid"
        )
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


def telegram_future_done(
    future
):
    try:
        future.result()

    except Exception as exc:
        logger.exception(
            "Telegram scheduled send failed: %s",
            exc
        )


def schedule_telegram_send(
    cycle
):
    """
    Called from the Auto Analysis thread.

    IMPORTANT:
    Never use asyncio.run() here.

    The coroutine is submitted to the
    SAME event loop used by Telegram.
    """

    global telegram_loop

    if telegram_loop is None:
        logger.warning(
            "Telegram loop not ready; "
            "signal not sent automatically"
        )
        return False

    if telegram_loop.is_closed():
        logger.error(
            "Telegram loop is already closed"
        )
        return False

    try:

        future = (
            asyncio
            .run_coroutine_threadsafe(
                send_cycle_to_owner(
                    cycle
                ),
                telegram_loop
            )
        )

        future.add_done_callback(
            telegram_future_done
        )

        return True

    except Exception as exc:

        logger.exception(
            "Could not schedule Telegram send: %s",
            exc
        )

        return False


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analysis_once():

    try:

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

        if not cycle:
            return

        logger.info(
            "NEW SIGNAL | "
            "%s | %s | %s | "
            "UP %s/18 | DOWN %s/18 | %s%%",
            cycle["symbol"],
            cycle["timeframe"],
            cycle["direction"],
            cycle["up_score"],
            cycle["down_score"],
            cycle["confidence"],
        )

        schedule_telegram_send(
            cycle
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
# WIN
# ============================================================

def mark_win():
    global current_cycle

    with state_lock:

        if not current_cycle:
            return None

        if (
            current_cycle.get(
                "status"
            )
            != "PENDING"
        ):
            return None

        current_cycle[
            "status"
        ] = "WIN"

        finished = (
            current_cycle.copy()
        )

    with stats_lock:
        stats["wins"] += 1

    add_history(
        finished,
        "WIN"
    )

    return finished


# ============================================================
# LOSS
# ============================================================

def mark_loss():
    global current_cycle

    with state_lock:

        if not current_cycle:
            return None

        if (
            current_cycle.get(
                "status"
            )
            != "PENDING"
        ):
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
    # BASE LOSS -> FRESH RECOVERY
    # --------------------------------------------------------

    if (
        losing_cycle.get(
            "trade_type"
        )
        == "BASE"
        and RECOVERY_LIMIT >= 1
    ):

        symbol = losing_cycle[
            "symbol"
        ]

        with mt5_lock:
            latest_data = (
                mt5_data.get(
                    symbol
                )
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

                    logger.info(
                        "RECOVERY SIGNAL | "
                        "%s | %s",
                        recovery_cycle[
                            "symbol"
                        ],
                        recovery_cycle[
                            "direction"
                        ],
                    )

                    return recovery_cycle

        with state_lock:
            current_cycle = None

        return None

    # --------------------------------------------------------
    # RECOVERY LOSS -> END
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
        wins = stats[
            "wins"
        ]

        losses = stats[
            "losses"
        ]

    total = (
        wins
        + losses
    )

    if total > 0:
        winrate = (
            wins
            / total
            * 100
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
        f"🎯 Win Rate: "
        f"{winrate:.1f}%\n\n"
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
            history[
                -HISTORY_DISPLAY_COUNT:
            ]
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
                f"RECOVERY "
                f"{recovery}/1"
            )

        else:
            label = "BASE"

        lines.append(
            f"{icon} "
            f"{item.get('symbol', '')} "
            f"{item.get('timeframe', '')}\n"

            f"   {label} | "
            f"{item.get('direction', '')}\n"

            f"   🎯 "
            f"{item.get('confidence', 0)}% | "
            f"📈 "
            f"{item.get('up_score', 0)}/18 "
            f"📉 "
            f"{item.get('down_score', 0)}/18\n"

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

    if (
        isinstance(
            result,
            dict
        )
        and result.get(
            "trade_type"
        )
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
        stats[
            "wins"
        ] = 0

        stats[
            "losses"
        ] = 0

    with state_lock:
        history.clear()

        current_cycle = None

        last_signal_time = 0.0

        last_setup_fingerprint = ""

    await update.message.reply_text(
        "♻️ All statistics, history "
        "and active cycle have been reset."
    )


# ============================================================
