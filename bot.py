import os
import json
import logging
import threading
import asyncio
import time
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
# ZinoProSignalAI V4
# MT5 -> Render -> Gemini -> Telegram
# ============================================================

APP_NAME = "ZinoProSignalAI"
VERSION = "V4-MT5"

# ------------------------------------------------------------
# ENVIRONMENT
# ------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()


# ------------------------------------------------------------
# GENERAL CONFIG
# ------------------------------------------------------------

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40
MAX_CLOSED_CANDLES = 150

HISTORY_DISPLAY_COUNT = 10

SIGNAL_COOLDOWN_SECONDS = 120
SETUP_REPEAT_BLOCK_SECONDS = 360

RECOVERY_LIMIT = 1

AUTO_ANALYSIS_INTERVAL_SECONDS = 60

ENTRY_DELAY_BY_TIMEFRAME = {
    "M1": 1,
    "M2": 2,
    "M3": 3,
    "M5": 5,
    "M10": 10,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


# ------------------------------------------------------------
# TIMEZONE
# ------------------------------------------------------------

ALGIERS_TZ = ZoneInfo("Africa/Algiers")


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

DATA_DIR = os.getenv("DATA_DIR", ".").strip() or "."

STATS_FILE = os.path.join(DATA_DIR, "zino_stats.json")
HISTORY_FILE = os.path.join(DATA_DIR, "zino_history.json")


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(APP_NAME)


# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------

mt5_data = {}
mt5_lock = threading.Lock()

state_lock = threading.Lock()

current_cycle = None

last_signal_time = 0.0
last_setup_fingerprint = ""
last_setup_time = 0.0

analysis_lock = threading.Lock()

bot_application = None


# ------------------------------------------------------------
# TELEGRAM OWNER
# ------------------------------------------------------------

try:
    OWNER_ID = int(OWNER_ID_RAW) if OWNER_ID_RAW else 0
except Exception:
    OWNER_ID = 0


# ============================================================
# DEFAULT STATS
# ============================================================

DEFAULT_STATS = {
    "wins": 0,
    "losses": 0,
    "total": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}


def load_json_file(path, default):
    try:
        if not os.path.exists(path):
            return default.copy() if isinstance(default, dict) else default

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(default, dict):
            result = default.copy()
            if isinstance(data, dict):
                result.update(data)
            return result

        return data

    except Exception as exc:
        logger.error("JSON LOAD ERROR %s: %s", path, exc)
        return default.copy() if isinstance(default, dict) else default


def save_json_file(path, data):
    temp_path = path + ".tmp"

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_path, path)

    except Exception as exc:
        logger.error("JSON SAVE ERROR %s: %s", path, exc)


stats = load_json_file(STATS_FILE, DEFAULT_STATS)
history = load_json_file(HISTORY_FILE, [])


# ============================================================
# UTILITY
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def format_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M")


def timeframe_minutes(timeframe):
    tf = str(timeframe or "M1").upper().strip()

    if tf in ENTRY_DELAY_BY_TIMEFRAME:
        return ENTRY_DELAY_BY_TIMEFRAME[tf]

    if tf.startswith("M"):
        try:
            return int(tf[1:])
        except Exception:
            return 1

    if tf.startswith("H"):
        try:
            return int(tf[1:]) * 60
        except Exception:
            return 60

    return 1


def safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def is_owner(update: Update):
    if not update or not update.effective_user:
        return False

    if OWNER_ID <= 0:
        return False

    return update.effective_user.id == OWNER_ID


async def owner_only(update: Update):
    if not is_owner(update):
        if update.message:
            await update.message.reply_text(
                "⛔ هذا البوت خاص بالمالك فقط."
            )
        return False

    return True


# ============================================================
# STATS / HISTORY
# ============================================================

def save_stats():
    save_json_file(STATS_FILE, stats)


def save_history():
    save_json_file(HISTORY_FILE, history)


def add_history(record):
    global history

    with state_lock:
        history.append(record)

        if len(history) > 200:
            history = history[-200:]

        save_history()


def calculate_win_rate():
    total = int(stats.get("wins", 0)) + int(stats.get("losses", 0))

    if total <= 0:
        return 0.0

    return round((stats.get("wins", 0) / total) * 100, 1)


# ============================================================
# INDICATORS
# ============================================================

def closes(candles):
    return [
        safe_float(c.get("close"))
        for c in candles
        if safe_float(c.get("close")) is not None
    ]


def highs(candles):
    return [
        safe_float(c.get("high"))
        for c in candles
        if safe_float(c.get("high")) is not None
    ]


def lows(candles):
    return [
        safe_float(c.get("low"))
        for c in candles
        if safe_float(c.get("low")) is not None
    ]


def ema(values, period):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if len(values) < period:
        return None

    seed = sum(values[:period]) / period
    result = seed

    multiplier = 2.0 / (period + 1.0)

    for value in values[period:]:
        result = (
            (value - result) * multiplier
        ) + result

    return result


def ema_series(values, period):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if len(values) < period:
        return []

    seed = sum(values[:period]) / period

    result = [None] * (period - 1)
    result.append(seed)

    multiplier = 2.0 / (period + 1.0)

    current = seed

    for value in values[period:]:
        current = (
            (value - current) * multiplier
        ) + current

        result.append(current)

    return result


def rsi(values, period=14):
    values = [
        float(x)
        for x in values
        if x is not None
    ]

    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0.0)

        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

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

    return 100.0 - (100.0 / (1.0 + rs))


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    hh = max(
        safe_float(c["high"], 0.0)
        for c in recent
    )

    ll = min(
        safe_float(c["low"], 0.0)
        for c in recent
    )

    close = safe_float(recent[-1]["close"])

    if close is None:
        return None

    if hh == ll:
        return -50.0

    return ((hh - close) / (hh - ll)) * -100.0


def true_ranges(candles):
    result = []

    previous_close = None

    for candle in candles:
        high = safe_float(candle.get("high"))
        low = safe_float(candle.get("low"))
        close = safe_float(candle.get("close"))

        if high is None or low is None:
            continue

        if previous_close is None:
            tr = high - low

        else:
            tr = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        result.append(tr)

        if close is not None:
            previous_close = close

    return result


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

    highs_v = [
        safe_float(c["high"])
        for c in candles
    ]

    lows_v = [
        safe_float(c["low"])
        for c in candles
    ]

    closes_v = [
        safe_float(c["close"])
        for c in candles
    ]

    tr_values = []
    plus_dm_values = []
    minus_dm_values = []

    for i in range(1, len(candles)):
        high = highs_v[i]
        low = lows_v[i]

        prev_high = highs_v[i - 1]
        prev_low = lows_v[i - 1]
        prev_close = closes_v[i - 1]

        if None in (
            high,
            low,
            prev_high,
            prev_low,
            prev_close,
        ):
            continue

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = 0.0
        minus_dm = 0.0

        if up_move > down_move and up_move > 0:
            plus_dm = up_move

        if down_move > up_move and down_move > 0:
            minus_dm = down_move

        tr_values.append(tr)
        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)

    if len(tr_values) < period:
        return None, None, None

    tr_avg = sum(tr_values[-period:]) / period
    plus_avg = sum(plus_dm_values[-period:]) / period
    minus_avg = sum(minus_dm_values[-period:]) / period

    if tr_avg == 0:
        return 0.0, 0.0, 0.0

    plus_di = (plus_avg / tr_avg) * 100.0
    minus_di = (minus_avg / tr_avg) * 100.0

    denominator = plus_di + minus_di

    if denominator == 0:
        dx = 0.0
    else:
        dx = (
            abs(plus_di - minus_di)
            / denominator
        ) * 100.0

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
        safe_float(c["high"])
        for c in first
    )

    last_high = max(
        safe_float(c["high"])
        for c in last
    )

    first_low = min(
        safe_float(c["low"])
        for c in first
    )

    last_low = min(
        safe_float(c["low"])
        for c in last
    )

    if last_high > first_high and last_low > first_low:
        return "BULLISH"

    if last_high < first_high and last_low < first_low:
        return "BEARISH"

    return "NEUTRAL"


def breakout_state(candles):
    if len(candles) < 9:
        return "NONE"

    previous = candles[-9:-1]
    last = candles[-1]

    previous_high = max(
        safe_float(c["high"])
        for c in previous
    )

    previous_low = min(
        safe_float(c["low"])
        for c in previous
    )

    close = safe_float(last["close"])

    if close > previous_high:
        return "UP"

    if close < previous_low:
        return "DOWN"

    return "NONE"


def candle_direction(candle):
    open_price = safe_float(candle.get("open"))
    close = safe_float(candle.get("close"))

    if open_price is None or close is None:
        return "NEUTRAL"

    if close > open_price:
        return "UP"

    if close < open_price:
        return "DOWN"

    return "NEUTRAL"


def candle_strength(candle):
    open_price = safe_float(candle.get("open"))
    high = safe_float(candle.get("high"))
    low = safe_float(candle.get("low"))
    close = safe_float(candle.get("close"))

    if None in (
        open_price,
        high,
        low,
        close,
    ):
        return 0.0

    rng = high - low

    if rng <= 0:
        return 0.0

    body = abs(close - open_price)

    return body / rng


# ============================================================
# KELTNER
# ============================================================

def keltner(candles):
    close_values = closes(candles)

    if len(close_values) < 20:
        return None, None, None

    middle = ema(close_values, 20)
    atr_value = atr(candles, 10)

    if middle is None or atr_value is None:
        return None, None, None

    multiplier = 5.0

    upper = middle + (
        atr_value * multiplier
    )

    lower = middle - (
        atr_value * multiplier
    )

    return middle, upper, lower


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def technical_snapshot(candles):
    if len(candles) < MIN_CLOSED_CANDLES:
        raise ValueError(
            f"Need at least {MIN_CLOSED_CANDLES} closed candles"
        )

    close_values = closes(candles)

    last = candles[-1]

    price = safe_float(last.get("close"))

    ema9 = ema(close_values, 9)
    ema21 = ema(close_values, 21)

    rsi14 = rsi(close_values, 14)

    wr14 = williams_r(candles, 14)

    atr10 = atr(candles, 10)

    adx14, plus_di14, minus_di14 = adx_di(
        candles,
        14
    )

    kc_middle, kc_upper, kc_lower = keltner(
        candles
    )

    structure = market_structure(candles)

    breakout = breakout_state(candles)

    recent8 = candles[-8:]

    recent8_low = min(
        safe_float(c["low"])
        for c in recent8
    )

    recent8_high = max(
        safe_float(c["high"])
        for c in recent8
    )

    body_strength = candle_strength(last)

    candle_dir = candle_direction(last)

    return {
        "price": price,
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi14,
        "williams_r14": wr14,
        "atr10": atr10,
        "adx14": adx14,
        "plus_di14": plus_di14,
        "minus_di14": minus_di14,
        "keltner_middle": kc_middle,
        "keltner_upper": kc_upper,
        "keltner_lower": kc_lower,
        "structure": structure,
        "breakout": breakout,
        "recent8_low": recent8_low,
        "recent8_high": recent8_high,
        "candle_direction": candle_dir,
        "candle_body_strength": body_strength,
    }


# ============================================================
# PRE-SCORE
# ============================================================

def directional_pre_score(snapshot):
    up = 0
    down = 0

    price = snapshot["price"]
    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    structure = snapshot["structure"]
    breakout = snapshot["breakout"]

    adx = snapshot["adx14"]
    plus_di = snapshot["plus_di14"]
    minus_di = snapshot["minus_di14"]

    rsi14 = snapshot["rsi14"]
    candle_dir = snapshot["candle_direction"]

    # --------------------------------------------------------
    # EMA 9 / EMA 21 = 3 points
    # --------------------------------------------------------

    if ema9 is not None and ema21 is not None:

        if ema9 > ema21:
            up += 3

        elif ema9 < ema21:
            down += 3

    # --------------------------------------------------------
    # Price vs EMA9 = 1
    # --------------------------------------------------------

    if price is not None and ema9 is not None:

        if price > ema9:
            up += 1

        elif price < ema9:
            down += 1

    # --------------------------------------------------------
    # Price vs EMA21 = 1
    # --------------------------------------------------------

    if price is not None and ema21 is not None:

        if price > ema21:
            up += 1

        elif price < ema21:
            down += 1

    # --------------------------------------------------------
    # Market Structure = 3
    # --------------------------------------------------------

    if structure == "BULLISH":
        up += 3

    elif structure == "BEARISH":
        down += 3

    # --------------------------------------------------------
    # Breakout = 3
    # --------------------------------------------------------

    if breakout == "UP":
        up += 3

    elif breakout == "DOWN":
        down += 3

    # --------------------------------------------------------
    # ADX / DI = 2
    # --------------------------------------------------------

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
    ):

        if adx >= 20:

            if plus_di > minus_di:
                up += 2

            elif minus_di > plus_di:
                down += 2

    # --------------------------------------------------------
    # RSI = 1
    # --------------------------------------------------------

    if rsi14 is not None:

        if 50 < rsi14 < 70:
            up += 1

        elif 30 < rsi14 < 50:
            down += 1

    # --------------------------------------------------------
    # Candle = 1
    # --------------------------------------------------------

    if candle_dir == "UP":
        up += 1

    elif candle_dir == "DOWN":
        down += 1

    if up > down:
        direction = "UP"

    elif down > up:
        direction = "DOWN"

    else:
        if (
            ema9 is not None
            and ema21 is not None
            and ema9 >= ema21
        ):
            direction = "UP"
        else:
            direction = "DOWN"

    return {
        "up": up,
        "down": down,
        "direction": direction,
        "gap": abs(up - down),
    }


# ============================================================
# ADDITIONAL CONFLUENCE
# ============================================================

def calculate_confluence(snapshot):
    up = 0
    down = 0

    price = snapshot["price"]

    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    rsi14 = snapshot["rsi14"]
    wr14 = snapshot["williams_r14"]

    adx = snapshot["adx14"]
    plus_di = snapshot["plus_di14"]
    minus_di = snapshot["minus_di14"]

    structure = snapshot["structure"]
    breakout = snapshot["breakout"]

    kc_middle = snapshot["keltner_middle"]

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if structure == "BULLISH":
        up += 2

    elif structure == "BEARISH":
        down += 2

    # --------------------------------------------------------
    # EMA alignment
    # --------------------------------------------------------

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:
            up += 1

        elif ema9 < ema21:
            down += 1

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    if breakout == "UP":
        up += 2

    elif breakout == "DOWN":
        down += 2

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi14 is not None:

        if 52 <= rsi14 <= 68:
            up += 1

        elif 32 <= rsi14 <= 48:
            down += 1

    # --------------------------------------------------------
    # Williams
    # --------------------------------------------------------

    if wr14 is not None:

        if wr14 > -50:
            up += 1

        elif wr14 < -50:
            down += 1

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
    ):

        if adx >= 20:

            if plus_di > minus_di:
                up += 1

            elif minus_di > plus_di:
                down += 1

    # --------------------------------------------------------
    # Keltner position
    # --------------------------------------------------------

    if price is not None and kc_middle is not None:

        if price > kc_middle:
            up += 1

        elif price < kc_middle:
            down += 1

    return {
        "up": up,
        "down": down,
        "gap": abs(up - down),
    }


def detect_market_quality(snapshot):
    """
    Returns a descriptive regime.

    This does NOT generate WAIT.
    It only affects confidence and explanation.
    """

    adx = snapshot["adx14"]
    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    structure = snapshot["structure"]
    breakout = snapshot["breakout"]

    if adx is not None and adx < 15:
        return "LOW_MOMENTUM"

    if (
        structure == "NEUTRAL"
        and breakout == "NONE"
        and ema9 is not None
        and ema21 is not None
        and abs(ema9 - ema21) < 0.00001
    ):
        return "CHOPPY"

    if (
        structure != "NEUTRAL"
        and adx is not None
        and adx >= 20
    ):
        return "TRENDING"

    if breakout != "NONE":
        return "BREAKOUT"

    return "MIXED"


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candles(raw):
    result = []

    if not isinstance(raw, list):
        return result

    for item in raw:

        if not isinstance(item, dict):
            continue

        required = (
            "time",
            "open",
            "high",
            "low",
            "close",
        )

        if not all(
            key in item
            for key in required
        ):
            continue

        try:
            candle = {
                "time": int(item["time"]),
                "open": float(item["open"]),
                "high": float(item["high"]),
                "low": float(item["low"]),
                "close": float(item["close"]),
                "tick_volume": int(
                    item.get("tick_volume", 0)
                ),
                "real_volume": int(
                    item.get("real_volume", 0)
                ),
                "spread": int(
                    item.get("spread", 0)
                ),
            }

            if candle["high"] < candle["low"]:
                continue

            if candle["open"] < 0:
                continue

            if candle["high"] < candle["open"]:
                continue

            if candle["high"] < candle["close"]:
                continue

            if candle["low"] > candle["open"]:
                continue

            if candle["low"] > candle["close"]:
                continue

            result.append(candle)

        except Exception:
            continue

    result.sort(
        key=lambda x: x["time"]
    )

    # Remove duplicate timestamps.
    unique = {}

    for candle in result:
        unique[candle["time"]] = candle

    return [
        unique[key]
        for key in sorted(unique)
    ]


# ============================================================
# MT5 DATA VALIDATION
# ============================================================

def validate_mt5_payload(payload):
    if not isinstance(payload, dict):
        return False, "Payload is not an object"

    incoming_key = str(
        payload.get("api_key", "")
    ).strip()

    header_key = str(
        payload.get("_header_api_key", "")
    ).strip()

    valid_key = (
        incoming_key == MT5_API_KEY
        or header_key == MT5_API_KEY
    )

    if not MT5_API_KEY:
        return False, "MT5_API_KEY is not configured"

    if not valid_key:
        return False, "Invalid API key"

    symbol = str(
        payload.get("symbol", "")
    ).strip()

    if not symbol:
        return False, "Missing symbol"

    candles = normalize_candles(
        payload.get("candles", [])
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return (
            False,
            f"Not enough candles: {len(candles)}"
        )

    timeframe = str(
        payload.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
    ).upper()

    if not timeframe:
        timeframe = ANALYSIS_TIMEFRAME

    return True, {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "batch_id": str(
            payload.get("batch_id", "")
        ),
        "closed_candle_time": safe_int(
            payload.get("closed_candle_time"),
            candles[-1]["time"]
        ),
        "current_bid": safe_float(
            payload.get("current_bid")
        ),
        "current_ask": safe_float(
            payload.get("current_ask")
        ),
        "digits": safe_int(
            payload.get("digits"),
            5
        ),
        "received_at": time.time(),
    }


# ============================================================
# MT5 HTTP HANDLER
# ============================================================

class MT5Handler(BaseHTTPRequestHandler):

    def _send_json(self, code, payload):
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

        try:
            self.wfile.write(body)
        except Exception:
            pass

    def do_GET(self):

        parsed = urlparse(self.path)

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
        ):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "app": APP_NAME,
                    "version": VERSION,
                    "mt5_connected": bool(mt5_data),
                }
            )
            return

        if parsed.path in (
            "/mt5status",
            "/api/mt5status",
        ):
            with mt5_lock:
                data = dict(mt5_data)

            self._send_json(
                200,
                {
                    "status": "ok",
                    "data": {
                        "symbol": data.get("symbol"),
                        "timeframe": data.get(
                            "timeframe"
                        ),
                        "candles": len(
                            data.get("candles", [])
                        ),
                        "batch_id": data.get(
                            "batch_id"
                        ),
                    }
                }
            )
            return

        self._send_json(
            404,
            {
                "error": "Not found"
            }
        )

    def do_POST(self):

        parsed = urlparse(self.path)

        if parsed.path not in (
            "/mt5",
            "/api/mt5",
        ):
            self._send_json(
                404,
                {
                    "error": "Not found"
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

        except Exception:
            content_length = 0

        if content_length <= 0:
            self._send_json(
                400,
                {
                    "error": "Empty body"
                }
            )
            return

        try:
            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )

        except Exception as exc:
            self._send_json(
                400,
                {
                    "error": "Invalid JSON",
                    "details": str(exc),
                }
            )
            return

        payload["_header_api_key"] = (
            self.headers.get(
                "X-MT5-API-Key",
                ""
            )
        )

        valid, result = validate_mt5_payload(
            payload
        )

        if not valid:
            status = 401 if (
                result == "Invalid API key"
            ) else 400

            self._send_json(
                status,
                {
                    "error": result
                }
            )
            return

        with mt5_lock:
            mt5_data.clear()
            mt5_data.update(result)

        logger.info(
            "MT5 DATA RECEIVED | %s | %s | candles=%d | batch=%s",
            result["symbol"],
            result["timeframe"],
            len(result["candles"]),
            result["batch_id"],
        )

        self._send_json(
            200,
            {
                "status": "accepted",
                "symbol": result["symbol"],
                "timeframe": result["timeframe"],
                "candles": len(
                    result["candles"]
                ),
                "batch_id": result["batch_id"],
            }
        )

    def log_message(self, fmt, *args):
        logger.info(
            "HTTP | " + fmt,
            *args
        )


def start_http_server():
    port = int(
        os.getenv(
            "PORT",
            "10000"
        )
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        MT5Handler
    )

    logger.info(
        "HTTP server listening on port %s",
        port
    )

    server.serve_forever()


# ============================================================
# MT5 DATA ACCESS
# ============================================================

def get_mt5_snapshot():
    with mt5_lock:
        if not mt5_data:
            return None

        return {
            key: value
            for key, value in mt5_data.items()
        }


def get_closed_candles(data):
    candles = normalize_candles(
        data.get("candles", [])
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return []

    # MT5 sends current forming candle as the latest
    # candle. We intentionally remove it.
    if len(candles) >= 2:
        candles = candles[:-1]

    if len(candles) > MAX_CLOSED_CANDLES:
        candles = candles[-MAX_CLOSED_CANDLES:]

    return candles


# ============================================================
# SETUP FINGERPRINT
# ============================================================

def create_setup_fingerprint(
    symbol,
    timeframe,
    candle_time,
    direction,
):
    return (
        f"{symbol}|"
        f"{timeframe}|"
        f"{candle_time}|"
        f"{direction}"
    )


# ============================================================
# ENTRY CALCULATION
# ============================================================

def calculate_entry(
    candles,
    direction,
    timeframe,
):
    last_closed = candles[-1]

    entry_price = safe_float(
        last_closed["close"]
    )

    delay = timeframe_minutes(
        timeframe
    )

    current = now_algiers()

    base = current.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        base
        + timedelta(minutes=delay)
    )

    recent = candles[-8:]

    if direction == "UP":
        cancellation_level = min(
            safe_float(c["low"])
            for c in recent
        )

        cancellation_text = (
            "إلغاء إذا أغلقت شمعة تحت "
            f"{cancellation_level:.8f}"
        )

    else:
        cancellation_level = max(
            safe_float(c["high"])
            for c in recent
        )

        cancellation_text = (
            "إلغاء إذا أغلقت شمعة فوق "
            f"{cancellation_level:.8f}"
        )

    return {
        "entry_price": entry_price,
        "delay": delay,
        "entry_time": entry_time,
        "cancellation_level": cancellation_level,
        "cancellation_text": cancellation_text,
    }


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
        logger.error(
            "Gemini initialization failed: %s",
            exc
        )


def compact_candle(c):
    return {
        "time": c["time"],
        "open": c["open"],
        "high": c["high"],
        "low": c["low"],
        "close": c["close"],
        "tick_volume": c.get(
            "tick_volume",
            0
        ),
    }


def build_gemini_prompt(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score,
    confluence,
    regime,
):
    last_candles = [
        compact_candle(c)
        for c in candles[-50:]
    ]

    technical_data = {
        "price": snapshot["price"],
        "ema9": snapshot["ema9"],
        "ema21": snapshot["ema21"],
        "rsi14": snapshot["rsi14"],
        "williams_r14": snapshot[
            "williams_r14"
        ],
        "atr10": snapshot["atr10"],
        "adx14": snapshot["adx14"],
        "plus_di14": snapshot[
            "plus_di14"
        ],
        "minus_di14": snapshot[
            "minus_di14"
        ],
        "keltner_middle": snapshot[
            "keltner_middle"
        ],
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
        "recent8_low": snapshot[
            "recent8_low"
        ],
        "recent8_high": snapshot[
            "recent8_high"
        ],
        "candle_direction": snapshot[
            "candle_direction"
        ],
        "candle_body_strength": snapshot[
            "candle_body_strength"
        ],
    }

    return f"""
You are the senior market-analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied MT5 closed-candle data.

IMPORTANT:
- Do not use external market data.
- Do not invent prices.
- Do not invent indicators.
- Do not assume an indicator value that is not supplied.
- Do not use the current forming candle.
- The supplied candle list consists of CLOSED candles.
- The final result must be UP or DOWN.
- Never return WAIT.
- Never return NEUTRAL.
- Never invent a future price.
- Confidence is NOT a guarantee.
- Do not give 90%+ confidence unless the evidence is exceptionally strong.
- Prefer realistic confidence values.

SYMBOL:
{symbol}

TIMEFRAME:
{timeframe}

MARKET REGIME:
{regime}

PRE-SCORE:
UP = {pre_score["up"]}
DOWN = {pre_score["down"]}
PRE-SCORE DIRECTION = {pre_score["direction"]}

ADDITIONAL CONFLUENCE:
UP = {confluence["up"]}
DOWN = {confluence["down"]}

TECHNICAL SNAPSHOT:
{json.dumps(technical_data, ensure_ascii=False)}

RECENT CLOSED CANDLES:
{json.dumps(last_candles, ensure_ascii=False)}

ANALYSIS PRIORITY:

1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity behavior
5. Momentum
6. Candle behavior
7. EMA 9 / EMA 21
8. RSI 14
9. Williams %R 14
10. Keltner Channel
11. ADX / DI

SCORING:

The final score must total EXACTLY 18.

Use the following categories:

Structure = 2
Breakout / Retest = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary / Overall Price Action = 2
Oscillators = 2
Moving Averages = 2
ADX / Trend Strength = 2

TOTAL = 18

The score must represent directional evidence.

If evidence supports UP:
UP should receive more points.

If evidence supports DOWN:
DOWN should receive more points.

Do not automatically give points merely because an indicator exists.

Avoid contradictory reasoning.

If the market is weak/choppy, still choose the stronger direction, but reduce confidence.

Use the actual supplied price action as the primary source.

LIQUIDITY:
Look for sweep/rejection behavior around recent highs/lows.
Do not invent liquidity levels.

BREAKOUT:
A breakout is meaningful only if supported by candle close and structure/momentum.
Do not call a simple wick a confirmed breakout.

CANDLE:
Evaluate body size, close location, rejection wick, and whether the candle agrees with the structure.

EMA:
Use EMA9 and EMA21 as trend confirmation, not as the main reason by themselves.

RSI:
Avoid treating overbought/oversold alone as an automatic reversal.

WILLIAMS:
Use it as secondary momentum/position information.

KELTNER:
Use only the supplied Keltner values.

ADX:
Use ADX with DI direction. Low ADX should reduce confidence.

OUTPUT JSON ONLY.

Required exact schema:

{{
  "direction": "UP" or "DOWN",
  "confidence": integer,
  "up_score": integer,
  "down_score": integer,
  "reason": "short explanation",
  "cancellation_reason": "short explanation"
}}

Rules:
- up_score + down_score MUST equal 18.
- Both scores must be integers.
- confidence must be between 1 and 89 normally.
- 90+ only for exceptionally strong multi-factor confluence.
- direction must match the stronger score.
- No markdown.
- No code fences.
- No extra keys.
"""


def extract_json(text):
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

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < 0:
        return None

    text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        return None


def gemini_analyze(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score,
    confluence,
    regime,
):
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client is not configured"
        )

    prompt = build_gemini_prompt(
        symbol=symbol,
        timeframe=timeframe,
        candles=candles,
        snapshot=snapshot,
        pre_score=pre_score,
        confluence=confluence,
        regime=regime,
    )

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

    parsed = extract_json(text)

    if not parsed:
        raise RuntimeError(
            "Gemini returned invalid JSON"
        )

    return parsed


# ============================================================
# DIRECTIONAL VALIDATION
# ============================================================

def validate_gemini_result(
    result,
    pre_score,
    confluence,
    snapshot,
):
    if not isinstance(result, dict):
        result = {}

    direction = str(
        result.get(
            "direction",
            ""
        )
    ).upper().strip()

    if direction not in (
        "UP",
        "DOWN",
    ):
        direction = pre_score["direction"]

    up_score = safe_int(
        result.get("up_score"),
        None
    )

    down_score = safe_int(
        result.get("down_score"),
        None
    )

    if up_score is None:
        up_score = 0

    if down_score is None:
        down_score = 0

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
    # Force exact 18 total
    # --------------------------------------------------------

    total = up_score + down_score

    if total <= 0:
        if pre_score["direction"] == "UP":
            up_score = 10
            down_score = 8
        else:
            up_score = 8
            down_score = 10

    elif total != 18:

        if up_score >= down_score:
            up_score = 18 - down_score

        else:
            down_score = 18 - up_score

    # --------------------------------------------------------
    # Direction must match score
    # --------------------------------------------------------

    if up_score > down_score:
        direction = "UP"

    elif down_score > up_score:
        direction = "DOWN"

    else:
        direction = pre_score["direction"]

        if direction == "UP":
            up_score = 10
            down_score = 8

        else:
            up_score = 8
            down_score = 10

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = safe_int(
        result.get("confidence"),
        None
    )

    if confidence is None:
        gap = abs(
            up_score - down_score
        )

        confidence = 55 + (
            gap * 3
        )

    confidence = clamp(
        confidence,
        1,
        99
    )

    # --------------------------------------------------------
    # Prevent fake high confidence
    # --------------------------------------------------------

    agreement_gap = abs(
        pre_score["up"]
        - pre_score["down"]
    )

    confluence_gap = abs(
        confluence["up"]
        - confluence["down"]
    )

    structure = snapshot[
        "structure"
    ]

    breakout = snapshot[
        "breakout"
    ]

    strong_alignment = (
        agreement_gap >= 5
        and confluence_gap >= 3
        and (
            (
                direction == "UP"
                and structure == "BULLISH"
            )
            or
            (
                direction == "DOWN"
                and structure == "BEARISH"
            )
        )
    )

    if confidence >= 90 and not strong_alignment:
        confidence = 89

    # --------------------------------------------------------
    # Choppy / low momentum penalty
    # --------------------------------------------------------

    regime = detect_market_quality(
        snapshot
    )

    if regime in (
        "CHOPPY",
        "LOW_MOMENTUM",
    ):
        confidence = min(
            confidence,
            74
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
        reason = (
            "Price action, structure and "
            "directional momentum favor "
            f"{direction}."
        )

    if not cancellation_reason:
        if direction == "UP":
            cancellation_reason = (
                "Cancel if a closed candle "
                "breaks the recent structural low."
            )
        else:
            cancellation_reason = (
                "Cancel if a closed candle "
                "breaks the recent structural high."
            )

    return {
        "direction": direction,
        "confidence": confidence,
        "up_score": up_score,
        "down_score": down_score,
        "reason": reason,
        "cancellation_reason": cancellation_reason,
    }


# ============================================================
# SIGNAL QUALITY
# ============================================================

def final_signal_quality(
    analysis,
    pre_score,
    confluence,
    snapshot,
):
    direction = analysis["direction"]

    if direction == "UP":
        primary = pre_score["up"]
        secondary = confluence["up"]

    else:
        primary = pre_score["down"]
        secondary = confluence["down"]

    opposite_pre = (
        pre_score["down"]
        if direction == "UP"
        else pre_score["up"]
    )

    primary_gap = (
        primary - opposite_pre
    )

    # --------------------------------------------------------
    # Prevent obviously contradictory signals
    # --------------------------------------------------------

    if primary_gap <= -2:
        analysis["confidence"] = min(
            analysis["confidence"],
            62
        )

    # --------------------------------------------------------
    # Strong alignment bonus only when justified
    # --------------------------------------------------------

    if (
        primary_gap >= 5
        and secondary >= 4
        and snapshot["structure"]
        == (
            "BULLISH"
            if direction == "UP"
            else "BEARISH"
        )
    ):
        analysis["confidence"] = min(
            analysis["confidence"] + 3,
            89
        )

    # --------------------------------------------------------
    # Never exceed 89 unless explicit exceptional condition.
    # We intentionally keep normal system below 90.
    # --------------------------------------------------------

    analysis["confidence"] = min(
        analysis["confidence"],
        89
    )

    return analysis


# ============================================================
# BEST PAIR
# ============================================================

def choose_best_pair():
    data = get_mt5_snapshot()

    if not data:
        return None

    candles = get_closed_candles(
        data
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    symbol = str(
        data.get(
            "symbol",
            ""
        )
    )

    timeframe = str(
        data.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
    ).upper()

    try:
        snapshot = technical_snapshot(
            candles
        )

        pre_score = directional_pre_score(
            snapshot
        )

        confluence = calculate_confluence(
            snapshot
        )

        regime = detect_market_quality(
            snapshot
        )

    except Exception as exc:
        logger.error(
            "TECHNICAL ANALYSIS ERROR: %s",
            exc
        )
        return None

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "snapshot": snapshot,
        "pre_score": pre_score,
        "confluence": confluence,
        "regime": regime,
        "batch_id": data.get(
            "batch_id",
            ""
        ),
    }


# ============================================================
# SIGNAL CREATION
# ============================================================

def create_signal():
    global last_signal_time
    global last_setup_fingerprint
    global last_setup_time

    if not analysis_lock.acquire(
        blocking=False
    ):
        logger.info(
            "Analysis already running"
        )
        return None

    try:
        data = choose_best_pair()

        if not data:
            logger.warning(
                "No valid MT5 analysis data"
            )
            return None

        symbol = data["symbol"]
        timeframe = data["timeframe"]
        candles = data["candles"]

        snapshot = data["snapshot"]
        pre_score = data["pre_score"]
        confluence = data["confluence"]
        regime = data["regime"]

        logger.info(
            "PRE ANALYSIS | %s | %s | "
            "UP=%s DOWN=%s | "
            "CONF_UP=%s CONF_DOWN=%s | "
            "STRUCT=%s BREAKOUT=%s REGIME=%s",
            symbol,
            timeframe,
            pre_score["up"],
            pre_score["down"],
            confluence["up"],
            confluence["down"],
            snapshot["structure"],
            snapshot["breakout"],
            regime,
        )

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        try:
            raw_analysis = gemini_analyze(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                snapshot=snapshot,
                pre_score=pre_score,
                confluence=confluence,
                regime=regime,
            )

        except Exception as exc:
            logger.error(
                "GEMINI ANALYSIS ERROR: %s",
                exc
            )

            # Strong deterministic fallback.
            raw_analysis = {
                "direction": pre_score[
                    "direction"
                ],
                "confidence": 58,
                "up_score": (
                    10
                    if pre_score["direction"]
                    == "UP"
                    else 8
                ),
                "down_score": (
                    8
                    if pre_score["direction"]
                    == "UP"
                    else 10
                ),
                "reason": (
                    "MT5 technical structure "
                    "and directional confluence "
                    f"favor {pre_score['direction']}."
                ),
                "cancellation_reason": (
                    "Cancel if price closes "
                    "beyond the recent "
                    "structural invalidation level."
                ),
            }

        analysis = validate_gemini_result(
            raw_analysis,
            pre_score,
            confluence,
            snapshot,
        )

        analysis = final_signal_quality(
            analysis,
            pre_score,
            confluence,
            snapshot,
        )

        direction = analysis[
            "direction"
        ]

        last_closed_time = candles[-1][
            "time"
        ]

        fingerprint = create_setup_fingerprint(
            symbol,
            timeframe,
            last_closed_time,
            direction,
        )

        current_time = time.time()

        # ----------------------------------------------------
        # Duplicate setup block
        # ----------------------------------------------------

        if (
            fingerprint == last_setup_fingerprint
            and (
                current_time
                - last_setup_time
            )
            < SETUP_REPEAT_BLOCK_SECONDS
        ):
            logger.info(
                "SETUP BLOCKED | %s",
                fingerprint
            )
            return None

        # ----------------------------------------------------
        # Cooldown
        # ----------------------------------------------------

        if (
            current_time
            - last_signal_time
        ) < SIGNAL_COOLDOWN_SECONDS:
            logger.info(
                "SIGNAL COOLDOWN ACTIVE"
            )
            return None

        # ----------------------------------------------------
        # Entry
        # ----------------------------------------------------

        entry = calculate_entry(
            candles,
            direction,
            timeframe,
        )

        # ----------------------------------------------------
        # Build signal
        # ----------------------------------------------------

        signal = {
            "id": int(
                time.time() * 1000
            ),
            "created_at": format_dt(
                now_algiers()
            ),
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "confidence": analysis[
                "confidence"
            ],
            "up_score": analysis[
                "up_score"
            ],
            "down_score": analysis[
                "down_score"
            ],
            "reason": analysis[
                "reason"
            ],
            "cancellation_reason": analysis[
                "cancellation_reason"
            ],
            "entry_price": entry[
                "entry_price"
            ],
            "entry_delay": entry[
                "delay"
            ],
            "entry_time": format_dt(
                entry["entry_time"]
            ),
            "cancellation_level": entry[
                "cancellation_level"
            ],
            "cancellation_text": entry[
                "cancellation_text"
            ],
            "structure": snapshot[
                "structure"
            ],
            "breakout": snapshot[
                "breakout"
            ],
            "regime": regime,
            "batch_id": data.get(
                "batch_id",
                ""
            ),
            "last_closed_candle_time":
                last_closed_time,
        }

        # ----------------------------------------------------
        # Start BASE cycle
        # ----------------------------------------------------

        global current_cycle

        with state_lock:

            current_cycle = {
                "status": "PENDING",
                "trade_type": "BASE",
                "trade_number": 1,
                "recovery_used": False,
                "signal": signal,
            }

        last_signal_time = current_time
        last_setup_fingerprint = fingerprint
        last_setup_time = current_time

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        add_history({
            "id": signal["id"],
            "status": "PENDING",
            "trade_type": "BASE",
            "trade_number": 1,
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "confidence": analysis[
                "confidence"
            ],
            "up_score": analysis[
                "up_score"
            ],
            "down_score": analysis[
                "down_score"
            ],
            "entry_price": entry[
                "entry_price"
            ],
            "entry_time": format_dt(
                entry["entry_time"]
            ),
            "created_at": signal[
                "created_at"
            ],
            "batch_id": signal[
                "batch_id"
            ],
        })

        logger.info(
            "SIGNAL CREATED | %s | %s | %s | "
            "confidence=%s | %s/%s",
            symbol,
            timeframe,
            direction,
            analysis["confidence"],
            analysis["up_score"],
            analysis["down_score"],
        )

        return signal

    finally:
        analysis_lock.release()


# ============================================================
# SIGNAL MESSAGE
# ============================================================

def format_signal_message(
    signal,
    recovery=False,
):
    direction = signal["direction"]

    if direction == "UP":
        direction_text = "🟢 UP"
    else:
        direction_text = "🔴 DOWN"

    if recovery:
        trade_label = "🔁 RECOVERY 1/1"
    else:
        trade_label = "🎯 BASE TRADE"

    confidence = signal["confidence"]

    up_score = signal["up_score"]
    down_score = signal["down_score"]

    entry_price = signal[
        "entry_price"
    ]

    cancellation_level = signal[
        "cancellation_level"
    ]

    reason = signal[
        "reason"
    ]

    return (
        f"🎓 {APP_NAME}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 {signal['symbol']} | "
        f"{signal['timeframe']}\n\n"
        f"{trade_label}\n"
        f"{direction_text}\n\n"
        f"🔥 Confidence: {confidence}%\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏱️ Entry after: "
        f"{signal['entry_delay']} minute"
        f"{'s' if signal['entry_delay'] != 1 else ''}\n"
        f"🕐 ENTRY TIME: "
        f"{signal['entry_time']} 🇩🇿\n"
        f"💰 ENTRY PRICE: "
        f"{entry_price:.8f}\n"
        f"⚠️ CANCEL LEVEL: "
        f"{cancellation_level:.8f}\n"
        f"   {signal['cancellation_text']}\n\n"
        f"🧠 Reason:\n"
        f"{reason}\n\n"
        f"📐 Structure: "
        f"{signal['structure']}\n"
        f"🚀 Breakout: "
        f"{signal['breakout']}\n"
        f"📈 Regime: "
        f"{signal['regime']}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# SEND TELEGRAM
# ============================================================

async def send_message_to_owner(text):
    global bot_application

    if not bot_application:
        logger.error(
            "Telegram application unavailable"
        )
        return False

    if OWNER_ID <= 0:
        logger.error(
            "OWNER_ID is not configured"
        )
        return False

    try:
        await bot_application.bot.send_message(
            chat_id=OWNER_ID,
            text=text,
        )

        return True

    except Exception as exc:
        logger.error(
            "TELEGRAM SEND ERROR: %s",
            exc
        )

        return False


# ============================================================
# AUTO ANALYSIS LOOP
# ============================================================

async def auto_analysis_loop():
    logger.info(
        "Auto analysis loop started"
    )

    while True:

        try:

            await asyncio.sleep(
                AUTO_ANALYSIS_INTERVAL_SECONDS
            )

            if not mt5_data:
                continue

            signal = await asyncio.to_thread(
                create_signal
            )

            if signal:
                await send_message_to_owner(
                    format_signal_message(
                        signal,
                        recovery=False
                    )
                )

        except asyncio.CancelledError:
            logger.info(
                "Auto analysis loop cancelled"
            )
            break

        except Exception as exc:
            logger.error(
                "AUTO LOOP ERROR: %s",
                exc
            )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        f"🎓 {APP_NAME} {VERSION}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ MT5 Analysis Engine\n"
        "✅ Closed-candle analysis\n"
        "✅ EMA 9/21\n"
        "✅ RSI 14\n"
        "✅ Williams %R 14\n"
        "✅ ADX/DI 14\n"
        "✅ Keltner\n"
        "✅ Structure / Breakout / Liquidity\n"
        "✅ Recovery 1/1\n\n"
        "الأوامر:\n"
        "/analyze\n"
        "/mt5status\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    data = get_mt5_snapshot()

    if not data:
        await update.message.reply_text(
            "🔴 MT5: لا توجد بيانات مستلمة."
        )
        return

    candles = data.get(
        "candles",
        []
    )

    received_at = data.get(
        "received_at",
        0
    )

    age = (
        time.time()
        - received_at
        if received_at
        else 999999
    )

    await update.message.reply_text(
        "🖥️ ZinoProSignalAI MT5 STATUS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Symbol: "
        f"{data.get('symbol')}\n"
        f"⏱️ Timeframe: "
        f"{data.get('timeframe')}\n"
        f"🕯️ Candles: "
        f"{len(candles)}\n"
        f"📦 Batch: "
        f"{data.get('batch_id')}\n"
        f"⏳ Data age: "
        f"{age:.1f}s\n"
        f"🕐 Closed candle: "
        f"{data.get('closed_candle_time')}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🧠 جاري تحليل بيانات MT5...\n"
        "التحليل يعتمد على آخر الشموع المغلقة."
    )

    signal = await asyncio.to_thread(
        create_signal
    )

    if not signal:
        await update.message.reply_text(
            "⚠️ لم يتم إنشاء إشارة.\n"
            "تأكد من وصول بيانات MT5 "
            "ووجود 40 شمعة مغلقة على الأقل."
        )
        return

    await update.message.reply_text(
        format_signal_message(
            signal,
            recovery=False
        )
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    wins = int(
        stats.get(
            "wins",
            0
        )
    )

    losses = int(
        stats.get(
            "losses",
            0
        )
    )

    total = wins + losses

    win_rate = calculate_win_rate()

    recovery_wins = int(
        stats.get(
            "recovery_wins",
            0
        )
    )

    recovery_losses = int(
        stats.get(
            "recovery_losses",
            0
        )
    )

    cycle_status = "NONE"

    if current_cycle:
        cycle_status = current_cycle.get(
            "status",
            "UNKNOWN"
        )

    await update.message.reply_text(
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📊 Total: {total}\n"
        f"🎯 Win Rate: {win_rate}%\n\n"
        f"🔁 Recovery Wins: "
        f"{recovery_wins}\n"
        f"🔁 Recovery Losses: "
        f"{recovery_losses}\n\n"
        f"🔄 Current Cycle: "
        f"{cycle_status}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    if not history:
        await update.message.reply_text(
            "📚 History فارغ."
        )
        return

    rows = history[
        -HISTORY_DISPLAY_COUNT:
    ]

    lines = [
        "📚 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in reversed(rows):

        status = item.get(
            "status",
            "UNKNOWN"
        )

        if status == "WIN":
            status_icon = "🟢"
        elif status == "LOSS":
            status_icon = "🔴"
        else:
            status_icon = "🟡"

        symbol = item.get(
            "symbol",
            "?"
        )

        timeframe = item.get(
            "timeframe",
            "?"
        )

        direction = item.get(
            "direction",
            "?"
        )

        confidence = item.get(
            "confidence",
            0
        )

        up = item.get(
            "up_score",
            0
        )

        down = item.get(
            "down_score",
            0
        )

        trade_type = item.get(
            "trade_type",
            "BASE"
        )

        lines.append(
            f"{status_icon} "
            f"{symbol} {timeframe}\n"
            f"   {trade_type} | "
            f"{direction}\n"
            f"   🎯 {confidence}% | "
            f"📈 {up}/18 📉 {down}/18\n"
            f"   ⏰ {item.get('created_at','')}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# WIN / LOSS
# ============================================================

def update_last_pending_history(
    status,
    trade_type=None,
):
    for item in reversed(history):

        if item.get("status") == "PENDING":

            item["status"] = status

            if trade_type:
                item["trade_type"] = trade_type

            item["result_time"] = format_dt(
                now_algiers()
            )

            save_history()

            return item

    return None


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    global current_cycle

    with state_lock:

        if not current_cycle:
            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة."
            )
            return

        trade_type = current_cycle.get(
            "trade_type",
            "BASE"
        )

        if trade_type == "RECOVERY":
            stats["recovery_wins"] += 1

        stats["wins"] += 1
        stats["total"] += 1

        update_last_pending_history(
            "WIN",
            trade_type
        )

        current_cycle = None

        save_stats()

    await update.message.reply_text(
        "🟢 WIN\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 {trade_type}\n"
        "تم تسجيل الصفقة كفوز.\n"
        "🔄 Cycle closed."
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    global current_cycle

    with state_lock:

        if not current_cycle:
            await update.message.reply_text(
                "⚠️ لا توجد صفقة معلقة."
            )
            return

        trade_type = current_cycle.get(
            "trade_type",
            "BASE"
        )

        # ----------------------------------------------------
        # BASE LOSS -> ONE RECOVERY
        # ----------------------------------------------------

        if (
            trade_type == "BASE"
            and not current_cycle.get(
                "recovery_used",
                False
            )
        ):

            stats["losses"] += 1
            stats["total"] += 1

            update_last_pending_history(
                "LOSS",
                "BASE"
            )

            old_signal = current_cycle[
                "signal"
            ]

            recovery_signal = dict(
                old_signal
            )

            recovery_signal["id"] = int(
                time.time() * 1000
            )

            recovery_signal[
                "created_at"
            ] = format_dt(
                now_algiers()
            )

            current_cycle = {
                "status": "PENDING",
                "trade_type": "RECOVERY",
                "trade_number": 2,
                "recovery_used": True,
                "signal": recovery_signal,
            }

            add_history({
                "id": recovery_signal["id"],
                "status": "PENDING",
                "trade_type": "RECOVERY",
                "trade_number": 2,
                "symbol": recovery_signal[
                    "symbol"
                ],
                "timeframe": recovery_signal[
                    "timeframe"
                ],
                "direction": recovery_signal[
                    "direction"
                ],
                "confidence": recovery_signal[
                    "confidence"
                ],
                "up_score": recovery_signal[
                    "up_score"
                ],
                "down_score": recovery_signal[
                    "down_score"
                ],
                "entry_price": recovery_signal[
                    "entry_price"
                ],
                "entry_time": recovery_signal[
                    "entry_time"
                ],
                "created_at": recovery_signal[
                    "created_at"
                ],
                "batch_id": recovery_signal[
                    "batch_id"
                ],
            })

            save_stats()

            message = format_signal_message(
                recovery_signal,
                recovery=True
            )

            await update.message.reply_text(
                "🔴 BASE LOSS\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔁 Recovery 1/1 activated.\n\n"
                + message
            )

            return

        # ----------------------------------------------------
        # RECOVERY LOSS -> END CYCLE
        # ----------------------------------------------------

        stats["losses"] += 1
        stats["total"] += 1

        if trade_type == "RECOVERY":
            stats["recovery_losses"] += 1

        update_last_pending_history(
            "LOSS",
            trade_type
        )

        current_cycle = None

        save_stats()

    await update.message.reply_text(
        "🔴 LOSS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 {trade_type}\n"
        "❌ Cycle انتهى.\n"
        "⛔ لا توجد Recovery أخرى."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not await owner_only(update):
        return

    global stats
    global history
    global current_cycle
    global last_signal_time
    global last_setup_fingerprint
    global last_setup_time

    with state_lock:

        stats = DEFAULT_STATS.copy()

        history = []

        current_cycle = None

        last_signal_time = 0.0

        last_setup_fingerprint = ""

        last_setup_time = 0.0

        save_stats()
        save_history()

    await update.message.reply_text(
        "♻️ RESET COMPLETED\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "Stats = 0\n"
        "History = cleared\n"
        "Cycle = cleared\n"
        "Cooldown = cleared"
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

    text = (
        update.message.text
        if update.message
        else ""
    )

    if not text:
        return

    await update.message.reply_text(
        "🤖 النظام يعمل عبر MT5.\n\n"
        "الأوامر المتاحة:\n"
        "/analyze\n"
        "/mt5status\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset"
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
        "📷 Image analysis غير مفعّل في نسخة MT5.\n"
        "مصدر التحليل الحالي هو MT5 مباشرة."
    )


# ============================================================
# TELEGRAM POST INIT
# ============================================================

async def post_init(
    application: Application,
):
    global bot_application

    bot_application = application

    logger.info(
        "Telegram application initialized"
    )

    application.create_task(
        auto_analysis_loop()
    )


# ============================================================
# MAIN
# ============================================================

def check_config():
    missing = []

    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if OWNER_ID <= 0:
        missing.append("OWNER_ID")

    if not MT5_API_KEY:
        missing.append("MT5_API_KEY")

    if missing:
        logger.warning(
            "Missing environment variables: %s",
            ", ".join(missing)
        )

    logger.info(
        "CONFIG | model=%s | timeframe=%s | "
        "min_candles=%s | recovery=%s",
        GEMINI_MODEL,
        ANALYSIS_TIMEFRAME,
        MIN_CLOSED_CANDLES,
        RECOVERY_LIMIT,
    )


def main():
    check_config()

    # --------------------------------------------------------
    # HTTP server for Render + MT5
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="HTTPServer"
    )

    http_thread.start()

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    if not BOT_TOKEN:
        logger.error(
            "BOT_TOKEN is missing. "
            "Telegram bot cannot start."
        )

        while True:
            time.sleep(60)

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
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
            "analyze",
            analyze_command
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

    logger.info(
        "========================================"
    )

    logger.info(
        "%s %s STARTING",
        APP_NAME,
        VERSION
    )

    logger.info(
        "Timeframe: %s",
        ANALYSIS_TIMEFRAME
    )

    logger.info(
        "Minimum closed candles: %s",
        MIN_CLOSED_CANDLES
    )

    logger.info(
        "Recovery limit: %s",
        RECOVERY_LIMIT
    )

    logger.info(
        "========================================"
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
