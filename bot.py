import os
import json
import logging
import threading
import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv(
    "MT4_API_KEY",
    ""
).strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")

TIMEFRAME = "M1"

MIN_CANDLES = 40
MAX_CANDLES = 150

# لا نحلل بيانات أقدم من هذا
MARKET_DATA_MAX_AGE = 90

# كل كم ثانية نفحص هل Batch جديد جاهز
BACKGROUND_INTERVAL = 3

# الدخول بعد دقيقتين
ENTRY_DELAY_MINUTES = 2

# بعد BASE LOSS ننتظر دقيقتين قبل Recovery
RECOVERY_DELAY_SECONDS = 120

RECOVERY_LIMIT = 1

# الحد الأدنى لجودة الفرصة
MIN_SCORE = 13

# عدد أفضل الفرص التي تدخل للمراجعة النهائية
TOP_CANDIDATES = 8

# لا نعيد نفس الزوج مباشرة إذا توجد فرصة أخرى قريبة منه
SAME_SYMBOL_PENALTY = 2

# حماية من تكرار نفس الاتجاه مرات كثيرة
STREAK_PENALTY_START = 3

# أقصى Confidence
MAX_CONFIDENCE = 89

HISTORY_FILE = "signal_history.json"


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
        logger.error(
            "Gemini initialization failed: %s",
            e
        )


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.RLock()

market_data = {}

latest_batch_id = None
latest_batch_started_at = 0
latest_batch_complete = False

processing_batch_id = None
last_processed_batch_id = None

# لكل رمز آخر شمعة تم تحليلها
last_processed_candle = {}

# fingerprint لكل رمز
last_processed_fingerprint = {}

# آخر إشارة
last_signal = None

# الصفقة الحالية
active_trade = None

# Recovery
recovery_pending = False
recovery_wait_until = 0
recovery_number = 0

# اتجاه آخر إشارة
last_signal_direction = None
same_direction_streak = 0

# آخر زوج
last_signal_symbol = None


# ============================================================
# STATS
# ============================================================

stats = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
    "signals": 0,
}

history = []


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history():
    global history
    global stats
    global last_signal_direction
    global same_direction_streak
    global last_signal_symbol

    try:
        if not os.path.exists(HISTORY_FILE):
            return

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            saved_stats = data.get("stats")

            if isinstance(saved_stats, dict):
                for key in stats:
                    if key in saved_stats:
                        stats[key] = int(
                            saved_stats[key]
                        )

            saved_history = data.get("history")

            if isinstance(saved_history, list):
                history = saved_history[-500:]

            if history:
                last = history[-1]

                direction = last.get(
                    "direction"
                )

                symbol = last.get(
                    "symbol"
                )

                if direction in (
                    "UP",
                    "DOWN"
                ):
                    last_signal_direction = direction

                if symbol:
                    last_signal_symbol = symbol

                streak = 0

                for item in reversed(history):
                    if item.get("direction") == direction:
                        streak += 1
                    else:
                        break

                same_direction_streak = streak

        logger.info(
            "History loaded | signals=%s wins=%s losses=%s",
            stats["signals"],
            stats["wins"],
            stats["losses"]
        )

    except Exception as e:
        logger.exception(
            "History load error: %s",
            e
        )


def save_history():
    try:
        data = {
            "stats": stats,
            "history": history[-500:]
        }

        tmp_file = HISTORY_FILE + ".tmp"

        with open(
            tmp_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        os.replace(
            tmp_file,
            HISTORY_FILE
        )

    except Exception as e:
        logger.exception(
            "History save error: %s",
            e
        )


load_history()


# ============================================================
# HELPERS
# ============================================================

def now_algiers():
    return datetime.now(TIMEZONE)


def now_timestamp():
    return time.time()


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


def clamp(value, low, high):
    return max(
        low,
        min(high, value)
    )


def normalize_direction(direction):
    if not direction:
        return ""

    d = str(direction).upper().strip()

    if d in (
        "UP",
        "CALL",
        "BUY"
    ):
        return "UP"

    if d in (
        "DOWN",
        "PUT",
        "SELL"
    ):
        return "DOWN"

    return ""


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1.0)

    result = sum(
        values[:period]
    ) / period

    for price in values[period:]:
        result = (
            (price - result) * multiplier
            + result
        )

    return result


def ema_series(values, period):
    if len(values) < period:
        return []

    multiplier = 2.0 / (period + 1.0)

    current = (
        sum(values[:period])
        / period
    )

    result = [current]

    for price in values[period:]:
        current = (
            (price - current)
            * multiplier
            + current
        )

        result.append(current)

    return result


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        diff = (
            values[i]
            - values[i - 1]
        )

        gains.append(
            max(diff, 0)
        )

        losses.append(
            max(-diff, 0)
        )

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
            (avg_gain * (period - 1))
            + gain
        ) / period

        avg_loss = (
            (avg_loss * (period - 1))
            + loss
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + rs)
    )


def williams_r(
    highs,
    lows,
    closes,
    period=14
):
    if len(closes) < period:
        return None

    highest = max(
        highs[-period:]
    )

    lowest = min(
        lows[-period:]
    )

    close = closes[-1]

    if highest == lowest:
        return -50.0

    return (
        (highest - close)
        / (highest - lowest)
    ) * -100.0


def atr(
    highs,
    lows,
    closes,
    period=14
):
    if len(closes) < period + 1:
        return None

    trs = []

    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
            )
        )

        trs.append(tr)

    if len(trs) < period:
        return None

    return (
        sum(trs[-period:])
        / period
    )


def adx_di(
    highs,
    lows,
    closes,
    period=14
):
    if len(closes) < period * 2:
        return None

    tr_values = []
    plus_dm = []
    minus_dm = []

    for i in range(
        1,
        len(closes)
    ):
        up_move = (
            highs[i]
            - highs[i - 1]
        )

        down_move = (
            lows[i - 1]
            - lows[i]
        )

        tr = max(
            highs[i] - lows[i],
            abs(
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
            )
        )

        tr_values.append(tr)

        plus_dm.append(
            up_move
            if (
                up_move > down_move
                and up_move > 0
            )
            else 0.0
        )

        minus_dm.append(
            down_move
            if (
                down_move > up_move
                and down_move > 0
            )
            else 0.0
        )

    if len(tr_values) < period:
        return None

    atr_value = (
        sum(tr_values[-period:])
        / period
    )

    if atr_value <= 0:
        return None

    plus_di = (
        100.0
        * (
            sum(
                plus_dm[-period:]
            ) / period
        )
        / atr_value
    )

    minus_di = (
        100.0
        * (
            sum(
                minus_dm[-period:]
            ) / period
        )
        / atr_value
    )

    dx_denominator = (
        plus_di
        + minus_di
    )

    if dx_denominator == 0:
        adx = 0.0
    else:
        adx = (
            100.0
            * abs(
                plus_di
                - minus_di
            )
            / dx_denominator
        )

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di
    }


# ============================================================
# CANDLE HELPERS
# ============================================================

def candle_body(c):
    return abs(
        c["close"]
        - c["open"]
    )


def candle_range(c):
    return max(
        c["high"]
        - c["low"],
        0.0000000001
    )


def candle_direction(c):
    if c["close"] > c["open"]:
        return "UP"

    if c["close"] < c["open"]:
        return "DOWN"

    return "FLAT"


def average_range(
    candles,
    period=14
):
    if len(candles) < period:
        return 0.0

    values = [
        max(
            c["high"] - c["low"],
            0.0
        )
        for c in candles[-period:]
    ]

    return sum(values) / len(values)


# ============================================================
# STRUCTURE
# ============================================================

def structure_signal(candles):
    if len(candles) < 12:
        return {
            "direction": None,
            "strength": 0,
            "reason": ""
        }

    recent = candles[-6:]
    previous = candles[-12:-6]

    recent_high = max(
        c["high"] for c in recent
    )

    recent_low = min(
        c["low"] for c in recent
    )

    previous_high = max(
        c["high"] for c in previous
    )

    previous_low = min(
        c["low"] for c in previous
    )

    close = candles[-1]["close"]

    # Break of structure
    if close > previous_high:
        return {
            "direction": "UP",
            "strength": 3,
            "reason": "bullish structure break"
        }

    if close < previous_low:
        return {
            "direction": "DOWN",
            "strength": 3,
            "reason": "bearish structure break"
        }

    # Higher/lower structure
    if (
        recent_high > previous_high
        and recent_low > previous_low
    ):
        return {
            "direction": "UP",
            "strength": 2,
            "reason": "higher-high/higher-low structure"
        }

    if (
        recent_high < previous_high
        and recent_low < previous_low
    ):
        return {
            "direction": "DOWN",
            "strength": 2,
            "reason": "lower-high/lower-low structure"
        }

    return {
        "direction": None,
        "strength": 0,
        "reason": ""
    }


# ============================================================
# BUILD SNAPSHOT
# ============================================================

def build_snapshot(
    symbol,
    data
):
    candles = data.get(
        "candles",
        []
    )

    if len(candles) < MIN_CANDLES:
        return None

    # Normalize
    clean = []

    for c in candles:
        try:
            clean.append({
                "time": safe_int(c.get("time")),
                "open": safe_float(c.get("open")),
                "high": safe_float(c.get("high")),
                "low": safe_float(c.get("low")),
                "close": safe_float(c.get("close")),
                "tick_volume": safe_int(
                    c.get("tick_volume")
                ),
                "spread": safe_int(
                    c.get("spread")
                ),
            })
        except Exception:
            continue

    if len(clean) < MIN_CANDLES:
        return None

    # Sort chronologically
    clean.sort(
        key=lambda x: x["time"]
    )

    opens = [
        c["open"] for c in clean
    ]

    highs = [
        c["high"] for c in clean
    ]

    lows = [
        c["low"] for c in clean
    ]

    closes = [
        c["close"] for c in clean
    ]

    current_price = safe_float(
        data.get("current_bid"),
        closes[-1]
    )

    # --------------------------------------------------------
    # INDICATORS
    # --------------------------------------------------------

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    rsi_value = rsi(
        closes,
        14
    )

    will_value = williams_r(
        highs,
        lows,
        closes,
        14
    )

    atr_value = atr(
        highs,
        lows,
        closes,
        14
    )

    adx = adx_di(
        highs,
        lows,
        closes,
        14
    )

    # Keltner
    keltner_ema = ema(
        closes,
        20
    )

    keltner_atr = atr(
        highs,
        lows,
        closes,
        10
    )

    # --------------------------------------------------------
    # CANDLE QUALITY
    # --------------------------------------------------------

    last = clean[-1]
    prev = clean[-2]

    body = candle_body(last)
    rng = candle_range(last)

    body_ratio = (
        body / rng
        if rng > 0
        else 0
    )

    candle_dir = candle_direction(
        last
    )

    # Last 3 candle momentum
    last3 = clean[-3:]

    up_count = sum(
        1
        for c in last3
        if c["close"] > c["open"]
    )

    down_count = sum(
        1
        for c in last3
        if c["close"] < c["open"]
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    structure = structure_signal(
        clean
    )

    # --------------------------------------------------------
    # SCORE BUCKETS
    #
    # IMPORTANT:
    # correlated indicators are grouped together.
    # They don't get full points independently.
    # --------------------------------------------------------

    up = 0
    down = 0

    up_reasons = []
    down_reasons = []

    independent_up = 0
    independent_down = 0

    # ========================================================
    # BUCKET 1: PRICE ACTION + STRUCTURE
    # MAX 4
    # ========================================================

    if structure["direction"] == "UP":
        points = min(
            structure["strength"] + 1,
            4
        )

        up += points
        independent_up += 1

        if structure["reason"]:
            up_reasons.append(
                structure["reason"]
            )

    elif structure["direction"] == "DOWN":
        points = min(
            structure["strength"] + 1,
            4
        )

        down += points
        independent_down += 1

        if structure["reason"]:
            down_reasons.append(
                structure["reason"]
            )

    # Recent price action
    if (
        up_count >= 2
        and closes[-1] > closes[-4]
    ):
        if structure["direction"] != "UP":
            up += 2
            independent_up += 1

        up_reasons.append(
            "bullish recent price action"
        )

    elif (
        down_count >= 2
        and closes[-1] < closes[-4]
    ):
        if structure["direction"] != "DOWN":
            down += 2
            independent_down += 1

        down_reasons.append(
            "bearish recent price action"
        )

    # ========================================================
    # BUCKET 2: BREAKOUT / LIQUIDITY
    # MAX 3
    # ========================================================

    lookback = clean[-8:-1]

    if lookback:
        local_high = max(
            c["high"] for c in lookback
        )

        local_low = min(
            c["low"] for c in lookback
        )

        # Close breakout
        if closes[-1] > local_high:
            up += 3
            independent_up += 1

            up_reasons.append(
                "clean upside breakout"
            )

        elif closes[-1] < local_low:
            down += 3
            independent_down += 1

            down_reasons.append(
                "clean downside breakout"
            )

        # Liquidity sweep / rejection
        else:
            recent_high = max(
                c["high"]
                for c in clean[-5:-1]
            )

            recent_low = min(
                c["low"]
                for c in clean[-5:-1]
            )

            if (
                last["high"] > recent_high
                and last["close"] < recent_high
            ):
                down += 2
                independent_down += 1

                down_reasons.append(
                    "bearish liquidity rejection"
                )

            elif (
                last["low"] < recent_low
                and last["close"] > recent_low
            ):
                up += 2
                independent_up += 1

                up_reasons.append(
                    "bullish liquidity rejection"
                )

    # ========================================================
    # BUCKET 3: MOMENTUM
    # MAX 3
    # ========================================================

    momentum_up = 0
    momentum_down = 0

    if closes[-1] > closes[-4]:
        momentum_up += 1

    if closes[-1] > closes[-8]:
        momentum_up += 1

    if closes[-1] < closes[-4]:
        momentum_down += 1

    if closes[-1] < closes[-8]:
        momentum_down += 1

    if (
        momentum_up >= 2
        and momentum_down == 0
    ):
        up += 3
        independent_up += 1

        up_reasons.append(
            "bullish momentum"
        )

    elif (
        momentum_down >= 2
        and momentum_up == 0
    ):
        down += 3
        independent_down += 1

        down_reasons.append(
            "bearish momentum"
        )

    elif momentum_up > momentum_down:
        up += 1

    elif momentum_down > momentum_up:
        down += 1

    # ========================================================
    # BUCKET 4: EMA + ADX/DI
    # MAX 3
    # ========================================================

    trend_up = False
    trend_down = False

    if (
        ema9 is not None
        and ema21 is not None
    ):
        if (
            ema9 > ema21
            and current_price > ema9
        ):
            trend_up = True

        elif (
            ema9 < ema21
            and current_price < ema9
        ):
            trend_down = True

    if adx:
        if (
            adx["adx"] >= 18
            and adx["plus_di"]
            > adx["minus_di"]
        ):
            if trend_up:
                up += 3
                independent_up += 1

                up_reasons.append(
                    "EMA/ADX bullish trend confirmation"
                )
            else:
                up += 1

        elif (
            adx["adx"] >= 18
            and adx["minus_di"]
            > adx["plus_di"]
        ):
            if trend_down:
                down += 3
                independent_down += 1

                down_reasons.append(
                    "EMA/ADX bearish trend confirmation"
                )
            else:
                down += 1

    # If EMA trend exists but ADX is weak
    if not adx:
        if trend_up:
            up += 2
            independent_up += 1

        elif trend_down:
            down += 2
            independent_down += 1

    # ========================================================
    # BUCKET 5: RSI + WILLIAMS
    # MAX 2
    # ========================================================

    oscillator_up = 0
    oscillator_down = 0

    if rsi_value is not None:
        if (
            rsi_value > 52
            and rsi_value < 70
        ):
            oscillator_up += 1

        elif (
            rsi_value < 48
            and rsi_value > 30
        ):
            oscillator_down += 1

    if will_value is not None:
        if (
            will_value > -50
            and will_value < -20
        ):
            oscillator_up += 1

        elif (
            will_value < -50
            and will_value > -80
        ):
            oscillator_down += 1

    if oscillator_up >= 2:
        up += 2
        independent_up += 1

        up_reasons.append(
            "RSI/Williams bullish confirmation"
        )

    elif oscillator_down >= 2:
        down += 2
        independent_down += 1

        down_reasons.append(
            "RSI/Williams bearish confirmation"
        )

    elif oscillator_up > oscillator_down:
        up += 1

    elif oscillator_down > oscillator_up:
        down += 1

    # ========================================================
    # BUCKET 6: KELTNER
    # MAX 1
    # ========================================================

    if (
        keltner_ema is not None
        and keltner_atr is not None
    ):
        upper = (
            keltner_ema
            + 5.0 * keltner_atr
        )

        lower = (
            keltner_ema
            - 5.0 * keltner_atr
        )

        if (
            current_price > keltner_ema
            and current_price < upper
        ):
            up += 1

        elif (
            current_price < keltner_ema
            and current_price > lower
        ):
            down += 1

    # ========================================================
    # BUCKET 7: CANDLE QUALITY
    # MAX 2
    # ========================================================

    if body_ratio >= 0.65:
        if candle_dir == "UP":
            up += 2

        elif candle_dir == "DOWN":
            down += 2

    elif body_ratio >= 0.45:
        if candle_dir == "UP":
            up += 1

        elif candle_dir == "DOWN":
            down += 1

    # ========================================================
    # CAP RAW SCORES
    # ========================================================

    up = min(up, 20)
    down = min(down, 20)

    # ========================================================
    # BALANCE THE SCORES
    #
    # We don't want 0/20 or 20/20 too easily.
    # Score represents directional edge, not raw indicator count.
    # ========================================================

    raw_diff = up - down

    # directional edge capped to avoid artificial 20/0
    edge = clamp(
        raw_diff,
        -8,
        8
    )

    balanced_up = int(
        round(10 + edge)
    )

    balanced_down = 20 - balanced_up

    balanced_up = clamp(
        balanced_up,
        2,
        18
    )

    balanced_down = 20 - balanced_up

    # ========================================================
    # DIRECTION
    # ========================================================

    if balanced_up > balanced_down:
        direction = "UP"
        score = balanced_up
        opposite_score = balanced_down
        reasons = up_reasons
        independent_groups = independent_up

    elif balanced_down > balanced_up:
        direction = "DOWN"
        score = balanced_down
        opposite_score = balanced_up
        reasons = down_reasons
        independent_groups = independent_down

    else:
        # tie: use price action
        if (
            candle_dir == "UP"
            and up_count >= down_count
        ):
            direction = "UP"
            score = balanced_up
            opposite_score = balanced_down
            reasons = up_reasons
            independent_groups = independent_up

        else:
            direction = "DOWN"
            score = balanced_down
            opposite_score = balanced_up
            reasons = down_reasons
            independent_groups = independent_down

    # ========================================================
    # CONFLICT
    # ========================================================

    conflict = (
        abs(raw_diff) <= 2
    )

    if conflict:
        confidence_penalty = 5
    else:
        confidence_penalty = 0

    # ========================================================
    # CONFIDENCE
    # ========================================================

    gap = abs(
        balanced_up
        - balanced_down
    )

    confidence = (
        56
        + gap * 3
        + independent_groups * 2
        - confidence_penalty
    )

    # Don't give extreme confidence to weak scores
    if score < 14:
        confidence -= 5

    if score >= 17 and independent_groups >= 4:
        confidence += 2

    confidence = int(
        clamp(
            confidence,
            55,
            MAX_CONFIDENCE
        )
    )

    # ========================================================
    # DATA FRESHNESS
    # ========================================================

    candle_time = clean[-1]["time"]

    # ========================================================
    # ATR
    # ========================================================

    if not atr_value:
        atr_value = average_range(
            clean,
            14
        )

    # ========================================================
    # SNAPSHOT FINGERPRINT
    # ========================================================

    fingerprint_source = "|".join(
        [
            symbol,
            str(candle_time),
            f"{clean[-1]['open']:.10f}",
            f"{clean[-1]['high']:.10f}",
            f"{clean[-1]['low']:.10f}",
            f"{clean[-1]['close']:.10f}",
            str(clean[-1]["tick_volume"]),
        ]
    )

    fingerprint = hashlib.sha256(
        fingerprint_source.encode(
            "utf-8"
        )
    ).hexdigest()

    # ========================================================
    # REASON
    # ========================================================

    unique_reasons = []

    for reason in reasons:
        if (
            reason
            and reason not in unique_reasons
        ):
            unique_reasons.append(
                reason
            )

    if not unique_reasons:
        unique_reasons.append(
            "balanced technical price-action analysis"
        )

    reason_text = "; ".join(
        unique_reasons[:3]
    )

    return {
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "direction": direction,

        "up_score": balanced_up,
        "down_score": balanced_down,

        "score": score,
        "opposite_score": opposite_score,

        "confidence": confidence,

        "price": current_price,
        "atr": atr_value,

        "rsi": rsi_value,
        "williams": will_value,

        "ema9": ema9,
        "ema21": ema21,

        "adx": adx,

        "candle_time": candle_time,

        "fingerprint": fingerprint,

        "independent_groups":
            independent_groups,

        "conflict": conflict,

        "reason": reason_text,

        "candles": len(clean)
    }


# ============================================================
# MARKET DATA FRESHNESS
# ============================================================

def is_data_fresh(data):
    received_at = safe_float(
        data.get("received_at")
    )

    if received_at <= 0:
        return False

    age = (
        now_timestamp()
        - received_at
    )

    return age <= MARKET_DATA_MAX_AGE


# ============================================================
# CANDIDATES
# ============================================================

def get_candidates():
    candidates = []

    with state_lock:
        items = list(
            market_data.items()
        )

    for symbol, data in items:

        if not is_data_fresh(data):
            continue

        snapshot = build_snapshot(
            symbol,
            data
        )

        if not snapshot:
            continue

        # ----------------------------------------------------
        # MUST HAVE NEW CANDLE
        # ----------------------------------------------------

        candle_time = snapshot[
            "candle_time"
        ]

        old_candle = (
            last_processed_candle
            .get(symbol)
        )

        if (
            old_candle is not None
            and candle_time <= old_candle
        ):
            continue

        # ----------------------------------------------------
        # MUST NOT BE SAME SNAPSHOT
        # ----------------------------------------------------

        old_fingerprint = (
            last_processed_fingerprint
            .get(symbol)
        )

        if (
            old_fingerprint
            and snapshot["fingerprint"]
            == old_fingerprint
        ):
            continue

        # ----------------------------------------------------
        # QUALITY FLOOR
        # ----------------------------------------------------

        if snapshot["score"] < MIN_SCORE:
            continue

        # ----------------------------------------------------
        # STREAK PENALTY
        #
        # This does NOT force opposite direction.
        # It only makes same-direction candidates
        # work harder to beat alternatives.
        # ----------------------------------------------------

        selection_score = float(
            snapshot["score"]
        )

        if (
            last_signal_direction
            and snapshot["direction"]
            == last_signal_direction
        ):
            if (
                same_direction_streak
                >= STREAK_PENALTY_START
            ):
                selection_score -= min(
                    3,
                    same_direction_streak
                    - STREAK_PENALTY_START
                    + 1
                )

        # ----------------------------------------------------
        # SAME SYMBOL PENALTY
        # ----------------------------------------------------

        if (
            last_signal_symbol
            and symbol
            == last_signal_symbol
        ):
            selection_score -= SAME_SYMBOL_PENALTY

        # ----------------------------------------------------
        # FRESHNESS BONUS
        # ----------------------------------------------------

        age = (
            now_timestamp()
            - safe_float(
                data.get("received_at")
            )
        )

        if age <= 20:
            selection_score += 1.0

        elif age > 60:
            selection_score -= 1.0

        # ----------------------------------------------------
        # INDEPENDENT EVIDENCE BONUS
        # ----------------------------------------------------

        if snapshot[
            "independent_groups"
        ] >= 4:
            selection_score += 1.5

        elif snapshot[
            "independent_groups"
        ] >= 3:
            selection_score += 0.5

        # ----------------------------------------------------
        # CONFLICT PENALTY
        # ----------------------------------------------------

        if snapshot["conflict"]:
            selection_score -= 1.5

        snapshot[
            "selection_score"
        ] = selection_score

        candidates.append(
            snapshot
        )

    candidates.sort(
        key=lambda x: (
            x["selection_score"],
            x["score"],
            x["independent_groups"],
            x["confidence"]
        ),
        reverse=True
    )

    return candidates


# ============================================================
# DIRECTION DIVERSITY WITHOUT FORCING
# ============================================================

def choose_best_candidate(
    candidates
):
    if not candidates:
        return None

    top = candidates[0]

    # --------------------------------------------------------
    # If we have been getting same direction repeatedly,
    # check whether a strong opposite opportunity exists.
    #
    # We NEVER force opposite if it is weak.
    # --------------------------------------------------------

    if (
        last_signal_direction
        and same_direction_streak
        >= STREAK_PENALTY_START
    ):
        opposite = None

        for candidate in candidates:
            if (
                candidate["direction"]
                != last_signal_direction
            ):
                opposite = candidate
                break

        if opposite:
            difference = (
                top["selection_score"]
                - opposite["selection_score"]
            )

            # If opposite is almost equally strong,
            # use it to prevent directional concentration.
            if (
                difference <= 1.5
                and opposite["score"] >= MIN_SCORE
            ):
                return opposite

    return top


# ============================================================
# GEMINI VALIDATION / REASON
# ============================================================

def ask_gemini(
    candidates
):
    if not gemini_client:
        return candidates[0] if candidates else None

    if not candidates:
        return None

    compact = []

    for c in candidates[:TOP_CANDIDATES]:
        compact.append({
            "symbol": c["symbol"],
            "direction": c["direction"],
            "up_score": c["up_score"],
            "down_score": c["down_score"],
            "score": c["score"],
            "confidence": c["confidence"],
            "price": c["price"],
            "rsi": c["rsi"],
            "williams": c["williams"],
            "independent_groups":
                c["independent_groups"],
            "reason": c["reason"],
            "candle_time":
                c["candle_time"],
        })

    prompt = f"""
You are the final quality-control layer for a short-term M1
market-data scanner.

IMPORTANT:
The deterministic engine has already calculated the direction
and scores.

Your job is NOT to invent a new direction.

Choose exactly ONE candidate from the supplied candidates.

Rules:
1. Choose only one supplied symbol.
2. Keep its supplied direction.
3. Do not change UP to DOWN.
4. Do not change DOWN to UP.
5. Never invent a price.
6. Prefer independent confluence.
7. Prefer fresh candle data.
8. Avoid a candidate with obvious conflicting evidence.
9. Do not select a weak candidate just to change direction.
10. Accuracy is more important than direction diversity.
11. Never output WAIT, NEUTRAL or NO SIGNAL.
12. Confidence must remain realistic.
13. The deterministic score is authoritative.

Candidates:
{json.dumps(compact, ensure_ascii=False)}

Return ONLY valid JSON:

{{
  "symbol": "EXACT_SYMBOL",
  "direction": "UP or DOWN",
  "quality": 1-10,
  "reason": "short technical reason"
}}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        text = (
            response.text
            if response
            else ""
        )

        parsed = json.loads(text)

        selected_symbol = str(
            parsed.get("symbol", "")
        ).strip()

        selected_direction = normalize_direction(
            parsed.get("direction")
        )

        if not selected_symbol:
            return candidates[0]

        # ----------------------------------------------------
        # Find exact candidate
        # ----------------------------------------------------

        for candidate in candidates[:TOP_CANDIDATES]:

            if (
                candidate["symbol"]
                != selected_symbol
            ):
                continue

            # Gemini is NOT allowed to reverse direction.
            if (
                candidate["direction"]
                != selected_direction
            ):
                continue

            candidate = dict(
                candidate
            )

            ai_reason = str(
                parsed.get(
                    "reason",
                    ""
                )
            ).strip()

            if ai_reason:
                candidate[
                    "ai_reason"
                ] = ai_reason

            return candidate

        # If Gemini tries to override deterministic direction,
        # reject it.
        logger.warning(
            "Gemini tried invalid override; deterministic candidate retained"
        )

        return candidates[0]

    except Exception as e:
        logger.error(
            "Gemini selection error: %s",
            e
        )

        return candidates[0]


# ============================================================
# ENTRY TIME
# ============================================================

def calculate_entry_time():
    now = now_algiers()

    target = (
        now.replace(
            second=0,
            microsecond=0
        )
        + timedelta(
            minutes=ENTRY_DELAY_MINUTES
        )
    )

    return target


# ============================================================
# CANCELLATION PRICE
# ============================================================

def calculate_cancel_price(
    direction,
    price,
    atr_value
):
    if not atr_value or atr_value <= 0:
        # fallback small distance
        distance = abs(price) * 0.0003
    else:
        distance = (
            atr_value * 0.45
        )

    if direction == "UP":
        return price - distance

    return price + distance


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(
    candidate,
    mode
):
    direction = candidate[
        "direction"
    ]

    price = candidate[
        "price"
    ]

    cancel_price = calculate_cancel_price(
        direction,
        price,
        candidate.get("atr", 0)
    )

    digits = 5

    symbol = candidate[
        "symbol"
    ]

    data = market_data.get(
        symbol,
        {}
    )

    try:
        digits = int(
            data.get(
                "digits",
                5
            )
        )
    except Exception:
        digits = 5

    entry_time = calculate_entry_time()

    if mode == "RECOVERY":
        title = "🔁 RECOVERY 1/1"
    else:
        title = "🎯 BASE TRADE"

    direction_text = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    if direction == "UP":
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{cancel_price:.{digits}f}"
        )
    else:
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{cancel_price:.{digits}f}"
        )

    reason = candidate.get(
        "ai_reason"
    ) or candidate.get(
        "reason",
        "technical confluence"
    )

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | M1\n\n"
        f"{title}\n"
        f"{direction_text}\n\n"
        f"🔥 Confidence: "
        f"{candidate['confidence']}%\n"
        f"🟢 UP Score: "
        f"{candidate['up_score']}/20\n"
        f"🔴 DOWN Score: "
        f"{candidate['down_score']}/20\n\n"
        f"⏱️ Entry after: "
        f"{ENTRY_DELAY_MINUTES} minutes\n"
        f"🕐 ENTRY TIME: "
        f"{entry_time.strftime('%H:%M:%S')}\n"
        f"💰 Entry Price: "
        f"{price:.{digits}f}\n"
        f"{cancel_text}\n\n"
        f"🧠 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    return message


# ============================================================
# SEND SIGNAL
# ============================================================

async def send_signal(
    application,
    candidate,
    mode
):
    global active_trade
    global last_signal
    global last_signal_direction
    global same_direction_streak
    global last_signal_symbol
    global recovery_pending
    global recovery_number
    global stats

    with state_lock:

        if active_trade is not None:
            logger.info(
                "Active trade exists; signal blocked"
            )
            return False

        direction = candidate[
            "direction"
        ]

        symbol = candidate[
            "symbol"
        ]

        # ----------------------------------------------------
        # Direction streak
        # ----------------------------------------------------

        if (
            last_signal_direction
            == direction
        ):
            same_direction_streak += 1
        else:
            same_direction_streak = 1

        last_signal_direction = direction
        last_signal_symbol = symbol

        # ----------------------------------------------------
        # Store active trade
        # ----------------------------------------------------

        entry_time = calculate_entry_time()

        active_trade = {
            "symbol": symbol,
            "direction": direction,
            "mode": mode,
            "confidence":
                candidate["confidence"],
            "up_score":
                candidate["up_score"],
            "down_score":
                candidate["down_score"],
            "entry_price":
                candidate["price"],
            "entry_time":
                entry_time.isoformat(),
            "candle_time":
                candidate["candle_time"],
            "created_at":
                now_algiers().isoformat(),
        }

        last_signal = dict(
            active_trade
        )

        stats["signals"] += 1

        # Mark this candle as processed
        last_processed_candle[
            symbol
        ] = candidate[
            "candle_time"
        ]

        last_processed_fingerprint[
            symbol
        ] = candidate[
            "fingerprint"
        ]

        save_history()

    message = format_signal(
        candidate,
        mode
    )

    try:
        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=message
        )

        logger.info(
            "SIGNAL SENT | %s %s | %s | %s/%s",
            symbol,
            direction,
            mode,
            candidate["up_score"],
            candidate["down_score"]
        )

        return True

    except Exception as e:
        logger.exception(
            "Telegram send error: %s",
            e
        )

        with state_lock:
            active_trade = None

        return False


# ============================================================
# PROCESS BATCH
# ============================================================

async def process_complete_batch(
    application
):
    global processing_batch_id
    global last_processed_batch_id
    global latest_batch_complete
    global recovery_pending
    global recovery_wait_until
    global recovery_number

    with state_lock:

        if not latest_batch_complete:
            return

        batch_id = latest_batch_id

        if not batch_id:
            return

        if (
            batch_id
            == last_processed_batch_id
        ):
            return

        if active_trade is not None:
            return

        # Recovery waiting period
        if recovery_pending:
            if (
                now_timestamp()
                < recovery_wait_until
            ):
                return

        if (
            processing_batch_id
            == batch_id
        ):
            return

        processing_batch_id = batch_id

    try:

        candidates = get_candidates()

        if not candidates:
            logger.info(
                "No fresh qualifying candidates in batch %s",
                batch_id
            )

            with state_lock:
                last_processed_batch_id = batch_id
                latest_batch_complete = False
                processing_batch_id = None

            return

        # ----------------------------------------------------
        # Select deterministic best candidate first
        # ----------------------------------------------------

        selected = choose_best_candidate(
            candidates
        )

        if not selected:
            with state_lock:
                processing_batch_id = None

            return

        # ----------------------------------------------------
        # Gemini checks only top candidates.
        # It cannot reverse deterministic direction.
        # ----------------------------------------------------

        gemini_selected = ask_gemini(
            candidates[:TOP_CANDIDATES]
        )

        if gemini_selected:
            selected = gemini_selected

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        if selected["score"] < MIN_SCORE:
            logger.info(
                "Selected candidate below minimum score"
            )

            with state_lock:
                last_processed_batch_id = batch_id
                latest_batch_complete = False
                processing_batch_id = None

            return

        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        if recovery_pending:
            mode = "RECOVERY"
        else:
            mode = "BASE"

        # ----------------------------------------------------
        # Send ONE signal
        # ----------------------------------------------------

        sent = await send_signal(
            application,
            selected,
            mode
        )

        with state_lock:

            last_processed_batch_id = batch_id
            latest_batch_complete = False
            processing_batch_id = None

            if sent and mode == "RECOVERY":
                recovery_pending = False
                recovery_number = 1

    except Exception as e:
        logger.exception(
            "Batch processing error: %s",
            e
        )

        with state_lock:
            processing_batch_id = None


# ============================================================
# BACKGROUND LOOP
# ============================================================

async def background_loop(
    application
):
    logger.info(
        "Background scanner started"
    )

    while True:
        try:
            await process_complete_batch(
                application
            )

        except Exception as e:
            logger.exception(
                "Background loop error: %s",
                e
            )

        await asyncio.sleep(
            BACKGROUND_INTERVAL
        )


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update):
    if not update:
        return False

    user = update.effective_user

    if not user:
        return False

    return user.id == OWNER_ID


async def owner_only(update):
    if not is_owner(update):
        try:
            await update.message.reply_text(
                "⛔ هذا البوت خاص بالمالك فقط."
            )
        except Exception:
            pass

        return False

    return True


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ Scanner: ON\n"
        "📊 Timeframe: M1\n"
        "🔎 Multi-pair analysis: ON\n"
        "🎯 One strongest trade only\n"
        "🔁 Recovery: 1/1\n\n"
        "الأوامر:\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt5status"
    )


# ============================================================
# /WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number

    if not await owner_only(update):
        return

    with state_lock:

        if active_trade is None:
            await update.message.reply_text(
                "ℹ️ ما كاش صفقة نشطة حاليًا."
            )
            return

        trade = dict(
            active_trade
        )

        mode = trade.get(
            "mode",
            "BASE"
        )

        stats["wins"] += 1

        if mode == "RECOVERY":
            stats["recovery_wins"] += 1
        else:
            stats["base_wins"] += 1

        history.append({
            "result": "WIN",
            "symbol":
                trade.get("symbol"),
            "direction":
                trade.get("direction"),
            "mode": mode,
            "confidence":
                trade.get("confidence"),
            "up_score":
                trade.get("up_score"),
            "down_score":
                trade.get("down_score"),
            "entry_price":
                trade.get("entry_price"),
            "entry_time":
                trade.get("entry_time"),
            "time":
                now_algiers().isoformat(),
        })

        active_trade = None

        # أي Recovery انتهت
        recovery_pending = False
        recovery_wait_until = 0
        recovery_number = 0

        save_history()

    await update.message.reply_text(
        "✅ WIN مسجلة\n\n"
        f"📊 {trade.get('symbol')} | "
        f"{trade.get('direction')}\n"
        f"🎯 {mode}\n\n"
        "🔎 الآن البوت يرجع يبحث على أقوى فرصة جديدة."
    )


# ============================================================
# /LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number

    if not await owner_only(update):
        return

    with state_lock:

        if active_trade is None:
            await update.message.reply_text(
                "ℹ️ ما كاش صفقة نشطة حاليًا."
            )
            return

        trade = dict(
            active_trade
        )

        mode = trade.get(
            "mode",
            "BASE"
        )

        stats["losses"] += 1

        if mode == "RECOVERY":
            stats["recovery_losses"] += 1
        else:
            stats["base_losses"] += 1

        history.append({
            "result": "LOSS",
            "symbol":
                trade.get("symbol"),
            "direction":
                trade.get("direction"),
            "mode": mode,
            "confidence":
                trade.get("confidence"),
            "up_score":
                trade.get("up_score"),
            "down_score":
                trade.get("down_score"),
            "entry_price":
                trade.get("entry_price"),
            "entry_time":
                trade.get("entry_time"),
            "time":
                now_algiers().isoformat(),
        })

        active_trade = None

        if mode == "BASE":
            recovery_pending = True

            recovery_number = 1

            recovery_wait_until = (
                now_timestamp()
                + RECOVERY_DELAY_SECONDS
            )

            save_history()

            await update.message.reply_text(
                "❌ LOSS مسجلة\n\n"
                f"📊 {trade.get('symbol')} | "
                f"{trade.get('direction')}\n"
                "🎯 BASE TRADE\n\n"
                "🔁 Recovery 1/1 مفعلة.\n"
                f"⏱️ انتظر {RECOVERY_DELAY_SECONDS // 60} "
                "دقائق لإعادة تحليل السوق.\n\n"
                "⚠️ Recovery لن تستعمل التحليل القديم."
            )

            return

        # ----------------------------------------------------
        # RECOVERY LOSS
        # ----------------------------------------------------

        recovery_pending = False
        recovery_wait_until = 0
        recovery_number = 0

        save_history()

    await update.message.reply_text(
        "❌ LOSS مسجلة\n\n"
        f"📊 {trade.get('symbol')} | "
        f"{trade.get('direction')}\n"
        "🔁 RECOVERY 1/1\n\n"
        "⛔ Recovery انتهت.\n"
        "🔎 البوت يرجع الآن يبحث عن BASE جديدة."
    )


# ============================================================
# /STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    with state_lock:
        wins = stats["wins"]
        losses = stats["losses"]
        signals = stats["signals"]

        total = wins + losses

        if total > 0:
            winrate = (
                wins
                / total
                * 100
            )
        else:
            winrate = 0.0

        active = (
            active_trade is not None
        )

        recovery = recovery_pending

        direction = (
            last_signal_direction
            or "-"
        )

        streak = same_direction_streak

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📌 Signals: {signals}\n"
        f"✅ Wins: {wins}\n"
        f"❌ Losses: {losses}\n"
        f"🎯 Win Rate: {winrate:.1f}%\n\n"
        f"🟢 Base Wins: {stats['base_wins']}\n"
        f"🔴 Base Losses: {stats['base_losses']}\n"
        f"🔁 Recovery Wins: {stats['recovery_wins']}\n"
        f"🔁 Recovery Losses: {stats['recovery_losses']}\n\n"
        f"📍 Active Trade: "
        f"{'YES' if active else 'NO'}\n"
        f"🔁 Recovery Pending: "
        f"{'YES' if recovery else 'NO'}\n"
        f"🧭 Last Direction: {direction}\n"
        f"📈 Direction Streak: {streak}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# /RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global stats
    global history
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number
    global last_signal_direction
    global same_direction_streak
    global last_signal_symbol

    if not await owner_only(update):
        return

    with state_lock:

        stats = {
            "wins": 0,
            "losses": 0,
            "base_wins": 0,
            "base_losses": 0,
            "recovery_wins": 0,
            "recovery_losses": 0,
            "signals": 0,
        }

        history = []

        active_trade = None

        recovery_pending = False
        recovery_wait_until = 0
        recovery_number = 0

        last_signal_direction = None
        same_direction_streak = 0
        last_signal_symbol = None

        save_history()

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات والحالة."
    )


# ============================================================
# /MT5STATUS
# ============================================================

async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    with state_lock:

        symbols = list(
            market_data.keys()
        )

        fresh = 0
        stale = 0

        for symbol in symbols:
            if is_data_fresh(
                market_data[symbol]
            ):
                fresh += 1
            else:
                stale += 1

        batch = latest_batch_id
        complete = latest_batch_complete

    await update.message.reply_text(
        "🖥️ MT5 STATUS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 Symbols received: {len(symbols)}\n"
        f"🟢 Fresh: {fresh}\n"
        f"🔴 Stale: {stale}\n"
        f"📦 Batch: {batch or '-'}\n"
        f"✅ Complete: "
        f"{'YES' if complete else 'NO'}\n"
        "━━━━━━━━━━━━━━━━━━"
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

        path = parsed.path

        if path in (
            "/",
            "/health",
            "/healthz"
        ):
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service":
                        "ZinoProSignalAI",
                    "timeframe": TIMEFRAME
                }
            )

            return

        if path in (
            "/mt5status",
            "/api/mt5status"
        ):
            with state_lock:
                symbols = len(
                    market_data
                )

            self.send_json(
                200,
                {
                    "status": "ok",
                    "symbols": symbols
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

        path = parsed.path

        if path not in (
            "/mt4",
            "/api/mt4",
            "/mt5",
            "/api/mt5"
        ):
            self.send_json(
                404,
                {
                    "error": "not found"
                }
            )

            return

        # --------------------------------------------------------
        # API KEY
        # --------------------------------------------------------

        received_key = (
            self.headers.get(
                "X-MT5-API-Key"
            )
            or
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

        if (
            not MT4_API_KEY
            or received_key
            != MT4_API_KEY
        ):
            logger.warning(
                "Unauthorized market-data request"
            )

            self.send_json(
                401,
                {
                    "error":
                        "unauthorized"
                }
            )

            return

        # --------------------------------------------------------
        # CONTENT LENGTH
        # --------------------------------------------------------

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
            self.send_json(
                400,
                {
                    "error":
                        "empty body"
                }
            )

            return

        if content_length > 10_000_000:
            self.send_json(
                413,
                {
                    "error":
                        "payload too large"
                }
            )

            return

        # --------------------------------------------------------
        # READ BODY
        # --------------------------------------------------------

        try:
            raw = self.rfile.read(
                content_length
            )

            body = raw.decode(
                "utf-8"
            )

            payload = json.loads(
                body
            )

        except Exception as e:
            logger.warning(
                "Invalid JSON: %s",
                e
            )

            self.send_json(
                400,
                {
                    "error":
                        "invalid json"
                }
            )

            return

        # --------------------------------------------------------
        # VALIDATE
        # --------------------------------------------------------

        symbol = str(
            payload.get(
                "symbol",
                ""
            )
        ).strip()

        timeframe = str(
            payload.get(
                "timeframe",
                ""
            )
        ).upper().strip()

        batch_id = str(
            payload.get(
                "batch_id",
                ""
            )
        ).strip()

        candles = payload.get(
            "candles"
        )

        if not symbol:
            self.send_json(
                400,
                {
                    "error":
                        "missing symbol"
                }
            )

            return

        if timeframe != "M1":
            self.send_json(
                400,
                {
                    "error":
                        "only M1 accepted"
                }
            )

            return

        if not batch_id:
            self.send_json(
                400,
                {
                    "error":
                        "missing batch_id"
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
                    "error":
                        "candles must be list"
                }
            )

            return

        if len(candles) < MIN_CANDLES:
            self.send_json(
                400,
                {
                    "error":
                        f"need at least {MIN_CANDLES} candles"
                }
            )

            return

        # --------------------------------------------------------
        # VALIDATE CANDLES
        # --------------------------------------------------------

        clean_candles = []

        try:
            for c in candles[-MAX_CANDLES:]:

                if not isinstance(
                    c,
                    dict
                ):
                    continue

                clean_candles.append({
                    "time": safe_int(
                        c.get("time")
                    ),
                    "open": safe_float(
                        c.get("open")
                    ),
                    "high": safe_float(
                        c.get("high")
                    ),
                    "low": safe_float(
                        c.get("low")
                    ),
                    "close": safe_float(
                        c.get("close")
                    ),
                    "tick_volume":
                        safe_int(
                            c.get(
                                "tick_volume"
                            )
                        ),
                    "real_volume":
                        safe_int(
                            c.get(
                                "real_volume"
                            )
                        ),
                    "spread":
                        safe_int(
                            c.get(
                                "spread"
                            )
                        ),
                })

        except Exception as e:
            self.send_json(
                400,
                {
                    "error":
                        "invalid candle data",
                    "detail":
                        str(e)
                }
            )

            return

        if len(clean_candles) < MIN_CANDLES:
            self.send_json(
                400,
                {
                    "error":
                        "not enough valid candles"
                }
            )

            return

        clean_candles.sort(
            key=lambda x: x["time"]
        )

        # --------------------------------------------------------
        # CLOSED CANDLE
        # --------------------------------------------------------

        supplied_closed_time = safe_int(
            payload.get(
                "closed_candle_time"
            )
        )

        if supplied_closed_time > 0:
            closed_candle_time = (
                supplied_closed_time
            )
        else:
            closed_candle_time = (
                clean_candles[-1]["time"]
            )

        # --------------------------------------------------------
        # CURRENT PRICE
        # --------------------------------------------------------

        current_bid = safe_float(
            payload.get(
                "current_bid"
            ),
            clean_candles[-1]["close"]
        )

        current_ask = safe_float(
            payload.get(
                "current_ask"
            ),
            current_bid
        )

        digits = safe_int(
            payload.get(
                "digits"
            ),
            5
        )

        batch_complete = bool(
            payload.get(
                "batch_complete",
                False
            )
        )

        # --------------------------------------------------------
        # STORE
        # --------------------------------------------------------

        with state_lock:

            # New batch
            if (
                latest_batch_id
                != batch_id
            ):
                latest_batch_id = batch_id

                latest_batch_started_at = (
                    now_timestamp()
                )

                latest_batch_complete = False

            market_data[symbol] = {
                "symbol": symbol,
                "timeframe": "M1",
                "batch_id": batch_id,
                "candles": clean_candles,
                "current_bid": current_bid,
                "current_ask": current_ask,
                "digits": digits,
                "closed_candle_time":
                    closed_candle_time,
                "batch_complete":
                    batch_complete,
                "received_at":
                    now_timestamp(),
            }

            # Only the last EA request should mark complete
            if (
                batch_complete
                and latest_batch_id
                == batch_id
            ):
                latest_batch_complete = True

        logger.info(
            "MT5 DATA | %s | candles=%s | "
            "closed=%s | complete=%s | batch=%s",
            symbol,
            len(clean_candles),
            closed_candle_time,
            batch_complete,
            batch_id
        )

        self.send_json(
            200,
            {
                "status": "accepted",
                "symbol": symbol,
                "batch_id": batch_id,
                "closed_candle_time":
                    closed_candle_time,
                "batch_complete":
                    batch_complete
            }
        )


# ============================================================
# START HTTP SERVER
# ============================================================

def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# POST INIT
# ============================================================

async def post_init(
    application
):
    application.create_task(
        background_loop(
            application
        )
    )

    logger.info(
        "ZinoProSignalAI background task started"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is missing"
        )

    if not MT4_API_KEY:
        logger.warning(
            "MT4_API_KEY is missing"
        )

    # --------------------------------------------------------
    # HTTP SERVER
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
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

    application.add_handler(
        CommandHandler(
            "start",
            start_command
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
            "stats",
            stats_command
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

    logger.info(
        "ZinoProSignalAI starting Telegram polling..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
