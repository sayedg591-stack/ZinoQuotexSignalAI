import os
import json
import logging
import threading
import asyncio
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
OWNER_ID = os.getenv("OWNER_ID", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

ENTRY_DELAY_MINUTES = 2
SIGNAL_COOLDOWN_SECONDS = 180

MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10

SIGNAL_HISTORY_FILE = "signal_history.json"
STATE_FILE = "bot_state.json"

GEMINI_MIN_INTERVAL_SECONDS = 900
GEMINI_429_COOLDOWN_SECONDS = 4 * 60 * 60

LOCAL_GEMINI_MIN_SCORE = 12


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBALS
# ============================================================

mt4_data = {}

telegram_application = None
telegram_loop = None

last_signal_sent_at = 0.0
next_signal_allowed_at = 0.0
pending_recovery = False

gemini_last_call = 0.0
gemini_disabled_until = 0.0


# ============================================================
# BASE / RECOVERY STATE
# ============================================================

active_cycle = {
    "active": False,
    "symbol": None,
    "timeframe": None,
    "trade_type": None,
    "direction": None,
    "recovery_used": False,
    "last_trade_time": 0.0,
    "trade_number": 0,
}


stats_data = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}


trade_history = []

current_trade_id = None


# ============================================================
# FILE HELPERS
# ============================================================

def atomic_write_json(filename, data):
    temp_file = filename + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temp_file, filename)


# ============================================================
# HISTORY
# ============================================================

def save_history():
    try:
        atomic_write_json(
            SIGNAL_HISTORY_FILE,
            trade_history,
        )

        logger.info(
            "signal_history.json saved | records=%s",
            len(trade_history),
        )

    except Exception as e:
        logger.exception(
            "Failed saving signal history: %s",
            e,
        )


def load_history():
    global trade_history

    try:
        if not os.path.exists(SIGNAL_HISTORY_FILE):
            trade_history = []
            return

        with open(
            SIGNAL_HISTORY_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            trade_history = data
        else:
            trade_history = []

        logger.info(
            "Loaded signal history | records=%s",
            len(trade_history),
        )

    except Exception as e:
        logger.exception(
            "Failed loading signal history: %s",
            e,
        )
        trade_history = []


# ============================================================
# STATE PERSISTENCE
# ============================================================

def save_state():
    try:
        state = {
            "active_cycle": active_cycle,
            "stats_data": stats_data,
            "current_trade_id": current_trade_id,
            "next_signal_allowed_at": next_signal_allowed_at,
            "pending_recovery": pending_recovery,
        }

        atomic_write_json(
            STATE_FILE,
            state,
        )

    except Exception as e:
        logger.exception(
            "Failed saving bot state: %s",
            e,
        )


def load_state():
    global active_cycle
    global stats_data
    global current_trade_id
    global next_signal_allowed_at
    global pending_recovery

    try:
        if not os.path.exists(STATE_FILE):
            return

        with open(
            STATE_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            state = json.load(f)

        saved_cycle = state.get("active_cycle")

        if isinstance(saved_cycle, dict):
            active_cycle.update(saved_cycle)

        saved_stats = state.get("stats_data")

        if isinstance(saved_stats, dict):
            stats_data.update(saved_stats)

        current_trade_id = state.get("current_trade_id")
        next_signal_allowed_at = float(state.get("next_signal_allowed_at", 0.0) or 0.0)
        pending_recovery = bool(state.get("pending_recovery", False))

        logger.info(
            "Bot state loaded | active=%s | trade_id=%s",
            active_cycle.get("active"),
            current_trade_id,
        )

    except Exception as e:
        logger.exception(
            "Failed loading bot state: %s",
            e,
        )


# ============================================================
# SYNC STATE FROM HISTORY
# ============================================================

def rebuild_state_from_history():
    global current_trade_id

    pending = None

    for record in reversed(trade_history):
        if record.get("result") == "PENDING":
            pending = record
            break

    if pending:
        current_trade_id = pending.get("id")

        trade_type = pending.get(
            "trade_type",
            "BASE",
        )

        if trade_type == "RECOVERY":
            active_cycle["active"] = True
            active_cycle["symbol"] = pending.get("symbol")
            active_cycle["timeframe"] = pending.get("timeframe")
            active_cycle["trade_type"] = "RECOVERY"
            active_cycle["direction"] = pending.get("direction")
            active_cycle["recovery_used"] = True
            active_cycle["trade_number"] = 2

        else:
            active_cycle["active"] = True
            active_cycle["symbol"] = pending.get("symbol")
            active_cycle["timeframe"] = pending.get("timeframe")
            active_cycle["trade_type"] = "BASE"
            active_cycle["direction"] = pending.get("direction")
            active_cycle["recovery_used"] = False
            active_cycle["trade_number"] = 1

        logger.info(
            "Recovered pending trade | id=%s | type=%s",
            current_trade_id,
            trade_type,
        )


# ============================================================
# OWNER
# ============================================================

def is_owner(update: Update):
    if not update.effective_user:
        return False

    if not OWNER_ID:
        return False

    try:
        return update.effective_user.id == int(OWNER_ID)

    except Exception:
        return False


# ============================================================
# TIME
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def format_dt(dt):
    if isinstance(dt, str):
        return dt

    return dt.astimezone(ALGIERS).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# TRADE ID
# ============================================================

def generate_trade_id():
    return (
        datetime.now().strftime("%Y%m%d%H%M%S")
        + "-"
        + str(int(time.time() * 1000))[-5:]
    )


# ============================================================
# TRADE HISTORY
# ============================================================

def create_trade_record(
    symbol,
    timeframe,
    trade_type,
    direction,
    confidence,
    up_score,
    down_score,
    entry_time,
    entry_price,
    cancellation_level,
    reason,
    message_id=None,
    signal_text="",
    recovery_level=0,
):
    global current_trade_id

    trade_id = generate_trade_id()

    record = {
        "id": trade_id,
        "created_at": format_dt(now_algiers()),
        "symbol": symbol,
        "timeframe": timeframe,
        "trade_type": trade_type,
        "recovery_level": recovery_level,
        "direction": direction,
        "confidence": confidence,
        "up_score": up_score,
        "down_score": down_score,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "cancellation_level": cancellation_level,
        "reason": reason,
        "result": "PENDING",
        "result_time": None,
        "message_id": message_id,
        "signal_text": signal_text,
    }

    trade_history.append(record)
    current_trade_id = trade_id

    save_history()
    save_state()

    logger.info(
        "Trade created | id=%s | type=%s | recovery=%s | %s | %s",
        trade_id,
        trade_type,
        recovery_level,
        symbol,
        direction,
    )

    return record

# ============================================================
# RESULT UPDATE
# ============================================================

def update_current_trade_result(result):
    global current_trade_id

    if result not in ("WIN", "LOSS"):
        return False

    record = None

    if current_trade_id:
        candidate = find_trade_by_id(current_trade_id)
        if candidate and candidate.get("result") == "PENDING":
            record = candidate

    if record is None:
        record = get_latest_pending_trade()

    if record is None:
        logger.warning(
            "No pending trade found for result=%s",
            result,
        )
        return False

    record["result"] = result
    record["result_time"] = format_dt(now_algiers())
    current_trade_id = record.get("id")

    save_history()
    save_state()

    # Update the original signal card so the result appears at its bottom.
    try:
        edit_signal_result_safely(record, result)
    except Exception:
        logger.exception("Failed to update signal card result.")

    logger.info(
        "Trade result recorded | id=%s | type=%s | result=%s",
        record.get("id"),
        record.get("trade_type"),
        result,
    )

    return True

# ============================================================
# CYCLE
# ============================================================

def start_base_cycle(
    symbol,
    timeframe,
    direction,
):
    active_cycle["active"] = True
    active_cycle["symbol"] = symbol
    active_cycle["timeframe"] = timeframe
    active_cycle["trade_type"] = "BASE"
    active_cycle["direction"] = direction
    active_cycle["recovery_used"] = False
    active_cycle["last_trade_time"] = time.time()
    active_cycle["trade_number"] = 1

    save_state()

    logger.info(
        "BASE cycle started | %s | %s",
        symbol,
        direction,
    )


def start_recovery_cycle():
    if not active_cycle["active"]:
        return False

    # --------------------------------------------------------
    # HARD LIMIT:
    # never create Recovery 2
    # --------------------------------------------------------

    if active_cycle.get("recovery_used"):
        logger.warning(
            "Recovery already used. Recovery 2 blocked."
        )
        return False

    active_cycle["trade_type"] = "RECOVERY"
    active_cycle["recovery_used"] = True
    active_cycle["trade_number"] = 2
    active_cycle["last_trade_time"] = time.time()

    save_state()

    logger.info(
        "RECOVERY 1/1 enabled | %s",
        active_cycle.get("symbol"),
    )

    return True


def reset_cycle():
    global current_trade_id

    active_cycle["active"] = False
    active_cycle["symbol"] = None
    active_cycle["timeframe"] = None
    active_cycle["trade_type"] = None
    active_cycle["direction"] = None
    active_cycle["recovery_used"] = False
    active_cycle["last_trade_time"] = 0.0
    active_cycle["trade_number"] = 0

    current_trade_id = None

    save_state()

    logger.info("Cycle reset")

# ============================================================
# STATISTICS
# ============================================================

def calculate_history_stats():
    wins = 0
    losses = 0
    pending = 0

    base_wins = 0
    base_losses = 0

    recovery_wins = 0
    recovery_losses = 0

    for record in trade_history:
        result = record.get("result")
        trade_type = record.get("trade_type")

        if result == "WIN":
            wins += 1

            if trade_type == "BASE":
                base_wins += 1

            elif trade_type == "RECOVERY":
                recovery_wins += 1

        elif result == "LOSS":
            losses += 1

            if trade_type == "BASE":
                base_losses += 1

            elif trade_type == "RECOVERY":
                recovery_losses += 1

        elif result == "PENDING":
            pending += 1

    return {
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "base_wins": base_wins,
        "base_losses": base_losses,
        "recovery_wins": recovery_wins,
        "recovery_losses": recovery_losses,
    }


# ============================================================
# CANDLE HELPERS
# ============================================================

def normalize_candle(candle):
    if not isinstance(candle, dict):
        return None

    try:
        timestamp = candle.get(
            "timestamp",
            candle.get(
                "time",
                candle.get("datetime"),
            ),
        )

        open_price = float(
            candle.get(
                "open",
                candle.get("o"),
            )
        )

        high_price = float(
            candle.get(
                "high",
                candle.get("h"),
            )
        )

        low_price = float(
            candle.get(
                "low",
                candle.get("l"),
            )
        )

        close_price = float(
            candle.get(
                "close",
                candle.get("c"),
            )
        )

        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
        }

    except Exception:
        return None


def extract_candles(data):
    if not isinstance(data, dict):
        return []

    candles = data.get("candles")

    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:
        normalized = normalize_candle(candle)

        if normalized:
            result.append(normalized)

    return result


def remove_forming_candle(candles):
    if len(candles) <= 1:
        return candles

    return candles[:-1]


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    current = sum(values[:period]) / period

    for value in values[period:]:
        current = (
            (value - current) * multiplier
        ) + current

    return current


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        change = values[i] - values[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]

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
    ) * -100


def atr(candles, period=10):
    if len(candles) <= period:
        return None

    trs = []

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

        trs.append(tr)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def adx_values(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

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
            if up_move > down_move and up_move > 0
            else 0
        )

        minus = (
            down_move
            if down_move > up_move and down_move > 0
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
        return None, None, None

    tr_sum = sum(trs[-period:])
    plus_sum = sum(plus_dm[-period:])
    minus_sum = sum(minus_dm[-period:])

    if tr_sum == 0:
        return 0.0, 0.0, 0.0

    plus_di = (
        100 * plus_sum / tr_sum
    )

    minus_di = (
        100 * minus_sum / tr_sum
    )

    denominator = plus_di + minus_di

    if denominator == 0:
        adx = 0.0
    else:
        adx = (
            100
            * abs(plus_di - minus_di)
            / denominator
        )

    return adx, plus_di, minus_di


# ============================================================
# LOCAL ANALYSIS
# ============================================================

def analyze_local(candles):
    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    closes = [
        c["close"]
        for c in candles
    ]

    current = candles[-1]
    previous = candles[-2]

    current_close = current["close"]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi14 = rsi(
        closes,
        14,
    )

    wr14 = williams_r(
        candles,
        14,
    )

    adx14, plus_di, minus_di = adx_values(
        candles,
        14,
    )

    atr10 = atr(
        candles,
        10,
    )

    up = 0
    down = 0

    reasons = []

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    recent = candles[-8:]

    recent_high = max(
        c["high"] for c in recent[:-1]
    )

    recent_low = min(
        c["low"] for c in recent[:-1]
    )

    if current_close > recent_high:
        up += 2
        reasons.append(
            "structure bullish"
        )

    elif current_close < recent_low:
        down += 2
        reasons.append(
            "structure bearish"
        )

    else:
        if current_close > previous["close"]:
            up += 1
        elif current_close < previous["close"]:
            down += 1

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    if current["close"] > previous["high"]:
        up += 2
        reasons.append(
            "bullish breakout"
        )

    elif current["close"] < previous["low"]:
        down += 2
        reasons.append(
            "bearish breakout"
        )

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    if current["low"] < previous["low"] and current["close"] > previous["low"]:
        up += 1
        reasons.append(
            "sell-side liquidity reaction"
        )

    elif current["high"] > previous["high"] and current["close"] < previous["high"]:
        down += 1
        reasons.append(
            "buy-side liquidity reaction"
        )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if len(closes) >= 4:
        momentum = (
            closes[-1] - closes[-4]
        )

        if momentum > 0:
            up += 2
        elif momentum < 0:
            down += 2

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    body = abs(
        current["close"]
        - current["open"]
    )

    candle_range = (
        current["high"]
        - current["low"]
    )

    if candle_range > 0:
        body_ratio = body / candle_range

        if (
            current["close"]
            > current["open"]
            and body_ratio >= 0.55
        ):
            up += 2

        elif (
            current["close"]
            < current["open"]
            and body_ratio >= 0.55
        ):
            down += 2

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi14 is not None:

        if rsi14 > 55 and rsi14 < 75:
            up += 1

        elif rsi14 < 45 and rsi14 > 25:
            down += 1

    # --------------------------------------------------------
    # Oscillators
    # Williams %R
    # --------------------------------------------------------

    if wr14 is not None:

        if wr14 > -50:
            up += 1

        elif wr14 < -50:
            down += 1

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if (
            current_close > ema9
            and ema9 > ema21
        ):
            up += 2
            reasons.append(
                "EMA 9/21 bullish"
            )

        elif (
            current_close < ema9
            and ema9 < ema21
        ):
            down += 2
            reasons.append(
                "EMA 9/21 bearish"
            )

    # --------------------------------------------------------
    # ADX / DI confirmation
    # --------------------------------------------------------

    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
    ):

        if adx14 >= 20:

            if plus_di > minus_di:
                up += 1

            elif minus_di > plus_di:
                down += 1

    # --------------------------------------------------------
    # Normalize to maximum displayed 18
    # --------------------------------------------------------

    raw_max = 16

    if up == down:
        direction = (
            "UP"
            if current["close"] >= previous["close"]
            else "DOWN"
        )

    else:
        direction = (
            "UP"
            if up > down
            else "DOWN"
        )

    if direction == "UP":
        confidence = (
            50
            + ((up - down) / raw_max) * 50
        )

    else:
        confidence = (
            50
            + ((down - up) / raw_max) * 50
        )

    confidence = max(
        50,
        min(89, confidence),
    )

    up_score = round(
        (up / raw_max) * 18
    )

    down_score = round(
        (down / raw_max) * 18
    )

    up_score = max(
        0,
        min(18, up_score),
    )

    down_score = max(
        0,
        min(18, down_score),
    )

    # --------------------------------------------------------
    # ATR / Keltner context
    # --------------------------------------------------------

    if atr10 is not None and ema21 is not None:

        keltner_upper = (
            ema21 + (atr10 * 5)
        )

        keltner_lower = (
            ema21 - (atr10 * 5)
        )

        if current_close > keltner_upper:
            reasons.append(
                "above Keltner upper"
            )

        elif current_close < keltner_lower:
            reasons.append(
                "below Keltner lower"
            )

    if adx14 is not None:
        reasons.append(
            f"ADX {adx14:.1f}"
        )

    if rsi14 is not None:
        reasons.append(
            f"RSI {rsi14:.1f}"
        )

    reason = ", ".join(reasons[:4])

    if not reason:
        reason = (
            "Price action and indicator confluence"
        )

    return {
        "direction": direction,
        "confidence": round(confidence),
        "up_score": up_score,
        "down_score": down_score,
        "reason": reason,
        "ema9": ema9,
        "ema21": ema21,
        "rsi": rsi14,
        "williams_r": wr14,
        "adx": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "atr": atr10,
    }


# ============================================================
# CANCELLATION LEVEL
# ============================================================

def calculate_cancellation_level(
    candles,
    direction,
    entry_price,
):
    recent = candles[-8:]

    if direction == "UP":
        level = min(
            c["low"]
            for c in recent
        )

        text = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{level:.6f}"
        )

    else:
        level = max(
            c["high"]
            for c in recent
        )

        text = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{level:.6f}"
        )

    return level, text


# ============================================================
# GEMINI
# ============================================================

def gemini_available():
    global gemini_disabled_until

    if not GEMINI_API_KEY:
        return False

    if time.time() < gemini_disabled_until:
        return False

    return True


def gemini_confirm(
    symbol,
    timeframe,
    candles,
    local_result,
):
    global gemini_last_call
    global gemini_disabled_until

    if not gemini_available():
        return None

    if local_result["up_score"] == local_result["down_score"]:
        return None

    if max(
        local_result["up_score"],
        local_result["down_score"],
    ) < LOCAL_GEMINI_MIN_SCORE:
        return None

    if (
        time.time() - gemini_last_call
        < GEMINI_MIN_INTERVAL_SECONDS
    ):
        return None

    try:
        gemini_last_call = time.time()

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        recent = candles[-25:]

        candle_text = []

        for c in recent:
            candle_text.append(
                {
                    "o": c["open"],
                    "h": c["high"],
                    "l": c["low"],
                    "c": c["close"],
                }
            )

        prompt = f"""
You are confirming a technical H1 market analysis.

Symbol: {symbol}
Timeframe: {timeframe}

Local direction:
{local_result["direction"]}

Local confidence:
{local_result["confidence"]}

UP score:
{local_result["up_score"]}/18

DOWN score:
{local_result["down_score"]}/18

Recent candles:
{json.dumps(candle_text, ensure_ascii=False)}

Indicators:
EMA9={local_result.get("ema9")}
EMA21={local_result.get("ema21")}
RSI={local_result.get("rsi")}
WilliamsR={local_result.get("williams_r")}
ADX={local_result.get("adx")}
DI+={local_result.get("plus_di")}
DI-={local_result.get("minus_di")}

Confirm only the existing direction.
Do not invent indicators.
Do not flip a strong local direction.

Return JSON only:

{{
  "direction": "UP or DOWN",
  "confidence": 0-100,
  "comment": "short reason"
}}
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        raw = response.text.strip()

        result = json.loads(raw)

        direction = result.get("direction")

        if direction not in ("UP", "DOWN"):
            return None

        # Never let Gemini flip the local direction.
        if direction != local_result["direction"]:
            return None

        confidence = result.get(
            "confidence",
            local_result["confidence"],
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = local_result["confidence"]

        confidence = max(
            50,
            min(89, confidence),
        )

        return {
            "direction": direction,
            "confidence": round(confidence),
            "comment": str(
                result.get(
                    "comment",
                    "",
                )
            )[:180],
        }

    except Exception as e:
        message = str(e)

        logger.warning(
            "Gemini confirmation failed: %s",
            message,
        )

        if (
            "429" in message
            or "RESOURCE_EXHAUSTED" in message
            or "quota" in message.lower()
        ):
            gemini_disabled_until = (
                time.time()
                + GEMINI_429_COOLDOWN_SECONDS
            )

            logger.warning(
                "Gemini disabled for 4 hours due to quota."
            )

        return None


# ============================================================
# MT4 DATA
# ============================================================

def store_mt4_data(payload):
    global mt4_data

    symbol = payload.get("symbol")

    if not symbol:
        return False

    timeframe = str(
        payload.get(
            "timeframe",
            "H1",
        )
    ).upper()

    candles = extract_candles(payload)

    if not candles:
        return False

    mt4_data[symbol] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "updated_at": time.time(),
    }

    logger.info(
        "MT4 data stored | %s | %s | candles=%s",
        symbol,
        timeframe,
        len(candles),
    )

    return True


# ============================================================
# BEST PAIR
# ============================================================

def choose_best_pair():
    candidates = []

    for symbol, data in mt4_data.items():

        timeframe = str(
            data.get(
                "timeframe",
                "H1",
            )
        ).upper()

        if timeframe != "H1":
            continue

        candles = data.get(
            "candles",
            [],
        )

        closed = remove_forming_candle(
            candles
        )

        if len(closed) < MIN_CLOSED_CANDLES:
            continue

        result = analyze_local(closed)

        if not result:
            continue

        if active_cycle["active"]:

            if (
                symbol
                == active_cycle.get("symbol")
            ):
                return {
                    "symbol": symbol,
                    "timeframe": "H1",
                    "analysis": result,
                }

            continue

        gap = abs(
            result["up_score"]
            - result["down_score"]
        )

        candidates.append(
            {
                "symbol": symbol,
                "timeframe": "H1",
                "analysis": result,
                "gap": gap,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["analysis"]["confidence"],
            max(
                x["analysis"]["up_score"],
                x["analysis"]["down_score"],
            ),
            x["gap"],
        ),
        reverse=True,
    )

    return candidates[0]


# ============================================================
# SIGNAL FORMAT
# ============================================================

def bold_digits(value):
    normal = "0123456789"
    bold = "𝟬𝟭𝟮𝟯𝟰𝟱𝟲𝟳𝟴𝟵"
    table = str.maketrans(normal, bold)
    return str(value).translate(table)


def format_signal(
    symbol,
    timeframe,
    trade_type,
    direction,
    confidence,
    up_score,
    down_score,
    entry_time,
    entry_price,
    cancellation_text,
    reason,
    recovery_level=0,
):
    direction_icon = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    recovery_text = "🔄 Recovery: 1" if recovery_level else "🔄 Recovery: 0"

    try:
        time_part = str(entry_time).split(" ")[-1]
    except Exception:
        time_part = str(entry_time)

    return f"""
🎓 ZinoProSignalAI
━━━━━━━━━━━━━━━━━━
📊 {symbol} | {timeframe}

{direction_icon}

🎯 Confidence: {confidence}%

📈 UP Score: {up_score}/18
📉 DOWN Score: {down_score}/18

⏱️ Entry after {ENTRY_DELAY_MINUTES} min

🕐 𝗘𝗡𝗧𝗥𝗬 𝗧𝗜𝗠𝗘
🕐 {bold_digits(time_part)}

💰 Entry Price: {entry_price:.6f}

⚠️ {cancellation_text}

🧠 {reason}

{recovery_text}
━━━━━━━━━━━━━━━━━━
""".strip()

# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_message(text):
    if telegram_application is None:
        return None

    try:
        message = await telegram_application.bot.send_message(
            chat_id=int(OWNER_ID),
            text=text,
        )
        return message.message_id

    except Exception as e:
        logger.exception(
            "Telegram send failed: %s",
            e,
        )
        return None


def send_signal_safely(text):
    global telegram_loop

    if telegram_loop is None:
        logger.warning(
            "Telegram loop unavailable."
        )
        return None

    try:
        future = asyncio.run_coroutine_threadsafe(
            send_message(text),
            telegram_loop,
        )

        return future.result(timeout=30)

    except Exception as e:
        logger.exception(
            "Safe Telegram send failed: %s",
            e,
        )
        return None


async def edit_signal_result(record, result):
    if telegram_application is None:
        return False

    message_id = record.get("message_id")
    if not message_id:
        return False

    icon = "✅" if result == "WIN" else "❌"
    label = "ربح" if result == "WIN" else "خسارة"

    base_text = record.get("signal_text", "")
    if not base_text:
        return False

    result_block = (
        "\n\n━━━━━━━━━━━━━━━━━━\n"
        f"{icon} النتيجة: {label}\n"
        "━━━━━━━━━━━━━━━━━━"
    )

    try:
        await telegram_application.bot.edit_message_text(
            chat_id=int(OWNER_ID),
            message_id=int(message_id),
            text=base_text + result_block,
        )
        return True

    except Exception as e:
        logger.exception(
            "Telegram result edit failed: %s",
            e,
        )
        return False


def edit_signal_result_safely(record, result):
    global telegram_loop

    if telegram_loop is None:
        return False

    try:
        future = asyncio.run_coroutine_threadsafe(
            edit_signal_result(record, result),
            telegram_loop,
        )
        return bool(future.result(timeout=30))
    except Exception as e:
        logger.exception(
            "Safe result edit failed: %s",
            e,
        )
        return False

# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(
    symbol,
    timeframe="H1",
):
    global last_signal_sent_at
    global next_signal_allowed_at
    global pending_recovery

    timeframe = str(timeframe).upper()

    if timeframe != "H1":
        return

    # Only one unresolved trade at a time.
    if active_cycle["active"]:
        logger.info(
            "Active trade waiting for /win or /loss | id=%s",
            current_trade_id,
        )
        return

    now_ts = time.time()

    # Signals are spaced exactly by at least 3 minutes.
    if now_ts < next_signal_allowed_at:
        return

    data = mt4_data.get(symbol)
    if not data:
        return

    candles = remove_forming_candle(
        data.get("candles", [])
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        logger.info(
            "Not enough closed candles | %s | %s",
            symbol,
            len(candles),
        )
        return

    local_result = analyze_local(candles)
    if not local_result:
        return

    gemini_result = gemini_confirm(
        symbol,
        timeframe,
        candles,
        local_result,
    )

    if gemini_result and gemini_result["direction"] == local_result["direction"]:
        local_result["confidence"] = min(
            local_result["confidence"],
            gemini_result["confidence"],
        )

        if gemini_result.get("comment"):
            local_result["reason"] += (
                " | " + gemini_result["comment"]
            )

    direction = local_result["direction"]
    confidence = local_result["confidence"]
    up_score = local_result["up_score"]
    down_score = local_result["down_score"]

    last_closed = candles[-1]
    entry_price = last_closed["close"]

    entry_dt = (
        now_algiers()
        + timedelta(minutes=ENTRY_DELAY_MINUTES)
    )
    entry_time = entry_dt.strftime("%Y-%m-%d %H:%M:%S")

    cancellation_level, cancellation_text = (
        calculate_cancellation_level(
            candles,
            direction,
            entry_price,
        )
    )

    # Recovery is manual: it is only marked on the next signal
    # after the owner records LOSS. No automatic multiplier is used.
    recovery_level = 1 if pending_recovery else 0
    trade_type = "RECOVERY" if recovery_level else "BASE"

    text = format_signal(
        symbol=symbol,
        timeframe=timeframe,
        trade_type=trade_type,
        direction=direction,
        confidence=confidence,
        up_score=up_score,
        down_score=down_score,
        entry_time=entry_time,
        entry_price=entry_price,
        cancellation_text=cancellation_text,
        reason=local_result["reason"],
        recovery_level=recovery_level,
    )

    message_id = send_signal_safely(text)

    if not message_id:
        return

    last_signal_sent_at = now_ts
    next_signal_allowed_at = now_ts + SIGNAL_COOLDOWN_SECONDS

    record = create_trade_record(
        symbol=symbol,
        timeframe=timeframe,
        trade_type=trade_type,
        direction=direction,
        confidence=confidence,
        up_score=up_score,
        down_score=down_score,
        entry_time=entry_time,
        entry_price=entry_price,
        cancellation_level=cancellation_level,
        reason=local_result["reason"],
        message_id=message_id,
        signal_text=text,
        recovery_level=recovery_level,
    )

    pending_recovery = False
    save_state()

    start_base_cycle(
        symbol,
        timeframe,
        direction,
    )

    # Keep the actual trade type for stats/history.
    active_cycle["trade_type"] = trade_type
    active_cycle["trade_number"] = 2 if recovery_level else 1
    active_cycle["recovery_used"] = bool(recovery_level)
    save_state()

    logger.info(
        "SIGNAL SENT | %s | %s | recovery=%s | next=%s",
        symbol,
        direction,
        recovery_level,
        datetime.fromtimestamp(
            next_signal_allowed_at,
            ALGIERS,
        ).strftime("%Y-%m-%d %H:%M:%S"),
    )

# ============================================================
# RECOVERY ANALYSIS
# ============================================================

def send_recovery_signal():
    """
    Called after BASE LOSS.

    Exactly one Recovery is allowed.
    """

    global last_signal_sent_at

    if not active_cycle["active"]:
        return False

    if active_cycle.get("recovery_used"):
        logger.warning(
            "Recovery already used. No second Recovery."
        )
        return False

    symbol = active_cycle.get(
        "symbol"
    )

    timeframe = active_cycle.get(
        "timeframe",
        "H1",
    )

    if not symbol:
        return False

    data = mt4_data.get(symbol)

    if not data:
        logger.warning(
            "No MT4 data for Recovery: %s",
            symbol,
        )
        return False

    candles = remove_forming_candle(
        data.get(
            "candles",
            [],
        )
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        return False

    result = analyze_local(
        candles
    )

    if not result:
        return False

    gemini_result = gemini_confirm(
        symbol,
        timeframe,
        candles,
        result,
    )

    if gemini_result:
        if (
            gemini_result["direction"]
            == result["direction"]
        ):
            result["confidence"] = min(
                result["confidence"],
                gemini_result["confidence"],
            )

            if gemini_result.get("comment"):
                result["reason"] += (
                    " | "
                    + gemini_result["comment"]
                )

    direction = result["direction"]

    entry_price = candles[-1]["close"]

    entry_dt = (
        now_algiers()
        + timedelta(
            minutes=ENTRY_DELAY_MINUTES
        )
    )

    entry_time = entry_dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cancellation_level, cancellation_text = (
        calculate_cancellation_level(
            candles,
            direction,
            entry_price,
        )
    )

    text = format_signal(
        symbol=symbol,
        timeframe=timeframe,
        trade_type="RECOVERY",
        direction=direction,
        confidence=result["confidence"],
        up_score=result["up_score"],
        down_score=result["down_score"],
        entry_time=entry_time,
        entry_price=entry_price,
        cancellation_text=cancellation_text,
        reason=result["reason"],
    )

    if not send_signal_safely(text):
        return False

    last_signal_sent_at = time.time()

    # --------------------------------------------------------
    # IMPORTANT:
    # Mark Recovery as used BEFORE creating the record.
    # This permanently blocks Recovery 2.
    # --------------------------------------------------------

    start_recovery_cycle()

    create_trade_record(
        symbol=symbol,
        timeframe=timeframe,
        trade_type="RECOVERY",
        direction=direction,
        confidence=result["confidence"],
        up_score=result["up_score"],
        down_score=result["down_score"],
        entry_time=entry_time,
        entry_price=entry_price,
        cancellation_level=cancellation_level,
        reason=result["reason"],
    )

    save_state()

    logger.info(
        "RECOVERY SIGNAL SENT | id=%s | %s | %s",
        current_trade_id,
        symbol,
        direction,
    )

    return True


# ============================================================
# /WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    if not active_cycle["active"]:
        await update.message.reply_text(
            "⚠️ لا توجد صفقة نشطة."
        )
        return

    trade_type = active_cycle.get("trade_type")

    updated = update_current_trade_result("WIN")

    if not updated:
        await update.message.reply_text(
            "⚠️ ما لقيتش صفقة PENDING لتسجيل WIN."
        )
        return

    stats_data["wins"] += 1

    if trade_type == "RECOVERY":
        stats_data["recovery_wins"] += 1
    else:
        stats_data["base_wins"] += 1

    save_state()
    reset_cycle()

    await update.message.reply_text(
        "✅ تم تسجيل الربح.\n"
        "🔄 Recovery القادم: 0\n"
        "⏱️ الإشارة التالية بعد انتهاء 3 دقائق من آخر إشارة."
    )

# ============================================================
# /LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global pending_recovery

    if not is_owner(update):
        return

    if not active_cycle["active"]:
        await update.message.reply_text(
            "⚠️ لا توجد صفقة نشطة."
        )
        return

    trade_type = active_cycle.get("trade_type")

    updated = update_current_trade_result("LOSS")

    if not updated:
        await update.message.reply_text(
            "⚠️ ما لقيتش صفقة PENDING لتسجيل LOSS."
        )
        return

    stats_data["losses"] += 1

    if trade_type == "RECOVERY":
        stats_data["recovery_losses"] += 1
        pending_recovery = False
        recovery_message = "🚫 Recovery انتهت. لا توجد Recovery ثانية."
    else:
        stats_data["base_losses"] += 1
        pending_recovery = True
        recovery_message = "🔄 الإشارة القادمة ستكون Recovery: 1 (يدوية)."

    save_state()
    reset_cycle()

    await update.message.reply_text(
        "❌ تم تسجيل الخسارة.\n"
        f"{recovery_message}\n"
        "⏱️ الإشارة التالية تبقى على دورة 3 دقائق، بدون مضاعفة تلقائية."
    )

# ============================================================
# /STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    stats = calculate_history_stats()

    total = stats["wins"] + stats["losses"]

    if total > 0:
        win_rate = (stats["wins"] / total) * 100
    else:
        win_rate = 0

    if next_signal_allowed_at > time.time():
        next_text = datetime.fromtimestamp(
            next_signal_allowed_at,
            ALGIERS,
        ).strftime("%H:%M:%S")
    else:
        next_text = "READY"

    message = (
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"✅ Wins: {stats['wins']}\n"
        f"❌ Losses: {stats['losses']}\n"
        f"⏳ Pending: {stats['pending']}\n"
        f"🎯 Win Rate: {win_rate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 BASE Wins: {stats['base_wins']}\n"
        f"🔴 BASE Losses: {stats['base_losses']}\n"
        f"🔁 Recovery Wins: {stats['recovery_wins']}\n"
        f"🔁 Recovery Losses: {stats['recovery_losses']}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🔄 Next Recovery: {'1' if pending_recovery else '0'}\n"
        f"⏱️ Next signal: {next_text}\n"
        "📌 Signal interval: 3 minutes"
    )

    await update.message.reply_text(message)

# ============================================================
# /HISTORY
# ============================================================

async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    if not trade_history:
        await update.message.reply_text(
            "📭 لا توجد صفقات مسجلة."
        )
        return

    records = trade_history[-HISTORY_DISPLAY_COUNT:]

    lines = [
        "📜 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for record in reversed(records):
        result = record.get("result", "PENDING")
        icon = "✅" if result == "WIN" else "❌" if result == "LOSS" else "⏳"
        recovery = record.get("recovery_level", 0)

        lines.append(
            f"{icon} {record.get('symbol')} | "
            f"{record.get('direction')} | "
            f"Recovery {recovery} | {result}"
        )

    await update.message.reply_text("\n".join(lines))

# ============================================================
# /RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global trade_history
    global stats_data
    global current_trade_id
    global next_signal_allowed_at
    global pending_recovery

    if not is_owner(update):
        return

    stats_data = {
        "wins": 0,
        "losses": 0,
        "base_wins": 0,
        "base_losses": 0,
        "recovery_wins": 0,
        "recovery_losses": 0,
    }

    reset_cycle()

    current_trade_id = None
    next_signal_allowed_at = 0.0
    pending_recovery = False

    trade_history = []

    save_history()
    save_state()

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات والسجل والدورة."
    )

# ============================================================
# /MT4STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    if not mt4_data:
        await update.message.reply_text(
            "📡 لا توجد بيانات MT4 حالياً."
        )
        return

    lines = [
        "📡 MT4 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for symbol, data in mt4_data.items():

        candles = data.get(
            "candles",
            [],
        )

        timeframe = data.get(
            "timeframe",
            "-",
        )

        lines.append(
            f"📊 {symbol} | {timeframe} | "
            f"{len(candles)} candles"
        )

    if gemini_available():
        gemini_status = "🟢 AVAILABLE"
    else:
        gemini_status = "🔴 COOLDOWN/OFF"

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🤖 Gemini: {gemini_status}"
    )

    await update.message.reply_text(
        "\n".join(lines)
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

    if active_cycle["active"]:
        await update.message.reply_text(
            "⏳ توجد دورة نشطة حالياً.\n"
            "استعمل /win أو /loss لتسجيل النتيجة."
        )
        return

    best = choose_best_pair()

    if not best:
        await update.message.reply_text(
            "⚠️ ما لقيتش زوج عنده بيانات MT4 كافية للتحليل."
        )
        return

    symbol = best["symbol"]

    await update.message.reply_text(
        f"🔎 تحليل {symbol} جاري..."
    )

    threading.Thread(
        target=auto_analyze_pair,
        args=(symbol, "H1"),
        daemon=True,
    ).start()


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ Bot running\n"
        "📡 MT4 connected mode\n"
        "🤖 Gemini analysis enabled\n"
        "⏱️ Signal interval: 3 minutes\n"
        "🔄 Recovery: manual only\n\n"
        "الأوامر:\n"
        "/analyze\n"
        "/stats\n"
        "/history\n"
        "/mt4status\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def background_analysis_loop():
    while True:
        try:
            if active_cycle["active"]:
                logger.info(
                    "Active trade waiting | type=%s | symbol=%s | trade_id=%s",
                    active_cycle.get("trade_type"),
                    active_cycle.get("symbol"),
                    current_trade_id,
                )
                time.sleep(15)
                continue

            # Do not send more than one signal every 3 minutes.
            if time.time() < next_signal_allowed_at:
                time.sleep(10)
                continue

            best = choose_best_pair()

            if best:
                symbol = best["symbol"]

                logger.info(
                    "AUTO ANALYSIS START | %s",
                    symbol,
                )

                auto_analyze_pair(
                    symbol,
                    "H1",
                )

        except Exception as e:
            logger.exception(
                "Background analysis error: %s",
                e,
            )

        time.sleep(10)

# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def _send(
        self,
        code,
        body,
        content_type="text/plain",
    ):
        body_bytes = body.encode(
            "utf-8"
        )

        self.send_response(code)

        self.send_header(
            "Content-Type",
            content_type
            + "; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body_bytes)),
        )

        self.end_headers()

        self.wfile.write(
            body_bytes
        )

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path in (
            "/",
            "/health",
            "/healthz",
        ):

            self._send(
                200,
                "ZinoProSignalAI is running",
            )

            return

        if path == "/mt4":

            self._send(
                200,
                "MT4 endpoint is running",
            )

            return

        self._send(
            404,
            "Not Found",
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        if parsed.path != "/mt4":

            self._send(
                404,
                "Not Found",
            )

            return

        provided_key = (
            self.headers.get(
                "X-MT4-API-Key",
                "",
            ).strip()
        )

        if not provided_key:

            provided_key = (
                self.headers.get(
                    "X-API-Key",
                    "",
                ).strip()
            )

        if (
            MT4_API_KEY
            and provided_key != MT4_API_KEY
        ):

            self._send(
                401,
                "Unauthorized",
            )

            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )

            raw = self.rfile.read(
                length
            )

            payload = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

            success = store_mt4_data(
                payload
            )

            if success:

                self._send(
                    200,
                    json.dumps(
                        {
                            "ok": True
                        }
                    ),
                    "application/json",
                )

                # --------------------------------------------
                # Trigger analysis when data arrives.
                # --------------------------------------------

                if not active_cycle["active"]:

                    symbol = payload.get(
                        "symbol"
                    )

                    if symbol:

                        threading.Thread(
                            target=auto_analyze_pair,
                            args=(
                                symbol,
                                "H1",
                            ),
                            daemon=True,
                        ).start()

            else:

                self._send(
                    400,
                    json.dumps(
                        {
                            "ok": False
                        }
                    ),
                    "application/json",
                )

        except Exception as e:

            logger.exception(
                "MT4 POST error: %s",
                e,
            )

            self._send(
                500,
                json.dumps(
                    {
                        "ok": False,
                        "error": str(e),
                    }
                ),
                "application/json",
            )

    def log_message(
        self,
        format,
        *args,
    ):
        return


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_application
    global telegram_loop

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:
        raise RuntimeError(
            "OWNER_ID is missing"
        )

    # --------------------------------------------------------
    # Load history and persistent state.
    # --------------------------------------------------------

    load_history()
    load_state()

    # --------------------------------------------------------
    # Recover pending BASE/RECOVERY trade.
    # --------------------------------------------------------

    rebuild_state_from_history()

    save_state()

    # --------------------------------------------------------
    # HTTP server for Render.
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # --------------------------------------------------------
    # Background analysis.
    # --------------------------------------------------------

    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
    )

    analysis_thread.start()

    # --------------------------------------------------------
    # Telegram.
    # --------------------------------------------------------

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_application = application

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "analyze",
            analyze_command,
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
            "history",
            history_command,
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

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "Time zone: Africa/Algiers"
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Entry delay: %s minutes",
        ENTRY_DELAY_MINUTES,
    )

    logger.info(
        "History records: %s",
        len(trade_history),
    )

    logger.info(
        "Active cycle: %s",
        active_cycle["active"],
    )

    # --------------------------------------------------------
    # Keep the Telegram event loop reference.
    # --------------------------------------------------------

    telegram_loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        telegram_loop
    )

    application.run_polling(
        close_loop=False
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
