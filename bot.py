import os
import json
import logging
import threading
import asyncio
import time
import math
from datetime import datetime
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

# Automatic analysis every minute
AUTO_ANALYSIS_INTERVAL_MINUTES = 1

# Entry is always 2 minutes after signal
ENTRY_DELAY_MINUTES = 2

# Minimum time between sent trades
SIGNAL_COOLDOWN_SECONDS = 120

# Minimum closed candles required
MIN_CLOSED_CANDLES = 40


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
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
    except Exception as e:
        logger.exception("Gemini initialization failed: %s", e)
else:
    logger.warning("GEMINI_API_KEY is missing")


# ============================================================
# GLOBAL STATE
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

last_auto_analysis = {}

telegram_application = None
telegram_loop = None

last_signal_sent_at = 0.0

signal_send_lock = threading.Lock()
cycle_lock = threading.Lock()

active_cycle = {
    "active": False,
    "symbol": None,
    "timeframe": None,
    "trade_type": None,
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


# ============================================================
# BASIC HELPERS
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
    tf = normalize_timeframe(timeframe)

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

    return values.get(tf, 60)


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

    if o is None or h is None or l is None or c is None:
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

    return result


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if not values or len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    current = sum(values[:period]) / period

    for price in values[period:]:
        current = (
            (price - current) * multiplier
        ) + current

    return current


def ema_series(values, period):
    if not values or len(values) < period:
        return []

    multiplier = 2 / (period + 1)

    first = sum(values[:period]) / period

    result = [None] * (period - 1)
    result.append(first)

    current = first

    for price in values[period:]:
        current = (
            (price - current) * multiplier
        ) + current

        result.append(current)

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]

        if diff >= 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]

        gain = max(diff, 0)
        loss = max(-diff, 0)

        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def williams_r(candles, period=14):
    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(x["high"] for x in recent)
    lowest = min(x["low"] for x in recent)

    if highest == lowest:
        return -50.0

    close = recent[-1]["close"]

    return ((highest - close) / (highest - lowest)) * -100


def true_ranges(candles):
    if len(candles) < 2:
        return []

    result = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        result.append(tr)

    return result


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def adx_di(candles, period=14):
    if len(candles) < period + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        up_move = current["high"] - previous["high"]
        down_move = previous["low"] - current["low"]

        plus = up_move if up_move > down_move and up_move > 0 else 0
        minus = down_move if down_move > up_move and down_move > 0 else 0

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(trs) < period:
        return None, None, None

    tr_avg = sum(trs[-period:]) / period
    plus_avg = sum(plus_dm[-period:]) / period
    minus_avg = sum(minus_dm[-period:]) / period

    if tr_avg == 0:
        return 0.0, 0.0, 0.0

    plus_di = 100 * plus_avg / tr_avg
    minus_di = 100 * minus_avg / tr_avg

    denominator = plus_di + minus_di

    if denominator == 0:
        dx = 0
    else:
        dx = 100 * abs(plus_di - minus_di) / denominator

    return dx, plus_di, minus_di


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(candles):
    if len(candles) < 8:
        return "UNKNOWN"

    recent = candles[-8:]

    first_high = max(x["high"] for x in recent[:4])
    second_high = max(x["high"] for x in recent[4:])

    first_low = min(x["low"] for x in recent[:4])
    second_low = min(x["low"] for x in recent[4:])

    if second_high > first_high and second_low > first_low:
        return "BULLISH"

    if second_high < first_high and second_low < first_low:
        return "BEARISH"

    return "RANGE"


def breakout_state(candles):
    if len(candles) < 10:
        return "NONE"

    recent = candles[-9:-1]

    previous_high = max(x["high"] for x in recent)
    previous_low = min(x["low"] for x in recent)

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
    closes = [x["close"] for x in candles]

    current = candles[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    current_rsi = rsi(closes, 14)
    current_wr = williams_r(candles, 14)

    current_atr = atr(candles, 10)

    adx, plus_di, minus_di = adx_di(candles, 14)

    keltner_mid = ema(closes, 20)

    keltner_atr = atr(candles, 10)

    keltner_upper = None
    keltner_lower = None

    if keltner_mid is not None and keltner_atr is not None:
        keltner_upper = keltner_mid + (keltner_atr * 5)
        keltner_lower = keltner_mid - (keltner_atr * 5)

    structure = market_structure(candles)

    breakout = breakout_state(candles)

    recent8 = candles[-8:]

    recent_low = min(x["low"] for x in recent8)
    recent_high = max(x["high"] for x in recent8)

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
        "structure": structure,
        "breakout": breakout,
        "recent_low": recent_low,
        "recent_high": recent_high,
    }


# ============================================================
# PRE-SCORE
# ============================================================

def technical_candidate_score(candles):
    if len(candles) < MIN_CLOSED_CANDLES:
        return -999

    snapshot = build_technical_snapshot(candles)

    score = 0

    price = snapshot["price"]
    ema9 = snapshot["ema9"]
    ema21 = snapshot["ema21"]

    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            score += 3
        elif ema9 < ema21:
            score += 3

    structure = snapshot["structure"]

    if structure in ("BULLISH", "BEARISH"):
        score += 3

    breakout = snapshot["breakout"]

    if breakout != "NONE":
        score += 3

    rsi_value = snapshot["rsi14"]

    if rsi_value is not None:
        if 40 <= rsi_value <= 60:
            score += 1
        elif rsi_value < 30 or rsi_value > 70:
            score += 1

    adx_value = snapshot["adx14"]

    if adx_value is not None and adx_value >= 20:
        score += 2

    return score


def choose_best_pair():
    with mt4_lock:
        candidates = []

        for symbol, timeframes in mt4_data.items():

            if not isinstance(timeframes, dict):
                continue

            candles = timeframes.get("H1")

            if not candles:
                continue

            closed = candles[:-1]

            if len(closed) < MIN_CLOSED_CANDLES:
                continue

            score = technical_candidate_score(closed)

            candidates.append({
                "symbol": symbol,
                "timeframe": "H1",
                "score": score,
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates[0]


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(symbol, timeframe, candles):
    snapshot = build_technical_snapshot(candles)

    recent = candles[-25:]

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles_count": len(candles),
        "technical_snapshot": snapshot,
        "recent_candles": recent,
    }

    return f"""
You are the analysis engine for ZinoProSignalAI.

Analyze the supplied market data only.

IMPORTANT RULES:

1. Do NOT invent any data.
2. Do NOT use external prices.
3. Do NOT assume indicators that are not supplied.
4. Direction must be exactly UP or DOWN.
5. Never return WAIT.
6. Never return NO SIGNAL.
7. Never return NEUTRAL.
8. The system requires a directional signal.
9. Confidence must reflect actual confluence.
10. Do not use 90%+ confidence unless the supplied evidence is exceptionally strong.
11. Price Action has highest priority.
12. Then Market Structure.
13. Then Breakout/Retest.
14. Then Liquidity.
15. Then Momentum.
16. Then Candle behavior.
17. Then EMA 9/21.
18. Then RSI.
19. Then Williams %R.
20. Then Keltner.
21. Then ADX/DI.

SCORING:

The total must be exactly 18 points.

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

Return JSON only.

Required format:

{{
  "signal": true,
  "direction": "UP",
  "confidence": 75,
  "up_score": 13,
  "down_score": 5,
  "reason": "Short factual reason based only on supplied data.",
  "cancellation_reason": "Cancel if the next closed candle invalidates the directional structure."
}}

The sum of up_score and down_score must be exactly 18.

MARKET DATA:

{json.dumps(payload, ensure_ascii=False)}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(symbol, timeframe, candles):
    if gemini_client is None:
        raise RuntimeError("Gemini client is not initialized")

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.15,
            response_mime_type="application/json",
        ),
    )

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned empty response")

    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    return result


# ============================================================
# FORCE VALID DIRECTIONAL SIGNAL
# ============================================================

def ensure_directional_signal(result, candles):
    if not isinstance(result, dict):
        result = {}

    direction = str(
        result.get("direction", "")
    ).upper().strip()

    up_score = int(
        safe_float(result.get("up_score"), 0) or 0
    )

    down_score = int(
        safe_float(result.get("down_score"), 0) or 0
    )

    # Keep total exactly 18
    up_score = max(0, min(18, up_score))
    down_score = max(0, min(18, down_score))

    total = up_score + down_score

    if total != 18:
        if total == 0:
            up_score = 9
            down_score = 9

        elif total < 18:
            if up_score >= down_score:
                up_score += 18 - total
            else:
                down_score += 18 - total

        else:
            difference = total - 18

            if up_score >= down_score:
                up_score = max(0, up_score - difference)
            else:
                down_score = max(0, down_score - difference)

    if direction not in ("UP", "DOWN"):

        if up_score > down_score:
            direction = "UP"

        elif down_score > up_score:
            direction = "DOWN"

        else:
            structure = market_structure(candles)

            if structure == "BULLISH":
                direction = "UP"

            elif structure == "BEARISH":
                direction = "DOWN"

            else:
                closes = [x["close"] for x in candles]

                e9 = ema(closes, 9)
                e21 = ema(closes, 21)

                if e9 is not None and e21 is not None:
                    direction = "UP" if e9 >= e21 else "DOWN"
                else:
                    direction = (
                        "UP"
                        if candles[-1]["close"] >= candles[-1]["open"]
                        else "DOWN"
                    )

    confidence = safe_float(
        result.get("confidence"),
        50
    )

    confidence = max(
        1,
        min(99, int(confidence))
    )

    reason = str(
        result.get("reason")
        or "Price action and technical confluence indicate the selected direction."
    ).strip()

    cancellation_reason = str(
        result.get("cancellation_reason")
        or "Cancel if the next closed candle invalidates the current structure."
    ).strip()

    return {
        "signal": True,
        "direction": direction,
        "confidence": confidence,
        "up_score": up_score,
        "down_score": down_score,
        "reason": reason,
        "cancellation_reason": cancellation_reason,
    }


# ============================================================
# ENTRY / CANCELLATION
# ============================================================

def get_entry_price(candles):
    return candles[-1]["close"]


def get_cancellation_level(candles, direction):
    recent = candles[-8:]

    if direction == "UP":
        return min(x["low"] for x in recent)

    return max(x["high"] for x in recent)


def format_price(price):
    if price is None:
        return "N/A"

    if abs(price) >= 100:
        return f"{price:.5f}"

    if abs(price) >= 1:
        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# SIGNAL FORMAT
# ============================================================

def format_signal(
    symbol,
    timeframe,
    analysis,
    candles,
    trade_type,
):
    direction = analysis["direction"]

    now = datetime.now(ALGIERS)

    entry_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = entry_time.replace(
        minute=entry_time.minute + ENTRY_DELAY_MINUTES
    )

    entry_price = get_entry_price(candles)

    cancellation_level = get_cancellation_level(
        candles,
        direction
    )

    if direction == "UP":
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(cancellation_level)}"
        )
    else:
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(cancellation_level)}"
        )

    if trade_type == "RECOVERY":
        trade_label = "🔁 RECOVERY 1/1"
    else:
        trade_label = "🎯 BASE TRADE"

    direction_icon = "🟢 UP" if direction == "UP" else "🔴 DOWN"

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n"
        f"{trade_label}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{direction_icon}\n"
        f"🎯 Confidence: {analysis['confidence']}%\n"
        f"📈 UP Score: {analysis['up_score']}/18\n"
        f"📉 DOWN Score: {analysis['down_score']}/18\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"⏳ Entry after: {ENTRY_DELAY_MINUTES} min\n"
        f"⏰ Entry Time: {entry_time.strftime('%H:%M:%S')} "
        "(Algiers)\n"
        f"💰 Entry Price: {format_price(entry_price)}\n"
        f"⚠️ {cancel_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧠 {analysis['reason']}\n"
    )

    return message


# ============================================================
# COOLDOWN
# ============================================================

def signal_cooldown_active():
    elapsed = time.time() - last_signal_sent_at

    if elapsed < SIGNAL_COOLDOWN_SECONDS:
        remaining = int(
            SIGNAL_COOLDOWN_SECONDS - elapsed
        )

        return True, remaining

    return False, 0


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_telegram_message(text_message):
    global telegram_application

    if telegram_application is None:
        logger.error("Telegram application is not ready")
        return False

    oid = owner_id_int()

    if oid is None:
        logger.error("OWNER_ID is invalid")
        return False

    try:
        await telegram_application.bot.send_message(
            chat_id=oid,
            text=text_message,
        )

        return True

    except Exception as e:
        logger.exception(
            "Telegram send failed: %s",
            e
        )

        return False


def send_signal_safely(message):
    global last_signal_sent_at

    with signal_send_lock:

        active, remaining = signal_cooldown_active()

        if active:
            logger.info(
                "Signal cooldown active: %s seconds",
                remaining
            )

            return False

        if telegram_loop is None:
            logger.error("Telegram event loop is not ready")
            return False

        future = asyncio.run_coroutine_threadsafe(
            send_telegram_message(message),
            telegram_loop
        )

        try:
            success = future.result(timeout=30)

        except Exception as e:
            logger.exception(
                "Telegram future failed: %s",
                e
            )

            return False

        if success:
            last_signal_sent_at = time.time()
            return True

        return False


# ============================================================
# CYCLE MANAGEMENT
# ============================================================

def get_active_cycle():
    with cycle_lock:
        return dict(active_cycle)


def start_base_cycle(symbol, timeframe):
    with cycle_lock:
        active_cycle["active"] = True
        active_cycle["symbol"] = symbol
        active_cycle["timeframe"] = timeframe
        active_cycle["trade_type"] = "BASE"
        active_cycle["recovery_used"] = False
        active_cycle["last_trade_time"] = time.time()
        active_cycle["trade_number"] = 1


def start_recovery_cycle():
    with cycle_lock:
        active_cycle["active"] = True
        active_cycle["trade_type"] = "RECOVERY"
        active_cycle["recovery_used"] = True
        active_cycle["trade_number"] = 2
        active_cycle["last_trade_time"] = time.time()


def reset_cycle():
    with cycle_lock:
        active_cycle["active"] = False
        active_cycle["symbol"] = None
        active_cycle["timeframe"] = None
        active_cycle["trade_type"] = None
        active_cycle["recovery_used"] = False
        active_cycle["last_trade_time"] = 0.0
        active_cycle["trade_number"] = 0


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(symbol, timeframe):
    timeframe = normalize_timeframe(timeframe)

    # System is configured for H1
    if timeframe != "H1":
        return

    cycle = get_active_cycle()

    # If there is an active cycle, only analyze that pair.
    if cycle["active"]:

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

    # Global send cooldown
    active, remaining = signal_cooldown_active()

    if active:
        logger.info(
            "Auto analysis skipped due to 2-minute cooldown: %ss",
            remaining
        )

        return

    with mt4_lock:
        timeframes = mt4_data.get(symbol)

        if not timeframes:
            return

        candles = timeframes.get(timeframe)

        if not candles:
            return

        candles = list(candles)

    if len(candles) < MIN_CLOSED_CANDLES + 1:
        logger.info(
            "%s %s: not enough candles (%s)",
            symbol,
            timeframe,
            len(candles)
        )

        return

    # Exclude current forming candle
    closed = candles[:-1]

    if len(closed) < MIN_CLOSED_CANDLES:
        return

    try:
        logger.info(
            "Analyzing %s %s with Gemini",
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

        cycle = get_active_cycle()

        if cycle["active"]:
            trade_type = cycle["trade_type"]

        else:
            trade_type = "BASE"

        message = format_signal(
            symbol,
            timeframe,
            analysis,
            closed,
            trade_type
        )

        sent = send_signal_safely(message)

        if not sent:
            logger.info(
                "Signal was not sent for %s %s",
                symbol,
                timeframe
            )

            return

        if not cycle["active"]:
            start_base_cycle(
                symbol,
                timeframe
            )

        else:
            with cycle_lock:
                active_cycle["last_trade_time"] = time.time()

        logger.info(
            "Signal sent: %s %s %s",
            symbol,
            timeframe,
            analysis["direction"]
        )

    except Exception as e:
        logger.exception(
            "Auto analysis failed for %s %s: %s",
            symbol,
            timeframe,
            e
        )


# ============================================================
# BACKGROUND ANALYSIS
# ============================================================

def background_analysis_loop():
    logger.info(
        "Background analysis loop started"
    )

    while True:

        try:
            with mt4_lock:
                symbols = list(mt4_data.keys())

            for symbol in symbols:
                auto_analyze_pair(
                    symbol,
                    "H1"
                )

        except Exception as e:
            logger.exception(
                "Background analysis loop error: %s",
                e
            )

        time.sleep(
            AUTO_ANALYSIS_INTERVAL_MINUTES * 60
        )


# ============================================================
# MT4 DATA HANDLING
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

    return symbol, timeframe, normalized


# ============================================================
# HTTP SERVER
# ============================================================

class MT4Handler(BaseHTTPRequestHandler):

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

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

        if parsed.path == "/mt4":

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

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        parsed = urlparse(self.path)

        if parsed.path != "/mt4":

            self.send_response(404)
            self.end_headers()

            return

        try:

            # API key
            received_key = (
                self.headers.get("X-MT4-API-Key")
                or self.headers.get("X-API-Key")
                or ""
            ).strip()

            if MT4_API_KEY:

                if received_key != MT4_API_KEY:

                    logger.warning(
                        "MT4 request rejected: invalid API key"
                    )

                    body = b"Invalid API key"

                    self.send_response(401)

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
                    "text/plain; charset=utf-8"
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

            symbol, timeframe, candles = store_mt4_data(
                payload
            )

            # Start analysis in another thread.
            analysis_thread = threading.Thread(
                target=auto_analyze_pair,
                args=(symbol, timeframe),
                daemon=True,
            )

            analysis_thread.start()

            response = {
                "ok": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
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

            logger.exception(
                "Invalid JSON received from MT4"
            )

            body = b"Invalid JSON"

            self.send_response(400)

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
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(body)

    def log_message(self, format_string, *args):
        logger.info(
            "HTTP %s - %s",
            self.address_string(),
            format_string % args
        )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
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
        "✅ Bot is running\n"
        "📡 MT4 → Render → Gemini → Telegram\n\n"
        "Commands:\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt4status\n"
        "/analyze"
    )


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

    if total > 0:
        winrate = (
            stats_data["wins"]
            / total
        ) * 100
    else:
        winrate = 0

    cycle = get_active_cycle()

    if cycle["active"]:
        cycle_text = (
            f"{cycle['trade_type']} | "
            f"{cycle['symbol']} | "
            f"{cycle['timeframe']}"
        )
    else:
        cycle_text = "No active cycle"

    text = (
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Total trades: {total}\n"
        f"🟢 Wins: {stats_data['wins']}\n"
        f"🔴 Losses: {stats_data['losses']}\n"
        f"📈 Win rate: {winrate:.1f}%\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Base wins: {stats_data['base_wins']}\n"
        f"Base losses: {stats_data['base_losses']}\n"
        f"Recovery wins: {stats_data['recovery_wins']}\n"
        f"Recovery losses: {stats_data['recovery_losses']}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"Cycle: {cycle_text}"
    )

    await update.message.reply_text(text)


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

    stats_data["wins"] += 1

    if cycle["trade_type"] == "BASE":

        stats_data["base_wins"] += 1

        result_text = (
            "🟢 BASE WIN\n\n"
            "الدورة انتهت بنجاح.\n"
            "🔎 سيتم البحث عن زوج جديد."
        )

    else:

        stats_data["recovery_wins"] += 1

        result_text = (
            "🟢 RECOVERY WIN\n\n"
            "تم تعويض الصفقة الأساسية.\n"
            "🔎 سيتم البحث عن زوج جديد."
        )

    reset_cycle()

    await update.message.reply_text(
        result_text
    )


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

    stats_data["losses"] += 1

    if cycle["trade_type"] == "BASE":

        stats_data["base_losses"] += 1

        start_recovery_cycle()

        await update.message.reply_text(
            "🔴 BASE LOSS\n\n"
            "🔁 Recovery 1/1 مسموح.\n"
            "لا توجد مضاعفة ثانية بعد Recovery."
        )

        return

    stats_data["recovery_losses"] += 1

    reset_cycle()

    await update.message.reply_text(
        "🔴 RECOVERY LOSS\n\n"
        "⛔ انتهت الدورة.\n"
        "لا توجد Recovery ثانية.\n"
        "🔎 سيتم البحث عن زوج جديد."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    stats_data["wins"] = 0
    stats_data["losses"] = 0
    stats_data["base_wins"] = 0
    stats_data["base_losses"] = 0
    stats_data["recovery_wins"] = 0
    stats_data["recovery_losses"] = 0

    reset_cycle()

    global last_signal_sent_at
    last_signal_sent_at = 0.0

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات والدورة."
    )


async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with mt4_lock:
        data_copy = {
            symbol: dict(timeframes)
            for symbol, timeframes in mt4_data.items()
        }

    if not data_copy:
        await update.message.reply_text(
            "📡 لا توجد بيانات MT4 مستلمة حتى الآن."
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


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    cycle = get_active_cycle()

    if cycle["active"]:

        await update.message.reply_text(
            "⚠️ توجد دورة نشطة بالفعل:\n"
            f"{cycle['symbol']} {cycle['timeframe']}\n"
            f"{cycle['trade_type']}"
        )

        return

    best = choose_best_pair()

    if not best:

        await update.message.reply_text(
            "⏳ لا توجد بيانات H1 كافية من MT4."
        )

        return

    await update.message.reply_text(
        f"🔎 أفضل مرشح حاليًا من البيانات المتوفرة:\n"
        f"📊 {best['symbol']} | H1\n"
        f"Score: {best['score']}\n\n"
        "سأبدأ التحليل."
    )

    thread = threading.Thread(
        target=auto_analyze_pair,
        args=(
            best["symbol"],
            "H1",
        ),
        daemon=True,
    )

    thread.start()


# ============================================================
# TEXT / PHOTO HANDLERS
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
        "📡 النظام الحالي يعتمد على بيانات MT4.\n"
        "أرسل البيانات من MT4 وسيتم تحليلها تلقائيًا."
    )


async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "📸 تم استلام الصورة، لكن الوضع الحالي يعتمد على "
        "بيانات MT4 المباشرة للتحليل التلقائي."
    )


# ============================================================
# TELEGRAM ERROR HANDLER
# ============================================================

async def telegram_error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):
    logger.exception(
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
    global telegram_loop

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
        "Analysis timeframe: H1"
    )

    logger.info(
        "Entry delay: %s minutes",
        ENTRY_DELAY_MINUTES
    )

    logger.info(
        "Signal cooldown: %s seconds",
        SIGNAL_COOLDOWN_SECONDS
    )

    logger.info(
        "Recovery limit: 1"
    )

    logger.info(
        "========================================"
    )

    # HTTP server
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # Background analysis
    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
    )

    analysis_thread.start()

    # Telegram
    application = build_telegram_application()

    telegram_loop = asyncio.new_event_loop()

    asyncio.set_event_loop(
        telegram_loop
    )

    logger.info(
        "Telegram bot starting"
    )

    application.run_polling(
        close_loop=False
    )


if __name__ == "__main__":
    main()
