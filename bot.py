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
OWNER_ID = int(os.getenv("OWNER_ID", "0").strip() or "0")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

# MT5 API key
MT5_API_KEY = os.getenv(
    "MT5_API_KEY",
    os.getenv("MT4_API_KEY", "")
).strip()

PORT = int(os.getenv("PORT", "10000"))

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10

SIGNAL_COOLDOWN_SECONDS = 120
SETUP_REPEAT_BLOCK_SECONDS = 360

RECOVERY_LIMIT = 1

AUTO_ANALYSIS_INTERVAL_SECONDS = 60

ALGIERS_TZ = ZoneInfo("Africa/Algiers")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBAL STATE
# ============================================================

mt5_data = {}
mt5_lock = threading.Lock()

analysis_lock = threading.Lock()

last_signal_sent_at = 0.0
last_setup_at = 0.0
last_setup_key = ""

active_cycle = None

stats = {
    "wins": 0,
    "losses": 0,
    "total": 0,
}

history = []

telegram_application = None


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
    except Exception as exc:
        logger.exception(
            "Gemini initialization failed: %s",
            exc
        )
else:
    logger.warning("GEMINI_API_KEY is not configured")


# ============================================================
# BASIC HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def timeframe_minutes(timeframe):
    text = str(timeframe).upper().strip()

    if text.startswith("M"):
        try:
            return max(1, int(text[1:]))
        except Exception:
            return 1

    if text.startswith("H"):
        try:
            return max(1, int(text[1:])) * 60
        except Exception:
            return 60

    return 1


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


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candles(candles):
    if not isinstance(candles, list):
        return []

    normalized = []

    for candle in candles:
        if not isinstance(candle, dict):
            continue

        try:
            item = {
                "time": safe_int(candle.get("time")),
                "open": safe_float(candle.get("open")),
                "high": safe_float(candle.get("high")),
                "low": safe_float(candle.get("low")),
                "close": safe_float(candle.get("close")),
                "tick_volume": safe_float(
                    candle.get("tick_volume", 0)
                ),
                "real_volume": safe_float(
                    candle.get("real_volume", 0)
                ),
                "spread": safe_float(
                    candle.get("spread", 0)
                ),
            }

            if item["time"] > 0:
                normalized.append(item)

        except Exception:
            continue

    normalized.sort(
        key=lambda x: x["time"]
    )

    return normalized


# ============================================================
# MT5 DATA STORE
# ============================================================

def store_mt5_data(payload):
    global mt5_data

    symbol = str(
        payload.get("symbol", "")
    ).strip()

    timeframe = str(
        payload.get(
            "timeframe",
            ANALYSIS_TIMEFRAME
        )
    ).strip().upper()

    candles = normalize_candles(
        payload.get("candles", [])
    )

    if not symbol:
        return False, "missing symbol"

    if not candles:
        return False, "missing candles"

    with mt5_lock:
        mt5_data[symbol] = {
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
                payload.get("digits"),
                5
            ),
            "closed_candle_time": safe_int(
                payload.get("closed_candle_time")
            ),
            "batch_id": str(
                payload.get("batch_id", "")
            ),
            "batch_complete": bool(
                payload.get("batch_complete", True)
            ),
            "received_at": time.time(),
        }

    return True, "stored"


def get_mt5_snapshot(symbol=None):
    with mt5_lock:
        if symbol:
            data = mt5_data.get(symbol)

            if not data:
                return None

            return dict(data)

        return {
            key: dict(value)
            for key, value in mt5_data.items()
        }


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if not values:
        return []

    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)

    seed = sum(
        values[:period]
    ) / period

    result[period - 1] = seed

    multiplier = 2.0 / (period + 1.0)

    previous = seed

    for i in range(period, len(values)):
        previous = (
            (values[i] - previous)
            * multiplier
            + previous
        )

        result[i] = previous

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return [None] * len(values)

    result = [None] * len(values)

    gains = []
    losses = []

    for i in range(1, period + 1):
        change = values[i] - values[i - 1]

        if change >= 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (
            100.0 / (1.0 + rs)
        )

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        avg_gain = (
            (avg_gain * (period - 1))
            + gain
        ) / period

        avg_loss = (
            (avg_loss * (period - 1))
            + loss
        ) / period

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (
                100.0 / (1.0 + rs)
            )

    return result


def williams_r(candles, period=14):
    result = [None] * len(candles)

    if len(candles) < period:
        return result

    for i in range(period - 1, len(candles)):
        window = candles[
            i - period + 1:
            i + 1
        ]

        highest = max(
            c["high"] for c in window
        )

        lowest = min(
            c["low"] for c in window
        )

        close = candles[i]["close"]

        denominator = highest - lowest

        if denominator == 0:
            result[i] = -50.0
        else:
            result[i] = (
                (highest - close)
                / denominator
            ) * -100.0

    return result


def true_ranges(candles):
    result = []

    previous_close = None

    for candle in candles:
        high = candle["high"]
        low = candle["low"]

        if previous_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )

        result.append(tr)
        previous_close = candle["close"]

    return result


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return None

    return sum(
        trs[-period:]
    ) / period


def adx_di(candles, period=14):
    if len(candles) < period + 1:
        return None, None, None

    trs = []
    plus_moves = []
    minus_moves = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        high = current["high"]
        low = current["low"]

        prev_high = previous["high"]
        prev_low = previous["low"]
        prev_close = previous["close"]

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm = up_move
        else:
            plus_dm = 0.0

        if down_move > up_move and down_move > 0:
            minus_dm = down_move
        else:
            minus_dm = 0.0

        trs.append(tr)
        plus_moves.append(plus_dm)
        minus_moves.append(minus_dm)

    if len(trs) < period:
        return None, None, None

    tr_avg = sum(
        trs[-period:]
    ) / period

    plus_avg = sum(
        plus_moves[-period:]
    ) / period

    minus_avg = sum(
        minus_moves[-period:]
    ) / period

    if tr_avg == 0:
        return 0.0, 0.0, 0.0

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
        plus_di + minus_di
    )

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

    last8 = candles[-8:]

    first4 = last8[:4]
    last4 = last8[4:]

    first_high = max(
        c["high"] for c in first4
    )

    last_high = max(
        c["high"] for c in last4
    )

    first_low = min(
        c["low"] for c in first4
    )

    last_low = min(
        c["low"] for c in last4
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

    return "NEUTRAL"


def breakout_state(candles):
    if len(candles) < 9:
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
        return "BULLISH"

    if last["close"] < previous_low:
        return "BEARISH"

    return "NONE"


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_technical_snapshot(candles):
    closes = [
        c["close"]
        for c in candles
    ]

    ema9_values = ema(
        closes,
        9
    )

    ema21_values = ema(
        closes,
        21
    )

    ema20_values = ema(
        closes,
        20
    )

    rsi_values = rsi(
        closes,
        14
    )

    wr_values = williams_r(
        candles,
        14
    )

    adx_value, plus_di, minus_di = adx_di(
        candles,
        14
    )

    atr10 = atr(
        candles,
        10
    )

    last_close = closes[-1]

    ema9_value = ema9_values[-1]
    ema21_value = ema21_values[-1]
    ema20_value = ema20_values[-1]

    rsi_value = rsi_values[-1]
    wr_value = wr_values[-1]

    if atr10 is not None and ema20_value is not None:
        keltner_upper = (
            ema20_value
            + atr10 * 5.0
        )

        keltner_lower = (
            ema20_value
            - atr10 * 5.0
        )
    else:
        keltner_upper = None
        keltner_lower = None

    return {
        "price": last_close,

        "ema9": ema9_value,
        "ema21": ema21_value,

        "rsi14": rsi_value,

        "williams_r14": wr_value,

        "atr10": atr10,

        "adx14": adx_value,
        "plus_di14": plus_di,
        "minus_di14": minus_di,

        "keltner_mid": ema20_value,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,

        "market_structure": market_structure(
            candles
        ),

        "breakout": breakout_state(
            candles
        ),

        "recent8_low": min(
            c["low"]
            for c in candles[-8:]
        ),

        "recent8_high": max(
            c["high"]
            for c in candles[-8:]
        ),
    }


# ============================================================
# DIRECTIONAL PRE-SCORE
# ============================================================

def directional_pre_score(snapshot, candles):
    up = 0
    down = 0

    ema9_value = snapshot.get("ema9")
    ema21_value = snapshot.get("ema21")

    price = snapshot.get("price")

    structure = snapshot.get(
        "market_structure"
    )

    breakout = snapshot.get(
        "breakout"
    )

    adx_value = snapshot.get("adx14")
    plus_di = snapshot.get("plus_di14")
    minus_di = snapshot.get("minus_di14")

    rsi_value = snapshot.get("rsi14")

    if (
        ema9_value is not None
        and ema21_value is not None
    ):
        if ema9_value > ema21_value:
            up += 3
        elif ema9_value < ema21_value:
            down += 3

    if (
        price is not None
        and ema9_value is not None
    ):
        if price > ema9_value:
            up += 1
        elif price < ema9_value:
            down += 1

    if (
        price is not None
        and ema21_value is not None
    ):
        if price > ema21_value:
            up += 1
        elif price < ema21_value:
            down += 1

    if structure == "BULLISH":
        up += 3

    elif structure == "BEARISH":
        down += 3

    if breakout == "BULLISH":
        up += 3

    elif breakout == "BEARISH":
        down += 3

    if (
        adx_value is not None
        and plus_di is not None
        and minus_di is not None
        and adx_value >= 20
    ):
        if plus_di > minus_di:
            up += 2

        elif minus_di > plus_di:
            down += 2

    if rsi_value is not None:
        if 50 < rsi_value < 70:
            up += 1

        elif 30 < rsi_value < 50:
            down += 1

    if candles:
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
        if (
            ema9_value is not None
            and ema21_value is not None
            and ema9_value >= ema21_value
        ):
            direction = "UP"
        else:
            direction = "DOWN"

    return {
        "up": up,
        "down": down,
        "direction": direction,
    }


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
    recent_candles = candles[-30:]

    data = {
        "symbol": symbol,
        "timeframe": timeframe,
        "technical_snapshot": snapshot,
        "pre_score": pre_score,
        "recent_closed_candles": recent_candles,
    }

    return f"""
You are the technical-analysis engine for ZinoProSignalAI.

Analyze ONLY the supplied closed-candle data.

IMPORTANT RULES:

1. Never invent prices, candles, indicators, news or market data.
2. Do not use external prices.
3. Use only the supplied data.
4. The latest candle in the supplied dataset is CLOSED.
5. The current forming candle must NOT be used.
6. The final direction MUST be UP or DOWN.
7. Never return WAIT.
8. Never return NO SIGNAL.
9. Never return NEUTRAL.
10. Confidence is an estimate, NOT a guarantee.
11. Do not artificially claim certainty.
12. Avoid 90%+ confidence unless there is exceptionally strong confluence.

ANALYSIS PRIORITY:

1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle behavior
7. EMA 9 / EMA 21
8. RSI 14
9. Williams %R 14
10. Keltner Channel
11. ADX / DI

INDICATORS:

EMA 9
EMA 21

RSI 14
RSI overbought = 70
RSI oversold = 30

Williams %R 14
overbought = -20
oversold = -80

ADX 14
DI length = 14

Keltner:
EMA 20
ATR 10
Multiplier 5

SCORING:

The final score MUST be exactly 18 points total.

Categories:

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

The direction must agree with the strongest evidence.

Do not blindly follow the preliminary score.
Use it as supporting information only.

DATA:

{json.dumps(data, ensure_ascii=False, separators=(",", ":"))}

Return ONLY valid JSON.

Required JSON structure:

{{
  "signal": "BASE TRADE",
  "direction": "UP",
  "confidence": 75,
  "up_score": 12,
  "down_score": 6,
  "reason": "Short technical reason.",
  "cancellation_reason": "Cancel if the candle closes below the specified cancellation level."
}}

The sum of up_score and down_score MUST equal 18.

direction MUST be either UP or DOWN.

signal MUST be BASE TRADE unless this is a recovery analysis.
"""


# ============================================================
# GEMINI RESPONSE NORMALIZATION
# ============================================================

def ensure_directional_signal(data, fallback_direction):
    if not isinstance(data, dict):
        data = {}

    direction = str(
        data.get(
            "direction",
            fallback_direction
        )
    ).upper().strip()

    if direction not in ("UP", "DOWN"):
        direction = fallback_direction

    try:
        up_score = int(
            round(
                float(
                    data.get(
                        "up_score",
                        0
                    )
                )
            )
        )
    except Exception:
        up_score = 0

    try:
        down_score = int(
            round(
                float(
                    data.get(
                        "down_score",
                        0
                    )
                )
            )
        )
    except Exception:
        down_score = 0

    up_score = max(
        0,
        min(18, up_score)
    )

    down_score = max(
        0,
        min(18, down_score)
    )

    total = up_score + down_score

    if total < 18:
        remaining = 18 - total

        if direction == "UP":
            up_score += remaining
        else:
            down_score += remaining

    elif total > 18:
        excess = total - 18

        if direction == "UP":
            up_score = max(
                0,
                up_score - excess
            )
        else:
            down_score = max(
                0,
                down_score - excess
            )

    if up_score == down_score:
        if direction == "UP":
            up_score = min(
                18,
                up_score + 1
            )
            down_score = 18 - up_score
        else:
            down_score = min(
                18,
                down_score + 1
            )
            up_score = 18 - down_score

    direction = (
        "UP"
        if up_score > down_score
        else "DOWN"
    )

    try:
        confidence = int(
            round(
                float(
                    data.get(
                        "confidence",
                        50
                    )
                )
            )
        )
    except Exception:
        confidence = 50

    confidence = max(
        1,
        min(99, confidence)
    )

    gap = abs(
        up_score - down_score
    )

    if confidence >= 90 and gap < 8:
        confidence = 89

    reason = str(
        data.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:
        reason = (
            "Signal based on the strongest "
            "available technical confluence."
        )

    cancellation_reason = str(
        data.get(
            "cancellation_reason",
            ""
        )
    ).strip()

    if not cancellation_reason:
        if direction == "UP":
            cancellation_reason = (
                "Cancel if the candle closes below "
                "the cancellation level."
            )
        else:
            cancellation_reason = (
                "Cancel if the candle closes above "
                "the cancellation level."
            )

    signal = str(
        data.get(
            "signal",
            "BASE TRADE"
        )
    ).strip()

    if not signal:
        signal = "BASE TRADE"

    return {
        "signal": signal,
        "direction": direction,
        "confidence": confidence,
        "up_score": up_score,
        "down_score": down_score,
        "reason": reason,
        "cancellation_reason": cancellation_reason,
    }


def parse_gemini_json(text):
    if not text:
        return {}

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

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            return {}

    return {}


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score,
    recovery=False,
):
    fallback_direction = pre_score["direction"]

    if not gemini_client:
        return ensure_directional_signal(
            {},
            fallback_direction
        )

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        snapshot,
        pre_score,
    )

    if recovery:
        prompt += """

This is a RECOVERY analysis.

The previous BASE TRADE lost.

Do not automatically reverse direction.
Re-evaluate the market from the supplied
closed candles.

Only choose the direction supported by the
current technical structure.
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

        text = getattr(
            response,
            "text",
            ""
        )

        parsed = parse_gemini_json(text)

        return ensure_directional_signal(
            parsed,
            fallback_direction
        )

    except Exception as exc:
        logger.exception(
            "Gemini analysis failed: %s",
            exc
        )

        return ensure_directional_signal(
            {},
            fallback_direction
        )


# ============================================================
# ENTRY / CANCELLATION
# ============================================================

def calculate_entry(
    candles,
    direction,
    timeframe,
):
    if not candles:
        return None

    last_price = candles[-1]["close"]

    delay = timeframe_minutes(
        timeframe
    )

    now = now_algiers()

    base_time = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        base_time
        + timedelta(minutes=delay)
    )

    recent = candles[-8:]

    if direction == "UP":
        cancellation = min(
            c["low"]
            for c in recent
        )

        cancellation_text = (
            "إلغاء إذا أغلقت الشمعة تحت "
            f"{cancellation:.5f}"
        )

    else:
        cancellation = max(
            c["high"]
            for c in recent
        )

        cancellation_text = (
            "إلغاء إذا أغلقت الشمعة فوق "
            f"{cancellation:.5f}"
        )

    return {
        "entry_price": last_price,
        "entry_time": entry_time,
        "delay_minutes": delay,
        "cancellation": cancellation,
        "cancellation_text": cancellation_text,
    }


# ============================================================
# TELEGRAM FORMAT
# ============================================================

def format_signal(
    symbol,
    timeframe,
    result,
    entry,
    trade_type,
    trade_number,
):
    direction = result["direction"]

    if direction == "UP":
        emoji = "🟢"
    else:
        emoji = "🔴"

    entry_time_text = (
        entry["entry_time"]
        .strftime("%Y-%m-%d %H:%M")
    )

    price = entry["entry_price"]

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"

        f"🔁 {trade_type}"
        f" {trade_number}/"
        f"{RECOVERY_LIMIT + 1}\n\n"

        f"{emoji} {direction}\n\n"

        f"🔥 Confidence: "
        f"{result['confidence']}%\n"

        f"🟢 UP Score: "
        f"{result['up_score']}/18\n"

        f"🔴 DOWN Score: "
        f"{result['down_score']}/18\n\n"

        f"⏱️ Entry after: "
        f"{entry['delay_minutes']} minute"
        f"{'s' if entry['delay_minutes'] != 1 else ''}\n"

        f"🕐 ENTRY TIME: "
        f"{entry_time_text}\n"

        f"💰 ENTRY PRICE: "
        f"{price:.5f}\n"

        f"⚠️ {entry['cancellation_text']}\n\n"

        f"📝 {result['reason']}"
    )


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_telegram_message(text):
    global telegram_application

    if not telegram_application:
        return False

    if OWNER_ID <= 0:
        return False

    try:
        await telegram_application.bot.send_message(
            chat_id=OWNER_ID,
            text=text,
        )

        return True

    except Exception as exc:
        logger.exception(
            "Telegram send failed: %s",
            exc
        )

        return False


def send_telegram_message_sync(text):
    try:
        loop = asyncio.new_event_loop()

        try:
            asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                send_telegram_message(text)
            )

        finally:
            loop.close()

    except Exception as exc:
        logger.exception(
            "Telegram sync send failed: %s",
            exc
        )

        return False


# ============================================================
# SETUP CONTROL
# ============================================================

def setup_fingerprint(
    symbol,
    timeframe,
    candles,
    direction,
):
    if not candles:
        return ""

    last_time = candles[-1]["time"]

    return (
        f"{symbol}|"
        f"{timeframe}|"
        f"{last_time}|"
        f"{direction}"
    )


def setup_blocked(key):
    global last_setup_key
    global last_setup_at

    if not key:
        return False

    if key != last_setup_key:
        return False

    if (
        time.time() - last_setup_at
        < SETUP_REPEAT_BLOCK_SECONDS
    ):
        return True

    return False


def signal_cooldown_active():
    if last_signal_sent_at <= 0:
        return False

    return (
        time.time()
        - last_signal_sent_at
        < SIGNAL_COOLDOWN_SECONDS
    )


# ============================================================
# PAIR SELECTION
# ============================================================

def choose_best_pair():
    snapshots = get_mt5_snapshot()

    if not snapshots:
        return None

    best = None

    for symbol, data in snapshots.items():

        timeframe = str(
            data.get(
                "timeframe",
                ANALYSIS_TIMEFRAME
            )
        ).upper()

        if timeframe != ANALYSIS_TIMEFRAME:
            continue

        candles = normalize_candles(
            data.get("candles", [])
        )

        if len(candles) < MIN_CLOSED_CANDLES + 1:
            continue

        closed = candles[:-1]

        if len(closed) < MIN_CLOSED_CANDLES:
            continue

        snapshot = build_technical_snapshot(
            closed
        )

        pre_score = directional_pre_score(
            snapshot,
            closed
        )

        strength = max(
            pre_score["up"],
            pre_score["down"]
        )

        gap = abs(
            pre_score["up"]
            - pre_score["down"]
        )

        candidate = {
            "symbol": symbol,
            "data": data,
            "candles": closed,
            "snapshot": snapshot,
            "pre_score": pre_score,
            "strength": strength,
            "gap": gap,
        }

        if best is None:
            best = candidate
            continue

        if candidate["strength"] > best["strength"]:
            best = candidate

        elif (
            candidate["strength"]
            == best["strength"]
            and candidate["gap"]
            > best["gap"]
        ):
            best = candidate

    return best


# ============================================================
# ACTIVE CYCLE
# ============================================================

def active_cycle_symbol():
    if not active_cycle:
        return None

    return active_cycle.get(
        "symbol"
    )


def create_base_cycle(
    symbol,
    direction,
    entry,
    result,
):
    global active_cycle

    active_cycle = {
        "symbol": symbol,
        "timeframe": ANALYSIS_TIMEFRAME,
        "direction": direction,

        "trade_type": "BASE TRADE",
        "trade_number": 1,

        "recovery_used": False,

        "entry_price": entry["entry_price"],
        "entry_time": entry["entry_time"].isoformat(),

        "cancellation": entry["cancellation"],

        "confidence": result["confidence"],

        "status": "PENDING",

        "created_at": now_algiers().isoformat(),
    }


# ============================================================
# HISTORY
# ============================================================

def add_history_item(
    symbol,
    timeframe,
    trade_type,
    direction,
    result,
    entry,
    status,
):
    item = {
        "symbol": symbol,
        "timeframe": timeframe,

        "trade_type": trade_type,
        "direction": direction,

        "confidence": result.get(
            "confidence",
            0
        ),

        "up_score": result.get(
            "up_score",
            0
        ),

        "down_score": result.get(
            "down_score",
            0
        ),

        "entry_price": entry.get(
            "entry_price"
        ),

        "entry_time": (
            entry["entry_time"].isoformat()
            if hasattr(
                entry.get("entry_time"),
                "isoformat"
            )
            else str(
                entry.get("entry_time", "")
            )
        ),

        "status": status,

        "timestamp": now_algiers().isoformat(),
    }

    history.append(item)

    if len(history) > 200:
        del history[:-200]


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(symbol=None):
    global last_signal_sent_at
    global last_setup_key
    global last_setup_at
    global active_cycle

    if not analysis_lock.acquire(
        blocking=False
    ):
        return False

    try:
        if signal_cooldown_active():
            return False

        candidate = None

        if symbol:
            data = get_mt5_snapshot(
                symbol
            )

            if data:
                candles = normalize_candles(
                    data.get("candles", [])
                )

                if len(candles) >= (
                    MIN_CLOSED_CANDLES + 1
                ):
                    closed = candles[:-1]

                    snapshot = build_technical_snapshot(
                        closed
                    )

                    pre_score = directional_pre_score(
                        snapshot,
                        closed
                    )

                    candidate = {
                        "symbol": symbol,
                        "data": data,
                        "candles": closed,
                        "snapshot": snapshot,
                        "pre_score": pre_score,
                    }

        if candidate is None:
            candidate = choose_best_pair()

        if not candidate:
            return False

        selected_symbol = candidate["symbol"]

        candles = candidate["candles"]

        if len(candles) < MIN_CLOSED_CANDLES:
            return False

        if (
            active_cycle
            and active_cycle.get("status")
            == "PENDING"
        ):
            if (
                active_cycle.get("symbol")
                != selected_symbol
            ):
                return False

        recovery = bool(
            active_cycle
            and active_cycle.get(
                "status"
            ) == "PENDING"
            and active_cycle.get(
                "recovery_used"
            )
        )

        result = analyze_with_gemini(
            selected_symbol,
            ANALYSIS_TIMEFRAME,
            candles,
            candidate["snapshot"],
            candidate["pre_score"],
            recovery=recovery,
        )

        direction = result["direction"]

        setup_key = setup_fingerprint(
            selected_symbol,
            ANALYSIS_TIMEFRAME,
            candles,
            direction,
        )

        if setup_blocked(setup_key):
            return False

        entry = calculate_entry(
            candles,
            direction,
            ANALYSIS_TIMEFRAME,
        )

        if not entry:
            return False

        if (
            active_cycle
            and active_cycle.get(
                "status"
            ) == "PENDING"
            and active_cycle.get(
                "recovery_used"
            )
        ):
            trade_type = "RECOVERY"
            trade_number = 2

            result["signal"] = (
                "RECOVERY 1/1"
            )

        else:
            trade_type = "BASE TRADE"
            trade_number = 1

            result["signal"] = (
                "BASE TRADE"
            )

        text = format_signal(
            selected_symbol,
            ANALYSIS_TIMEFRAME,
            result,
            entry,
            trade_type,
            trade_number,
        )

        sent = send_telegram_message_sync(
            text
        )

        if not sent:
            return False

        last_signal_sent_at = time.time()

        last_setup_key = setup_key
        last_setup_at = time.time()

        if trade_type == "BASE TRADE":

            create_base_cycle(
                selected_symbol,
                direction,
                entry,
                result,
            )

        else:

            if active_cycle:
                active_cycle[
                    "trade_number"
                ] = 2

                active_cycle[
                    "direction"
                ] = direction

                active_cycle[
                    "entry_price"
                ] = entry[
                    "entry_price"
                ]

                active_cycle[
                    "entry_time"
                ] = entry[
                    "entry_time"
                ].isoformat()

                active_cycle[
                    "cancellation"
                ] = entry[
                    "cancellation"
                ]

                active_cycle[
                    "confidence"
                ] = result[
                    "confidence"
                ]

        add_history_item(
            selected_symbol,
            ANALYSIS_TIMEFRAME,
            trade_type,
            direction,
            result,
            entry,
            "PENDING",
        )

        logger.info(
            "Signal sent | %s | %s | %s",
            selected_symbol,
            trade_type,
            direction,
        )

        return True

    except Exception as exc:
        logger.exception(
            "auto_analyze_pair failed: %s",
            exc
        )

        return False

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
            auto_analyze_pair()

        except Exception as exc:
            logger.exception(
                "Background loop error: %s",
                exc
            )

        time.sleep(
            AUTO_ANALYSIS_INTERVAL_SECONDS
        )


# ============================================================
# STATS
# ============================================================

def stats_text():
    total = stats["total"]
    wins = stats["wins"]
    losses = stats["losses"]

    if total > 0:
        winrate = (
            wins / total
        ) * 100
    else:
        winrate = 0.0

    return (
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📈 Total: {total}\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"🎯 Win Rate: {winrate:.1f}%"
    )


# ============================================================
# HISTORY TEXT
# ============================================================

def history_text():
    if not history:
        return (
            "📚 ZinoProSignalAI HISTORY\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "No history yet."
        )

    items = history[
        -HISTORY_DISPLAY_COUNT:
    ]

    lines = [
        "📚 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in reversed(items):

        status = item.get(
            "status",
            "PENDING"
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
            "M1"
        )

        trade_type = item.get(
            "trade_type",
            "BASE TRADE"
        )

        direction = item.get(
            "direction",
            "?"
        )

        confidence = item.get(
            "confidence",
            0
        )

        up_score = item.get(
            "up_score",
            0
        )

        down_score = item.get(
            "down_score",
            0
        )

        price = item.get(
            "entry_price",
            0
        )

        entry_time = item.get(
            "entry_time",
            ""
        )

        lines.append(
            f"{status_icon} {symbol} {timeframe}"
        )

        lines.append(
            f"   {trade_type} | {direction}"
        )

        lines.append(
            f"   🎯 {confidence}% | "
            f"📈 {up_score}/18 "
            f"📉 {down_score}/18"
        )

        lines.append(
            f"   💰 {safe_float(price):.5f}"
        )

        lines.append(
            f"   ⏰ {entry_time}"
        )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# TELEGRAM AUTH
# ============================================================

def owner_only(update):
    if not update or not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID
    )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "MT5 M1 → Render → Gemini → Telegram\n\n"
        "📌 Commands:\n"
        "/analyze\n"
        "/mt5status\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    await update.message.reply_text(
        "🔎 Starting MT5 analysis..."
    )

    def worker():
        auto_analyze_pair()

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    await update.message.reply_text(
        stats_text()
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    await update.message.reply_text(
        history_text()
    )


async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    snapshots = get_mt5_snapshot()

    if not snapshots:
        await update.message.reply_text(
            "📡 MT5 STATUS\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "🔴 No MT5 data received."
        )
        return

    lines = [
        "📡 ZinoProSignalAI MT5 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for symbol, data in snapshots.items():

        candles = data.get(
            "candles",
            []
        )

        timeframe = data.get(
            "timeframe",
            "?"
        )

        batch_id = data.get(
            "batch_id",
            ""
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

        if age < 120:
            status = "🟢"
        else:
            status = "🟡"

        lines.append(
            f"{status} {symbol} | {timeframe}"
        )

        lines.append(
            f"   Candles: {len(candles)}"
        )

        lines.append(
            f"   Age: {age:.0f}s"
        )

        lines.append(
            f"   Batch: {batch_id}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global active_cycle

    if not owner_only(update):
        return

    if not active_cycle:
        await update.message.reply_text(
            "⚠️ No active trade."
        )
        return

    stats["wins"] += 1
    stats["total"] += 1

    symbol = active_cycle.get(
        "symbol",
        "?"
    )

    trade_type = active_cycle.get(
        "trade_type",
        "BASE TRADE"
    )

    direction = active_cycle.get(
        "direction",
        "?"
    )

    for item in reversed(history):
        if (
            item.get("symbol")
            == symbol
            and item.get("status")
            == "PENDING"
        ):
            item["status"] = "WIN"
            break

    await update.message.reply_text(
        "🟢 WIN\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol}\n"
        f"🔁 {trade_type}\n"
        f"📈 {direction}\n\n"
        f"{stats_text()}"
    )

    active_cycle = None


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global active_cycle
    global last_signal_sent_at

    if not owner_only(update):
        return

    if not active_cycle:
        await update.message.reply_text(
            "⚠️ No active trade."
        )
        return

    symbol = active_cycle.get(
        "symbol",
        "?"
    )

    trade_number = safe_int(
        active_cycle.get(
            "trade_number",
            1
        ),
        1
    )

    # --------------------------------------------------------
    # BASE LOSS -> ONE RECOVERY
    # --------------------------------------------------------

    if (
        trade_number == 1
        and not active_cycle.get(
            "recovery_used",
            False
        )
    ):

        stats["losses"] += 1
        stats["total"] += 1

        for item in reversed(history):
            if (
                item.get("symbol")
                == symbol
                and item.get("status")
                == "PENDING"
            ):
                item["status"] = "LOSS"
                break

        active_cycle[
            "recovery_used"
        ] = True

        active_cycle[
            "trade_number"
        ] = 2

        # Allow recovery signal immediately.
        last_signal_sent_at = 0

        await update.message.reply_text(
            "🔴 BASE LOSS\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {symbol}\n"
            "🔁 Recovery 1/1 enabled.\n\n"
            "🔎 Re-analyzing market..."
        )

        threading.Thread(
            target=lambda: auto_analyze_pair(
                symbol
            ),
            daemon=True,
        ).start()

        return

    # --------------------------------------------------------
    # RECOVERY LOSS -> END CYCLE
    # --------------------------------------------------------

    stats["losses"] += 1
    stats["total"] += 1

    for item in reversed(history):
        if (
            item.get("symbol")
            == symbol
            and item.get("status")
            == "PENDING"
        ):
            item["status"] = "LOSS"
            break

    active_cycle = None

    await update.message.reply_text(
        "🔴 LOSS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol}\n"
        "❌ Recovery 1/1 also lost.\n"
        "🛑 Cycle closed.\n\n"
        f"{stats_text()}"
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global active_cycle
    global history
    global last_signal_sent_at
    global last_setup_at
    global last_setup_key

    if not owner_only(update):
        return

    stats["wins"] = 0
    stats["losses"] = 0
    stats["total"] = 0

    history.clear()

    active_cycle = None

    last_signal_sent_at = 0
    last_setup_at = 0
    last_setup_key = ""

    await update.message.reply_text(
        "♻️ ZinoProSignalAI reset completed."
    )


# ============================================================
# COMPATIBILITY COMMAND
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await mt5status_command(
        update,
        context
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    text = (
        update.message.text
        if update.message
        else ""
    )

    if not text:
        return

    await update.message.reply_text(
        "📡 ZinoProSignalAI يعمل عبر MT5.\n\n"
        "أرسل /analyze لبدء التحليل."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not owner_only(update):
        return

    await update.message.reply_text(
        "📷 Image analysis is disabled.\n\n"
        "ZinoProSignalAI الآن يعتمد على "
        "بيانات MT5 المباشرة."
    )


# ============================================================
# HTTP SERVER
# ============================================================

class MT5Handler(BaseHTTPRequestHandler):

    def _send_json(
        self,
        status,
        payload,
    ):
        body = json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8")

        self.send_response(status)

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

    def _authorized(self):
        key = (
            self.headers.get(
                "X-MT5-API-Key",
                ""
            ).strip()
        )

        if not key:
            key = (
                self.headers.get(
                    "X-API-Key",
                    ""
                ).strip()
            )

        if not key:
            key = (
                self.headers.get(
                    "X-MT4-API-Key",
                    ""
                ).strip()
            )

        if not MT5_API_KEY:
            return False

        return key == MT5_API_KEY

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
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": (
                        "ZinoProSignalAI"
                    ),
                    "source": "MT5",
                },
            )
            return

        if path == "/mt5status":
            snapshots = get_mt5_snapshot()

            self._send_json(
                200,
                {
                    "status": "ok",
                    "symbols": list(
                        snapshots.keys()
                    ),
                    "count": len(
                        snapshots
                    ),
                },
            )
            return

        if path == "/mt4status":
            snapshots = get_mt5_snapshot()

            self._send_json(
                200,
                {
                    "status": "ok",
                    "source": "MT5",
                    "symbols": list(
                        snapshots.keys()
                    ),
                    "count": len(
                        snapshots
                    ),
                },
            )
            return

        self._send_json(
            404,
            {
                "error": "not found"
            },
        )

    def do_POST(self):
        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path not in (
            "/mt5",
            "/api/mt5",

            # compatibility
            "/mt4",
            "/api/mt4",
        ):
            self._send_json(
                404,
                {
                    "error": "not found"
                },
            )
            return

        if not self._authorized():
            logger.warning(
                "Unauthorized MT5 request"
            )

            self._send_json(
                401,
                {
                    "error": "unauthorized"
                },
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
                self._send_json(
                    400,
                    {
                        "error":
                        "empty body"
                    },
                )
                return

            raw = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw.decode("utf-8")
            )

            if not isinstance(
                payload,
                dict
            ):
                self._send_json(
                    400,
                    {
                        "error":
                        "JSON object required"
                    },
                )
                return

            ok, message = store_mt5_data(
                payload
            )

            if not ok:
                self._send_json(
                    400,
                    {
                        "error": message
                    },
                )
                return

            symbol = str(
                payload.get(
                    "symbol",
                    ""
                )
            )

            candles = normalize_candles(
                payload.get(
                    "candles",
                    []
                )
            )

            logger.info(
                "MT5 data received | "
                "%s | candles=%s",
                symbol,
                len(candles),
            )

            self._send_json(
                200,
                {
                    "status": "ok",
                    "stored": True,
                    "symbol": symbol,
                    "candles": len(
                        candles
                    ),
                },
            )

        except json.JSONDecodeError:
            self._send_json(
                400,
                {
                    "error":
                    "invalid JSON"
                },
            )

        except Exception as exc:
            logger.exception(
                "MT5 POST error: %s",
                exc
            )

            self._send_json(
                500,
                {
                    "error":
                    "server error"
                },
            )

    def log_message(
        self,
        format,
        *args,
    ):
        logger.info(
            "HTTP | " + format,
            *args
        )


# ============================================================
# HTTP SERVER
# ============================================================

def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        MT5Handler,
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# TELEGRAM APPLICATION
# ============================================================

async def post_init(
    application,
):
    logger.info(
        "Telegram application initialized"
    )


def build_telegram_application():
    global telegram_application

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

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

    telegram_application = application

    return application


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info(
        "================================================"
    )

    logger.info(
        "Starting ZinoProSignalAI"
    )

    logger.info(
        "MT5 M1 -> Render -> Gemini -> Telegram"
    )

    logger.info(
        "================================================"
    )

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is not configured"
        )

    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is not configured"
        )

    if not MT5_API_KEY:
        logger.warning(
            "MT5_API_KEY is not configured"
        )

    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="HTTPServer",
    )

    http_thread.start()

    # --------------------------------------------------------
    # Background analysis
    # --------------------------------------------------------

    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
        name="AnalysisLoop",
    )

    analysis_thread.start()

    # --------------------------------------------------------
    # Telegram
    # --------------------------------------------------------

    application = (
        build_telegram_application()
    )

    logger.info(
        "Telegram bot starting..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
    )


if __name__ == "__main__":
    main()
