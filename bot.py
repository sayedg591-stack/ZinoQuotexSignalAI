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
)

from google import genai
from google.genai import types


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0") or 0)

MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()

# Backward compatibility only
if not MT5_API_KEY:
    MT5_API_KEY = os.getenv("MT4_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

PORT = int(os.getenv("PORT", "10000"))

TIMEZONE = ZoneInfo("Africa/Algiers")

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40

SIGNAL_COOLDOWN_SECONDS = 120
SETUP_REPEAT_BLOCK_SECONDS = 360

RECOVERY_LIMIT = 1

AUTO_ANALYSIS_INTERVAL_SECONDS = 60

HISTORY_FILE = "signal_history.json"
HISTORY_DISPLAY_COUNT = 10


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
        logger.info(
            "Gemini client initialized | model=%s",
            GEMINI_MODEL
        )
    except Exception as e:
        logger.exception(
            "Gemini initialization failed: %s",
            e
        )


# ============================================================
# GLOBAL MT5 DATA
# ============================================================

mt5_data = {}

data_lock = threading.Lock()

analysis_lock = threading.Lock()

last_analysis_time = {}

last_setup_time = {}

cycle_state = {}

signal_history = []


# ============================================================
# HISTORY
# ============================================================

def load_history():
    global signal_history

    try:
        if not os.path.exists(HISTORY_FILE):
            signal_history = []
            return

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

        if isinstance(data, list):
            signal_history = data
        else:
            signal_history = []

        logger.info(
            "History loaded: %d signals",
            len(signal_history)
        )

    except Exception as e:
        logger.exception(
            "History load failed: %s",
            e
        )
        signal_history = []


def save_history():
    try:
        tmp = HISTORY_FILE + ".tmp"

        with open(
            tmp,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                signal_history,
                f,
                ensure_ascii=False,
                indent=2
            )

        os.replace(
            tmp,
            HISTORY_FILE
        )

    except Exception as e:
        logger.exception(
            "History save failed: %s",
            e
        )


# ============================================================
# AUTH
# ============================================================

def is_owner(update: Update) -> bool:
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


async def owner_only(update: Update) -> bool:
    if not is_owner(update):
        if update.effective_message:
            await update.effective_message.reply_text(
                "⛔ Unauthorized."
            )
        return False

    return True


# ============================================================
# TIME
# ============================================================

def now_algiers() -> datetime:
    return datetime.now(TIMEZONE)


def format_time(dt=None):
    if dt is None:
        dt = now_algiers()

    return dt.astimezone(TIMEZONE).strftime(
        "%Y-%m-%d %H:%M"
    )


def timeframe_minutes(timeframe: str) -> int:
    tf = str(timeframe or "M1").upper().strip()

    aliases = {
        "M1": 1,
        "1M": 1,
        "M2": 2,
        "2M": 2,
        "M3": 3,
        "3M": 3,
        "M5": 5,
        "5M": 5,
        "M15": 15,
        "15M": 15,
        "M30": 30,
        "30M": 30,
        "H1": 60,
        "1H": 60,
        "H4": 240,
        "4H": 240,
    }

    return aliases.get(tf, 1)


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candles(candles):
    if not isinstance(candles, list):
        return []

    clean = []

    for c in candles:
        if not isinstance(c, dict):
            continue

        try:
            item = {
                "time": int(c.get("time", 0)),
                "open": float(c.get("open", 0)),
                "high": float(c.get("high", 0)),
                "low": float(c.get("low", 0)),
                "close": float(c.get("close", 0)),
                "tick_volume": float(
                    c.get("tick_volume", 0)
                ),
                "real_volume": float(
                    c.get("real_volume", 0)
                ),
                "spread": float(
                    c.get("spread", 0)
                ),
            }

            if item["time"] > 0:
                clean.append(item)

        except Exception:
            continue

    clean.sort(
        key=lambda x: x["time"]
    )

    return clean


def get_closed_candles(candles):
    candles = normalize_candles(candles)

    if len(candles) < 2:
        return []

    # MT5 sends the current forming candle as the last candle.
    return candles[:-1]


# ============================================================
# BASIC INDICATORS
# ============================================================

def ema(values, period):
    values = [float(x) for x in values]

    if len(values) < period:
        return None

    seed = sum(values[:period]) / period

    multiplier = 2.0 / (period + 1.0)

    result = seed

    for price in values[period:]:
        result = (
            (price - result) * multiplier
        ) + result

    return result


def rsi(values, period=14):
    values = [float(x) for x in values]

    if len(values) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, period + 1):
        diff = values[i] - values[i - 1]

        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(diff))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period + 1, len(values)):
        diff = values[i] - values[i - 1]

        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)

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
        return 0.0

    return (
        (highest - close)
        /
        (highest - lowest)
    ) * -100.0


def atr(candles, period=10):
    if len(candles) < period + 1:
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

    value = sum(trs[:period]) / period

    for tr in trs[period:]:
        value = (
            (value * (period - 1))
            + tr
        ) / period

    return value


# ============================================================
# ADX / DI
# Compatibility implementation from old MT4 logic
# ============================================================

def adx_di(candles, period=14):
    if len(candles) < period + 1:
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

        if (
            up_move > down_move
            and up_move > 0
        ):
            plus = up_move
        else:
            plus = 0.0

        if (
            down_move > up_move
            and down_move > 0
        ):
            minus = down_move
        else:
            minus = 0.0

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
        return None, None, None

    atr_value = sum(
        trs[-period:]
    ) / period

    if atr_value == 0:
        return 0.0, 0.0, 0.0

    plus_di = (
        sum(plus_dm[-period:])
        / period
    ) / atr_value * 100.0

    minus_di = (
        sum(minus_dm[-period:])
        / period
    ) / atr_value * 100.0

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
        return "RANGE"

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

    return "RANGE"


# ============================================================
# BREAKOUT
# ============================================================

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
        return "BULLISH_BREAKOUT"

    if last["close"] < previous_low:
        return "BEARISH_BREAKOUT"

    return "NONE"


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_snapshot(candles):
    closes = [
        c["close"]
        for c in candles
    ]

    last = candles[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

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

    adx, plus_di, minus_di = adx_di(
        candles,
        14
    )

    keltner_mid = ema(
        closes,
        20
    )

    if (
        keltner_mid is not None
        and atr10 is not None
    ):
        keltner_upper = (
            keltner_mid
            + atr10 * 5.0
        )

        keltner_lower = (
            keltner_mid
            - atr10 * 5.0
        )
    else:
        keltner_upper = None
        keltner_lower = None

    recent8 = candles[-8:]

    recent8_low = min(
        c["low"]
        for c in recent8
    )

    recent8_high = max(
        c["high"]
        for c in recent8
    )

    return {
        "price": last["close"],

        "open": last["open"],
        "high": last["high"],
        "low": last["low"],
        "close": last["close"],

        "ema9": ema9,
        "ema21": ema21,

        "rsi14": rsi14,

        "williams_r14": wr14,

        "atr10": atr10,

        "adx14": adx,
        "plus_di14": plus_di,
        "minus_di14": minus_di,

        "keltner_mid20": keltner_mid,
        "keltner_upper": keltner_upper,
        "keltner_lower": keltner_lower,

        "structure": market_structure(
            candles
        ),

        "breakout": breakout_state(
            candles
        ),

        "recent8_low": recent8_low,
        "recent8_high": recent8_high,
    }


# ============================================================
# OLD MT4 PRE-SCORE
# ============================================================

def directional_pre_score(
    snapshot,
    candles
):
    up = 0
    down = 0

    ema9 = snapshot.get("ema9")
    ema21 = snapshot.get("ema21")
    price = snapshot.get("price")

    structure = snapshot.get(
        "structure"
    )

    breakout = snapshot.get(
        "breakout"
    )

    adx = snapshot.get(
        "adx14"
    )

    plus_di = snapshot.get(
        "plus_di14"
    )

    minus_di = snapshot.get(
        "minus_di14"
    )

    rsi_value = snapshot.get(
        "rsi14"
    )

    # EMA 9/21 = 3 points
    if (
        ema9 is not None
        and ema21 is not None
    ):
        if ema9 > ema21:
            up += 3
        elif ema9 < ema21:
            down += 3

    # Price vs EMA9 = 1 point
    if (
        ema9 is not None
        and price is not None
    ):
        if price > ema9:
            up += 1
        elif price < ema9:
            down += 1

    # Price vs EMA21 = 1 point
    if (
        ema21 is not None
        and price is not None
    ):
        if price > ema21:
            up += 1
        elif price < ema21:
            down += 1

    # Structure = 3 points
    if structure == "BULLISH":
        up += 3
    elif structure == "BEARISH":
        down += 3

    # Breakout = 3 points
    if breakout == "BULLISH_BREAKOUT":
        up += 3
    elif breakout == "BEARISH_BREAKOUT":
        down += 3

    # ADX / DI = 2 points
    if (
        adx is not None
        and plus_di is not None
        and minus_di is not None
        and adx >= 20
    ):
        if plus_di > minus_di:
            up += 2
        elif minus_di > plus_di:
            down += 2

    # RSI = 1 point
    if rsi_value is not None:
        if 50 < rsi_value < 70:
            up += 1
        elif 30 < rsi_value < 50:
            down += 1

    # Last candle body = 1 point
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
            ema9 is not None
            and ema21 is not None
            and ema9 >= ema21
        ):
            direction = "UP"
        else:
            direction = "DOWN"

    return {
        "direction": direction,
        "up": up,
        "down": down,
    }


# ============================================================
# GEMINI PROMPT
# ============================================================

MT5_ANALYSIS_PROMPT = r"""
You are ZinoProSignalAI, a strict binary-options technical-analysis engine.

IMPORTANT:
This system is compatible with the old MT4 ZinoProSignalAI analysis logic.

TIMEFRAME:
M1 ONLY.

You receive CLOSED M1 candles and calculated technical data.

DO NOT invent indicators or prices.

PRIMARY PRIORITY:
1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle behavior
7. EMA 9/21
8. RSI 14
9. Williams %R 14
10. Keltner Channel
11. ADX / DI

The final decision MUST be either:
UP
or
DOWN

NEVER return:
WAIT
NO SIGNAL
NEUTRAL
HOLD

The system must always choose the stronger direction.

SCORING:

The historical MT4 engine used an 18-point final display.

The analytical categories are:

Structure: 2
Breakout: 2
Liquidity: 1
Momentum: 2
Candle: 2
RSI: 1
Summary: 2
Oscillators: 2
Moving Averages: 2

The original engine normalizes the final display to 18 points.

Use the supplied technical snapshot as the source of truth.

Do not invent a technical condition that is not supported by the data.

Do not give 90%+ confidence unless there is exceptionally strong multi-factor confluence.

Confidence should normally remain moderate when the market is mixed, ranging, or contradictory.

DIRECTION RULE:

If bullish evidence clearly dominates:
UP

If bearish evidence clearly dominates:
DOWN

If the market is mixed:
choose the direction supported by the stronger evidence and lower confidence.

ENTRY:

The entry price is the close of the latest CLOSED candle.

For M1:
Entry after = 1 minute.

CANCELLATION:

For UP:
use the supplied recent 8-candle low.

For DOWN:
use the supplied recent 8-candle high.

RETURN ONLY VALID JSON.

JSON FORMAT:

{
  "direction": "UP" or "DOWN",
  "confidence": 1-99,
  "up_score": 0-18,
  "down_score": 0-18,
  "reason": "short technical reason",
  "cancellation_reason": "short cancellation condition"
}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles,
    snapshot,
    pre_score
):
    if gemini_client is None:
        raise RuntimeError(
            "Gemini client is not initialized"
        )

    payload = {
        "symbol": symbol,
        "timeframe": timeframe,
        "closed_candles_count": len(candles),

        "pre_score": pre_score,

        "technical_snapshot": snapshot,

        "latest_candles": candles[-20:],
    }

    prompt = (
        MT5_ANALYSIS_PROMPT
        + "\n\nDATA:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":")
        )
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    text = response.text or ""

    try:
        result = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            result = json.loads(
                text[start:end + 1]
            )
        else:
            raise ValueError(
                "Gemini returned invalid JSON"
            )

    return result


# ============================================================
# NORMALIZE FINAL SIGNAL
# ============================================================

def ensure_directional_signal(
    result,
    pre_score
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
        "DOWN"
    ):
        direction = pre_score["direction"]

    try:
        confidence = int(
            float(
                result.get(
                    "confidence",
                    50
                )
            )
        )
    except Exception:
        confidence = 50

    confidence = max(
        1,
        min(
            99,
            confidence
        )
    )

    try:
        up = int(
            float(
                result.get(
                    "up_score",
                    pre_score["up"]
                )
            )
        )
    except Exception:
        up = pre_score["up"]

    try:
        down = int(
            float(
                result.get(
                    "down_score",
                    pre_score["down"]
                )
            )
        )
    except Exception:
        down = pre_score["down"]

    up = max(
        0,
        min(
            18,
            up
        )
    )

    down = max(
        0,
        min(
            18,
            down
        )
    )

    # Compatibility with the old MT4 engine:
    # normalize the displayed total to 18.
    total = up + down

    if total < 18:
        missing = 18 - total

        if direction == "UP":
            up += missing
        else:
            down += missing

    elif total > 18:
        excess = total - 18

        if direction == "UP":
            up = max(
                0,
                up - excess
            )
        else:
            down = max(
                0,
                down - excess
            )

    # Direction must follow stronger displayed score.
    if up > down:
        direction = "UP"

    elif down > up:
        direction = "DOWN"

    else:
        direction = pre_score["direction"]

    # Old behavior:
    # do not show very high confidence without a large gap.
    if confidence >= 90:
        if abs(up - down) < 8:
            confidence = 89

    reason = str(
        result.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:
        if direction == "UP":
            reason = (
                "Bullish confluence from "
                "structure, momentum and trend."
            )
        else:
            reason = (
                "Bearish confluence from "
                "structure, momentum and trend."
            )

    cancellation_reason = str(
        result.get(
            "cancellation_reason",
            ""
        )
    ).strip()

    return {
        "direction": direction,
        "confidence": confidence,
        "up_score": up,
        "down_score": down,
        "reason": reason,
        "cancellation_reason": cancellation_reason,
    }


# ============================================================
# ENTRY CALCULATION
# ============================================================

def calculate_entry(
    candles,
    direction,
    timeframe
):
    if not candles:
        return None

    last = candles[-1]

    entry_price = float(
        last["close"]
    )

    recent = candles[-8:]

    if direction == "UP":
        cancellation = min(
            c["low"]
            for c in recent
        )

    else:
        cancellation = max(
            c["high"]
            for c in recent
        )

    delay = timeframe_minutes(
        timeframe
    )

    now = now_algiers()

    entry_time = now.replace(
        second=0,
        microsecond=0
    ) + timedelta(
        minutes=delay
    )

    return {
        "entry_price": entry_price,
        "cancellation": cancellation,
        "delay": delay,
        "entry_time": entry_time,
    }


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(
    symbol,
    timeframe,
    signal,
    entry,
    mode="BASE",
    recovery_number=0
):
    direction = signal["direction"]

    arrow = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    entry_price = entry[
        "entry_price"
    ]

    cancellation = entry[
        "cancellation"
    ]

    if direction == "UP":
        cancel_text = (
            f"❌ إلغاء إذا أغلقت شمعة تحت "
            f"{cancellation:.5f}"
        )
    else:
        cancel_text = (
            f"❌ إلغاء إذا أغلقت شمعة فوق "
            f"{cancellation:.5f}"
        )

    if mode == "RECOVERY":
        trade_title = (
            f"🔁 RECOVERY "
            f"{recovery_number}/{RECOVERY_LIMIT}"
        )
    else:
        trade_title = "🎯 BASE TRADE"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n\n"
        f"{trade_title}\n"
        f"{arrow}\n\n"
        f"🔥 Confidence: "
        f"{signal['confidence']}%\n"
        f"🟢 UP Score: "
        f"{signal['up_score']}/18\n"
        f"🔴 DOWN Score: "
        f"{signal['down_score']}/18\n\n"
        f"⏱️ Entry after: "
        f"{entry['delay']} minute"
        f"{'s' if entry['delay'] != 1 else ''}\n"
        f"🕐 ENTRY TIME: "
        f"{format_time(entry['entry_time'])}\n"
        f"💰 ENTRY PRICE: "
        f"{entry_price:.5f}\n"
        f"{cancel_text}\n\n"
        f"🧠 {signal['reason']}"
    )


# ============================================================
# HISTORY RECORD
# ============================================================

def add_history_record(
    symbol,
    timeframe,
    signal,
    entry,
    mode,
    recovery_number
):
    record = {
        "id": int(
            time.time() * 1000
        ),

        "symbol": symbol,
        "timeframe": timeframe,

        "mode": mode,

        "recovery_number": recovery_number,

        "direction": signal["direction"],

        "confidence": signal["confidence"],

        "up_score": signal["up_score"],
        "down_score": signal["down_score"],

        "entry_price": entry["entry_price"],

        "cancellation": entry["cancellation"],

        "entry_time": format_time(
            entry["entry_time"]
        ),

        "created_at": format_time(),

        "result": "PENDING",
    }

    signal_history.append(record)

    save_history()

    return record


# ============================================================
# CYCLE
# ============================================================

def get_cycle(symbol, timeframe):
    key = f"{symbol}|{timeframe}"

    return cycle_state.get(
        key,
        {
            "active": False,
            "recovery_used": 0,
            "last_direction": None,
        }
    )


def set_cycle(
    symbol,
    timeframe,
    state
):
    key = f"{symbol}|{timeframe}"

    cycle_state[key] = state


def start_base_cycle(
    symbol,
    timeframe,
    direction
):
    set_cycle(
        symbol,
        timeframe,
        {
            "active": True,
            "recovery_used": 0,
            "last_direction": direction,
        }
    )


def start_recovery_cycle(
    symbol,
    timeframe,
    direction
):
    set_cycle(
        symbol,
        timeframe,
        {
            "active": True,
            "recovery_used": 1,
            "last_direction": direction,
        }
    )


def end_cycle(
    symbol,
    timeframe
):
    set_cycle(
        symbol,
        timeframe,
        {
            "active": False,
            "recovery_used": 0,
            "last_direction": None,
        }
    )


# ============================================================
# FIND PENDING
# ============================================================

def find_pending_signal():
    for record in reversed(
        signal_history
    ):
        if record.get("result") == "PENDING":
            return record

    return None


# ============================================================
# ANALYZE PAIR
# ============================================================

def analyze_pair(
    symbol,
    data,
    mode="BASE",
    recovery_number=0
):
    timeframe = str(
        data.get(
            "timeframe",
            "M1"
        )
    ).upper()

    # M1 ONLY
    timeframe = "M1"

    candles = get_closed_candles(
        data.get(
            "candles",
            []
        )
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        logger.warning(
            "%s: insufficient closed candles: %d",
            symbol,
            len(candles)
        )
        return None

    snapshot = build_snapshot(
        candles
    )

    pre_score = directional_pre_score(
        snapshot,
        candles
    )

    logger.info(
        "%s pre-score | UP=%d DOWN=%d direction=%s",
        symbol,
        pre_score["up"],
        pre_score["down"],
        pre_score["direction"]
    )

    try:
        raw_result = analyze_with_gemini(
            symbol,
            timeframe,
            candles,
            snapshot,
            pre_score
        )

        signal = ensure_directional_signal(
            raw_result,
            pre_score
        )

    except Exception as e:
        logger.exception(
            "%s Gemini analysis failed: %s",
            symbol,
            e
        )

        # Safe fallback to old deterministic pre-score.
        signal = ensure_directional_signal(
            {
                "direction": pre_score["direction"],
                "confidence": 50,
                "up_score": pre_score["up"],
                "down_score": pre_score["down"],
                "reason": (
                    "Gemini unavailable; "
                    "using technical pre-score."
                ),
            },
            pre_score
        )

    entry = calculate_entry(
        candles,
        signal["direction"],
        timeframe
    )

    if entry is None:
        return None

    message = format_signal(
        symbol,
        timeframe,
        signal,
        entry,
        mode,
        recovery_number
    )

    record = add_history_record(
        symbol,
        timeframe,
        signal,
        entry,
        mode,
        recovery_number
    )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "entry": entry,
        "message": message,
        "record": record,
        "snapshot": snapshot,
        "pre_score": pre_score,
    }


# ============================================================
# BEST PAIR
# ============================================================

def choose_best_pair():
    best = None

    with data_lock:
        items = list(
            mt5_data.items()
        )

    for symbol, data in items:
        timeframe = str(
            data.get(
                "timeframe",
                "M1"
            )
        ).upper()

        if timeframe != "M1":
            continue

        candles = get_closed_candles(
            data.get(
                "candles",
                []
            )
        )

        if len(candles) < MIN_CLOSED_CANDLES:
            continue

        try:
            snapshot = build_snapshot(
                candles
            )

            score = directional_pre_score(
                snapshot,
                candles
            )

            strength = max(
                score["up"],
                score["down"]
            )

            gap = abs(
                score["up"]
                - score["down"]
            )

            candidate = (
                strength,
                gap,
                symbol
            )

            if best is None or candidate > best[0]:
                best = (
                    candidate,
                    symbol,
                    data
                )

        except Exception as e:
            logger.warning(
                "Pre-score failed for %s: %s",
                symbol,
                e
            )

    if best is None:
        return None

    return best[1], best[2]


# ============================================================
# AUTO ANALYSIS
# ============================================================

async def auto_analyze_pair(
    symbol,
    force=False
):
    if not analysis_lock.acquire(
        blocking=False
    ):
        return

    try:
        with data_lock:
            data = mt5_data.get(
                symbol
            )

        if not data:
            return

        timeframe = str(
            data.get(
                "timeframe",
                "M1"
            )
        ).upper()

        if timeframe != "M1":
            return

        candles = get_closed_candles(
            data.get(
                "candles",
                []
            )
        )

        if len(candles) < MIN_CLOSED_CANDLES:
            return

        last_closed_time = candles[-1][
            "time"
        ]

        # Avoid analyzing the same closed candle repeatedly.
        last_done = last_analysis_time.get(
            symbol
        )

        if (
            not force
            and last_done == last_closed_time
        ):
            return

        now_ts = time.time()

        previous_setup = last_setup_time.get(
            symbol,
            0
        )

        if (
            not force
            and now_ts - previous_setup
            < SETUP_REPEAT_BLOCK_SECONDS
        ):
            return

        result = analyze_pair(
            symbol,
            data,
            mode="BASE",
            recovery_number=0
        )

        if not result:
            return

        last_analysis_time[
            symbol
        ] = last_closed_time

        last_setup_time[
            symbol
        ] = now_ts

        start_base_cycle(
            symbol,
            "M1",
            result["signal"]["direction"]
        )

        await send_owner_message(
            result["message"]
        )

    finally:
        analysis_lock.release()


# ============================================================
# OWNER TELEGRAM SEND
# ============================================================

telegram_app = None


async def send_owner_message(text):
    global telegram_app

    if telegram_app is None:
        return

    try:
        await telegram_app.bot.send_message(
            chat_id=OWNER_ID,
            text=text
        )
    except Exception as e:
        logger.exception(
            "Telegram send failed: %s",
            e
        )


# ============================================================
# MT5 HTTP DATA
# ============================================================

def validate_mt5_request(
    headers,
    body
):
    header_key = (
        headers.get("X-MT5-API-Key")
        or headers.get("X-API-Key")
        or headers.get("X-MT4-API-Key")
        or ""
    ).strip()

    body_key = str(
        body.get(
            "api_key",
            ""
        )
    ).strip()

    provided = (
        header_key
        or body_key
    )

    if not MT5_API_KEY:
        return False

    return provided == MT5_API_KEY


def store_mt5_data(body):
    symbol = str(
        body.get(
            "symbol",
            ""
        )
    ).strip()

    if not symbol:
        return False, "missing symbol"

    timeframe = str(
        body.get(
            "timeframe",
            "M1"
        )
    ).upper().strip()

    # M1 ONLY
    if timeframe != "M1":
        return False, (
            "Only M1 is enabled"
        )

    candles = normalize_candles(
        body.get(
            "candles",
            []
        )
    )

    if len(candles) < 2:
        return False, (
            "insufficient candles"
        )

    current_bid = body.get(
        "current_bid"
    )

    current_ask = body.get(
        "current_ask"
    )

    with data_lock:
        mt5_data[symbol] = {
            "symbol": symbol,
            "timeframe": "M1",
            "candles": candles,
            "current_bid": current_bid,
            "current_ask": current_ask,
            "digits": body.get(
                "digits"
            ),
            "closed_candle_time": body.get(
                "closed_candle_time"
            ),
            "batch_id": body.get(
                "batch_id"
            ),
            "received_at": time.time(),
        }

    return True, "stored"


# ============================================================
# HTTP SERVER
# ============================================================

class HTTPHandler(
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
        code,
        payload
    ):
        data = json.dumps(
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
            str(len(data))
        )

        self.end_headers()

        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(
            self.path
        ).path

        if path in (
            "/",
            "/health",
            "/healthz"
        ):
            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "source": "MT5",
                    "timeframe": "M1",
                    "pairs": len(mt5_data),
                }
            )
            return

        if path in (
            "/mt5status",
            "/api/mt5/status"
        ):
            with data_lock:
                status = {}

                for symbol, data in mt5_data.items():
                    status[symbol] = {
                        "timeframe": data.get(
                            "timeframe"
                        ),
                        "candles": len(
                            data.get(
                                "candles",
                                []
                            )
                        ),
                        "received_at": data.get(
                            "received_at"
                        ),
                    }

            self.send_json(
                200,
                {
                    "status": "ok",
                    "markets": status
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
        path = urlparse(
            self.path
        ).path

        if path not in (
            "/mt5",
            "/api/mt5",
            # backward compatibility
            "/mt4",
            "/api/mt4",
        ):
            self.send_json(
                404,
                {
                    "error": "not found"
                }
            )
            return

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw = self.rfile.read(
                length
            )

            body = json.loads(
                raw.decode("utf-8")
            )

        except Exception as e:
            self.send_json(
                400,
                {
                    "error": "invalid json",
                    "detail": str(e)
                }
            )
            return

        if not validate_mt5_request(
            self.headers,
            body
        ):
            self.send_json(
                401,
                {
                    "error": "unauthorized"
                }
            )
            return

        ok, message = store_mt5_data(
            body
        )

        if not ok:
            self.send_json(
                400,
                {
                    "error": message
                }
            )
            return

        symbol = str(
            body.get(
                "symbol",
                ""
            )
        ).strip()

        # Immediate analysis in a separate thread/event.
        def trigger():
            try:
                asyncio.run(
                    auto_analyze_pair(
                        symbol
                    )
                )
            except Exception as e:
                logger.exception(
                    "Immediate MT5 analysis failed: %s",
                    e
                )

        threading.Thread(
            target=trigger,
            daemon=True
        ).start()

        self.send_json(
            200,
            {
                "ok": True,
                "service": "ZinoProSignalAI",
                "source": "MT5",
                "symbol": symbol,
                "timeframe": "M1",
                "candles": len(
                    body.get(
                        "candles",
                        []
                    )
                )
            }
        )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HTTPHandler
    )

    logger.info(
        "HTTP server started on port %d",
        PORT
    )

    server.serve_forever()


# ============================================================
# BACKGROUND LOOP
# ============================================================

async def background_loop():
    await asyncio.sleep(5)

    while True:
        try:
            best = choose_best_pair()

            if best:
                symbol, _ = best

                await auto_analyze_pair(
                    symbol
                )

        except Exception as e:
            logger.exception(
                "Background analysis error: %s",
                e
            )

        await asyncio.sleep(
            AUTO_ANALYSIS_INTERVAL_SECONDS
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

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ MT5 connected mode\n"
        "📊 Timeframe: M1\n"
        "🧠 MT4-compatible analysis engine\n"
        "🔁 Recovery: 1/1\n\n"
        "Commands:\n"
        "/stats\n"
        "/history\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt5status\n"
        "/analyze"
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    wins = sum(
        1
        for x in signal_history
        if x.get("result") == "WIN"
    )

    losses = sum(
        1
        for x in signal_history
        if x.get("result") == "LOSS"
    )

    pending = sum(
        1
        for x in signal_history
        if x.get("result") == "PENDING"
    )

    total = wins + losses

    if total > 0:
        winrate = (
            wins / total
        ) * 100
    else:
        winrate = 0

    text = (
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"⏳ Pending: {pending}\n"
        f"📈 Win Rate: {winrate:.1f}%\n"
        f"📚 Total: {total}"
    )

    await update.message.reply_text(
        text
    )


async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    recent = signal_history[
        -HISTORY_DISPLAY_COUNT:
    ]

    if not recent:
        await update.message.reply_text(
            "📚 History is empty."
        )
        return

    lines = [
        "📚 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━"
    ]

    for record in reversed(
        recent
    ):
        result = record.get(
            "result",
            "PENDING"
        )

        if result == "WIN":
            icon = "🟢"
        elif result == "LOSS":
            icon = "🔴"
        else:
            icon = "⏳"

        mode = record.get(
            "mode",
            "BASE"
        )

        direction = record.get(
            "direction",
            "?"
        )

        up = record.get(
            "up_score",
            0
        )

        down = record.get(
            "down_score",
            0
        )

        confidence = record.get(
            "confidence",
            0
        )

        lines.append(
            f"{icon} "
            f"{record.get('symbol','?')} "
            f"{record.get('timeframe','M1')}"
        )

        lines.append(
            f"   {mode} | "
            f"{'🟢' if direction == 'UP' else '🔴'} "
            f"{direction}"
        )

        lines.append(
            f"   🎯 {confidence}% | "
            f"📈 {up}/18 "
            f"📉 {down}/18"
        )

        lines.append(
            f"   💰 "
            f"{record.get('entry_price', 0):.5f}"
        )

        lines.append(
            f"   ⏰ "
            f"{record.get('entry_time','')}"
        )

        lines.append(
            f"   📌 {result}"
        )

        lines.append("")

    await update.message.reply_text(
        "\n".join(lines)
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    pending = find_pending_signal()

    if pending is None:
        await update.message.reply_text(
            "ℹ️ No pending signal."
        )
        return

    pending["result"] = "WIN"

    save_history()

    symbol = pending.get(
        "symbol"
    )

    timeframe = pending.get(
        "timeframe",
        "M1"
    )

    end_cycle(
        symbol,
        timeframe
    )

    await update.message.reply_text(
        "🟢 WIN recorded.\n"
        "✅ Cycle completed."
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    pending = find_pending_signal()

    if pending is None:
        await update.message.reply_text(
            "ℹ️ No pending signal."
        )
        return

    pending["result"] = "LOSS"

    save_history()

    symbol = pending.get(
        "symbol"
    )

    timeframe = pending.get(
        "timeframe",
        "M1"
    )

    mode = pending.get(
        "mode",
        "BASE"
    )

    if mode == "RECOVERY":
        end_cycle(
            symbol,
            timeframe
        )

        await update.message.reply_text(
            "🔴 RECOVERY LOSS.\n"
            "⛔ Cycle completed."
        )

        return

    cycle = get_cycle(
        symbol,
        timeframe
    )

    if cycle.get(
        "recovery_used",
        0
    ) >= RECOVERY_LIMIT:
        end_cycle(
            symbol,
            timeframe
        )

        await update.message.reply_text(
            "🔴 LOSS.\n"
            "⛔ Recovery limit reached."
        )

        return

    # One recovery only.
    with data_lock:
        data = mt5_data.get(
            symbol
        )

    if not data:
        end_cycle(
            symbol,
            timeframe
        )

        await update.message.reply_text(
            "🔴 LOSS.\n"
            "⚠️ MT5 data unavailable."
        )

        return

    candles = get_closed_candles(
        data.get(
            "candles",
            []
        )
    )

    if len(candles) < MIN_CLOSED_CANDLES:
        end_cycle(
            symbol,
            timeframe
        )

        await update.message.reply_text(
            "🔴 LOSS.\n"
            "⚠️ Not enough candles for recovery."
        )

        return

    result = analyze_pair(
        symbol,
        data,
        mode="RECOVERY",
        recovery_number=1
    )

    if not result:
        end_cycle(
            symbol,
            timeframe
        )

        await update.message.reply_text(
            "🔴 LOSS.\n"
            "⚠️ Recovery analysis failed."
        )

        return

    start_recovery_cycle(
        symbol,
        timeframe,
        result["signal"]["direction"]
    )

    await update.message.reply_text(
        result["message"]
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    signal_history.clear()

    cycle_state.clear()

    last_analysis_time.clear()

    last_setup_time.clear()

    save_history()

    await update.message.reply_text(
        "♻️ Everything reset.\n"
        "📚 History cleared."
    )


async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    with data_lock:
        items = list(
            mt5_data.items()
        )

    if not items:
        await update.message.reply_text(
            "🔴 MT5: No data received."
        )
        return

    lines = [
        "🟢 ZinoProSignalAI MT5 STATUS",
        "━━━━━━━━━━━━━━━━━━"
    ]

    for symbol, data in items:
        candles = data.get(
            "candles",
            []
        )

        age = (
            time.time()
            - data.get(
                "received_at",
                time.time()
            )
        )

        lines.append(
            f"📊 {symbol} | M1"
        )

        lines.append(
            f"🕯 Candles: {len(candles)}"
        )

        lines.append(
            f"📡 Last update: "
            f"{age:.0f}s ago"
        )

        lines.append("")

    await update.message.reply_text(
        "\n".join(lines)
    )


async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not await owner_only(update):
        return

    best = choose_best_pair()

    if not best:
        await update.message.reply_text(
            "⚠️ No MT5 pair has enough M1 candles."
        )
        return

    symbol, _ = best

    await auto_analyze_pair(
        symbol,
        force=True
    )

    await update.message.reply_text(
        f"🔎 Analysis requested for {symbol}."
    )


# ============================================================
# MAIN
# ============================================================

async def post_init(
    application: Application
):
    global telegram_app

    telegram_app = application

    asyncio.create_task(
        background_loop()
    )


def main():
    load_history()

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is missing"
        )

    if not MT5_API_KEY:
        logger.warning(
            "MT5_API_KEY is missing"
        )

    # HTTP server for Render / MT5
    threading.Thread(
        target=start_http_server,
        daemon=True
    ).start()

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
            "analyze",
            analyze_command
        )
    )

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "Analysis timeframe: M1"
    )

    logger.info(
        "Recovery limit: %d",
        RECOVERY_LIMIT
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
