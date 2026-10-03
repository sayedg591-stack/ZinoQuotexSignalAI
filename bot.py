import os
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
    MessageHandler,
    ContextTypes,
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

# تحليل البيانات كل دقيقة
AUTO_ANALYSIS_INTERVAL_MINUTES = 1

# الدخول بعد دقيقتين
ENTRY_DELAY_MINUTES = 2

# أقل مدة بين أي صفقتين
SIGNAL_COOLDOWN_SECONDS = 120

# أقل عدد شموع مغلقة
MIN_CLOSED_CANDLES = 40


# ============================================================
# GLOBAL DATA
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

last_auto_analysis = {}

last_processed_closed_candle = {}

telegram_application = None
telegram_loop = None

# وقت آخر صفقة مرسلة
last_signal_sent_at = 0.0

# حماية من إرسال صفقتين بنفس الوقت
signal_send_lock = threading.Lock()


# ============================================================
# TRADING CYCLE
# ============================================================

cycle_lock = threading.Lock()

# لا توجد دورة نشطة في البداية
active_cycle = {
    "active": False,

    # الزوج والفريم الحالي
    "symbol": None,
    "timeframe": None,

    # نوع الصفقة الحالية
    # BASE = أساسية
    # RECOVERY = تعويض
    "trade_type": None,

    # هل استعملنا التعويض؟
    "recovery_used": False,

    # وقت آخر صفقة في الدورة
    "last_trade_time": 0.0,

    # رقم الصفقة داخل الدورة
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
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# TIME
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def get_next_entry_time(timeframe=None):
    """
    الدخول دائما بعد دقيقتين.
    الفريم لا يغير مدة الدخول.
    """
    return now_algiers() + timedelta(
        minutes=ENTRY_DELAY_MINUTES
    )


def format_dt(dt):
    return dt.strftime("%H:%M:%S")


# ============================================================
# TIMEFRAME
# ============================================================

def timeframe_to_minutes(timeframe):
    if not timeframe:
        return 1

    tf = str(timeframe).upper().strip()

    match = re.match(
        r"^(\d+(?:\.\d+)?)(MN|MO|W|D|H|M)$",
        tf,
    )

    if not match:
        return 1

    value = float(match.group(1))
    unit = match.group(2)

    if unit == "M":
        return max(1, int(value))

    if unit == "H":
        return max(1, int(value * 60))

    if unit == "D":
        return max(1, int(value * 1440))

    if unit == "W":
        return max(1, int(value * 10080))

    if unit in ("MN", "MO"):
        return max(1, int(value * 43200))

    return 1


# ============================================================
# OWNER
# ============================================================

def is_owner(update: Update):

    if not OWNER_ID:
        return False

    try:
        user = update.effective_user

        if not user:
            return False

        return user.id == int(OWNER_ID)

    except Exception:
        return False


# ============================================================
# SAFE CONVERSION
# ============================================================

def safe_float(value, default=0.0):

    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):

    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        return int(float(value))

    except Exception:
        return default


# ============================================================
# CANDLES
# ============================================================

def normalize_candle(candle):

    if not isinstance(candle, dict):
        return None

    timestamp = (
        candle.get("timestamp")
        or candle.get("time")
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

    if (
        timestamp is None
        or open_price is None
        or high_price is None
        or low_price is None
        or close_price is None
    ):
        return None

    return {
        "timestamp": safe_int(timestamp),
        "open": safe_float(open_price),
        "high": safe_float(high_price),
        "low": safe_float(low_price),
        "close": safe_float(close_price),
        "volume": safe_float(volume),
    }


def normalize_candles(candles):

    if not isinstance(candles, list):
        return []

    result = []

    for candle in candles:

        normalized = normalize_candle(
            candle
        )

        if normalized:
            result.append(normalized)

    result.sort(
        key=lambda x: x["timestamp"]
    )

    return result


# ============================================================
# EMA
# ============================================================

def calculate_ema(values, period):

    if len(values) < period:
        return None

    ema = sum(
        values[:period]
    ) / period

    multiplier = 2 / (period + 1)

    for price in values[period:]:

        ema = (
            (price - ema)
            * multiplier
        ) + ema

    return ema


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    closes,
    period=14,
):

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):

        change = (
            closes[i]
            - closes[i - 1]
        )

        if change > 0:

            gains.append(change)
            losses.append(0)

        else:

            gains.append(0)
            losses.append(
                abs(change)
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

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# WILLIAMS %R
# ============================================================

def calculate_williams_r(
    candles,
    period=14,
):

    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest_high = max(
        c["high"]
        for c in recent
    )

    lowest_low = min(
        c["low"]
        for c in recent
    )

    close = candles[-1]["close"]

    if highest_high == lowest_low:
        return -50.0

    return (
        (
            highest_high
            - close
        )
        /
        (
            highest_high
            - lowest_low
        )
    ) * -100


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    candles,
    period=10,
):

    if len(candles) < period + 1:
        return None

    trs = []

    for i in range(
        1,
        len(candles)
    ):

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

        trs.append(tr)

    if len(trs) < period:
        return None

    atr = (
        sum(trs[:period])
        / period
    )

    for tr in trs[period:]:

        atr = (
            (
                atr
                * (period - 1)
            )
            + tr
        ) / period

    return atr


# ============================================================
# ADX
# ============================================================

def calculate_adx(
    candles,
    period=14,
):

    if len(candles) < (
        period * 2 + 1
    ):
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    trs = []
    plus_dm = []
    minus_dm = []

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
        return {
            "adx": None,
            "plus_di": None,
            "minus_di": None,
        }

    atr = (
        sum(trs[:period])
        / period
    )

    plus_smoothed = (
        sum(plus_dm[:period])
        / period
    )

    minus_smoothed = (
        sum(minus_dm[:period])
        / period
    )

    dx_values = []

    plus_di = None
    minus_di = None

    for i in range(
        period,
        len(trs)
    ):

        atr = (
            (
                atr
                * (period - 1)
            )
            + trs[i]
        ) / period

        plus_smoothed = (
            (
                plus_smoothed
                * (period - 1)
            )
            + plus_dm[i]
        ) / period

        minus_smoothed = (
            (
                minus_smoothed
                * (period - 1)
            )
            + minus_dm[i]
        ) / period

        if atr == 0:

            plus_di = 0
            minus_di = 0

        else:

            plus_di = (
                100
                * plus_smoothed
                / atr
            )

            minus_di = (
                100
                * minus_smoothed
                / atr
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

        dx_values.append(dx)

    if not dx_values:

        return {
            "adx": None,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    if len(dx_values) < period:

        adx = (
            sum(dx_values)
            / len(dx_values)
        )

    else:

        adx = (
            sum(
                dx_values[:period]
            )
            / period
        )

        for dx in dx_values[period:]:

            adx = (
                (
                    adx
                    * (period - 1)
                )
                + dx
            ) / period

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


# ============================================================
# STRUCTURE
# ============================================================

def calculate_market_structure(
    candles
):

    if len(candles) < 6:

        return {
            "trend": "UNKNOWN"
        }

    recent = candles[-6:]

    highs = [
        c["high"]
        for c in recent
    ]

    lows = [
        c["low"]
        for c in recent
    ]

    higher_highs = (
        highs[-1] > highs[-3]
        and highs[-3] > highs[-5]
    )

    higher_lows = (
        lows[-1] > lows[-3]
        and lows[-3] > lows[-5]
    )

    lower_highs = (
        highs[-1] < highs[-3]
        and highs[-3] < highs[-5]
    )

    lower_lows = (
        lows[-1] < lows[-3]
        and lows[-3] < lows[-5]
    )

    if (
        higher_highs
        and higher_lows
    ):

        trend = "UP"

    elif (
        lower_highs
        and lower_lows
    ):

        trend = "DOWN"

    else:

        trend = "RANGE"

    return {
        "trend": trend,
        "higher_highs": higher_highs,
        "higher_lows": higher_lows,
        "lower_highs": lower_highs,
        "lower_lows": lower_lows,
    }


# ============================================================
# BREAKOUT
# ============================================================

def calculate_breakout(
    candles
):

    if len(candles) < 12:

        return {
            "direction": "NONE",
            "strength": 0,
        }

    previous = candles[-11:-1]

    highest = max(
        c["high"]
        for c in previous
    )

    lowest = min(
        c["low"]
        for c in previous
    )

    last = candles[-1]

    if last["close"] > highest:

        return {
            "direction": "UP",
            "strength": 1,
        }

    if last["close"] < lowest:

        return {
            "direction": "DOWN",
            "strength": 1,
        }

    return {
        "direction": "NONE",
        "strength": 0,
    }


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_technical_snapshot(
    candles
):

    if not candles:
        return {}

    closes = [
        c["close"]
        for c in candles
    ]

    ema9 = calculate_ema(
        closes,
        9
    )

    ema21 = calculate_ema(
        closes,
        21
    )

    rsi14 = calculate_rsi(
        closes,
        14
    )

    williams_r14 = (
        calculate_williams_r(
            candles,
            14
        )
    )

    atr10 = calculate_atr(
        candles,
        10
    )

    adx_data = calculate_adx(
        candles,
        14
    )

    structure = (
        calculate_market_structure(
            candles
        )
    )

    breakout = (
        calculate_breakout(
            candles
        )
    )

    return {
        "last_close": candles[-1]["close"],
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi14,
        "williams_r14": williams_r14,
        "atr10": atr10,
        "adx14": adx_data,
        "keltner": {
            "ema20": calculate_ema(
                closes,
                20
            ),
            "atr10": atr10,
            "multiplier": 5,
        },
        "structure": structure,
        "breakout": breakout,
    }


# ============================================================
# PRE-SCORE
# اختيار زوج واحد فقط قبل Gemini
# ============================================================

def technical_candidate_score(
    candles
):

    if len(candles) < MIN_CLOSED_CANDLES:
        return -999

    snapshot = (
        build_technical_snapshot(
            candles
        )
    )

    score = 0

    structure = (
        snapshot
        .get("structure", {})
        .get("trend")
    )

    breakout = (
        snapshot
        .get("breakout", {})
        .get("direction")
    )

    adx_data = (
        snapshot.get(
            "adx14",
            {}
        )
    )

    adx = safe_float(
        adx_data.get("adx")
    )

    plus_di = safe_float(
        adx_data.get("plus_di")
    )

    minus_di = safe_float(
        adx_data.get("minus_di")
    )

    ema9 = snapshot.get(
        "ema9"
    )

    ema21 = snapshot.get(
        "ema21"
    )

    rsi = snapshot.get(
        "rsi14"
    )

    williams = snapshot.get(
        "williams_r14"
    )

    last = candles[-1]

    previous = candles[-2]

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    if structure in (
        "UP",
        "DOWN",
    ):

        score += 3

    elif structure == "RANGE":

        score -= 1

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    if breakout in (
        "UP",
        "DOWN",
    ):

        score += 2

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if abs(
            ema9 - ema21
        ) > 0:

            score += 2

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    if adx >= 20:

        score += 2

    elif adx >= 15:

        score += 1

    # --------------------------------------------------------
    # Candle
    # --------------------------------------------------------

    body = abs(
        last["close"]
        - last["open"]
    )

    previous_range = (
        previous["high"]
        - previous["low"]
    )

    if (
        previous_range > 0
        and body
        > previous_range * 0.25
    ):

        score += 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi is not None:

        if (
            35
            <= rsi
            <= 65
        ):

            score += 1

    # --------------------------------------------------------
    # Williams
    # --------------------------------------------------------

    if williams is not None:

        if (
            -80
            < williams
            < -20
        ):

            score += 1

    # --------------------------------------------------------
    # DI
    # --------------------------------------------------------

    if (
        plus_di > minus_di
        or minus_di > plus_di
    ):

        score += 1

    return score


def choose_best_pair():

    candidates = []

    with mt4_lock:

        datasets = list(
            mt4_data.items()
        )

    for key, data in datasets:

        symbol = data.get(
            "symbol"
        )

        timeframe = data.get(
            "timeframe"
        )

        # التحليل المطلوب يبقى H1
        if str(
            timeframe
        ).upper() != "H1":

            continue

        candles = normalize_candles(
            data.get(
                "candles",
                []
            )
        )

        if len(candles) < (
            MIN_CLOSED_CANDLES + 1
        ):

            continue

        closed = candles[:-1]

        if len(closed) < MIN_CLOSED_CANDLES:

            continue

        score = technical_candidate_score(
            closed
        )

        candidates.append(
            (
                score,
                symbol,
                timeframe,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    best = candidates[0]

    logger.info(
        "Selected ONE pair: "
        f"{best[1]} {best[2]} "
        f"technical score={best[0]}"
    )

    return {
        "symbol": best[1],
        "timeframe": best[2],
        "score": best[0],
    }


# ============================================================
# GEMINI
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles,
    recovery=False,
):

    if not GEMINI_API_KEY:

        logger.error(
            "GEMINI_API_KEY is missing."
        )

        return None

    try:

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        snapshot = (
            build_technical_snapshot(
                candles
            )
        )

        recent_candles = candles[-60:]

        trade_mode = (
            "RECOVERY"
            if recovery
            else "BASE"
        )

        prompt = f"""
You are the technical analysis engine
for ZinoProSignalAI.

MARKET:
Symbol: {symbol}
Timeframe: {timeframe}

TRADE MODE:
{trade_mode}

IMPORTANT:
Return exactly ONE direction:
UP or DOWN.

Never return:
WAIT
NO SIGNAL
NEUTRAL

Do not invent data.

Use only supplied candles and indicators.

Priority:

1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle behavior
7. EMA 9 / EMA 21
8. RSI 14
9. Williams %R 14
10. Keltner
11. ADX / DI

Technical snapshot:

{json.dumps(
    snapshot,
    ensure_ascii=False,
    indent=2
)}

Recent closed candles:

{json.dumps(
    recent_candles,
    ensure_ascii=False,
    indent=2
)}

Scoring:

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 3
Moving Averages = 3

TOTAL = 18

Calculate:

up_score
down_score

Select the stronger direction.

If scores are close:
use structure,
then EMA,
then latest candle.

Confidence must reflect actual confluence.

Do NOT use 90%+ unless exceptionally strong.

signal must be true.

direction must be UP or DOWN.

Return JSON ONLY.

Format:

{{
  "signal": true,
  "direction": "UP",
  "confidence": 75,
  "up_score": 11,
  "down_score": 7,
  "structure": 2,
  "breakout": 2,
  "liquidity": 1,
  "momentum": 2,
  "candle": 2,
  "rsi": 1,
  "summary": 2,
  "oscillators": 2,
  "moving_averages": 2,
  "confirmations": 4,
  "contradictions": 0,
  "reason": "Short factual reason based only on the supplied market data."
}}

No markdown.
No text outside JSON.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.10,
                response_mime_type="application/json",
            ),
        )

        text = response.text

        if not text:
            return None

        try:

            analysis = json.loads(
                text
            )

        except json.JSONDecodeError:

            logger.error(
                "Invalid Gemini JSON: %s",
                text[:1000]
            )

            return None

        if not isinstance(
            analysis,
            dict
        ):

            return None

        return analysis

    except Exception as e:

        logger.exception(
            "Gemini error: %s",
            e
        )

        return None


# ============================================================
# FORCE DIRECTION
# ============================================================

def ensure_directional_signal(
    analysis,
    candles,
):

    if not isinstance(
        analysis,
        dict
    ):

        return None

    up_score = safe_int(
        analysis.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        analysis.get(
            "down_score",
            0
        )
    )

    direction = str(
        analysis.get(
            "direction",
            ""
        )
    ).upper().strip()

    if direction not in (
        "UP",
        "DOWN",
    ):

        if up_score > down_score:

            direction = "UP"

        elif down_score > up_score:

            direction = "DOWN"

        else:

            snapshot = (
                build_technical_snapshot(
                    candles
                )
            )

            structure = (
                snapshot
                .get("structure", {})
                .get("trend")
            )

            if structure in (
                "UP",
                "DOWN",
            ):

                direction = structure

            else:

                ema9 = snapshot.get(
                    "ema9"
                )

                ema21 = snapshot.get(
                    "ema21"
                )

                if (
                    ema9 is not None
                    and ema21 is not None
                ):

                    direction = (
                        "UP"
                        if ema9 >= ema21
                        else "DOWN"
                    )

                elif len(candles) >= 2:

                    direction = (
                        "UP"
                        if candles[-1]["close"]
                        >= candles[-2]["close"]
                        else "DOWN"
                    )

                else:

                    direction = "UP"

    analysis["signal"] = True
    analysis["direction"] = direction

    confidence = safe_float(
        analysis.get(
            "confidence",
            0
        )
    )

    analysis["confidence"] = max(
        0,
        min(
            100,
            confidence
        )
    )

    if not str(
        analysis.get(
            "reason",
            ""
        )
    ).strip():

        analysis["reason"] = (
            "تم اختيار الاتجاه الأقوى "
            "حسب بيانات الشموع والمؤشرات."
        )

    return analysis


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

    direction = str(
        analysis.get(
            "direction",
            "UP"
        )
    ).upper()

    confidence = safe_float(
        analysis.get(
            "confidence",
            0
        )
    )

    up_score = safe_int(
        analysis.get(
            "up_score",
            0
        )
    )

    down_score = safe_int(
        analysis.get(
            "down_score",
            0
        )
    )

    reason = str(
        analysis.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:

        reason = (
            "تحليل مبني على حركة السعر "
            "والمؤشرات المتاحة."
        )

    entry_time = (
        get_next_entry_time(
            timeframe
        )
    )

    entry_price = candles[-1]["close"]

    if direction == "UP":

        cancellation = min(
            c["low"]
            for c in candles[-8:]
        )

        decision = "🟢 UP"

        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة "
            f"تحت {cancellation:.5f}"
        )

    else:

        cancellation = max(
            c["high"]
            for c in candles[-8:]
        )

        decision = "🔴 DOWN"

        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة "
            f"فوق {cancellation:.5f}"
        )

    if trade_type == "RECOVERY":

        mode_text = (
            "🔁 RECOVERY 1/1"
        )

    else:

        mode_text = (
            "🎯 BASE TRADE"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | {timeframe}\n"
        f"{mode_text}\n\n"
        f"{decision}\n"
        f"🎯 Confidence: {confidence:.0f}%\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏳ Entry after: "
        f"{ENTRY_DELAY_MINUTES} min\n"
        f"⏰ Entry Time: "
        f"{format_dt(entry_time)} "
        f"(Algiers)\n"
        f"💰 Entry Price: "
        f"{entry_price:.5f}\n"
        f"{cancel_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# COOLDOWN
# ============================================================

def signal_cooldown_active():

    global last_signal_sent_at

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
# SEND SIGNAL
# ============================================================

async def send_signal_safely(
    symbol,
    timeframe,
    message,
    trade_type,
):

    global last_signal_sent_at

    if telegram_application is None:

        logger.error(
            "Telegram application unavailable."
        )

        return False

    if not OWNER_ID:

        logger.error(
            "OWNER_ID missing."
        )

        return False

    with signal_send_lock:

        active, remaining = (
            signal_cooldown_active()
        )

        if active:

            logger.info(
                f"Signal blocked. "
                f"Remaining {remaining}s."
            )

            return False

        try:

            await telegram_application.bot.send_message(
                chat_id=int(OWNER_ID),
                text=message,
            )

            last_signal_sent_at = time.time()

            logger.info(
                f"SIGNAL SENT | "
                f"{symbol} {timeframe} | "
                f"{trade_type}"
            )

            return True

        except Exception as e:

            logger.exception(
                "Telegram send error: %s",
                e
            )

            return False


# ============================================================
# CHECK CYCLE
# ============================================================

def get_cycle():

    with cycle_lock:

        return dict(
            active_cycle
        )


def start_base_cycle(
    symbol,
    timeframe,
):

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

        active_cycle["trade_type"] = "RECOVERY"

        active_cycle["recovery_used"] = True

        active_cycle["last_trade_time"] = time.time()

        active_cycle["trade_number"] = 2


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

def auto_analyze_pair(
    symbol,
    timeframe,
):

    # ========================================================
    # مهم:
    # نمنع أزواج أخرى من التحليل والإرسال أثناء دورة نشطة.
    # ========================================================

    cycle = get_cycle()

    if cycle["active"]:

        active_symbol = cycle["symbol"]
        active_timeframe = cycle["timeframe"]

        if (
            symbol.upper()
            != str(active_symbol).upper()
            or timeframe.upper()
            != str(active_timeframe).upper()
        ):

            return

    key = (
        symbol.upper(),
        timeframe.upper()
    )

    now_ts = time.time()

    previous = last_auto_analysis.get(
        key,
        0
    )

    if (
        now_ts - previous
        < AUTO_ANALYSIS_INTERVAL_MINUTES * 60
    ):

        return

    last_auto_analysis[key] = now_ts

    # --------------------------------------------------------
    # إذا لا توجد دورة:
    # نختار زوجا واحدا فقط.
    # --------------------------------------------------------

    if not cycle["active"]:

        selected = choose_best_pair()

        if not selected:
            return

        selected_symbol = (
            selected["symbol"]
        )

        selected_tf = (
            selected["timeframe"]
        )

        if (
            symbol.upper()
            != selected_symbol.upper()
            or timeframe.upper()
            != selected_tf.upper()
        ):

            return

        symbol = selected_symbol
        timeframe = selected_tf

        trade_type = "BASE"

    else:

        symbol = cycle["symbol"]
        timeframe = cycle["timeframe"]

        trade_type = cycle["trade_type"]

    # --------------------------------------------------------
    # GET DATA
    # --------------------------------------------------------

    with mt4_lock:

        data = mt4_data.get(
            (
                symbol.upper(),
                timeframe.upper()
            )
        )

        if not data:
            return

        raw_candles = data.get(
            "candles",
            []
        )

    candles = normalize_candles(
        raw_candles
    )

    if len(candles) < (
        MIN_CLOSED_CANDLES + 1
    ):

        logger.info(
            f"Not enough candles: "
            f"{symbol} {timeframe} "
            f"{len(candles)}"
        )

        return

    closed = candles[:-1]

    if len(closed) < MIN_CLOSED_CANDLES:

        return

    # --------------------------------------------------------
    # GLOBAL 2-MINUTE COOLDOWN
    # --------------------------------------------------------

    active, remaining = (
        signal_cooldown_active()
    )

    if active:

        logger.info(
            f"Global cooldown: "
            f"{remaining}s"
        )

        return

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    recovery = (
        trade_type == "RECOVERY"
    )

    analysis = analyze_with_gemini(
        symbol,
        timeframe,
        closed,
        recovery=recovery,
    )

    if not analysis:
        return

    logger.info(
        "Gemini result: %s",
        json.dumps(
            analysis,
            ensure_ascii=False
        )
    )

    analysis = ensure_directional_signal(
        analysis,
        closed,
    )

    if not analysis:
        return

    # --------------------------------------------------------
    # MESSAGE
    # --------------------------------------------------------

    message = format_signal(
        symbol,
        timeframe,
        analysis,
        closed,
        trade_type,
    )

    # --------------------------------------------------------
    # SEND
    # --------------------------------------------------------

    if telegram_loop is None:
        return

    try:

        future = (
            asyncio.run_coroutine_threadsafe(
                send_signal_safely(
                    symbol,
                    timeframe,
                    message,
                    trade_type,
                ),
                telegram_loop,
            )
        )

        success = future.result(
            timeout=30
        )

        if success:

            if not cycle["active"]:

                start_base_cycle(
                    symbol,
                    timeframe,
                )

            elif (
                trade_type == "RECOVERY"
            ):

                with cycle_lock:

                    active_cycle[
                        "last_trade_time"
                    ] = time.time()

    except Exception as e:

        logger.exception(
            "Signal scheduling error: %s",
            e
        )


# ============================================================
# CLOSED CANDLE DETECTION
# ============================================================

def detect_new_closed_candle(
    symbol,
    timeframe,
    candles,
):

    key = (
        symbol.upper(),
        timeframe.upper()
    )

    if len(candles) < 2:
        return False

    closed = candles[:-1]

    if not closed:
        return False

    timestamp = closed[-1][
        "timestamp"
    ]

    previous = (
        last_processed_closed_candle.get(
            key
        )
    )

    if previous is None:

        last_processed_closed_candle[
            key
        ] = timestamp

        return False

    if timestamp > previous:

        last_processed_closed_candle[
            key
        ] = timestamp

        return True

    return False


# ============================================================
# RESULT SYSTEM
# ============================================================

async def handle_win():

    cycle = get_cycle()

    if not cycle["active"]:

        await telegram_application.bot.send_message(
            chat_id=int(OWNER_ID),
            text=(
                "ℹ️ لا توجد صفقة نشطة."
            ),
        )

        return

    trade_type = cycle["trade_type"]

    stats_data["wins"] += 1

    if trade_type == "BASE":

        stats_data["base_wins"] += 1

        result_text = (
            "🟢 BASE WIN"
        )

    else:

        stats_data["recovery_wins"] += 1

        result_text = (
            "🟢 RECOVERY WIN"
        )

    symbol = cycle["symbol"]

    reset_cycle()

    await telegram_application.bot.send_message(
        chat_id=int(OWNER_ID),
        text=(
            f"{result_text}\n"
            f"📊 {symbol}\n\n"
            "♻️ انتهت الدورة.\n"
            "🎯 البوت سيبحث عن زوج واحد "
            "جديد للإشارة القادمة."
        ),
    )


async def handle_loss():

    cycle = get_cycle()

    if not cycle["active"]:

        await telegram_application.bot.send_message(
            chat_id=int(OWNER_ID),
            text=(
                "ℹ️ لا توجد صفقة نشطة."
            ),
        )

        return

    trade_type = cycle["trade_type"]

    stats_data["losses"] += 1

    if trade_type == "BASE":

        stats_data["base_losses"] += 1

        # -----------------------------------------------
        # أول خسارة:
        # يسمح بتعويض واحد فقط
        # -----------------------------------------------

        with cycle_lock:

            if active_cycle[
                "recovery_used"
            ]:

                reset_cycle()

                await telegram_application.bot.send_message(
                    chat_id=int(OWNER_ID),
                    text=(
                        "🔴 LOSS\n"
                        "🛑 التعويض مستعمل مسبقا.\n"
                        "♻️ بدأت دورة جديدة."
                    ),
                )

                return

        start_recovery_cycle()

        await telegram_application.bot.send_message(
            chat_id=int(OWNER_ID),
            text=(
                "🔴 BASE LOSS\n"
                "━━━━━━━━━━━━━━━━━━\n"
                "🔁 يسمح الآن بتعويض واحد فقط.\n"
                "🛑 لا يوجد تعويض ثاني.\n"
                "⏱️ البوت سيحافظ على فارق "
                "دقيقتين بين الصفقات."
            ),
        )

    else:

        stats_data["recovery_losses"] += 1

        symbol = cycle["symbol"]

        # -----------------------------------------------
        # خسارة التعويض:
        # إيقاف الدورة
        # -----------------------------------------------

        reset_cycle()

        await telegram_application.bot.send_message(
            chat_id=int(OWNER_ID),
            text=(
                "🔴 RECOVERY LOSS\n"
                "━━━━━━━━━━━━━━━━━━\n"
                f"📊 {symbol}\n"
                "🛑 انتهت الدورة.\n"
                "🚫 ممنوع تعويض ثاني.\n"
                "♻️ سيبحث البوت عن زوج جديد "
                "للدورة القادمة."
            ),
        )


# ============================================================
# TELEGRAM COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Bot is running.\n\n"
        "📊 Analysis: H1\n"
        "🔎 One pair only\n"
        "⏱️ Analysis: every 1 min\n"
        "⏳ Entry delay: 2 min\n"
        "🛑 Between trades: 2 min\n"
        "🔁 Recovery: maximum 1\n\n"
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
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    wins = stats_data["wins"]
    losses = stats_data["losses"]

    total = wins + losses

    winrate = (
        (wins / total) * 100
        if total
        else 0
    )

    cycle = get_cycle()

    if cycle["active"]:

        current = (
            f"{cycle['symbol']} | "
            f"{cycle['timeframe']} | "
            f"{cycle['trade_type']}"
        )

    else:

        current = "لا توجد دورة نشطة"

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {winrate:.1f}%\n\n"
        f"🟢 Base Wins: "
        f"{stats_data['base_wins']}\n"
        f"🔴 Base Losses: "
        f"{stats_data['base_losses']}\n"
        f"🟢 Recovery Wins: "
        f"{stats_data['recovery_wins']}\n"
        f"🔴 Recovery Losses: "
        f"{stats_data['recovery_losses']}\n\n"
        f"📌 Current: {current}"
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await handle_win()


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await handle_loss()


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    reset_cycle()

    stats_data["wins"] = 0
    stats_data["losses"] = 0

    stats_data["base_wins"] = 0
    stats_data["base_losses"] = 0

    stats_data["recovery_wins"] = 0
    stats_data["recovery_losses"] = 0

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات.\n"
        "♻️ تم تصفير دورة التداول.\n\n"
        "🎯 البوت جاهز لاختيار زوج واحد."
    )


# ============================================================
# MT4 STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    with mt4_lock:

        if not mt4_data:

            await update.message.reply_text(
                "📡 MT4\n\n"
                "🔴 لا توجد بيانات."
            )

            return

        lines = [
            "📡 MT4 STATUS",
            "━━━━━━━━━━━━━━━━━━",
        ]

        for key, data in mt4_data.items():

            symbol = data.get(
                "symbol",
                "?"
            )

            timeframe = data.get(
                "timeframe",
                "?"
            )

            candles = len(
                data.get(
                    "candles",
                    []
                )
            )

            lines.append(
                f"🟢 {symbol} | "
                f"{timeframe} | "
                f"{candles} candles"
            )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# MANUAL ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    cycle = get_cycle()

    if cycle["active"]:

        await update.message.reply_text(
            "📌 توجد دورة نشطة حاليا.\n"
            f"📊 الزوج: {cycle['symbol']}\n"
            f"🔁 النوع: {cycle['trade_type']}\n\n"
            "لن يتم تحليل أزواج أخرى."
        )

        return

    selected = choose_best_pair()

    if not selected:

        await update.message.reply_text(
            "❌ لم أجد زوج H1 صالحا "
            "بـ40 شمعة مغلقة."
        )

        return

    symbol = selected["symbol"]
    timeframe = selected["timeframe"]

    await update.message.reply_text(
        f"🔎 تحليل زوج واحد فقط:\n"
        f"📊 {symbol} | {timeframe}"
    )

    auto_analyze_pair(
        symbol,
        timeframe,
    )


# ============================================================
# PHOTO
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not GEMINI_API_KEY:

        await update.message.reply_text(
            "❌ GEMINI_API_KEY is missing."
        )

        return

    try:

        photo = update.message.photo[-1]

        file = await photo.get_file()

        image_bytes = (
            await file.download_as_bytearray()
        )

        client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        prompt = """
Analyze this trading chart.

Return exactly one direction:
UP or DOWN.

Do not return WAIT,
NO SIGNAL or NEUTRAL.

Use price action, structure,
momentum, EMA 9/21, RSI,
Williams %R, ADX/DI,
Keltner and candles.

Do not invent values.
"""

        response = client.models.generate_content(
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
                temperature=0.10
            ),
        )

        await update.message.reply_text(
            response.text
        )

    except Exception as e:

        logger.exception(
            "Photo analysis error: %s",
            e
        )

        await update.message.reply_text(
            "❌ Image analysis failed."
        )


# ============================================================
# HTTP HEALTH SERVER
# ============================================================

class HealthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
        ):

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.end_headers()

            self.wfile.write(
                b"ZinoProSignalAI is running"
            )

            return

        self.send_response(404)
        self.end_headers()

    def log_message(
        self,
        format,
        *args
    ):

        return


def start_http_server():

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
    )

    logger.info(
        f"HTTP server listening on {PORT}"
    )

    server.serve_forever()


# ============================================================
# MT4 HTTP
# ============================================================

class MT4Handler(
    BaseHTTPRequestHandler
):

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        if parsed.path != "/mt4":

            self.send_response(404)
            self.end_headers()

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                body.decode(
                    "utf-8"
                )
            )

        except Exception as e:

            logger.error(
                "Invalid MT4 JSON: %s",
                e
            )

            self.send_response(400)
            self.end_headers()

            return

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        received_key = (
            self.headers.get(
                "X-MT4-API-KEY",
                ""
            ).strip()
        )

        if not received_key:

            received_key = str(
                payload.get(
                    "api_key",
                    ""
                )
            ).strip()

        if (
            MT4_API_KEY
            and received_key != MT4_API_KEY
        ):

            logger.warning(
                "Invalid MT4 API key."
            )

            self.send_response(401)
            self.end_headers()

            return

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

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
        ).strip()

        candles = payload.get(
            "candles",
            []
        )

        if not symbol or not timeframe:

            self.send_response(400)
            self.end_headers()

            return

        normalized = normalize_candles(
            candles
        )

        if not normalized:

            self.send_response(400)
            self.end_headers()

            return

        key = (
            symbol.upper(),
            timeframe.upper()
        )

        with mt4_lock:

            mt4_data[key] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": normalized,
                "received_at": (
                    now_algiers().isoformat()
                ),
            }

        is_new_candle = (
            detect_new_closed_candle(
                symbol,
                timeframe,
                normalized,
            )
        )

        logger.info(
            f"MT4 DATA | "
            f"{symbol} {timeframe} | "
            f"{len(normalized)} candles | "
            f"new={is_new_candle}"
        )

        # ----------------------------------------------------
        # AUTO ANALYSIS
        #
        # لا ننتظر إغلاق شمعة H1.
        # إذا MT4 يرسل البيانات كل دقيقة،
        # البوت يستطيع إعادة التحليل كل دقيقة.
        # ----------------------------------------------------

        threading.Thread(
            target=auto_analyze_pair,
            args=(
                symbol,
                timeframe,
            ),
            daemon=True,
        ).start()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.end_headers()

        response = {
            "ok": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": len(normalized),
            "new_closed_candle": is_new_candle,
        }

        self.wfile.write(
            json.dumps(
                response
            ).encode(
                "utf-8"
            )
        )

    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_application
    global telegram_loop

    logger.info(
        "================================="
    )

    logger.info(
        "ZinoProSignalAI STARTING"
    )

    logger.info(
        f"Gemini: {GEMINI_MODEL}"
    )

    logger.info(
        "Analysis interval: 1 minute"
    )

    logger.info(
        "Entry delay: 2 minutes"
    )

    logger.info(
        "Signal cooldown: 120 seconds"
    )

    logger.info(
        "ONE PAIR MODE: ENABLED"
    )

    logger.info(
        "ONE RECOVERY MAX: ENABLED"
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "================================="
    )

    # ========================================================
    # HTTP
    # ========================================================

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    # ========================================================
    # TELEGRAM
    # ========================================================

    if not BOT_TOKEN:

        logger.error(
            "BOT_TOKEN missing."
        )

        return

    if not OWNER_ID:

        logger.error(
            "OWNER_ID missing."
        )

        return

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_application = application

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

    logger.info(
        "Starting Telegram polling..."
    )

    async def run_bot():

        global telegram_loop

        telegram_loop = (
            asyncio.get_running_loop()
        )

        await application.initialize()

        await application.start()

        await application.updater.start_polling(
            drop_pending_updates=True
        )

        logger.info(
            "Telegram bot is running."
        )

        try:

            while True:

                await asyncio.sleep(
                    3600
                )

        finally:

            await application.updater.stop()

            await application.stop()

            await application.shutdown()

    asyncio.run(
        run_bot()
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
