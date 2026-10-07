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

# ============================================================
# ZinoProSignalAI
# MT4 / MT5 -> Render -> Telegram
# ============================================================

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()

API_KEY = os.getenv("API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

# Gemini optional
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

TELEGRAM_CHAT_ID = OWNER_ID_RAW

try:
    OWNER_ID = int(OWNER_ID_RAW)
except Exception:
    OWNER_ID = 0

# Algeria timezone
ALGIERS_TZ = ZoneInfo("Africa/Algiers")

# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")

# ------------------------------------------------------------
# OPTIONAL GEMINI
# ------------------------------------------------------------

gemini_client = None

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types

        gemini_client = genai.Client(api_key=GEMINI_API_KEY)

        logger.info(
            "Gemini enabled | model=%s",
            GEMINI_MODEL
        )

    except Exception as exc:
        logger.warning(
            "Gemini unavailable: %s",
            exc
        )
        gemini_client = None
else:
    logger.info("Gemini disabled - deterministic analysis mode")

# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------

state_lock = threading.Lock()

stats = {
    "wins": 0,
    "losses": 0,
    "signals": 0,
}

last_signal = None

signal_history = []

# Maximum history kept in memory
MAX_HISTORY = 100

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def format_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default

        if isinstance(value, bool):
            return default

        if isinstance(value, str):
            value = value.strip().replace(",", ".")

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def normalize_direction(value):
    if value is None:
        return ""

    value = str(value).upper().strip()

    mapping = {
        "CALL": "UP",
        "BUY": "UP",
        "LONG": "UP",
        "UP": "UP",

        "PUT": "DOWN",
        "SELL": "DOWN",
        "SHORT": "DOWN",
        "DOWN": "DOWN",
    }

    return mapping.get(value, "")


def normalize_timeframe(value):
    if value is None:
        return "M1"

    text = str(value).upper().strip()

    replacements = {
        "1": "M1",
        "1M": "M1",
        "M01": "M1",

        "2": "M2",
        "2M": "M2",
        "M02": "M2",

        "3": "M3",
        "3M": "M3",
        "M03": "M3",

        "5": "M5",
        "5M": "M5",
        "M05": "M5",

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

    return replacements.get(text, text)


def timeframe_minutes(timeframe):
    tf = normalize_timeframe(timeframe)

    table = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
    }

    return table.get(tf, 1)


def extract_value(data, *names, default=None):
    """
    Flexible field reader.
    Supports:
        data["rsi"]
        data["RSI"]
        data["indicators"]["rsi"]
    """

    if not isinstance(data, dict):
        return default

    indicators = data.get("indicators")

    for name in names:
        if name in data:
            return data[name]

        if isinstance(indicators, dict):
            if name in indicators:
                return indicators[name]

    # case-insensitive
    lower_map = {
        str(k).lower(): v
        for k, v in data.items()
    }

    for name in names:
        if str(name).lower() in lower_map:
            return lower_map[str(name).lower()]

    if isinstance(indicators, dict):
        lower_indicators = {
            str(k).lower(): v
            for k, v in indicators.items()
        }

        for name in names:
            if str(name).lower() in lower_indicators:
                return lower_indicators[str(name).lower()]

    return default


def get_nested(data, *paths, default=None):
    for path in paths:
        current = data

        try:
            for key in path:
                if isinstance(current, dict):
                    if key in current:
                        current = current[key]
                    else:
                        current = None
                        break
                else:
                    current = None
                    break

            if current is not None:
                return current

        except Exception:
            pass

    return default


# ============================================================
# DETERMINISTIC ANALYSIS
# ============================================================

def analyze_market(data):
    """
    Main technical analysis.

    Maximum score:
        UP   /20
        DOWN /20

    Priority:
        Price Action
        Structure
        Breakout / Retest
        Liquidity
        Momentum
        Candle
        EMA 9/21
        RSI
        Williams %R
        Keltner
        ADX / DI
    """

    symbol = str(
        extract_value(
            data,
            "symbol",
            "asset",
            "pair",
            "instrument",
            default="UNKNOWN"
        )
    ).upper()

    timeframe = normalize_timeframe(
        extract_value(
            data,
            "timeframe",
            "tf",
            "period",
            default="M1"
        )
    )

    price = safe_float(
        extract_value(
            data,
            "price",
            "bid",
            "close",
            "entry_price",
            default=0
        )
    )

    # --------------------------------------------------------
    # Raw indicators
    # --------------------------------------------------------

    ema9 = safe_float(
        extract_value(
            data,
            "ema9",
            "ema_9",
            "EMA9",
            default=0
        )
    )

    ema21 = safe_float(
        extract_value(
            data,
            "ema21",
            "ema_21",
            "EMA21",
            default=0
        )
    )

    rsi = safe_float(
        extract_value(
            data,
            "rsi",
            "RSI",
            default=50
        ),
        50
    )

    williams = safe_float(
        extract_value(
            data,
            "williams",
            "williams_r",
            "wpr",
            "WPR",
            default=-50
        ),
        -50
    )

    adx = safe_float(
        extract_value(
            data,
            "adx",
            "ADX",
            default=0
        )
    )

    plus_di = safe_float(
        extract_value(
            data,
            "plus_di",
            "di_plus",
            "plusDI",
            "DIPlus",
            default=0
        )
    )

    minus_di = safe_float(
        extract_value(
            data,
            "minus_di",
            "di_minus",
            "minusDI",
            "DIMinus",
            default=0
        )
    )

    keltner_mid = safe_float(
        extract_value(
            data,
            "keltner_mid",
            "kc_mid",
            "keltner_ema",
            default=0
        )
    )

    keltner_upper = safe_float(
        extract_value(
            data,
            "keltner_upper",
            "kc_upper",
            default=0
        )
    )

    keltner_lower = safe_float(
        extract_value(
            data,
            "keltner_lower",
            "kc_lower",
            default=0
        )
    )

    # --------------------------------------------------------
    # Explicit signals from MT4/MT5
    # --------------------------------------------------------

    structure = str(
        extract_value(
            data,
            "structure",
            "market_structure",
            default=""
        )
    ).upper()

    breakout = str(
        extract_value(
            data,
            "breakout",
            "breakout_direction",
            default=""
        )
    ).upper()

    retest = str(
        extract_value(
            data,
            "retest",
            "retest_direction",
            default=""
        )
    ).upper()

    liquidity = str(
        extract_value(
            data,
            "liquidity",
            "liquidity_direction",
            default=""
        )
    ).upper()

    momentum = str(
        extract_value(
            data,
            "momentum",
            "momentum_direction",
            default=""
        )
    ).upper()

    candle = str(
        extract_value(
            data,
            "candle",
            "candle_direction",
            "last_candle",
            default=""
        )
    ).upper()

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    up = 0
    down = 0

    reasons_up = []
    reasons_down = []

    # --------------------------------------------------------
    # 1. Structure / Price Action - 4 points
    # --------------------------------------------------------

    structure_up_words = (
        "BULLISH",
        "UP",
        "HH",
        "HL",
        "BOS_UP",
        "CHOCH_UP",
        "BUY",
    )

    structure_down_words = (
        "BEARISH",
        "DOWN",
        "LH",
        "LL",
        "BOS_DOWN",
        "CHOCH_DOWN",
        "SELL",
    )

    if any(x in structure for x in structure_up_words):
        up += 4
        reasons_up.append("Bullish structure")

    elif any(x in structure for x in structure_down_words):
        down += 4
        reasons_down.append("Bearish structure")

    # --------------------------------------------------------
    # 2. Breakout - 3 points
    # --------------------------------------------------------

    if any(x in breakout for x in (
        "UP",
        "BULLISH",
        "BUY",
        "BREAK_UP",
    )):
        up += 3
        reasons_up.append("Bullish breakout")

    elif any(x in breakout for x in (
        "DOWN",
        "BEARISH",
        "SELL",
        "BREAK_DOWN",
    )):
        down += 3
        reasons_down.append("Bearish breakout")

    # Retest
    if any(x in retest for x in (
        "UP",
        "BULLISH",
        "BUY",
    )):
        up += 1
        reasons_up.append("Bullish retest")

    elif any(x in retest for x in (
        "DOWN",
        "BEARISH",
        "SELL",
    )):
        down += 1
        reasons_down.append("Bearish retest")

    # --------------------------------------------------------
    # 3. Liquidity - 2 points
    # --------------------------------------------------------

    if any(x in liquidity for x in (
        "UP",
        "BULLISH",
        "BUY",
        "LOWER",
        "SELL_SIDE",
        "SSL",
    )):
        up += 2
        reasons_up.append("Liquidity supports UP")

    elif any(x in liquidity for x in (
        "DOWN",
        "BEARISH",
        "SELL",
        "UPPER",
        "BUY_SIDE",
        "BSL",
    )):
        down += 2
        reasons_down.append("Liquidity supports DOWN")

    # --------------------------------------------------------
    # 4. Momentum - 2 points
    # --------------------------------------------------------

    if any(x in momentum for x in (
        "UP",
        "BULLISH",
        "BUY",
        "STRONG_UP",
    )):
        up += 2
        reasons_up.append("Bullish momentum")

    elif any(x in momentum for x in (
        "DOWN",
        "BEARISH",
        "SELL",
        "STRONG_DOWN",
    )):
        down += 2
        reasons_down.append("Bearish momentum")

    # --------------------------------------------------------
    # 5. Candle - 2 points
    # --------------------------------------------------------

    if any(x in candle for x in (
        "UP",
        "BULLISH",
        "BUY",
        "GREEN",
        "HAMMER",
        "ENGULFING_BULL",
    )):
        up += 2
        reasons_up.append("Bullish candle")

    elif any(x in candle for x in (
        "DOWN",
        "BEARISH",
        "SELL",
        "RED",
        "SHOOTING",
        "ENGULFING_BEAR",
    )):
        down += 2
        reasons_down.append("Bearish candle")

    # --------------------------------------------------------
    # 6. EMA 9/21 - 2 points
    # --------------------------------------------------------

    if ema9 > 0 and ema21 > 0:

        if ema9 > ema21:
            up += 2
            reasons_up.append("EMA 9 > EMA 21")

        elif ema9 < ema21:
            down += 2
            reasons_down.append("EMA 9 < EMA 21")

    # --------------------------------------------------------
    # 7. RSI - 1 point
    # --------------------------------------------------------

    if rsi > 55 and rsi < 75:
        up += 1
        reasons_up.append("RSI bullish")

    elif rsi < 45 and rsi > 25:
        down += 1
        reasons_down.append("RSI bearish")

    # --------------------------------------------------------
    # 8. Williams %R - 1 point
    # --------------------------------------------------------

    if williams > -50 and williams < -5:
        up += 1
        reasons_up.append("Williams %R bullish")

    elif williams < -50 and williams > -95:
        down += 1
        reasons_down.append("Williams %R bearish")

    # --------------------------------------------------------
    # 9. Keltner - 1 point
    # --------------------------------------------------------

    if price > 0 and keltner_mid > 0:

        if price > keltner_mid:
            up += 1
            reasons_up.append("Price above Keltner mid")

        elif price < keltner_mid:
            down += 1
            reasons_down.append("Price below Keltner mid")

    # --------------------------------------------------------
    # 10. ADX / DI - 1 point
    # --------------------------------------------------------

    if adx >= 18:

        if plus_di > minus_di:
            up += 1
            reasons_up.append("ADX/DI bullish")

        elif minus_di > plus_di:
            down += 1
            reasons_down.append("ADX/DI bearish")

    # --------------------------------------------------------
    # Clamp
    # --------------------------------------------------------

    up = min(up, 20)
    down = min(down, 20)

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    if up > down:
        direction = "UP"
        winning_score = up
        losing_score = down
        reasons = reasons_up

    elif down > up:
        direction = "DOWN"
        winning_score = down
        losing_score = up
        reasons = reasons_down

    else:
        # No WAIT allowed.
        # If tied, use EMA/RSI/price fallback.
        if ema9 > ema21 or rsi >= 50 or price > keltner_mid:
            direction = "UP"
            winning_score = up
            losing_score = down
            reasons = reasons_up
        else:
            direction = "DOWN"
            winning_score = down
            losing_score = up
            reasons = reasons_down

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    difference = abs(up - down)

    confidence = 50 + (difference * 3)

    if winning_score >= 16:
        confidence += 5

    elif winning_score >= 14:
        confidence += 3

    elif winning_score <= 8:
        confidence -= 5

    # Do not blindly output unrealistic 90%+
    confidence = max(55, min(89, confidence))

    # --------------------------------------------------------
    # Weak / contradictory filter
    # --------------------------------------------------------

    contradiction = False

    if winning_score <= 9:
        contradiction = True

    if difference <= 2:
        contradiction = True

    # Still UP/DOWN only.
    # In weak situations choose direction but lower confidence.
    if contradiction:
        confidence = min(confidence, 62)

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    minutes = timeframe_minutes(timeframe)

    signal_created = now_algiers()

    entry_time = signal_created + timedelta(minutes=minutes)

    entry_price = price

    # --------------------------------------------------------
    # Cancellation
    # --------------------------------------------------------

    # Approximate cancellation based on price precision.
    # EA can provide exact cancellation_level.
    supplied_cancel = safe_float(
        extract_value(
            data,
            "cancellation_level",
            "cancel_level",
            "invalidation",
            default=0
        )
    )

    if supplied_cancel > 0:
        cancellation = supplied_cancel

    else:
        digits = safe_int(
            extract_value(
                data,
                "digits",
                "precision",
                default=5
            ),
            5
        )

        point = 10 ** (-digits)

        # Small protective distance.
        # This is only fallback when EA doesn't provide one.
        distance = point * 20

        if direction == "UP":
            cancellation = price - distance
        else:
            cancellation = price + distance

        cancellation = round(cancellation, digits)

    # --------------------------------------------------------
    # Reason
    # --------------------------------------------------------

    if reasons:
        reason = " • ".join(reasons[:4])
    else:
        reason = "Technical momentum bias"

    if direction == "UP":
        cancel_text = (
            f"إلغاء إذا أغلقت شمعة تحت "
            f"{cancellation}"
        )
    else:
        cancel_text = (
            f"إلغاء إذا أغلقت شمعة فوق "
            f"{cancellation}"
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": int(confidence),
        "up_score": int(up),
        "down_score": int(down),
        "entry_price": entry_price,
        "cancellation": cancellation,
        "entry_after": minutes,
        "entry_time": entry_time,
        "created_at": signal_created,
        "reason": reason,
        "cancel_text": cancel_text,
        "weak": contradiction,
        "raw": data,
    }


# ============================================================
# GEMINI OPTIONAL VALIDATION
# ============================================================

def gemini_validate(analysis):
    """
    Optional AI validation.

    Important:
    The final signal remains UP/DOWN.
    Gemini cannot return WAIT.
    """

    if gemini_client is None:
        return analysis

    try:

        prompt = f"""
You are a strict binary-options technical-analysis validator.

You MUST choose exactly UP or DOWN.

Never output:
WAIT
NO SIGNAL
NEUTRAL

Asset: {analysis["symbol"]}
Timeframe: {analysis["timeframe"]}

Current deterministic analysis:

Direction:
{analysis["direction"]}

UP Score:
{analysis["up_score"]}/20

DOWN Score:
{analysis["down_score"]}/20

Confidence:
{analysis["confidence"]}%

Reason:
{analysis["reason"]}

Raw MT4/MT5 data:
{json.dumps(analysis["raw"], ensure_ascii=False)}

Rules:

1. Price action has highest priority.
2. Market structure is more important than oscillators.
3. Breakout/retest is important.
4. Liquidity matters.
5. Momentum matters.
6. Candle confirmation matters.
7. EMA 9/21 is secondary.
8. RSI is secondary.
9. Williams %R is secondary.
10. Keltner and ADX/DI are confirmation only.
11. Never invent indicators.
12. Never invent market data.
13. Do not automatically favor UP.
14. Do not automatically favor DOWN.
15. Avoid confidence above 89%.
16. If evidence is weak, keep the stronger direction but reduce confidence.
17. Do not reverse a strong 16+/20 deterministic signal unless there is clear contradiction.

Return ONLY valid JSON:

{{
  "direction": "UP",
  "confidence": 70,
  "reason": "short technical reason"
}}
"""

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        text = response.text.strip()

        result = json.loads(text)

        direction = normalize_direction(
            result.get("direction")
        )

        if direction not in ("UP", "DOWN"):
            return analysis

        confidence = safe_int(
            result.get("confidence"),
            analysis["confidence"]
        )

        confidence = max(55, min(89, confidence))

        # Do not allow AI to reverse very strong deterministic signals
        if (
            analysis["up_score"] >= 16
            and direction == "DOWN"
        ):
            direction = "UP"
            confidence = min(confidence, analysis["confidence"])

        elif (
            analysis["down_score"] >= 16
            and direction == "UP"
        ):
            direction = "DOWN"
            confidence = min(confidence, analysis["confidence"])

        reason = str(
            result.get(
                "reason",
                analysis["reason"]
            )
        ).strip()

        analysis["direction"] = direction
        analysis["confidence"] = confidence
        analysis["reason"] = reason[:300]

        return analysis

    except Exception as exc:

        logger.warning(
            "Gemini validation failed: %s",
            exc
        )

        return analysis


# ============================================================
# TELEGRAM CARD
# ============================================================

def build_signal_message(a):
    direction = a["direction"]

    if direction == "UP":
        emoji = "🟢"
        action = "UP"
        cancel_word = "تحت"
    else:
        emoji = "🔴"
        action = "DOWN"
        cancel_word = "فوق"

    entry_time = a["entry_time"].strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    price = a["entry_price"]

    if price:
        price_text = f"{price:.8f}".rstrip("0").rstrip(".")
    else:
        price_text = "N/A"

    cancellation = a["cancellation"]

    if cancellation:
        cancel_text = (
            f"{cancellation:.8f}"
            .rstrip("0")
            .rstrip(".")
        )
    else:
        cancel_text = "N/A"

    recovery_text = ""

    with state_lock:
        if signal_history:
            last = signal_history[-1]

            if last.get("result") == "LOSS":
                recovery_text = "\n🔁 RECOVERY 1/1\n"

    message = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {a['symbol']} | {a['timeframe']}\n"
        f"{recovery_text}"
        f"{emoji} {action}\n\n"
        f"🔥 Confidence: {a['confidence']}%\n"
        f"🟢 UP Score: {a['up_score']}/20\n"
        f"🔴 DOWN Score: {a['down_score']}/20\n\n"
        f"⏱️ Entry after: {a['entry_after']} minute"
        f"{'s' if a['entry_after'] != 1 else ''}\n"
        f"🕐 ENTRY TIME: {entry_time} (Algiers)\n"
        f"💰 Entry Price: {price_text}\n"
        f"⚠️ Cancellation: {cancel_text}\n"
        f"   إلغاء إذا أغلقت شمعة {cancel_word} {cancel_text}\n\n"
        f"🧠 Reason: {a['reason']}"
    )

    return message


# ============================================================
# SEND TELEGRAM
# ============================================================

async def send_signal_to_owner(application, analysis):
    global last_signal

    message = build_signal_message(analysis)

    if OWNER_ID <= 0:
        logger.error("OWNER_ID is not configured")
        return

    try:

        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=message,
        )

        with state_lock:

            last_signal = analysis

            stats["signals"] += 1

            record = {
                "symbol": analysis["symbol"],
                "timeframe": analysis["timeframe"],
                "direction": analysis["direction"],
                "confidence": analysis["confidence"],
                "up_score": analysis["up_score"],
                "down_score": analysis["down_score"],
                "entry_price": analysis["entry_price"],
                "entry_time": analysis["entry_time"].isoformat(),
                "created_at": analysis["created_at"].isoformat(),
                "result": None,
            }

            signal_history.append(record)

            if len(signal_history) > MAX_HISTORY:
                del signal_history[:-MAX_HISTORY]

        logger.info(
            "Signal sent | %s | %s | %s%%",
            analysis["symbol"],
            analysis["direction"],
            analysis["confidence"],
        )

    except Exception as exc:
        logger.exception(
            "Failed to send Telegram signal: %s",
            exc
        )


# ============================================================
# MT4 / MT5 DATA PROCESSING
# ============================================================

def process_mt_data(payload, application):

    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")

    analysis = analyze_market(payload)

    # Optional AI validation
    if GEMINI_API_KEY:
        analysis = gemini_validate(analysis)

    # Run Telegram send in its own event loop
    def runner():

        try:
            asyncio.run(
                send_signal_to_owner(
                    application,
                    analysis
                )
            )

        except Exception as exc:
            logger.exception(
                "Telegram async runner error: %s",
                exc
            )

    threading.Thread(
        target=runner,
        daemon=True,
    ).start()

    return analysis


# ============================================================
# HTTP SERVER
# ============================================================

telegram_application = None


class HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Avoid noisy Render HTTP logs
        logger.info(
            "HTTP | " + format,
            *args
        )

    def send_json(
        self,
        status,
        payload
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

    def read_json_body(self):

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        if content_length <= 0:
            raise ValueError(
                "Empty request body"
            )

        # Safety limit: 1 MB
        if content_length > 1024 * 1024:
            raise ValueError(
                "Request body too large"
            )

        raw = self.rfile.read(
            content_length
        )

        if not raw:
            raise ValueError(
                "Empty request body"
            )

        try:
            return json.loads(
                raw.decode("utf-8")
            )

        except Exception:
            raise ValueError(
                "Invalid JSON"
            )

    def authorized(self):

        # If API_KEY is not configured,
        # endpoint remains open for compatibility.
        if not API_KEY:
            return True

        received = (
            self.headers.get("X-API-Key", "")
            or self.headers.get("Authorization", "")
        ).strip()

        if received.startswith("Bearer "):
            received = received[7:].strip()

        return received == API_KEY

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path.rstrip("/")

        if path in ("", "/", "/health"):

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "time": format_dt(
                        now_algiers()
                    ),
                }
            )

            return

        if path == "/stats":

            with state_lock:
                data = dict(stats)

            total = (
                data["wins"]
                + data["losses"]
            )

            winrate = (
                (data["wins"] / total) * 100
                if total
                else 0
            )

            self.send_json(
                200,
                {
                    **data,
                    "winrate": round(
                        winrate,
                        2
                    ),
                }
            )

            return

        self.send_json(
            404,
            {
                "error": "Not found"
            }
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path.rstrip("/")

        # ----------------------------------------------------
        # MT4 / MT5
        # ----------------------------------------------------

        if path == "/mt4":

            if not self.authorized():

                self.send_json(
                    401,
                    {
                        "error": "Unauthorized"
                    }
                )

                return

            try:

                payload = (
                    self.read_json_body()
                )

                analysis = process_mt_data(
                    payload,
                    telegram_application
                )

                self.send_json(
                    200,
                    {
                        "status": "signal_sent",
                        "symbol": analysis["symbol"],
                        "timeframe": analysis["timeframe"],
                        "direction": analysis["direction"],
                        "confidence": analysis["confidence"],
                        "up_score": analysis["up_score"],
                        "down_score": analysis["down_score"],
                        "entry_after": analysis["entry_after"],
                        "entry_time": analysis["entry_time"].isoformat(),
                    }
                )

            except Exception as exc:

                logger.exception(
                    "MT4 endpoint error"
                )

                self.send_json(
                    400,
                    {
                        "status": "error",
                        "error": str(exc),
                    }
                )

            return

        # Alias for MT5
        if path == "/mt5":

            if not self.authorized():

                self.send_json(
                    401,
                    {
                        "error": "Unauthorized"
                    }
                )

                return

            try:

                payload = (
                    self.read_json_body()
                )

                analysis = process_mt_data(
                    payload,
                    telegram_application
                )

                self.send_json(
                    200,
                    {
                        "status": "signal_sent",
                        "symbol": analysis["symbol"],
                        "timeframe": analysis["timeframe"],
                        "direction": analysis["direction"],
                        "confidence": analysis["confidence"],
                        "up_score": analysis["up_score"],
                        "down_score": analysis["down_score"],
                        "entry_after": analysis["entry_after"],
                        "entry_time": analysis["entry_time"].isoformat(),
                    }
                )

            except Exception as exc:

                logger.exception(
                    "MT5 endpoint error"
                )

                self.send_json(
                    400,
                    {
                        "status": "error",
                        "error": str(exc),
                    }
                )

            return

        self.send_json(
            404,
            {
                "error": "Not found"
            }
        )


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
# TELEGRAM COMMANDS
# ============================================================

def owner_only(func):

    async def wrapper(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ):

        user = update.effective_user

        if not user:
            return

        if user.id != OWNER_ID:

            if update.message:
                await update.message.reply_text(
                    "⛔ Unauthorized."
                )

            return

        return await func(
            update,
            context
        )

    return wrapper


@owner_only
async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ Bot is online.\n"
        "📡 MT4/MT5 endpoint: /mt4\n"
        "📊 /stats\n"
        "🟢 /win\n"
        "🔴 /loss\n"
        "♻️ /reset"
    )


@owner_only
async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    with state_lock:
        wins = stats["wins"]
        losses = stats["losses"]
        signals = stats["signals"]

    total_results = wins + losses

    winrate = (
        wins / total_results * 100
        if total_results
        else 0
    )

    await update.message.reply_text(
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📡 Signals: {signals}\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"🎯 Winrate: {winrate:.2f}%"
    )


@owner_only
async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    with state_lock:

        stats["wins"] += 1

        if signal_history:
            signal_history[-1]["result"] = "WIN"

    await update.message.reply_text(
        "🟢 WIN registered."
    )


@owner_only
async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    with state_lock:

        stats["losses"] += 1

        if signal_history:
            signal_history[-1]["result"] = "LOSS"

    await update.message.reply_text(
        "🔴 LOSS registered.\n"
        "🔁 Next signal can use RECOVERY 1/1."
    )


@owner_only
async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    with state_lock:

        stats["wins"] = 0
        stats["losses"] = 0
        stats["signals"] = 0

        signal_history.clear()

    await update.message.reply_text(
        "♻️ Statistics reset."
    )


@owner_only
async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    with state_lock:
        history = list(
            signal_history[-10:]
        )

    if not history:

        await update.message.reply_text(
            "📚 History is empty."
        )

        return

    lines = [
        "📚 ZinoProSignalAI HISTORY",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in reversed(history):

        result = item.get("result")

        if result == "WIN":
            result_icon = "🟢"
        elif result == "LOSS":
            result_icon = "🔴"
        else:
            result_icon = "⚪"

        lines.append(
            f"{result_icon} "
            f"{item['symbol']} "
            f"{item['timeframe']} | "
            f"{item['direction']} | "
            f"{item['confidence']}%"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# MAIN
# ============================================================

def validate_config():

    errors = []

    if not BOT_TOKEN:
        errors.append(
            "BOT_TOKEN is missing"
        )

    if OWNER_ID <= 0:
        errors.append(
            "OWNER_ID is missing or invalid"
        )

    if errors:

        for error in errors:
            logger.error(error)

        raise RuntimeError(
            " | ".join(errors)
        )


async def post_init(application):

    global telegram_application

    telegram_application = application

    logger.info(
        "Telegram application initialized"
    )


def main():

    validate_config()

    logger.info(
        "Starting ZinoProSignalAI..."
    )

    logger.info(
        "OWNER_ID=%s",
        OWNER_ID
    )

    logger.info(
        "PORT=%s",
        PORT
    )

    # --------------------------------------------------------
    # HTTP server FIRST
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="HTTP-Server",
    )

    http_thread.start()

    # --------------------------------------------------------
    # Telegram
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

    logger.info(
        "Telegram polling starting..."
    )

    # Only ONE Telegram polling instance
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )


if __name__ == "__main__":
    main()
