import os
import json
import logging
import threading
import asyncio
import time
import re
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()

OWNER_ID_RAW = os.getenv("OWNER_ID", "").strip()
PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

# One signal per cycle
CYCLE_MINUTES = 3

# Only these timeframes are candidates for automatic signals
AUTO_TIMEFRAMES = {"M1", "M3"}

# MT4 data freshness
MAX_DATA_AGE_SECONDS = 180

# Minimum number of closed candles required
MIN_CLOSED_CANDLES = 40

# Minimum lead time before entry
MIN_ENTRY_LEAD_SECONDS = 20

# Persistent runtime storage
DATA_FILE = Path(os.getenv("MT4_DATA_FILE", "mt4_data.json"))

# Do not allow excessive Gemini confidence
MAX_CONFIDENCE = 89

# Score must have a meaningful edge
MIN_SELECTED_SCORE = 11
MIN_SCORE_DIFFERENCE = 5
MIN_CONFIDENCE = 70


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not MT4_API_KEY:
    raise RuntimeError("MT4_API_KEY is missing")

if not OWNER_ID_RAW:
    raise RuntimeError("OWNER_ID is missing")

try:
    OWNER_ID = int(OWNER_ID_RAW)
except ValueError:
    raise RuntimeError("OWNER_ID must be an integer")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.RLock()

mt4_data: Dict[str, Dict[str, Any]] = {}

# Current active trade
active_trade: Optional[Dict[str, Any]] = None

# Last completed trade
last_completed_trade: Optional[Dict[str, Any]] = None

# Statistics
stats = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}

# Last forming candle used for a base signal per symbol/timeframe
last_signal_candles: Dict[str, str] = {}

# Prevent duplicate automatic cycles
auto_cycle_started = False


# ============================================================
# BASIC HELPERS
# ============================================================

def normalize_symbol(symbol: Any) -> str:
    if symbol is None:
        return ""

    value = str(symbol).strip().upper()

    value = value.replace("/", "")
    value = value.replace("\\", "")
    value = value.replace(" ", "")

    return value


def normalize_timeframe(value: Any) -> str:
    if value is None:
        return ""

    value = str(value).strip().upper()

    value = value.replace(" ", "")

    mapping = {
        "1": "M1",
        "01": "M1",
        "1M": "M1",
        "M01": "M1",
        "M1": "M1",

        "2": "M2",
        "02": "M2",
        "2M": "M2",
        "M02": "M2",
        "M2": "M2",

        "3": "M3",
        "03": "M3",
        "3M": "M3",
        "M03": "M3",
        "M3": "M3",

        "5": "M5",
        "05": "M5",
        "5M": "M5",
        "M05": "M5",
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

        "D1": "D1",
        "DAY": "D1",
    }

    return mapping.get(value, value)


def timeframe_minutes(timeframe: str) -> int:
    tf = normalize_timeframe(timeframe)

    match = re.match(r"^M(\d+)$", tf)
    if match:
        return max(1, int(match.group(1)))

    if tf == "H1":
        return 60

    if tf == "H4":
        return 240

    if tf == "D1":
        return 1440

    return 1


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()

        if not text:
            return None

        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"

            dt = datetime.fromisoformat(text)
        except Exception:
            # Unix timestamp
            try:
                timestamp = float(text)
                dt = datetime.fromtimestamp(timestamp, tz=ZoneInfo("UTC"))
            except Exception:
                return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ALGIERS)

    return dt


def now_algiers() -> datetime:
    return datetime.now(ALGIERS)


def iso_now() -> str:
    return now_algiers().isoformat()


# ============================================================
# PERSISTENT MT4 STORAGE
# ============================================================

def save_mt4_data() -> None:
    """
    Save MT4 data to JSON.
    This prevents losing data merely because another command
    accesses the process later.
    """

    with state_lock:
        payload = {
            "saved_at": iso_now(),
            "data": mt4_data,
        }

        temporary = DATA_FILE.with_suffix(".tmp")

        try:
            with open(temporary, "w", encoding="utf-8") as file:
                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            os.replace(temporary, DATA_FILE)

        except Exception:
            logger.exception("Could not save MT4 data")


def load_mt4_data() -> None:
    global mt4_data

    if not DATA_FILE.exists():
        logger.info("No existing MT4 data file")

        return

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            payload = json.load(file)

        loaded = payload.get("data", {})

        if isinstance(loaded, dict):
            with state_lock:
                mt4_data = loaded

            logger.info(
                "Loaded persistent MT4 data | symbols=%s",
                len(mt4_data),
            )

    except Exception:
        logger.exception("Could not load persistent MT4 data")


# ============================================================
# MT4 DATA NORMALIZATION
# ============================================================

def normalize_candle(candle: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(candle)

    # Normalize common MT4 field names
    if "time" not in result:
        for key in ("timestamp", "datetime", "date"):
            if key in result:
                result["time"] = result[key]
                break

    if "open" not in result:
        for key in ("o", "Open"):
            if key in result:
                result["open"] = result[key]
                break

    if "high" not in result:
        for key in ("h", "High"):
            if key in result:
                result["high"] = result[key]
                break

    if "low" not in result:
        for key in ("l", "Low"):
            if key in result:
                result["low"] = result[key]
                break

    if "close" not in result:
        for key in ("c", "Close"):
            if key in result:
                result["close"] = result[key]
                break

    if "volume" not in result:
        for key in ("v", "tick_volume", "tickVolume"):
            if key in result:
                result["volume"] = result[key]
                break

    result["open"] = safe_float(result.get("open"))
    result["high"] = safe_float(result.get("high"))
    result["low"] = safe_float(result.get("low"))
    result["close"] = safe_float(result.get("close"))
    result["volume"] = safe_float(result.get("volume"))

    return result


def extract_candles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = payload.get("candles")

    if raw is None:
        raw = payload.get("bars")

    if raw is None:
        raw = payload.get("data")

    if isinstance(raw, dict):
        raw = raw.get("candles") or raw.get("bars") or []

    if not isinstance(raw, list):
        return []

    candles = []

    for item in raw:
        if isinstance(item, dict):
            candles.append(normalize_candle(item))

    return candles


def candle_time_value(candle: Dict[str, Any]) -> str:
    value = candle.get("time")

    if value is None:
        return ""

    return str(value)


# ============================================================
# DATA AGE
# ============================================================

def data_age_seconds(item: Dict[str, Any]) -> float:
    received_at = parse_datetime(item.get("received_at"))

    if received_at is None:
        return 999999.0

    now = datetime.now(received_at.tzinfo)

    return max(0.0, (now - received_at).total_seconds())


# ============================================================
# CLOSED CANDLES
# ============================================================

def get_closed_candles(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candles:
        return []

    # MT4 sends newest forming candle last.
    # We intentionally exclude it.
    if len(candles) >= 2:
        return candles[:-1]

    return []


# ============================================================
# INDICATORS
# ============================================================

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1)

    current = sum(values[:period]) / period

    for value in values[period:]:
        current = (value - current) * multiplier + current

    return current


def calculate_rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
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

    if len(gains) < period:
        return None

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


def calculate_williams_r(
    candles: List[Dict[str, Any]],
    period: int = 14,
) -> Optional[float]:

    if len(candles) < period:
        return None

    recent = candles[-period:]

    highest = max(c["high"] for c in recent)
    lowest = min(c["low"] for c in recent)
    close = recent[-1]["close"]

    if highest == lowest:
        return -50.0

    return ((highest - close) / (highest - lowest)) * -100.0


def calculate_atr(
    candles: List[Dict[str, Any]],
    period: int = 10,
) -> Optional[float]:

    if len(candles) < period + 1:
        return None

    trs = []

    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i - 1]

        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)

    if len(trs) < period:
        return None

    return sum(trs[-period:]) / period


def calculate_adx_di(
    candles: List[Dict[str, Any]],
    period: int = 14,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:

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
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )

        trs.append(tr)
        plus_dm.append(pdm)
        minus_dm.append(mdm)

    if len(trs) < period:
        return None, None, None

    atr = sum(trs[-period:]) / period

    if atr <= 0:
        return None, None, None

    plus = sum(plus_dm[-period:]) / period
    minus = sum(minus_dm[-period:]) / period

    plus_di = 100.0 * plus / atr
    minus_di = 100.0 * minus / atr

    denominator = plus_di + minus_di

    if denominator == 0:
        return 0.0, plus_di, minus_di

    dx = 100.0 * abs(plus_di - minus_di) / denominator

    # Lightweight ADX estimate from current directional movement.
    adx = dx

    return adx, plus_di, minus_di


def calculate_indicators(
    candles: List[Dict[str, Any]],
) -> Dict[str, Any]:

    closes = [c["close"] for c in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi14 = calculate_rsi(closes, 14)

    williams14 = calculate_williams_r(candles, 14)

    atr10 = calculate_atr(candles, 10)

    adx14, plus_di14, minus_di14 = calculate_adx_di(
        candles,
        14,
    )

    keltner_mid = ema(closes, 20)

    keltner_atr = calculate_atr(candles, 10)

    keltner_upper = None
    keltner_lower = None

    if keltner_mid is not None and keltner_atr is not None:
        keltner_upper = keltner_mid + (5.0 * keltner_atr)
        keltner_lower = keltner_mid - (5.0 * keltner_atr)

    return {
        "EMA9": ema9,
        "EMA21": ema21,
        "RSI14": rsi14,
        "WilliamsR14": williams14,
        "ATR10": atr10,
        "ADX14": adx14,
        "PlusDI14": plus_di14,
        "MinusDI14": minus_di14,
        "KeltnerEMA20": keltner_mid,
        "KeltnerUpper": keltner_upper,
        "KeltnerLower": keltner_lower,
    }


# ============================================================
# STRUCTURE
# ============================================================

def analyze_structure(
    candles: List[Dict[str, Any]],
) -> str:

    if len(candles) < 8:
        return "MIXED"

    recent = candles[-8:]

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]

    first_half_high = max(highs[:4])
    second_half_high = max(highs[4:])

    first_half_low = min(lows[:4])
    second_half_low = min(lows[4:])

    if second_half_high > first_half_high and second_half_low > first_half_low:
        return "BULLISH"

    if second_half_high < first_half_high and second_half_low < first_half_low:
        return "BEARISH"

    return "MIXED"


# ============================================================
# BREAKOUT
# ============================================================

def analyze_breakout(
    candles: List[Dict[str, Any]],
) -> str:

    if len(candles) < 13:
        return "NONE"

    previous = candles[-11:-1]
    latest = candles[-1]

    resistance = max(c["high"] for c in previous)
    support = min(c["low"] for c in previous)

    if latest["close"] > resistance:
        return "UP"

    if latest["close"] < support:
        return "DOWN"

    return "NONE"


# ============================================================
# CANDLE
# ============================================================

def candle_signal(
    candle: Dict[str, Any],
) -> str:

    open_price = candle["open"]
    close_price = candle["close"]

    if close_price > open_price:
        return "UP"

    if close_price < open_price:
        return "DOWN"

    return "NEUTRAL"


# ============================================================
# LOCAL CANDIDATE SCORE
# ============================================================

def local_direction_score(
    candles: List[Dict[str, Any]],
) -> Tuple[str, float]:

    if len(candles) < MIN_CLOSED_CANDLES:
        return "NONE", 0.0

    indicators = calculate_indicators(candles)

    latest = candles[-1]

    up = 0.0
    down = 0.0

    structure = analyze_structure(candles)
    breakout = analyze_breakout(candles)
    candle = candle_signal(latest)

    ema9 = indicators.get("EMA9")
    ema21 = indicators.get("EMA21")
    rsi = indicators.get("RSI14")
    williams = indicators.get("WilliamsR14")
    plus_di = indicators.get("PlusDI14")
    minus_di = indicators.get("MinusDI14")
    adx = indicators.get("ADX14")

    # Structure
    if structure == "BULLISH":
        up += 3
    elif structure == "BEARISH":
        down += 3

    # Breakout
    if breakout == "UP":
        up += 3
    elif breakout == "DOWN":
        down += 3

    # EMA
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            up += 2
        elif ema9 < ema21:
            down += 2

    # Price vs EMA9
    if ema9 is not None:
        if latest["close"] > ema9:
            up += 1
        elif latest["close"] < ema9:
            down += 1

    # RSI
    if rsi is not None:
        if 50 < rsi < 68:
            up += 1
        elif 32 < rsi < 50:
            down += 1

    # Williams
    if williams is not None:
        if -80 < williams < -20:
            if williams > -50:
                up += 1
            else:
                down += 1

    # DI
    if plus_di is not None and minus_di is not None:
        if plus_di > minus_di:
            up += 1
        elif minus_di > plus_di:
            down += 1

    # ADX directional confirmation
    if adx is not None and adx >= 20:
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di:
                up += 1
            elif minus_di > plus_di:
                down += 1

    # Candle
    if candle == "UP":
        up += 1
    elif candle == "DOWN":
        down += 1

    if up > down:
        return "UP", up - down

    if down > up:
        return "DOWN", down - up

    return "NONE", 0.0


# ============================================================
# ENTRY TIME
# ============================================================

def get_next_entry_time(
    timeframe: str,
) -> datetime:

    minutes = timeframe_minutes(timeframe)

    now = now_algiers()

    # Find next timeframe boundary.
    base = now.replace(
        second=0,
        microsecond=0,
    )

    current_minute = base.minute

    next_block = ((current_minute // minutes) + 1) * minutes

    if next_block >= 60:
        candidate = (
            base.replace(
                minute=0,
            )
            + timedelta(hours=1)
        )
    else:
        candidate = base.replace(
            minute=next_block,
        )

    # Ensure enough time for the user to enter.
    if (
        candidate - now
    ).total_seconds() < MIN_ENTRY_LEAD_SECONDS:

        candidate += timedelta(
            minutes=minutes
        )

    return candidate


# ============================================================
# CANCELLATION LEVEL
# ============================================================

def cancellation_level(
    candles: List[Dict[str, Any]],
    direction: str,
    entry_price: float,
) -> float:

    recent = candles[-10:]

    if direction == "UP":
        swing = min(c["low"] for c in recent)

        # Keep cancellation below entry.
        if swing >= entry_price:
            swing = entry_price * 0.999

        return swing

    swing = max(c["high"] for c in recent)

    # Keep cancellation above entry.
    if swing <= entry_price:
        swing = entry_price * 1.001

    return swing


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
    indicators: Dict[str, Any],
) -> str:

    compact_candles = []

    for candle in candles[-100:]:
        compact_candles.append(
            {
                "time": candle.get("time"),
                "open": candle.get("open"),
                "high": candle.get("high"),
                "low": candle.get("low"),
                "close": candle.get("close"),
                "volume": candle.get("volume"),
            }
        )

    return f"""
You are the technical-analysis engine for ZinoProSignalAI.

Analyze {symbol} on {timeframe}.

IMPORTANT:
- This is a binary-direction analysis.
- Return UP or DOWN only.
- Never return WAIT.
- Never return NO SIGNAL.
- Never return NEUTRAL.
- Do not invent missing indicators.
- Price Action has priority over indicators.
- Structure has priority over weak oscillator readings.
- Breakout/retest and liquidity matter.
- Momentum and candle behavior matter.
- EMA 9/21 is secondary confirmation.
- RSI 14 and Williams %R 14 are secondary confirmation.
- Keltner and ADX/DI are supporting confirmation.
- Do not claim certainty.
- Confidence must normally remain below 90.
- Maximum confidence is 89.

SCORING MUST TOTAL EXACTLY 18:

Structure: 2
Breakout: 2
Liquidity: 1
Momentum: 2
Candle: 2
RSI: 1
Summary: 2
Oscillators: 3
Moving Averages: 3

Total = 18.

Return ONLY valid JSON in this exact structure:

{{
  "direction": "UP",
  "confidence": 72,
  "scores": {{
    "structure": 2,
    "breakout": 2,
    "liquidity": 1,
    "momentum": 2,
    "candle": 2,
    "rsi": 1,
    "summary": 2,
    "oscillators": 2,
    "moving_averages": 2
  }},
  "reason": "short technical reason"
}}

The larger score must match the direction.

INDICATORS:

{json.dumps(indicators, ensure_ascii=False)}

CLOSED CANDLES:

{json.dumps(compact_candles, ensure_ascii=False)}
""".strip()


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol: str,
    timeframe: str,
    candles: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:

    if len(candles) < MIN_CLOSED_CANDLES:
        logger.warning(
            "Not enough closed candles | %s | %s | %s",
            symbol,
            timeframe,
            len(candles),
        )

        return None

    indicators = calculate_indicators(candles)

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        indicators,
    )

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""

        result = json.loads(text)

        direction = str(
            result.get("direction", "")
        ).strip().upper()

        if direction not in {"UP", "DOWN"}:
            return None

        scores_raw = result.get("scores", {})

        score_keys = [
            "structure",
            "breakout",
            "liquidity",
            "momentum",
            "candle",
            "rsi",
            "summary",
            "oscillators",
            "moving_averages",
        ]

        scores = {}

        for key in score_keys:
            value = scores_raw.get(key, 0)

            try:
                value = int(value)
            except Exception:
                value = 0

            scores[key] = max(0, value)

        total = sum(scores.values())

        if total != 18:
            logger.warning(
                "Gemini score total invalid | %s | total=%s",
                symbol,
                total,
            )

            return None

        up_score = 0
        down_score = 0

        # Gemini gives category scores, so we use the final direction
        # plus a conservative score split based on technical evidence.
        #
        # The total remains exactly 18.
        local_direction, local_edge = local_direction_score(candles)

        if local_direction == direction and local_edge >= 4:
            selected = 14
        elif local_direction == direction and local_edge >= 2:
            selected = 12
        else:
            selected = 11

        selected = min(17, selected)
        opposite = 18 - selected

        if direction == "UP":
            up_score = selected
            down_score = opposite
        else:
            down_score = selected
            up_score = opposite

        confidence = result.get("confidence", 70)

        try:
            confidence = int(float(confidence))
        except Exception:
            confidence = 70

        confidence = max(
            MIN_CONFIDENCE,
            min(MAX_CONFIDENCE, confidence),
        )

        # Do not permit weak local structure to produce an exaggerated
        # confidence.
        if local_direction != direction:
            confidence = min(confidence, 74)

        if local_edge < 2:
            confidence = min(confidence, 76)

        reason = str(
            result.get(
                "reason",
                "Price action and technical confluence.",
            )
        ).strip()

        if not reason:
            reason = "Price action and technical confluence."

        return {
            "direction": direction,
            "confidence": confidence,
            "up_score": up_score,
            "down_score": down_score,
            "reason": reason[:300],
            "indicators": indicators,
            "local_direction": local_direction,
            "local_edge": local_edge,
        }

    except Exception:
        logger.exception(
            "Gemini analysis failed | %s | %s",
            symbol,
            timeframe,
        )

        return None


# ============================================================
# QUALITY FILTER
# ============================================================

def quality_check(
    analysis: Dict[str, Any],
) -> bool:

    direction = analysis["direction"]

    up_score = int(analysis["up_score"])
    down_score = int(analysis["down_score"])

    confidence = int(analysis["confidence"])

    selected_score = (
        up_score
        if direction == "UP"
        else down_score
    )

    opposite_score = (
        down_score
        if direction == "UP"
        else up_score
    )

    difference = selected_score - opposite_score

    if selected_score < MIN_SELECTED_SCORE:
        return False

    if difference < MIN_SCORE_DIFFERENCE:
        return False

    if confidence < MIN_CONFIDENCE:
        return False

    if analysis["local_direction"] != direction:
        return False

    if analysis["local_edge"] < 2:
        return False

    return True


# ============================================================
# FIND BEST CANDIDATE
# ============================================================

def get_candidates() -> List[Dict[str, Any]]:

    candidates = []

    with state_lock:
        snapshot = dict(mt4_data)

    for key, item in snapshot.items():

        symbol = normalize_symbol(
            item.get("symbol")
        )

        timeframe = normalize_timeframe(
            item.get("timeframe")
        )

        if timeframe not in AUTO_TIMEFRAMES:
            continue

        age = data_age_seconds(item)

        if age > MAX_DATA_AGE_SECONDS:
            continue

        candles = extract_candles(item)

        closed = get_closed_candles(candles)

        if len(closed) < MIN_CLOSED_CANDLES:
            continue

        local_direction, local_edge = local_direction_score(
            closed
        )

        if local_direction not in {"UP", "DOWN"}:
            continue

        candle_id = candle_time_value(
            candles[-1]
        ) if candles else ""

        signal_key = f"{symbol}|{timeframe}"

        # Do not analyze the same forming candle repeatedly.
        if last_signal_candles.get(signal_key) == candle_id:
            continue

        candidates.append(
            {
                "key": key,
                "symbol": symbol,
                "timeframe": timeframe,
                "item": item,
                "candles": candles,
                "closed": closed,
                "age": age,
                "local_direction": local_direction,
                "local_edge": local_edge,
            }
        )

    candidates.sort(
        key=lambda x: (
            x["local_edge"],
            -x["age"],
        ),
        reverse=True,
    )

    return candidates


# ============================================================
# ANALYZE BEST SIGNAL
# ============================================================

def find_best_signal() -> Optional[Dict[str, Any]]:

    candidates = get_candidates()

    if not candidates:
        logger.info("No eligible M1/M3 MT4 candidate found")

        return None

    for candidate in candidates:

        symbol = candidate["symbol"]
        timeframe = candidate["timeframe"]

        logger.info(
            "Analyzing candidate | %s | %s | local=%s edge=%.1f age=%.1f",
            symbol,
            timeframe,
            candidate["local_direction"],
            candidate["local_edge"],
            candidate["age"],
        )

        analysis = analyze_with_gemini(
            symbol,
            timeframe,
            candidate["closed"],
        )

        if not analysis:
            continue

        if not quality_check(analysis):
            logger.info(
                "Candidate rejected | %s | %s | direction=%s confidence=%s scores=%s/%s",
                symbol,
                timeframe,
                analysis["direction"],
                analysis["confidence"],
                analysis["up_score"],
                analysis["down_score"],
            )

            continue

        entry_time = get_next_entry_time(timeframe)

        entry_price = candidate["closed"][-1]["close"]

        cancel_price = cancellation_level(
            candidate["closed"],
            analysis["direction"],
            entry_price,
        )

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": analysis["direction"],
            "confidence": analysis["confidence"],
            "up_score": analysis["up_score"],
            "down_score": analysis["down_score"],
            "entry_time": entry_time.isoformat(),
            "entry_price": entry_price,
            "cancel_price": cancel_price,
            "reason": analysis["reason"],
            "candle_id": candle_time_value(
                candidate["candles"][-1]
            ),
        }

    return None


# ============================================================
# FORMAT SIGNAL CARD
# ============================================================

def format_signal_card(
    trade: Dict[str, Any],
    result: Optional[str] = None,
) -> str:

    direction = trade["direction"]

    if direction == "UP":
        direction_text = "🟢 UP"
        cancel_text = "إلغاء إذا أغلقت الشمعة تحت"
    else:
        direction_text = "🔴 DOWN"
        cancel_text = "إلغاء إذا أغلقت الشمعة فوق"

    entry_dt = parse_datetime(
        trade["entry_time"]
    )

    if entry_dt is None:
        entry_text = "--:--:--"
    else:
        entry_text = entry_dt.astimezone(
            ALGIERS
        ).strftime("%H:%M:%S")

    price = trade["entry_price"]
    cancel_price = trade["cancel_price"]

    # Display precision based on price magnitude.
    if price >= 100:
        price_fmt = ".3f"
    elif price >= 10:
        price_fmt = ".5f"
    elif price >= 1:
        price_fmt = ".5f"
    else:
        price_fmt = ".5f"

    result_line = ""

    if result == "WIN":
        result_line = "\n\n🟢 WIN"
    elif result == "LOSS":
        result_line = "\n\n🔴 LOSS"

    recovery = trade.get(
        "trade_type",
        "BASE",
    )

    recovery_number = trade.get(
        "recovery_number",
        0,
    )

    if recovery == "RECOVERY":
        cycle_text = f"🔁 RECOVERY {recovery_number}/1"
    else:
        cycle_text = "🟦 BASE"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {trade['symbol']} | {trade['timeframe']}\n\n"
        f"{cycle_text}\n\n"
        f"{direction_text}\n"
        f"📈 Confidence: {trade['confidence']}%\n"
        f"🎯 UP Score: {trade['up_score']}/18\n"
        f"🎯 DOWN Score: {trade['down_score']}/18\n\n"
        f"⏰ ENTRY: {entry_text}\n"
        f"💰 Price: {price:{price_fmt}}\n"
        f"⚠️ {cancel_text} "
        f"{cancel_price:{price_fmt}}\n\n"
        f"🧠 {trade['reason']}"
        f"{result_line}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# TELEGRAM HELPERS
# ============================================================

async def send_owner_message(
    application: Application,
    text: str,
) -> None:

    try:
        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=text,
        )

    except Exception:
        logger.exception(
            "Could not send Telegram message"
        )


# ============================================================
# REGISTER TRADE
# ============================================================

def register_trade(
    signal: Dict[str, Any],
    trade_type: str = "BASE",
    recovery_number: int = 0,
) -> Dict[str, Any]:

    global active_trade

    trade = dict(signal)

    trade["trade_id"] = datetime.now(
        ALGIERS
    ).strftime("%Y%m%d%H%M%S-%f")[:-3]

    trade["trade_type"] = trade_type
    trade["recovery_number"] = recovery_number
    trade["status"] = "ACTIVE"
    trade["created_at"] = iso_now()
    trade["result"] = None

    with state_lock:
        active_trade = trade

    return trade


# ============================================================
# SEND NEW SIGNAL
# ============================================================

async def send_new_signal(
    application: Application,
    trade_type: str = "BASE",
    recovery_number: int = 0,
) -> bool:

    global active_trade

    with state_lock:
        if active_trade is not None:
            if active_trade.get("status") == "ACTIVE":
                logger.info(
                    "Active trade already exists | trade_id=%s",
                    active_trade.get("trade_id"),
                )

                return False

    signal = find_best_signal()

    if signal is None:
        logger.info(
            "No signal available"
        )

        return False

    # Mark the forming candle before sending.
    signal_key = (
        f"{signal['symbol']}|{signal['timeframe']}"
    )

    last_signal_candles[
        signal_key
    ] = signal["candle_id"]

    trade = register_trade(
        signal,
        trade_type=trade_type,
        recovery_number=recovery_number,
    )

    text = format_signal_card(trade)

    await send_owner_message(
        application,
        text,
    )

    logger.info(
        "SIGNAL SENT | type=%s symbol=%s timeframe=%s direction=%s confidence=%s trade_id=%s",
        trade_type,
        trade["symbol"],
        trade["timeframe"],
        trade["direction"],
        trade["confidence"],
        trade["trade_id"],
    )

    return True


# ============================================================
# RESULT HANDLING
# ============================================================

def apply_result(result: str) -> Tuple[bool, str]:
    global active_trade
    global last_completed_trade

    result = result.upper()

    if result not in {"WIN", "LOSS"}:
        return False, "Invalid result"

    with state_lock:

        if active_trade is None:
            return False, "NO_ACTIVE_TRADE"

        if active_trade.get("status") != "ACTIVE":
            return False, "NO_ACTIVE_TRADE"

        trade = dict(active_trade)

        trade["status"] = "COMPLETED"
        trade["result"] = result
        trade["completed_at"] = iso_now()

        last_completed_trade = trade
        active_trade = None

        stats["wins" if result == "WIN" else "losses"] += 1

        if trade.get("trade_type") == "RECOVERY":
            stats[
                "recovery_wins"
                if result == "WIN"
                else "recovery_losses"
            ] += 1
        else:
            stats[
                "base_wins"
                if result == "WIN"
                else "base_losses"
            ] += 1

    return True, trade


# ============================================================
# AUTO LOOP
# ============================================================

async def auto_loop(
    application: Application,
) -> None:

    global auto_cycle_started

    if auto_cycle_started:
        return

    auto_cycle_started = True

    logger.info(
        "Automatic signal loop started"
    )

    # Wait until next clean 3-minute boundary.
    while True:

        now = now_algiers()

        next_minute = (
            (now.minute // CYCLE_MINUTES) + 1
        ) * CYCLE_MINUTES

        base = now.replace(
            second=0,
            microsecond=0,
        )

        if next_minute >= 60:
            next_run = (
                base.replace(
                    minute=0,
                )
                + timedelta(hours=1)
            )
        else:
            next_run = base.replace(
                minute=next_minute,
            )

        wait_seconds = max(
            1,
            (
                next_run - now
            ).total_seconds(),
        )

        logger.info(
            "Next automatic cycle | %s | wait=%.1fs",
            next_run.strftime("%H:%M:%S"),
            wait_seconds,
        )

        await asyncio.sleep(
            wait_seconds
        )

        with state_lock:
            current_trade = (
                dict(active_trade)
                if active_trade
                else None
            )

        # Never send a second trade while one is active.
        if current_trade is not None:
            logger.info(
                "Cycle skipped | active trade=%s",
                current_trade.get("trade_id"),
            )

            continue

        # If the previous completed trade was LOSS,
        # the next signal is Recovery 1/1.
        with state_lock:
            previous = (
                dict(last_completed_trade)
                if last_completed_trade
                else None
            )

        if (
            previous is not None
            and previous.get("result") == "LOSS"
        ):
            logger.info(
                "Starting RECOVERY 1/1"
            )

            sent = await send_new_signal(
                application,
                trade_type="RECOVERY",
                recovery_number=1,
            )

            if not sent:
                logger.info(
                    "Recovery not sent: no valid signal"
                )

        else:
            logger.info(
                "Starting BASE signal"
            )

            sent = await send_new_signal(
                application,
                trade_type="BASE",
                recovery_number=0,
            )

            if not sent:
                logger.info(
                    "Base signal not sent: no valid signal"
                )


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update: Update) -> bool:

    if update.effective_user is None:
        return False

    return update.effective_user.id == OWNER_ID


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "✅ Bot is running.\n"
        "✅ MT4 connection supported.\n"
        "✅ One signal per cycle.\n"
        "✅ BASE + RECOVERY 1/1.\n\n"
        "Commands:\n"
        "/mt4status\n"
        "/analyze SYMBOL M1\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


# ============================================================
# /MT4STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    with state_lock:
        items = list(mt4_data.values())

    if not items:
        await update.message.reply_text(
            "❌ لا توجد بيانات MT4 حتى الآن."
        )

        return

    lines = [
        "📡 MT4 STATUS",
        "━━━━━━━━━━━━━━━━━━",
    ]

    for item in sorted(
        items,
        key=lambda x: (
            normalize_symbol(
                x.get("symbol")
            ),
            normalize_timeframe(
                x.get("timeframe")
            ),
        ),
    ):

        symbol = normalize_symbol(
            item.get("symbol")
        )

        timeframe = normalize_timeframe(
            item.get("timeframe")
        )

        candles = extract_candles(item)

        age = data_age_seconds(item)

        lines.append(
            f"📊 {symbol} | {timeframe} | "
            f"candles={len(candles)} | "
            f"age={age:.0f}s"
        )

    with state_lock:
        current = (
            dict(active_trade)
            if active_trade
            else None
        )

    if current:
        lines.append("")
        lines.append(
            f"🔴 Active: "
            f"{current['symbol']} "
            f"{current['timeframe']} "
            f"{current['direction']}"
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
) -> None:

    if not is_owner(update):
        return

    if len(context.args) < 2:
        await update.message.reply_text(
            "استعمل:\n"
            "/analyze EURCAD.wt M1"
        )

        return

    symbol = normalize_symbol(
        context.args[0]
    )

    timeframe = normalize_timeframe(
        context.args[1]
    )

    target = None

    with state_lock:
        for item in mt4_data.values():

            item_symbol = normalize_symbol(
                item.get("symbol")
            )

            item_tf = normalize_timeframe(
                item.get("timeframe")
            )

            if (
                item_symbol == symbol
                and item_tf == timeframe
            ):
                target = dict(item)
                break

    if target is None:
        await update.message.reply_text(
            f"❌ لا توجد بيانات MT4 لـ "
            f"{symbol} {timeframe}"
        )

        return

    age = data_age_seconds(target)

    if age > MAX_DATA_AGE_SECONDS:
        await update.message.reply_text(
            f"❌ بيانات {symbol} {timeframe} قديمة.\n"
            f"العمر: {age:.0f} ثانية."
        )

        return

    candles = extract_candles(target)

    closed = get_closed_candles(candles)

    if len(closed) < MIN_CLOSED_CANDLES:
        await update.message.reply_text(
            f"❌ الشموع المغلقة غير كافية.\n"
            f"الموجود: {len(closed)}\n"
            f"المطلوب: {MIN_CLOSED_CANDLES}"
        )

        return

    await update.message.reply_text(
        f"🔎 جاري تحليل {symbol} {timeframe}..."
    )

    analysis = analyze_with_gemini(
        symbol,
        timeframe,
        closed,
    )

    if analysis is None:
        await update.message.reply_text(
            "❌ التحليل لم يعطِ نتيجة صالحة."
        )

        return

    if not quality_check(analysis):
        await update.message.reply_text(
            "❌ التحليل موجود، لكن الجودة الحالية "
            "لا تحقق شروط الإشارة."
        )

        return

    entry_time = get_next_entry_time(
        timeframe
    )

    entry_price = closed[-1]["close"]

    cancel_price = cancellation_level(
        closed,
        analysis["direction"],
        entry_price,
    )

    signal = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": analysis["direction"],
        "confidence": analysis["confidence"],
        "up_score": analysis["up_score"],
        "down_score": analysis["down_score"],
        "entry_time": entry_time.isoformat(),
        "entry_price": entry_price,
        "cancel_price": cancel_price,
        "reason": analysis["reason"],
        "candle_id": candle_time_value(
            candles[-1]
        ),
        "trade_type": "BASE",
        "recovery_number": 0,
    }

    await update.message.reply_text(
        format_signal_card(signal)
    )


# ============================================================
# /WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    success, result = apply_result(
        "WIN"
    )

    if not success:

        if result == "NO_ACTIVE_TRADE":
            await update.message.reply_text(
                "⚠️ لا توجد صفقة نشطة لتسجيل WIN."
            )

        else:
            await update.message.reply_text(
                "❌ تعذر تسجيل النتيجة."
            )

        return

    trade = result

    await update.message.reply_text(
        format_signal_card(
            trade,
            result="WIN",
        )
    )


# ============================================================
# /LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    success, result = apply_result(
        "LOSS"
    )

    if not success:

        if result == "NO_ACTIVE_TRADE":
            await update.message.reply_text(
                "⚠️ لا توجد صفقة نشطة لتسجيل LOSS."
            )

        else:
            await update.message.reply_text(
                "❌ تعذر تسجيل النتيجة."
            )

        return

    trade = result

    await update.message.reply_text(
        format_signal_card(
            trade,
            result="LOSS",
        )
        + "\n\n"
        "🔁 Recovery 1/1 ستكون في الدورة التالية "
        "إذا توفرت إشارة مؤهلة."
    )


# ============================================================
# /STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    with state_lock:
        total_wins = stats["wins"]
        total_losses = stats["losses"]

        base_wins = stats["base_wins"]
        base_losses = stats["base_losses"]

        recovery_wins = stats["recovery_wins"]
        recovery_losses = stats["recovery_losses"]

        current = (
            dict(active_trade)
            if active_trade
            else None
        )

    total = total_wins + total_losses

    if total > 0:
        win_rate = (
            total_wins / total
        ) * 100
    else:
        win_rate = 0.0

    active_text = "NONE"

    if current:
        active_text = (
            f"{current['symbol']} "
            f"{current['timeframe']} "
            f"{current['direction']} "
            f"{current['trade_type']}"
        )

    await update.message.reply_text(
        "📊 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 WIN: {total_wins}\n"
        f"🔴 LOSS: {total_losses}\n"
        f"📈 Win rate: {win_rate:.1f}%\n\n"
        f"🟦 BASE WIN: {base_wins}\n"
        f"🟥 BASE LOSS: {base_losses}\n\n"
        f"🔁 RECOVERY WIN: {recovery_wins}\n"
        f"🔁 RECOVERY LOSS: {recovery_losses}\n\n"
        f"🎯 Active trade:\n{active_text}"
    )


# ============================================================
# /RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    global active_trade
    global last_completed_trade
    global last_signal_candles

    if not is_owner(update):
        return

    with state_lock:

        stats["wins"] = 0
        stats["losses"] = 0
        stats["base_wins"] = 0
        stats["base_losses"] = 0
        stats["recovery_wins"] = 0
        stats["recovery_losses"] = 0

        active_trade = None
        last_completed_trade = None
        last_signal_candles = {}

    await update.message.reply_text(
        "♻️ الإحصائيات وحالة الصفقات تم تصفيرها."
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:

    if not is_owner(update):
        return

    if not update.message:
        return

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:
        return

    parts = text.split()

    if len(parts) != 2:
        return

    symbol = normalize_symbol(
        parts[0]
    )

    timeframe = normalize_timeframe(
        parts[1]
    )

    if not re.match(
        r"^[A-Z0-9._-]+$",
        symbol,
    ):
        return

    if timeframe not in AUTO_TIMEFRAMES:
        await update.message.reply_text(
            "⚠️ التحليل التلقائي يدعم حالياً M1 و M3."
        )

        return

    # Reuse analyze logic.
    context.args = [
        symbol,
        timeframe,
    ]

    await analyze_command(
        update,
        context,
    )


# ============================================================
# HTTP SERVER
# ============================================================

class MT4HTTPHandler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format_string: str,
        *args: Any,
    ) -> None:
        return

    def send_json(
        self,
        status: int,
        payload: Dict[str, Any],
    ) -> None:

        body = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self) -> None:

        if self.path in {"/", "/health"}:

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "time": iso_now(),
                },
            )

            return

        if self.path == "/mt4status":

            with state_lock:
                count = len(mt4_data)

            self.send_json(
                200,
                {
                    "status": "ok",
                    "mt4_symbols": count,
                    "time": iso_now(),
                },
            )

            return

        self.send_json(
            404,
            {
                "error": "not_found"
            },
        )

    def do_POST(self) -> None:

        if self.path != "/mt4":

            self.send_json(
                404,
                {
                    "error": "not_found"
                },
            )

            return

        provided_key = (
            self.headers.get(
                "X-API-Key",
                "",
            ).strip()
        )

        if provided_key != MT4_API_KEY:

            self.send_json(
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
                    "0",
                )
            )

            if content_length <= 0:
                self.send_json(
                    400,
                    {
                        "error": "empty_body"
                    },
                )

                return

            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode("utf-8")
            )

            if not isinstance(payload, dict):
                raise ValueError(
                    "JSON root must be object"
                )

            symbol = normalize_symbol(
                payload.get("symbol")
            )

            timeframe = normalize_timeframe(
                payload.get("timeframe")
            )

            if not symbol:
                self.send_json(
                    400,
                    {
                        "error": "symbol_missing"
                    },
                )

                return

            if not timeframe:
                self.send_json(
                    400,
                    {
                        "error": "timeframe_missing"
                    },
                )

                return

            candles = extract_candles(
                payload
            )

            if not candles:
                self.send_json(
                    400,
                    {
                        "error": "candles_missing"
                    },
                )

                return

            # Keep the latest 200 candles.
            candles = candles[-200:]

            key = f"{symbol}|{timeframe}"

            stored = {
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": candles,
                "received_at": iso_now(),
            }

            with state_lock:
                mt4_data[key] = stored

            save_mt4_data()

            logger.info(
                "MT4 data stored | %s | %s | candles=%s",
                symbol,
                timeframe,
                len(candles),
            )

            self.send_json(
                200,
                {
                    "status": "ok",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": len(candles),
                    "received_at": stored[
                        "received_at"
                    ],
                },
            )

        except Exception as exc:

            logger.exception(
                "MT4 POST error"
            )

            self.send_json(
                400,
                {
                    "error": str(exc)
                },
            )


# ============================================================
# HTTP SERVER THREAD
# ============================================================

def start_http_server() -> None:

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        MT4HTTPHandler,
    )

    logger.info(
        "HTTP server started on port %s",
        PORT,
    )

    server.serve_forever()


# ============================================================
# POST INIT
# ============================================================

async def post_init(
    application: Application,
) -> None:

    asyncio.create_task(
        auto_loop(application)
    )

    logger.info(
        "Automatic loop scheduled"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    load_mt4_data()

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
        name="MT4-HTTP",
    )

    http_thread.start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
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

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    logger.info(
        "ZinoProSignalAI starting..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
