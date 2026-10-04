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

# الاسم يبقى MT4_API_KEY حتى لا نضطر لتغيير Render.
# MT5 EA سيرسل نفس المفتاح.
MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

ANALYSIS_TIMEFRAME = "M1"

# عدد الشموع الأدنى
MIN_CLOSED_CANDLES = 40

# العدد الذي يستقبله MT5
EXPECTED_CANDLES = 150

# نحلل أفضل عدد من المرشحين فقط بواسطة Gemini
TOP_CANDIDATES_FOR_AI = 5

# أقل Score مقبول للإشارة
MIN_SIGNAL_SCORE = 13

# Recovery
RECOVERY_LIMIT = 1
RECOVERY_DELAY_SECONDS = 120

# البيانات تعتبر قديمة بعد هذا الوقت
MARKET_DATA_MAX_AGE_SECONDS = 130

# ملف النتائج
HISTORY_FILE = "signal_history.json"

# فحص الخلفية
BACKGROUND_INTERVAL_SECONDS = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBAL STATE
# ============================================================

data_lock = threading.RLock()

# symbol -> {
#     symbol,
#     timeframe,
#     candles,
#     received_at,
#     batch_id,
#     ...
# }
market_data = {}

# آخر Batch وصل من MT5
latest_batch_id = None

# هل وصلنا scan_complete من MT5؟
latest_batch_complete = False

# آخر Batch تم تحليله
last_analyzed_batch_id = None

# Candidate محفوظ أثناء Recovery
pending_recovery_candidate = None

# دورة الصفقة الحالية
active_cycle = {
    "active": False,
    "symbol": None,
    "timeframe": ANALYSIS_TIMEFRAME,
    "direction": None,
    "entry_price": None,
    "entry_time": None,
    "trade_type": "BASE",
    "recovery_number": 0,

    # Recovery
    "recovery_ready_at": None,

    # رقم الصفقة
    "trade_id": None,
}

# إحصائيات
stats = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}

stats_lock = threading.RLock()

# آخر إشارة مرسلة
last_signal_sent_at = 0

# منع إرسال إشارتين في نفس اللحظة
signal_send_lock = threading.Lock()


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized.")
    except Exception as e:
        logger.exception("Gemini initialization failed: %s", e)
else:
    logger.warning("GEMINI_API_KEY is missing.")


# ============================================================
# HISTORY
# ============================================================

def load_history():
    global stats

    try:
        if not os.path.exists(HISTORY_FILE):
            return

        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)

        if isinstance(saved, dict):
            for key in stats:
                if key in saved:
                    stats[key] = int(saved[key])

        logger.info("History loaded: %s", stats)

    except Exception as e:
        logger.exception("History load error: %s", e)


def save_history():
    try:
        with stats_lock:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.exception("History save error: %s", e)


load_history()


# ============================================================
# TIME HELPERS
# ============================================================

def now_algiers():
    return datetime.now(ALGIERS)


def format_algiers_time(dt=None):
    if dt is None:
        dt = now_algiers()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ALGIERS)

    dt = dt.astimezone(ALGIERS)

    return dt.strftime("%H:%M:%S")


def format_entry_time(timestamp):
    try:
        dt = datetime.fromtimestamp(timestamp, tz=ALGIERS)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return format_algiers_time()


# ============================================================
# BASIC HELPERS
# ============================================================

def is_owner(update: Update):
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


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


def normalize_symbol(symbol):
    return str(symbol or "").strip().upper()


def normalize_timeframe(tf):
    tf = str(tf or "M1").strip().upper()

    if tf in ("1", "1M", "M1"):
        return "M1"

    return tf


# ============================================================
# CANDLE HELPERS
# ============================================================

def normalize_candles(raw_candles):
    result = []

    if not isinstance(raw_candles, list):
        return result

    for c in raw_candles:
        if not isinstance(c, dict):
            continue

        try:
            item = {
                "time": safe_int(c.get("time")),
                "open": safe_float(c.get("open")),
                "high": safe_float(c.get("high")),
                "low": safe_float(c.get("low")),
                "close": safe_float(c.get("close")),
                "volume": safe_float(c.get("volume", 0)),
            }

            if item["time"] <= 0:
                continue

            if item["high"] <= 0 or item["low"] <= 0:
                continue

            if item["high"] < item["low"]:
                continue

            result.append(item)

        except Exception:
            continue

    result.sort(key=lambda x: x["time"])

    # إزالة التكرار
    unique = {}

    for candle in result:
        unique[candle["time"]] = candle

    result = list(unique.values())
    result.sort(key=lambda x: x["time"])

    return result


def latest_closed_candle(candles):
    if not candles:
        return None

    return candles[-1]


def candle_body(c):
    return abs(c["close"] - c["open"])


def candle_range(c):
    return max(c["high"] - c["low"], 0.0000000001)


def candle_direction(c):
    if c["close"] > c["open"]:
        return "UP"

    if c["close"] < c["open"]:
        return "DOWN"

    return "FLAT"


# ============================================================
# INDICATORS
# ============================================================

def ema(values, period):
    if not values:
        return []

    if len(values) < period:
        return [None] * len(values)

    result = [None] * len(values)

    sma = sum(values[:period]) / period
    result[period - 1] = sma

    multiplier = 2.0 / (period + 1)

    previous = sma

    for i in range(period, len(values)):
        previous = (
            (values[i] - previous) * multiplier
            + previous
        )
        result[i] = previous

    return result


def rsi(values, period=14):
    result = [None] * len(values)

    if len(values) <= period:
        return result

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]

        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    for i in range(period + 1, len(values)):
        gain = gains[i - 1]
        loss = losses[i - 1]

        avg_gain = (
            (avg_gain * (period - 1) + gain)
            / period
        )

        avg_loss = (
            (avg_loss * (period - 1) + loss)
            / period
        )

        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

    return result


def williams_r(candles, period=14):
    result = [None] * len(candles)

    if len(candles) < period:
        return result

    for i in range(period - 1, len(candles)):
        window = candles[i - period + 1:i + 1]

        highest = max(x["high"] for x in window)
        lowest = min(x["low"] for x in window)

        if highest == lowest:
            result[i] = -50.0
        else:
            result[i] = (
                (highest - candles[i]["close"])
                / (highest - lowest)
            ) * -100

    return result


def true_ranges(candles):
    result = []

    for i, c in enumerate(candles):
        if i == 0:
            tr = c["high"] - c["low"]
        else:
            prev_close = candles[i - 1]["close"]

            tr = max(
                c["high"] - c["low"],
                abs(c["high"] - prev_close),
                abs(c["low"] - prev_close),
            )

        result.append(max(tr, 0.0))

    return result


def atr(candles, period=10):
    trs = true_ranges(candles)

    if len(trs) < period:
        return [None] * len(candles)

    result = [None] * len(candles)

    value = sum(trs[:period]) / period
    result[period - 1] = value

    for i in range(period, len(trs)):
        value = (
            (value * (period - 1) + trs[i])
            / period
        )

        result[i] = value

    return result


def adx_data(candles, period=14):
    n = len(candles)

    adx = [None] * n
    plus_di = [None] * n
    minus_di = [None] * n

    if n <= period + 1:
        return adx, plus_di, minus_di

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):
        high = candles[i]["high"]
        low = candles[i]["low"]

        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        tr[i] = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

        up_move = high - prev_high
        down_move = prev_low - low

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move

        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    atr_value = sum(tr[1:period + 1]) / period
    plus_value = sum(plus_dm[1:period + 1]) / period
    minus_value = sum(minus_dm[1:period + 1]) / period

    dx_values = []

    for i in range(period + 1, n):
        atr_value = (
            (atr_value * (period - 1) + tr[i])
            / period
        )

        plus_value = (
            (plus_value * (period - 1) + plus_dm[i])
            / period
        )

        minus_value = (
            (minus_value * (period - 1) + minus_dm[i])
            / period
        )

        if atr_value <= 0:
            continue

        pdi = 100 * plus_value / atr_value
        mdi = 100 * minus_value / atr_value

        plus_di[i] = pdi
        minus_di[i] = mdi

        denominator = pdi + mdi

        if denominator <= 0:
            dx = 0
        else:
            dx = 100 * abs(pdi - mdi) / denominator

        dx_values.append((i, dx))

        if len(dx_values) >= period:
            recent = dx_values[-period:]
            adx_value = sum(x[1] for x in recent) / period
            adx[i] = adx_value

    return adx, plus_di, minus_di


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_snapshot(symbol, candles):
    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    closes = [x["close"] for x in candles]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi_values = rsi(closes, 14)
    wr_values = williams_r(candles, 14)

    atr_values = atr(candles, 10)

    adx_values, plus_di, minus_di = adx_data(
        candles,
        14
    )

    i = len(candles) - 1

    latest = candles[i]

    e9 = ema9[i]
    e21 = ema21[i]

    rv = rsi_values[i]
    wr = wr_values[i]

    atrv = atr_values[i]

    adxv = adx_values[i]
    pd = plus_di[i]
    md = minus_di[i]

    recent = candles[-20:]

    recent_high = max(x["high"] for x in recent)
    recent_low = min(x["low"] for x in recent)

    previous_5 = candles[-6:-1]

    prev_high = max(x["high"] for x in previous_5)
    prev_low = min(x["low"] for x in previous_5)

    body = candle_body(latest)
    rng = candle_range(latest)

    body_ratio = body / rng

    direction = candle_direction(latest)

    # --------------------------------------------------------
    # Price Action
    # --------------------------------------------------------

    pa_up = 0
    pa_down = 0

    if latest["close"] > latest["open"]:
        pa_up += 1

    elif latest["close"] < latest["open"]:
        pa_down += 1

    if len(candles) >= 3:
        c1 = candles[-1]
        c2 = candles[-2]
        c3 = candles[-3]

        if (
            c1["close"] > c2["high"]
            and c2["close"] >= c3["close"]
        ):
            pa_up += 2

        elif (
            c1["close"] < c2["low"]
            and c2["close"] <= c3["close"]
        ):
            pa_down += 2

    pa_up = min(pa_up, 3)
    pa_down = min(pa_down, 3)

    # --------------------------------------------------------
    # Structure
    # --------------------------------------------------------

    structure_up = 0
    structure_down = 0

    highs = [x["high"] for x in candles[-8:]]
    lows = [x["low"] for x in candles[-8:]]

    if len(highs) >= 4:
        if highs[-1] >= highs[-3]:
            structure_up += 1

        if lows[-1] >= lows[-3]:
            structure_up += 1

        if highs[-1] <= highs[-3]:
            structure_down += 1

        if lows[-1] <= lows[-3]:
            structure_down += 1

    structure_up = min(structure_up, 3)
    structure_down = min(structure_down, 3)

    # --------------------------------------------------------
    # Liquidity
    # --------------------------------------------------------

    liquidity_up = 0
    liquidity_down = 0

    if latest["low"] < prev_low and latest["close"] > prev_low:
        liquidity_up = 2

    elif latest["high"] > prev_high and latest["close"] < prev_high:
        liquidity_down = 2

    # --------------------------------------------------------
    # Breakout / Retest
    # --------------------------------------------------------

    breakout_up = 0
    breakout_down = 0

    if latest["close"] > prev_high:
        breakout_up = 2

    elif latest["close"] < prev_low:
        breakout_down = 2

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    momentum_up = 0
    momentum_down = 0

    if len(candles) >= 5:
        momentum_change = (
            candles[-1]["close"]
            - candles[-5]["close"]
        )

        if momentum_change > 0:
            momentum_up = 2

        elif momentum_change < 0:
            momentum_down = 2

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    ema_up = 0
    ema_down = 0

    if e9 is not None and e21 is not None:
        if e9 > e21 and latest["close"] > e9:
            ema_up = 2

        elif e9 < e21 and latest["close"] < e9:
            ema_down = 2

    # --------------------------------------------------------
    # ADX + DI
    # --------------------------------------------------------

    adx_up = 0
    adx_down = 0

    if adxv is not None:
        if adxv >= 18:
            if pd is not None and md is not None:
                if pd > md:
                    adx_up = 2

                elif md > pd:
                    adx_down = 2

    # --------------------------------------------------------
    # Candle Strength
    # --------------------------------------------------------

    candle_up = 0
    candle_down = 0

    if body_ratio >= 0.60:
        if direction == "UP":
            candle_up = 1

        elif direction == "DOWN":
            candle_down = 1

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_up = 0
    rsi_down = 0

    if rv is not None:
        if 50 <= rv <= 68:
            rsi_up = 1

        elif 32 <= rv < 50:
            rsi_down = 1

    # --------------------------------------------------------
    # Williams %R
    # --------------------------------------------------------

    wr_up = 0
    wr_down = 0

    if wr is not None:
        if -80 <= wr <= -50:
            wr_up = 1

        elif -50 < wr <= -20:
            wr_down = 1

    # --------------------------------------------------------
    # Keltner
    # --------------------------------------------------------

    keltner_up = 0
    keltner_down = 0

    if e21 is not None and atrv is not None:
        k_upper = e21 + (atrv * 5)
        k_lower = e21 - (atrv * 5)

        if latest["close"] > e21:
            keltner_up = 1

        elif latest["close"] < e21:
            keltner_down = 1

        if latest["close"] >= k_upper:
            keltner_up = 1

        elif latest["close"] <= k_lower:
            keltner_down = 1

    # --------------------------------------------------------
    # Raw scores
    # --------------------------------------------------------

    up = (
        pa_up
        + structure_up
        + liquidity_up
        + breakout_up
        + momentum_up
        + ema_up
        + adx_up
        + candle_up
        + rsi_up
        + wr_up
        + keltner_up
    )

    down = (
        pa_down
        + structure_down
        + liquidity_down
        + breakout_down
        + momentum_down
        + ema_down
        + adx_down
        + candle_down
        + rsi_down
        + wr_down
        + keltner_down
    )

    # تحويل إلى توزيع /20
    total = up + down

    if total <= 0:
        up20 = 10
        down20 = 10
    else:
        up20 = round((up / total) * 20)
        down20 = 20 - up20

    if up20 >= down20:
        pre_direction = "UP"
        pre_score = up20
    else:
        pre_direction = "DOWN"
        pre_score = down20

    return {
        "symbol": symbol,
        "timeframe": ANALYSIS_TIMEFRAME,

        "latest": latest,

        "ema9": e9,
        "ema21": e21,

        "rsi": rv,
        "williams_r": wr,

        "atr": atrv,

        "adx": adxv,
        "plus_di": pd,
        "minus_di": md,

        "recent_high": recent_high,
        "recent_low": recent_low,

        "body_ratio": body_ratio,

        "up_score": up20,
        "down_score": down20,

        "pre_direction": pre_direction,
        "pre_score": pre_score,

        "factors": {
            "price_action_up": pa_up,
            "price_action_down": pa_down,

            "structure_up": structure_up,
            "structure_down": structure_down,

            "liquidity_up": liquidity_up,
            "liquidity_down": liquidity_down,

            "breakout_up": breakout_up,
            "breakout_down": breakout_down,

            "momentum_up": momentum_up,
            "momentum_down": momentum_down,

            "ema_up": ema_up,
            "ema_down": ema_down,

            "adx_up": adx_up,
            "adx_down": adx_down,

            "candle_up": candle_up,
            "candle_down": candle_down,

            "rsi_up": rsi_up,
            "rsi_down": rsi_down,

            "williams_up": wr_up,
            "williams_down": wr_down,

            "keltner_up": keltner_up,
            "keltner_down": keltner_down,
        },
    }


# ============================================================
# MARKET CANDIDATES
# ============================================================

def get_fresh_candidates():
    candidates = []

    now = time.time()

    with data_lock:
        items = list(market_data.items())

    for symbol, item in items:

        if normalize_timeframe(item.get("timeframe")) != ANALYSIS_TIMEFRAME:
            continue

        received_at = safe_float(
            item.get("received_at"),
            0
        )

        if now - received_at > MARKET_DATA_MAX_AGE_SECONDS:
            continue

        candles = item.get("candles") or []

        if len(candles) < MIN_CLOSED_CANDLES:
            continue

        snapshot = build_snapshot(
            symbol,
            candles
        )

        if snapshot:
            candidates.append(snapshot)

    candidates.sort(
        key=lambda x: x["pre_score"],
        reverse=True
    )

    return candidates


# ============================================================
# GEMINI
# ============================================================

def compact_candidate(candidate):
    latest = candidate["latest"]

    return {
        "symbol": candidate["symbol"],
        "timeframe": candidate["timeframe"],

        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "close": latest["close"],

        "ema9": candidate["ema9"],
        "ema21": candidate["ema21"],

        "rsi": candidate["rsi"],
        "williams_r": candidate["williams_r"],

        "atr": candidate["atr"],

        "adx": candidate["adx"],
        "plus_di": candidate["plus_di"],
        "minus_di": candidate["minus_di"],

        "recent_high": candidate["recent_high"],
        "recent_low": candidate["recent_low"],

        "body_ratio": candidate["body_ratio"],

        "pre_score": candidate["pre_score"],
        "pre_direction": candidate["pre_direction"],

        "up_score_pre": candidate["up_score"],
        "down_score_pre": candidate["down_score"],

        "factors": candidate["factors"],
    }


def analyze_candidates_with_gemini(candidates):
    if not gemini_client:
        logger.error("Gemini client unavailable.")
        return None

    if not candidates:
        return None

    data = [
        compact_candidate(x)
        for x in candidates[:TOP_CANDIDATES_FOR_AI]
    ]

    prompt = f"""
You are the final technical decision engine for ZinoProSignalAI.

The market timeframe is M1.

You are comparing several forex/OTC candidates.
Your job is to select EXACTLY ONE best opportunity.

IMPORTANT:
- Return only UP or DOWN.
- Never return WAIT.
- Never return NEUTRAL.
- Do not invent data.
- Use only supplied market data.
- Price Action and Market Structure have the highest priority.
- Then Breakout/Retest.
- Then Liquidity.
- Then Momentum.
- Then Candle Strength.
- EMA 9/21, RSI, Williams %R, Keltner and ADX/DI are supporting evidence.
- Do not give 90%+ confidence unless the confluence is exceptionally strong.
- Confidence must realistically reflect the evidence.
- The score must be EXACTLY /20.
- UP score + DOWN score MUST equal 20.

SCORING WEIGHTS:

Price Action = 3
Market Structure = 3
Liquidity = 2
Breakout/Retest = 2
Momentum = 2
EMA 9/21 = 2
ADX + DI = 2
Candle Strength = 1
RSI = 1
Williams %R = 1
Keltner = 1

TOTAL = 20

For every factor, allocate its points between UP and DOWN.

Example:
Price Action weight 3:
UP 3 / DOWN 0
or
UP 2 / DOWN 1
or
UP 1 / DOWN 2
or
UP 0 / DOWN 3

Do the same according to each factor's weight.

The final UP_SCORE + DOWN_SCORE must equal 20.

Choose the single strongest candidate.

ENTRY:
The entry price should normally be the latest supplied closed candle close.

CANCELLATION:
For UP:
cancel if a candle closes below the relevant entry/structure level.

For DOWN:
cancel if a candle closes above the relevant entry/structure level.

Do NOT output support/resistance levels in the final human card.
Only output one concise reason.

Return STRICT JSON only:

{{
  "symbol": "EURUSD",
  "timeframe": "M1",
  "direction": "UP",
  "confidence": 78,
  "up_score": 15,
  "down_score": 5,
  "entry_price": 1.12345,
  "cancellation_price": 1.12290,
  "reason": "Strong bullish structure with momentum and EMA alignment.",
  "quality": "STRONG"
}}

Candidates:

{json.dumps(data, ensure_ascii=False)}
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

        text = (response.text or "").strip()

        if not text:
            return None

        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        return result

    except Exception as e:
        logger.exception("Gemini analysis error: %s", e)
        return None


# ============================================================
# VALIDATE AI RESULT
# ============================================================

def validate_ai_result(result, candidates):
    if not isinstance(result, dict):
        return None

    symbol = normalize_symbol(
        result.get("symbol")
    )

    allowed = {
        normalize_symbol(x["symbol"])
        for x in candidates
    }

    if symbol not in allowed:
        return None

    direction = str(
        result.get("direction", "")
    ).upper().strip()

    if direction not in ("UP", "DOWN"):
        return None

    up = safe_int(
        result.get("up_score"),
        0
    )

    down = safe_int(
        result.get("down_score"),
        0
    )

    # يجب أن يكون /20
    if up < 0:
        up = 0

    if down < 0:
        down = 0

    if up > 20:
        up = 20

    if down > 20:
        down = 20

    total = up + down

    if total <= 0:
        return None

    # normalize to 20
    if total != 20:
        up = round((up / total) * 20)
        down = 20 - up

    if direction == "UP":
        score = up
    else:
        score = down

    if score < MIN_SIGNAL_SCORE:
        return None

    confidence = safe_int(
        result.get("confidence"),
        50
    )

    confidence = max(
        50,
        min(89, confidence)
    )

    # لا نسمح بـ 90+
    # إلا إذا كانت النتيجة شديدة القوة
    if score >= 19:
        confidence = min(92, max(confidence, 88))

    entry_price = safe_float(
        result.get("entry_price"),
        0
    )

    cancellation_price = safe_float(
        result.get("cancellation_price"),
        0
    )

    # إذا Gemini لم يعط سعراً صالحاً،
    # نستخدم آخر close.
    candidate = next(
        (
            x for x in candidates
            if normalize_symbol(x["symbol"]) == symbol
        ),
        None
    )

    if not candidate:
        return None

    latest_close = safe_float(
        candidate["latest"]["close"],
        0
    )

    if entry_price <= 0:
        entry_price = latest_close

    if cancellation_price <= 0:
        if direction == "UP":
            cancellation_price = candidate["latest"]["low"]
        else:
            cancellation_price = candidate["latest"]["high"]

    reason = str(
        result.get("reason")
        or "Strongest current market confluence."
    ).strip()

    if len(reason) > 220:
        reason = reason[:217] + "..."

    return {
        "symbol": symbol,
        "timeframe": ANALYSIS_TIMEFRAME,
        "direction": direction,

        "confidence": confidence,

        "up_score": up,
        "down_score": down,

        "entry_price": entry_price,
        "cancellation_price": cancellation_price,

        "reason": reason,

        "quality": str(
            result.get("quality")
            or "STRONG"
        ).strip().upper(),

        "score": score,

        "created_at": time.time(),
    }


# ============================================================
# FIND BEST TRADE
# ============================================================

def find_best_trade():
    candidates = get_fresh_candidates()

    if not candidates:
        logger.info("No fresh market candidates.")
        return None

    logger.info(
        "Fresh candidates: %s",
        ", ".join(
            f'{x["symbol"]}:{x["pre_score"]}/20'
            for x in candidates[:10]
        )
    )

    # Gemini يقارن فقط أفضل المرشحين
    ai_result = analyze_candidates_with_gemini(
        candidates[:TOP_CANDIDATES_FOR_AI]
    )

    if not ai_result:
        logger.warning("Gemini did not return a valid result.")
        return None

    final = validate_ai_result(
        ai_result,
        candidates[:TOP_CANDIDATES_FOR_AI]
    )

    if not final:
        logger.warning("Gemini result failed validation.")
        return None

    logger.info(
        "BEST TRADE: %s %s score=%s/20 confidence=%s",
        final["symbol"],
        final["direction"],
        final["score"],
        final["confidence"],
    )

    return final


# ============================================================
# SIGNAL CARD
# ============================================================

def format_signal(signal, trade_type="BASE", recovery_number=0):
    direction = signal["direction"]

    arrow = "🟢 UP" if direction == "UP" else "🔴 DOWN"

    if trade_type == "RECOVERY":
        cycle_text = f"🔁 RECOVERY {recovery_number}/{RECOVERY_LIMIT}"
    else:
        cycle_text = "🎯 BASE TRADE"

    now = now_algiers()

    # Entry after M1 = 1 minute
    entry_dt = now + timedelta(minutes=1)

    entry_time = entry_dt.strftime("%H:%M:%S")

    entry_price = signal["entry_price"]
    cancel_price = signal["cancellation_price"]

    if direction == "UP":
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{cancel_price:.6f}"
        )
    else:
        cancel_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{cancel_price:.6f}"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {signal['symbol']} | {signal['timeframe']}\n\n"
        f"{cycle_text}\n"
        f"{arrow}\n\n"
        f"🔥 Confidence: {signal['confidence']}%\n"
        f"🟢 UP Score: {signal['up_score']}/20\n"
        f"🔴 DOWN Score: {signal['down_score']}/20\n\n"
        "⏱️ Entry after: 1 minute\n"
        f"🕐 Entry Time: {entry_time} 🇩🇿\n"
        f"💰 Entry Price: {entry_price:.6f}\n"
        f"{cancel_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧠 {signal['reason']}"
    )


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_signal(application, signal, trade_type="BASE", recovery_number=0):
    global last_signal_sent_at

    if not signal:
        return False

    with signal_send_lock:

        now = time.time()

        # حماية من duplicate
        if now - last_signal_sent_at < 10:
            logger.warning("Signal send blocked: duplicate protection.")
            return False

        text = format_signal(
            signal,
            trade_type=trade_type,
            recovery_number=recovery_number,
        )

        try:
            await application.bot.send_message(
                chat_id=OWNER_ID,
                text=text,
            )

            last_signal_sent_at = now

            logger.info(
                "Signal sent: %s %s",
                signal["symbol"],
                signal["direction"],
            )

            return True

        except Exception as e:
            logger.exception(
                "Telegram signal send error: %s",
                e
            )

            return False


# ============================================================
# CYCLE MANAGEMENT
# ============================================================

def reset_cycle():
    global active_cycle
    global pending_recovery_candidate

    with data_lock:
        active_cycle = {
            "active": False,
            "symbol": None,
            "timeframe": ANALYSIS_TIMEFRAME,
            "direction": None,
            "entry_price": None,
            "entry_time": None,
            "trade_type": "BASE",
            "recovery_number": 0,
            "recovery_ready_at": None,
            "trade_id": None,
        }

        pending_recovery_candidate = None


def activate_base_trade(signal):
    global active_cycle

    with data_lock:
        active_cycle = {
            "active": True,

            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],

            "direction": signal["direction"],

            "entry_price": signal["entry_price"],

            "entry_time": time.time(),

            "trade_type": "BASE",

            "recovery_number": 0,

            "recovery_ready_at": None,

            "trade_id": str(
                int(time.time() * 1000)
            ),
        }


def activate_recovery_trade(signal):
    global active_cycle
    global pending_recovery_candidate

    with data_lock:
        active_cycle = {
            "active": True,

            "symbol": signal["symbol"],
            "timeframe": signal["timeframe"],

            "direction": signal["direction"],

            "entry_price": signal["entry_price"],

            "entry_time": time.time(),

            "trade_type": "RECOVERY",

            "recovery_number": 1,

            "recovery_ready_at": None,

            "trade_id": str(
                int(time.time() * 1000)
            ),
        }

        pending_recovery_candidate = None


def start_recovery_wait():
    global active_cycle

    ready_at = time.time() + RECOVERY_DELAY_SECONDS

    with data_lock:
        active_cycle["recovery_ready_at"] = ready_at
        active_cycle["trade_type"] = "RECOVERY"
        active_cycle["recovery_number"] = 1


def recovery_seconds_remaining():
    with data_lock:
        ready_at = active_cycle.get(
            "recovery_ready_at"
        )

    if not ready_at:
        return 0

    return max(
        0,
        int(ready_at - time.time())
    )


# ============================================================
# BATCH SCAN
# ============================================================

async def process_completed_batch(application):
    global last_analyzed_batch_id
    global pending_recovery_candidate

    with data_lock:
        batch_id = latest_batch_id
        complete = latest_batch_complete

        current_cycle = dict(active_cycle)

    if not complete:
        return

    if not batch_id:
        return

    if batch_id == last_analyzed_batch_id:
        return

    # نضعه مباشرة حتى لا تتكرر العملية
    last_analyzed_batch_id = batch_id

    logger.info(
        "Processing completed MT5 batch: %s",
        batch_id
    )

    # --------------------------------------------------------
    # أثناء Recovery:
    # نستمر في التحليل لكن لا نرسل حتى انتهاء الدقيقتين.
    # --------------------------------------------------------

    if current_cycle["active"]:

        if current_cycle["trade_type"] == "RECOVERY":
            remaining = recovery_seconds_remaining()

            if remaining > 0:
                candidate = find_best_trade()

                if candidate:
                    pending_recovery_candidate = candidate

                logger.info(
                    "Recovery waiting: %ss remaining.",
                    remaining
                )

                return

            # Recovery أصبحت جاهزة
            candidate = find_best_trade()

            if not candidate:
                candidate = pending_recovery_candidate

            if not candidate:
                logger.info(
                    "Recovery ready but no valid candidate yet."
                )

                # نرجع ننتظر Batch جديد
                return

            sent = await send_signal(
                application,
                candidate,
                trade_type="RECOVERY",
                recovery_number=1,
            )

            if sent:
                activate_recovery_trade(candidate)

            return

        # ----------------------------------------------------
        # Base trade active:
        # لا نرسل صفقة ثانية.
        # نستمر فقط في تحديث السوق.
        # ----------------------------------------------------

        logger.info(
            "Base trade active. Market scanned, no new signal."
        )

        return

    # --------------------------------------------------------
    # لا توجد صفقة نشطة -> ابحث عن أفضل صفقة
    # --------------------------------------------------------

    candidate = find_best_trade()

    if not candidate:
        logger.info(
            "No qualified trade from completed batch."
        )
        return

    sent = await send_signal(
        application,
        candidate,
        trade_type="BASE",
        recovery_number=0,
    )

    if sent:
        activate_base_trade(candidate)


# ============================================================
# BACKGROUND LOOP
# ============================================================

async def background_analysis_loop(application):
    logger.info("Background analysis loop started.")

    while True:
        try:

            await process_completed_batch(
                application
            )

            # ------------------------------------------------
            # Recovery timer
            # ------------------------------------------------

            with data_lock:
                cycle = dict(active_cycle)

            if (
                cycle["active"]
                and cycle["trade_type"] == "RECOVERY"
            ):
                ready_at = cycle.get(
                    "recovery_ready_at"
                )

                if ready_at:
                    if time.time() >= ready_at:

                        # نحتاج Batch جديد/حديث
                        # لكن إذا pending candidate موجود
                        # نحاول استعماله.
                        candidate = find_best_trade()

                        if not candidate:
                            candidate = pending_recovery_candidate

                        if candidate:
                            sent = await send_signal(
                                application,
                                candidate,
                                trade_type="RECOVERY",
                                recovery_number=1,
                            )

                            if sent:
                                activate_recovery_trade(
                                    candidate
                                )

        except Exception as e:
            logger.exception(
                "Background loop error: %s",
                e
            )

        await asyncio.sleep(
            BACKGROUND_INTERVAL_SECONDS
        )


# ============================================================
# MT5 DATA STORAGE
# ============================================================

def store_mt5_data(payload):
    global latest_batch_id
    global latest_batch_complete

    symbol = normalize_symbol(
        payload.get("symbol")
    )

    timeframe = normalize_timeframe(
        payload.get("timeframe")
    )

    candles = normalize_candles(
        payload.get("candles")
    )

    batch_id = str(
        payload.get("batch_id")
        or int(time.time())
    )

    batch_complete = bool(
        payload.get("batch_complete", False)
    )

    if not symbol:
        return False, "missing symbol"

    if timeframe != ANALYSIS_TIMEFRAME:
        return False, (
            f"unsupported timeframe: {timeframe}"
        )

    if len(candles) < MIN_CLOSED_CANDLES:
        return False, (
            f"not enough candles: {len(candles)}"
        )

    item = {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles[-EXPECTED_CANDLES:],
        "received_at": time.time(),
        "batch_id": batch_id,

        "source": payload.get(
            "source",
            "MT5"
        ),

        "current_bid": safe_float(
            payload.get("current_bid"),
            0
        ),

        "current_ask": safe_float(
            payload.get("current_ask"),
            0
        ),

        "digits": safe_int(
            payload.get("digits"),
            0
        ),
    }

    with data_lock:
        market_data[symbol] = item

        latest_batch_id = batch_id

        if batch_complete:
            latest_batch_complete = True

    logger.info(
        "MT5 DATA | %s | %s | candles=%s | batch=%s | complete=%s",
        symbol,
        timeframe,
        len(candles),
        batch_id,
        batch_complete,
    )

    return True, "stored"


# ============================================================
# HTTP SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        return

    def send_json(self, status, payload):
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

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path.rstrip("/")

        if path in ("", "/"):
            self.send_response(200)

            body = (
                "ZinoProSignalAI is running"
            ).encode("utf-8")

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

        if path in (
            "/health",
            "/healthz",
        ):
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "ZinoProSignalAI",
                    "status": "running",
                },
            )

            return

        if path in (
            "/mt4",
            "/api/mt4",
            "/mt5",
            "/api/mt5",
        ):
            self.send_json(
                200,
                {
                    "ok": True,
                    "service": "ZinoProSignalAI",
                    "endpoint": "market-data",
                    "source": "MT5",
                },
            )

            return

        self.send_json(
            404,
            {
                "ok": False,
                "error": "not_found",
            },
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path.rstrip("/")

        if path not in (
            "/mt4",
            "/api/mt4",
            "/mt5",
            "/api/mt5",
        ):
            self.send_json(
                404,
                {
                    "ok": False,
                    "error": "not_found",
                },
            )

            return

        # ----------------------------------------------------
        # API KEY
        # ----------------------------------------------------

        if MT4_API_KEY:

            received_key = (
                self.headers.get("X-MT4-API-Key")
                or self.headers.get("X-MT5-API-Key")
                or self.headers.get("X-API-Key")
                or ""
            ).strip()

            if received_key != MT4_API_KEY:
                logger.warning(
                    "Unauthorized market-data request."
                )

                self.send_json(
                    401,
                    {
                        "ok": False,
                        "error": "unauthorized",
                    },
                )

                return

        # ----------------------------------------------------
        # BODY
        # ----------------------------------------------------

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
                    "ok": False,
                    "error": "empty_body",
                },
            )

            return

        try:
            raw = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except Exception as e:
            logger.warning(
                "Invalid JSON: %s",
                e
            )

            self.send_json(
                400,
                {
                    "ok": False,
                    "error": "invalid_json",
                },
            )

            return

        if not isinstance(payload, dict):
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": "json_must_be_object",
                },
            )

            return

        ok, message = store_mt5_data(
            payload
        )

        if not ok:
            self.send_json(
                400,
                {
                    "ok": False,
                    "error": message,
                },
            )

            return

        symbol = normalize_symbol(
            payload.get("symbol")
        )

        timeframe = normalize_timeframe(
            payload.get("timeframe")
        )

        candles = normalize_candles(
            payload.get("candles")
        )

        self.send_json(
            200,
            {
                "ok": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(candles),
                "batch_id": str(
                    payload.get("batch_id", "")
                ),
                "batch_complete": bool(
                    payload.get(
                        "batch_complete",
                        False
                    )
                ),
                "analysis_timeframe": ANALYSIS_TIMEFRAME,
            },
        )


def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler,
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

    text = (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🟢 النظام يعمل\n\n"
        "📡 MT5 → Render → Gemini\n"
        "📊 Timeframe: M1\n"
        "🎯 اختيار أفضل صفقة واحدة\n"
        "🏆 Score: /20\n"
        "🔁 Recovery: 1/1\n"
        "⏱️ Recovery delay: 2 minutes\n"
        "🇩🇿 Timezone: Africa/Algiers\n\n"
        "الأوامر:\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset\n"
        "/mt5status"
    )

    await update.message.reply_text(
        text
    )


async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with stats_lock:
        total = (
            stats["wins"]
            + stats["losses"]
        )

        wins = stats["wins"]
        losses = stats["losses"]

        if total:
            winrate = (
                wins / total
            ) * 100
        else:
            winrate = 0

        text = (
            "📊 ZinoProSignalAI Stats\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"🟢 WIN: {wins}\n"
            f"🔴 LOSS: {losses}\n"
            f"📈 Win Rate: {winrate:.1f}%\n\n"
            f"🎯 Base WIN: {stats['base_wins']}\n"
            f"🎯 Base LOSS: {stats['base_losses']}\n"
            f"🔁 Recovery WIN: {stats['recovery_wins']}\n"
            f"🔁 Recovery LOSS: {stats['recovery_losses']}\n"
        )

    await update.message.reply_text(
        text
    )


async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with data_lock:
        trade_type = active_cycle.get(
            "trade_type",
            "BASE"
        )

        active = active_cycle.get(
            "active",
            False
        )

    if not active:
        await update.message.reply_text(
            "ℹ️ ما كاش صفقة نشطة."
        )
        return

    with stats_lock:
        stats["wins"] += 1

        if trade_type == "RECOVERY":
            stats["recovery_wins"] += 1
        else:
            stats["base_wins"] += 1

    save_history()

    reset_cycle()

    await update.message.reply_text(
        "🟢 WIN ✅\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "تم تسجيل الربح.\n"
        "🔎 البوت رجع يبحث عن أفضل صفقة جديدة."
    )


async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with data_lock:
        active = active_cycle.get(
            "active",
            False
        )

        trade_type = active_cycle.get(
            "trade_type",
            "BASE"
        )

    if not active:
        await update.message.reply_text(
            "ℹ️ ما كاش صفقة نشطة."
        )
        return

    with stats_lock:
        stats["losses"] += 1

        if trade_type == "RECOVERY":
            stats["recovery_losses"] += 1
        else:
            stats["base_losses"] += 1

    save_history()

    # --------------------------------------------------------
    # إذا كانت Recovery وخسرت:
    # لا توجد Recovery 2.
    # --------------------------------------------------------

    if trade_type == "RECOVERY":

        reset_cycle()

        await update.message.reply_text(
            "🔴 LOSS ❌\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "Recovery 1/1 خسرت.\n"
            "🚫 لا توجد Recovery 2.\n"
            "🔎 البوت رجع للبحث عن صفقة BASE جديدة."
        )

        return

    # --------------------------------------------------------
    # Base loss -> Recovery بعد دقيقتين
    # --------------------------------------------------------

    start_recovery_wait()

    with data_lock:
        pending_recovery_candidate = None

    await update.message.reply_text(
        "🔴 LOSS ❌\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "🔁 سيتم تفعيل Recovery 1/1.\n"
        "⏱️ الانتظار: دقيقتان بالضبط.\n"
        "📡 خلال الدقيقتين البوت يستمر في تحليل السوق.\n"
        "🎯 بعد انتهاء الدقيقتين سيختار أفضل فرصة حالية، "
        "وليس شرطًا نفس الزوج."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    reset_cycle()

    await update.message.reply_text(
        "♻️ تم Reset للدورة الحالية.\n"
        "🔎 البوت جاهز للبحث عن أفضل صفقة."
    )


async def mt5status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with data_lock:
        items = list(
            market_data.items()
        )

        batch = latest_batch_id
        complete = latest_batch_complete

        cycle = dict(active_cycle)

    if not items:
        await update.message.reply_text(
            "📡 لا توجد بيانات MT5 حتى الآن."
        )
        return

    now = time.time()

    lines = [
        "📡 MT5 STATUS",
        "━━━━━━━━━━━━━━━━━━",
        f"Batch: {batch}",
        f"Complete: {complete}",
        f"Pairs stored: {len(items)}",
        "",
    ]

    for symbol, item in sorted(items):
        age = int(
            now - item["received_at"]
        )

        candles = len(
            item.get("candles", [])
        )

        lines.append(
            f"• {symbol} | "
            f"{candles} candles | "
            f"{age}s ago"
        )

    lines.append("")
    lines.append(
        f"Cycle: "
        f"{'ACTIVE' if cycle['active'] else 'IDLE'}"
    )

    if cycle["active"]:
        lines.append(
            f"Trade: {cycle['trade_type']}"
        )

        lines.append(
            f"Symbol: {cycle['symbol']}"
        )

        if cycle["trade_type"] == "RECOVERY":
            lines.append(
                f"Recovery remaining: "
                f"{recovery_seconds_remaining()}s"
            )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    text = (
        update.message.text or ""
    ).strip().lower()

    if text in ("win", "ربح"):
        await win_command(
            update,
            context
        )

    elif text in ("loss", "خسارة"):
        await loss_command(
            update,
            context
        )


# ============================================================
# ERROR HANDLER
# ============================================================

async def telegram_error_handler(
    update,
    context
):
    logger.exception(
        "Telegram error: %s",
        context.error
    )


# ============================================================
# MAIN
# ============================================================

async def post_init(
    application: Application
):
    asyncio.create_task(
        background_analysis_loop(
            application
        )
    )

    logger.info(
        "Background task created."
    )


def main():

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing."
        )

    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is missing."
        )

    # HTTP server for Render
    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
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
            "mt5status",
            mt5status_command
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

    logger.info(
        "ZinoProSignalAI starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
