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

# Keep the old ENV name so Render does not need changing.
MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")

ANALYSIS_TIMEFRAME = "M1"

MIN_CLOSED_CANDLES = 40
EXPECTED_CANDLES = 150

TOP_CANDIDATES_FOR_AI = 5

MIN_SIGNAL_SCORE = 13

RECOVERY_LIMIT = 1
RECOVERY_DELAY_SECONDS = 120

MARKET_DATA_MAX_AGE_SECONDS = 130

BACKGROUND_INTERVAL_SECONDS = 5

HISTORY_FILE = "signal_history.json"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
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
    except Exception as exc:
        logger.exception(
            "Gemini initialization failed: %s",
            exc
        )


# ============================================================
# GLOBAL STATE
# ============================================================

state_lock = threading.RLock()

market_data = {}

latest_batch_id = None
latest_batch_complete = False
latest_batch_started_at = 0.0

processing_batch_id = None

# ------------------------------------------------------------
# ACTIVE TRADE
#
# None = no active trade, scanner may search
# BASE = normal trade
# RECOVERY = recovery trade
# ------------------------------------------------------------

active_trade = None

# ------------------------------------------------------------
# RECOVERY STATE
# ------------------------------------------------------------

recovery_wait_until = None
recovery_pending = False
recovery_number = 0

# ------------------------------------------------------------
# STATS
# ------------------------------------------------------------

stats = {
    "wins": 0,
    "losses": 0,
    "base_wins": 0,
    "base_losses": 0,
    "recovery_wins": 0,
    "recovery_losses": 0,
}

# ------------------------------------------------------------
# LAST SIGNAL
# ------------------------------------------------------------

last_signal = None

# ------------------------------------------------------------
# HISTORY
# ------------------------------------------------------------

history = []


# ============================================================
# FILE STORAGE
# ============================================================

def load_history():
    global history
    global stats

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
            saved_history = data.get(
                "history",
                []
            )

            saved_stats = data.get(
                "stats",
                {}
            )

            if isinstance(saved_history, list):
                history = saved_history

            if isinstance(saved_stats, dict):
                for key in stats:
                    if key in saved_stats:
                        try:
                            stats[key] = int(
                                saved_stats[key]
                            )
                        except Exception:
                            pass

        logger.info(
            "History loaded: %d records",
            len(history)
        )

    except Exception as exc:
        logger.exception(
            "History load failed: %s",
            exc
        )


def save_history():
    try:
        payload = {
            "history": history[-500:],
            "stats": stats,
        }

        temp_file = HISTORY_FILE + ".tmp"

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                payload,
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temp_file,
            HISTORY_FILE
        )

    except Exception as exc:
        logger.exception(
            "History save failed: %s",
            exc
        )


load_history()


# ============================================================
# HELPERS
# ============================================================

def is_owner(update: Update) -> bool:
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


def now_algiers() -> datetime:
    return datetime.now(ALGIERS)


def now_timestamp() -> float:
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
    return max(low, min(high, value))


def candle_close(candle):
    return safe_float(candle.get("close"))


def candle_open(candle):
    return safe_float(candle.get("open"))


def candle_high(candle):
    return safe_float(candle.get("high"))


def candle_low(candle):
    return safe_float(candle.get("low"))


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2.0 / (period + 1.0)

    result = sum(values[:period]) / period

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

    result = sum(values[:period]) / period

    output = [None] * (period - 1)
    output.append(result)

    for price in values[period:]:
        result = (
            (price - result) * multiplier
            + result
        )
        output.append(result)

    return output


def rsi(values, period=14):
    if len(values) <= period:
        return None

    gains = []
    losses = []

    for i in range(1, len(values)):
        change = values[i] - values[i - 1]

        gains.append(
            max(change, 0.0)
        )

        losses.append(
            max(-change, 0.0)
        )

    avg_gain = sum(
        gains[:period]
    ) / period

    avg_loss = sum(
        losses[:period]
    ) / period

    if avg_loss == 0:
        return 100.0

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

    if highest == lowest:
        return -50.0

    return (
        (highest - closes[-1])
        / (highest - lowest)
    ) * -100.0


def true_ranges(
    highs,
    lows,
    closes
):
    if not closes:
        return []

    output = []

    for i in range(len(closes)):
        if i == 0:
            output.append(
                highs[i] - lows[i]
            )
            continue

        tr = max(
            highs[i] - lows[i],
            abs(
                highs[i]
                - closes[i - 1]
            ),
            abs(
                lows[i]
                - closes[i - 1]
            ),
        )

        output.append(tr)

    return output


def atr(
    highs,
    lows,
    closes,
    period=10
):
    trs = true_ranges(
        highs,
        lows,
        closes
    )

    if len(trs) < period:
        return None

    return sum(
        trs[-period:]
    ) / period


def adx_di(
    highs,
    lows,
    closes,
    period=14
):
    if len(closes) < period + 2:
        return None, None, None

    trs = []
    plus_dm = []
    minus_dm = []

    for i in range(1, len(closes)):
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
            ),
        )

        trs.append(tr)

        if (
            up_move > down_move
            and up_move > 0
        ):
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if (
            down_move > up_move
            and down_move > 0
        ):
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

    if len(trs) < period:
        return None, None, None

    atr_value = sum(
        trs[-period:]
    ) / period

    if atr_value <= 0:
        return None, None, None

    plus_di = (
        100.0
        * (
            sum(plus_dm[-period:])
            / period
        )
        / atr_value
    )

    minus_di = (
        100.0
        * (
            sum(minus_dm[-period:])
            / period
        )
        / atr_value
    )

    denominator = (
        plus_di
        + minus_di
    )

    if denominator <= 0:
        return 0.0, plus_di, minus_di

    dx = (
        100.0
        * abs(
            plus_di
            - minus_di
        )
        / denominator
    )

    return dx, plus_di, minus_di


# ============================================================
# DETERMINISTIC MARKET SNAPSHOT
# ============================================================

def build_snapshot(symbol, data):
    candles = data.get("candles", [])

    if len(candles) < MIN_CLOSED_CANDLES:
        return None

    opens = [
        candle_open(c)
        for c in candles
    ]

    highs = [
        candle_high(c)
        for c in candles
    ]

    lows = [
        candle_low(c)
        for c in candles
    ]

    closes = [
        candle_close(c)
        for c in candles
    ]

    if not closes:
        return None

    current = closes[-1]

    previous = closes[-2]

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    rsi14 = rsi(
        closes,
        14
    )

    wr14 = williams_r(
        highs,
        lows,
        closes,
        14
    )

    atr10 = atr(
        highs,
        lows,
        closes,
        10
    )

    adx14, plus_di, minus_di = adx_di(
        highs,
        lows,
        closes,
        14
    )

    up = 0
    down = 0

    reasons = []

    # --------------------------------------------------------
    # PRICE ACTION 3
    # --------------------------------------------------------

    last_range = (
        highs[-1]
        - lows[-1]
    )

    last_body = abs(
        closes[-1]
        - opens[-1]
    )

    if (
        closes[-1] > opens[-1]
        and last_body > last_range * 0.55
    ):
        up += 3
        reasons.append(
            "strong bullish price action"
        )

    elif (
        closes[-1] < opens[-1]
        and last_body > last_range * 0.55
    ):
        down += 3
        reasons.append(
            "strong bearish price action"
        )

    else:
        if closes[-1] > previous:
            up += 1
        elif closes[-1] < previous:
            down += 1

    # --------------------------------------------------------
    # MARKET STRUCTURE 3
    # --------------------------------------------------------

    recent_high = max(
        highs[-6:-1]
    )

    recent_low = min(
        lows[-6:-1]
    )

    if current > recent_high:
        up += 3
        reasons.append(
            "bullish structure break"
        )

    elif current < recent_low:
        down += 3
        reasons.append(
            "bearish structure break"
        )

    else:
        if current > (
            sum(closes[-6:-1]) / 5
        ):
            up += 1
        elif current < (
            sum(closes[-6:-1]) / 5
        ):
            down += 1

    # --------------------------------------------------------
    # LIQUIDITY 2
    # --------------------------------------------------------

    if len(candles) >= 10:
        old_high = max(
            highs[-10:-2]
        )

        old_low = min(
            lows[-10:-2]
        )

        if (
            highs[-1] > old_high
            and closes[-1] < old_high
        ):
            down += 2
            reasons.append(
                "possible buy-side liquidity rejection"
            )

        elif (
            lows[-1] < old_low
            and closes[-1] > old_low
        ):
            up += 2
            reasons.append(
                "possible sell-side liquidity rejection"
            )

    # --------------------------------------------------------
    # BREAKOUT / RETEST 2
    # --------------------------------------------------------

    if current > recent_high:
        up += 2

    elif current < recent_low:
        down += 2

    # --------------------------------------------------------
    # MOMENTUM 2
    # --------------------------------------------------------

    if (
        len(closes) >= 4
        and closes[-1] > closes[-4]
    ):
        up += 2
        reasons.append(
            "bullish short-term momentum"
        )

    elif (
        len(closes) >= 4
        and closes[-1] < closes[-4]
    ):
        down += 2
        reasons.append(
            "bearish short-term momentum"
        )

    # --------------------------------------------------------
    # EMA 9/21 2
    # --------------------------------------------------------

    if ema9 is not None and ema21 is not None:
        if (
            ema9 > ema21
            and current > ema9
        ):
            up += 2

        elif (
            ema9 < ema21
            and current < ema9
        ):
            down += 2

    # --------------------------------------------------------
    # ADX + DI 2
    # --------------------------------------------------------

    if (
        adx14 is not None
        and plus_di is not None
        and minus_di is not None
    ):
        if (
            adx14 >= 20
            and plus_di > minus_di
        ):
            up += 2

        elif (
            adx14 >= 20
            and minus_di > plus_di
        ):
            down += 2

    # --------------------------------------------------------
    # CANDLE STRENGTH 1
    # --------------------------------------------------------

    if last_range > 0:
        body_ratio = (
            last_body
            / last_range
        )

        if body_ratio >= 0.65:
            if closes[-1] > opens[-1]:
                up += 1
            elif closes[-1] < opens[-1]:
                down += 1

    # --------------------------------------------------------
    # RSI 1
    # --------------------------------------------------------

    if rsi14 is not None:
        if (
            rsi14 >= 52
            and rsi14 <= 68
        ):
            up += 1

        elif (
            rsi14 <= 48
            and rsi14 >= 32
        ):
            down += 1

    # --------------------------------------------------------
    # WILLIAMS %R 1
    # --------------------------------------------------------

    if wr14 is not None:
        if wr14 > -50:
            up += 1
        elif wr14 < -50:
            down += 1

    # --------------------------------------------------------
    # KELTNER 1
    # EMA20 +/- ATR*5
    # --------------------------------------------------------

    ema20 = ema(
        closes,
        20
    )

    if (
        ema20 is not None
        and atr10 is not None
    ):
        upper = (
            ema20
            + (atr10 * 5.0)
        )

        lower = (
            ema20
            - (atr10 * 5.0)
        )

        if current > ema20 and current < upper:
            up += 1

        elif current < ema20 and current > lower:
            down += 1

    # --------------------------------------------------------
    # NORMALIZE TO 20
    #
    # Raw factors can exceed 20 because some secondary
    # factors overlap. We normalize directionally.
    # --------------------------------------------------------

    raw_up = up
    raw_down = down

    total_raw = raw_up + raw_down

    if total_raw <= 0:
        up_score = 10
        down_score = 10
    else:
        up_score = round(
            (raw_up / total_raw) * 20
        )

        down_score = 20 - up_score

    # Prevent fake 20/20.
    # A direction needs meaningful opposing evidence
    # before being allowed to reach 20.
    if up_score >= 20 and raw_down > 0:
        up_score = 19
        down_score = 1

    if down_score >= 20 and raw_up > 0:
        down_score = 19
        up_score = 1

    if up_score >= down_score:
        direction = "UP"
        best_score = up_score
    else:
        direction = "DOWN"
        best_score = down_score

    # --------------------------------------------------------
    # DETERMINISTIC CONFIDENCE
    # Keep 90+ rare.
    # --------------------------------------------------------

    confidence = 50 + (
        abs(up_score - down_score) * 2
    )

    if best_score >= 19:
        confidence += 2

    confidence = int(
        clamp(
            confidence,
            55,
            89
        )
    )

    return {
        "symbol": symbol,
        "current_price": safe_float(
            data.get("current_bid"),
            current
        ),
        "up_score": int(up_score),
        "down_score": int(down_score),
        "confidence": confidence,
        "direction": direction,
        "best_score": int(best_score),
        "ema9": ema9,
        "ema21": ema21,
        "rsi14": rsi14,
        "williams_r": wr14,
        "atr10": atr10,
        "adx14": adx14,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "reason": "; ".join(
            reasons[:4]
        ),
    }


# ============================================================
# MARKET DATA VALIDATION
# ============================================================

def is_fresh(data):
    received = safe_float(
        data.get(
            "received_at",
            0
        )
    )

    if received <= 0:
        return False

    return (
        now_timestamp() - received
        <= MARKET_DATA_MAX_AGE_SECONDS
    )


def get_candidates():
    candidates = []

    with state_lock:
        for symbol, data in market_data.items():
            if not is_fresh(data):
                continue

            snapshot = build_snapshot(
                symbol,
                data
            )

            if snapshot is None:
                continue

            if snapshot["best_score"] < MIN_SIGNAL_SCORE:
                continue

            candidates.append(snapshot)

    candidates.sort(
        key=lambda x: (
            x["best_score"],
            x["confidence"],
            abs(
                x["up_score"]
                - x["down_score"]
            ),
        ),
        reverse=True,
    )

    return candidates


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def build_ai_payload(candidates):
    output = []

    for candidate in candidates:
        output.append({
            "symbol": candidate["symbol"],
            "current_price": candidate["current_price"],
            "up_score": candidate["up_score"],
            "down_score": candidate["down_score"],
            "confidence": candidate["confidence"],
            "direction": candidate["direction"],
            "best_score": candidate["best_score"],
            "reason": candidate["reason"],
        })

    return output


def ask_gemini(candidates):
    if not gemini_client:
        return None

    if not candidates:
        return None

    payload = build_ai_payload(
        candidates
    )

    prompt = f"""
You are the final selector for a short-term M1 binary-options
market scanner.

The market data comes from MT5.
Do not invent indicators or prices.

You must select EXACTLY ONE candidate.

Priority:
1. Price Action
2. Market Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle Strength
7. EMA 9/21
8. RSI
9. Williams %R
10. Keltner
11. ADX/DI as trend confirmation

The deterministic technical engine has already scored every
candidate.

Important:
- Do NOT automatically choose the first candidate.
- Do NOT inflate confidence.
- 90%+ confidence is extremely rare.
- Never output WAIT, NEUTRAL or NO SIGNAL.
- Direction must be UP or DOWN.
- up_score + down_score MUST equal 20.
- Prefer the candidate with the strongest current confluence.
- Do not change the candidate's scores unless there is a clear
  technical reason.
- Do not invent candle values.
- Entry price must be based on the supplied current_price.

Candidates:

{json.dumps(payload, ensure_ascii=False)}

Return ONLY valid JSON:

{{
  "symbol": "PAIR",
  "direction": "UP or DOWN",
  "confidence": 55,
  "up_score": 10,
  "down_score": 10,
  "reason": "short technical reason"
}}
"""

    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        text = (response.text or "").strip()

        if not text:
            return None

        result = json.loads(text)

        if not isinstance(result, dict):
            return None

        return result

    except Exception as exc:
        logger.exception(
            "Gemini analysis failed: %s",
            exc
        )

        return None


# ============================================================
# FINAL SIGNAL VALIDATION
# ============================================================

def validate_ai_signal(ai_result, candidates):
    if not ai_result:
        return None

    symbol = str(
        ai_result.get(
            "symbol",
            ""
        )
    ).strip()

    direction = str(
        ai_result.get(
            "direction",
            ""
        )
    ).upper().strip()

    if direction not in (
        "UP",
        "DOWN"
    ):
        return None

    candidate = None

    for item in candidates:
        if item["symbol"] == symbol:
            candidate = item
            break

    if candidate is None:
        return None

    up_score = safe_int(
        ai_result.get(
            "up_score",
            candidate["up_score"]
        ),
        candidate["up_score"]
    )

    down_score = safe_int(
        ai_result.get(
            "down_score",
            candidate["down_score"]
        ),
        candidate["down_score"]
    )

    # --------------------------------------------------------
    # Scores MUST equal 20.
    # Use deterministic scores as authority.
    # --------------------------------------------------------

    up_score = candidate["up_score"]
    down_score = candidate["down_score"]

    if up_score + down_score != 20:
        up_score = clamp(
            up_score,
            0,
            20
        )

        down_score = 20 - up_score

    if direction == "UP":
        best_score = up_score
    else:
        best_score = down_score

    if best_score < MIN_SIGNAL_SCORE:
        return None

    confidence = safe_int(
        ai_result.get(
            "confidence",
            candidate["confidence"]
        ),
        candidate["confidence"]
    )

    # Never allow arbitrary Gemini inflation.
    confidence = min(
        confidence,
        candidate["confidence"]
    )

    confidence = int(
        clamp(
            confidence,
            55,
            89
        )
    )

    reason = str(
        ai_result.get(
            "reason",
            candidate["reason"]
        )
    ).strip()

    if not reason:
        reason = candidate["reason"]

    # --------------------------------------------------------
    # Current price MUST come from newest MT5 data.
    # --------------------------------------------------------

    with state_lock:
        fresh_data = market_data.get(
            symbol
        )

    if not fresh_data:
        return None

    if not is_fresh(fresh_data):
        return None

    entry_price = safe_float(
        fresh_data.get(
            "current_bid",
            candidate["current_price"]
        ),
        candidate["current_price"]
    )

    # --------------------------------------------------------
    # Cancellation level
    # Use ATR-based protective distance.
    # --------------------------------------------------------

    atr_value = candidate.get(
        "atr10"
    )

    if not atr_value or atr_value <= 0:
        atr_value = abs(
            entry_price * 0.0005
        )

    cancellation_distance = (
        atr_value * 0.45
    )

    if direction == "UP":
        cancellation_price = (
            entry_price
            - cancellation_distance
        )
    else:
        cancellation_price = (
            entry_price
            + cancellation_distance
        )

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "up_score": int(up_score),
        "down_score": int(down_score),
        "entry_price": entry_price,
        "cancellation_price": cancellation_price,
        "reason": reason,
        "created_at": now_timestamp(),
    }


# ============================================================
# SIGNAL CARD
# ============================================================

def format_signal(signal, mode):
    direction = signal["direction"]

    direction_icon = (
        "🟢 UP"
        if direction == "UP"
        else "🔴 DOWN"
    )

    symbol = signal["symbol"]

    entry_time = (
        now_algiers()
        + timedelta(minutes=1)
    )

    entry_time_text = entry_time.strftime(
        "%H:%M:%S"
    )

    mode_text = (
        "🎯 BASE TRADE"
        if mode == "BASE"
        else "🔁 RECOVERY 1/1"
    )

    if direction == "UP":
        cancellation_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة تحت "
            f"{signal['cancellation_price']:.6f}"
        )
    else:
        cancellation_text = (
            f"⚠️ إلغاء إذا أغلقت الشمعة فوق "
            f"{signal['cancellation_price']:.6f}"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {symbol} | M1\n\n"
        f"{mode_text}\n"
        f"{direction_icon}\n\n"
        f"🔥 Confidence: {signal['confidence']}%\n"
        f"🟢 UP Score: {signal['up_score']}/20\n"
        f"🔴 DOWN Score: {signal['down_score']}/20\n\n"
        "⏱️ Entry after: 1 minute\n"
        f"🕐 Entry Time: {entry_time_text} 🇩🇿\n"
        f"💰 Entry Price: {signal['entry_price']:.6f}\n"
        f"{cancellation_text}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🧠 {signal['reason']}"
    )


# ============================================================
# SEND SIGNAL
# ============================================================

async def send_signal(
    application,
    signal,
    mode
):
    global active_trade
    global last_signal

    # --------------------------------------------------------
    # CRITICAL LOCK:
    # Never send another signal while active_trade exists.
    # --------------------------------------------------------

    with state_lock:
        if active_trade is not None:
            logger.info(
                "SIGNAL BLOCKED: active trade already exists"
            )
            return False

        active_trade = {
            "mode": mode,
            "signal": signal,
            "created_at": now_timestamp(),
            "result": None,
        }

        last_signal = signal.copy()

    message = format_signal(
        signal,
        mode
    )

    try:
        await application.bot.send_message(
            chat_id=OWNER_ID,
            text=message
        )

        logger.info(
            "SIGNAL SENT: %s %s | %s",
            mode,
            signal["symbol"],
            signal["direction"]
        )

        return True

    except Exception as exc:
        logger.exception(
            "Telegram signal send failed: %s",
            exc
        )

        # Unlock if sending failed.
        with state_lock:
            active_trade = None

        return False


# ============================================================
# PROCESS COMPLETE BATCH
# ============================================================

async def process_complete_batch(
    application
):
    global processing_batch_id

    with state_lock:
        if not latest_batch_complete:
            return

        batch_id = latest_batch_id

        if not batch_id:
            return

        if processing_batch_id == batch_id:
            return

        processing_batch_id = batch_id

        # ----------------------------------------------------
        # HARD LOCK:
        # If a trade is active, DO NOT send anything.
        # ----------------------------------------------------

        if active_trade is not None:
            processing_batch_id = None
            return

    try:
        # ----------------------------------------------------
        # Recovery waiting period
        # ----------------------------------------------------

        with state_lock:
            recovery_active = (
                recovery_pending
            )
            wait_until = (
                recovery_wait_until
            )

        if recovery_active:
            if (
                wait_until is not None
                and now_timestamp()
                < wait_until
            ):
                # Continue scanning silently.
                logger.info(
                    "RECOVERY WAIT: market scanning continues"
                )
                return

        candidates = get_candidates()

        if not candidates:
            logger.info(
                "No valid candidates"
            )
            return

        # ----------------------------------------------------
        # TOP 5 ONLY
        # ----------------------------------------------------

        top_candidates = candidates[
            :TOP_CANDIDATES_FOR_AI
        ]

        ai_result = ask_gemini(
            top_candidates
        )

        signal = validate_ai_signal(
            ai_result,
            top_candidates
        )

        if signal is None:
            logger.info(
                "No valid final signal"
            )
            return

        # ----------------------------------------------------
        # Decide BASE or RECOVERY
        # ----------------------------------------------------

        with state_lock:
            if active_trade is not None:
                return

            if recovery_pending:
                mode = "RECOVERY"

                # Recovery wait is now consumed.
                recovery_pending = False
                recovery_wait_until = None

            else:
                mode = "BASE"

        await send_signal(
            application,
            signal,
            mode
        )

    finally:
        with state_lock:
            processing_batch_id = None


# ============================================================
# BACKGROUND LOOP
# ============================================================

async def background_loop(
    application
):
    while True:
        try:
            await process_complete_batch(
                application
            )

        except Exception as exc:
            logger.exception(
                "Background processing error: %s",
                exc
            )

        await asyncio.sleep(
            BACKGROUND_INTERVAL_SECONDS
        )


# ============================================================
# RECORD WIN
# ============================================================

async def handle_win(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number

    if not is_owner(update):
        return

    with state_lock:
        trade = active_trade

        if trade is None:
            await update.message.reply_text(
                "لا توجد صفقة نشطة."
            )
            return

        mode = trade["mode"]

        stats["wins"] += 1

        if mode == "BASE":
            stats["base_wins"] += 1
        else:
            stats["recovery_wins"] += 1

        history.append({
            "time": now_algiers().isoformat(),
            "result": "WIN",
            "mode": mode,
            "signal": trade["signal"],
        })

        # ----------------------------------------------------
        # WIN always resets the cycle.
        # ----------------------------------------------------

        active_trade = None
        recovery_pending = False
        recovery_wait_until = None
        recovery_number = 0

        save_history()

    await update.message.reply_text(
        "✅ WIN مسجلة.\n"
        "🔓 انتهت الصفقة.\n"
        "🔎 البوت رجع للبحث عن BASE TRADE جديدة."
    )


# ============================================================
# RECORD LOSS
# ============================================================

async def handle_loss(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number

    if not is_owner(update):
        return

    with state_lock:
        trade = active_trade

        if trade is None:
            await update.message.reply_text(
                "لا توجد صفقة نشطة."
            )
            return

        mode = trade["mode"]

        stats["losses"] += 1

        if mode == "BASE":
            stats["base_losses"] += 1
        else:
            stats["recovery_losses"] += 1

        history.append({
            "time": now_algiers().isoformat(),
            "result": "LOSS",
            "mode": mode,
            "signal": trade["signal"],
        })

        active_trade = None

        # ----------------------------------------------------
        # BASE LOSS -> Recovery 1/1
        # ----------------------------------------------------

        if mode == "BASE":
            recovery_number = 1

            recovery_pending = True

            recovery_wait_until = (
                now_timestamp()
                + RECOVERY_DELAY_SECONDS
            )

            save_history()

            wait_time = (
                now_algiers()
                + timedelta(
                    seconds=RECOVERY_DELAY_SECONDS
                )
            ).strftime("%H:%M:%S")

            await update.message.reply_text(
                "❌ LOSS مسجلة.\n\n"
                "🔁 RECOVERY 1/1\n"
                "⏳ الانتظار: دقيقتان\n"
                f"🕐 Recovery بعد: {wait_time} 🇩🇿\n\n"
                "🔎 خلال الدقيقتين البوت يواصل تحليل السوق "
                "لكن لن يرسل أي صفقة.\n"
                "بعد انتهاء الدقيقتين يختار أفضل فرصة حالية."
            )

            return

        # ----------------------------------------------------
        # RECOVERY LOSS -> NO RECOVERY 2
        # ----------------------------------------------------

        recovery_pending = False
        recovery_wait_until = None
        recovery_number = 0

        save_history()

    await update.message.reply_text(
        "❌ RECOVERY LOSS.\n\n"
        "🚫 لا توجد Recovery 2.\n"
        "🔎 البوت رجع مباشرة للبحث عن BASE TRADE جديدة."
    )


# ============================================================
# STATS
# ============================================================

async def handle_stats(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with state_lock:
        wins = stats["wins"]
        losses = stats["losses"]

        total = wins + losses

        if total > 0:
            winrate = (
                wins / total
            ) * 100
        else:
            winrate = 0.0

        if active_trade:
            active_text = (
                f"{active_trade['mode']} | "
                f"{active_trade['signal']['symbol']} | "
                f"{active_trade['signal']['direction']}"
            )
        else:
            active_text = "لا توجد صفقة"

        if recovery_pending:
            if recovery_wait_until:
                remaining = max(
                    0,
                    int(
                        recovery_wait_until
                        - now_timestamp()
                    )
                )
            else:
                remaining = 0

            recovery_text = (
                f"Recovery 1/1 قيد الانتظار "
                f"({remaining}s)"
            )
        else:
            recovery_text = "غير نشطة"

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"✅ WIN: {wins}\n"
        f"❌ LOSS: {losses}\n"
        f"📈 Win Rate: {winrate:.1f}%\n\n"
        f"🟢 Base WIN: {stats['base_wins']}\n"
        f"🔴 Base LOSS: {stats['base_losses']}\n"
        f"🔁 Recovery WIN: {stats['recovery_wins']}\n"
        f"🔁 Recovery LOSS: {stats['recovery_losses']}\n\n"
        f"🎯 Active: {active_text}\n"
        f"⏳ Recovery: {recovery_text}"
    )


# ============================================================
# RESET
# ============================================================

async def handle_reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    global active_trade
    global recovery_pending
    global recovery_wait_until
    global recovery_number
    global last_signal

    if not is_owner(update):
        return

    with state_lock:
        for key in stats:
            stats[key] = 0

        history.clear()

        active_trade = None

        recovery_pending = False
        recovery_wait_until = None
        recovery_number = 0

        last_signal = None

        save_history()

    await update.message.reply_text(
        "♻️ تم Reset بالكامل.\n"
        "🔎 البوت الآن يبحث عن BASE TRADE جديدة."
    )


# ============================================================
# MT5 STATUS
# ============================================================

async def handle_mt5status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    with state_lock:
        items = []

        for symbol, data in market_data.items():
            age = (
                now_timestamp()
                - safe_float(
                    data.get(
                        "received_at",
                        0
                    )
                )
            )

            items.append(
                f"{symbol}: {age:.0f}s"
            )

        batch = latest_batch_id or "none"

    if not items:
        text = (
            "📡 MT5 Status\n"
            "━━━━━━━━━━━━━━━━━━\n"
            "لا توجد بيانات من MT5."
        )
    else:
        text = (
            "📡 MT5 Status\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"Batch: {batch}\n\n"
            + "\n".join(items)
        )

    await update.message.reply_text(
        text
    )


# ============================================================
# START
# ============================================================

async def handle_start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):
    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "MT5 Market Scanner جاهز.\n\n"
        "/stats - الإحصائيات\n"
        "/win - تسجيل ربح\n"
        "/loss - تسجيل خسارة\n"
        "/reset - تصفير الدورة\n"
        "/mt5status - حالة MT5"
    )


# ============================================================
# HTTP HEALTH + MT5 API
# ============================================================

class RequestHandler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format,
        *args
    ):
        return

    def _send_json(
        self,
        code,
        payload
    ):
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

        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(
            self.path
        )

        if parsed.path in (
            "/",
            "/health",
            "/api/health",
        ):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                    "time": now_algiers().isoformat(),
                }
            )
            return

        if parsed.path == "/mt5status":
            with state_lock:
                symbols = list(
                    market_data.keys()
                )

            self._send_json(
                200,
                {
                    "status": "ok",
                    "symbols": symbols,
                    "count": len(symbols),
                }
            )
            return

        self._send_json(
            404,
            {
                "error": "not_found"
            }
        )

    def do_POST(self):
        parsed = urlparse(
            self.path
        )

        if parsed.path not in (
            "/mt4",
            "/api/mt4",
            "/mt5",
            "/api/mt5",
        ):
            self._send_json(
                404,
                {
                    "error": "not_found"
                }
            )
            return

        # --------------------------------------------------------
        # API KEY
        # --------------------------------------------------------

        supplied_key = (
            self.headers.get(
                "X-MT4-API-Key"
            )
            or self.headers.get(
                "X-MT5-API-Key"
            )
            or self.headers.get(
                "X-API-Key"
            )
            or ""
        ).strip()

        if (
            not MT4_API_KEY
            or supplied_key != MT4_API_KEY
        ):
            logger.warning(
                "Unauthorized MT5 request"
            )

            self._send_json(
                401,
                {
                    "error": "unauthorized"
                }
            )

            return

        # --------------------------------------------------------
        # READ BODY
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
            self._send_json(
                400,
                {
                    "error": "empty_body"
                }
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

        except Exception as exc:
            logger.warning(
                "Invalid JSON: %s",
                exc
            )

            self._send_json(
                400,
                {
                    "error": "invalid_json"
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

        candles = payload.get(
            "candles",
            []
        )

        batch_id = str(
            payload.get(
                "batch_id",
                ""
            )
        ).strip()

        batch_complete = bool(
            payload.get(
                "batch_complete",
                False
            )
        )

        if not symbol:
            self._send_json(
                400,
                {
                    "error": "missing_symbol"
                }
            )
            return

        if timeframe != "M1":
            self._send_json(
                400,
                {
                    "error": "only_M1_supported"
                }
            )
            return

        if not isinstance(
            candles,
            list
        ):
            self._send_json(
                400,
                {
                    "error": "candles_must_be_list"
                }
            )
            return

        if len(candles) < MIN_CLOSED_CANDLES:
            self._send_json(
                400,
                {
                    "error": "not_enough_candles",
                    "minimum": MIN_CLOSED_CANDLES,
                }
            )
            return

        if not batch_id:
            self._send_json(
                400,
                {
                    "error": "missing_batch_id"
                }
            )
            return

        # --------------------------------------------------------
        # STORE DATA
        # --------------------------------------------------------

        with state_lock:
            global latest_batch_id
            global latest_batch_complete
            global latest_batch_started_at

            if (
                latest_batch_id != batch_id
            ):
                latest_batch_id = batch_id

                latest_batch_complete = False

                latest_batch_started_at = (
                    now_timestamp()
                )

            market_data[symbol] = {
                "source": "MT5",
                "symbol": symbol,
                "timeframe": "M1",
                "batch_id": batch_id,
                "batch_complete": batch_complete,
                "current_bid": safe_float(
                    payload.get(
                        "current_bid"
                    )
                ),
                "current_ask": safe_float(
                    payload.get(
                        "current_ask"
                    )
                ),
                "digits": safe_int(
                    payload.get(
                        "digits"
                    ),
                    5
                ),
                "candles": candles[
                    -EXPECTED_CANDLES:
                ],
                "received_at": now_timestamp(),
            }

            if batch_complete:
                latest_batch_complete = True

        logger.info(
            "MT5 DATA | %s | candles=%d | batch=%s | complete=%s",
            symbol,
            len(candles),
            batch_id,
            batch_complete,
        )

        self._send_json(
            200,
            {
                "status": "ok",
                "symbol": symbol,
                "batch_id": batch_id,
                "batch_complete": batch_complete,
                "candles": len(candles),
            }
        )


# ============================================================
# START HTTP SERVER
# ============================================================

def start_http_server():
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        RequestHandler
    )

    logger.info(
        "HTTP server listening on port %d",
        PORT
    )

    server.serve_forever()


# ============================================================
# MAIN
# ============================================================

async def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN is missing"
        )

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing"
        )

    if not OWNER_ID:
        raise RuntimeError(
            "OWNER_ID is missing"
        )

    if not MT4_API_KEY:
        raise RuntimeError(
            "MT4_API_KEY is missing"
        )

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True
    )

    http_thread.start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            handle_start
        )
    )

    application.add_handler(
        CommandHandler(
            "stats",
            handle_stats
        )
    )

    application.add_handler(
        CommandHandler(
            "win",
            handle_win
        )
    )

    application.add_handler(
        CommandHandler(
            "loss",
            handle_loss
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            handle_reset
        )
    )

    application.add_handler(
        CommandHandler(
            "mt5status",
            handle_mt5status
        )
    )

    background_task = None

    try:
        await application.initialize()

        await application.start()

        await application.updater.start_polling()

        background_task = asyncio.create_task(
            background_loop(
                application
            )
        )

        logger.info(
            "ZinoProSignalAI Telegram bot started"
        )

        await asyncio.Event().wait()

    finally:
        if background_task:
            background_task.cancel()

            try:
                await background_task
            except asyncio.CancelledError:
                pass

        await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info(
            "Bot stopped"
        )
