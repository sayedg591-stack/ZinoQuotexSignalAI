import os
import json
import time
import threading
import asyncio
import logging
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from google import genai
from google.genai import types

# ============================================================
# ZinoProSignalAI - MT5 -> Render -> Telegram
# Strict, low-frequency signal engine
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = os.getenv("OWNER_ID", "").strip()
MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
PORT = int(os.getenv("PORT", "10000"))

ANALYSIS_TIMEFRAME = "M1"
MIN_CLOSED_CANDLES = 40
HISTORY_DISPLAY_COUNT = 10
SIGNAL_COOLDOWN_SECONDS = 60
SETUP_REPEAT_BLOCK_SECONDS = 360
AUTO_ANALYSIS_INTERVAL_SECONDS = 30
RECOVERY_LIMIT = 1
MAX_DATA_AGE_SECONDS = 150
ALGIERS_TZ = ZoneInfo("Africa/Algiers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ZinoProSignalAI")

try:
    OWNER_ID_INT = int(OWNER_ID) if OWNER_ID else 0
except ValueError:
    OWNER_ID_INT = 0

telegram_application = None
telegram_loop = None
auto_thread_started = False
state_lock = threading.RLock()
mt5_lock = threading.RLock()

mt5_data = {}
stats = {"wins": 0, "losses": 0, "signals": 0}
history = []
active_cycle = None
last_signal_time = 0.0
last_setup = None

# ------------------------------------------------------------
# Persistence
# ------------------------------------------------------------
STATE_FILE = os.getenv("STATE_FILE", "/tmp/zinopro_state.json")


def load_state():
    global stats, history, active_cycle, last_signal_time, last_setup
    try:
        if not os.path.exists(STATE_FILE):
            return
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        with state_lock:
            stats = data.get("stats", stats)
            history = data.get("history", [])[-100:]
            active_cycle = data.get("active_cycle")
            last_signal_time = float(data.get("last_signal_time", 0))
            last_setup = data.get("last_setup")
        logger.info("State loaded")
    except Exception:
        logger.exception("Could not load state")


def save_state():
    try:
        tmp = STATE_FILE + ".tmp"
        with state_lock:
            data = {
                "stats": stats,
                "history": history[-100:],
                "active_cycle": active_cycle,
                "last_signal_time": last_signal_time,
                "last_setup": last_setup,
            }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, STATE_FILE)
    except Exception:
        logger.exception("Could not save state")


# ------------------------------------------------------------
# Time / helpers
# ------------------------------------------------------------

def now_algiers():
    return datetime.now(ALGIERS_TZ)


def parse_timeframe_minutes(tf):
    tf = str(tf or ANALYSIS_TIMEFRAME).upper().strip()
    if tf.startswith("M"):
        try:
            return max(1, int(tf[1:]))
        except ValueError:
            return 1
    if tf.startswith("H"):
        try:
            return max(1, int(tf[1:])) * 60
        except ValueError:
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


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ------------------------------------------------------------
# Indicator calculations
# ------------------------------------------------------------

def ema(values, period):
    values = [safe_float(x) for x in values]
    if len(values) < period:
        return None
    result = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for value in values[period:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def ema_series(values, period):
    values = [safe_float(x) for x in values]
    if len(values) < period:
        return []
    out = [None] * (period - 1)
    current = sum(values[:period]) / period
    out.append(current)
    alpha = 2.0 / (period + 1.0)
    for value in values[period:]:
        current = alpha * value + (1.0 - alpha) * current
        out.append(current)
    return out


def rsi(values, period=14):
    values = [safe_float(x) for x in values]
    if len(values) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_series(values, period=14):
    values = [safe_float(x) for x in values]
    if len(values) <= period:
        return []
    gains = []
    losses = []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    result = [None] * period
    result.append(100.0 if avg_loss == 0 and avg_gain > 0 else 50.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)))
    for i in range(period, len(gains)):
        avg_gain = ((avg_gain * (period - 1)) + gains[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + losses[i]) / period
        value = 100.0 if avg_loss == 0 and avg_gain > 0 else 50.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        result.append(value)
    return result


def williams_r(candles, period=14):
    if len(candles) < period:
        return None
    window = candles[-period:]
    highest = max(safe_float(c["high"]) for c in window)
    lowest = min(safe_float(c["low"]) for c in window)
    close = safe_float(window[-1]["close"])
    if highest == lowest:
        return -50.0
    return -100.0 * (highest - close) / (highest - lowest)


def williams_r_previous(candles, period=14):
    if len(candles) < period + 1:
        return None
    return williams_r(candles[:-1], period)


def true_ranges(candles):
    out = []
    prev_close = None
    for c in candles:
        high = safe_float(c["high"])
        low = safe_float(c["low"])
        close = safe_float(c["close"])
        if prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        out.append(max(0.0, tr))
        prev_close = close
    return out


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
        cur = candles[i]
        prev = candles[i - 1]
        high = safe_float(cur["high"])
        low = safe_float(cur["low"])
        prev_high = safe_float(prev["high"])
        prev_low = safe_float(prev["low"])
        prev_close = safe_float(prev["close"])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        up_move = high - prev_high
        down_move = prev_low - low
        pdm = up_move if up_move > down_move and up_move > 0 else 0.0
        mdm = down_move if down_move > up_move and down_move > 0 else 0.0
        trs.append(tr)
        plus_dm.append(pdm)
        minus_dm.append(mdm)
    if len(trs) < period:
        return None, None, None
    tr_avg = sum(trs[-period:]) / period
    p_avg = sum(plus_dm[-period:]) / period
    m_avg = sum(minus_dm[-period:]) / period
    if tr_avg == 0:
        return 0.0, 0.0, 0.0
    plus_di = 100.0 * p_avg / tr_avg
    minus_di = 100.0 * m_avg / tr_avg
    denom = plus_di + minus_di
    dx = 100.0 * abs(plus_di - minus_di) / denom if denom else 0.0
    return dx, plus_di, minus_di


# ------------------------------------------------------------
# Structure / price action
# ------------------------------------------------------------

def market_structure(candles):
    if len(candles) < 8:
        return "NEUTRAL"
    window = candles[-8:]
    first = window[:4]
    last = window[4:]
    first_high = max(safe_float(c["high"]) for c in first)
    last_high = max(safe_float(c["high"]) for c in last)
    first_low = min(safe_float(c["low"]) for c in first)
    last_low = min(safe_float(c["low"]) for c in last)
    if last_high > first_high and last_low > first_low:
        return "BULLISH"
    if last_high < first_high and last_low < first_low:
        return "BEARISH"
    return "NEUTRAL"


def breakout_state(candles):
    if len(candles) < 9:
        return "NONE"
    last = candles[-1]
    prior = candles[-9:-1]
    prior_high = max(safe_float(c["high"]) for c in prior)
    prior_low = min(safe_float(c["low"]) for c in prior)
    close = safe_float(last["close"])
    high = safe_float(last["high"])
    low = safe_float(last["low"])
    if close > prior_high:
        return "BULL_BREAKOUT"
    if close < prior_low:
        return "BEAR_BREAKOUT"
    if high > prior_high and close < prior_high:
        return "BEAR_REJECTION"
    if low < prior_low and close > prior_low:
        return "BULL_REJECTION"
    return "NONE"


def momentum_state(candles):
    if len(candles) < 4:
        return "NEUTRAL"
    closes = [safe_float(c["close"]) for c in candles[-4:]]
    ups = sum(closes[i] > closes[i - 1] for i in range(1, 4))
    downs = sum(closes[i] < closes[i - 1] for i in range(1, 4))
    if ups >= 2 and ups > downs:
        return "UP"
    if downs >= 2 and downs > ups:
        return "DOWN"
    return "NEUTRAL"


def liquidity_sweep(candles):
    if len(candles) < 9:
        return "NONE"
    last = candles[-1]
    prior = candles[-9:-1]
    prior_high = max(safe_float(c["high"]) for c in prior)
    prior_low = min(safe_float(c["low"]) for c in prior)
    high = safe_float(last["high"])
    low = safe_float(last["low"])
    close = safe_float(last["close"])
    if low < prior_low and close > prior_low:
        return "BULL_SWEEP"
    if high > prior_high and close < prior_high:
        return "BEAR_SWEEP"
    return "NONE"


# ------------------------------------------------------------
# Snapshot
# ------------------------------------------------------------

def technical_snapshot(candles):
    closes = [safe_float(c["close"]) for c in candles]
    close = closes[-1]
    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    ema20 = ema(closes, 20)
    rsi14 = rsi(closes, 14)
    wr14 = williams_r(candles, 14)
    wr_prev = williams_r_previous(candles, 14)
    atr10 = atr(candles, 10)
    adx, plus_di, minus_di = adx_di(candles, 14)
    structure = market_structure(candles)
    breakout = breakout_state(candles)
    momentum = momentum_state(candles)
    sweep = liquidity_sweep(candles)
    k_mid = ema20
    k_upper = (ema20 + atr10 * 5.0) if ema20 is not None and atr10 is not None else None
    k_lower = (ema20 - atr10 * 5.0) if ema20 is not None and atr10 is not None else None
    last = candles[-1]
    body = safe_float(last["close"]) - safe_float(last["open"])
    return {
        "close": close,
        "ema9": ema9,
        "ema21": ema21,
        "ema20": ema20,
        "rsi14": rsi14,
        "williams_r14": wr14,
        "williams_r14_previous": wr_prev,
        "atr10": atr10,
        "adx14": adx,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "keltner_mid": k_mid,
        "keltner_upper": k_upper,
        "keltner_lower": k_lower,
        "structure": structure,
        "breakout": breakout,
        "momentum": momentum,
        "liquidity_sweep": sweep,
        "last_body": "UP" if body > 0 else "DOWN" if body < 0 else "FLAT",
        "recent8_high": max(safe_float(c["high"]) for c in candles[-8:]),
        "recent8_low": min(safe_float(c["low"]) for c in candles[-8:]),
    }


# ------------------------------------------------------------
# Strict deterministic engine
# ------------------------------------------------------------

def directional_engine(candles, snap):
    up = 0
    down = 0
    up_reasons = []
    down_reasons = []

    close = snap["close"]
    ema9 = snap["ema9"]
    ema21 = snap["ema21"]
    ema20 = snap["ema20"]
    rsi14 = snap["rsi14"]
    wr = snap["williams_r14"]
    wr_prev = snap["williams_r14_previous"]
    adx = snap["adx14"]
    plus_di = snap["plus_di14"]
    minus_di = snap["minus_di14"]

    # Trend / moving averages: symmetric weights.
    if ema9 is not None and ema21 is not None:
        if ema9 > ema21:
            up += 2; up_reasons.append("EMA9>EMA21")
        elif ema9 < ema21:
            down += 2; down_reasons.append("EMA9<EMA21")

    if ema9 is not None:
        if close > ema9:
            up += 1; up_reasons.append("price>EMA9")
        elif close < ema9:
            down += 1; down_reasons.append("price<EMA9")

    if ema21 is not None:
        if close > ema21:
            up += 1; up_reasons.append("price>EMA21")
        elif close < ema21:
            down += 1; down_reasons.append("price<EMA21")

    # Structure: only true HH/HL or LH/LL style block, no artificial fallback.
    if snap["structure"] == "BULLISH":
        up += 2; up_reasons.append("bullish structure")
    elif snap["structure"] == "BEARISH":
        down += 2; down_reasons.append("bearish structure")

    # Breakout / rejection / liquidity.
    b = snap["breakout"]
    if b == "BULL_BREAKOUT":
        up += 2; up_reasons.append("bull breakout")
    elif b == "BEAR_BREAKOUT":
        down += 2; down_reasons.append("bear breakout")
    elif b == "BULL_REJECTION":
        up += 1; up_reasons.append("bull rejection")
    elif b == "BEAR_REJECTION":
        down += 1; down_reasons.append("bear rejection")

    sweep = snap["liquidity_sweep"]
    if sweep == "BULL_SWEEP":
        up += 2; up_reasons.append("bull liquidity sweep")
    elif sweep == "BEAR_SWEEP":
        down += 2; down_reasons.append("bear liquidity sweep")

    # Momentum.
    if snap["momentum"] == "UP":
        up += 1; up_reasons.append("short momentum up")
    elif snap["momentum"] == "DOWN":
        down += 1; down_reasons.append("short momentum down")

    # Candle body.
    if snap["last_body"] == "UP":
        up += 1; up_reasons.append("last candle bullish")
    elif snap["last_body"] == "DOWN":
        down += 1; down_reasons.append("last candle bearish")

    # ADX/DI: only directional when trend strength is meaningful.
    if adx is not None and plus_di is not None and minus_di is not None and adx >= 20:
        if plus_di > minus_di:
            up += 2; up_reasons.append("ADX/DI bullish")
        elif minus_di > plus_di:
            down += 2; down_reasons.append("ADX/DI bearish")

    # RSI: deliberately conservative. Do not force direction in extremes.
    if rsi14 is not None:
        if 50 < rsi14 < 68:
            up += 1; up_reasons.append("RSI bullish zone")
        elif 32 < rsi14 < 50:
            down += 1; down_reasons.append("RSI bearish zone")
        elif rsi14 >= 68 and ema9 is not None and close > ema9:
            up += 1; up_reasons.append("RSI strong with trend")
        elif rsi14 <= 32 and ema9 is not None and close < ema9:
            down += 1; down_reasons.append("RSI weak with trend")

    # Williams %R: direction only when it has momentum confirmation, not merely >/< -50.
    if wr is not None and wr_prev is not None:
        if wr > -50 and wr > wr_prev:
            up += 1; up_reasons.append("Williams momentum up")
        elif wr < -50 and wr < wr_prev:
            down += 1; down_reasons.append("Williams momentum down")
        elif wr_prev <= -80 < wr:
            up += 1; up_reasons.append("Williams exited oversold")
        elif wr_prev >= -20 > wr:
            down += 1; down_reasons.append("Williams exited overbought")

    # Keltner midline confirmation.
    if ema20 is not None:
        if close > ema20:
            up += 1; up_reasons.append("price above Keltner mid")
        elif close < ema20:
            down += 1; down_reasons.append("price below Keltner mid")

    # Tie-break is symmetric and deterministic.
    if up > down:
        direction = "UP"
    elif down > up:
        direction = "DOWN"
    else:
        if snap["momentum"] == "UP":
            direction = "UP"
        elif snap["momentum"] == "DOWN":
            direction = "DOWN"
        elif ema9 is not None and ema21 is not None:
            direction = "UP" if ema9 >= ema21 else "DOWN"
        else:
            direction = "UP" if snap["last_body"] != "DOWN" else "DOWN"

    total = up + down
    if total <= 0:
        up_score, down_score = (10, 8) if direction == "UP" else (8, 10)
    else:
        up_score = round((up / total) * 18)
        down_score = 18 - up_score
        if direction == "UP" and up_score <= down_score:
            up_score, down_score = 10, 8
        elif direction == "DOWN" and down_score <= up_score:
            up_score, down_score = 8, 10

    gap = abs(up - down)
    ratio = gap / max(1, total)
    confidence = int(round(55 + min(32, ratio * 42)))
    # Require multiple independent confirmations for higher confidence.
    side_reasons = up_reasons if direction == "UP" else down_reasons
    unique_groups = len(side_reasons)
    if unique_groups < 4:
        confidence = min(confidence, 67)
    elif unique_groups < 6:
        confidence = min(confidence, 76)
    else:
        confidence = min(confidence, 88)

    logger.info(
        "ENGINE | %s | UP=%d %s | DOWN=%d %s | FINAL=%s | conf=%d",
        direction, up, ", ".join(up_reasons), down, ", ".join(down_reasons), direction, confidence
    )

    return {
        "direction": direction,
        "up_raw": up,
        "down_raw": down,
        "up_score": int(clamp(up_score, 0, 18)),
        "down_score": int(clamp(down_score, 0, 18)),
        "confidence": int(clamp(confidence, 55, 88)),
        "up_reasons": up_reasons,
        "down_reasons": down_reasons,
        "gap": gap,
    }


# ------------------------------------------------------------
# Gemini explanation / validation
# ------------------------------------------------------------

def gemini_client():
    if not GEMINI_API_KEY:
        return None
    try:
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        logger.exception("Gemini client initialization failed")
        return None


def ask_gemini(candles, snap, engine, symbol, timeframe):
    client = gemini_client()
    if client is None:
        return None

    compact = []
    for c in candles[-40:]:
        compact.append({
            "time": safe_int(c.get("time")),
            "open": safe_float(c.get("open")),
            "high": safe_float(c.get("high")),
            "low": safe_float(c.get("low")),
            "close": safe_float(c.get("close")),
        })

    prompt = f"""
You are the confirmation analyst for ZinoProSignalAI.
Symbol: {symbol}
Timeframe: {timeframe}
Only closed candles are supplied. Never invent prices, candles, indicators, or external market data.
The deterministic engine has already selected {engine['direction']}.
Your job is NOT to flip the direction. Check whether the evidence supports it and provide a concise reason.
Use this priority:
1 Price Action
2 Market Structure
3 Breakout/Retest and Liquidity
4 Momentum
5 Candle behavior
6 EMA 9/21
7 RSI 14
8 Williams %R 14
9 Keltner EMA20/ATR10*5
10 ADX/DI 14
Be conservative. If evidence conflicts, explicitly say that confidence is limited.
Return JSON only:
{{"supported":true/false,"reason":"short reason","cancellation_reason":"short cancellation condition"}}

TECHNICAL SNAPSHOT:
{json.dumps(snap, ensure_ascii=False)}

ENGINE:
{json.dumps(engine, ensure_ascii=False)}

CLOSED CANDLES:
{json.dumps(compact, ensure_ascii=False)}
"""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        raw = getattr(response, "text", "") or ""
        result = json.loads(raw)
        if not isinstance(result, dict):
            return None
        return result
    except Exception:
        logger.exception("Gemini analysis failed")
        return None


# ------------------------------------------------------------
# MT5 input
# ------------------------------------------------------------

def validate_mt5_payload(payload, header_key=None):
    if not isinstance(payload, dict):
        return False, "invalid json"
    supplied = str(header_key or payload.get("api_key") or "").strip()
    if MT5_API_KEY and supplied != MT5_API_KEY:
        return False, "unauthorized"
    symbol = str(payload.get("symbol", "")).strip()
    candles = payload.get("candles")
    if not symbol or not isinstance(candles, list):
        return False, "missing symbol/candles"
    if len(candles) < MIN_CLOSED_CANDLES + 1:
        return False, "not enough candles"
    required = ("time", "open", "high", "low", "close")
    clean = []
    for c in candles:
        if not isinstance(c, dict) or any(k not in c for k in required):
            continue
        clean.append({
            "time": safe_int(c["time"]),
            "open": safe_float(c["open"]),
            "high": safe_float(c["high"]),
            "low": safe_float(c["low"]),
            "close": safe_float(c["close"]),
            "tick_volume": safe_int(c.get("tick_volume")),
            "real_volume": safe_int(c.get("real_volume")),
            "spread": safe_int(c.get("spread")),
        })
    if len(clean) < MIN_CLOSED_CANDLES + 1:
        return False, "invalid candles"
    return True, {
        "symbol": symbol,
        "timeframe": str(payload.get("timeframe") or ANALYSIS_TIMEFRAME).upper(),
        "candles": clean,
        "received_at": time.time(),
        "current_bid": safe_float(payload.get("current_bid")),
        "current_ask": safe_float(payload.get("current_ask")),
        "digits": safe_int(payload.get("digits"), 5),
        "closed_candle_time": safe_int(payload.get("closed_candle_time")),
        "batch_id": str(payload.get("batch_id", "")),
    }


def store_mt5(payload, header_key=None):
    ok, result = validate_mt5_payload(payload, header_key)
    if not ok:
        return False, result
    with mt5_lock:
        mt5_data[result["symbol"]] = result
    return True, result


def closed_candles_for(data):
    candles = data.get("candles", [])
    if len(candles) < MIN_CLOSED_CANDLES + 1:
        return []
    # MT5 sends current forming candle last. Exclude it.
    return candles[:-1]


def get_candidates():
    with mt5_lock:
        items = list(mt5_data.items())
    out = []
    now = time.time()
    for symbol, data in items:
        age = now - safe_float(data.get("received_at"), 0)
        if age > MAX_DATA_AGE_SECONDS:
            continue
        candles = closed_candles_for(data)
        if len(candles) < MIN_CLOSED_CANDLES:
            continue
        out.append((symbol, data, candles, age))
    return out


# ------------------------------------------------------------
# Signal creation
# ------------------------------------------------------------

def choose_best_candidate():
    candidates = get_candidates()
    if not candidates:
        return None

    if active_cycle:
        target = active_cycle.get("symbol")
        same = [x for x in candidates if x[0] == target]
        if same:
            candidates = same

    ranked = []
    for symbol, data, candles, age in candidates:
        snap = technical_snapshot(candles)
        engine = directional_engine(candles, snap)
        quality = engine["gap"] * 3 + len(engine["up_reasons"] if engine["direction"] == "UP" else engine["down_reasons"])
        ranked.append((quality, engine["confidence"], symbol, data, candles, snap, engine, age))

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = ranked[0]
    return {
        "symbol": best[2],
        "data": best[3],
        "candles": best[4],
        "snapshot": best[5],
        "engine": best[6],
        "age": best[7],
    }


def calculate_entry(candles, direction, timeframe):
    delay = parse_timeframe_minutes(timeframe)
    entry_price = safe_float(candles[-1]["close"])
    recent = candles[-8:]
    if direction == "UP":
        cancellation = min(safe_float(c["low"]) for c in recent)
        cancellation_text = f"إلغاء إذا أغلقت شمعة تحت {cancellation:.5f}"
    else:
        cancellation = max(safe_float(c["high"]) for c in recent)
        cancellation_text = f"إلغاء إذا أغلقت شمعة فوق {cancellation:.5f}"
    base = now_algiers().replace(second=0, microsecond=0)
    entry_time = base + timedelta(minutes=delay)
    return {
        "delay": delay,
        "entry_price": entry_price,
        "cancellation": cancellation,
        "cancellation_text": cancellation_text,
        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
    }


def setup_fingerprint(symbol, timeframe, candle_time, direction, trade_type, recovery_number):
    return f"{symbol}|{timeframe}|{candle_time}|{direction}|{trade_type}|{recovery_number}"


def strict_enough(engine):
    reasons = engine["up_reasons"] if engine["direction"] == "UP" else engine["down_reasons"]
    # Minimum independent evidence. This intentionally produces fewer signals.
    if len(reasons) < 4:
        return False
    if engine["gap"] < 2:
        return False
    return True


def build_signal(candidate, trade_type="BASE", recovery_number=0):
    global last_signal_time, last_setup, active_cycle

    symbol = candidate["symbol"]
    data = candidate["data"]
    candles = candidate["candles"]
    snap = candidate["snapshot"]
    engine = candidate["engine"]
    timeframe = str(data.get("timeframe") or ANALYSIS_TIMEFRAME).upper()
    direction = engine["direction"]
    candle_time = safe_int(candles[-1]["time"])
    fingerprint = setup_fingerprint(symbol, timeframe, candle_time, direction, trade_type, recovery_number)

    with state_lock:
        now = time.time()
        if trade_type == "BASE" and now - last_signal_time < SIGNAL_COOLDOWN_SECONDS:
            return None
        if last_setup and last_setup.get("fingerprint") == fingerprint and now - safe_float(last_setup.get("time")) < SETUP_REPEAT_BLOCK_SECONDS:
            return None

    if not strict_enough(engine):
        logger.info("FILTERED | %s | insufficient confluence", symbol)
        return None

    gem = ask_gemini(candles, snap, engine, symbol, timeframe)
    if gem and gem.get("supported") is False:
        # Gemini is a confirmation layer; disagreement suppresses weak setups.
        if engine["confidence"] < 78:
            logger.info("FILTERED | %s | Gemini did not support setup", symbol)
            return None

    entry = calculate_entry(candles, direction, timeframe)
    reason = "; ".join((engine["up_reasons"] if direction == "UP" else engine["down_reasons"])[:4])
    if gem and isinstance(gem.get("reason"), str) and gem["reason"].strip():
        reason = gem["reason"].strip()
    cancellation_reason = entry["cancellation_text"]
    if gem and isinstance(gem.get("cancellation_reason"), str) and gem["cancellation_reason"].strip():
        cancellation_reason = gem["cancellation_reason"].strip()

    signal = {
        "symbol": symbol,
        "timeframe": timeframe,
        "trade_type": trade_type,
        "recovery_number": recovery_number,
        "direction": direction,
        "confidence": engine["confidence"],
        "up_score": engine["up_score"],
        "down_score": engine["down_score"],
        "entry_after": entry["delay"],
        "entry_time": entry["entry_time"],
        "entry_price": entry["entry_price"],
        "cancellation": entry["cancellation"],
        "cancellation_text": cancellation_reason,
        "reason": reason,
        "candle_time": candle_time,
        "created_at": now_algiers().strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": fingerprint,
        "status": "PENDING",
    }

    with state_lock:
        last_signal_time = time.time()
        last_setup = {"fingerprint": fingerprint, "time": last_signal_time}
        active_cycle = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "trade_type": trade_type,
            "recovery_number": recovery_number,
            "trade_number": 1 if trade_type == "BASE" else 2,
            "started_at": signal["created_at"],
            "signal": signal,
            "recovery_used": recovery_number > 0,
        }
        history.append(signal.copy())
        stats["signals"] = stats.get("signals", 0) + 1
    save_state()
    return signal


# ------------------------------------------------------------
# Telegram formatting / commands
# ------------------------------------------------------------

def fmt_price(price, digits=5):
    return f"{safe_float(price):.{max(2, min(8, digits))}f}"


def format_signal(signal):
    arrow = "🟢 UP" if signal["direction"] == "UP" else "🔴 DOWN"
    trade_label = "BASE TRADE" if signal["trade_type"] == "BASE" else f"RECOVERY {signal['recovery_number']}/{RECOVERY_LIMIT}"
    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {signal['symbol']} | {signal['timeframe']}\n\n"
        f"🎯 {trade_label}\n"
        f"{arrow}\n\n"
        f"🔥 Confidence: {signal['confidence']}%\n"
        f"🟢 UP Score: {signal['up_score']}/18\n"
        f"🔴 DOWN Score: {signal['down_score']}/18\n\n"
        f"⏱️ Entry after: {signal['entry_after']} minute(s)\n"
        f"🕐 ENTRY TIME: {signal['entry_time']} (Algiers)\n"
        f"💰 Entry Price: {fmt_price(signal['entry_price'])}\n"
        f"⚠️ {signal['cancellation_text']}\n\n"
        f"📝 Reason: {signal['reason']}"
    )


def owner_only(update):
    return bool(update.effective_user and OWNER_ID_INT and update.effective_user.id == OWNER_ID_INT)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    await update.message.reply_text(
        "ZinoProSignalAI MT5 is online.\n"
        "MT5 → Render → strict analysis → Telegram.\n"
        "Commands: /analyze /mt5status /stats /history /win /loss /reset"
    )


async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    if active_cycle:
        await update.message.reply_text("⚠️ كاين cycle مفتوح. استعمل /win أو /loss أولاً.")
        return
    candidate = choose_best_candidate()
    if not candidate:
        await update.message.reply_text("⚠️ ما كاش MT5 data صالحة حالياً.")
        return
    signal = build_signal(candidate)
    if not signal:
        await update.message.reply_text("🔎 ما كاش setup قوي كفاية حالياً. البوت فلتر الإشارة.")
        return
    await update.message.reply_text(format_signal(signal))


async def cmd_mt5status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    candidates = get_candidates()
    if not candidates:
        await update.message.reply_text("❌ No fresh MT5 data.")
        return
    lines = ["📡 MT5 STATUS", "━━━━━━━━━━━━━━━━━━"]
    for symbol, data, candles, age in candidates:
        lines.append(f"• {symbol} | {data.get('timeframe')} | {len(candles)} closed | age {age:.0f}s")
    await update.message.reply_text("\n".join(lines))


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    with state_lock:
        w = stats.get("wins", 0)
        l = stats.get("losses", 0)
        s = stats.get("signals", 0)
    total = w + l
    rate = (100.0 * w / total) if total else 0.0
    await update.message.reply_text(f"📊 STATS\nSignals: {s}\nWins: {w}\nLosses: {l}\nWin rate: {rate:.1f}%")


async def cmd_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    with state_lock:
        items = history[-HISTORY_DISPLAY_COUNT:]
    if not items:
        await update.message.reply_text("📚 History empty.")
        return
    lines = ["📚 ZinoProSignalAI HISTORY", "━━━━━━━━━━━━━━━━━━"]
    for x in reversed(items):
        icon = "🟢" if x.get("direction") == "UP" else "🔴"
        lines.append(
            f"{icon} {x.get('symbol')} {x.get('timeframe')} | {x.get('trade_type')} | "
            f"{x.get('direction')} | {x.get('confidence')}% | {x.get('status')}"
        )
    await update.message.reply_text("\n".join(lines))


async def cmd_win(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    await finish_trade("WIN", update)


async def cmd_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    await finish_trade("LOSS", update)


async def finish_trade(result, update):
    global active_cycle
    with state_lock:
        cycle = active_cycle
        if not cycle:
            await update.message.reply_text("❌ ما كاش trade active.")
            return
        sig = cycle.get("signal", {})
        for item in reversed(history):
            if item.get("fingerprint") == sig.get("fingerprint") and item.get("status") == "PENDING":
                item["status"] = result
                item["closed_at"] = now_algiers().strftime("%Y-%m-%d %H:%M:%S")
                break
        if result == "WIN":
            stats["wins"] = stats.get("wins", 0) + 1
            active_cycle = None
            save_state()
            await update.message.reply_text("✅ WIN recorded. Cycle closed.")
            return
        stats["losses"] = stats.get("losses", 0) + 1
        if cycle.get("recovery_used") or cycle.get("recovery_number", 0) >= RECOVERY_LIMIT:
            active_cycle = None
            save_state()
            await update.message.reply_text("❌ LOSS recorded. Recovery limit reached; cycle closed.")
            return
        symbol = cycle.get("symbol")

    # Fresh recovery analysis outside the lock.
    candidate = None
    for item in get_candidates():
        if item[0] == symbol:
            _, data, candles, age = item
            snap = technical_snapshot(candles)
            engine = directional_engine(candles, snap)
            candidate = {"symbol": symbol, "data": data, "candles": candles, "snapshot": snap, "engine": engine, "age": age}
            break

    if not candidate:
        with state_lock:
            active_cycle = None
        save_state()
        await update.message.reply_text("❌ LOSS recorded. MT5 data unavailable for recovery; cycle closed.")
        return

    signal = build_signal(candidate, trade_type="RECOVERY", recovery_number=1)
    if not signal:
        with state_lock:
            active_cycle = None
        save_state()
        await update.message.reply_text("❌ LOSS recorded. No strong recovery setup; cycle closed.")
        return
    await update.message.reply_text(format_signal(signal))


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global active_cycle, last_signal_time, last_setup
    if not owner_only(update):
        return
    with state_lock:
        stats["wins"] = 0
        stats["losses"] = 0
        stats["signals"] = 0
        history.clear()
        active_cycle = None
        last_signal_time = 0.0
        last_setup = None
    save_state()
    await update.message.reply_text("♻️ Stats, history and active cycle reset.")


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    await update.message.reply_text("📡 النظام يعتمد على بيانات MT5. استعمل /analyze أو أرسل بيانات MT5 عبر EA.")


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not owner_only(update):
        return
    await update.message.reply_text("📷 Image mode disabled. ZinoProSignalAI الآن يعمل عبر MT5 مباشرة.")


# ------------------------------------------------------------
# Telegram async loop bridge
# ------------------------------------------------------------

async def telegram_post_init(application):
    global telegram_loop, auto_thread_started
    telegram_loop = asyncio.get_running_loop()
    logger.info("Telegram event loop captured")
    if not auto_thread_started:
        auto_thread_started = True
        threading.Thread(target=auto_analysis_loop, daemon=True, name="AutoAnalysis").start()
        logger.info("Auto analysis thread started")


async def telegram_error_handler(update, context):
    logger.error("Telegram update error: %r", context.error, exc_info=context.error)


async def send_cycle_to_owner(signal):
    if telegram_application is None or OWNER_ID_INT == 0:
        return
    await telegram_application.bot.send_message(chat_id=OWNER_ID_INT, text=format_signal(signal))


def schedule_telegram_send(signal):
    if telegram_loop is None:
        logger.warning("Telegram loop not ready; signal not sent")
        return
    future = asyncio.run_coroutine_threadsafe(send_cycle_to_owner(signal), telegram_loop)

    def done_callback(f):
        try:
            f.result()
        except Exception:
            logger.exception("Telegram async send failed")

    future.add_done_callback(done_callback)


# ------------------------------------------------------------
# Automatic analysis
# ------------------------------------------------------------

def auto_analysis_once():
    if active_cycle:
        return
    candidate = choose_best_candidate()
    if not candidate:
        return
    signal = build_signal(candidate)
    if signal:
        logger.info("SIGNAL CREATED | %s | %s | %s", signal["symbol"], signal["direction"], signal["confidence"])
        schedule_telegram_send(signal)


def auto_analysis_loop():
    while True:
        try:
            auto_analysis_once()
        except Exception:
            logger.exception("Auto analysis error")
        time.sleep(AUTO_ANALYSIS_INTERVAL_SECONDS)


# ------------------------------------------------------------
# HTTP server for MT5
# ------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("HTTP | " + fmt, *args)

    def _send(self, code, body, content_type="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/health", "/healthz"):
            self._send(200, json.dumps({"status": "ok", "service": "ZinoProSignalAI", "mt5_symbols": len(mt5_data)}))
            return
        if self.path in ("/mt4status", "/mt5status"):
            self._send(200, json.dumps({"status": "ok", "symbols": list(mt5_data.keys()), "active_cycle": bool(active_cycle)}))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path not in ("/mt5", "/api/mt5", "/mt4", "/api/mt4"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 5_000_000:
                self._send(400, json.dumps({"error": "invalid content length"}))
                return
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            ok, result = store_mt5(payload, self.headers.get("X-MT5-API-Key", ""))
            if not ok:
                status = 401 if result == "unauthorized" else 400
                self._send(status, json.dumps({"ok": False, "error": result}))
                return
            logger.info(
                "MT5 DATA | %s | %s | candles=%d | batch=%s",
                result["symbol"], result["timeframe"], len(result["candles"]), result.get("batch_id", "")
            )
            self._send(200, json.dumps({"ok": True, "symbol": result["symbol"], "closed": len(closed_candles_for(result))}))
        except json.JSONDecodeError:
            self._send(400, json.dumps({"ok": False, "error": "invalid json"}))
        except Exception as exc:
            logger.exception("MT5 POST error")
            self._send(500, json.dumps({"ok": False, "error": str(exc)}))


def start_http_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    logger.info("HTTP server listening on port %d", PORT)
    server.serve_forever()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    global telegram_application

    load_state()

    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN is missing")
    if OWNER_ID_INT == 0:
        raise RuntimeError("OWNER_ID is missing or invalid")
    if not MT5_API_KEY:
        logger.warning("MT5_API_KEY is empty; MT5 authentication is disabled")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY is empty; deterministic engine will be used without Gemini")

    threading.Thread(target=start_http_server, daemon=True, name="HTTPServer").start()

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(telegram_post_init)
        .build()
    )
    telegram_application = app

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("mt5status", cmd_mt5status))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("history", cmd_history))
    app.add_handler(CommandHandler("win", cmd_win))
    app.add_handler(CommandHandler("loss", cmd_loss))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
    app.add_error_handler(telegram_error_handler)

    logger.info("Starting Telegram polling")
    # drop_pending_updates prevents a large backlog after redeploy.
    app.run_polling(drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
