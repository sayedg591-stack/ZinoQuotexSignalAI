 import os
import json
import time
import threading
import urllib.parse
import urllib.request
import logging

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_KEY = os.getenv("ZINO_API_KEY", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
PORT = int(os.getenv("PORT", "10000"))

TZ = ZoneInfo("Africa/Algiers")

MIN_SCORE = 15
MAX_SCORE = 20

# بعد إرسال إشارة، ننتظر قبل إشارة BASE جديدة
GLOBAL_COOLDOWN = 120

# لا نختار الفائز مباشرة عند وصول أول زوج
# نجمع الأزواج خلال هذه المدة ثم نختار الأفضل
RANKING_WINDOW = 8

# لا نكرر نفس الزوج بسرعة
SYMBOL_COOLDOWN = 300

# Recovery واحد فقط
RECOVERY_DELAY = 60

MAX_HISTORY = 100


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# STATE
# ============================================================

lock = threading.Lock()

stats = {
    "wins": 0,
    "losses": 0,
}

history = []

candidates = {}

last_base_time = 0.0
last_signal_symbol = ""
last_signal_direction = ""

active_trade = None

ranking_started = 0.0

MAX_CANDIDATE_AGE = 15


# ============================================================
# HELPERS
# ============================================================

def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def ema(values, period):
    if not values:
        return 0.0

    if len(values) < period:
        period = len(values)

    k = 2.0 / (period + 1.0)

    result = values[0]

    for value in values[1:]:
        result = value * k + result * (1.0 - k)

    return result


def atr(candles, period=14):

    if len(candles) < 2:
        return 0.0

    tr = []

    previous_close = None

    for candle in candles:

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        if previous_close is None:
            value = high - low
        else:
            value = max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close)
            )

        tr.append(max(value, 1e-12))

        previous_close = close

    return mean(tr[-period:])


# ============================================================
# RSI
# ============================================================

def rsi(closes, period=14):

    if len(closes) <= period:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(closes)):

        diff = closes[i] - closes[i - 1]

        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))

    avg_gain = mean(gains[-period:])
    avg_loss = mean(losses[-period:])

    if avg_loss == 0:

        if avg_gain > 0:
            return 100.0

        return 50.0

    rs = avg_gain / avg_loss

    return 100.0 - (100.0 / (1.0 + rs))


# ============================================================
# WILLIAMS %R
# ============================================================

def williams(candles, period=14):

    if len(candles) < period:
        return -50.0

    window = candles[-period:]

    highest = max(float(x["high"]) for x in window)
    lowest = min(float(x["low"]) for x in window)

    close = float(candles[-1]["close"])

    if highest == lowest:
        return -50.0

    return -100.0 * (highest - close) / (highest - lowest)


# ============================================================
# ADX / DI APPROXIMATION
# ============================================================

def adx_di(candles, period=14):

    if len(candles) < period + 2:
        return 0.0, 0.0, 0.0

    plus_dm = []
    minus_dm = []
    true_ranges = []

    previous_high = float(candles[-period - 1]["high"])
    previous_low = float(candles[-period - 1]["low"])
    previous_close = float(candles[-period - 1]["close"])

    for candle in candles[-period:]:

        high = float(candle["high"])
        low = float(candle["low"])
        close = float(candle["close"])

        up_move = high - previous_high
        down_move = previous_low - low

        plus = up_move if up_move > down_move and up_move > 0 else 0
        minus = down_move if down_move > up_move and down_move > 0 else 0

        tr = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )

        plus_dm.append(plus)
        minus_dm.append(minus)
        true_ranges.append(max(tr, 1e-12))

        previous_high = high
        previous_low = low
        previous_close = close

    avg_tr = mean(true_ranges)

    if avg_tr <= 0:
        return 0.0, 0.0, 0.0

    plus_di = mean(plus_dm) / avg_tr * 100
    minus_di = mean(minus_dm) / avg_tr * 100

    denominator = plus_di + minus_di

    if denominator <= 0:
        adx = 0.0
    else:
        adx = abs(plus_di - minus_di) / denominator * 100

    return adx, plus_di, minus_di


# ============================================================
# KELTNER
# ============================================================

def keltner(candles):

    closes = [float(x["close"]) for x in candles]

    middle = ema(closes[-40:], 20)

    atr_value = atr(candles, 10)

    upper = middle + atr_value * 1.5
    lower = middle - atr_value * 1.5

    return middle, upper, lower, atr_value


# ============================================================
# ANALYSIS
# ============================================================

def analyze(symbol, timeframe, candles):

    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    if len(candles) < 50:
        raise ValueError("Need at least 50 candles")

    candles = sorted(
        candles,
        key=lambda x: int(x["time"])
    )

    closes = [float(x["close"]) for x in candles]
    highs = [float(x["high"]) for x in candles]
    lows = [float(x["low"]) for x in candles]

    current = candles[-1]
    previous = candles[-2]

    open_price = float(current["open"])
    high_price = float(current["high"])
    low_price = float(current["low"])
    close_price = float(current["close"])

    previous_high = float(previous["high"])
    previous_low = float(previous["low"])

    candle_range = max(
        high_price - low_price,
        1e-12
    )

    body = abs(close_price - open_price)

    body_ratio = body / candle_range

    upper_wick = high_price - max(
        open_price,
        close_price
    )

    lower_wick = min(
        open_price,
        close_price
    ) - low_price

    atr_value = atr(candles, 14)

    if atr_value <= 0:
        raise ValueError("Invalid ATR")

    e9 = ema(closes[-80:], 9)
    e21 = ema(closes[-80:], 21)

    rsi_value = rsi(closes, 14)
    williams_value = williams(candles, 14)

    adx_value, plus_di, minus_di = adx_di(
        candles,
        14
    )

    kc_middle, kc_upper, kc_lower, kc_atr = keltner(
        candles
    )

    up = 0
    down = 0

    reasons_up = []
    reasons_down = []

    # ========================================================
    # 1. EMA TREND
    # ========================================================

    if e9 > e21:

        up += 2
        reasons_up.append("EMA 9/21 bullish")

    elif e9 < e21:

        down += 2
        reasons_down.append("EMA 9/21 bearish")


    # ========================================================
    # 2. PRICE VS EMA
    # ========================================================

    if close_price > e9:

        up += 1

    elif close_price < e9:

        down += 1


    # ========================================================
    # 3. STRUCTURE
    # ========================================================

    recent_high = max(highs[-9:-1])
    recent_low = min(lows[-9:-1])

    if close_price > recent_high:

        up += 2
        reasons_up.append("structure breakout")

    elif close_price < recent_low:

        down += 2
        reasons_down.append("structure breakdown")


    # ========================================================
    # 4. IMMEDIATE BREAK
    # ========================================================

    if close_price > previous_high:

        up += 1
        reasons_up.append("bullish break")

    elif close_price < previous_low:

        down += 1
        reasons_down.append("bearish break")


    # ========================================================
    # 5. MOMENTUM
    # ========================================================

    momentum = closes[-1] - closes[-4]

    if momentum > atr_value * 0.25:

        up += 2
        reasons_up.append("strong momentum")

    elif momentum < -atr_value * 0.25:

        down += 2
        reasons_down.append("strong momentum")


    # ========================================================
    # 6. CANDLE QUALITY
    # ========================================================

    if body_ratio >= 0.60:

        if close_price > open_price:

            up += 2
            reasons_up.append("strong bullish candle")

        else:

            down += 2
            reasons_down.append("strong bearish candle")

    elif body_ratio >= 0.40:

        if close_price > open_price:
            up += 1
        else:
            down += 1


    # ========================================================
    # 7. WICK REJECTION
    # ========================================================

    if lower_wick > upper_wick * 1.5 and close_price > open_price:

        up += 1
        reasons_up.append("lower wick rejection")

    elif upper_wick > lower_wick * 1.5 and close_price < open_price:

        down += 1
        reasons_down.append("upper wick rejection")


    # ========================================================
    # 8. RSI
    # ========================================================

    if 52 <= rsi_value <= 68:

        up += 1
        reasons_up.append("RSI bullish")

    elif 32 <= rsi_value <= 48:

        down += 1
        reasons_down.append("RSI bearish")


    # ========================================================
    # 9. WILLIAMS
    # ========================================================

    if williams_value > -50:

        up += 1

    elif williams_value < -50:

        down += 1


    # ========================================================
    # 10. ADX + DI
    # ========================================================

    if adx_value >= 20:

        if plus_di > minus_di:

            up += 2
            reasons_up.append("ADX/DI bullish")

        elif minus_di > plus_di:

            down += 2
            reasons_down.append("ADX/DI bearish")


    # ========================================================
    # 11. KELTNER
    # ========================================================

    if close_price > kc_middle:

        up += 1

    elif close_price < kc_middle:

        down += 1


    # ========================================================
    # 12. MARKET QUALITY FILTER
    # ========================================================

    # Candle abnormalement large
    if candle_range > atr_value * 2.5:

        up -= 2
        down -= 2

        market_warning = "abnormal candle"

    else:

        market_warning = ""


    # Very small body = indecision
    if body_ratio < 0.20:

        up -= 1
        down -= 1

        market_warning = "indecision candle"


    # Very low ADX = range
    if adx_value < 16:

        up -= 2
        down -= 2

        market_warning = "weak trend"


    # ========================================================
    # CLAMP
    # ========================================================

    up = int(clamp(up, 0, MAX_SCORE))
    down = int(clamp(down, 0, MAX_SCORE))

    # ========================================================
    # CHOOSE DIRECTION
    # ========================================================

    if up > down:

        direction = "UP"
        score = up
        opposite = down
        reasons = reasons_up

    elif down > up:

        direction = "DOWN"
        score = down
        opposite = up
        reasons = reasons_down

    else:

        return {
            "valid": False,
            "symbol": symbol,
            "reason": "balanced market"
        }


    # ========================================================
    # MINIMUM SCORE
    # ========================================================

    if score < MIN_SCORE:

        return {
            "valid": False,
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "up_score": up,
            "down_score": down,
            "reason": "score below minimum"
        }


    # ========================================================
    # EDGE FILTER
    # ========================================================

    if score - opposite < 3:

        return {
            "valid": False,
            "symbol": symbol,
            "direction": direction,
            "score": score,
            "up_score": up,
            "down_score": down,
            "reason": "weak directional edge"
        }


    # ========================================================
    # CONFIDENCE
    # ========================================================

    edge = score - opposite

    confidence = 62 + edge * 4

    if score >= 18:
        confidence += 5

    if adx_value >= 25:
        confidence += 4

    if body_ratio >= 0.60:
        confidence += 3

    if market_warning:
        confidence -= 8

    confidence = int(
        clamp(
            confidence,
            60,
            89
        )
    )


    # ========================================================
    # ENTRY
    # ========================================================

    now = datetime.now(TZ)

    entry_time = now + timedelta(
        minutes=1
    )

    if direction == "UP":

        cancel_price = close_price - atr_value * 0.35

    else:

        cancel_price = close_price + atr_value * 0.35


    # ========================================================
    # QUALITY SCORE
    # ========================================================

    quality = (
        score * 10
        + edge * 5
        + min(adx_value, 40)
    )

    # bonus for clean candle
    if body_ratio >= 0.60:
        quality += 10

    # bonus trend
    if adx_value >= 25:
        quality += 10

    signal = {

        "valid": True,

        "symbol": symbol,

        "timeframe": timeframe.upper(),

        "direction": direction,

        "score": score,

        "opposite_score": opposite,

        "up_score": up,

        "down_score": down,

        "confidence": confidence,

        "quality": round(quality, 2),

        "entry_after": 1,

        "entry_time": entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "entry_price": close_price,

        "cancel_price": cancel_price,

        "rsi": round(rsi_value, 2),

        "williams": round(williams_value, 2),

        "adx": round(adx_value, 2),

        "plus_di": round(plus_di, 2),

        "minus_di": round(minus_di, 2),

        "ema9": e9,

        "ema21": e21,

        "reason": ", ".join(
            reasons[:5]
        ) or "multi-factor setup",

        "generated_at": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "candle_time": int(
            current["time"]
        )
    }

    return signal


# ============================================================
# FORMAT TELEGRAM
# ============================================================

def format_signal(signal):

    if signal["direction"] == "UP":

        direction = "🟢 UP"

        cancel = (
            f"⚠️ CANCEL IF CLOSE < "
            f"{signal['cancel_price']:.8f}"
        )

    else:

        direction = "🔴 DOWN"

        cancel = (
            f"⚠️ CANCEL IF CLOSE > "
            f"{signal['cancel_price']:.8f}"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🏆 BEST SETUP\n"
        f"📊 {signal['symbol']} | "
        f"{signal['timeframe']}\n\n"

        "🎯 BASE TRADE\n"
        f"{direction}\n\n"

        f"🔥 Confidence: "
        f"{signal['confidence']}%\n"

        f"🏆 QUALITY: "
        f"{signal['score']}/20\n\n"

        f"🟢 UP Score: "
        f"{signal['up_score']}/20\n"

        f"🔴 DOWN Score: "
        f"{signal['down_score']}/20\n\n"

        "⏱️ Entry after: 1 minute\n"

        f"🕐 ENTRY TIME: "
        f"{signal['entry_time']} 🇩🇿\n"

        f"💰 ENTRY PRICE: "
        f"{signal['entry_price']:.8f}\n\n"

        f"{cancel}\n\n"

        f"🧠 {signal['reason']}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"
        "🔒 Recovery: 1 MAX"
    )


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram_message(text):

    if not BOT_TOKEN:
        logger.error("BOT_TOKEN missing")
        return False

    if not OWNER_ID:
        logger.error("OWNER_ID missing")
        return False

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{BOT_TOKEN}/sendMessage"
        )

        payload = urllib.parse.urlencode({
            "chat_id": str(OWNER_ID),
            "text": text
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=15
        ) as response:

            result = response.read().decode(
                "utf-8",
                errors="replace"
            )

        logger.info(
            "Telegram sendMessage: %s",
            result
        )

        return True

    except Exception:

        logger.exception(
            "Telegram send failed"
        )

        return False


# ============================================================
# BEST CANDIDATE
# ============================================================

def choose_best_candidate():

    global candidates

    now = time.time()

    valid = []

    with lock:

        for symbol, item in list(
            candidates.items()
        ):

            if now - item["received_at"] > MAX_CANDIDATE_AGE:

                del candidates[symbol]

                continue

            signal = item["signal"]

            if not signal.get("valid"):
                continue

            valid.append(signal)

    if not valid:
        return None

    valid.sort(
        key=lambda x: (
            x["quality"],
            x["score"],
            x["confidence"]
        ),
        reverse=True
    )

    return valid[0]


# ============================================================
# SIGNAL MANAGER
# ============================================================

def process_candidate(signal):

    global ranking_started
    global last_base_time
    global last_signal_symbol
    global last_signal_direction
    global active_trade

    now = time.time()

    # --------------------------------------------------------
    # Recovery active?
    # --------------------------------------------------------

    with lock:

        if active_trade is not None:

            logger.info(
                "Recovery/trade state active - "
                "new BASE ignored"
            )

            return False


    # --------------------------------------------------------
    # Global cooldown
    # --------------------------------------------------------

    if now - last_base_time < GLOBAL_COOLDOWN:

        logger.info(
            "Global cooldown active"
        )

        return False


    # --------------------------------------------------------
    # Store candidate
    # --------------------------------------------------------

    with lock:

        candidates[
            signal["symbol"]
        ] = {
            "signal": signal,
            "received_at": now
        }

        if ranking_started == 0:
            ranking_started = now


    # --------------------------------------------------------
    # Wait for ranking window
    # --------------------------------------------------------

    if now - ranking_started < RANKING_WINDOW:

        return False


    # --------------------------------------------------------
    # Select best
    # --------------------------------------------------------

    best = choose_best_candidate()

    with lock:

        ranking_started = 0

        candidates.clear()


    if best is None:

        logger.info(
            "No valid setup >= %s/20",
            MIN_SCORE
        )

        return False


    # --------------------------------------------------------
    # Avoid same pair/direction
    # --------------------------------------------------------

    if (
        best["symbol"] == last_signal_symbol
        and
        best["direction"] == last_signal_direction
    ):

        logger.info(
            "Same symbol/direction rejected: %s %s",
            best["symbol"],
            best["direction"]
        )

        return False


    # --------------------------------------------------------
    # Send
    # --------------------------------------------------------

    text = format_signal(best)

    sent = send_telegram_message(text)

    if not sent:
        return False


    # --------------------------------------------------------
    # Save active BASE
    # --------------------------------------------------------

    with lock:

        active_trade = {
            "symbol": best["symbol"],
            "direction": best["direction"],
            "confidence": best["confidence"],
            "score": best["score"],
            "entry_price": best["entry_price"],
            "entry_time": best["entry_time"],
            "recovery_used": False,
            "created_at": now
        }

        history.append(best)

        if len(history) > MAX_HISTORY:

            del history[
                :-MAX_HISTORY
            ]


    last_base_time = now

    last_signal_symbol = best["symbol"]

    last_signal_direction = best["direction"]

    logger.info(
        "BEST SIGNAL SENT: %s %s %s/20",
        best["symbol"],
        best["direction"],
        best["score"]
    )

    return True


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def send_json(self, code, obj):

        data = json.dumps(
            obj,
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

        if self.path in (
            "/",
            "/health"
        ):

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI"
                }
            )

            return

        self.send_json(
            404,
            {"error": "not found"}
        )


    def do_POST(self):

        if self.path != "/mt4":

            self.send_json(
                404,
                {"error": "not found"}
            )

            return

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw = self.rfile.read(length)

            data = json.loads(
                raw.decode("utf-8")
            )

            received_key = str(
                data.get(
                    "api_key",
                    ""
                )
            ).strip()


            # ------------------------------------------------
            # API KEY
            # ------------------------------------------------

            if not API_KEY:

                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error":
                            "ZINO_API_KEY missing"
                    }
                )

                return


            if received_key != API_KEY:

                self.send_json(
                    401,
                    {
                        "ok": False,
                        "error":
                            "unauthorized"
                    }
                )

                return


            symbol = str(
                data.get(
                    "symbol",
                    ""
                )
            ).strip().upper()


            timeframe = str(
                data.get(
                    "timeframe",
                    "M1"
                )
            ).strip().upper()


            candles = data.get(
                "candles",
                []
            )


            # ------------------------------------------------
            # ANALYZE
            # ------------------------------------------------

            signal = analyze(
                symbol,
                timeframe,
                candles
            )


            if not signal.get("valid"):

                logger.info(
                    "%s rejected: %s",
                    symbol,
                    signal.get("reason")
                )

                self.send_json(
                    200,
                    {
                        "ok": True,
                        "accepted": False,
                        "signal": signal
                    }
                )

                return


            # ------------------------------------------------
            # ADD TO RANKING
            # ------------------------------------------------

            sent = process_candidate(
                signal
            )


            self.send_json(
                200,
                {
                    "ok": True,
                    "accepted": True,
                    "sent": sent,
                    "signal": signal
                }
            )


        except Exception as exc:

            logger.exception(
                "MT4 request failed"
            )

            self.send_json(
                400,
                {
                    "ok": False,
                    "error": str(exc)
                }
            )


    def log_message(
        self,
        fmt,
        *args
    ):

        logger.info(
            "HTTP " + fmt,
            *args
        )


# ============================================================
# HTTP THREAD
# ============================================================

def run_http_server():

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        Handler
    )

    logger.info(
        "ZinoProSignalAI HTTP online "
        "port=%s",
        PORT
    )

    server.serve_forever()


# ============================================================
# OWNER
# ============================================================

async def is_owner(update):

    if not update.effective_user:
        return False

    return (
        update.effective_user.id
        == OWNER_ID
    )


async def start(update, context):

    if not await is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ ONLINE\n"
        "📡 MT4 connected\n"
        "🏆 Best Setup Ranking\n"
        "🎯 Minimum: 15/20\n"
        "🔒 Recovery: 1 MAX\n\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


async def stats_command(
    update,
    context
):

    if not await is_owner(update):
        return

    with lock:

        wins = stats["wins"]
        losses = stats["losses"]

        active = (
            active_trade is not None
        )

    total = wins + losses

    accuracy = (
        wins / total * 100
        if total
        else 0
    )

    await update.message.reply_text(
        "📚 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 WIN: {wins}\n"
        f"🔴 LOSS: {losses}\n"
        f"📊 TOTAL: {total}\n"
        f"🎯 Accuracy: {accuracy:.1f}%\n"
        f"🔄 Active trade: "
        f"{'YES' if active else 'NO'}"
    )


async def win_command(
    update,
    context
):

    global active_trade

    if not await is_owner(update):
        return

    with lock:

        stats["wins"] += 1

        active_trade = None

    await update.message.reply_text(
        "🟢 WIN recorded\n"
        "🔎 Searching for a new BEST SETUP..."
    )


async def loss_command(
    update,
    context
):

    global active_trade

    if not await is_owner(update):
        return

    with lock:

        stats["losses"] += 1

        trade = active_trade

    if trade is None:

        await update.message.reply_text(
            "⚠️ No active BASE trade."
        )

        return


    # ========================================================
    # FIRST LOSS -> RECOVERY
    # ========================================================

    if not trade["recovery_used"]:

        trade["recovery_used"] = True

        recovery_time = (
            datetime.now(TZ)
            + timedelta(
                seconds=RECOVERY_DELAY
            )
        )

        await update.message.reply_text(
            "🔴 BASE LOSS\n"
            "━━━━━━━━━━━━━━━━━━\n"
            f"📊 {trade['symbol']}\n"
            f"📉 {trade['direction']}\n\n"
            "🔄 RECOVERY 1/1 ALLOWED\n"
            f"⏱️ Wait: {RECOVERY_DELAY} seconds\n"
            f"🕐 Recovery time: "
            f"{recovery_time.strftime('%H:%M:%S')} 🇩🇿\n\n"
            "❌ No Recovery 2"
        )

        return


    # ========================================================
    # SECOND LOSS = END
    # ========================================================

    with lock:

        active_trade = None

    await update.message.reply_text(
        "🔴 RECOVERY LOSS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "❌ Recovery 1 failed\n"
        "🚫 No Recovery 2\n"
        "♻️ Trade sequence closed\n\n"
        "🔎 Searching for a new BEST SETUP..."
    )


async def reset_command(
    update,
    context
):

    global active_trade
    global last_base_time
    global last_signal_symbol
    global last_signal_direction

    if not await is_owner(update):
        return

    with lock:

        stats["wins"] = 0
        stats["losses"] = 0

        history.clear()
        candidates.clear()

        active_trade = None

    last_base_time = 0
    last_signal_symbol = ""
    last_signal_direction = ""

    await update.message.reply_text(
        "♻️ ZinoProSignalAI RESET\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "✅ Statistics cleared\n"
        "✅ Ranking cleared\n"
        "✅ Active trade cleared"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    thread = threading.Thread(
        target=run_http_server,
        daemon=True
    )

    thread.start()

    logger.info(
        "HTTP server started"
    )

    if not BOT_TOKEN:
        logger.error(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:
        logger.error(
            "OWNER_ID is missing"
        )

    if not API_KEY:
        logger.error(
            "ZINO_API_KEY is missing"
        )

    if not BOT_TOKEN:

        logger.info(
            "Telegram polling disabled"
        )

        while True:
            time.sleep(3600)


    application = (
        Application
        .builder()
        .token(BOT_TOKEN)
        .build()
    )


    application.add_handler(
        CommandHandler(
            "start",
            start
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


    logger.info(
        "Telegram polling starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
