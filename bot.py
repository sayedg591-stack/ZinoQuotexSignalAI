import os
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
API_KEY = os.getenv("ZINO_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "10000"))

TZ = ZoneInfo("Africa/Algiers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GLOBAL STATE
# ============================================================

stats = {
    "wins": 0,
    "losses": 0,
}

history = []

lock = threading.Lock()

# Prevent duplicate Telegram signals
last_sent_key = ""
last_sent_time = 0.0

# Minimum time between automatically sent signals
SIGNAL_COOLDOWN = 120

# Keep only latest signals in memory
MAX_HISTORY = 100


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def mean(values):
    return sum(values) / len(values) if values else 0.0


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


def adx_proxy(candles, period=14):
    """
    Lightweight ADX/DI-style strength calculation.
    This is intentionally deterministic and does not use Gemini.
    """

    if len(candles) < period + 2:
        return 15.0, 0.0

    ups = []
    downs = []
    trs = []

    previous_close = float(candles[-period - 1]["close"])

    for candle in candles[-period:]:
        high = float(candle["high"])
        low = float(candle["low"])

        ups.append(max(high - previous_close, 0.0))
        downs.append(max(previous_close - low, 0.0))

        trs.append(max(high - low, 1e-12))

        previous_close = float(candle["close"])

    avg_tr = mean(trs)

    if avg_tr <= 0:
        return 15.0, 0.0

    plus_di = mean(ups) / avg_tr * 100.0
    minus_di = mean(downs) / avg_tr * 100.0

    strength = 10.0 + clamp(
        abs(plus_di - minus_di) * 2.0,
        0.0,
        50.0
    )

    return strength, plus_di - minus_di


# ============================================================
# ANALYSIS
# ============================================================

def analyze(symbol, timeframe, candles):

    if not isinstance(candles, list):
        raise ValueError("candles must be a list")

    if len(candles) < 35:
        raise ValueError(
            "Need at least 35 candles"
        )

    candles = sorted(
        candles,
        key=lambda x: int(x["time"])
    )

    closes = [
        float(x["close"])
        for x in candles
    ]

    highs = [
        float(x["high"])
        for x in candles
    ]

    lows = [
        float(x["low"])
        for x in candles
    ]

    e9 = ema(closes[-80:], 9)
    e21 = ema(closes[-80:], 21)

    rsi_value = rsi(closes, 14)
    williams_value = williams(candles, 14)
    adx_value, di_value = adx_proxy(candles, 14)
    atr_value = atr(candles, 14)

    if atr_value <= 0:
        raise ValueError("Invalid ATR")

    current = candles[-1]
    previous = candles[-2]

    current_open = float(current["open"])
    current_high = float(current["high"])
    current_low = float(current["low"])
    current_close = float(current["close"])

    previous_high = float(previous["high"])
    previous_low = float(previous["low"])

    up = 0.0
    down = 0.0

    reasons_up = []
    reasons_down = []

    # --------------------------------------------------------
    # 1. STRUCTURE / BREAKOUT
    # --------------------------------------------------------

    recent_high = max(highs[-8:-1])
    recent_low = min(lows[-8:-1])

    if current_close > recent_high:
        up += 2
        reasons_up.append("structure breakout")

    elif current_close < recent_low:
        down += 2
        reasons_down.append("structure breakdown")

    else:
        short_average = mean(closes[-3:])
        previous_average = mean(closes[-8:-3])

        if short_average > previous_average:
            up += 1

        elif short_average < previous_average:
            down += 1

    # --------------------------------------------------------
    # 2. IMMEDIATE CANDLE BREAK
    # --------------------------------------------------------

    if current_close > previous_high:
        up += 2
        reasons_up.append("bullish break")

    elif current_close < previous_low:
        down += 2
        reasons_down.append("bearish break")

    # --------------------------------------------------------
    # 3. LIQUIDITY / WICK REJECTION
    # --------------------------------------------------------

    body = abs(current_close - current_open)

    candle_range = max(
        current_high - current_low,
        1e-12
    )

    upper_wick = (
        current_high -
        max(current_open, current_close)
    )

    lower_wick = (
        min(current_open, current_close) -
        current_low
    )

    if (
        lower_wick > upper_wick * 1.35
        and current_close > current_open
    ):
        up += 1
        reasons_up.append("lower-wick rejection")

    elif (
        upper_wick > lower_wick * 1.35
        and current_close < current_open
    ):
        down += 1
        reasons_down.append("upper-wick rejection")

    # --------------------------------------------------------
    # 4. MOMENTUM
    # --------------------------------------------------------

    momentum = closes[-1] - closes[-4]

    if momentum > atr_value * 0.20:
        up += 2
        reasons_up.append("positive momentum")

    elif momentum < -atr_value * 0.20:
        down += 2
        reasons_down.append("negative momentum")

    elif momentum > 0:
        up += 1

    elif momentum < 0:
        down += 1

    # --------------------------------------------------------
    # 5. CANDLE QUALITY
    # --------------------------------------------------------

    body_ratio = body / candle_range

    if current_close > current_open and body_ratio >= 0.55:
        up += 2
        reasons_up.append("strong bullish candle")

    elif current_close < current_open and body_ratio >= 0.55:
        down += 2
        reasons_down.append("strong bearish candle")

    elif current_close > current_open:
        up += 1

    else:
        down += 1

    # --------------------------------------------------------
    # 6. RSI 14
    # --------------------------------------------------------

    if 52 <= rsi_value <= 72:
        up += 1
        reasons_up.append("RSI supports up")

    elif 28 <= rsi_value <= 48:
        down += 1
        reasons_down.append("RSI supports down")

    # --------------------------------------------------------
    # 7. EMA 9 / 21
    # --------------------------------------------------------

    if e9 > e21 and current_close > e9:
        up += 2
        reasons_up.append("EMA trend")

    elif e9 < e21 and current_close < e9:
        down += 2
        reasons_down.append("EMA trend")

    elif e9 > e21:
        up += 1

    elif e9 < e21:
        down += 1

    # --------------------------------------------------------
    # 8. WILLIAMS %R + RSI
    # --------------------------------------------------------

    if williams_value > -50 and rsi_value > 50:
        up += 2
        reasons_up.append("oscillators aligned")

    elif williams_value < -50 and rsi_value < 50:
        down += 2
        reasons_down.append("oscillators aligned")

    elif rsi_value > 50:
        up += 1

    elif rsi_value < 50:
        down += 1

    # --------------------------------------------------------
    # 9. MOVING AVERAGE CONFIRMATION
    # --------------------------------------------------------

    if e9 > e21:
        up += 2

    elif e9 < e21:
        down += 2

    # --------------------------------------------------------
    # 10. ADX / DI
    # --------------------------------------------------------

    if adx_value >= 20:

        if di_value > 2:
            up += 2
            reasons_up.append("ADX/DI strength")

        elif di_value < -2:
            down += 2
            reasons_down.append("ADX/DI strength")

        elif up >= down:
            up += 1

        else:
            down += 1

    else:

        if up >= down:
            up += 0.5

        else:
            down += 0.5

    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if up >= down:
        direction = "UP"
        selected_reasons = reasons_up
    else:
        direction = "DOWN"
        selected_reasons = reasons_down

    total = up + down

    if total <= 0:
        total = 1

    edge = abs(up - down)

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    confidence = 55 + (
        edge / total
    ) * 30

    # Weak trend
    if adx_value < 18:
        confidence -= 6

    # Weak candle
    if body_ratio < 0.35:
        confidence -= 3

    # Prevent same direction domination
    with lock:
        recent_directions = [
            item["direction"]
            for item in history[-6:]
        ]

    if recent_directions.count(direction) >= 5:
        confidence -= 7

    confidence = int(
        round(
            clamp(
                confidence,
                55,
                89
            )
        )
    )

    # --------------------------------------------------------
    # ENTRY TIME
    # --------------------------------------------------------

    tf = timeframe.upper()

    delay = {
        "M1": 1,
        "M2": 2,
        "M3": 3
    }.get(tf, 1)

    now = datetime.now(TZ)

    entry_time = now + timedelta(
        minutes=delay
    )

    # --------------------------------------------------------
    # CANCELLATION PRICE
    # --------------------------------------------------------

    if direction == "UP":
        cancel_price = (
            current_close -
            atr_value * 0.35
        )
    else:
        cancel_price = (
            current_close +
            atr_value * 0.35
        )

    reason = ", ".join(
        selected_reasons[:4]
    )

    if not reason:
        reason = "mixed market structure"

    signal = {
        "symbol": symbol,
        "timeframe": tf,
        "direction": direction,
        "confidence": confidence,

        "up_score": round(up, 1),
        "down_score": round(down, 1),

        "entry_after": delay,

        "entry_time": entry_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "entry_price": current_close,

        "cancel_price": cancel_price,

        "reason": reason,

        "generated_at": now.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "candle_time": int(
            current["time"]
        )
    }

    with lock:
        history.append(signal)

        if len(history) > MAX_HISTORY:
            del history[:-MAX_HISTORY]

    return signal


# ============================================================
# FORMAT TELEGRAM SIGNAL
# ============================================================

def format_signal(signal):

    direction = signal["direction"]

    if direction == "UP":
        arrow = "🟢 UP"
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة تحت "
            f"{signal['cancel_price']:.8f}"
        )
    else:
        arrow = "🔴 DOWN"
        cancel_text = (
            f"إلغاء إذا أغلقت الشمعة فوق "
            f"{signal['cancel_price']:.8f}"
        )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {signal['symbol']} | "
        f"{signal['timeframe']}\n\n"

        "🎯 BASE TRADE\n"
        f"{arrow}\n\n"

        f"🔥 Confidence: "
        f"{signal['confidence']}%\n"

        f"🟢 UP Score: "
        f"{signal['up_score']}/18\n"

        f"🔴 DOWN Score: "
        f"{signal['down_score']}/18\n\n"

        f"⏱️ Entry after: "
        f"{signal['entry_after']} minute(s)\n"

        f"🕐 ENTRY TIME: "
        f"{signal['entry_time']} 🇩🇿\n"

        f"💰 ENTRY PRICE: "
        f"{signal['entry_price']:.8f}\n"

        f"⚠️ {cancel_text}\n\n"

        f"🧠 {signal['reason']}"
    )


# ============================================================
# TELEGRAM DIRECT SEND
# ============================================================

def send_telegram_message(text):

    if not BOT_TOKEN:
        logger.warning(
            "BOT_TOKEN is empty - Telegram message not sent"
        )
        return False

    if not OWNER_ID:
        logger.warning(
            "OWNER_ID is empty - Telegram message not sent"
        )
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

        request.add_header(
            "Content-Type",
            "application/x-www-form-urlencoded"
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
# DUPLICATE / COOLDOWN CONTROL
# ============================================================

def should_send_signal(signal):

    global last_sent_key
    global last_sent_time

    now = time.time()

    signal_key = (
        f"{signal['symbol']}_"
        f"{signal['timeframe']}_"
        f"{signal['candle_time']}"
    )

    with lock:

        # Same candle
        if signal_key == last_sent_key:
            logger.info(
                "Duplicate signal ignored: %s",
                signal_key
            )
            return False

        # Cooldown
        if now - last_sent_time < SIGNAL_COOLDOWN:

            logger.info(
                "Signal cooldown active: %s",
                signal["symbol"]
            )

            return False

        last_sent_key = signal_key
        last_sent_time = now

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

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        if self.path in (
            "/",
            "/health"
        ):

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI MT4"
                }
            )

            return

        self.send_json(
            404,
            {
                "error": "not found"
            }
        )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        if self.path != "/mt4":

            self.send_json(
                404,
                {
                    "error": "not found"
                }
            )

            return

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            raw = self.rfile.read(
                content_length
            )

            data = json.loads(
                raw.decode("utf-8")
            )

            # ------------------------------------------------
            # API KEY
            # ------------------------------------------------

            received_key = str(
                data.get(
                    "api_key",
                    ""
                )
            ).strip()

            if not API_KEY:

                logger.error(
                    "ZINO_API_KEY is not configured"
                )

                self.send_json(
                    500,
                    {
                        "ok": False,
                        "error": "server API key not configured"
                    }
                )

                return

            if received_key != API_KEY:

                logger.warning(
                    "Unauthorized MT4 request"
                )

                self.send_json(
                    401,
                    {
                        "ok": False,
                        "error": "unauthorized"
                    }
                )

                return

            # ------------------------------------------------
            # DATA
            # ------------------------------------------------

            symbol = str(
                data.get(
                    "symbol",
                    "UNKNOWN"
                )
            ).strip()

            timeframe = str(
                data.get(
                    "timeframe",
                    "M1"
                )
            ).upper().strip()

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

            telegram_text = format_signal(
                signal
            )

            # ------------------------------------------------
            # SEND TELEGRAM
            # ------------------------------------------------

            telegram_sent = False

            if should_send_signal(signal):

                telegram_sent = (
                    send_telegram_message(
                        telegram_text
                    )
                )

            else:

                logger.info(
                    "Signal not sent because of duplicate/cooldown"
                )

            # ------------------------------------------------
            # RESPONSE TO MT4
            # ------------------------------------------------

            self.send_json(
                200,
                {
                    "ok": True,
                    "signal": signal,
                    "telegram_sent": telegram_sent,
                    "telegram_text": telegram_text
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

    def log_message(self, fmt, *args):

        logger.info(
            "HTTP " + fmt,
            *args
        )


# ============================================================
# HTTP SERVER THREAD
# ============================================================

def run_http_server():

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        Handler
    )

    logger.info(
        "ZinoProSignalAI MT4 HTTP online on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# TELEGRAM COMMANDS
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
        "✅ Online\n"
        "📡 MT4 → Render connected\n"
        "📊 M1 Analysis ready\n\n"
        "/stats\n"
        "/win\n"
        "/loss\n"
        "/reset"
    )


async def stats_command(update, context):

    if not await is_owner(update):
        return

    with lock:

        wins = stats["wins"]
        losses = stats["losses"]

    total = wins + losses

    accuracy = (
        wins / total * 100
        if total > 0
        else 0
    )

    await update.message.reply_text(
        "📚 ZinoProSignalAI STATS\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🟢 WIN: {wins}\n"
        f"🔴 LOSS: {losses}\n"
        f"📊 TOTAL: {total}\n"
        f"🎯 Accuracy: {accuracy:.1f}%"
    )


async def win_command(update, context):

    if not await is_owner(update):
        return

    with lock:
        stats["wins"] += 1

    await update.message.reply_text(
        "🟢 WIN recorded"
    )


async def loss_command(update, context):

    if not await is_owner(update):
        return

    with lock:
        stats["losses"] += 1

    await update.message.reply_text(
        "🔴 LOSS recorded"
    )


async def reset_command(update, context):

    if not await is_owner(update):
        return

    with lock:

        stats["wins"] = 0
        stats["losses"] = 0
        history.clear()

    await update.message.reply_text(
        "♻️ ZinoProSignalAI reset done"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # START HTTP SERVER
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=run_http_server,
        daemon=True
    )

    http_thread.start()

    logger.info(
        "HTTP server started"
    )

    # --------------------------------------------------------
    # CHECK CONFIG
    # --------------------------------------------------------

    if not BOT_TOKEN:

        logger.warning(
            "BOT_TOKEN is missing"
        )

    if not OWNER_ID:

        logger.warning(
            "OWNER_ID is missing"
        )

    if not API_KEY:

        logger.error(
            "ZINO_API_KEY is missing"
        )

    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
