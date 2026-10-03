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
OWNER_ID = os.getenv("OWNER_ID", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MT4_API_KEY = os.getenv("MT4_API_KEY", "").strip()

PORT = int(os.getenv("PORT", "10000"))

ALGIERS = ZoneInfo("Africa/Algiers")


# ============================================================
# ANALYSIS SETTINGS
# ============================================================

# فحص البيانات كل دقيقة
AUTO_ANALYSIS_INTERVAL_MINUTES = 1

# الدخول بعد دقيقتين
ENTRY_DELAY_MINUTES = 2

# أقل مدة بين أي إشارتين
SIGNAL_COOLDOWN_SECONDS = 120

# أقل عدد شموع مغلقة
MIN_CLOSED_CANDLES = 40

# عدد الصفقات في /history
HISTORY_DISPLAY_COUNT = 10

# ملف السجل
HISTORY_FILE = "signal_history.json"


# ============================================================
# GEMINI QUOTA PROTECTION
# ============================================================

# لا نستعمل Gemini في كل دورة.
# أقل مدة بين طلبين Gemini.
GEMINI_MIN_INTERVAL_SECONDS = 900  # 15 دقيقة

# إذا ظهر 429، نوقف Gemini مؤقتًا.
# القيمة الافتراضية 4 ساعات.
GEMINI_429_COOLDOWN_SECONDS = 4 * 60 * 60

# Gemini يستعمل فقط عندما يكون التحليل المحلي قويًا.
LOCAL_GEMINI_MIN_SCORE = 12

# Gemini لا يستطيع تعطيل التحليل المحلي.
GEMINI_OPTIONAL = True


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        logger.info(
            "Gemini client initialized"
        )

    except Exception as e:
        logger.exception(
            "Gemini initialization failed: %s",
            e
        )

else:
    logger.warning(
        "GEMINI_API_KEY is missing"
    )


# آخر طلب Gemini
last_gemini_request_at = 0.0

# حتى متى Gemini متوقف بسبب 429
gemini_blocked_until = 0.0

gemini_lock = threading.Lock()


# ============================================================
# GLOBAL STATE
# ============================================================

mt4_lock = threading.Lock()

mt4_data = {}

telegram_application = None
telegram_loop = None

# آخر وقت أرسلت فيه إشارة
last_signal_sent_at = 0.0

signal_send_lock = threading.Lock()

cycle_lock = threading.Lock()

# يمنع تحليلين متزامنين
analysis_lock = threading.Lock()


# ============================================================
# ACTIVE CYCLE
# ============================================================

active_cycle = {
    "active": False,
    "symbol": None,
    "timeframe": None,
    "trade_type": None,
    "direction": None,
    "recovery_used": False,
    "last_trade_time": 0.0,
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
# TRADE HISTORY
# ============================================================

history_lock = threading.Lock()

trade_history = []

current_trade_id = None


# ============================================================
# HISTORY
# ============================================================

def load_history():

    global trade_history

    try:

        if not os.path.exists(HISTORY_FILE):

            trade_history = []

            return

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            trade_history = data

        else:

            trade_history = []

        logger.info(
            "Trade history loaded: %s records",
            len(trade_history)
        )

    except Exception as e:

        logger.exception(
            "Failed to load history: %s",
            e
        )

        trade_history = []


def save_history():

    try:

        with history_lock:

            data = list(
                trade_history
            )

        temp_file = (
            HISTORY_FILE
            + ".tmp"
        )

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

        os.replace(
            temp_file,
            HISTORY_FILE
        )

    except Exception as e:

        logger.exception(
            "Failed to save history: %s",
            e
        )


# ============================================================
# BASIC HELPERS
# ============================================================

def owner_id_int():

    try:

        return int(
            OWNER_ID
        )

    except Exception:

        return None


def is_owner(
    update: Update
):

    oid = owner_id_int()

    if oid is None:
        return False

    user = update.effective_user

    if user is None:
        return False

    return user.id == oid


def safe_float(
    value,
    default=None
):

    try:

        if value is None:
            return default

        if isinstance(
            value,
            str
        ):

            value = value.strip()

        return float(value)

    except Exception:

        return default


def normalize_timeframe(
    value
):

    if value is None:
        return ""

    value = str(
        value
    ).upper().strip()

    aliases = {

        "1": "M1",
        "1M": "M1",
        "M1": "M1",

        "2": "M2",
        "2M": "M2",
        "M2": "M2",

        "3": "M3",
        "3M": "M3",
        "M3": "M3",

        "5": "M5",
        "5M": "M5",
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
    }

    return aliases.get(
        value,
        value
    )


def timeframe_minutes(
    timeframe
):

    tf = normalize_timeframe(
        timeframe
    )

    values = {
        "M1": 1,
        "M2": 2,
        "M3": 3,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
    }

    return values.get(
        tf,
        60
    )


# ============================================================
# CANDLE NORMALIZATION
# ============================================================

def normalize_candle(
    candle
):

    if not isinstance(
        candle,
        dict
    ):

        return None

    timestamp = (
        candle.get("time")
        or candle.get("timestamp")
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

    o = safe_float(
        open_price
    )

    h = safe_float(
        high_price
    )

    l = safe_float(
        low_price
    )

    c = safe_float(
        close_price
    )

    v = safe_float(
        volume,
        0
    )

    if (
        o is None
        or h is None
        or l is None
        or c is None
    ):

        return None

    return {

        "time": timestamp,

        "open": o,

        "high": h,

        "low": l,

        "close": c,

        "volume": v,
    }


def normalize_candles(
    candles
):

    if not isinstance(
        candles,
        list
    ):

        return []

    result = []

    for candle in candles:

        normalized = normalize_candle(
            candle
        )

        if normalized:

            result.append(
                normalized
            )

    return result


# ============================================================
# INDICATORS
# ============================================================

def ema(
    values,
    period
):

    if (
        not values
        or len(values) < period
    ):

        return None

    multiplier = (
        2
        /
        (period + 1)
    )

    current = (
        sum(
            values[:period]
        )
        /
        period
    )

    for price in values[period:]:

        current = (
            (
                price
                - current
            )
            * multiplier
        ) + current

    return current


def rsi(
    values,
    period=14
):

    if len(values) < period + 1:

        return None

    gains = []
    losses = []

    for i in range(
        1,
        period + 1
    ):

        diff = (
            values[i]
            -
            values[i - 1]
        )

        if diff >= 0:

            gains.append(diff)
            losses.append(0)

        else:

            gains.append(0)
            losses.append(abs(diff))

    avg_gain = (
        sum(gains)
        /
        period
    )

    avg_loss = (
        sum(losses)
        /
        period
    )

    for i in range(
        period + 1,
        len(values)
    ):

        diff = (
            values[i]
            -
            values[i - 1]
        )

        gain = max(
            diff,
            0
        )

        loss = max(
            -diff,
            0
        )

        avg_gain = (
            (
                avg_gain
                *
                (period - 1)
            )
            + gain
        ) / period

        avg_loss = (
            (
                avg_loss
                *
                (period - 1)
            )
            + loss
        ) / period

    if avg_loss == 0:

        return 100.0

    rs = (
        avg_gain
        /
        avg_loss
    )

    return (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )


def williams_r(
    candles,
    period=14
):

    if len(candles) < period:

        return None

    recent = candles[-period:]

    highest = max(
        x["high"]
        for x in recent
    )

    lowest = min(
        x["low"]
        for x in recent
    )

    if highest == lowest:

        return -50.0

    close = recent[-1]["close"]

    return (
        (
            highest
            - close
        )
        /
        (
            highest
            - lowest
        )
    ) * -100


def true_ranges(
    candles
):

    if len(candles) < 2:

        return []

    result = []

    for i in range(
        1,
        len(candles)
    ):

        current = candles[i]

        previous = candles[i - 1]

        tr = max(

            current["high"]
            -
            current["low"],

            abs(
                current["high"]
                -
                previous["close"]
            ),

            abs(
                current["low"]
                -
                previous["close"]
            ),
        )

        result.append(
            tr
        )

    return result


def atr(
    candles,
    period=10
):

    trs = true_ranges(
        candles
    )

    if len(trs) < period:

        return None

    return (
        sum(
            trs[-period:]
        )
        /
        period
    )


def adx_di(
    candles,
    period=14
):

    if len(candles) < period + 2:

        return (
            None,
            None,
            None
        )

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
            -
            previous["high"]
        )

        down_move = (
            previous["low"]
            -
            current["low"]
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
            -
            current["low"],

            abs(
                current["high"]
                -
                previous["close"]
            ),

            abs(
                current["low"]
                -
                previous["close"]
            ),
        )

        trs.append(tr)

        plus_dm.append(
            plus
        )

        minus_dm.append(
            minus
        )

    if len(trs) < period:

        return (
            None,
            None,
            None
        )

    tr_avg = (
        sum(
            trs[-period:]
        )
        /
        period
    )

    plus_avg = (
        sum(
            plus_dm[-period:]
        )
        /
        period
    )

    minus_avg = (
        sum(
            minus_dm[-period:]
        )
        /
        period
    )

    if tr_avg == 0:

        return (
            0.0,
            0.0,
            0.0
        )

    plus_di = (
        100
        *
        plus_avg
        /
        tr_avg
    )

    minus_di = (
        100
        *
        minus_avg
        /
        tr_avg
    )

    denominator = (
        plus_di
        +
        minus_di
    )

    if denominator == 0:

        dx = 0

    else:

        dx = (
            100
            *
            abs(
                plus_di
                -
                minus_di
            )
            /
            denominator
        )

    return (
        dx,
        plus_di,
        minus_di
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

def market_structure(
    candles
):

    if len(candles) < 8:

        return "UNKNOWN"

    recent = candles[-8:]

    first_high = max(
        x["high"]
        for x in recent[:4]
    )

    second_high = max(
        x["high"]
        for x in recent[4:]
    )

    first_low = min(
        x["low"]
        for x in recent[:4]
    )

    second_low = min(
        x["low"]
        for x in recent[4:]
    )

    if (
        second_high > first_high
        and second_low > first_low
    ):

        return "BULLISH"

    if (
        second_high < first_high
        and second_low < first_low
    ):

        return "BEARISH"

    return "RANGE"


def breakout_state(
    candles
):

    if len(candles) < 10:

        return "NONE"

    previous = candles[-9:-1]

    previous_high = max(
        x["high"]
        for x in previous
    )

    previous_low = min(
        x["low"]
        for x in previous
    )

    last_close = candles[-1]["close"]

    if last_close > previous_high:

        return "UP_BREAKOUT"

    if last_close < previous_low:

        return "DOWN_BREAKOUT"

    return "NONE"


# ============================================================
# TECHNICAL SNAPSHOT
# ============================================================

def build_technical_snapshot(
    candles
):

    closes = [
        x["close"]
        for x in candles
    ]

    current = candles[-1]

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    current_rsi = rsi(
        closes,
        14
    )

    current_wr = williams_r(
        candles,
        14
    )

    current_atr = atr(
        candles,
        10
    )

    adx, plus_di, minus_di = (
        adx_di(
            candles,
            14
        )
    )

    keltner_mid = ema(
        closes,
        20
    )

    keltner_atr = atr(
        candles,
        10
    )

    keltner_upper = None
    keltner_lower = None

    if (
        keltner_mid is not None
        and keltner_atr is not None
    ):

        keltner_upper = (
            keltner_mid
            +
            (
                keltner_atr
                * 5
            )
        )

        keltner_lower = (
            keltner_mid
            -
            (
                keltner_atr
                * 5
            )
        )

    structure = market_structure(
        candles
    )

    breakout = breakout_state(
        candles
    )

    recent8 = candles[-8:]

    recent_low = min(
        x["low"]
        for x in recent8
    )

    recent_high = max(
        x["high"]
        for x in recent8
    )

    return {

        "price": current["close"],

        "open": current["open"],

        "high": current["high"],

        "low": current["low"],

        "ema9": ema9,

        "ema21": ema21,

        "rsi14": current_rsi,

        "williams_r14": current_wr,

        "atr10": current_atr,

        "adx14": adx,

        "plus_di14": plus_di,

        "minus_di14": minus_di,

        "keltner_mid": keltner_mid,

        "keltner_upper": keltner_upper,

        "keltner_lower": keltner_lower,

        "structure": structure,

        "breakout": breakout,

        "recent_low": recent_low,

        "recent_high": recent_high,
    }


# ============================================================
# LOCAL 18-POINT ANALYSIS
# ============================================================

def local_directional_analysis(
    candles
):

    if len(candles) < MIN_CLOSED_CANDLES:

        return None

    snapshot = build_technical_snapshot(
        candles
    )

    price = snapshot["price"]

    open_price = snapshot["open"]

    high = snapshot["high"]

    low = snapshot["low"]

    ema9 = snapshot["ema9"]

    ema21 = snapshot["ema21"]

    rsi_value = snapshot["rsi14"]

    wr_value = snapshot["williams_r14"]

    adx_value = snapshot["adx14"]

    plus_di = snapshot["plus_di14"]

    minus_di = snapshot["minus_di14"]

    structure = snapshot["structure"]

    breakout = snapshot["breakout"]

    atr_value = snapshot["atr10"]

    k_mid = snapshot["keltner_mid"]

    k_upper = snapshot["keltner_upper"]

    k_lower = snapshot["keltner_lower"]


    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    up = 0
    down = 0

    reasons_up = []
    reasons_down = []


    # --------------------------------------------------------
    # 1. STRUCTURE = 2
    # --------------------------------------------------------

    if structure == "BULLISH":

        up += 2
        reasons_up.append(
            "bullish structure"
        )

    elif structure == "BEARISH":

        down += 2
        reasons_down.append(
            "bearish structure"
        )

    else:

        # RANGE: no forced structure points
        pass


    # --------------------------------------------------------
    # 2. BREAKOUT = 2
    # --------------------------------------------------------

    if breakout == "UP_BREAKOUT":

        up += 2
        reasons_up.append(
            "upward breakout"
        )

    elif breakout == "DOWN_BREAKOUT":

        down += 2
        reasons_down.append(
            "downward breakout"
        )


    # --------------------------------------------------------
    # 3. LIQUIDITY = 1
    # --------------------------------------------------------

    recent20 = candles[-20:]

    previous_high = max(
        x["high"]
        for x in recent20[:-1]
    )

    previous_low = min(
        x["low"]
        for x in recent20[:-1]
    )

    current_close = candles[-1]["close"]

    liquidity_up = (
        current_close > previous_high
    )

    liquidity_down = (
        current_close < previous_low
    )

    if liquidity_up:

        up += 1

        reasons_up.append(
            "liquidity high taken"
        )

    elif liquidity_down:

        down += 1

        reasons_down.append(
            "liquidity low taken"
        )

    else:

        # no forced point
        pass


    # --------------------------------------------------------
    # 4. MOMENTUM = 2
    # --------------------------------------------------------

    if len(candles) >= 4:

        c1 = candles[-1]["close"]
        c2 = candles[-2]["close"]
        c3 = candles[-3]["close"]
        c4 = candles[-4]["close"]

        up_moves = 0
        down_moves = 0

        if c1 > c2:
            up_moves += 1

        elif c1 < c2:
            down_moves += 1

        if c2 > c3:
            up_moves += 1

        elif c2 < c3:
            down_moves += 1

        if c3 > c4:
            up_moves += 1

        elif c3 < c4:
            down_moves += 1

        if up_moves >= 2:

            up += 2

            reasons_up.append(
                "positive momentum"
            )

        elif down_moves >= 2:

            down += 2

            reasons_down.append(
                "negative momentum"
            )


    # --------------------------------------------------------
    # 5. CANDLE = 2
    # --------------------------------------------------------

    body = abs(
        current_close
        -
        open_price
    )

    candle_range = (
        high
        -
        low
    )

    if candle_range > 0:

        body_ratio = (
            body
            /
            candle_range
        )

        upper_wick = (
            high
            -
            max(
                open_price,
                current_close
            )
        )

        lower_wick = (
            min(
                open_price,
                current_close
            )
            -
            low
        )

        if (
            current_close > open_price
            and body_ratio >= 0.55
        ):

            up += 2

            reasons_up.append(
                "strong bullish candle"
            )

        elif (
            current_close < open_price
            and body_ratio >= 0.55
        ):

            down += 2

            reasons_down.append(
                "strong bearish candle"
            )

        else:

            # weaker candle:
            # assign 1 point only if directional evidence exists

            if (
                current_close > open_price
                and lower_wick > upper_wick
            ):

                up += 1

                reasons_up.append(
                    "bullish rejection"
                )

            elif (
                current_close < open_price
                and upper_wick > lower_wick
            ):

                down += 1

                reasons_down.append(
                    "bearish rejection"
                )


    # --------------------------------------------------------
    # 6. RSI = 1
    # --------------------------------------------------------

    if rsi_value is not None:

        if (
            rsi_value > 50
            and rsi_value < 70
        ):

            up += 1

            reasons_up.append(
                "RSI supports upside"
            )

        elif (
            rsi_value < 50
            and rsi_value > 30
        ):

            down += 1

            reasons_down.append(
                "RSI supports downside"
            )

        # extreme RSI gives no automatic direction
        # because reversal/trend conflict is possible


    # --------------------------------------------------------
    # 7. SUMMARY = 2
    # --------------------------------------------------------

    summary_up = 0
    summary_down = 0

    if structure == "BULLISH":
        summary_up += 1

    elif structure == "BEARISH":
        summary_down += 1

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:
            summary_up += 1

        elif ema9 < ema21:
            summary_down += 1

    if (
        summary_up >= 2
    ):

        up += 2

        reasons_up.append(
            "overall structure alignment"
        )

    elif (
        summary_down >= 2
    ):

        down += 2

        reasons_down.append(
            "overall structure alignment"
        )


    # --------------------------------------------------------
    # 8. OSCILLATORS = 2
    # Williams %R + ADX/DI supporting momentum
    # --------------------------------------------------------

    oscillator_up = 0
    oscillator_down = 0

    if wr_value is not None:

        if (
            wr_value > -80
            and wr_value < -20
        ):

            if wr_value > -50:

                oscillator_up += 1

            elif wr_value < -50:

                oscillator_down += 1

        elif wr_value <= -80:

            # Oversold alone is NOT enough.
            # Only count if price momentum also supports UP.
            if (
                current_close > open_price
            ):

                oscillator_up += 1

        elif wr_value >= -20:

            # Overbought alone is NOT enough.
            # Only count if candle supports DOWN.
            if (
                current_close < open_price
            ):

                oscillator_down += 1


    if (
        adx_value is not None
        and adx_value >= 20
        and plus_di is not None
        and minus_di is not None
    ):

        if plus_di > minus_di:

            oscillator_up += 1

        elif minus_di > plus_di:

            oscillator_down += 1


    if oscillator_up >= 2:

        up += 2

        reasons_up.append(
            "oscillators aligned"
        )

    elif oscillator_down >= 2:

        down += 2

        reasons_down.append(
            "oscillators aligned"
        )

    elif oscillator_up == 1 and oscillator_down == 0:

        up += 1

        reasons_up.append(
            "oscillator support"
        )

    elif oscillator_down == 1 and oscillator_up == 0:

        down += 1

        reasons_down.append(
            "oscillator support"
        )


    # --------------------------------------------------------
    # 9. MOVING AVERAGES = 2
    # --------------------------------------------------------

    ma_up = 0
    ma_down = 0

    if (
        ema9 is not None
        and ema21 is not None
    ):

        if ema9 > ema21:

            ma_up += 1

        elif ema9 < ema21:

            ma_down += 1

        if price > ema9 and price > ema21:

            ma_up += 1

        elif price < ema9 and price < ema21:

            ma_down += 1


    if ma_up >= 2:

        up += 2

        reasons_up.append(
            "EMA 9/21 aligned"
        )

    elif ma_down >= 2:

        down += 2

        reasons_down.append(
            "EMA 9/21 aligned"
        )

    elif ma_up == 1 and ma_down == 0:

        up += 1

        reasons_up.append(
            "EMA support"
        )

    elif ma_down == 1 and ma_up == 0:

        down += 1

        reasons_down.append(
            "EMA support"
        )


    # --------------------------------------------------------
    # NORMALIZE TO EXACTLY 18
    #
    # The category weights above can produce fewer than 18
    # because conflicting/neutral categories don't award both.
    #
    # We allocate the remaining evidence to the stronger side
    # so the displayed pair is always /18.
    # --------------------------------------------------------

    raw_up = up
    raw_down = down

    raw_total = (
        raw_up
        +
        raw_down
    )

    if raw_total <= 0:

        # fallback based on EMA/structure
        if structure == "BULLISH":

            raw_up = 1
            raw_down = 0

        elif structure == "BEARISH":

            raw_up = 0
            raw_down = 1

        elif (
            ema9 is not None
            and ema21 is not None
            and ema9 >= ema21
        ):

            raw_up = 1
            raw_down = 0

        else:

            raw_up = 0
            raw_down = 1

        raw_total = (
            raw_up
            +
            raw_down
        )


    # Convert to 18-point displayed score.
    up_score = round(
        (
            raw_up
            /
            raw_total
        )
        *
        18
    )

    down_score = (
        18
        -
        up_score
    )

    # Avoid 0/18 only when evidence is genuinely weak.
    # Keep the stronger direction.
    if up_score == down_score:

        if raw_up > raw_down:

            up_score = 10
            down_score = 8

        elif raw_down > raw_up:

            up_score = 8
            down_score = 10

        else:

            if structure == "BULLISH":

                up_score = 10
                down_score = 8

            elif structure == "BEARISH":

                up_score = 8
                down_score = 10

            elif (
                ema9 is not None
                and ema21 is not None
                and ema9 >= ema21
            ):

                up_score = 10
                down_score = 8

            else:

                up_score = 8
                down_score = 10


    if up_score > down_score:

        direction = "UP"

    else:

        direction = "DOWN"


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    gap = abs(
        up_score
        -
        down_score
    )

    confidence = (
        50
        +
        int(
            gap
            *
            2.4
        )
    )

    # Do not create fake 90%+ confidence.
    confidence = max(
        51,
        min(
            89,
            confidence
        )
    )

    # If structure strongly contradicts the direction,
    # lower confidence.
    if (
        structure == "BULLISH"
        and direction == "DOWN"
    ):

        confidence = min(
            confidence,
            68
        )

    if (
        structure == "BEARISH"
        and direction == "UP"
    ):

        confidence = min(
            confidence,
            68
        )


    # --------------------------------------------------------
    # REASON
    # --------------------------------------------------------

    if direction == "UP":

        selected_reasons = reasons_up[:4]

    else:

        selected_reasons = reasons_down[:4]

    if selected_reasons:

        reason = (
            " + ".join(
                selected_reasons
            )
        )

    else:

        reason = (
            "Directional evidence from the supplied "
            "closed-candle data."
        )


    cancellation_reason = (
        "Cancel if a closed candle invalidates "
        "the current directional structure."
    )


    # --------------------------------------------------------
    # DATA QUALITY
    # --------------------------------------------------------

    data_quality = "GOOD"

    if len(candles) < 60:

        data_quality = "LIMITED"

    if (
        atr_value is None
        or ema9 is None
        or ema21 is None
    ):

        data_quality = "LIMITED"


    # Keltner informational values
    keltner_position = "MIDDLE"

    if (
        k_upper is not None
        and price > k_upper
    ):

        keltner_position = "ABOVE_UPPER"

    elif (
        k_lower is not None
        and price < k_lower
    ):

        keltner_position = "BELOW_LOWER"


    return {

        "signal": True,

        "direction": direction,

        "confidence": confidence,

        "up_score": int(up_score),

        "down_score": int(down_score),

        "reason": reason,

        "cancellation_reason":
            cancellation_reason,

        "data_quality":
            data_quality,

        "raw_up":
            raw_up,

        "raw_down":
            raw_down,

        "structure":
            structure,

        "breakout":
            breakout,

        "rsi":
            rsi_value,

        "williams_r":
            wr_value,

        "adx":
            adx_value,

        "plus_di":
            plus_di,

        "minus_di":
            minus_di,

        "keltner_position":
            keltner_position,
    }


# ============================================================
# GEMINI QUOTA HELPERS
# ============================================================

def gemini_is_available():

    if not GEMINI_OPTIONAL:
        return False

    if gemini_client is None:
        return False

    now = time.time()

    with gemini_lock:

        if now < gemini_blocked_until:

            return False

        if (
            now
            -
            last_gemini_request_at
            <
            GEMINI_MIN_INTERVAL_SECONDS
        ):

            return False

    return True


def extract_retry_seconds(
    error_text
):

    text = str(
        error_text
    )

    # محاولة استخراج retryDelay
    # مثال: 4h16m43s
    import re

    match = re.search(
        r"(\d+)h(\d+)m(\d+(?:\.\d+)?)s",
        text
    )

    if match:

        hours = int(
            match.group(1)
        )

        minutes = int(
            match.group(2)
        )

        seconds = float(
            match.group(3)
        )

        return int(
            hours * 3600
            +
            minutes * 60
            +
            seconds
        )

    match = re.search(
        r"retryDelay[^0-9]*(\d+)s",
        text,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return GEMINI_429_COOLDOWN_SECONDS


def block_gemini(
    seconds
):

    global gemini_blocked_until

    with gemini_lock:

        gemini_blocked_until = (
            time.time()
            +
            max(
                60,
                int(seconds)
            )
        )

    logger.warning(
        "Gemini temporarily blocked for %s seconds",
        int(seconds)
    )


def gemini_status_text():

    now = time.time()

    with gemini_lock:

        blocked_until = (
            gemini_blocked_until
        )

        last_request = (
            last_gemini_request_at
        )

    if now < blocked_until:

        remaining = int(
            blocked_until
            -
            now
        )

        hours = remaining // 3600

        minutes = (
            remaining % 3600
        ) // 60

        seconds = (
            remaining % 60
        )

        return (
            f"429 cooldown: "
            f"{hours}h {minutes}m {seconds}s"
        )

    elapsed = (
        now
        -
        last_request
    )

    if last_request <= 0:

        return "READY"

    remaining = max(
        0,
        int(
            GEMINI_MIN_INTERVAL_SECONDS
            -
            elapsed
        )
    )

    if remaining > 0:

        return (
            f"rate protection: "
            f"{remaining}s"
        )

    return "READY"


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(
    symbol,
    timeframe,
    candles,
    local_analysis
):

    snapshot = build_technical_snapshot(
        candles
    )

    recent = candles[-25:]

    payload = {

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "candles_count":
            len(candles),

        "technical_snapshot":
            snapshot,

        "local_analysis":
            local_analysis,

        "recent_candles":
            recent,
    }

    return f"""
You are an optional confirmation engine for
ZinoProSignalAI.

Analyze ONLY the supplied closed H1 candle data.

Do NOT invent data.
Do NOT use external market data.
Do NOT assume missing indicators.

The local engine has already calculated a directional
analysis. Your job is ONLY to check whether the local
direction is supported by the supplied evidence.

Direction must be UP or DOWN.

Do not return WAIT.
Do not return NO SIGNAL.
Do not return NEUTRAL.

Do not create 90%+ confidence unless evidence is
exceptionally strong.

Do not reverse the local direction without clear
contradictory evidence.

The displayed score must total exactly 18.

Weights:

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

Return JSON ONLY:

{{
  "direction": "UP",
  "confidence": 75,
  "up_score": 13,
  "down_score": 5,
  "reason": "Short factual reason.",
  "confirmation": true
}}

DATA:

{json.dumps(
    payload,
    ensure_ascii=False
)}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    symbol,
    timeframe,
    candles,
    local_analysis
):

    global last_gemini_request_at

    if not gemini_is_available():

        return None

    prompt = build_gemini_prompt(
        symbol,
        timeframe,
        candles,
        local_analysis
    )

    with gemini_lock:

        last_gemini_request_at = (
            time.time()
        )

    try:

        response = (
            gemini_client
            .models
            .generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.10,
                    response_mime_type="application/json",
                ),
            )
        )

        text = getattr(
            response,
            "text",
            None
        )

        if not text:

            logger.warning(
                "Gemini returned empty response"
            )

            return None

        text = text.strip()

        if text.startswith("```"):

            text = text.replace(
                "```json",
                ""
            )

            text = text.replace(
                "```",
                ""
            )

            text = text.strip()

        result = json.loads(
            text
        )

        return result

    except Exception as e:

        error_text = str(
            e
        )

        if (
            "429"
            in error_text
            or
            "RESOURCE_EXHAUSTED"
            in error_text
        ):

            retry_seconds = (
                extract_retry_seconds(
                    error_text
                )
            )

            block_gemini(
                retry_seconds
            )

            logger.warning(
                "Gemini 429 detected. "
                "Local engine will continue."
            )

            return None

        logger.exception(
            "Gemini analysis failed: %s",
            e
        )

        return None


# ============================================================
# VALIDATE / MERGE ANALYSIS
# ============================================================

def ensure_directional_signal(
    result,
    candles,
    local_analysis
):

    # --------------------------------------------------------
    # If Gemini unavailable -> local analysis
    # --------------------------------------------------------

    if not isinstance(
        result,
        dict
    ):

        return dict(
            local_analysis
        )


    local_direction = (
        local_analysis[
            "direction"
        ]
    )

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

        return dict(
            local_analysis
        )


    # --------------------------------------------------------
    # Do not allow Gemini to arbitrarily flip a strong local
    # setup. It may confirm or reduce confidence.
    # --------------------------------------------------------

    local_up = int(
        local_analysis[
            "up_score"
        ]
    )

    local_down = int(
        local_analysis[
            "down_score"
        ]
    )

    local_gap = abs(
        local_up
        -
        local_down
    )


    if (
        direction != local_direction
        and
        local_gap >= 4
    ):

        logger.info(
            "Gemini direction conflicts with "
            "strong local direction. "
            "Keeping local direction."
        )

        return dict(
            local_analysis
        )


    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    up_score = int(
        safe_float(
            result.get(
                "up_score"
            ),
            local_up
        )
        or local_up
    )

    down_score = int(
        safe_float(
            result.get(
                "down_score"
            ),
            local_down
        )
        or local_down
    )

    up_score = max(
        0,
        min(
            18,
            up_score
        )
    )

    down_score = max(
        0,
        min(
            18,
            down_score
        )
    )

    total = (
        up_score
        +
        down_score
    )

    if total <= 0:

        up_score = local_up
        down_score = local_down

    elif total != 18:

        # Keep ratio while forcing exact 18.
        up_score = round(
            (
                up_score
                /
                total
            )
            *
            18
        )

        up_score = max(
            0,
            min(
                18,
                up_score
            )
        )

        down_score = (
            18
            -
            up_score
        )


    # --------------------------------------------------------
    # Direction from scores
    # --------------------------------------------------------

    if up_score > down_score:

        final_direction = "UP"

    elif down_score > up_score:

        final_direction = "DOWN"

    else:

        final_direction = local_direction


    # Do not let weak Gemini confirmation destroy strong
    # local direction.
    if (
        local_gap >= 4
        and
        final_direction != local_direction
    ):

        final_direction = local_direction

        up_score = local_up
        down_score = local_down


    confidence = safe_float(
        result.get(
            "confidence"
        ),
        local_analysis[
            "confidence"
        ]
    )

    confidence = max(
        1,
        min(
            89,
            int(
                confidence
            )
        )
    )


    reason = str(
        result.get(
            "reason"
        )
        or
        local_analysis[
            "reason"
        ]
    ).strip()


    return {

        "signal": True,

        "direction":
            final_direction,

        "confidence":
            confidence,

        "up_score":
            int(up_score),

        "down_score":
            int(down_score),

        "reason":
            reason,

        "cancellation_reason":
            local_analysis[
                "cancellation_reason"
            ],

        "data_quality":
            local_analysis[
                "data_quality"
            ],

        "structure":
            local_analysis[
                "structure"
            ],

        "breakout":
            local_analysis[
                "breakout"
            ],

        "rsi":
            local_analysis[
                "rsi"
            ],

        "williams_r":
            local_analysis[
                "williams_r"
            ],

        "adx":
            local_analysis[
                "adx"
            ],

        "plus_di":
            local_analysis[
                "plus_di"
            ],

        "minus_di":
            local_analysis[
                "minus_di"
            ],
    }


# ============================================================
# ENTRY / CANCELLATION
# ============================================================

def get_entry_price(
    candles
):

    return candles[-1]["close"]


def get_cancellation_level(
    candles,
    direction
):

    recent = candles[-8:]

    if direction == "UP":

        return min(
            x["low"]
            for x in recent
        )

    return max(
        x["high"]
        for x in recent
    )


def format_price(
    price
):

    if price is None:

        return "N/A"

    if abs(price) >= 1:

        return f"{price:.5f}"

    return f"{price:.6f}"


# ============================================================
# SIGNAL FORMAT
# ============================================================

def format_signal(
    symbol,
    timeframe,
    analysis,
    candles,
    trade_type
):

    direction = analysis[
        "direction"
    ]

    now = datetime.now(
        ALGIERS
    )

    entry_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        +
        timedelta(
            minutes=ENTRY_DELAY_MINUTES
        )
    )

    entry_price = get_entry_price(
        candles
    )

    cancellation_level = (
        get_cancellation_level(
            candles,
            direction
        )
    )

    if direction == "UP":

        cancel_text = (
            "إلغاء إذا أغلقت الشمعة تحت "
            f"{format_price(cancellation_level)}"
        )

    else:

        cancel_text = (
            "إلغاء إذا أغلقت الشمعة فوق "
            f"{format_price(cancellation_level)}"
        )


    if trade_type == "RECOVERY":

        trade_label = (
            "🔁 RECOVERY 1/1"
        )

    else:

        trade_label = (
            "🎯 BASE TRADE"
        )


    direction_icon = (
        "🟢 UP"
        if direction == "UP"
        else
        "🔴 DOWN"
    )


    message = (

        "🎓 ZinoProSignalAI\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"📊 {symbol} | {timeframe}\n"

        f"{trade_label}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"{direction_icon}\n"

        f"🎯 Confidence: "
        f"{analysis['confidence']}%\n"

        f"📈 UP Score: "
        f"{analysis['up_score']}/18\n"

        f"📉 DOWN Score: "
        f"{analysis['down_score']}/18\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"⏳ Entry after: "
        f"{ENTRY_DELAY_MINUTES} min\n"

        f"⏰ Entry Time: "
        f"{entry_time.strftime('%H:%M:%S')} "
        "(Algiers)\n"

        f"💰 Entry Price: "
        f"{format_price(entry_price)}\n"

        f"⚠️ {cancel_text}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"🧠 {analysis['reason']}\n"
    )

    return (
        message,
        entry_time,
        entry_price,
        cancellation_level
    )


# ============================================================
# TRADE HISTORY
# ============================================================

def create_trade_record(
    symbol,
    timeframe,
    analysis,
    candles,
    trade_type,
    entry_time,
    entry_price,
    cancellation_level
):

    now = datetime.now(
        ALGIERS
    )

    trade_id = (
        f"{now.strftime('%Y%m%d%H%M%S')}"
        f"-"
        f"{int(time.time() * 1000) % 1000:03d}"
    )

    return {

        "id":
            trade_id,

        "created_at":
            now.isoformat(),

        "symbol":
            symbol,

        "timeframe":
            timeframe,

        "trade_type":
            trade_type,

        "direction":
            analysis[
                "direction"
            ],

        "confidence":
            analysis[
                "confidence"
            ],

        "up_score":
            analysis[
                "up_score"
            ],

        "down_score":
            analysis[
                "down_score"
            ],

        "entry_time":
            entry_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "entry_price":
            entry_price,

        "cancellation_level":
            cancellation_level,

        "reason":
            analysis[
                "reason"
            ],

        "cancellation_reason":
            analysis[
                "cancellation_reason"
            ],

        "result":
            "PENDING",

        "result_time":
            None,
    }


def add_trade_record(
    record
):

    global current_trade_id

    with history_lock:

        trade_history.append(
            record
        )

        current_trade_id = (
            record["id"]
        )

    save_history()

    logger.info(
        "TRADE RECORDED | %s | %s | %s | %s | %s/%s",
        record["symbol"],
        record["trade_type"],
        record["direction"],
        record["confidence"],
        record["up_score"],
        record["down_score"],
    )


def update_current_trade_result(
    result
):

    global current_trade_id

    if result not in (
        "WIN",
        "LOSS"
    ):

        return False

    updated = False

    now = datetime.now(
        ALGIERS
    )

    with history_lock:

        target_id = (
            current_trade_id
        )

        if target_id is not None:

            for record in reversed(
                trade_history
            ):

                if (
                    record.get("id")
                    ==
                    target_id
                    and
                    record.get("result")
                    ==
                    "PENDING"
                ):

                    record["result"] = (
                        result
                    )

                    record["result_time"] = (
                        now.isoformat()
                    )

                    updated = True

                    break

        if not updated:

            for record in reversed(
                trade_history
            ):

                if (
                    record.get("result")
                    ==
                    "PENDING"
                ):

                    record["result"] = (
                        result
                    )

                    record["result_time"] = (
                        now.isoformat()
                    )

                    current_trade_id = (
                        record.get("id")
                    )

                    updated = True

                    break

    if updated:

        save_history()

        logger.info(
            "TRADE RESULT RECORDED: %s",
            result
        )

    return updated


def history_summary():

    with history_lock:

        records = list(
            trade_history
        )

    total = len(
        records
    )

    wins = sum(
        1
        for x in records
        if x.get("result")
        ==
        "WIN"
    )

    losses = sum(
        1
        for x in records
        if x.get("result")
        ==
        "LOSS"
    )

    pending = sum(
        1
        for x in records
        if x.get("result")
        ==
        "PENDING"
    )

    return (
        records,
        total,
        wins,
        losses,
        pending
    )


# ============================================================
# SIGNAL COOLDOWN
# ============================================================

def signal_cooldown_active():

    elapsed = (
        time.time()
        -
        last_signal_sent_at
    )

    if (
        elapsed
        <
        SIGNAL_COOLDOWN_SECONDS
    ):

        remaining = int(
            SIGNAL_COOLDOWN_SECONDS
            -
            elapsed
        )

        return (
            True,
            remaining
        )

    return (
        False,
        0
    )


# ============================================================
# TELEGRAM SEND
# ============================================================

async def send_telegram_message(
    text_message
):

    global telegram_application

    if telegram_application is None:

        logger.error(
            "Telegram application is not ready"
        )

        return False

    oid = owner_id_int()

    if oid is None:

        logger.error(
            "OWNER_ID is invalid"
        )

        return False

    try:

        await telegram_application.bot.send_message(
            chat_id=oid,
            text=text_message,
        )

        return True

    except Exception as e:

        logger.exception(
            "Telegram send failed: %s",
            e
        )

        return False


def send_signal_safely(
    message
):

    global last_signal_sent_at

    with signal_send_lock:

        active, remaining = (
            signal_cooldown_active()
        )

        if active:

            logger.info(
                "Signal cooldown active: %ss",
                remaining
            )

            return False

        if telegram_loop is None:

            logger.error(
                "Telegram event loop is not ready"
            )

            return False

        future = (
            asyncio.run_coroutine_threadsafe(
                send_telegram_message(
                    message
                ),
                telegram_loop
            )
        )

        try:

            success = future.result(
                timeout=30
            )

        except Exception as e:

            logger.exception(
                "Telegram future failed: %s",
                e
            )

            return False

        if success:

            last_signal_sent_at = (
                time.time()
            )

            return True

        return False


# ============================================================
# CYCLE MANAGEMENT
# ============================================================

def get_active_cycle():

    with cycle_lock:

        return dict(
            active_cycle
        )


def start_base_cycle(
    symbol,
    timeframe,
    direction
):

    with cycle_lock:

        active_cycle[
            "active"
        ] = True

        active_cycle[
            "symbol"
        ] = symbol

        active_cycle[
            "timeframe"
        ] = timeframe

        active_cycle[
            "trade_type"
        ] = "BASE"

        active_cycle[
            "direction"
        ] = direction

        active_cycle[
            "recovery_used"
        ] = False

        active_cycle[
            "last_trade_time"
        ] = time.time()

        active_cycle[
            "trade_number"
        ] = 1


def start_recovery_cycle():

    with cycle_lock:

        active_cycle[
            "active"
        ] = True

        active_cycle[
            "trade_type"
        ] = "RECOVERY"

        active_cycle[
            "recovery_used"
        ] = True

        active_cycle[
            "trade_number"
        ] = 2

        active_cycle[
            "last_trade_time"
        ] = time.time()


def reset_cycle():

    with cycle_lock:

        active_cycle[
            "active"
        ] = False

        active_cycle[
            "symbol"
        ] = None

        active_cycle[
            "timeframe"
        ] = None

        active_cycle[
            "trade_type"
        ] = None

        active_cycle[
            "direction"
        ] = None

        active_cycle[
            "recovery_used"
        ] = False

        active_cycle[
            "last_trade_time"
        ] = 0.0

        active_cycle[
            "trade_number"
        ] = 0


# ============================================================
# BEST PAIR SELECTION
# ============================================================

def choose_best_pair():

    with mt4_lock:

        candidates = []

        for symbol, timeframes in (
            mt4_data.items()
        ):

            if not isinstance(
                timeframes,
                dict
            ):

                continue

            candles = timeframes.get(
                "H1"
            )

            if not candles:

                continue

            closed = candles[:-1]

            if len(closed) < (
                MIN_CLOSED_CANDLES
            ):

                continue

            analysis = (
                local_directional_analysis(
                    closed
                )
            )

            if not analysis:

                continue

            candidates.append({

                "symbol":
                    symbol,

                "timeframe":
                    "H1",

                "score":
                    max(
                        analysis[
                            "up_score"
                        ],
                        analysis[
                            "down_score"
                        ]
                    ),

                "direction":
                    analysis[
                        "direction"
                    ],

                "up":
                    analysis[
                        "up_score"
                    ],

                "down":
                    analysis[
                        "down_score"
                    ],

                "confidence":
                    analysis[
                        "confidence"
                    ],
            })


    if not candidates:

        return None


    candidates.sort(
        key=lambda x: (
            x["confidence"],
            x["score"],
            abs(
                x["up"]
                -
                x["down"]
            ),
        ),
        reverse=True
    )


    cycle = get_active_cycle()

    if cycle["active"]:

        for candidate in candidates:

            if (
                candidate["symbol"]
                ==
                cycle["symbol"]
            ):

                return candidate

        return None


    return candidates[0]


# ============================================================
# AUTO ANALYSIS
# ============================================================

def auto_analyze_pair(
    symbol,
    timeframe
):

    if not analysis_lock.acquire(
        blocking=False
    ):

        logger.info(
            "Another analysis is already running"
        )

        return

    try:

        timeframe = normalize_timeframe(
            timeframe
        )

        if timeframe != "H1":

            return


        # ----------------------------------------------------
        # Cycle state
        # ----------------------------------------------------

        cycle = get_active_cycle()


        # ----------------------------------------------------
        # IMPORTANT:
        # If BASE/RECOVERY is already pending,
        # DO NOT send another trade.
        # User must use /win or /loss.
        # ----------------------------------------------------

        if cycle["active"]:

            if (
                cycle["symbol"]
                != symbol
            ):

                return

            if (
                cycle["timeframe"]
                != timeframe
            ):

                return

            if (
                cycle["trade_type"]
                in (
                    "BASE",
                    "RECOVERY"
                )
            ):

                logger.info(
                    "Active %s trade is pending. "
                    "Waiting for /win or /loss.",
                    cycle["trade_type"]
                )

                return


        # ----------------------------------------------------
        # Get MT4 data
        # ----------------------------------------------------

        with mt4_lock:

            timeframes = mt4_data.get(
                symbol
            )

            if not timeframes:

                return

            candles = timeframes.get(
                timeframe
            )

            if not candles:

                return

            candles = list(
                candles
            )


        if len(candles) < (
            MIN_CLOSED_CANDLES
            +
            1
        ):

            logger.info(
                "%s %s: not enough candles: %s",
                symbol,
                timeframe,
                len(candles)
            )

            return


        # ----------------------------------------------------
        # Remove current forming candle
        # ----------------------------------------------------

        closed = candles[:-1]

        if len(closed) < (
            MIN_CLOSED_CANDLES
        ):

            return


        # ----------------------------------------------------
        # LOCAL ANALYSIS
        # ----------------------------------------------------

        local_analysis = (
            local_directional_analysis(
                closed
            )
        )

        if not local_analysis:

            return


        logger.info(
            "LOCAL ANALYSIS | %s %s | %s | "
            "UP=%s DOWN=%s CONF=%s",
            symbol,
            timeframe,
            local_analysis[
                "direction"
            ],
            local_analysis[
                "up_score"
            ],
            local_analysis[
                "down_score"
            ],
            local_analysis[
                "confidence"
            ],
        )


        # ----------------------------------------------------
        # Gemini only as OPTIONAL confirmation.
        #
        # It is NOT called every minute.
        # ----------------------------------------------------

        analysis = local_analysis

        strongest_score = max(
            local_analysis[
                "up_score"
            ],
            local_analysis[
                "down_score"
            ]
        )


        if (
            strongest_score
            >=
            LOCAL_GEMINI_MIN_SCORE
            and
            gemini_is_available()
        ):

            logger.info(
                "Strong local setup. "
                "Gemini confirmation may be used."
            )

            gemini_result = (
                analyze_with_gemini(
                    symbol,
                    timeframe,
                    closed,
                    local_analysis
                )
            )

            if gemini_result is not None:

                analysis = (
                    ensure_directional_signal(
                        gemini_result,
                        closed,
                        local_analysis
                    )
                )

                logger.info(
                    "Gemini confirmation received."
                )

            else:

                logger.info(
                    "Gemini unavailable. "
                    "Using local analysis."
                )


        # ----------------------------------------------------
        # Trade type
        # ----------------------------------------------------

        cycle = get_active_cycle()

        if cycle["active"]:

            trade_type = (
                cycle["trade_type"]
            )

        else:

            trade_type = "BASE"


        # ----------------------------------------------------
        # Final score validation
        # ----------------------------------------------------

        up_score = int(
            analysis[
                "up_score"
            ]
        )

        down_score = int(
            analysis[
                "down_score"
            ]
        )

        if (
            up_score
            +
            down_score
            !=
            18
        ):

            # Force exact 18
            if up_score >= down_score:

                up_score = min(
                    18,
                    max(
                        0,
                        up_score
                    )
                )

                down_score = (
                    18
                    -
                    up_score
                )

            else:

                down_score = min(
                    18,
                    max(
                        0,
                        down_score
                    )
                )

                up_score = (
                    18
                    -
                    down_score
                )

            analysis[
                "up_score"
            ] = up_score

            analysis[
                "down_score"
            ] = down_score


        # ----------------------------------------------------
        # Direction must come from actual scores
        # ----------------------------------------------------

        if up_score > down_score:

            analysis[
                "direction"
            ] = "UP"

        elif down_score > up_score:

            analysis[
                "direction"
            ] = "DOWN"

        else:

            # Tie breaker from local engine
            analysis[
                "direction"
            ] = local_analysis[
                "direction"
            ]


        # ----------------------------------------------------
        # Global signal cooldown
        # ----------------------------------------------------

        active, remaining = (
            signal_cooldown_active()
        )

        if active:

            logger.info(
                "Global signal cooldown: %ss",
                remaining
            )

            return


        # ----------------------------------------------------
        # Build signal
        # ----------------------------------------------------

        (
            message,
            entry_time,
            entry_price,
            cancellation_level
        ) = format_signal(
            symbol,
            timeframe,
            analysis,
            closed,
            trade_type
        )


        # ----------------------------------------------------
        # SEND
        # ----------------------------------------------------

        sent = send_signal_safely(
            message
        )

        if not sent:

            logger.info(
                "Signal was not sent."
            )

            return


        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        record = create_trade_record(
            symbol=symbol,
            timeframe=timeframe,
            analysis=analysis,
            candles=closed,
            trade_type=trade_type,
            entry_time=entry_time,
            entry_price=entry_price,
            cancellation_level=cancellation_level,
        )

        add_trade_record(
            record
        )


        # ----------------------------------------------------
        # START BASE / UPDATE RECOVERY
        # ----------------------------------------------------

        if not cycle["active"]:

            start_base_cycle(
                symbol,
                timeframe,
                analysis[
                    "direction"
                ]
            )

        else:

            with cycle_lock:

                active_cycle[
                    "last_trade_time"
                ] = time.time()

                active_cycle[
                    "direction"
                ] = analysis[
                    "direction"
                ]


        logger.info(
            "SIGNAL SENT | %s | %s | %s | %s | "
            "UP=%s DOWN=%s",
            symbol,
            timeframe,
            trade_type,
            analysis[
                "direction"
            ],
            analysis[
                "up_score"
            ],
            analysis[
                "down_score"
            ]
        )


    except Exception as e:

        logger.exception(
            "Auto analysis failed for %s %s: %s",
            symbol,
            timeframe,
            e
        )

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

            cycle = get_active_cycle()

            # ------------------------------------------------
            # If a trade is pending, wait for /win or /loss.
            # ------------------------------------------------

            if cycle["active"]:

                logger.info(
                    "Cycle active: %s %s %s. "
                    "Waiting for result.",
                    cycle["trade_type"],
                    cycle["symbol"],
                    cycle["direction"]
                )

            else:

                best = choose_best_pair()

                if best:

                    auto_analyze_pair(
                        best["symbol"],
                        "H1"
                    )

                else:

                    logger.info(
                        "No suitable H1 candidate yet"
                    )

        except Exception as e:

            logger.exception(
                "Background loop error: %s",
                e
            )

        time.sleep(
            AUTO_ANALYSIS_INTERVAL_MINUTES
            *
            60
        )


# ============================================================
# MT4 DATA
# ============================================================

def store_mt4_data(
    payload
):

    symbol = str(
        payload.get("symbol")
        or
        payload.get("Symbol")
        or
        ""
    ).upper().strip()

    timeframe = normalize_timeframe(
        payload.get("timeframe")
        or
        payload.get("Timeframe")
        or
        payload.get("tf")
    )

    candles = (
        payload.get("candles")
        or
        payload.get("data")
        or
        []
    )

    if not symbol:

        raise ValueError(
            "Missing symbol"
        )

    if not timeframe:

        raise ValueError(
            "Missing timeframe"
        )

    normalized = normalize_candles(
        candles
    )

    if not normalized:

        raise ValueError(
            "No valid candles"
        )

    with mt4_lock:

        if symbol not in mt4_data:

            mt4_data[symbol] = {}

        mt4_data[symbol][
            timeframe
        ] = normalized

    logger.info(
        "MT4 data stored: %s %s candles=%s",
        symbol,
        timeframe,
        len(normalized)
    )

    return (
        symbol,
        timeframe,
        normalized
    )


# ============================================================
# HTTP SERVER
# ============================================================

class MT4Handler(
    BaseHTTPRequestHandler
):

    def do_GET(
        self
    ):

        parsed = urlparse(
            self.path
        )

        if parsed.path in (
            "/",
            "/health",
            "/healthz",
        ):

            body = (
                "ZinoProSignalAI is running"
            ).encode(
                "utf-8"
            )

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )

            return


        if parsed.path == "/mt4":

            body = (
                "ZinoProSignalAI MT4 endpoint"
            ).encode(
                "utf-8"
            )

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )

            return


        self.send_response(
            404
        )

        self.end_headers()


    def do_POST(
        self
    ):

        parsed = urlparse(
            self.path
        )

        if parsed.path != "/mt4":

            self.send_response(
                404
            )

            self.end_headers()

            return


        try:

            received_key = (
                self.headers.get(
                    "X-MT4-API-Key"
                )
                or
                self.headers.get(
                    "X-API-Key"
                )
                or
                ""
            ).strip()


            if MT4_API_KEY:

                if (
                    received_key
                    !=
                    MT4_API_KEY
                ):

                    logger.warning(
                        "MT4 request rejected: invalid API key"
                    )

                    body = (
                        b"Invalid API key"
                    )

                    self.send_response(
                        401
                    )

                    self.send_header(
                        "Content-Type",
                        "text/plain; charset=utf-8"
                    )

                    self.send_header(
                        "Content-Length",
                        str(len(body))
                    )

                    self.end_headers()

                    self.wfile.write(
                        body
                    )

                    return


            content_length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )

            if content_length <= 0:

                body = (
                    b"Empty request"
                )

                self.send_response(
                    400
                )

                self.send_header(
                    "Content-Type",
                    "text/plain; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(
                    body
                )

                return


            raw_body = self.rfile.read(
                content_length
            )

            payload = json.loads(
                raw_body.decode(
                    "utf-8"
                )
            )


            (
                symbol,
                timeframe,
                candles
            ) = store_mt4_data(
                payload
            )


            # التحليل في thread مستقل
            analysis_thread = threading.Thread(
                target=auto_analyze_pair,
                args=(
                    symbol,
                    timeframe,
                ),
                daemon=True,
            )

            analysis_thread.start()


            response = {

                "ok": True,

                "symbol":
                    symbol,

                "timeframe":
                    timeframe,

                "candles":
                    len(candles),

                "analysis":
                    "local_first_gemini_optional",

                "gemini":
                    gemini_status_text(),
            }


            body = json.dumps(
                response
            ).encode(
                "utf-8"
            )


            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "application/json; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )


        except json.JSONDecodeError:

            logger.exception(
                "Invalid JSON received from MT4"
            )

            body = (
                b"Invalid JSON"
            )

            self.send_response(
                400
            )

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )


        except Exception as e:

            logger.exception(
                "MT4 POST error: %s",
                e
            )

            body = (
                f"Server error: {e}"
            ).encode(
                "utf-8"
            )

            self.send_response(
                500
            )

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )


    def log_message(
        self,
        format_string,
        *args
    ):

        logger.info(
            "HTTP %s - %s",
            self.address_string(),
            format_string % args
        )


def start_http_server():

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        MT4Handler,
    )

    logger.info(
        "HTTP server listening on port %s",
        PORT
    )

    server.serve_forever()


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return

    await update.message.reply_text(

        "🎓 ZinoProSignalAI\n\n"

        "✅ Bot is running\n\n"

        "📡 MT4 → Render → Local Analysis\n"

        "🧠 Gemini = Optional Confirmation\n\n"

        "⚙️ Analysis: H1\n"

        "⏱️ Entry delay: 2 min\n"

        "⏳ Signal cooldown: 2 min\n"

        "🔁 Recovery limit: 1/1\n\n"

        "📊 /stats\n"

        "📚 /history\n"

        "📡 /mt4status\n"

        "🔎 /analyze\n\n"

        "🟢 /win\n"

        "🔴 /loss\n"

        "♻️ /reset"
    )


# ============================================================
# /STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    total = (
        stats_data["wins"]
        +
        stats_data["losses"]
    )


    if total > 0:

        winrate = (
            stats_data["wins"]
            /
            total
        ) * 100

    else:

        winrate = 0


    cycle = get_active_cycle()


    if cycle["active"]:

        cycle_text = (
            f"{cycle['trade_type']} | "
            f"{cycle['symbol']} | "
            f"{cycle['timeframe']} | "
            f"{cycle['direction']}"
        )

    else:

        cycle_text = (
            "No active cycle"
        )


    (
        records,
        history_total,
        history_wins,
        history_losses,
        history_pending
    ) = history_summary()


    if (
        history_wins
        +
        history_losses
        >
        0
    ):

        history_winrate = (
            history_wins
            /
            (
                history_wins
                +
                history_losses
            )
        ) * 100

    else:

        history_winrate = 0


    text = (

        "📊 ZinoProSignalAI Stats\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"Current session trades: "
        f"{total}\n"

        f"🟢 Wins: "
        f"{stats_data['wins']}\n"

        f"🔴 Losses: "
        f"{stats_data['losses']}\n"

        f"📈 Win rate: "
        f"{winrate:.1f}%\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"🎯 BASE wins: "
        f"{stats_data['base_wins']}\n"

        f"🎯 BASE losses: "
        f"{stats_data['base_losses']}\n"

        f"🔁 RECOVERY wins: "
        f"{stats_data['recovery_wins']}\n"

        f"🔁 RECOVERY losses: "
        f"{stats_data['recovery_losses']}\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"📚 Recorded history: "
        f"{history_total}\n"

        f"🟢 WIN: "
        f"{history_wins}\n"

        f"🔴 LOSS: "
        f"{history_losses}\n"

        f"⏳ Pending: "
        f"{history_pending}\n"

        f"📈 History win rate: "
        f"{history_winrate:.1f}%\n"

        "━━━━━━━━━━━━━━━━━━\n"

        f"🧠 Gemini: "
        f"{gemini_status_text()}\n"

        f"🔄 Cycle: "
        f"{cycle_text}"
    )


    await update.message.reply_text(
        text
    )


# ============================================================
# /HISTORY
# ============================================================

async def history_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    with history_lock:

        records = list(
            trade_history[
                -HISTORY_DISPLAY_COUNT:
            ]
        )


    if not records:

        await update.message.reply_text(
            "📚 لا توجد صفقات مسجلة حتى الآن."
        )

        return


    lines = [

        "📚 ZinoProSignalAI HISTORY",

        "━━━━━━━━━━━━━━━━━━"
    ]


    for record in reversed(
        records
    ):

        result = record.get(
            "result",
            "PENDING"
        )


        if result == "WIN":

            result_icon = "🟢"

        elif result == "LOSS":

            result_icon = "🔴"

        else:

            result_icon = "⏳"


        direction = record.get(
            "direction",
            "?"
        )


        direction_icon = (
            "🟢"
            if direction == "UP"
            else
            "🔴"
        )


        trade_type = record.get(
            "trade_type",
            "?"
        )


        confidence = record.get(
            "confidence",
            "?"
        )


        up_score = record.get(
            "up_score",
            "?"
        )


        down_score = record.get(
            "down_score",
            "?"
        )


        entry_price = format_price(
            safe_float(
                record.get(
                    "entry_price"
                )
            )
        )


        lines.append(

            f"{result_icon} "
            f"{record.get('symbol', '?')} "
            f"{record.get('timeframe', '?')}\n"

            f"   {trade_type} | "
            f"{direction_icon} {direction}\n"

            f"   🎯 {confidence}% | "
            f"📈 {up_score}/18 "
            f"📉 {down_score}/18\n"

            f"   💰 {entry_price}\n"

            f"   ⏰ "
            f"{record.get('entry_time', '?')}"
        )


        lines.append(
            "━━━━━━━━━━━━━━━━━━"
        )


    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# /WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    cycle = get_active_cycle()


    if not cycle["active"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة نشطة."
        )

        return


    updated = update_current_trade_result(
        "WIN"
    )


    if not updated:

        await update.message.reply_text(
            "⚠️ لم أجد صفقة PENDING لتسجيل WIN."
        )

        return


    stats_data["wins"] += 1


    if cycle["trade_type"] == "BASE":

        stats_data[
            "base_wins"
        ] += 1

        result_text = (

            "🟢 BASE WIN\n\n"

            "📚 تم تسجيل الصفقة: WIN\n\n"

            "✅ انتهت الدورة بنجاح.\n"

            "🔎 سيتم البحث عن زوج جديد."
        )

    else:

        stats_data[
            "recovery_wins"
        ] += 1

        result_text = (

            "🟢 RECOVERY WIN\n\n"

            "📚 تم تسجيل الصفقة: WIN\n\n"

            "✅ Recovery نجحت.\n"

            "🔎 انتهت الدورة وسيتم البحث عن زوج جديد."
        )


    reset_cycle()


    await update.message.reply_text(
        result_text
    )


# ============================================================
# /LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    cycle = get_active_cycle()


    if not cycle["active"]:

        await update.message.reply_text(
            "ℹ️ لا توجد صفقة نشطة."
        )

        return


    updated = update_current_trade_result(
        "LOSS"
    )


    if not updated:

        await update.message.reply_text(
            "⚠️ لم أجد صفقة PENDING لتسجيل LOSS."
        )

        return


    stats_data["losses"] += 1


    if cycle["trade_type"] == "BASE":

        stats_data[
            "base_losses"
        ] += 1


        start_recovery_cycle()


        await update.message.reply_text(

            "🔴 BASE LOSS\n\n"

            "📚 تم تسجيل الصفقة: LOSS\n\n"

            "🔁 Recovery 1/1 مسموح.\n"

            "⏱️ سيتم تحليل نفس الزوج.\n"

            "🚫 لا توجد Recovery ثانية."
        )

        return


    stats_data[
        "recovery_losses"
    ] += 1


    reset_cycle()


    await update.message.reply_text(

        "🔴 RECOVERY LOSS\n\n"

        "📚 تم تسجيل الصفقة: LOSS\n\n"

        "⛔ انتهت الدورة.\n"

        "🚫 لا توجد Recovery ثانية.\n"

        "🔎 سيتم البحث عن زوج جديد."
    )


# ============================================================
# /RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    stats_data[
        "wins"
    ] = 0

    stats_data[
        "losses"
    ] = 0

    stats_data[
        "base_wins"
    ] = 0

    stats_data[
        "base_losses"
    ] = 0

    stats_data[
        "recovery_wins"
    ] = 0

    stats_data[
        "recovery_losses"
    ] = 0


    reset_cycle()


    global last_signal_sent_at
    global current_trade_id

    last_signal_sent_at = 0.0

    current_trade_id = None


    with history_lock:

        trade_history.clear()


    save_history()


    await update.message.reply_text(

        "♻️ تم تصفير:\n\n"

        "• الإحصائيات\n"

        "• الدورة\n"

        "• سجل الصفقات\n\n"

        "✅ التحليل المحلي وGemini protection "
        "مازالا يعملان."
    )


# ============================================================
# /MT4STATUS
# ============================================================

async def mt4status_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    with mt4_lock:

        data_copy = {

            symbol: dict(
                timeframes
            )

            for symbol, timeframes
            in mt4_data.items()
        }


    if not data_copy:

        await update.message.reply_text(

            "📡 لا توجد بيانات MT4 "
            "مستلمة حتى الآن."
        )

        return


    lines = [

        "📡 MT4 STATUS",

        "━━━━━━━━━━━━━━━━━━"
    ]


    for symbol, timeframes in (
        data_copy.items()
    ):

        for timeframe, candles in (
            timeframes.items()
        ):

            lines.append(

                f"📊 {symbol} {timeframe}: "
                f"{len(candles)} candles"
            )


    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🧠 Gemini: {gemini_status_text()}"
    )


    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# /ANALYZE
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    cycle = get_active_cycle()


    if cycle["active"]:

        await update.message.reply_text(

            "⚠️ توجد دورة نشطة بالفعل:\n\n"

            f"📊 {cycle['symbol']} "
            f"{cycle['timeframe']}\n"

            f"🎯 {cycle['trade_type']}\n"

            f"🧭 {cycle['direction']}\n\n"

            "استعمل /win أو /loss لتسجيل النتيجة."
        )

        return


    best = choose_best_pair()


    if not best:

        await update.message.reply_text(

            "⏳ لا توجد بيانات H1 "
            "كافية من MT4."
        )

        return


    await update.message.reply_text(

        "🔎 أقوى مرشح محلي حاليًا:\n\n"

        f"📊 {best['symbol']} | H1\n"

        f"🧭 Bias: {best['direction']}\n"

        f"📈 UP: {best['up']}/18\n"

        f"📉 DOWN: {best['down']}/18\n"

        f"🎯 Confidence: {best['confidence']}%\n\n"

        "🧠 سيبدأ التحليل المحلي، "
        "وGemini سيكون تأكيدًا اختياريًا."
    )


    thread = threading.Thread(

        target=auto_analyze_pair,

        args=(

            best["symbol"],

            "H1",

        ),

        daemon=True,
    )


    thread.start()


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
        update.message.text
        or
        ""
    )


    if text.startswith("/"):

        return


    await update.message.reply_text(

        "📡 النظام الحالي يعتمد على بيانات MT4.\n\n"

        "MT4 → Render → Local Analysis\n"

        "🧠 Gemini = Optional Confirmation\n\n"

        "استعمل /mt4status لمعرفة حالة البيانات."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        return


    await update.message.reply_text(

        "📸 الوضع الحالي لا يعتمد على الصور.\n\n"

        "التحليل يتم من بيانات MT4 المباشرة."
    )


# ============================================================
# TELEGRAM ERROR
# ============================================================

async def telegram_error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.exception(
        "Telegram error: %s",
        context.error
    )


# ============================================================
# TELEGRAM APPLICATION
# ============================================================

def build_telegram_application():

    global telegram_application


    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN is missing"
        )


    application = (
        Application.builder()
        .token(BOT_TOKEN)
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


    application.add_handler(
        MessageHandler(
            filters.TEXT
            &
            ~filters.COMMAND,
            text_handler
        )
    )


    application.add_error_handler(
        telegram_error_handler
    )


    telegram_application = (
        application
    )


    return application


# ============================================================
# MAIN
# ============================================================

def main():

    global telegram_loop


    logger.info(
        "========================================"
    )

    logger.info(
        "Starting ZinoProSignalAI"
    )

    logger.info(
        "Model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Port: %s",
        PORT
    )

    logger.info(
        "Timezone: Africa/Algiers"
    )

    logger.info(
        "Analysis timeframe: H1"
    )

    logger.info(
        "Entry delay: %s minutes",
        ENTRY_DELAY_MINUTES
    )

    logger.info(
        "Signal cooldown: %s seconds",
        SIGNAL_COOLDOWN_SECONDS
    )

    logger.info(
        "Local analysis: ENABLED"
    )

    logger.info(
        "Gemini optional: %s",
        GEMINI_OPTIONAL
    )

    logger.info(
        "Gemini minimum interval: %s seconds",
        GEMINI_MIN_INTERVAL_SECONDS
    )

    logger.info(
        "Recovery limit: 1"
    )

    logger.info(
        "Trade history file: %s",
        HISTORY_FILE
    )

    logger.info(
        "========================================"
    )


    # --------------------------------------------------------
    # LOAD HISTORY
    # --------------------------------------------------------

    load_history()


    # --------------------------------------------------------
    # HTTP
    # --------------------------------------------------------

    http_thread = threading.Thread(
        target=start_http_server,
        daemon=True,
    )

    http_thread.start()


    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    analysis_thread = threading.Thread(
        target=background_analysis_loop,
        daemon=True,
    )

    analysis_thread.start()


    # --------------------------------------------------------
    # TELEGRAM
    # --------------------------------------------------------

    application = (
        build_telegram_application()
    )


    telegram_loop = (
        asyncio.new_event_loop()
    )


    asyncio.set_event_loop(
        telegram_loop
    )


    logger.info(
        "Telegram bot starting"
    )


    application.run_polling(
        close_loop=False
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
