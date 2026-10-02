import os
import io
import json
import logging
import threading
import asyncio
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
# SETTINGS
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_ID_RAW = os.getenv("OWNER_ID")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

MT4_API_KEY = os.getenv("MT4_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID_RAW:
    raise RuntimeError("OWNER_ID is missing")

if not MT4_API_KEY:
    raise RuntimeError("MT4_API_KEY is missing")

try:
    OWNER_ID = int(OWNER_ID_RAW)
except ValueError:
    raise RuntimeError("OWNER_ID must be an integer")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GEMINI RETRY
# ============================================================

async def gemini_generate_with_retry(
    model,
    contents,
    config,
    max_retries=4,
):
    """
    إعادة المحاولة تلقائياً عند أخطاء Gemini المؤقتة.
    
    المحاولات:
    1) مباشرة
    2) بعد 3 ثواني
    3) بعد 7 ثواني
    4) بعد 15 ثانية
    
    ملاحظة:
    إذا كان الخطأ غير مؤقت، يتم إظهاره مباشرة.
    """

    delays = [3, 7, 15]

    for attempt in range(max_retries):

        try:

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )

            return response

        except Exception as error:

            error_text = str(error)

            retryable = any(
                code in error_text
                for code in (
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                    "UNAVAILABLE",
                    "RESOURCE_EXHAUSTED",
                    "DEADLINE_EXCEEDED",
                )
            )

            if not retryable:
                raise

            if attempt >= max_retries - 1:
                raise

            delay = delays[attempt]

            logger.warning(
                "Gemini temporary error: %s | retrying in %s seconds | attempt %s/%s",
                error_text[:250],
                delay,
                attempt + 1,
                max_retries,
            )

            await asyncio.sleep(delay)

    raise RuntimeError(
        "Gemini failed after multiple retries"
    )


# ============================================================
# STATS
# ============================================================

wins = 0
losses = 0


# ============================================================
# MT4 DATA STORAGE
# ============================================================

mt4_data = {}

mt4_lock = threading.Lock()


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def send_json(
        self,
        status_code,
        data,
    ):

        body = json.dumps(
            data,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(status_code)

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )

        self.send_header(
            "Content-Length",
            str(len(body)),
        )

        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )

        self.end_headers()

        self.wfile.write(body)

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path == "/":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                },
            )

            return

        if path == "/health":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "service": "ZinoProSignalAI",
                },
            )

            return

        if path == "/mt4":

            self.send_json(
                200,
                {
                    "status": "ok",
                    "endpoint": "/mt4",
                    "message": "MT4 endpoint is ready",
                },
            )

            return

        self.send_json(
            404,
            {
                "error": "Not found"
            },
        )

    def do_HEAD(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )

        self.end_headers()

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        if path != "/mt4":

            self.send_json(
                404,
                {
                    "error": "Not found"
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

        except Exception:

            self.send_json(
                400,
                {
                    "error": "Invalid Content-Length"
                },
            )

            return

        if content_length <= 0:

            self.send_json(
                400,
                {
                    "error": "Empty request"
                },
            )

            return

        if content_length > 2_000_000:

            self.send_json(
                413,
                {
                    "error": "Request too large"
                },
            )

            return

        try:

            raw_body = self.rfile.read(
                content_length
            )

            body = raw_body.decode(
                "utf-8"
            )

            data = json.loads(body)

        except Exception as error:

            logger.error(
                "Invalid MT4 JSON: %s",
                error,
            )

            self.send_json(
                400,
                {
                    "error": "Invalid JSON"
                },
            )

            return

        if not isinstance(data, dict):

            self.send_json(
                400,
                {
                    "error": "JSON must be an object"
                },
            )

            return

        received_key = str(
            data.get(
                "api_key",
                "",
            )
        )

        if received_key != MT4_API_KEY:

            logger.warning(
                "Unauthorized MT4 request"
            )

            self.send_json(
                401,
                {
                    "error": "Unauthorized"
                },
            )

            return

        symbol = str(
            data.get(
                "symbol",
                "",
            )
        ).strip().upper()

        timeframe = str(
            data.get(
                "timeframe",
                "",
            )
        ).strip().upper()

        candles = data.get(
            "candles",
            [],
        )

        if not symbol:

            self.send_json(
                400,
                {
                    "error": "symbol is required"
                },
            )

            return

        if not timeframe:

            self.send_json(
                400,
                {
                    "error": "timeframe is required"
                },
            )

            return

        if not isinstance(
            candles,
            list,
        ):

            self.send_json(
                400,
                {
                    "error": "candles must be a list"
                },
            )

            return

        if len(candles) > 200:

            candles = candles[:200]

        clean_candles = []

        for candle in candles:

            if not isinstance(
                candle,
                dict,
            ):
                continue

            clean_candles.append(
                {
                    "time": candle.get(
                        "time",
                        "",
                    ),
                    "open": candle.get(
                        "open",
                        None,
                    ),
                    "high": candle.get(
                        "high",
                        None,
                    ),
                    "low": candle.get(
                        "low",
                        None,
                    ),
                    "close": candle.get(
                        "close",
                        None,
                    ),
                    "volume": candle.get(
                        "volume",
                        0,
                    ),
                }
            )

        stored_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "price": data.get(
                "price",
                None,
            ),
            "digits": data.get(
                "digits",
                None,
            ),
            "server_time": data.get(
                "server_time",
                "",
            ),
            "received_at": datetime.now(
                ZoneInfo("Africa/Algiers")
            ).isoformat(),
            "candles": clean_candles,
        }

        key = (
            f"{symbol}:{timeframe}"
        )

        with mt4_lock:

            mt4_data[key] = stored_data

        logger.info(
            "MT4 data received: %s | %s | candles=%s",
            symbol,
            timeframe,
            len(clean_candles),
        )

        self.send_json(
            200,
            {
                "status": "ok",
                "symbol": symbol,
                "timeframe": timeframe,
                "candles": len(clean_candles),
            },
        )

    def log_message(
        self,
        format,
        *args,
    ):
        return


def start_web_server():

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            port,
        ),
        HealthHandler,
    )

    logger.info(
        "HTTP server started on port %s",
        port,
    )

    server.serve_forever()


# ============================================================
# OWNER
# ============================================================

def is_owner(
    update: Update,
) -> bool:

    return (
        update.effective_user is not None
        and update.effective_user.id
        == OWNER_ID
    )


# ============================================================
# START COMMAND
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "📸 أرسل صورة الشارت للتحليل بالصورة.\n\n"
        "📡 أو استخدم MT4 لإرسال بيانات الشموع.\n"
        "مثال:\n"
        "/analyze EURUSD M1\n\n"
        "الأوامر:\n"
        "/stats - الإحصائيات\n"
        "/win - تسجيل WIN\n"
        "/loss - تسجيل LOSS\n"
        "/reset - تصفير الإحصائيات\n"
        "/mt4status - حالة بيانات MT4\n\n"
        "تحليل الصورة يعتمد على:\n"
        "EMA 9 / EMA 21\n"
        "RSI 14\n"
        "Williams %R 14\n"
        "ADX 14 + DI 14\n"
        "Keltner EMA 20 / ATR 10 / Multiplier 5"
    )


# ============================================================
# STATS
# ============================================================

async def stats_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    total = wins + losses

    win_rate = (
        wins / total * 100
        if total
        else 0
    )

    await update.message.reply_text(
        "📊 ZinoProSignalAI Stats\n\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📌 Total: {total}\n"
        f"🎯 Win Rate: {win_rate:.1f}%"
    )


# ============================================================
# WIN
# ============================================================

async def win_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    global wins

    if not is_owner(update):
        return

    wins += 1

    await update.message.reply_text(
        "🟢 WIN مسجلة\n\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}"
    )


# ============================================================
# LOSS
# ============================================================

async def loss_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    global losses

    if not is_owner(update):
        return

    losses += 1

    await update.message.reply_text(
        "🔴 LOSS مسجلة\n\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}"
    )


# ============================================================
# RESET
# ============================================================

async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    global wins
    global losses

    if not is_owner(update):
        return

    wins = 0
    losses = 0

    await update.message.reply_text(
        "♻️ تم تصفير الإحصائيات."
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
                "📡 لا توجد بيانات MT4 حتى الآن."
            )

            return

        items = list(
            mt4_data.values()
        )

    items.sort(
        key=lambda item: item.get(
            "received_at",
            "",
        ),
        reverse=True,
    )

    lines = [
        "📡 MT4 Data Status",
        "",
    ]

    for item in items[:10]:

        symbol = item.get(
            "symbol",
            "UNKNOWN",
        )

        timeframe = item.get(
            "timeframe",
            "UNKNOWN",
        )

        candles = len(
            item.get(
                "candles",
                [],
            )
        )

        received = item.get(
            "received_at",
            "",
        )

        lines.append(
            f"• {symbol} {timeframe}"
        )

        lines.append(
            f"  Candles: {candles}"
        )

        lines.append(
            f"  Received: {received}"
        )

    await update.message.reply_text(
        "\n".join(lines)
    )


# ============================================================
# ANALYSIS PROMPT FOR MT4
# ============================================================

MT4_ANALYSIS_PROMPT = r"""
أنت ZinoProSignalAI، محلل فني للبيانات الرقمية القادمة من MT4.

أنت لا ترى صورة.

أنت تعتمد فقط على بيانات OHLC والوقت والحجم والسعر الحالي التي أرسلها MT4.

ممنوع اختلاق أي قيمة غير موجودة.

==================================================
DATA
==================================================

البيانات تحتوي على شموع مرتبة من الأقدم إلى الأحدث.

كل شمعة تحتوي:

time
open
high
low
close
volume

==================================================
IMPORTANT
==================================================

استخدم آخر الشموع في التحليل.

لا تستخدم الإنترنت.

لا تفترض سعرًا غير موجود.

لا تخترع RSI أو ADX أو Keltner أو EMA.

إذا لم يتم إرسال قيم المؤشرات، يجب حسابها رياضيًا من بيانات OHLC إذا كانت البيانات كافية.

إذا لم تكن البيانات كافية لحساب مؤشر معين:
اكتب "غير متاح".

==================================================
INDICATORS
==================================================

EMA 9
EMA 21

RSI 14

Williams %R 14

ADX 14 + DI 14

Keltner Channel:

EMA 20
ATR 10
Multiplier 5

==================================================
PRICE ACTION
==================================================

حلل:

- Higher High
- Higher Low
- Lower High
- Lower Low
- Trend
- Range
- Consolidation

الاتجاه الصاعد يحتاج بنية صاعدة واضحة.

الاتجاه الهابط يحتاج بنية هابطة واضحة.

==================================================
BREAKOUT
==================================================

ابحث عن:

- Breakout
- Candle close بعد الاختراق
- Retest
- Failed breakout

لا تعتبر مجرد لمس مستوى اختراقًا.

==================================================
LIQUIDITY
==================================================

ابحث عن:

- Sweep
- False breakout
- أخذ قمة
- أخذ قاع
- Rejection

==================================================
MOMENTUM
==================================================

حلل:

- حجم الشموع
- سرعة الحركة
- استمرار الحركة
- ضعف الحركة
- تسلسل الإغلاقات

==================================================
CANDLE
==================================================

حلل آخر الشموع:

- Bullish engulfing
- Bearish engulfing
- Pin bar
- Hammer
- Shooting star
- Rejection
- قوة الإغلاق

==================================================
EMA
==================================================

EMA 9 و EMA 21.

افحص:

- EMA 9 فوق EMA 21
- EMA 9 تحت EMA 21
- التقاطع
- الميل
- موقع السعر

التقاطع وحده لا يكفي.

==================================================
RSI
==================================================

RSI 14.

70 = تشبع شرائي.

30 = تشبع بيعي.

حلل:

- القيمة
- الاتجاه
- divergence إذا كان واضحًا

لا تجعل RSI وحده سبب الدخول.

==================================================
WILLIAMS
==================================================

Williams %R 14.

-20 = منطقة تشبع شرائي.

-80 = منطقة تشبع بيعي.

راقب:

- الدخول إلى التشبع
- الخروج من التشبع
- اتجاه المؤشر

==================================================
KELTNER
==================================================

EMA 20

ATR 10

Multiplier 5

احسب القناة إذا كانت البيانات كافية.

راقب:

- Upper
- Middle
- Lower
- Rejection
- Breakout
- Continuation

==================================================
ADX
==================================================

ADX 14.

DI Length 14.

راقب:

- قوة الاتجاه
- DI+
- DI-
- ADX

لا تستخدم ADX وحده.

==================================================
SCORING
==================================================

المجموع = 18.

Structure = 2

Breakout = 2

Liquidity = 1

Momentum = 2

Candle = 2

RSI = 1

Summary = 2

Oscillators = 2

Moving Averages = 2

TOTAL = 18

==================================================
DIRECTION
==================================================

اختر اتجاهًا واحدًا:

UP

أو

DOWN

ممنوع:

WAIT

NO SIGNAL

NEUTRAL

إذا كانت الأدلة ضعيفة:

اختر الاتجاه الذي لديه دعم أكبر.

لكن اخفض confidence.

==================================================
BEST ENTRY TIME
==================================================

لا تعطِ وقت دخول عشوائيًا.

إذا كان الفريم:

M1 = الشمعة التالية بعد دقيقة

M2 = الشمعة التالية بعد دقيقتين

M3 = الشمعة التالية بعد 3 دقائق

M5 = الشمعة التالية بعد 5 دقائق

يمكن زيادة delay إذا كان هناك سبب فني واضح.

يجب أن يكون:

entry_delay_minutes

رقمًا صحيحًا.

==================================================
ENTRY PRICE
==================================================

استخدم السعر الحالي أو مستوى الدخول المنطقي بناءً على البيانات.

لا تخترع سعرًا.

==================================================
CANCELLATION
==================================================

UP:

الإلغاء عادة تحت القاع البنيوي الأخير.

DOWN:

الإلغاء عادة فوق القمة البنيوية الأخيرة.

يجب أن يكون المستوى مبنيًا على بيانات فعلية.

==================================================
CONFIDENCE
==================================================

Confidence ليست مساوية للـscore تلقائيًا.

إذا كان:

Structure + Price Action + Momentum + Candle

متوافقة بقوة:

ارفع confidence.

إذا كانت المؤشرات متعارضة:

اخفض confidence.

==================================================
OUTPUT
==================================================

أخرج JSON فقط.

الشكل:

{
  "asset": "EURUSD",
  "timeframe": "M1",
  "direction": "UP",
  "confidence": 82,
  "up_score": 15,
  "down_score": 5,
  "entry_delay_minutes": 1,
  "entry_price": "1.17452",
  "cancellation_level": "1.17430",
  "cancellation_text": "إلغاء إذا أغلقت شمعة تحت 1.17430",
  "structure": "Higher High + Higher Low",
  "breakout": "Bullish breakout confirmed",
  "liquidity": "Bullish liquidity sweep",
  "momentum": "Positive",
  "candle": "Bullish continuation",
  "rsi": "61",
  "williams_r": "-34",
  "moving_averages": "EMA 9 above EMA 21",
  "keltner": "Price above middle band",
  "adx": "ADX rising with DI+ stronger",
  "reason": "بنية صاعدة مع زخم إيجابي وتأكيد من المتوسطات"
}

ممنوع إضافة نص خارج JSON.
"""


# ============================================================
# GET MT4 DATA
# ============================================================

def get_mt4_data(
    symbol,
    timeframe,
):

    key = (
        f"{symbol.upper()}:{timeframe.upper()}"
    )

    with mt4_lock:

        data = mt4_data.get(
            key
        )

        if data is None:
            return None

        return json.loads(
            json.dumps(
                data
            )
        )


# ============================================================
# CALCULATE ENTRY TIME
# ============================================================

def calculate_entry_time(
    delay,
):

    try:

        delay = int(delay)

    except Exception:

        delay = 1

    delay = max(
        1,
        min(120, delay),
    )

    now = datetime.now(
        ZoneInfo(
            "Africa/Algiers"
        )
    )

    return (
        now
        + timedelta(
            minutes=delay
        )
    )


# ============================================================
# SAFE SCORE
# ============================================================

def safe_score(
    value,
) -> int:

    try:

        number = int(
            value
        )

    except Exception:

        return 0

    return max(
        0,
        min(
            18,
            number,
        ),
    )


# ============================================================
# FORMAT MT4 SIGNAL
# ============================================================

def format_mt4_signal(
    data,
):

    asset = str(
        data.get(
            "asset",
            "غير واضح",
        )
    )

    timeframe = str(
        data.get(
            "timeframe",
            "غير واضح",
        )
    )

    direction = str(
        data.get(
            "direction",
            "UP",
        )
    ).upper()

    if direction not in (
        "UP",
        "DOWN",
    ):

        direction = "UP"

    try:

        confidence = float(
            data.get(
                "confidence",
                0,
            )
        )

        confidence = max(
            0,
            min(
                100,
                confidence,
            )
        )

        confidence_text = (
            f"{confidence:.0f}%"
        )

    except Exception:

        confidence_text = (
            "غير واضح"
        )

    up_score = safe_score(
        data.get(
            "up_score",
            0,
        )
    )

    down_score = safe_score(
        data.get(
            "down_score",
            0,
        )
    )

    try:

        delay = int(
            data.get(
                "entry_delay_minutes",
                1,
            )
        )

    except Exception:

        delay = 1

    delay = max(
        1,
        min(
            120,
            delay,
        )
    )

    entry_time = calculate_entry_time(
        delay
    )

    entry_price = str(
        data.get(
            "entry_price",
            "غير واضح",
        )
    )

    cancellation_level = str(
        data.get(
            "cancellation_level",
            "غير واضح",
        )
    )

    cancellation_text = str(
        data.get(
            "cancellation_text",
            "",
        )
    ).strip()

    if not cancellation_text:

        if (
            cancellation_level
            != "غير واضح"
        ):

            if direction == "DOWN":

                cancellation_text = (
                    "إلغاء إذا أغلقت شمعة فوق "
                    f"{cancellation_level}"
                )

            else:

                cancellation_text = (
                    "إلغاء إذا أغلقت شمعة تحت "
                    f"{cancellation_level}"
                )

        else:

            cancellation_text = (
                "إلغاء إذا أغلقت شمعة عكس الاتجاه"
            )

    if direction == "UP":

        direction_text = "🟢 UP"

    else:

        direction_text = "🔴 DOWN"

    structure = str(
        data.get(
            "structure",
            "غير متاح",
        )
    )

    breakout = str(
        data.get(
            "breakout",
            "غير متاح",
        )
    )

    liquidity = str(
        data.get(
            "liquidity",
            "غير متاح",
        )
    )

    momentum = str(
        data.get(
            "momentum",
            "غير متاح",
        )
    )

    candle = str(
        data.get(
            "candle",
            "غير متاح",
        )
    )

    rsi = str(
        data.get(
            "rsi",
            "غير متاح",
        )
    )

    williams = str(
        data.get(
            "williams_r",
            "غير متاح",
        )
    )

    ma = str(
        data.get(
            "moving_averages",
            "غير متاح",
        )
    )

    keltner = str(
        data.get(
            "keltner",
            "غير متاح",
        )
    )

    adx = str(
        data.get(
            "adx",
            "غير متاح",
        )
    )

    reason = str(
        data.get(
            "reason",
            "لا يوجد سبب متاح.",
        )
    )

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {asset} | {timeframe}\n"
        f"🎯 Confidence: {confidence_text}\n\n"
        f"📌 Decision: {direction_text}\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"

        "📊 ANALYSIS\n"
        f"🏗 Structure: {structure}\n"
        f"💥 Breakout: {breakout}\n"
        f"💧 Liquidity: {liquidity}\n"
        f"⚡ Momentum: {momentum}\n"
        f"🕯 Candle: {candle}\n"
        f"📈 RSI 14: {rsi}\n"
        f"📉 Williams %R: {williams}\n"
        f"📊 EMA 9/21: {ma}\n"
        f"〰️ Keltner: {keltner}\n"
        f"📐 ADX/DI: {adx}\n\n"

        f"⏳ Entry after: {delay} min\n"
        f"🕐 Entry Time: "
        f"{entry_time.strftime('%H:%M:%S')}\n"
        f"💰 Entry Price: {entry_price}\n"
        f"🚫 {cancellation_text}\n\n"

        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# CLEAN JSON
# ============================================================

def clean_json(
    text: str,
) -> str:

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip()
            == "```"
        ):

            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

    return text


# ============================================================
# GEMINI MT4 ANALYSIS
# ============================================================

async def analyze_mt4_data(
    market_data,
):

    payload = json.dumps(
        market_data,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )

    response = await gemini_generate_with_retry(
        model=GEMINI_MODEL,
        contents=[
            MT4_ANALYSIS_PROMPT,
            "\n\nMT4 MARKET DATA:\n",
            payload,
        ],
        config=types.GenerateContentConfig(
            temperature=0.10,
            response_mime_type="application/json",
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response"
        )

    raw = clean_json(
        response.text
    )

    result = json.loads(
        raw
    )

    if not isinstance(
        result,
        dict,
    ):

        raise RuntimeError(
            "Gemini response is not a JSON object"
        )

    return result


# ============================================================
# ANALYZE COMMAND
# ============================================================

async def analyze_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    args = context.args

    if len(args) < 2:

        await update.message.reply_text(
            "❌ استخدم الأمر هكذا:\n\n"
            "/analyze EURUSD M1\n\n"
            "مثال آخر:\n"
            "/analyze GBPUSD M3"
        )

        return

    symbol = args[0].strip().upper()

    timeframe = args[1].strip().upper()

    market_data = get_mt4_data(
        symbol,
        timeframe,
    )

    if market_data is None:

        await update.message.reply_text(
            "❌ لا توجد بيانات MT4 لهذا الزوج والفريم.\n\n"
            f"الزوج: {symbol}\n"
            f"الفريم: {timeframe}\n\n"
            "تأكد أن EA يعمل وأنه أرسل البيانات."
        )

        return

    candles = market_data.get(
        "candles",
        [],
    )

    if len(candles) < 20:

        await update.message.reply_text(
            "⏳ البيانات غير كافية للتحليل.\n\n"
            f"المتاح: {len(candles)} شمعة\n"
            "المطلوب حاليًا: 20 شمعة على الأقل."
        )

        return

    processing = await update.message.reply_text(
        "📡 استلام بيانات MT4...\n"
        "🔎 تحليل Structure + Price Action + "
        "EMA + RSI + Williams + ADX + Keltner..."
    )

    try:

        result = await analyze_mt4_data(
            market_data
        )

        signal = format_mt4_signal(
            result
        )

        await processing.edit_text(
            signal
        )

    except json.JSONDecodeError:

        logger.exception(
            "Invalid JSON from Gemini MT4"
        )

        await processing.edit_text(
            "❌ Gemini رجّع نتيجة غير قابلة للقراءة."
        )

    except Exception as error:

        logger.exception(
            "MT4 analysis error"
        )

        message = str(
            error
        )

        if len(message) > 350:

            message = message[:350]

        await processing.edit_text(
            "❌ حدث خطأ أثناء تحليل MT4.\n\n"
            f"{message}"
        )


# ============================================================
# ANALYSIS PROMPT FOR SCREENSHOT
# ============================================================

ANALYSIS_PROMPT = r"""
أنت محلل فني صارم لشارت تداول قصير الأجل.

حلل صورة الشارت المرفقة فقط.

ممنوع اختلاق أي معلومة غير ظاهرة بوضوح في الصورة.

==================================================
INDICATORS SETTINGS
==================================================

الشارت يفترض أن يحتوي على:

1. EMA 9
- Period = 9
- لون أخضر

2. EMA 21
- Period = 21
- لون أحمر

3. RSI
- Period = 14
- Overbought = 70
- Oversold = 30

4. Williams %R
- Period = 14
- Overbought = -20
- Oversold = -80

5. ADX
- ADX = 14
- DI Length = 14

6. Keltner Channel
- EMA = 20
- ATR = 10
- Multiplier = 5

لا تفترض أن أي مؤشر ظاهر لمجرد أن الإعدادات المفترضة موجودة.

إذا كان المؤشر غير واضح أو غير ظاهر:
اكتب "غير متاح".

لا تخترع أي رقم.

==================================================
PRIORITY
==================================================

ترتيب أهمية التحليل:

1. Price Action
2. Structure
3. Breakout / Retest
4. Liquidity
5. Momentum
6. Candle
7. EMA 9 / EMA 21
8. RSI
9. Williams %R
10. Keltner Channel
11. ADX / DI

المؤشرات ليست بديلًا عن حركة السعر.

لا تعطِ إشارة فقط لأن مؤشرًا واحدًا صاعد أو هابط.

==================================================
STRUCTURE
==================================================

حدد:

- Higher High
- Higher Low
- Lower High
- Lower Low
- Trend
- Range / Consolidation

==================================================
BREAKOUT
==================================================

ابحث عن:

- Breakout واضح
- Candle close
- Retest
- Breakout failure

==================================================
LIQUIDITY
==================================================

ابحث عن:

- Liquidity sweep
- False breakout
- Rejection
- أخذ قمة أو قاع قريب ثم انعكاس

==================================================
MOMENTUM
==================================================

افحص:

- قوة الحركة
- سرعة الحركة
- حجم الشموع
- استمرار أو ضعف الزخم
- تسلسل الشموع

==================================================
CANDLE
==================================================

افحص:

- Bullish engulfing
- Bearish engulfing
- Pin bar
- Hammer
- Shooting star
- Rejection candle
- قوة الإغلاق

==================================================
EMA 9 + EMA 21
==================================================

إذا ظهرا بوضوح:

EMA 9 أخضر.
EMA 21 أحمر.

افحص:

- EMA 9 فوق EMA 21
- EMA 9 تحت EMA 21
- تقاطع
- ميل الخطوط
- موقع السعر

لا تستخدم التقاطع وحده.

==================================================
RSI 14
==================================================

70 = تشبع شرائي.

30 = تشبع بيعي.

افحص اتجاه RSI وdivergence إذا كان واضحًا.

==================================================
WILLIAMS %R 14
==================================================

-20 = تشبع شرائي.

-80 = تشبع بيعي.

راقب الخروج من التشبع.

==================================================
KELTNER
==================================================

EMA 20
ATR 10
Multiplier 5

افحص:

- Upper
- Middle
- Lower
- rejection
- breakout
- continuation

==================================================
ADX + DI
==================================================

ADX = 14
DI Length = 14

افحص:

- قوة الاتجاه
- DI+ مقابل DI-
- اتجاه ADX

==================================================
SCORING
==================================================

المجموع = 18.

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

TOTAL = 18

==================================================
DIRECTION
==================================================

اختر:

UP

أو

DOWN

ممنوع:

WAIT
NO SIGNAL
NEUTRAL

إذا كانت الأدلة ضعيفة اختر الاتجاه الذي لديه أدلة أكثر وخفض confidence.

==================================================
ENTRY
==================================================

استخرج:

Entry Price
Timeframe

إذا كان غير واضح:
"غير واضح"

==================================================
ENTRY DELAY
==================================================

1M = 1 دقيقة

2M = 2 دقائق

3M = 3 دقائق

الفريم الآخر = عدد دقائق الفريم إذا كان واضحًا.

أرسل فقط:

entry_delay_minutes

==================================================
CANCELLATION
==================================================

UP:

أسفل البنية أو القاع الأخير إذا كان واضحًا.

DOWN:

فوق البنية أو القمة الأخيرة إذا كان واضحًا.

==================================================
OUTPUT
==================================================

أخرج JSON فقط:

{
  "asset": "EUR/USD",
  "timeframe": "2M",
  "direction": "DOWN",
  "confidence": 76,
  "up_score": 5,
  "down_score": 13,
  "entry_delay_minutes": 2,
  "entry_price": "1.13460",
  "cancellation_level": "1.13490",
  "cancellation_text": "إلغاء إذا أغلقت شمعة فوق 1.13490",
  "structure": "Lower High + Lower Low",
  "breakout": "Bearish breakout confirmed",
  "liquidity": "Bearish liquidity sweep",
  "momentum": "Negative",
  "candle": "Bearish rejection",
  "rsi": "Below 30",
  "williams_r": "خرج من منطقة التشبع الشرائي",
  "moving_averages": "EMA 9 below EMA 21",
  "keltner": "Bearish rejection",
  "adx": "DI- stronger than DI+",
  "reason": "بنية هابطة مع زخم سلبي وتأكيد من EMA"
}

ممنوع إضافة نص خارج JSON.
"""


# ============================================================
# SCREENSHOT ANALYSIS
# ============================================================

async def analyze_chart(
    image_bytes: bytes,
):

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )

    response = await gemini_generate_with_retry(
        model=GEMINI_MODEL,
        contents=[
            image_part,
            ANALYSIS_PROMPT,
        ],
        config=types.GenerateContentConfig(
            temperature=0.10,
            response_mime_type="application/json",
        ),
    )

    if not response.text:

        raise RuntimeError(
            "Gemini returned an empty response"
        )

    raw = clean_json(
        response.text
    )

    data = json.loads(
        raw
    )

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            "Gemini response is not a JSON object"
        )

    return data


# ============================================================
# SCREENSHOT SIGNAL FORMAT
# ============================================================

def format_signal(
    data: dict,
) -> str:

    asset = str(
        data.get(
            "asset",
            "غير واضح",
        )
    )

    timeframe = str(
        data.get(
            "timeframe",
            "غير واضح",
        )
    )

    direction = str(
        data.get(
            "direction",
            "UP",
        )
    ).upper()

    if direction not in (
        "UP",
        "DOWN",
    ):

        direction = "UP"

    try:

        confidence = float(
            data.get(
                "confidence",
                0,
            )
        )

        confidence = max(
            0,
            min(
                100,
                confidence,
            )
        )

        confidence_text = (
            f"{confidence:.0f}%"
        )

    except Exception:

        confidence_text = (
            "غير واضح"
        )

    up_score = safe_score(
        data.get(
            "up_score",
            0,
        )
    )

    down_score = safe_score(
        data.get(
            "down_score",
            0,
        )
    )

    try:

        delay = int(
            data.get(
                "entry_delay_minutes",
                1,
            )
        )

    except Exception:

        delay = 1

    delay = max(
        1,
        min(
            60,
            delay,
        )
    )

    entry_price = str(
        data.get(
            "entry_price",
            "غير واضح",
        )
    )

    cancellation_level = str(
        data.get(
            "cancellation_level",
            "غير واضح",
        )
    )

    cancellation_text = str(
        data.get(
            "cancellation_text",
            "",
        )
    ).strip()

    if not cancellation_text:

        if (
            cancellation_level
            != "غير واضح"
        ):

            if direction == "DOWN":

                cancellation_text = (
                    "إلغاء إذا أغلقت شمعة فوق "
                    f"{cancellation_level}"
                )

            else:

                cancellation_text = (
                    "إلغاء إذا أغلقت شمعة تحت "
                    f"{cancellation_level}"
                )

        else:

            cancellation_text = (
                "إلغاء إذا أغلقت شمعة عكس الاتجاه"
            )

    reason = str(
        data.get(
            "reason",
            "تحليل مبني على الأدلة الظاهرة في الشارت.",
        )
    )

    now = datetime.now(
        ZoneInfo(
            "Africa/Algiers"
        )
    )

    entry_time = (
        now
        + timedelta(
            minutes=delay
        )
    )

    if direction == "DOWN":

        direction_text = "🔴 DOWN"

    else:

        direction_text = "🟢 UP"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {asset} | {timeframe}\n"
        f"🎯 Confidence: {confidence_text}\n\n"
        f"📌 Decision: {direction_text}\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏳ Entry after: {delay} min\n"
        f"🕐 Entry Time: "
        f"{entry_time.strftime('%H:%M:%S')}\n"
        f"💰 Entry Price: {entry_price}\n"
        f"🚫 {cancellation_text}\n\n"
        f"📝 {reason}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not update.message:
        return

    if not update.message.photo:
        return

    processing = await update.message.reply_text(
        "🔎 تحليل الشارت...\n"
        "Price Action + Structure + "
        "EMA 9/21 + RSI 14 + "
        "Williams %R 14 + ADX/DI 14 + "
        "Keltner 20/10/5"
    )

    try:

        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        image_buffer = io.BytesIO()

        await telegram_file.download_to_memory(
            image_buffer
        )

        image_bytes = (
            image_buffer.getvalue()
        )

        if not image_bytes:

            raise RuntimeError(
                "الصورة فارغة"
            )

        result = await analyze_chart(
            image_bytes
        )

        signal = format_signal(
            result
        )

        await processing.edit_text(
            signal
        )

    except json.JSONDecodeError:

        logger.exception(
            "Invalid JSON from Gemini"
        )

        await processing.edit_text(
            "❌ Gemini رجّع نتيجة غير قابلة للقراءة.\n"
            "أعد إرسال الصورة."
        )

    except Exception as error:

        logger.exception(
            "Analysis error"
        )

        message = str(
            error
        )

        if len(message) > 350:

            message = message[:350]

        await processing.edit_text(
            "❌ حدث خطأ أثناء التحليل.\n\n"
            f"{message}"
        )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    if not update.message:
        return

    text = update.message.text.strip()

    parts = text.split()

    if len(parts) == 2:

        symbol = parts[0].upper()

        timeframe = parts[1].upper()

        valid_timeframes = {
            "M1",
            "M2",
            "M3",
            "M5",
            "M15",
            "M30",
            "H1",
            "H4",
            "D1",
        }

        if timeframe in valid_timeframes:

            market_data = get_mt4_data(
                symbol,
                timeframe,
            )

            if market_data is not None:

                await analyze_command(
                    update,
                    context,
                )

                return

    await update.message.reply_text(
        "📸 أرسل صورة الشارت مباشرة.\n\n"
        "أو لتحليل بيانات MT4:\n"
        "اكتب مثلًا:\n"
        "EURUSD M1"
    )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.error(
        "Telegram error: %s",
        context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting ZinoProSignalAI..."
    )

    threading.Thread(
        target=start_web_server,
        daemon=True,
    ).start()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
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
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram bot is running"
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
