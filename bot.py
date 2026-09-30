import os
import io
import json
import logging
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
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

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID_RAW:
    raise RuntimeError("OWNER_ID is missing")

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
# STATS
# ============================================================

wins = 0
losses = 0


# ============================================================
# RENDER HEALTH SERVER
# ============================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )
        self.end_headers()

        self.wfile.write(
            b"ZinoProSignalAI is running"
        )

    def do_HEAD(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8",
        )
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_web_server():

    port = int(
        os.getenv("PORT", "10000")
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    logger.info(
        "Health server started on port %s",
        port,
    )

    server.serve_forever()


# ============================================================
# OWNER
# ============================================================

def is_owner(update: Update) -> bool:

    return (
        update.effective_user is not None
        and update.effective_user.id == OWNER_ID
    )


# ============================================================
# COMMANDS
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    if not is_owner(update):
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "📸 أرسل صورة الشارت مباشرة.\n"
        "سيبدأ التحليل تلقائيًا.\n\n"
        "المؤشرات المعتمدة:\n"
        "🟢 EMA 9\n"
        "🔴 EMA 21\n"
        "🟣 RSI 14\n"
        "🟠 Williams %R 14\n"
        "🔵 ADX 14 + DI 14\n"
        "🟡 Keltner EMA 20 / ATR 10 / Multiplier 5\n\n"
        "الأوامر:\n"
        "/stats - الإحصائيات\n"
        "/win - تسجيل WIN\n"
        "/loss - تسجيل LOSS\n"
        "/reset - تصفير الإحصائيات"
    )


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
# ANALYSIS PROMPT
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
1. STRUCTURE
==================================================

حدد إن كان واضحًا:

- Higher High
- Higher Low
- Lower High
- Lower Low
- Trend
- Range / Consolidation

الاتجاه الصاعد يحتاج بنية Higher High / Higher Low أو دليل صاعد واضح.

الاتجاه الهابط يحتاج بنية Lower High / Lower Low أو دليل هابط واضح.

==================================================
2. BREAKOUT
==================================================

ابحث عن:

- Breakout واضح
- Candle close بعد الاختراق
- Retest
- Breakout failure

لا تعتبر مجرد لمس المستوى اختراقًا.

==================================================
3. LIQUIDITY
==================================================

ابحث عن:

- Liquidity sweep
- False breakout
- Rejection
- أخذ قمة أو قاع قريب ثم انعكاس

استخدم فقط ما يظهر في الصورة.

==================================================
4. MOMENTUM
==================================================

افحص:

- قوة آخر الحركة
- سرعة الحركة
- حجم الشموع
- استمرار أو ضعف الزخم
- تسلسل الشموع

==================================================
5. CANDLE
==================================================

افحص:

- Bullish engulfing
- Bearish engulfing
- Pin bar
- Hammer
- Shooting star
- Rejection candle
- قوة الإغلاق

لا تعتبر شكلًا صغيرًا جدًا Pattern مؤكدًا.

==================================================
6. EMA 9 + EMA 21
==================================================

إذا ظهرا بوضوح:

EMA 9 أخضر.
EMA 21 أحمر.

افحص:

- EMA 9 فوق EMA 21
- EMA 9 تحت EMA 21
- تقاطع واضح
- ميل الخطوط
- موقع السعر بالنسبة لهما

لا تستخدم التقاطع وحده.

==================================================
7. RSI 14
==================================================

إذا كان RSI ظاهرًا بوضوح:

- فوق 70 = منطقة تشبع شرائي
- تحت 30 = منطقة تشبع بيعي
- افحص اتجاه RSI
- افحص divergence فقط إذا كان واضحًا جدًا

الوصول إلى 70 أو 30 وحده ليس دخولًا.

==================================================
8. WILLIAMS %R 14
==================================================

إذا كان ظاهرًا:

- فوق -20 = تشبع شرائي
- تحت -80 = تشبع بيعي
- راقب الخروج من منطقة التشبع
- راقب اتجاه المؤشر

مجرد لمس -20 أو -80 لا يكفي.

Williams %R عامل تأكيد فقط.

==================================================
9. KELTNER CHANNEL
==================================================

الإعداد:

EMA 20
ATR 10
Multiplier 5

إذا كان ظاهرًا:

افحص:

- السعر بالنسبة للحد العلوي
- السعر بالنسبة للحد الأوسط
- السعر بالنسبة للحد السفلي
- rejection
- breakout
- استمرار الاتجاه

لا تعتمد على Keltner وحده.

==================================================
10. ADX + DI
==================================================

الإعداد:

ADX = 14
DI Length = 14

إذا ظهر بوضوح:

افحص:

- قوة الاتجاه
- DI+ مقابل DI-
- اتجاه ADX
- هل الحركة اتجاهية أم ضعيفة

لا تخترع قيمة ADX.

لا تستخدم ADX وحده.

==================================================
SCORING SYSTEM
==================================================

المجموع = 18 نقطة.

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
SCORING RULES
==================================================

لكل عنصر:

Structure:
0 إلى 2

Breakout:
0 إلى 2

Liquidity:
0 إلى 1

Momentum:
0 إلى 2

Candle:
0 إلى 2

RSI:
0 إلى 1

Summary:
0 إلى 2

Oscillators:
0 إلى 2

Moving Averages:
0 إلى 2

كل نقطة يجب أن تكون مدعومة بدليل من الصورة.

لا تمنح النقاط لمؤشر غير ظاهر.

==================================================
OSCILLATORS
==================================================

Oscillators = 2 نقاط.

استخدم داخلها:

- RSI
- Williams %R
- ADX/DI عند الحاجة كتأكيد للاتجاه

لا تعطي نقطتين تلقائيًا.

إذا كانت المؤشرات متعارضة، خفّض نقاط Oscillators.

==================================================
MOVING AVERAGES
==================================================

Moving Averages = 2 نقاط.

استخدم:

EMA 9
EMA 21

إذا كان EMA 9 فوق EMA 21 مع ميل صاعد وتأكيد سعري:
يميل إلى UP.

إذا كان EMA 9 تحت EMA 21 مع ميل هابط وتأكيد سعري:
يميل إلى DOWN.

التقاطع وحده لا يكفي.

==================================================
SUMMARY
==================================================

Summary = 2 نقاط.

لخّص توافق:

Structure
Momentum
Candle
Breakout
Indicators

إذا كان هناك توافق قوي وواضح:
يمكن إعطاء 2.

إذا كان التوافق جزئيًا:
1.

إذا لم يوجد دعم واضح:
0.

==================================================
DIRECTION
==================================================

اختر اتجاهًا واحدًا فقط:

UP
أو
DOWN

ممنوع:

WAIT
NO SIGNAL
NEUTRAL

إذا كانت الأدلة ضعيفة أو متعارضة:

اختر الاتجاه الذي لديه أدلة أكثر.

لكن اخفض confidence.

لا ترفع confidence فقط بسبب ارتفاع score.

==================================================
CONFIDENCE
==================================================

Confidence تقدير لقوة الأدلة الظاهرة في الصورة فقط.

لا تجعل confidence يساوي score بشكل آلي.

لا تعطِ 90% أو أكثر إلا إذا كانت الصورة تحتوي على توافق واضح جدًا بين:

Structure
Price Action
Momentum
Candle
EMA
والتأكيدات الأخرى.

إذا كانت المؤشرات متعارضة:
اخفض confidence.

==================================================
ENTRY
==================================================

استخرج:

- Entry Price
- Timeframe

من الصورة إذا كانا واضحين.

إذا كان السعر غير واضح:
اكتب "غير واضح".

لا تخترع سعرًا.

==================================================
ENTRY DELAY
==================================================

إذا كان الفريم:

1M:
entry_delay_minutes = 1

2M:
entry_delay_minutes = 2

3M:
entry_delay_minutes = 3

إذا كان فريم آخر:
استخدم عدد دقائق الفريم إذا كان واضحًا.

لا تغير الفريم الموجود في الصورة.

==================================================
ENTRY TIME
==================================================

التوقيت النهائي سيتم حسابه بواسطة البرنامج بتوقيت الجزائر:

Africa/Algiers

لا تحسب Entry Time بنفسك.

أرسل فقط:

entry_delay_minutes

==================================================
CANCELLATION
==================================================

Cancellation Level يجب أن يكون منطقيًا مع الاتجاه والبنية الأخيرة.

في UP:
عادة يكون مستوى الإلغاء أسفل البنية أو القاع الأخير إذا كان واضحًا.

في DOWN:
عادة يكون مستوى الإلغاء فوق البنية أو القمة الأخيرة إذا كان واضحًا.

لا تضع مستوى عشوائي.

إذا لم يمكن تحديده:
اكتب "غير واضح".

==================================================
IMPORTANT
==================================================

لا تستخدم بيانات السوق الحية.

لا تستخدم الإنترنت.

لا تدّعي أنك ترى شموعًا غير موجودة.

لا تخترع:

RSI
Williams %R
ADX
DI
Keltner
EMA
السعر
الفريم

إذا لم يظهر المؤشر:
"غير متاح"

==================================================
OUTPUT
==================================================

أخرج JSON فقط.

يجب أن يكون الشكل:

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
  "keltner": "Bearish rejection from upper area",
  "adx": "ADX 14 with DI- stronger than DI+",
  "reason": "بنية هابطة مع زخم سلبي ورفض سعري وتأكيد من EMA وWilliams %R"
}

ممنوع إضافة نص خارج JSON.
"""


# ============================================================
# JSON CLEANER
# ============================================================

def clean_json(text: str) -> str:

    text = text.strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip() == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# GEMINI ANALYSIS
# ============================================================

async def analyze_chart(
    image_bytes: bytes,
):

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/jpeg",
    )

    response = client.models.generate_content(
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

    data = json.loads(raw)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Gemini response is not a JSON object"
        )

    return data


# ============================================================
# SAFE SCORE
# ============================================================

def safe_score(
    value,
) -> int:

    try:
        number = int(value)
    except Exception:
        return 0

    return max(
        0,
        min(18, number),
    )


# ============================================================
# SIGNAL FORMAT
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

    confidence = data.get(
        "confidence",
        0,
    )

    try:
        confidence = float(
            confidence
        )

        confidence = max(
            0,
            min(100, confidence),
        )

        confidence_text = (
            f"{confidence:.0f}%"
        )

    except Exception:
        confidence_text = "غير واضح"

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

    delay = data.get(
        "entry_delay_minutes",
        1,
    )

    try:
        delay = int(delay)

    except Exception:
        delay = 1

    delay = max(
        1,
        min(60, delay),
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
        ZoneInfo("Africa/Algiers")
    )

    entry_time = (
        now
        + timedelta(minutes=delay)
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
        f"🕐 Entry Time: {entry_time.strftime('%H:%M:%S')}\n"
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

        message = str(error)

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

    await update.message.reply_text(
        "📸 أرسل صورة الشارت مباشرة."
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
        MessageHandler(
            filters.PHOTO,
            photo_handler,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
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
