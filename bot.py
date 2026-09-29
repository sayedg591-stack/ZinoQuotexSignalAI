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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

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

client = genai.Client(api_key=GEMINI_API_KEY)


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
    port = int(os.getenv("PORT", "10000"))

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

    if total:
        win_rate = wins / total * 100
    else:
        win_rate = 0

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
        f"🟢 WIN مسجلة\n\n"
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
        f"🔴 LOSS مسجلة\n\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}"
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    global wins, losses

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

أولًا:
اقرأ:
- اسم الأصل إن كان ظاهرًا.
- الفريم إن كان ظاهرًا.
- السعر الحالي إن كان ظاهرًا.
- آخر الشموع.
- البنية السعرية.

ركز على Price Action قبل المؤشرات.

حلل:

1. STRUCTURE
- Higher High
- Higher Low
- Lower High
- Lower Low
- الاتجاه أو التذبذب.

2. BREAKOUT
- هل يوجد اختراق واضح؟
- هل حصل إغلاق بعد الاختراق؟
- هل يوجد Retest؟
- لا تعتبر مجرد لمس المستوى اختراقًا.

3. LIQUIDITY
- مناطق سحب السيولة الظاهرة.
- False Breakout إن كان واضحًا.
- رفض سعري واضح.

4. MOMENTUM
- قوة الحركة الأخيرة.
- تسارع أو ضعف الحركة.
- مقارنة الشموع الأخيرة.

5. CANDLE
ابحث عن:
- Engulfing
- Pin Bar
- Hammer
- Shooting Star
- Rejection
- قوة الإغلاق.

6. RSI
إذا كان RSI ظاهرًا فقط:
- افحص 70 / 30.
- افحص الاتجاه.
- لا تستخدم RSI وحده.

7. WILLIAMS %R
إذا كان Williams %R ظاهرًا فقط:
- افحص منطقة -20.
- افحص منطقة -80.
- راقب الخروج من منطقة التشبع.
- لا تعتبر مجرد وصول المؤشر إلى -20 أو -80 إشارة دخول.
- استخدمه كتأكيد لحركة السعر.

8. MOVING AVERAGES
إذا كانت ظاهرة:
- اتجاه السعر بالنسبة لها.
- التقاطع إن كان واضحًا.
- لا تخترع قيمًا غير موجودة.

9. KELTNER CHANNEL
إذا كان ظاهرًا:
- لمس/اختراق الحد.
- rejection.
- استمرار الاتجاه.
- لا تعتمد عليه منفردًا.

10. ADX
إذا كان ظاهرًا:
- قوة الاتجاه.
- DI+ / DI- إذا كانت واضحة.
- لا تستخدم رقمًا غير ظاهر.

==================================================
SCORING
==================================================

Structure = 2
Breakout = 2
Liquidity = 1
Momentum = 2
Candle = 2
RSI = 1
Summary = 2
Oscillators = 2
Moving Averages = 2

المجموع = 18.

Williams %R يستخدم كعامل تأكيد داخل Oscillators.
لا تضف نقاطًا وهمية بسبب عدم ظهور المؤشر.

احسب:

UP score
DOWN score

كل نقطة يجب أن تكون مدعومة بدليل ظاهر.

==================================================
الاتجاه
==================================================

اختر:
UP
أو
DOWN

لا تعط:
WAIT
NO SIGNAL
NEUTRAL

لكن إذا كانت الصورة ضعيفة أو غير واضحة:
- اختر الاتجاه الذي لديه أدلة أكثر.
- اخفض confidence.
- لا تدّعي أن الإشارة قوية.

لا تجعل confidence مرتفعًا فقط لأن UP score أو DOWN score مرتفع.

==================================================
التوقيت
==================================================

استخدم توقيت الجزائر:
Africa/Algiers

إذا كان الفريم:
1M → دخول بعد حوالي دقيقة.
2M → دخول بعد حوالي دقيقتين.
3M → دخول بعد حوالي 3 دقائق.

لا تغير الفريم الظاهر في الصورة.

لا تجعل وقت الدخول مساويًا لوقت رفع الصورة.

==================================================
ENTRY
==================================================

استخرج Entry Price من السعر الظاهر.

Cancellation Level يجب أن يكون مرتبطًا بالبنية السعرية الأخيرة.

لا تضع Cancellation Level فوق Entry Price في حالة DOWN إذا كان ذلك غير منطقي.

ولا تضعه تحت Entry Price في حالة UP إذا كان ذلك غير منطقي.

==================================================
مهم جدًا
==================================================

لا تستخدم بيانات السوق الحية.
لا تدّعي أنك ترى شموعًا غير موجودة.
لا تخترع RSI.
لا تخترع Williams %R.
لا تخترع ADX.
لا تخترع Keltner.
لا تخترع Moving Average.

إذا كان المؤشر غير ظاهر:
اجعل قيمته "غير متاح".

==================================================
OUTPUT
==================================================

أخرج JSON فقط.

الشكل:

{
  "asset": "EUR/USD",
  "timeframe": "2M",
  "direction": "DOWN",
  "confidence": 76,
  "up_score": 5,
  "down_score": 14,
  "entry_delay_minutes": 2,
  "entry_price": "1.13460",
  "cancellation_level": "1.13490",
  "cancellation_text": "إلغاء إذا أغلقت شمعة فوق 1.13490",
  "structure": "Lower High + Lower Low",
  "breakout": "Bearish breakout confirmed",
  "liquidity": "Bearish rejection",
  "momentum": "Negative",
  "candle": "Bearish rejection",
  "rsi": "غير متاح",
  "williams_r": "خرج من منطقة التشبع الشرائي",
  "moving_averages": "Bearish",
  "keltner": "غير متاح",
  "adx": "غير متاح",
  "reason": "اتجاه هابط مع رفض سعري وزخم سلبي وتأكيد من Williams %R"
}

لا تضف أي نص خارج JSON.
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

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


# ============================================================
# GEMINI ANALYSIS
# ============================================================

async def analyze_chart(image_bytes: bytes):

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

    raw = clean_json(response.text)

    return json.loads(raw)


# ============================================================
# SIGNAL FORMAT
# ============================================================

def format_signal(data: dict) -> str:

    asset = str(
        data.get("asset", "غير واضح")
    )

    timeframe = str(
        data.get("timeframe", "غير واضح")
    )

    direction = str(
        data.get("direction", "UP")
    ).upper()

    confidence = data.get(
        "confidence",
        0,
    )

    up_score = data.get(
        "up_score",
        0,
    )

    down_score = data.get(
        "down_score",
        0,
    )

    delay = data.get(
        "entry_delay_minutes",
        1,
    )

    try:
        delay = max(
            1,
            int(delay),
        )
    except Exception:
        delay = 1

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
            f"إلغاء إذا أغلقت شمعة عند {cancellation_level}",
        )
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

    entry_time = now + timedelta(
        minutes=delay
    )

    if direction == "DOWN":
        direction_text = "🔴 DOWN"
    else:
        direction_text = "🟢 UP"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 {asset} | {timeframe}\n"
        f"🎯 Confidence: {confidence}%\n\n"
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
        "Price Action + Structure + Williams %R + المؤشرات الظاهرة"
    )

    try:

        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(
            photo.file_id
        )

        image_buffer = io.BytesIO()

        await telegram_file.download_to_memory(
            image_buffer
        )

        image_bytes = image_buffer.getvalue()

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
