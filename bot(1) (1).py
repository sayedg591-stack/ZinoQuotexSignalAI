import os
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
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
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"ZinoProSignalAI is running")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()

    def log_message(self, format, *args):
        return


def start_web_server():
    port = int(os.getenv("PORT", "10000"))

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        HealthHandler,
    )

    logger.info("Health server started on port %s", port)

    server.serve_forever()


# ============================================================
# OWNER CHECK
# ============================================================

def is_owner(update: Update) -> bool:
    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


# ============================================================
# START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    if not is_owner(update):
        await update.message.reply_text(
            "❌ هذا البوت خاص."
        )
        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "أرسل صورة شارت Quotex مباشرة.\n"
        "سيتم تحليل الصورة وإرسال الإشارة تلقائيًا.\n\n"
        "📊 الأوامر:\n"
        "/stats - الإحصائيات\n"
        "/reset - تصفير الإحصائيات"
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

    if total > 0:
        win_rate = (wins / total) * 100
    else:
        win_rate = 0

    await update.message.reply_text(
        "📊 إحصائيات ZinoProSignalAI\n\n"
        f"🟢 Wins: {wins}\n"
        f"🔴 Losses: {losses}\n"
        f"📈 Total: {total}\n"
        f"🎯 Win Rate: {win_rate:.1f}%"
    )


# ============================================================
# RESET
# ============================================================

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
# WIN / LOSS
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
        f"🟢 WIN\n\n"
        f"Total Wins: {wins}\n"
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
        f"🔴 LOSS\n\n"
        f"Wins: {wins}\n"
        f"Total Losses: {losses}"
    )


# ============================================================
# GEMINI PROMPT
# ============================================================

ANALYSIS_PROMPT = r"""
أنت محلل فني لشارتات التداول قصير الأجل.

حلل صورة الشارت المرفقة فقط.

مهم جدًا:
- لا تخترع بيانات غير ظاهرة في الصورة.
- اقرأ اسم الأصل إذا كان ظاهرًا.
- اقرأ الفريم إذا كان ظاهرًا.
- ركز على حركة السعر والشموع والبنية السعرية.
- افحص الاتجاه.
- افحص Higher High / Higher Low.
- افحص Lower High / Lower Low.
- افحص الاختراقات وإعادة الاختبار.
- افحص مناطق الرفض.
- افحص الزخم.
- افحص شكل آخر الشموع.
- افحص RSI إذا كان ظاهرًا.
- افحص Moving Averages إذا كانت ظاهرة.
- افحص Keltner Channel إذا كان ظاهرًا.
- افحص ADX إذا كان ظاهرًا.
- لا تستخدم مؤشرًا غير ظاهر في الصورة وكأنه موجود.
- أعطِ اتجاهًا واحدًا فقط: UP أو DOWN.
- لا تعطِ WAIT.
- لا تعطِ NO SIGNAL.
- لا تعطِ الاتجاهين معًا.

نظام التقييم المطلوب من 18:

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

احسب UP score و DOWN score بناءً على الأدلة الظاهرة.

اختيار الاتجاه:
- إذا كانت أدلة الصعود أقوى اختر UP.
- إذا كانت أدلة الهبوط أقوى اختر DOWN.
- إذا كانت الصورة ضعيفة، اختر الاتجاه الذي تدعمه الأدلة الأقوى، لكن خفّض confidence.
- لا تجعل confidence مرتفعًا بدون أدلة واضحة.

التوقيت:
- الوقت المطلوب بتوقيت الجزائر Africa/Algiers.
- لا تجعل وقت الدخول مساويًا بالضرورة لوقت رفع الصورة.
- اقترح دخولًا على الشمعة التالية أو بعد تأخير مناسب حسب الفريم الظاهر.
- إذا كان الفريم 1M فالتأخير المعتاد حوالي دقيقة.
- إذا كان 2M فالتأخير المعتاد حوالي دقيقتين.
- إذا كان 3M فالتأخير المعتاد حوالي ثلاث دقائق.
- لا تغيّر الفريم الموجود في الصورة.

السعر:
- استخرج Entry Price من السعر الظاهر في الصورة.
- اقترح Cancellation Level منطقيًا من البنية السعرية.
- لا تضع Cancellation Level في نفس سطر Entry Price.

أخرج JSON فقط بهذا الشكل:

{
  "asset": "EUR/USD",
  "timeframe": "1M",
  "direction": "UP",
  "confidence": 84,
  "up_score": 15,
  "down_score": 3,
  "entry_delay_minutes": 1,
  "entry_price": "1.13460",
  "cancellation_level": "1.13435",
  "cancellation_text": "إلغاء إذا أغلقت شمعة تحت 1.13435",
  "reason": "سبب مختصر جدًا مبني على الشارت"
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

        if len(lines) >= 3:
            lines = lines[1:]

            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]

            text = "\n".join(lines).strip()

    return text


# ============================================================
# IMAGE ANALYSIS
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
            temperature=0.15,
            response_mime_type="application/json",
        ),
    )

    text = response.text

    if not text:
        raise RuntimeError("Gemini returned an empty response")

    text = clean_json(text)

    data = json.loads(text)

    return data


# ============================================================
# FORMAT SIGNAL
# ============================================================

def format_signal(data: dict) -> str:

    asset = str(data.get("asset", "Unknown"))
    timeframe = str(data.get("timeframe", "Unknown"))
    direction = str(data.get("direction", "UP")).upper()

    confidence = data.get("confidence", 0)
    up_score = data.get("up_score", 0)
    down_score = data.get("down_score", 0)

    delay = data.get("entry_delay_minutes", 1)

    entry_price = str(
        data.get("entry_price", "غير واضح")
    )

    cancellation_level = str(
        data.get("cancellation_level", "غير واضح")
    )

    cancellation_text = str(
        data.get(
            "cancellation_text",
            f"إلغاء إذا أغلقت شمعة عند المستوى {cancellation_level}",
        )
    )

    reason = str(
        data.get(
            "reason",
            "تم اعتماد الاتجاه بناءً على البنية والزخم والشموع الظاهرة.",
        )
    )

    now = datetime.now(
        ZoneInfo("Africa/Algiers")
    )

    entry_time = now

    try:
        delay_minutes = int(delay)
    except Exception:
        delay_minutes = 1

    from datetime import timedelta

    entry_time = now + timedelta(
        minutes=max(1, delay_minutes)
    )

    direction_text = "🟢 UP" if direction == "UP" else "🔴 DOWN"

    return (
        "🎓 ZinoProSignalAI\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 الأصل: {asset}\n"
        f"⏱ الفريم: {timeframe}\n"
        f"🎯 الثقة: {confidence}%\n\n"
        f"📌 القرار: {direction_text}\n"
        f"🟢 UP Score: {up_score}/18\n"
        f"🔴 DOWN Score: {down_score}/18\n\n"
        f"⏳ الدخول بعد: {max(1, delay_minutes)} دقيقة\n"
        f"🕐 وقت الدخول: {entry_time.strftime('%H:%M:%S')}\n\n"
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
        await update.message.reply_text(
            "❌ هذا البوت خاص."
        )
        return

    message = update.message

    if not message or not message.photo:
        return

    processing_message = await message.reply_text(
        "🔎 جاري تحليل الشارت..."
    )

    try:

        photo = message.photo[-1]

        file = await context.bot.get_file(
            photo.file_id
        )

        image_buffer = io.BytesIO()

        await file.download_to_memory(
            image_buffer
        )

        image_bytes = image_buffer.getvalue()

        if not image_bytes:
            raise RuntimeError(
                "Could not download image"
            )

        result = await analyze_chart(
            image_bytes
        )

        signal = format_signal(result)

        await processing_message.edit_text(
            signal
        )

    except json.JSONDecodeError:

        logger.exception(
            "Gemini returned invalid JSON"
        )

        await processing_message.edit_text(
            "❌ لم أستطع قراءة نتيجة التحليل.\n"
            "أعد إرسال صورة الشارت."
        )

    except Exception as e:

        logger.exception(
            "Chart analysis failed"
        )

        error_text = str(e)

        if len(error_text) > 300:
            error_text = error_text[:300]

        await processing_message.edit_text(
            "❌ حدث خطأ أثناء تحليل الشارت.\n\n"
            f"{error_text}"
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
        "📸 أرسل صورة الشارت وسأبدأ التحليل مباشرة."
    )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Telegram error",
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting ZinoProSignalAI..."
    )

    web_thread = threading.Thread(
        target=start_web_server,
        daemon=True,
    )

    web_thread.start()

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
            "reset",
            reset_command,
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
        "Telegram bot is starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
