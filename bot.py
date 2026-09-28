import os
import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime, timedelta, timezone
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


# =========================================================
# SETTINGS
# =========================================================

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OWNER_ID = os.getenv("OWNER_ID")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

PORT = int(os.getenv("PORT", "10000"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is missing")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

if not OWNER_ID:
    raise RuntimeError("OWNER_ID is missing")

OWNER_ID = int(OWNER_ID)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("ZinoProSignalAI")


# =========================================================
# GEMINI
# =========================================================

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# UTC-3
# =========================================================

SIGNAL_TZ = timezone(
    timedelta(hours=-3)
)


# =========================================================
# HEALTH SERVER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )

        self.end_headers()

        if self.path == "/health":
            self.wfile.write(
                b"ZinoProSignalAI is running"
            )
        else:
            self.wfile.write(
                b"ZinoProSignalAI"
            )

    def log_message(self, format, *args):
        return


def start_health_server():

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        HealthHandler
    )

    logger.info(
        "Health server running on port %s",
        PORT
    )

    server.serve_forever()


# =========================================================
# OWNER
# =========================================================

def is_owner(update: Update):

    if not update.effective_user:
        return False

    return update.effective_user.id == OWNER_ID


# =========================================================
# START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        await update.message.reply_text(
            "⛔ هذا البوت خاص."
        )

        return

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "📸 أرسل Screenshot للشارت.\n\n"
        "⚡ سيتم تحليل الشارت وإعطاؤك "
        "الاتجاه ووقت الدخول وسعر الدخول "
        "ومستوى الإلغاء."
    )


# =========================================================
# ANALYSIS PROMPT
# =========================================================

ANALYSIS_PROMPT = """
أنت محرك التحليل الفني الرئيسي لبوت ZinoProSignalAI.

حلل Screenshot حقيقية لشارت Quotex.

الهدف هو اختيار اتجاه واحد فقط:

UP
أو
DOWN

لا تعتمد على مؤشر واحد.

الأولوية:

1. Market Structure
2. Candle Close / Open
3. Highs / Lows
4. Breakout
5. Liquidity Sweep
6. Momentum
7. Candle Confirmation
8. RSI
9. Oscillators
10. Moving Averages

==================================================
IMPORTANT
==================================================

لا تجعل UP افتراضياً.

لا تجعل DOWN افتراضياً.

احسب الأدلة لصالح UP و DOWN بشكل منفصل.

إذا كان السوق صاعداً:
يجب أن تظهر الأدلة في UP.

إذا كان السوق هابطاً:
يجب أن تظهر الأدلة في DOWN.

إذا كان عامل غير ظاهر في الصورة:
لا تخترع بياناته.

==================================================
SCORING
==================================================

المجموع الأقصى = 18 نقطة.

Structure       = 2
Breakout        = 2
Liquidity       = 1
Momentum        = 2
Candle          = 2
RSI             = 1
Summary         = 2
Oscillators     = 3
Moving Averages = 3

TOTAL = 18

لكل عامل، احسب نقاط UP ونقاط DOWN بشكل مستقل.

==================================================
STRUCTURE
==================================================

راقب:

Higher High
Higher Low
Lower High
Lower Low
Break of Structure
Change of Character

Higher High + Higher Low
يدعم UP.

Lower High + Lower Low
يدعم DOWN.

==================================================
CANDLE
==================================================

ركز على آخر شمعة مغلقة.

افحص:

Open
Close
Body
Upper Wick
Lower Wick
Strong Close
Weak Close

راقب:

Bullish Engulfing
Bearish Engulfing
Hammer
Shooting Star
Pin Bar
Strong Close
Continuation

==================================================
BREAKOUT
==================================================

ميز بين:

True Breakout
Fake Breakout
Breakout Confirmation

لا تعتبر Wick وحده Breakout مؤكداً.

==================================================
LIQUIDITY
==================================================

راقب:

High Sweep
Low Sweep
Liquidity Sweep
Rejection

Sweep لقاع ثم إغلاق فوقه
قد يدعم UP.

Sweep لقمة ثم إغلاق تحتها
قد يدعم DOWN.

==================================================
MOMENTUM
==================================================

افحص:

قوة جسم الشموع
تتابع الشموع
سرعة الحركة
Wicks

واستخدم ADX إذا كان ظاهراً.

==================================================
RSI
==================================================

إذا كان RSI ظاهراً:

Above 50
Below 50
Overbought
Oversold
Divergence

لا تخترع قيمة RSI.

==================================================
OSCILLATORS
==================================================

حلل المؤشرات الظاهرة فقط.

مثل:

RSI
MACD
Stochastic

==================================================
MOVING AVERAGES
==================================================

إذا كانت ظاهرة:

السعر فوق أو تحت المتوسطات
اتجاه المتوسطات
Cross
Alignment

لا تخترع أرقاماً غير ظاهرة.

==================================================
KELTNER / ADX
==================================================

إذا كان Keltner Channel ظاهراً:
راقب:

Upper band rejection
Lower band rejection
Middle line
Breakout

إذا كان ADX ظاهراً:
راقب قوة الاتجاه و DI+ و DI-.

لا تخترع القيم.

==================================================
ENTRY
==================================================

اختر:

0 دقيقة
أو
1 دقيقة
أو
2 دقيقة

إذا كانت الحركة جاهزة:
0

إذا كان يحتاج تأكيد:
1

إذا كان يحتاج تأكيد أقوى:
2

==================================================
ENTRY PRICE
==================================================

استخدم السعر الظاهر في الصورة
أو أقرب سعر منطقي للدخول.

لا تخترع سعراً عشوائياً.

==================================================
CANCELLATION
==================================================

UP:

cancellation_price يكون أسفل الدخول
أو أسفل البنية المهمة.

الإلغاء إذا أغلقت شمعة تحته.

DOWN:

cancellation_price يكون فوق الدخول
أو فوق البنية المهمة.

الإلغاء إذا أغلقت شمعة فوقه.

==================================================
CONFIDENCE
==================================================

Confidence تعبر عن قوة توافق الأدلة.

لا تستخدم 90 أو 95 لمجرد رفع الرقم.

استخدم تقريباً:

50-60 = ضعيف
61-72 = متوسط
73-82 = جيد
83-90 = قوي

لا تتجاوز 90.

==================================================
OUTPUT
==================================================

أرجع JSON فقط.

ممنوع Markdown.

ممنوع ```json.

ممنوع أي نص قبل أو بعد JSON.

الشكل:

{
  "asset": "",
  "timeframe": "",
  "current_price": "",
  "direction": "UP",
  "confidence": 0,
  "entry_delay_minutes": 1,
  "entry_price": "",
  "cancellation_price": "",

  "up_scores": {
    "structure": 0,
    "breakout": 0,
    "liquidity": 0,
    "momentum": 0,
    "candle": 0,
    "rsi": 0,
    "summary": 0,
    "oscillators": 0,
    "moving_averages": 0
  },

  "down_scores": {
    "structure": 0,
    "breakout": 0,
    "liquidity": 0,
    "momentum": 0,
    "candle": 0,
    "rsi": 0,
    "summary": 0,
    "oscillators": 0,
    "moving_averages": 0
  },

  "analysis": {
    "structure": "",
    "breakout": "",
    "liquidity": "",
    "momentum": "",
    "candle": "",
    "rsi": "",
    "summary": "",
    "oscillators": "",
    "moving_averages": ""
  },

  "reason": ""
}
"""


# =========================================================
# JSON CLEANER
# =========================================================

def clean_json(text):

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# =========================================================
# SAFE INTEGER
# =========================================================

def safe_int(value, default=0):

    try:
        return int(float(value))
    except Exception:
        return default


# =========================================================
# SCORE LIMITS
# =========================================================

SCORE_LIMITS = {
    "structure": 2,
    "breakout": 2,
    "liquidity": 1,
    "momentum": 2,
    "candle": 2,
    "rsi": 1,
    "summary": 2,
    "oscillators": 3,
    "moving_averages": 3,
}


# =========================================================
# NORMALIZE SCORE
# =========================================================

def normalize_scores(scores):

    if not isinstance(scores, dict):
        scores = {}

    result = {}

    for name, maximum in SCORE_LIMITS.items():

        value = safe_int(
            scores.get(name, 0)
        )

        if value < 0:
            value = 0

        if value > maximum:
            value = maximum

        result[name] = value

    return result


# =========================================================
# TOTAL
# =========================================================

def total_score(scores):

    return sum(
        scores.get(name, 0)
        for name in SCORE_LIMITS
    )


# =========================================================
# GEMINI ANALYSIS
# =========================================================

async def analyze_chart(image_bytes):

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            ),
            ANALYSIS_PROMPT
        ],
        config=types.GenerateContentConfig(
            temperature=0.05,
            response_mime_type="application/json"
        )
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    data = json.loads(
        clean_json(response.text)
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Invalid Gemini response"
        )

    # =====================================================
    # SCORES
    # =====================================================

    up_scores = normalize_scores(
        data.get("up_scores", {})
    )

    down_scores = normalize_scores(
        data.get("down_scores", {})
    )

    up_total = total_score(
        up_scores
    )

    down_total = total_score(
        down_scores
    )

    data["up_scores"] = up_scores
    data["down_scores"] = down_scores

    # =====================================================
    # DIRECTION
    # =====================================================

    if up_total > down_total:
        direction = "UP"

    elif down_total > up_total:
        direction = "DOWN"

    else:

        direction = str(
            data.get("direction", "UP")
        ).upper()

        if direction not in ("UP", "DOWN"):
            direction = "UP"

    data["direction"] = direction

    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = safe_int(
        data.get("confidence", 50)
    )

    if confidence < 50:
        confidence = 50

    if confidence > 90:
        confidence = 90

    difference = abs(
        up_total - down_total
    )

    if difference >= 7:
        confidence = max(
            confidence,
            82
        )

    elif difference >= 5:
        confidence = max(
            confidence,
            76
        )

    elif difference >= 3:
        confidence = max(
            confidence,
            68
        )

    data["confidence"] = confidence

    # =====================================================
    # ENTRY DELAY
    # =====================================================

    delay = safe_int(
        data.get(
            "entry_delay_minutes",
            1
        )
    )

    if delay < 0:
        delay = 0

    if delay > 2:
        delay = 2

    data["entry_delay_minutes"] = delay

    return data


# =========================================================
# PRICE
# =========================================================

def clean_price(value):

    if value is None:
        return "N/A"

    value = str(value).strip()

    if not value:
        return "N/A"

    return value


# =========================================================
# FORMAT SIGNAL
# =========================================================

def format_signal(data):

    direction = str(
        data.get("direction", "UP")
    ).upper()

    if direction not in ("UP", "DOWN"):
        direction = "UP"

    if direction == "UP":
        icon = "🟢"
        cancel_text = "إذا أغلقت شمعة تحت"
    else:
        icon = "🔴"
        cancel_text = "إذا أغلقت شمعة فوق"

    up_scores = normalize_scores(
        data.get("up_scores", {})
    )

    down_scores = normalize_scores(
        data.get("down_scores", {})
    )

    up_total = total_score(
        up_scores
    )

    down_total = total_score(
        down_scores
    )

    delay = safe_int(
        data.get(
            "entry_delay_minutes",
            1
        )
    )

    if delay < 0:
        delay = 0

    if delay > 2:
        delay = 2

    now = datetime.now(
        SIGNAL_TZ
    )

    entry_time = (
        now.replace(
            second=0,
            microsecond=0
        )
        + timedelta(minutes=delay)
    )

    entry_time_text = (
        entry_time.strftime("%H:%M")
    )

    if delay == 0:
        entry_label = "الآن"
    elif delay == 1:
        entry_label = "بعد 1 دقيقة"
    else:
        entry_label = "بعد 2 دقيقة"

    analysis = data.get(
        "analysis",
        {}
    )

    if not isinstance(analysis, dict):
        analysis = {}

    confidence = safe_int(
        data.get("confidence", 50)
    )

    return (
        "🎓 ZinoProSignalAI\n\n"

        f"{icon} SIGNAL: {direction}\n"
        f"🎯 Confidence: {confidence}%\n\n"

        f"📊 {data.get('asset', 'Unknown')}"
        f" · ⏱ {data.get('timeframe', 'Unknown')}\n"

        "━━━━━━━━━━━━━━━━━━\n\n"

        f"📈 UP Score: {up_total}/18\n"
        f"📉 DOWN Score: {down_total}/18\n\n"

        f"🕐 الدخول: {entry_label}\n"
        f"⏰ وقت الدخول UTC-3: {entry_time_text}\n\n"

        f"💵 سعر الدخول: "
        f"{clean_price(data.get('entry_price'))}\n"

        f"🛑 إلغاء {cancel_text} "
        f"{clean_price(data.get('cancellation_price'))}\n\n"

        "━━━━━━━━━━━━━━━━━━\n"
        "📊 SIGNAL SCORE\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        "🟢 UP\n"
        f"Structure: {up_scores['structure']}/2\n"
        f"Breakout: {up_scores['breakout']}/2\n"
        f"Liquidity: {up_scores['liquidity']}/1\n"
        f"Momentum: {up_scores['momentum']}/2\n"
        f"Candle: {up_scores['candle']}/2\n"
        f"RSI: {up_scores['rsi']}/1\n"
        f"Summary: {up_scores['summary']}/2\n"
        f"Oscillators: {up_scores['oscillators']}/3\n"
        f"Moving Averages: "
        f"{up_scores['moving_averages']}/3\n"
        f"TOTAL: {up_total}/18\n\n"

        "🔴 DOWN\n"
        f"Structure: {down_scores['structure']}/2\n"
        f"Breakout: {down_scores['breakout']}/2\n"
        f"Liquidity: {down_scores['liquidity']}/1\n"
        f"Momentum: {down_scores['momentum']}/2\n"
        f"Candle: {down_scores['candle']}/2\n"
        f"RSI: {down_scores['rsi']}/1\n"
        f"Summary: {down_scores['summary']}/2\n"
        f"Oscillators: {down_scores['oscillators']}/3\n"
        f"Moving Averages: "
        f"{down_scores['moving_averages']}/3\n"
        f"TOTAL: {down_total}/18\n\n"

        "━━━━━━━━━━━━━━━━━━\n"
        "📌 ANALYSIS\n"
        "━━━━━━━━━━━━━━━━━━\n\n"

        f"Structure: "
        f"{analysis.get('structure', '')}\n\n"

        f"Breakout: "
        f"{analysis.get('breakout', '')}\n\n"

        f"Liquidity: "
        f"{analysis.get('liquidity', '')}\n\n"

        f"Momentum: "
        f"{analysis.get('momentum', '')}\n\n"

        f"Candle: "
        f"{analysis.get('candle', '')}\n\n"

        f"RSI: "
        f"{analysis.get('rsi', '')}\n\n"

        f"Summary: "
        f"{analysis.get('summary', '')}\n\n"

        f"Oscillators: "
        f"{analysis.get('oscillators', '')}\n\n"

        f"Moving Averages: "
        f"{analysis.get('moving_averages', '')}\n\n"

        f"📝 السبب:\n"
        f"{data.get('reason', '')}"
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not is_owner(update):

        await update.message.reply_text(
            "⛔ هذا البوت خاص."
        )

        return

    status = await update.message.reply_text(
        "🔎 جاري تحليل الشارت..."
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

        logger.info(
            "Screenshot received: %s bytes",
            len(image_bytes)
        )

        data = await analyze_chart(
            image_bytes
        )

        signal = format_signal(
            data
        )

        await status.edit_text(
            signal
        )

        logger.info(
            "SIGNAL=%s UP=%s DOWN=%s CONF=%s",
            data.get("direction"),
            total_score(data.get("up_scores", {})),
            total_score(data.get("down_scores", {})),
            data.get("confidence")
        )

    except json.JSONDecodeError:

        logger.exception(
            "Invalid JSON from Gemini"
        )

        await status.edit_text(
            "⚠️ Gemini أرسل نتيجة غير صالحة.\n"
            "أعد إرسال Screenshot."
        )

    except Exception as error:

        logger.exception(
            "Analysis error"
        )

        await status.edit_text(
            "⚠️ حدث خطأ أثناء تحليل الشارت.\n\n"
            f"{str(error)[:500]}"
        )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    logger.error(
        "Unhandled error: %s",
        context.error,
        exc_info=context.error
    )


# =========================================================
# MAIN
# =========================================================

def main():

    health_thread = threading.Thread(
        target=start_health_server,
        daemon=True
    )

    health_thread.start()

    logger.info(
        "ZinoProSignalAI starting..."
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Render port: %s",
        PORT
    )

    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    app.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    app.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    app.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram application starting..."
    )

    app.run_polling(
        drop_pending_updates=True
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
