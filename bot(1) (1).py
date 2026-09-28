import os
import io
import json
import logging
import asyncio
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

GEMINI_MODEL = os.getenv(
"GEMINI_MODEL",
"gemini-3.8-flash"
)

PORT = int(
os.getenv("PORT", "10000")
)

# =========================================================

# ENVIRONMENT CHECK

# =========================================================

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

try:
OWNER_ID = int(OWNER_ID)

except ValueError:
raise RuntimeError(
"OWNER_ID must be an integer"
)

# =========================================================

# LOGGING

# =========================================================

logging.basicConfig(
format="%(asctime)s - %(levelname)s - %(message)s",
level=logging.INFO,
)

logger = logging.getLogger(
"ZinoProSignalAI"
)

# =========================================================

# GEMINI CLIENT

# =========================================================

gemini = genai.Client(
api_key=GEMINI_API_KEY
)

# =========================================================

# ALGERIA TIME

# UTC +1

# =========================================================

SIGNAL_TZ = timezone(
timedelta(hours=1)
)

# =========================================================

# RENDER HEALTH SERVER

# =========================================================

class HealthHandler(
BaseHTTPRequestHandler
):

```
def do_GET(self):

    try:

        if self.path == "/health":

            body = (
                b"ZinoProSignalAI is running"
            )

        else:

            body = b"ZinoProSignalAI"

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.send_header(
            "Connection",
            "close"
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    except Exception:

        logger.exception(
            "Health request error"
        )

def log_message(
    self,
    format,
    *args
):
    return
```

def create_health_server():

```
server = ThreadingHTTPServer(
    (
        "0.0.0.0",
        PORT
    ),
    HealthHandler
)

logger.info(
    "=================================================="
)

logger.info(
    "HEALTH SERVER STARTED"
)

logger.info(
    "Listening on 0.0.0.0:%s",
    PORT
)

logger.info(
    "Health endpoint: /health"
)

logger.info(
    "=================================================="
)

return server
```

def run_health_server(
server
):

```
try:

    server.serve_forever()

except Exception:

    logger.exception(
        "Health server stopped"
    )
```

# =========================================================

# OWNER CHECK

# =========================================================

def is_owner(
update: Update
):

```
if not update.effective_user:
    return False

return (
    update.effective_user.id
    == OWNER_ID
)
```

# =========================================================

# START

# =========================================================

async def start(
update: Update,
context: ContextTypes.DEFAULT_TYPE
):

```
if not is_owner(update):

    if update.message:

        await update.message.reply_text(
            "⛔ هذا البوت خاص."
        )

    return

if update.message:

    await update.message.reply_text(
        "🎓 ZinoProSignalAI\n\n"
        "📸 أرسل Screenshot للشارت.\n\n"
        "⚡ سيتم تحليل الصورة مباشرة.\n"
        "📊 الاتجاه + الثقة + النقاط\n"
        "⏱️ وقت الدخول\n"
        "💰 سعر الدخول\n"
        "🚫 مستوى الإلغاء"
    )
```

# =========================================================

# ANALYSIS PROMPT

# =========================================================

ANALYSIS_PROMPT = """
أنت محرك التحليل الفني الرئيسي لبوت
ZinoProSignalAI.

حلل Screenshot حقيقية لشارت Quotex.

يجب اختيار اتجاه واحد فقط:

UP
أو
DOWN

ممنوع:

NO SIGNAL
WAIT
NEUTRAL

==================================================
IMPORTANT
=========

لا تجعل UP افتراضياً.

لا تجعل DOWN افتراضياً.

احسب الأدلة لصالح UP و DOWN بشكل منفصل.

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

إذا كان عامل غير ظاهر في الصورة:
لا تخترع بياناته.

==================================================
SCORING
=======

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

احسب UP و DOWN بشكل مستقل.

لا تعطِ نقاطاً مرتفعة بدون دليل واضح.

==================================================
STRUCTURE
=========

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
======

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
========

ميز بين:

True Breakout
Fake Breakout
Breakout Confirmation

لا تعتبر Wick وحده Breakout مؤكداً.

==================================================
LIQUIDITY
=========

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
========

افحص:

قوة جسم الشموع
تتابع الشموع
سرعة الحركة
Wicks

إذا كان ADX ظاهراً:
حلله.

لا تخترع ADX.

==================================================
RSI
===

إذا كان RSI ظاهراً:

Above 50
Below 50
Overbought
Oversold
Divergence

لا تخترع قيمة RSI.

==================================================
OSCILLATORS
===========

حلل المؤشرات الظاهرة فقط.

مثل:

RSI
MACD
Stochastic

إذا لم تكن ظاهرة:
لا تخترعها.

==================================================
MOVING AVERAGES
===============

إذا كانت ظاهرة:

السعر فوق أو تحت المتوسطات
اتجاه المتوسطات
Cross
Alignment

لا تخترع أرقاماً.

==================================================
KELTNER / ADX
=============

إذا كان Keltner Channel ظاهراً:

Upper band rejection
Lower band rejection
Middle line
Breakout

إذا كان ADX ظاهراً:

ADX strength
DI+
DI-

لا تخترع القيم.

==================================================
ENTRY
=====

اختر:

0
أو
1
أو
2

إذا كانت الحركة جاهزة:
0

إذا كانت تحتاج تأكيد:
1

إذا كانت تحتاج تأكيد أقوى:
2

==================================================
ENTRY PRICE
===========

استخدم السعر الظاهر في الصورة.

إذا كان السعر واضحاً:
استخدمه.

إذا لم يكن واضحاً:
استخدم أقرب سعر منطقي يمكن قراءته من الشارت.

لا تخترع سعراً عشوائياً.

==================================================
CANCELLATION
============

UP:

cancellation_price أسفل الدخول
أو أسفل بنية مهمة.

DOWN:

cancellation_price فوق الدخول
أو فوق بنية مهمة.

==================================================
CONFIDENCE
==========

الثقة تعبر عن توافق الأدلة.

50-60 = ضعيف
61-72 = متوسط
73-82 = جيد
83-90 = قوي

لا تتجاوز 90.

لا تستخدم 90 لمجرد رفع الرقم.

==================================================
FINAL CHECK
===========

قبل إخراج JSON:

* direction = UP أو DOWN فقط.
* confidence بين 50 و90.
* entry_delay_minutes = 0 أو 1 أو 2.
* UP scores لا تتجاوز 18.
* DOWN scores لا تتجاوز 18.
* لا تخترع المؤشرات غير الظاهرة.
* cancellation_price منطقي للاتجاه.
* entry_price مأخوذ من السعر الظاهر قدر الإمكان.

==================================================
OUTPUT
======

أرجع JSON فقط.

ممنوع Markdown.

ممنوع ```json.

ممنوع أي كلام قبل أو بعد JSON.

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

def clean_json(
text
):

````
if not text:
    return ""

text = text.strip()

if text.startswith(
    "```json"
):

    text = text[7:]

elif text.startswith(
    "```"
):

    text = text[3:]

if text.endswith(
    "```"
):

    text = text[:-3]

return text.strip()
````

# =========================================================

# SAFE INTEGER

# =========================================================

def safe_int(
value,
default=0
):

```
try:

    return int(
        float(value)
    )

except Exception:

    return default
```

# =========================================================

# SCORE LIMITS

# =========================================================

SCORE_LIMITS = {

```
"structure": 2,

"breakout": 2,

"liquidity": 1,

"momentum": 2,

"candle": 2,

"rsi": 1,

"summary": 2,

"oscillators": 3,

"moving_averages": 3,
```

}

# =========================================================

# NORMALIZE SCORES

# =========================================================

def normalize_scores(
scores
):

```
if not isinstance(
    scores,
    dict
):

    scores = {}

result = {}

for name, maximum in SCORE_LIMITS.items():

    value = safe_int(
        scores.get(
            name,
            0
        )
    )

    if value < 0:
        value = 0

    if value > maximum:
        value = maximum

    result[name] = value

return result
```

# =========================================================

# TOTAL SCORE

# =========================================================

def total_score(
scores
):

```
return sum(
    scores.get(
        name,
        0
    )
    for name in SCORE_LIMITS
)
```

# =========================================================

# TEMPORARY GEMINI ERROR

# =========================================================

def is_temporary_gemini_error(
error
):

```
text = str(
    error
).upper()

temporary_errors = [

    "503",

    "UNAVAILABLE",

    "429",

    "RESOURCE_EXHAUSTED",

    "504",

    "DEADLINE_EXCEEDED",

    "TIMEOUT",

    "INTERNAL",
]

return any(
    item in text
    for item in temporary_errors
)
```

# =========================================================

# GEMINI REQUEST

# =========================================================

async def request_gemini(
image_bytes
):

```
last_error = None

max_attempts = 3

for attempt in range(
    1,
    max_attempts + 1
):

    try:

        logger.info(
            "Gemini request %s/%s using model=%s",
            attempt,
            max_attempts,
            GEMINI_MODEL
        )

        response = (
            await asyncio.to_thread(
                gemini.models.generate_content,
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
        )

        if not response.text:

            raise RuntimeError(
                "Gemini returned an empty response"
            )

        return response.text

    except Exception as error:

        last_error = error

        logger.error(
            "Gemini request failed (%s/%s): %s",
            attempt,
            max_attempts,
            error
        )

        if (
            attempt < max_attempts
            and
            is_temporary_gemini_error(
                error
            )
        ):

            wait_seconds = (
                attempt * 5
            )

            logger.info(
                "Temporary Gemini error. "
                "Retrying in %s seconds...",
                wait_seconds
            )

            await asyncio.sleep(
                wait_seconds
            )

            continue

        raise

raise last_error
```

# =========================================================

# GEMINI ANALYSIS

# =========================================================

async def analyze_chart(
image_bytes
):

```
raw_text = await request_gemini(
    image_bytes
)

raw_text = clean_json(
    raw_text
)

try:

    data = json.loads(
        raw_text
    )

except json.JSONDecodeError as error:

    logger.error(
        "Gemini invalid JSON: %s",
        raw_text[:2000]
    )

    raise error

if not isinstance(
    data,
    dict
):

    raise RuntimeError(
        "Invalid Gemini response"
    )

# =====================================================
# SCORES
# =====================================================

up_scores = normalize_scores(
    data.get(
        "up_scores",
        {}
    )
)

down_scores = normalize_scores(
    data.get(
        "down_scores",
        {}
    )
)

data["up_scores"] = (
    up_scores
)

data["down_scores"] = (
    down_scores
)

up_total = total_score(
    up_scores
)

down_total = total_score(
    down_scores
)

# =====================================================
# DIRECTION
# =====================================================

direction = str(
    data.get(
        "direction",
        ""
    )
).upper().strip()

if direction not in (
    "UP",
    "DOWN"
):

    if up_total >= down_total:

        direction = "UP"

    else:

        direction = "DOWN"

data["direction"] = (
    direction
)

# =====================================================
# CONFIDENCE
# =====================================================

confidence = safe_int(
    data.get(
        "confidence",
        0
    )
)

if confidence < 50:
    confidence = 50

if confidence > 90:
    confidence = 90

data["confidence"] = (
    confidence
)

# =====================================================
# ENTRY DELAY
# =====================================================

delay = safe_int(
    data.get(
        "entry_delay_minutes",
        1
    ),
    1
)

if delay < 0:
    delay = 0

if delay > 2:
    delay = 2

data["entry_delay_minutes"] = (
    delay
)

# =====================================================
# BASIC FIELDS
# =====================================================

data["asset"] = str(
    data.get(
        "asset",
        "Unknown"
    )
).strip()

data["timeframe"] = str(
    data.get(
        "timeframe",
        "Unknown"
    )
).strip()

data["current_price"] = str(
    data.get(
        "current_price",
        ""
    )
).strip()

data["entry_price"] = str(
    data.get(
        "entry_price",
        ""
    )
).strip()

data["cancellation_price"] = str(
    data.get(
        "cancellation_price",
        ""
    )
).strip()

data["reason"] = str(
    data.get(
        "reason",
        ""
    )
).strip()

# =====================================================
# ANALYSIS FIELDS
# =====================================================

analysis = data.get(
    "analysis",
    {}
)

if not isinstance(
    analysis,
    dict
):

    analysis = {}

normalized_analysis = {}

for key in [

    "structure",

    "breakout",

    "liquidity",

    "momentum",

    "candle",

    "rsi",

    "summary",

    "oscillators",

    "moving_averages",

]:

    normalized_analysis[key] = str(
        analysis.get(
            key,
            ""
        )
    ).strip()

data["analysis"] = (
    normalized_analysis
)

logger.info(
    "Analysis successful | "
    "direction=%s | "
    "UP=%s/18 | "
    "DOWN=%s/18 | "
    "confidence=%s | "
    "delay=%s",
    direction,
    up_total,
    down_total,
    confidence,
    delay
)

return data
```

# =========================================================

# ENTRY TIME

# =========================================================

def calculate_entry_time(
received_at,
delay_minutes
):

```
base = received_at.astimezone(
    SIGNAL_TZ
)

# أول شمعة كاملة بعد استلام الصورة
entry_time = (
    base.replace(
        second=0,
        microsecond=0
    )
    + timedelta(
        minutes=1
    )
)

# إضافة التأخير الذي اختاره التحليل
if delay_minutes > 0:

    entry_time += timedelta(
        minutes=delay_minutes
    )

return entry_time
```

# =========================================================

# FORMAT SIGNAL

# =========================================================

def format_signal(
data,
received_at
):

```
direction = (
    data["direction"]
)

if direction == "UP":

    direction_text = "🟢 UP"

    cancel_text = (
        "🚫 إلغاء إذا أغلقت شمعة تحت "
        f"{data['cancellation_price']}"
    )

else:

    direction_text = "🔴 DOWN"

    cancel_text = (
        "🚫 إلغاء إذا أغلقت شمعة فوق "
        f"{data['cancellation_price']}"
    )

up_total = total_score(
    data["up_scores"]
)

down_total = total_score(
    data["down_scores"]
)

entry_time = calculate_entry_time(
    received_at,
    data["entry_delay_minutes"]
)

time_text = entry_time.strftime(
    "%H:%M"
)

delay = (
    data["entry_delay_minutes"]
)

if delay == 0:

    delay_text = (
        "الشمعة القادمة"
    )

elif delay == 1:

    delay_text = (
        "بعد 1 دقيقة"
    )

else:

    delay_text = (
        f"بعد {delay} دقائق"
    )

analysis = (
    data["analysis"]
)

message = (

    "🎓 ZinoProSignalAI\n"

    "━━━━━━━━━━━━━━━━━━\n"

    f"📊 الأصل: {data['asset']}\n"

    f"⏱️ الفريم: {data['timeframe']}\n"

    f"🎯 الاتجاه: {direction_text}\n"

    f"🔥 الثقة: {data['confidence']}%\n"

    "\n"

    f"🟢 UP: {up_total}/18\n"

    f"🔴 DOWN: {down_total}/18\n"

    "\n"

    f"⏳ الدخول: {delay_text}\n"

    f"🕐 الوقت: {time_text} الجزائر\n"

    f"💰 سعر الدخول: {data['entry_price']}\n"

    f"{cancel_text}\n"

    "━━━━━━━━━━━━━━━━━━\n"

    "📋 التحليل\n"

    f"• Structure: "
    f"{analysis['structure']}\n"

    f"• Breakout: "
    f"{analysis['breakout']}\n"

    f"• Liquidity: "
    f"{analysis['liquidity']}\n"

    f"• Momentum: "
    f"{analysis['momentum']}\n"

    f"• Candle: "
    f"{analysis['candle']}\n"

    f"• RSI: "
    f"{analysis['rsi']}\n"

    f"• Summary: "
    f"{analysis['summary']}\n"

    f"• Oscillators: "
    f"{analysis['oscillators']}\n"

    f"• Moving Averages: "
    f"{analysis['moving_averages']}\n"

    "\n"

    f"💡 السبب: {data['reason']}"
)

return message
```

# =========================================================

# PHOTO HANDLER

# =========================================================

async def handle_photo(
update: Update,
context: ContextTypes.DEFAULT_TYPE
):

```
if not is_owner(update):

    if update.message:

        await update.message.reply_text(
            "⛔ هذا البوت خاص."
        )

    return

if not update.message:
    return

# =====================================================
# وقت استلام الصورة
# =====================================================

received_at = datetime.now(
    SIGNAL_TZ
)

processing_message = (
    await update.message.reply_text(
        "🔎 جاري تحليل الشارت..."
    )
)

try:

    # =================================================
    # GET ORIGINAL PHOTO
    # =================================================

    photo = (
        update.message.photo[-1]
    )

    telegram_file = (
        await context.bot.get_file(
            photo.file_id
        )
    )

    image_buffer = (
        io.BytesIO()
    )

    # تحميل الصورة بدون Resize
    await telegram_file.download_to_memory(
        image_buffer
    )

    image_bytes = (
        image_buffer.getvalue()
    )

    if not image_bytes:

        raise RuntimeError(
            "Empty image received"
        )

    logger.info(
        "Screenshot received: %s bytes",
        len(image_bytes)
    )

    # =================================================
    # GEMINI
    # =================================================

    data = await analyze_chart(
        image_bytes
    )

    # =================================================
    # FORMAT
    # =================================================

    signal_message = format_signal(
        data,
        received_at
    )

    # =================================================
    # SEND
    # =================================================

    await processing_message.edit_text(
        signal_message
    )

except Exception as error:

    logger.exception(
        "Screenshot analysis failed"
    )

    error_text = str(
        error
    )

    if len(error_text) > 700:

        error_text = (
            error_text[:700]
            + "..."
        )

    try:

        await processing_message.edit_text(

            "❌ فشل تحليل الشارت.\n\n"

            f"الخطأ:\n{error_text}\n\n"

            "💡 إذا كان الخطأ 503 أو 429 "
            "سيتم إعادة المحاولة تلقائياً."
        )

    except Exception:

        pass
```

# =========================================================

# ERROR HANDLER

# =========================================================

async def error_handler(
update: object,
context: ContextTypes.DEFAULT_TYPE
):

```
logger.error(
    "Telegram error: %s",
    context.error
)
```

# =========================================================

# MAIN

# =========================================================

def main():

```
logger.info(
    "=================================================="
)

logger.info(
    "Starting ZinoProSignalAI"
)

logger.info(
    "Render PORT=%s",
    PORT
)

logger.info(
    "Gemini model=%s",
    GEMINI_MODEL
)

logger.info(
    "=================================================="
)

# =====================================================
# CREATE PORT FIRST
# =====================================================

try:

    health_server = (
        create_health_server()
    )

except Exception:

    logger.exception(
        "FAILED TO BIND PORT %s",
        PORT
    )

    raise

health_thread = threading.Thread(
    target=run_health_server,
    args=(health_server,),
    daemon=True,
    name="HealthServer"
)

health_thread.start()

# =====================================================
# TELEGRAM
# =====================================================

application = (
    Application.builder()
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
    MessageHandler(
        filters.PHOTO,
        handle_photo
    )
)

application.add_error_handler(
    error_handler
)

logger.info(
    "Telegram application starting..."
)

try:

    application.run_polling(
        drop_pending_updates=True
    )

finally:

    logger.info(
        "Stopping health server..."
    )

    try:

        health_server.shutdown()

        health_server.server_close()

    except Exception:

        pass

    logger.info(
        "ZinoProSignalAI stopped."
    )
```

# =========================================================

# RUN

# =========================================================

if **name** == "**main**":

```
main()
```
