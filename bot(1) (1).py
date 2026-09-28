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

try:
OWNER_ID = int(OWNER_ID)
except ValueError:
raise RuntimeError("OWNER_ID must be an integer")

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

# ALGERIA TIME UTC+1

# =========================================================

SIGNAL_TZ = timezone(
timedelta(hours=1)
)

# =========================================================

# HEALTH SERVER FOR RENDER

# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

```
def do_GET(self):

    try:
        if self.path == "/health":
            body = b"ZinoProSignalAI is running"
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

        self.wfile.write(body)

    except Exception:
        logger.exception(
            "Health request error"
        )

def log_message(self, format, *args):
    return
```

def start_health_server():

```
try:

    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
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

    server.serve_forever()

except Exception:
    logger.exception(
        "FAILED TO START HEALTH SERVER"
    )
```

# =========================================================

# OWNER CHECK

# =========================================================

def is_owner(update: Update):

```
if not update.effective_user:
    return False

return update.effective_user.id == OWNER_ID
```

# =========================================================

# START COMMAND

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

await update.message.reply_text(
    "🎓 ZinoProSignalAI\n\n"
    "📸 أرسل Screenshot للشارت.\n\n"
    "⚡ سيتم تحليل الشارت مباشرة "
    "وإعطاؤك الاتجاه ووقت الدخول "
    "وسعر الدخول ومستوى الإلغاء."
)
```

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
=========

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

لكل عامل، احسب نقاط UP ونقاط DOWN بشكل مستقل.

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

واستخدم ADX إذا كان ظاهراً.

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

==================================================
MOVING AVERAGES
===============

إذا كانت ظاهرة:

السعر فوق أو تحت المتوسطات
اتجاه المتوسطات
Cross
Alignment

لا تخترع أرقاماً غير ظاهرة.

==================================================
KELTNER / ADX
=============

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
=====

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
===========

استخدم السعر الظاهر في الصورة
أو أقرب سعر منطقي للدخول.

لا تخترع سعراً عشوائياً.

==================================================
CANCELLATION
============

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
==========

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
======

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

````
if not text:
    return ""

text = text.strip()

if text.startswith("```json"):
    text = text[7:]

elif text.startswith("```"):
    text = text[3:]

if text.endswith("```"):
    text = text[:-3]

return text.strip()
````

# =========================================================

# SAFE INTEGER

# =========================================================

def safe_int(value, default=0):

```
try:
    return int(float(value))
except Exception:
    return default
```

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

# NORMALIZE SCORES

# =========================================================

def normalize_scores(scores):

```
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
```

# =========================================================

# TOTAL SCORE

# =========================================================

def total_score(scores):

```
return sum(
    scores.get(name, 0)
    for name in SCORE_LIMITS
)
```

# =========================================================

# GEMINI ANALYSIS

# =========================================================

async def analyze_chart(image_bytes):

```
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

raw_text = clean_json(
    response.text
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
    data.get("down
```
