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
# SIGNAL TIMEZONE
# UTC-3
# =========================================================

SIGNAL_TZ = timezone(
    timedelta(hours=-3)
)


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

class HealthHandler(BaseHTTPRequestHandler):

    def do_GET(self):

        if self.path == "/health":

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8"
            )

            self.end_headers()

            self.wfile.write(
                b"ZinoProSignalAI is running"
            )

            return

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain; charset=utf-8"
        )

        self.end_headers()

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
# OWNER CHECK
# =========================================================

def is_owner(update: Update):

    if not update.effective_user:
        return False

    return (
        update.effective_user.id == OWNER_ID
    )


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
        "الاتجاه + وقت الدخول + سعر الدخول "
        "+ مستوى الإلغاء."
    )


# =========================================================
# ANALYSIS PROMPT
# =========================================================

ANALYSIS_PROMPT = """
أنت محرك تحليل فني متقدم لبوت ZinoProSignalAI.

مهمتك تحليل Screenshot حقيقية لشارت Quotex
وإنتاج اتجاه واحد فقط:

UP
أو
DOWN

لا تعتمد على مؤشر واحد.

الأولوية للتحليل بهذا الترتيب:

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
قاعدة مهمة جداً
==================================================

لا تجعل UP افتراضياً.

لا تجعل DOWN افتراضياً.

احسب الأدلة لصالح UP والأدلة لصالح DOWN
بشكل منفصل.

بعد ذلك اختر الاتجاه الذي لديه الأدلة الأقوى.

إذا كان هناك تعادل، استخدم:

Market Structure
ثم Candle Close
ثم Breakout
ثم Momentum

لتحديد الاتجاه.

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

لكل عامل:

UP يحصل على نقاط إذا كانت الأدلة تدعم UP.

DOWN يحصل على نقاط إذا كانت الأدلة تدعم DOWN.

لا تعطِ نقاطاً لمجرد التخمين.

إذا كان المؤشر غير ظاهر في الصورة:
اجعل نقاطه 0.

==================================================
MARKET STRUCTURE
==================================================

افحص:

Higher High
Higher Low
Lower High
Lower Low

و:

Break of Structure
Change of Character

إذا كان السعر يصنع Lower High + Lower Low
فهذه أدلة لصالح DOWN.

إذا كان السعر يصنع Higher High + Higher Low
فهذه أدلة لصالح UP.

==================================================
CANDLE CLOSE
==================================================

ركز بشكل خاص على:

آخر شمعة مغلقة.

لا تعتمد فقط على لون شمعة لم تغلق.

افحص:

Open
Close
Body
Upper Wick
Lower Wick
Strong Close
Weak Close

إغلاق قوي فوق مستوى مهم:
دليل UP.

إغلاق قوي تحت مستوى مهم:
دليل DOWN.

==================================================
BREAKOUT
==================================================

ميز بين:

True Breakout
Fake Breakout
Breakout Confirmation

الكسر الحقيقي يحتاج إغلاقاً واضحاً
وليس مجرد Wick.

إذا حدث Breakout ثم رجوع سريع داخل المستوى:
قد يكون Fake Breakout.

==================================================
LIQUIDITY
==================================================

افحص:

High Sweep
Low Sweep
Liquidity Sweep
Rejection

مثال:

السعر يأخذ قاعاً سابقاً ثم يغلق فوقه
مع رفض واضح:
قد يكون دليلاً لصالح UP.

السعر يأخذ قمة سابقة ثم يغلق تحتها
مع رفض واضح:
قد يكون دليلاً لصالح DOWN.

==================================================
MOMENTUM
==================================================

افحص قوة الحركة:

Strong bullish candles
Strong bearish candles
Consecutive candles
Body size
Wicks

وADX إذا كان ظاهراً.

==================================================
CANDLE PATTERNS
==================================================

افحص:

Bullish Engulfing
Bearish Engulfing
Hammer
Shooting Star
Pin Bar
Strong Close
Weak Close
Continuation

لا تعتبر Pattern صحيحاً إذا لم يكن واضحاً
في الصورة.

==================================================
RSI
==================================================

إذا كان RSI ظاهراً:

Above 50
Below 50
Overbought
Oversold
Divergence

لا تخترع قيمة RSI إذا لم تكن واضحة.

==================================================
OSCILLATORS
==================================================

حلل فقط المؤشرات المتذبذبة الظاهرة.

مثل:

RSI
MACD
Stochastic

إذا لم يظهر المؤشر:
لا تخترع قراءة.

==================================================
MOVING AVERAGES
==================================================

إذا كانت Moving Averages ظاهرة:

السعر فوق/تحت المتوسطات
اتجاه المتوسطات
Cross
Alignment

لا تخترع أرقام المتوسطات.

==================================================
ENTRY DELAY
==================================================

اختر فقط:

0
1
2

إذا كانت الحركة جاهزة:
0

إذا كان يحتاج تأكيد:
1

إذا كان يحتاج تأكيد أقوى:
2

لا تستخدم أكثر من دقيقتين.

==================================================
ENTRY PRICE
==================================================

حدد سعر الدخول من السعر الظاهر في الصورة
أو من أقرب سعر منطقي للدخول.

لا تضف أرقاماً عشوائية.

==================================================
CANCELLATION
==================================================

إذا كان الاتجاه UP:

cancellation_price يجب أن يكون مستوى
أسفل منطقة الدخول/البنية المهمة.

الإلغاء:
إذا أغلقت شمعة تحت هذا المستوى.

إذا كان الاتجاه DOWN:

cancellation_price يجب أن يكون مستوى
فوق منطقة الدخول/البنية المهمة.

الإلغاء:
إذا أغلقت شمعة فوق هذا المستوى.

==================================================
CONFIDENCE
==================================================

Confidence ليست ضماناً للفوز.

اجعلها تعكس قوة توافق الأدلة.

قوة منخفضة:
50-60

قوة متوسطة:
61-72

قوة جيدة:
73-82

قوة قوية:
83-90

لا تضع 90+ إلا إذا كانت الأدلة
متعددة وواضحة جداً.

==================================================
IMPORTANT
==================================================

لا تستخدم Support/Resistance كقسم مستقل
في النتيجة النهائية.

لا تضف مؤشرات غير موجودة.

لا تخترع أسعاراً غير ظاهرة.

لا تخترع اسم الزوج.

لا تخترع الفريم.

==================================================
OUTPUT
==================================================

أرجع JSON فقط.

لا Markdown.

لا ```json.

لا نص قبل JSON.

استخدم هذا الشكل:

{
  "asset": "",
  "timeframe": "",
  "current_price": "",

  "direction": "UP",

  "confidence": 0,

  "entry_delay_minutes": 0,

  "entry_price": "",
  "cancellation_price": "",

  "up_score": 0,
  "down_score": 0,

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
# SAFE JSON CLEANER
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
# CLAMP SCORE
# =========================================================

def clamp_score(value, maximum):

    value = safe_int(value)

    if value < 0:
        return 0

    if value > maximum:
        return maximum

    return value


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

def normalize_scores(raw_scores):

    if not isinstance(raw_scores, dict):
        raw_scores = {}

    result = {}

    for name, maximum in SCORE_LIMITS.items():

        result[name] = clamp_score(
            raw_scores.get(name, 0),
            maximum
        )

    return result


# =========================================================
# TOTAL SCORE
# =========================================================

def calculate_total(scores):

    return sum(
        safe_int(scores.get(name, 0))
        for name in SCORE_LIMITS
    )


# =========================================================
# ANALYSIS
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

    text = response.text

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response"
        )

    text = clean_json(text)

    data = json.loads(text)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Gemini response is not an object"
        )

    # =====================================================
    # NORMALIZE UP / DOWN SCORES
    # =====================================================

    up_scores = normalize_scores(
        data.get("up_scores", {})
    )

    down_scores = normalize_scores(
        data.get("down_scores", {})
    )

    up_total = calculate_total(
        up_scores
    )

    down_total = calculate_total(
        down_scores
    )

    # =====================================================
    # FORCE SCORE VALUES FROM COMPONENTS
    # =====================================================

    data["up_scores"] = up_scores
    data["down_scores"] = down_scores

    data["up_score"] = up_total
    data["down_score"] = down_total

    # =====================================================
    # DIRECTION
    # =====================================================

    direction = str(
        data.get("direction", "")
    ).upper()

    if direction not in ("UP", "DOWN"):

        if up_total > down_total:
            direction = "UP"

        elif down_total > up_total:
            direction = "DOWN"

        else:
            direction = "UP"

    # =====================================================
    # DO NOT ALLOW GEMINI TO IGNORE SCORE DIFFERENCE
    # =====================================================

    score_difference = abs(
        up_total - down_total
    )

    if score_difference >= 3:

        if up_total > down_total:
            direction = "UP"
        else:
            direction = "DOWN"

    elif score_difference >= 1:

        if up_total > down_total:
            direction = "UP"
        else:
            direction = "DOWN"

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

    # Stronger score agreement increases confidence.
    if score_difference >= 7:
        confidence = max(
            confidence,
            82
        )

    elif score_difference >= 5:
        confidence = max(
            confidence,
            76
        )

    elif score_difference >= 3:
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
# PRICE FORMAT
# =========================================================

def clean_price(value):

    if value is None:
        return "N/A"

    text = str(value).strip()

    if not text:
        return "N/A"

    return text


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

        cancel_text = (
            "إذا أغلقت شمعة تحت"
        )

    else:

        icon = "🔴"

        cancel_text = (
            "إذا أغلقت شمعة فوق"
        )

    up_scores = normalize_scores(
        data.get("up_scores", {})
    )

    down_scores = normalize_scores(
        data.get("down_scores", {})
    )

    up_total = calculate_total(
        up_scores
    )

    down_total = calculate_total(
        down_scores
    )

    # =====================================================
    # ENTRY TIME UTC-3
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

    now = datetime.now(
        SIGNAL_TZ
    )

    current_minute = now.replace(
        second=0,
        microsecond=0
    )

    entry_time = (
        current_minute
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

    # =====================================================
    # ANALYSIS
    # =====================================================

    analysis = data.get(
        "analysis",
        {}
    )

    if not isinstance(analysis, dict):
        analysis = {}

    confidence = safe_int(
        data.get("
