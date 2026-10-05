import os
import json
import logging
import threading
import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from google import genai
from google.genai import types

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OWNER_ID = int(os.getenv("OWNER_ID", "0"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
MT5_API_KEY = os.getenv("MT5_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "10000"))
TIMEZONE = ZoneInfo("Africa/Algiers")
TIMEFRAME = "M1"
MIN_CANDLES = 40
MAX_CANDLES = 150
MARKET_DATA_MAX_AGE = 90
BACKGROUND_INTERVAL = 3
ENTRY_DELAY_MINUTES = 2
RECOVERY_DELAY_SECONDS = 120
RECOVERY_LIMIT = 1
MIN_SCORE = 13
TOP_CANDIDATES = 8
MAX_CONFIDENCE = 89
HISTORY_FILE = "signal_history.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("ZinoProSignalAI")

gemini_client = None
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized")
    except Exception as e:
        logger.error("Gemini initialization failed: %s", e)

state_lock = threading.RLock()
market_data = {}
latest_batch_id = None
latest_batch_complete = False
last_processed_batch_id = None
last_processed_candle = {}
last_processed_fingerprint = {}
active_trade = None
recovery_pending = False
recovery_wait_until = 0
recovery_number = 0
last_signal_direction = None
same_direction_streak = 0
last_signal_symbol = None
background_task = None
background_stop_event = None

stats = {"wins": 0, "losses": 0, "base_wins": 0, "base_losses": 0, "recovery_wins": 0, "recovery_losses": 0, "signals": 0}
history = []

def load_history():
    global history, stats, last_signal_direction, same_direction_streak, last_signal_symbol
    try:
        if not os.path.exists(HISTORY_FILE): return
        with open(HISTORY_FILE, "r", encoding="utf-8") as f: data = json.load(f)
        if isinstance(data, dict):
            saved = data.get("stats", {})
            for k in stats:
                if k in saved: stats[k] = int(saved[k])
            history = data.get("history", [])[-500:]
            if history:
                d = history[-1].get("direction")
                last_signal_direction = d if d in ("UP", "DOWN") else None
                last_signal_symbol = history[-1].get("symbol")
                same_direction_streak = 0
                for item in reversed(history):
                    if item.get("direction") == d: same_direction_streak += 1
                    else: break
        logger.info("History loaded | signals=%s wins=%s losses=%s", stats["signals"], stats["wins"], stats["losses"])
    except Exception as e: logger.exception("History load error: %s", e)

def save_history():
    try:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"stats": stats, "history": history[-500:]}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)
    except Exception as e: logger.exception("History save error: %s", e)

load_history()

def now_algiers(): return datetime.now(TIMEZONE)
def now_timestamp(): return time.time()
def safe_float(v, default=0.0):
    try: return float(v)
    except Exception: return default
def safe_int(v, default=0):
    try: return int(v)
    except Exception: return default
def clamp(v, lo, hi): return max(lo, min(hi, v))
def normalize_direction(d):
    d = str(d or "").upper().strip()
    return "UP" if d in ("UP", "CALL", "BUY") else "DOWN" if d in ("DOWN", "PUT", "SELL") else ""

def ema(values, period):
    if len(values) < period: return None
    m = 2.0 / (period + 1.0)
    r = sum(values[:period]) / period
    for p in values[period:]: r = (p-r)*m+r
    return r

def rsi(values, period=14):
    if len(values) <= period: return None
    gains=[]; losses=[]
    for i in range(1, period+1):
        d=values[i]-values[i-1]; gains.append(max(d,0)); losses.append(max(-d,0))
    ag=sum(gains)/period; al=sum(losses)/period
    for i in range(period+1,len(values)):
        d=values[i]-values[i-1]; ag=((ag*(period-1))+max(d,0))/period; al=((al*(period-1))+max(-d,0))/period
    return 100.0 if al==0 else 100.0-(100.0/(1.0+ag/al))

def williams_r(highs,lows,closes,period=14):
    if len(closes)<period:return None
    hi=max(highs[-period:]); lo=min(lows[-period:]); c=closes[-1]
    return -50.0 if hi==lo else ((hi-c)/(hi-lo))*-100.0

def atr(highs,lows,closes,period=14):
    if len(closes)<period+1:return None
    trs=[]
    for i in range(1,len(closes)):
        trs.append(max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])))
    return sum(trs[-period:])/period if len(trs)>=period else None

def adx_di(highs,lows,closes,period=14):
    if len(closes)<period*2:return None
    trs=[]; plus=[]; minus=[]
    for i in range(1,len(closes)):
        up=highs[i]-highs[i-1]; dn=lows[i-1]-lows[i]
        trs.append(max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])))
        plus.append(up if up>dn and up>0 else 0.0); minus.append(dn if dn>up and dn>0 else 0.0)
    av=sum(trs[-period:])/period
    if av<=0:return None
    p=100*sum(plus[-period:])/period/av; m=100*sum(minus[-period:])/period/av
    den=p+m; adx=0 if den==0 else 100*abs(p-m)/den
    return {"adx":adx,"plus_di":p,"minus_di":m}

def candle_body(c): return abs(c["close"]-c["open"])
def candle_range(c): return max(c["high"]-c["low"],1e-12)
def candle_direction(c): return "UP" if c["close"]>c["open"] else "DOWN" if c["close"]<c["open"] else "FLAT"

def structure_signal(c):
    if len(c)<12:return {"direction":None,"strength":0,"reason":""}
    r=c[-6:]; p=c[-12:-6]; rh=max(x["high"] for x in r); rl=min(x["low"] for x in r); ph=max(x["high"] for x in p); pl=min(x["low"] for x in p); close=c[-1]["close"]
    if close>ph:return {"direction":"UP","strength":3,"reason":"bullish structure break"}
    if close<pl:return {"direction":"DOWN","strength":3,"reason":"bearish structure break"}
    if rh>ph and rl>pl:return {"direction":"UP","strength":2,"reason":"higher-high/higher-low structure"}
    if rh<ph and rl<pl:return {"direction":"DOWN","strength":2,"reason":"lower-high/lower-low structure"}
    return {"direction":None,"strength":0,"reason":""}

def build_snapshot(symbol,data):
    raw=data.get("candles",[])
    clean=[]
    for x in raw:
        if not isinstance(x,dict):continue
        clean.append({"time":safe_int(x.get("time")),"open":safe_float(x.get("open")),"high":safe_float(x.get("high")),"low":safe_float(x.get("low")),"close":safe_float(x.get("close")),"tick_volume":safe_int(x.get("tick_volume")),"spread":safe_int(x.get("spread"))})
    clean=sorted(clean,key=lambda x:x["time"])[-MAX_CANDLES:]
    if len(clean)<MIN_CANDLES:return None
    o=[x["open"] for x in clean]; h=[x["high"] for x in clean]; l=[x["low"] for x in clean]; c=[x["close"] for x in clean]
    price=safe_float(data.get("current_bid"),c[-1]); e9=ema(c,9); e21=ema(c,21); rv=rsi(c,14); wr=williams_r(h,l,c,14); av=atr(h,l,c,14); ad=adx_di(h,l,c,14); ke=ema(c,20); ka=atr(h,l,c,10)
    last=clean[-1]; body_ratio=candle_body(last)/candle_range(last); cd=candle_direction(last); last3=clean[-3:]; uc=sum(x["close"]>x["open"] for x in last3); dc=sum(x["close"]<x["open"] for x in last3)
    st=structure_signal(clean); up=down=0; ur=[]; dr=[]; iu=idn=0
    if st["direction"]=="UP": up+=min(st["strength"]+1,4); iu+=1; ur.append(st["reason"])
    elif st["direction"]=="DOWN": down+=min(st["strength"]+1,4); idn+=1; dr.append(st["reason"])
    if uc>=2 and c[-1]>c[-4]:
        if st["direction"]!="UP": up+=2; iu+=1
        ur.append("bullish recent price action")
    elif dc>=2 and c[-1]<c[-4]:
        if st["direction"]!="DOWN": down+=2; idn+=1
        dr.append("bearish recent price action")
    lb=clean[-8:-1]
    if lb:
        lh=max(x["high"] for x in lb); ll=min(x["low"] for x in lb)
        if c[-1]>lh: up+=3; iu+=1; ur.append("clean upside breakout")
        elif c[-1]<ll: down+=3; idn+=1; dr.append("clean downside breakout")
        else:
            rh=max(x["high"] for x in clean[-5:-1]); rl=min(x["low"] for x in clean[-5:-1])
            if last["high"]>rh and last["close"]<rh: down+=2; idn+=1; dr.append("bearish liquidity rejection")
            elif last["low"]<rl and last["close"]>rl: up+=2; iu+=1; ur.append("bullish liquidity rejection")
    mu=(c[-1]>c[-4])+(c[-1]>c[-8]); md=(c[-1]<c[-4])+(c[-1]<c[-8])
    if mu>=2 and md==0: up+=3; iu+=1; ur.append("bullish momentum")
    elif md>=2 and mu==0: down+=3; idn+=1; dr.append("bearish momentum")
    elif mu>md: up+=1
    elif md>mu: down+=1
    tu=e9 is not None and e21 is not None and e9>e21 and price>e9; td=e9 is not None and e21 is not None and e9<e21 and price<e9
    if ad:
        if ad["adx"]>=18 and ad["plus_di"]>ad["minus_di"]:
            if tu: up+=3; iu+=1; ur.append("EMA/ADX bullish trend confirmation")
            else: up+=1
        elif ad["adx"]>=18 and ad["minus_di"]>ad["plus_di"]:
            if td: down+=3; idn+=1; dr.append("EMA/ADX bearish trend confirmation")
            else: down+=1
    else:
        if tu: up+=2; iu+=1
        elif td: down+=2; idn+=1
    ou=od=0
    if rv is not None:
        if 52<rv<70:ou+=1
        elif 30<rv<48:od+=1
    if wr is not None:
        if -50<wr<-20:ou+=1
        elif -80<wr<-50:od+=1
    if ou>=2:up+=2; iu+=1; ur.append("RSI/Williams bullish confirmation")
    elif od>=2:down+=2; idn+=1; dr.append("RSI/Williams bearish confirmation")
    elif ou>od:up+=1
    elif od>ou:down+=1
    if ke is not None and ka is not None:
        upper=ke+5*ka; lower=ke-5*ka
        if ke<price<upper:up+=1
        elif lower<price<ke:down+=1
    if body_ratio>=.65:
        if cd=="UP":up+=2
        elif cd=="DOWN":down+=2
    elif body_ratio>=.45:
        if cd=="UP":up+=1
        elif cd=="DOWN":down+=1
    rawdiff=up-down; edge=clamp(rawdiff,-8,8); bu=clamp(int(round(10+edge)),2,18); bd=20-bu
    direction="UP" if bu>bd else "DOWN" if bd>bu else ("UP" if cd=="UP" else "DOWN")
    score=bu if direction=="UP" else bd; opp=bd if direction=="UP" else bu; indep=iu if direction=="UP" else idn
    conflict=abs(rawdiff)<=2; gap=abs(bu-bd); conf=56+gap*3+indep*2-(5 if conflict else 0)
    if score<14:conf-=5
    if score>=17 and indep>=4:conf+=2
    conf=int(clamp(conf,55,MAX_CONFIDENCE))
    reasons=ur if direction=="UP" else dr; reasons=[x for i,x in enumerate(reasons) if x and x not in reasons[:i]]
    fp=hashlib.sha256(f"{symbol}|{clean[-1]['time']}|{clean[-1]['open']:.10f}|{clean[-1]['high']:.10f}|{clean[-1]['low']:.10f}|{clean[-1]['close']:.10f}|{clean[-1]['tick_volume']}".encode()).hexdigest()
    return {"symbol":symbol,"timeframe":"M1","direction":direction,"up_score":bu,"down_score":bd,"score":score,"opposite_score":opp,"confidence":conf,"price":price,"atr":av or 0,"rsi":rv,"williams":wr,"ema9":e9,"ema21":e21,"adx":ad,"candle_time":clean[-1]["time"],"fingerprint":fp,"independent_groups":indep,"conflict":conflict,"reason":"; ".join(reasons[:3]) or "technical price-action confluence","candles":len(clean)}

def is_data_fresh(d): return now_timestamp()-safe_float(d.get("received_at"))<=MARKET_DATA_MAX_AGE

def get_candidates():
    out=[]
    with state_lock: items=list(market_data.items())
    for sym,d in items:
        if not is_data_fresh(d):continue
        s=build_snapshot(sym,d)
        if not s or s["score"]<MIN_SCORE:continue
        if s["candle_time"]<=last_processed_candle.get(sym,-1):continue
        if last_processed_fingerprint.get(sym)==s["fingerprint"]:continue
        sel=float(s["score"])
        if last_signal_direction==s["direction"] and same_direction_streak>=3:sel-=min(3,same_direction_streak-2)
        if last_signal_symbol==sym:sel-=2
        age=now_timestamp()-safe_float(d.get("received_at")); sel += 1 if age<=20 else -1 if age>60 else 0
        sel += 1.5 if s["independent_groups"]>=4 else .5 if s["independent_groups"]>=3 else 0
        if s["conflict"]:sel-=1.5
        s["selection_score"]=sel; out.append(s)
    out.sort(key=lambda x:(x["selection_score"],x["score"],x["independent_groups"],x["confidence"]),reverse=True)
    return out

def ask_gemini(candidates):
    if not candidates:return None
    if not gemini_client:return candidates[0]
    compact=[{k:c[k] for k in ("symbol","direction","up_score","down_score","score","confidence","price","rsi","williams","independent_groups","reason","candle_time")} for c in candidates[:TOP_CANDIDATES]]
    prompt=f'''Choose exactly one candidate. Do not change its direction. Do not invent price. Prefer fresh data and independent confluence. Return JSON only: {{"symbol":"EXACT_SYMBOL","direction":"UP or DOWN","quality":1-10,"reason":"short reason"}}\nCandidates:\n{json.dumps(compact,ensure_ascii=False)}'''
    try:
        r=gemini_client.models.generate_content(model=GEMINI_MODEL,contents=prompt,config=types.GenerateContentConfig(temperature=0.1,response_mime_type="application/json"))
        p=json.loads(r.text or "{}"); sym=str(p.get("symbol","")).strip(); d=normalize_direction(p.get("direction"))
        for c in candidates[:TOP_CANDIDATES]:
            if c["symbol"]==sym and c["direction"]==d:
                c=dict(c); c["ai_reason"]=str(p.get("reason","")).strip() or c["reason"]; return c
    except Exception as e: logger.warning("Gemini selection error: %s",e)
    return candidates[0]

def entry_time(): return now_algiers().replace(second=0,microsecond=0)+timedelta(minutes=ENTRY_DELAY_MINUTES)
def cancel_price(direction,price,av):
    dist=(av*.45) if av and av>0 else abs(price)*.0003
    return price-dist if direction=="UP" else price+dist

def format_signal(c,mode):
    d=c["direction"]; p=c["price"]; data=market_data.get(c["symbol"],{}); digits=clamp(safe_int(data.get("digits"),5),0,10); ep=entry_time(); cp=cancel_price(d,p,c.get("atr",0)); cancel=(f"⚠️ إلغاء إذا أغلقت الشمعة تحت {cp:.{digits}f}" if d=="UP" else f"⚠️ إلغاء إذا أغلقت الشمعة فوق {cp:.{digits}f}")
    return ("🎓 ZinoProSignalAI\n━━━━━━━━━━━━━━━━━━\n" f"📊 {c['symbol']} | M1\n\n" f"{'🔁 RECOVERY 1/1' if mode=='RECOVERY' else '🎯 BASE TRADE'}\n" f"{'🟢 UP' if d=='UP' else '🔴 DOWN'}\n\n" f"🔥 Confidence: {c['confidence']}%\n🟢 UP Score: {c['up_score']}/20\n🔴 DOWN Score: {c['down_score']}/20\n\n" f"⏱️ Entry after: {ENTRY_DELAY_MINUTES} minutes\n🕐 ENTRY TIME: {ep.strftime('%H:%M:%S')}\n💰 Entry Price: {p:.{digits}f}\n{cancel}\n\n🧠 {c.get('ai_reason') or c['reason']}\n━━━━━━━━━━━━━━━━━━")

async def send_signal(app,c,mode):
    global active_trade,last_signal_direction,same_direction_streak,last_signal_symbol,recovery_pending,recovery_number
    with state_lock:
        if active_trade is not None:return False
        d=c["direction"]; sym=c["symbol"]
        same_direction_streak=same_direction_streak+1 if last_signal_direction==d else 1; last_signal_direction=d; last_signal_symbol=sym
        active_trade={"symbol":sym,"direction":d,"mode":mode,"confidence":c["confidence"],"up_score":c["up_score"],"down_score":c["down_score"],"entry_price":c["price"],"entry_time":entry_time().isoformat(),"candle_time":c["candle_time"],"created_at":now_algiers().isoformat()}
        stats["signals"]+=1; last_processed_candle[sym]=c["candle_time"]; last_processed_fingerprint[sym]=c["fingerprint"]; save_history()
    try:
        await app.bot.send_message(chat_id=OWNER_ID,text=format_signal(c,mode)); logger.info("SIGNAL SENT | %s %s | %s",sym,d,mode); return True
    except Exception as e:
        logger.exception("Telegram send error: %s",e)
        with state_lock: active_trade=None
        return False

async def process_batch(app):
    global last_processed_batch_id,latest_batch_complete,recovery_pending,recovery_number
    with state_lock:
        if not latest_batch_complete or not latest_batch_id or latest_batch_id==last_processed_batch_id or active_trade is not None:return
        if recovery_pending and now_timestamp()<recovery_wait_until:return
        batch=latest_batch_id
    candidates=get_candidates()
    if not candidates:
        with state_lock: last_processed_batch_id=batch; latest_batch_complete=False
        return
    selected=ask_gemini(candidates)
    mode="RECOVERY" if recovery_pending else "BASE"
    sent=await send_signal(app,selected,mode)
    with state_lock:
        last_processed_batch_id=batch; latest_batch_complete=False
        if sent and mode=="RECOVERY": recovery_pending=False; recovery_number=1

async def background_loop(app,stop):
    logger.info("Background scanner started")
    while not stop.is_set():
        try: await process_batch(app)
        except asyncio.CancelledError: raise
        except Exception as e: logger.exception("Background loop error: %s",e)
        try: await asyncio.wait_for(stop.wait(),timeout=BACKGROUND_INTERVAL)
        except asyncio.TimeoutError: pass

def is_owner(update): return bool(update and update.effective_user and update.effective_user.id==OWNER_ID)
async def owner_only(update):
    if not is_owner(update):
        try: await update.message.reply_text("⛔ هذا البوت خاص بالمالك فقط.")
        except Exception: pass
        return False
    return True

async def start_command(update,context):
    if await owner_only(update): await update.message.reply_text("🎓 ZinoProSignalAI\n\n✅ MT5 Scanner: ON\n📊 Timeframe: M1\n🎯 One strongest trade only\n🔁 Recovery: 1/1\n\n/stats\n/win\n/loss\n/reset\n/mt5status")

async def win_command(update,context):
    global active_trade,recovery_pending,recovery_wait_until,recovery_number
    if not await owner_only(update):return
    with state_lock:
        if active_trade is None: await update.message.reply_text("ℹ️ ما كاش صفقة نشطة حاليًا."); return
        t=dict(active_trade); mode=t.get("mode","BASE"); stats["wins"]+=1; stats["recovery_wins" if mode=="RECOVERY" else "base_wins"]+=1
        history.append({"result":"WIN",**{k:t.get(k) for k in ("symbol","direction","mode","confidence","up_score","down_score","entry_price","entry_time")},"time":now_algiers().isoformat()}); active_trade=None; recovery_pending=False; recovery_wait_until=0; recovery_number=0; save_history()
    await update.message.reply_text(f"✅ WIN مسجلة\n\n📊 {t.get('symbol')} | {t.get('direction')}\n🎯 {mode}\n\n🔎 البوت يرجع يبحث على أقوى فرصة جديدة.")

async def loss_command(update,context):
    global active_trade,recovery_pending,recovery_wait_until,recovery_number
    if not await owner_only(update):return
    with state_lock:
        if active_trade is None: await update.message.reply_text("ℹ️ ما كاش صفقة نشطة حاليًا."); return
        t=dict(active_trade); mode=t.get("mode","BASE"); stats["losses"]+=1; stats["recovery_losses" if mode=="RECOVERY" else "base_losses"]+=1
        history.append({"result":"LOSS",**{k:t.get(k) for k in ("symbol","direction","mode","confidence","up_score","down_score","entry_price","entry_time")},"time":now_algiers().isoformat()}); active_trade=None
        if mode=="BASE": recovery_pending=True; recovery_number=1; recovery_wait_until=now_timestamp()+RECOVERY_DELAY_SECONDS; save_history(); msg=f"❌ LOSS مسجلة\n\n📊 {t.get('symbol')} | {t.get('direction')}\n🎯 BASE TRADE\n\n🔁 Recovery 1/1 مفعلة.\n⏱️ انتظر {RECOVERY_DELAY_SECONDS//60} دقائق لإعادة تحليل السوق."
        else: recovery_pending=False; recovery_wait_until=0; recovery_number=0; save_history(); msg="❌ LOSS مسجلة\n\n🔁 RECOVERY 1/1\n⛔ Recovery انتهت.\n🔎 البوت يرجع الآن يبحث عن BASE جديدة."
    await update.message.reply_text(msg)

async def stats_command(update,context):
    if not await owner_only(update):return
    with state_lock:
        total=stats["wins"]+stats["losses"]; wr=stats["wins"]/total*100 if total else 0; active=active_trade is not None; rec=recovery_pending
        msg=("📊 ZinoProSignalAI Stats\n━━━━━━━━━━━━━━━━━━\n" f"📌 Signals: {stats['signals']}\n✅ Wins: {stats['wins']}\n❌ Losses: {stats['losses']}\n🎯 Win Rate: {wr:.1f}%\n\n" f"🟢 Base Wins: {stats['base_wins']}\n🔴 Base Losses: {stats['base_losses']}\n🔁 Recovery Wins: {stats['recovery_wins']}\n🔁 Recovery Losses: {stats['recovery_losses']}\n\n📍 Active Trade: {'YES' if active else 'NO'}\n🔁 Recovery Pending: {'YES' if rec else 'NO'}\n🧭 Last Direction: {last_signal_direction or '-'}\n📈 Direction Streak: {same_direction_streak}\n━━━━━━━━━━━━━━━━━━")
    await update.message.reply_text(msg)

async def reset_command(update,context):
    global stats,history,active_trade,recovery_pending,recovery_wait_until,recovery_number,last_signal_direction,same_direction_streak,last_signal_symbol
    if not await owner_only(update):return
    with state_lock:
        stats={"wins":0,"losses":0,"base_wins":0,"base_losses":0,"recovery_wins":0,"recovery_losses":0,"signals":0}; history=[]; active_trade=None; recovery_pending=False; recovery_wait_until=0; recovery_number=0; last_signal_direction=None; same_direction_streak=0; last_signal_symbol=None; save_history()
    await update.message.reply_text("♻️ تم تصفير الإحصائيات والحالة.")

async def mt5status_command(update,context):
    if not await owner_only(update):return
    with state_lock:
        syms=list(market_data); fresh=sum(is_data_fresh(d) for d in market_data.values()); stale=len(syms)-fresh; batch=latest_batch_id
    await update.message.reply_text(f"🖥️ MT5 STATUS\n━━━━━━━━━━━━━━━━━━\n📊 Symbols received: {len(syms)}\n🟢 Fresh: {fresh}\n🔴 Stale: {stale}\n📦 Batch: {batch or '-'}\n━━━━━━━━━━━━━━━━━━")

class HealthHandler(BaseHTTPRequestHandler):
    def log_message(self,format,*args):return
    def send_json(self,status,payload):
        b=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        path=urlparse(self.path).path
        if path in ("/","/health","/healthz"): self.send_json(200,{"status":"ok","service":"ZinoProSignalAI","timeframe":"M1"}); return
        if path in ("/mt5status","/api/mt5status"): self.send_json(200,{"status":"ok","symbols":len(market_data)}); return
        self.send_json(404,{"error":"not found"})
    def do_POST(self):
        global latest_batch_id,latest_batch_complete
        path=urlparse(self.path).path
        if path not in ("/mt5","/api/mt5","/mt4","/api/mt4"): self.send_json(404,{"error":"not found"}); return
        received=(self.headers.get("X-MT5-API-Key") or self.headers.get("X-MT4-API-Key") or self.headers.get("X-API-Key") or "").strip()
        if not MT5_API_KEY or received!=MT5_API_KEY:
            logger.warning("Unauthorized market-data request"); self.send_json(401,{"error":"unauthorized"}); return
        try:n=int(self.headers.get("Content-Length","0"))
        except:n=0
        if n<=0:self.send_json(400,{"error":"empty body"});return
        if n>10000000:self.send_json(413,{"error":"payload too large"});return
        try: payload=json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:self.send_json(400,{"error":"invalid json","detail":str(e)});return
        symbol=str(payload.get("symbol","")).strip(); tf=str(payload.get("timeframe","")).upper().strip(); batch=str(payload.get("batch_id","")).strip(); candles=payload.get("candles")
        if not symbol:return self.send_json(400,{"error":"missing symbol"})
        if tf!="M1":return self.send_json(400,{"error":"only M1 accepted"})
        if not batch:return self.send_json(400,{"error":"missing batch_id"})
        if not isinstance(candles,list):return self.send_json(400,{"error":"candles must be list"})
        clean=[]
        for x in candles[-MAX_CANDLES:]:
            if isinstance(x,dict):clean.append({"time":safe_int(x.get("time")),"open":safe_float(x.get("open")),"high":safe_float(x.get("high")),"low":safe_float(x.get("low")),"close":safe_float(x.get("close")),"tick_volume":safe_int(x.get("tick_volume")),"real_volume":safe_int(x.get("real_volume")),"spread":safe_int(x.get("spread"))})
        if len(clean)<MIN_CANDLES:return self.send_json(400,{"error":f"need at least {MIN_CANDLES} candles"})
        clean.sort(key=lambda x:x["time"]); current_bid=safe_float(payload.get("current_bid"),clean[-1]["close"]); current_ask=safe_float(payload.get("current_ask"),current_bid); digits=safe_int(payload.get("digits"),5); complete=bool(payload.get("batch_complete",True)); closed=safe_int(payload.get("closed_candle_time"),clean[-1]["time"])
        with state_lock:
            latest_batch_id=batch; latest_batch_complete=complete; market_data[symbol]={"symbol":symbol,"timeframe":"M1","batch_id":batch,"candles":clean,"current_bid":current_bid,"current_ask":current_ask,"digits":digits,"closed_candle_time":closed,"batch_complete":complete,"received_at":now_timestamp()}
        logger.info("MT5 DATA | %s | candles=%s | closed=%s | complete=%s | batch=%s",symbol,len(clean),closed,complete,batch)
        self.send_json(200,{"status":"accepted","symbol":symbol,"batch_id":batch,"closed_candle_time":closed,"batch_complete":complete})

def start_http_server():
    server=ThreadingHTTPServer(("0.0.0.0",PORT),HealthHandler); logger.info("HTTP server listening on port %s",PORT); server.serve_forever()

async def post_init(app):
    global background_task,background_stop_event
    background_stop_event=asyncio.Event(); background_task=asyncio.create_task(background_loop(app,background_stop_event)); logger.info("ZinoProSignalAI background task started")
async def post_stop(app):
    global background_task,background_stop_event
    if background_stop_event: background_stop_event.set()
    if background_task:
        background_task.cancel()
        try: await background_task
        except asyncio.CancelledError: pass
    background_task=None; background_stop_event=None
async def post_shutdown(app): pass

def main():
    if not BOT_TOKEN: raise RuntimeError("BOT_TOKEN is missing")
    if not GEMINI_API_KEY: logger.warning("GEMINI_API_KEY is missing")
    if not MT5_API_KEY: logger.warning("MT5_API_KEY is missing")
    threading.Thread(target=start_http_server,daemon=True).start()
    app=Application.builder().token(BOT_TOKEN).post_init(post_init).post_stop(post_stop).post_shutdown(post_shutdown).build()
    app.add_handler(CommandHandler("start",start_command)); app.add_handler(CommandHandler("win",win_command)); app.add_handler(CommandHandler("loss",loss_command)); app.add_handler(CommandHandler("stats",stats_command)); app.add_handler(CommandHandler("reset",reset_command)); app.add_handler(CommandHandler("mt5status",mt5status_command))
    logger.info("ZinoProSignalAI starting Telegram polling..."); app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__=="__main__": main()
