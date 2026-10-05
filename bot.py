import os, json, logging, threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN=os.getenv("BOT_TOKEN","")
OWNER_ID=int(os.getenv("OWNER_ID","0"))
API_KEY=os.getenv("ZINO_API_KEY","CHANGE_ME")
PORT=int(os.getenv("PORT","10000"))
TZ=ZoneInfo("Africa/Algiers")
logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s")
stats={"wins":0,"losses":0}; history=[]; lock=threading.Lock()

def mean(a): return sum(a)/len(a) if a else 0.0
def clamp(x,a,b): return max(a,min(b,x))
def ema(v,p):
    if not v:return 0.0
    k=2/(p+1); e=v[0]
    for x in v[1:]: e=x*k+e*(1-k)
    return e
def atr(c,p=14):
    tr=[]; pc=None
    for x in c:
        if pc is None:t=x['high']-x['low']
        else:t=max(x['high']-x['low'],abs(x['high']-pc),abs(x['low']-pc))
        tr.append(max(t,1e-12)); pc=x['close']
    return mean(tr[-p:])
def rsi(cl,p=14):
    if len(cl)<=p:return 50
    g=[];l=[]
    for i in range(1,len(cl)):
        d=cl[i]-cl[i-1];g.append(max(d,0));l.append(max(-d,0))
    ag,al=mean(g[-p:]),mean(l[-p:])
    return 100 if al==0 and ag>0 else 50 if al==0 else 100-100/(1+ag/al)
def williams(c,p=14):
    w=c[-p:];hh=max(x['high'] for x in w);ll=min(x['low'] for x in w)
    return -50 if hh==ll else -100*(hh-c[-1]['close'])/(hh-ll)
def adx_proxy(c,p=14):
    if len(c)<p+2:return 15,0
    ups=[];dns=[];trs=[];pc=c[-p-1]['close']
    for x in c[-p:]:
        ups.append(max(x['high']-pc,0));dns.append(max(pc-x['low'],0));trs.append(max(x['high']-x['low'],1e-12));pc=x['close']
    pd=mean(ups)/mean(trs)*100;md=mean(dns)/mean(trs)*100
    return 10+clamp(abs(pd-md)*2,0,50),pd-md

def analyze(symbol,tf,c):
    if len(c)<35: raise ValueError("Need at least 35 candles")
    c=sorted(c,key=lambda x:x['time']);cl=[x['close'] for x in c];hi=[x['high'] for x in c];lo=[x['low'] for x in c]
    e9=ema(cl[-80:],9);e21=ema(cl[-80:],21);rr=rsi(cl);wr=williams(c);adx,di=adx_proxy(c);a=atr(c)
    x=c[-1];p=c[-2];up=down=0;ru=[];rd=[]
    rh=max(hi[-8:-1]);rl=min(lo[-8:-1])
    if x['close']>rh:up+=2;ru.append('structure breakout')
    elif x['close']<rl:down+=2;rd.append('structure breakdown')
    else:
        s=mean(cl[-3:])-mean(cl[-8:-3]);up+=1 if s>0 else 0;down+=1 if s<0 else 0
    if x['close']>p['high']:up+=2;ru.append('bullish break')
    elif x['close']<p['low']:down+=2;rd.append('bearish break')
    body=abs(x['close']-x['open']);rng=max(x['high']-x['low'],1e-12);upper=x['high']-max(x['open'],x['close']);lower=min(x['open'],x['close'])-x['low']
    if lower>upper*1.35 and x['close']>x['open']:up+=1;ru.append('lower-wick rejection')
    elif upper>lower*1.35 and x['close']<x['open']:down+=1;rd.append('upper-wick rejection')
    mom=cl[-1]-cl[-4]
    if mom>a*.20:up+=2;ru.append('positive momentum')
    elif mom<-a*.20:down+=2;rd.append('negative momentum')
    elif mom>0:up+=1
    elif mom<0:down+=1
    br=body/rng
    if x['close']>x['open'] and br>=.55:up+=2;ru.append('strong bullish candle')
    elif x['close']<x['open'] and br>=.55:down+=2;rd.append('strong bearish candle')
    elif x['close']>x['open']:up+=1
    else:down+=1
    if 52<=rr<=72:up+=1;ru.append('RSI supports up')
    elif 28<=rr<=48:down+=1;rd.append('RSI supports down')
    if e9>e21 and x['close']>e9:up+=2;ru.append('EMA trend')
    elif e9<e21 and x['close']<e9:down+=2;rd.append('EMA trend')
    elif e9>e21:up+=1
    elif e9<e21:down+=1
    if wr>-50 and rr>50:up+=2;ru.append('oscillators aligned')
    elif wr<-50 and rr<50:down+=2;rd.append('oscillators aligned')
    elif rr>50:up+=1
    elif rr<50:down+=1
    if e9>e21:up+=2
    elif e9<e21:down+=2
    if adx>=20:
        if di>2:up+=2;ru.append('ADX/DI strength')
        elif di<-2:down+=2;rd.append('ADX/DI strength')
        elif up>=down:up+=1
        else:down+=1
    else:
        if up>=down:up+=.5
        else:down+=.5
    direction='UP' if up>=down else 'DOWN';total=up+down;edge=abs(up-down)
    conf=55+(edge/max(total,1))*30
    if adx<18:conf-=6
    if br<.35:conf-=3
    with lock: recent=[h['direction'] for h in history[-6:]]
    if recent.count(direction)>=5:conf-=7
    conf=int(round(clamp(conf,55,89)))
    delay={'M1':1,'M2':2,'M3':3}.get(tf.upper(),1);now=datetime.now(TZ);entry=now+timedelta(minutes=delay)
    cancel=x['close']-a*.35 if direction=='UP' else x['close']+a*.35
    s={'symbol':symbol,'timeframe':tf,'direction':direction,'confidence':conf,'up_score':round(up,1),'down_score':round(down,1),'entry_after':delay,'entry_time':entry.strftime('%Y-%m-%d %H:%M:%S'),'entry_price':x['close'],'cancel_price':cancel,'reason':', '.join((ru if direction=='UP' else rd)[:4]) or 'mixed market structure','generated_at':now.strftime('%Y-%m-%d %H:%M:%S')}
    with lock: history.append(s);del history[:-100]
    return s

def fmt(s):
    arrow='🟢 UP' if s['direction']=='UP' else '🔴 DOWN'
    cancel=(f"إلغاء إذا أغلقت الشمعة تحت {s['cancel_price']:.8f}" if s['direction']=='UP' else f"إلغاء إذا أغلقت الشمعة فوق {s['cancel_price']:.8f}")
    return ("🎓 ZinoProSignalAI\n━━━━━━━━━━━━━━━━━━\n"+f"📊 {s['symbol']} | {s['timeframe']}\n\n🎯 BASE TRADE\n{arrow}\n\n"+f"🔥 Confidence: {s['confidence']}%\n🟢 UP Score: {s['up_score']}/18\n🔴 DOWN Score: {s['down_score']}/18\n\n⏱️ Entry after: {s['entry_after']} minute(s)\n🕐 ENTRY TIME: {s['entry_time']} (Algiers)\n💰 ENTRY PRICE: {s['entry_price']:.8f}\n⚠️ {cancel}\n\n🧠 {s['reason']}")

class H(BaseHTTPRequestHandler):
    def sendj(self,code,obj):
        b=json.dumps(obj).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def do_GET(self): self.sendj(200,{'status':'ok','service':'ZinoProSignalAI MT4'}) if self.path in ('/','/health') else self.sendj(404,{'error':'not found'})
    def do_POST(self):
        if self.path!='/mt4':return self.sendj(404,{'error':'not found'})
        try:
            n=int(self.headers.get('Content-Length','0'));d=json.loads(self.rfile.read(n).decode())
            if d.get('api_key')!=API_KEY:return self.sendj(401,{'error':'unauthorized'})
            s=analyze(str(d.get('symbol','UNKNOWN')),str(d.get('timeframe','M1')).upper(),d.get('candles',[]));self.sendj(200,{'ok':True,'signal':s,'telegram_text':fmt(s)})
        except Exception as e: logging.exception('MT4 request failed');self.sendj(400,{'ok':False,'error':str(e)})
    def log_message(self,f,*a):logging.info('HTTP '+f,*a)
def http():ThreadingHTTPServer(('0.0.0.0',PORT),H).serve_forever()
async def owner(update): return update.effective_user and update.effective_user.id==OWNER_ID
async def start(u,c):
    if await owner(u): await u.message.reply_text('🎓 ZinoProSignalAI MT4\n✅ Online\n🌙 OTC/Night ready\n📊 /stats /win /loss /reset')
async def stats_cmd(u,c):
    if not await owner(u):return
    with lock:w,l=stats['wins'],stats['losses']
    t=w+l;await u.message.reply_text(f'📚 HISTORY\n🟢 WIN: {w}\n🔴 LOSS: {l}\n📊 TOTAL: {t}\n🎯 Accuracy: {(w/t*100 if t else 0):.1f}%')
async def win(u,c):
    if await owner(u):
        with lock:stats['wins']+=1
        await u.message.reply_text('🟢 WIN recorded')
async def loss(u,c):
    if await owner(u):
        with lock:stats['losses']+=1
        await u.message.reply_text('🔴 LOSS recorded')
async def reset(u,c):
    if await owner(u):
        with lock:stats['wins']=stats['losses']=0;history.clear()
        await u.message.reply_text('♻️ Reset done')
def main():
    threading.Thread(target=http,daemon=True).start();logging.info('ZinoProSignalAI MT4 HTTP online on %s',PORT)
    if not BOT_TOKEN:return threading.Event().wait()
    app=Application.builder().token(BOT_TOKEN).build();app.add_handler(CommandHandler('start',start));app.add_handler(CommandHandler('stats',stats_cmd));app.add_handler(CommandHandler('win',win));app.add_handler(CommandHandler('loss',loss));app.add_handler(CommandHandler('reset',reset));app.run_polling(drop_pending_updates=True)
if __name__=='__main__':main()
