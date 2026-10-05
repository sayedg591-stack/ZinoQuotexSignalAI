#property strict
#property version "1.0"
input string RenderURL="https://YOUR-RENDER-SERVICE.onrender.com/mt4";
input string API_KEY="CHANGE_ME";
input ENUM_TIMEFRAMES SignalTimeframe=PERIOD_M1;
input int CandleCount=80;
input int SendEverySeconds=60;
input bool SendOnlyOnNewCandle=true;
input string SymbolsCSV="";
datetime lastBar=0,lastSend=0;
string TF(ENUM_TIMEFRAMES t){if(t==PERIOD_M1)return "M1";if(t==PERIOD_M2)return "M2";if(t==PERIOD_M3)return "M3";if(t==PERIOD_M5)return "M5";if(t==PERIOD_M15)return "M15";if(t==PERIOD_M30)return "M30";if(t==PERIOD_H1)return "H1";return "M1";}
string Esc(string s){StringReplace(s,"\\","\\\\");StringReplace(s,"\"","\\\"");return s;}
bool Candles(string sym,string &out){if(iBars(sym,SignalTimeframe)<CandleCount+5)return false;out="[";for(int i=CandleCount-1;i>=0;i--){if(i<CandleCount-1)out+=",";out+="{\"time\":"+IntegerToString((int)iTime(sym,SignalTimeframe,i))+",\"open\":"+DoubleToString(iOpen(sym,SignalTimeframe,i),Digits)+",\"high\":"+DoubleToString(iHigh(sym,SignalTimeframe,i),Digits)+",\"low\":"+DoubleToString(iLow(sym,SignalTimeframe,i),Digits)+",\"close\":"+DoubleToString(iClose(sym,SignalTimeframe,i),Digits)+"}";}out+="]";return true;}
bool Send(string sym){string c;if(!Candles(sym,c)){Print("Not enough bars: ",sym);return false;}string body="{\"api_key\":\""+Esc(API_KEY)+"\",\"symbol\":\""+Esc(sym)+"\",\"timeframe\":\""+TF(SignalTimeframe)+"\",\"candles\":"+c+"}";char data[],res[];StringToCharArray(body,data,0,StringLen(body),CP_UTF8);string hdr="Content-Type: application/json\r\n",rh="";ResetLastError();int code=WebRequest("POST",RenderURL,hdr,15000,data,res,rh);if(code<0){Print("WEBREQUEST ERROR: ",GetLastError()," | Add Render URL in Tools > Options > Expert Advisors");return false;}Print("HTTP CODE: ",code);Print("SIGNAL RESPONSE: ",CharArrayToString(res,0,-1,CP_UTF8));return code>=200&&code<300;}
void SendAll(){if(StringTrimLeft(StringTrimRight(SymbolsCSV))==""){Send(Symbol());return;}string a[];int n=StringSplit(SymbolsCSV,',',a);for(int i=0;i<n;i++){string s=StringTrimLeft(StringTrimRight(a[i]));if(s!=""){SymbolSelect(s,true);Send(s);}}}
int OnInit(){Print("ZinoProSignalAI MT4 EA started | ",Symbol()," | ",TF(SignalTimeframe));return INIT_SUCCEEDED;}
void OnTick(){datetime b=iTime(Symbol(),SignalTimeframe,0);if(SendOnlyOnNewCandle){if(b==lastBar)return;lastBar=b;SendAll();}else if(TimeCurrent()-lastSend>=SendEverySeconds){lastSend=TimeCurrent();SendAll();}}
