ZinoProSignalAI MT4 - fresh installation

1) Upload these files to Render. Environment variables:
BOT_TOKEN = Telegram bot token
OWNER_ID = your Telegram numeric ID
ZINO_API_KEY = any long secret, e.g. ZinoMT4-2026-ChangeMe-9xQ7
PORT = 10000

2) After Render deploys, copy the Render service URL into EA input RenderURL and append /mt4.
Example: https://your-service.onrender.com/mt4
Put the same ZINO_API_KEY in EA input API_KEY.

3) MT4: File > Open Data Folder > MQL4 > Experts. Put ZinoProSignalAI_MT4.mq4 there. Open MetaEditor and Compile.

4) MT4: Tools > Options > Expert Advisors > Allow WebRequest for listed URL. Add the base Render URL (without /mt4), e.g. https://your-service.onrender.com

5) Attach EA to any chart. SignalTimeframe=M1. Leave SymbolsCSV empty for current chart, or put exact broker symbols separated by commas.

Important: OTC names are broker-specific. If the broker calls a symbol EURUSD-OTC, use exactly that. If it calls EURUSD_otc, use that exact name. Do not invent suffixes.

The engine always chooses UP or DOWN. It lowers confidence when evidence is weak and reduces confidence after a long same-direction streak; it does not force UP/DOWN alternation.
