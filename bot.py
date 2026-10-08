 //+------------------------------------------------------------------+
//|                                             ZinoM1_EMA_RSI.mq5   |
//|                        M1 Short-Term Expert Advisor               |
//+------------------------------------------------------------------+
#property copyright "ZinoProSignalAI"
#property version   "1.00"
#property description "M1 EMA 9/21 + RSI momentum Expert Advisor"
#property description "Uses only standard MetaTrader 5 indicators/functions."
#property strict

#include <Trade/Trade.mqh>

//====================================================================
// TRADE OBJECT
//====================================================================
CTrade trade;

//====================================================================
// INPUT PARAMETERS
//====================================================================

//--- Strategy settings
input int      InpFastEMA              = 9;       // Fast EMA period
input int      InpSlowEMA              = 21;      // Slow EMA period
input int      InpRSIPeriod             = 14;      // RSI period
input double   InpRSIBuyLevel           = 50.0;    // Minimum RSI for BUY
input double   InpRSISellLevel          = 50.0;    // Maximum RSI for SELL

//--- Risk management
input double   InpRiskPercent           = 1.0;     // Risk per trade (% equity)
input double   InpStopLossPips          = 10.0;    // Stop loss in pips
input double   InpTakeProfitPips        = 15.0;    // Take profit in pips

//--- Trading restrictions
input bool     InpAllowBuy             = true;    // Allow BUY trades
input bool     InpAllowSell            = true;    // Allow SELL trades
input bool     InpOnePositionOnly      = true;    // Only one EA position
input int      InpMaxSpreadPoints      = 30;      // Maximum spread in points
input ulong    InpMagicNumber           = 26081001;// EA magic number

//--- Execution settings
input int      InpDeviationPoints      = 10;      // Maximum price deviation
input bool     InpUseTrailingStop      = false;   // Enable trailing stop
input double   InpTrailingStartPips    = 8.0;     // Start trailing after X pips
input double   InpTrailingDistancePips = 5.0;     // Trailing distance

//====================================================================
// GLOBAL VARIABLES
//====================================================================

int      g_fastEMAHandle = INVALID_HANDLE;
int      g_slowEMAHandle = INVALID_HANDLE;
int      g_rsiHandle     = INVALID_HANDLE;

datetime g_lastBarTime   = 0;

//====================================================================
// HELPER: POINT VALUE
//====================================================================

double PipSize()
{
   /*
      Forex symbols normally have either:
      - 5 digits: 1 pip = 10 points
      - 3 digits: 1 pip = 10 points
      - 4/2 digits: 1 pip = 1 point

      This function converts a pip into the corresponding
      price distance.
   */

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   if(digits == 3 || digits == 5)
      return _Point * 10.0;

   return _Point;
}

//====================================================================
// HELPER: CHECK NEW M1 BAR
//====================================================================

bool IsNewBar()
{
   datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);

   if(currentBarTime <= 0)
      return false;

   if(currentBarTime != g_lastBarTime)
   {
      g_lastBarTime = currentBarTime;
      return true;
   }

   return false;
}

//====================================================================
// HELPER: GET CURRENT SPREAD
//====================================================================

double CurrentSpreadPoints()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(ask <= 0 || bid <= 0)
      return 999999.0;

   return (ask - bid) / _Point;
}

//====================================================================
// HELPER: CHECK WHETHER EA ALREADY HAS A POSITION
//====================================================================

bool HasOurPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);

      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      ulong magic   = (ulong)PositionGetInteger(POSITION_MAGIC);

      if(symbol == _Symbol && magic == InpMagicNumber)
         return true;
   }

   return false;
}

//====================================================================
// HELPER: NORMALIZE VOLUME
//====================================================================

double NormalizeVolume(double volume)
{
   double minVolume  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxVolume  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volumeStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(volumeStep <= 0)
      return 0.0;

   //--- Clamp volume to broker limits
   volume = MathMax(volume, minVolume);
   volume = MathMin(volume, maxVolume);

   //--- Round down to valid volume step
   volume = MathFloor(volume / volumeStep) * volumeStep;

   int volumeDigits = 2;

   if(volumeStep == 1.0)
      volumeDigits = 0;
   else if(volumeStep == 0.1)
      volumeDigits = 1;
   else if(volumeStep == 0.01)
      volumeDigits = 2;

   return NormalizeDouble(volume, volumeDigits);
}

//====================================================================
// CALCULATE POSITION SIZE BASED ON RISK
//====================================================================

double CalculateRiskVolume(double stopLossDistance)
{
   /*
      The goal is to risk approximately InpRiskPercent of equity.

      Example:
         Equity = $100
         Risk = 1%
         Maximum loss = $1

      The EA determines how much volume corresponds to that
      maximum loss using the symbol's tick value and tick size.

      This is much safer than simply using a fixed lot size.
   */

   if(stopLossDistance <= 0)
      return 0.0;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   if(equity <= 0)
      return 0.0;

   double riskMoney = equity * (InpRiskPercent / 100.0);

   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);

   if(tickSize <= 0 || tickValue <= 0)
   {
      Print("ERROR: Invalid tick size/value for ", _Symbol);
      return 0.0;
   }

   /*
      Money lost by one lot if price moves by stopLossDistance.
   */
   double lossPerLot =
      (stopLossDistance / tickSize) * tickValue;

   if(lossPerLot <= 0)
      return 0.0;

   double volume = riskMoney / lossPerLot;

   return NormalizeVolume(volume);
}

//====================================================================
// CHECK BROKER STOP DISTANCE
//====================================================================

double MinimumStopDistance()
{
   long stopsLevel =
      SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);

   return (double)stopsLevel * _Point;
}

//====================================================================
// OPEN BUY
//====================================================================

bool OpenBuy()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(ask <= 0)
   {
      Print("BUY ERROR: Invalid ASK price.");
      return false;
   }

   double pip = PipSize();

   double requestedSLDistance =
      InpStopLossPips * pip;

   double requestedTPDistance =
      InpTakeProfitPips * pip;

   //--- Make sure SL/TP satisfy broker minimum distance
   double minimumDistance = MinimumStopDistance();

   double slDistance =
      MathMax(requestedSLDistance, minimumDistance);

   double tpDistance =
      MathMax(requestedTPDistance, minimumDistance);

   double volume =
      CalculateRiskVolume(slDistance);

   if(volume <= 0)
   {
      Print("BUY ERROR: Calculated volume is invalid.");
      return false;
   }

   double sl = ask - slDistance;
   double tp = ask + tpDistance;

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   bool result =
      trade.Buy(
         volume,
         _Symbol,
         0.0,
         sl,
         tp,
         "ZinoM1 EMA RSI BUY"
      );

   if(!result)
   {
      Print(
         "BUY FAILED | Retcode=",
         trade.ResultRetcode(),
         " | ",
         trade.ResultRetcodeDescription()
      );

      return false;
   }

   Print(
      "BUY OPENED | Symbol=",
      _Symbol,
      " | Volume=",
      volume,
      " | SL=",
      sl,
      " | TP=",
      tp
   );

   return true;
}

//====================================================================
// OPEN SELL
//====================================================================

bool OpenSell()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

   if(bid <= 0)
   {
      Print("SELL ERROR: Invalid BID price.");
      return false;
   }

   double pip = PipSize();

   double requestedSLDistance =
      InpStopLossPips * pip;

   double requestedTPDistance =
      InpTakeProfitPips * pip;

   double minimumDistance = MinimumStopDistance();

   double slDistance =
      MathMax(requestedSLDistance, minimumDistance);

   double tpDistance =
      MathMax(requestedTPDistance, minimumDistance);

   double volume =
      CalculateRiskVolume(slDistance);

   if(volume <= 0)
   {
      Print("SELL ERROR: Calculated volume is invalid.");
      return false;
   }

   double sl = bid + slDistance;
   double tp = bid - tpDistance;

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   bool result =
      trade.Sell(
         volume,
         _Symbol,
         0.0,
         sl,
         tp,
         "ZinoM1 EMA RSI SELL"
      );

   if(!result)
   {
      Print(
         "SELL FAILED | Retcode=",
         trade.ResultRetcode(),
         " | ",
         trade.ResultRetcodeDescription()
      );

      return false;
   }

   Print(
      "SELL OPENED | Symbol=",
      _Symbol,
      " | Volume=",
      volume,
      " | SL=",
      sl,
      " | TP=",
      tp
   );

   return true;
}

//====================================================================
// TRAILING STOP
//====================================================================

void ManageTrailingStop()
{
   if(!InpUseTrailingStop)
      return;

   double pip = PipSize();

   double startDistance =
      InpTrailingStartPips * pip;

   double trailingDistance =
      InpTrailingDistancePips * pip;

   if(startDistance <= 0 || trailingDistance <= 0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);

      if(ticket == 0)
         continue;

      if(!PositionSelectByTicket(ticket))
         continue;

      string symbol = PositionGetString(POSITION_SYMBOL);
      ulong magic   = (ulong)PositionGetInteger(POSITION_MAGIC);

      if(symbol != _Symbol || magic != InpMagicNumber)
         continue;

      ENUM_POSITION_TYPE type =
         (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      double openPrice =
         PositionGetDouble(POSITION_PRICE_OPEN);

      double currentSL =
         PositionGetDouble(POSITION_SL);

      double currentTP =
         PositionGetDouble(POSITION_TP);

      double bid =
         SymbolInfoDouble(_Symbol, SYMBOL_BID);

      double ask =
         SymbolInfoDouble(_Symbol, SYMBOL_ASK);

      //==============================================================
      // BUY TRAILING STOP
      //==============================================================
      if(type == POSITION_TYPE_BUY)
      {
         double profitDistance =
            bid - openPrice;

         if(profitDistance < startDistance)
            continue;

         double newSL =
            bid - trailingDistance;

         newSL =
            NormalizeDouble(
               newSL,
               (int)SymbolInfoInteger(
                  _Symbol,
                  SYMBOL_DIGITS
               )
            );

         /*
            Only move SL upward.
         */
         if(currentSL == 0.0 || newSL > currentSL)
         {
            if(!trade.PositionModify(
                  ticket,
                  newSL,
                  currentTP))
            {
               Print(
                  "Trailing BUY failed | ",
                  trade.ResultRetcodeDescription()
               );
            }
         }
      }

      //==============================================================
      // SELL TRAILING STOP
      //==============================================================
      if(type == POSITION_TYPE_SELL)
      {
         double profitDistance =
            openPrice - ask;

         if(profitDistance < startDistance)
            continue;

         double newSL =
            ask + trailingDistance;

         newSL =
            NormalizeDouble(
               newSL,
               (int)SymbolInfoInteger(
                  _Symbol,
                  SYMBOL_DIGITS
               )
            );

         /*
            Only move SL downward.
         */
         if(currentSL == 0.0 || newSL < currentSL)
         {
            if(!trade.PositionModify(
                  ticket,
                  newSL,
                  currentTP))
            {
               Print(
                  "Trailing SELL failed | ",
                  trade.ResultRetcodeDescription()
               );
            }
         }
      }
   }
}

//====================================================================
// INITIALIZATION
//====================================================================

int OnInit()
{
   Print("=================================================");
   Print("ZinoM1 EMA + RSI EA starting...");
   Print("Symbol: ", _Symbol);
   Print("Timeframe required: M1");
   Print("=================================================");

   //--- Force strategy to M1
   if(_Period != PERIOD_M1)
   {
      Print(
         "WARNING: EA is designed for M1. ",
         "Current chart timeframe is not M1."
      );
   }

   //--- Validate inputs
   if(InpFastEMA <= 0 ||
      InpSlowEMA <= 0 ||
      InpRSIPeriod <= 0)
   {
      Print("ERROR: Invalid indicator periods.");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(InpFastEMA >= InpSlowEMA)
   {
      Print(
         "ERROR: Fast EMA must be smaller than Slow EMA."
      );

      return INIT_PARAMETERS_INCORRECT;
   }

   if(InpRiskPercent <= 0)
   {
      Print("ERROR: Risk percentage must be > 0.");
      return INIT_PARAMETERS_INCORRECT;
   }

   if(InpStopLossPips <= 0 ||
      InpTakeProfitPips <= 0)
   {
      Print("ERROR: SL and TP must be > 0.");
      return INIT_PARAMETERS_INCORRECT;
   }

   //================================================================
   // CREATE STANDARD MT5 INDICATORS
   //================================================================

   g_fastEMAHandle =
      iMA(
         _Symbol,
         PERIOD_M1,
         InpFastEMA,
         0,
         MODE_EMA,
         PRICE_CLOSE
      );

   if(g_fastEMAHandle == INVALID_HANDLE)
   {
      Print(
         "ERROR: Failed to create Fast EMA handle. ",
         "Error=", GetLastError()
      );

      return INIT_FAILED;
   }

   g_slowEMAHandle =
      iMA(
         _Symbol,
         PERIOD_M1,
         InpSlowEMA,
         0,
         MODE_EMA,
         PRICE_CLOSE
      );

   if(g_slowEMAHandle == INVALID_HANDLE)
   {
      Print(
         "ERROR: Failed to create Slow EMA handle. ",
         "Error=", GetLastError()
      );

      return INIT_FAILED;
   }

   g_rsiHandle =
      iRSI(
         _Symbol,
         PERIOD_M1,
         InpRSIPeriod,
         PRICE_CLOSE
      );

   if(g_rsiHandle == INVALID_HANDLE)
   {
      Print(
         "ERROR: Failed to create RSI handle. ",
         "Error=", GetLastError()
      );

      return INIT_FAILED;
   }

   //================================================================
   // CONFIGURE TRADE OBJECT
   //================================================================

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);

   /*
      This asks MT5 to automatically select a suitable
      filling mode for the broker/symbol.
   */
   trade.SetTypeFillingBySymbol(_Symbol);

   //--- Initialize bar time
   g_lastBarTime =
      iTime(_Symbol, PERIOD_M1, 0);

   Print("Initialization completed successfully.");

   return INIT_SUCCEEDED;
}

//====================================================================
// DEINITIALIZATION
//====================================================================

void OnDeinit(const int reason)
{
   /*
      Release indicator handles to avoid unnecessary resource usage.
   */

   if(g_fastEMAHandle != INVALID_HANDLE)
      IndicatorRelease(g_fastEMAHandle);

   if(g_slowEMAHandle != INVALID_HANDLE)
      IndicatorRelease(g_slowEMAHandle);

   if(g_rsiHandle != INVALID_HANDLE)
      IndicatorRelease(g_rsiHandle);

   Print(
      "ZinoM1 EMA + RSI EA stopped. Reason=",
      reason
   );
}

//====================================================================
// MAIN TICK FUNCTION
//====================================================================

void OnTick()
{
   /*
      OnTick() is called whenever a new market tick arrives.

      Important optimization:
      We do NOT perform the full strategy calculation on every tick.

      Instead:
      1. Manage an existing trailing stop if enabled.
      2. Check whether a NEW M1 candle has appeared.
      3. Only then evaluate a new trading signal.

      This greatly reduces unnecessary calculations.
   */

   //--- Manage existing position
   ManageTrailingStop();

   //--- Strategy is evaluated only once per new M1 candle
   if(!IsNewBar())
      return;

   //================================================================
   // SPREAD FILTER
   //================================================================

   double spread =
      CurrentSpreadPoints();

   if(spread > InpMaxSpreadPoints)
   {
      Print(
         "Signal skipped: spread too high. ",
         "Spread=",
         DoubleToString(spread, 1),
         " points"
      );

      return;
   }

   //================================================================
   // POSITION FILTER
   //================================================================

   if(InpOnePositionOnly && HasOurPosition())
   {
      /*
         We don't open another position while an EA position
         already exists on this symbol.
      */

      return;
   }

   //================================================================
   // READ INDICATOR DATA
   //================================================================

   double fastEMA[3];
   double slowEMA[3];
   double rsi[3];

   ArraySetAsSeries(fastEMA, true);
   ArraySetAsSeries(slowEMA, true);
   ArraySetAsSeries(rsi, true);

   /*
      We need:
         index 0 = current candle
         index 1 = last CLOSED candle
         index 2 = candle before that

      The actual signal is generated using CLOSED candles.
      This prevents the EA from reacting to an unfinished candle.
   */

   if(CopyBuffer(
         g_fastEMAHandle,
         0,
         0,
         3,
         fastEMA) != 3)
   {
      Print(
         "ERROR: Failed to read Fast EMA. ",
         "Error=",
         GetLastError()
      );

      return;
   }

   if(CopyBuffer(
         g_slowEMAHandle,
         0,
         0,
         3,
         slowEMA) != 3)
   {
      Print(
         "ERROR: Failed to read Slow EMA. ",
         "Error=",
         GetLastError()
      );

      return;
   }

   if(CopyBuffer(
         g_rsiHandle,
         0,
         0,
         3,
         rsi) != 3)
   {
      Print(
         "ERROR: Failed to read RSI. ",
         "Error=",
         GetLastError()
      );

      return;
   }

   //================================================================
   // READ PRICE DATA
   //================================================================

   MqlRates rates[3];

   ArraySetAsSeries(rates, true);

   if(CopyRates(
         _Symbol,
         PERIOD_M1,
         0,
         3,
         rates) != 3)
   {
      Print(
         "ERROR: Failed to read M1 price data. ",
         "Error=",
         GetLastError()
      );

      return;
   }

   //================================================================
   // PREVIOUS CLOSED CANDLE
   //================================================================

   double previousOpen  = rates[1].open;
   double previousClose = rates[1].close;

   bool previousBullish =
      previousClose > previousOpen;

   bool previousBearish =
      previousClose < previousOpen;

   //================================================================
   // CROSSOVER DETECTION
   //================================================================

   /*
      BUY crossover:

         Previous older candle:
             Fast EMA <= Slow EMA

         Last closed candle:
             Fast EMA > Slow EMA

      SELL crossover:

         Previous older candle:
             Fast EMA >= Slow EMA

         Last closed candle:
             Fast EMA < Slow EMA
   */

   bool bullishCross =
      fastEMA[2] <= slowEMA[2] &&
      fastEMA[1] > slowEMA[1];

   bool bearishCross =
      fastEMA[2] >= slowEMA[2] &&
      fastEMA[1] < slowEMA[1];

   //================================================================
   // RSI CONFIRMATION
   //================================================================

   bool buyRSI =
      rsi[1] > InpRSIBuyLevel;

   bool sellRSI =
      rsi[1] < InpRSISellLevel;

   //================================================================
   // FINAL SIGNAL
   //================================================================

   bool buySignal =
      bullishCross &&
      buyRSI &&
      previousBullish;

   bool sellSignal =
      bearishCross &&
      sellRSI &&
      previousBearish;

   //================================================================
   // LOG SIGNAL INFORMATION
   //================================================================

   Print(
      "M1 ANALYSIS | ",
      _Symbol,
      " | FastEMA=",
      DoubleToString(fastEMA[1], 6),
      " | SlowEMA=",
      DoubleToString(slowEMA[1], 6),
      " | RSI=",
      DoubleToString(rsi[1], 2),
      " | BullishCross=",
      bullishCross,
      " | BearishCross=",
      bearishCross
   );

   //================================================================
   // EXECUTE BUY
   //================================================================

   if(buySignal && InpAllowBuy)
   {
      Print(
         "BUY SIGNAL CONFIRMED | ",
         _Symbol,
         " | RSI=",
         DoubleToString(rsi[1], 2)
      );

      OpenBuy();

      return;
   }

   //================================================================
   // EXECUTE SELL
   //================================================================

   if(sellSignal && InpAllowSell)
   {
      Print(
         "SELL SIGNAL CONFIRMED | ",
         _Symbol,
         " | RSI=",
         DoubleToString(rsi[1], 2)
      );

      OpenSell();

      return;
   }

   //================================================================
   // NO SIGNAL
   //================================================================

   Print(
      "No valid M1 signal on ",
      _Symbol
   );
}
//+------------------------------------------------------------------+
