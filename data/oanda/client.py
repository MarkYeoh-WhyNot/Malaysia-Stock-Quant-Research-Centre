import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import aiohttp
from config.settings import OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_BASE_URL, DEFAULT_PAIRS

logger = logging.getLogger(__name__)
HEADERS = {"Authorization": f"Bearer {OANDA_API_KEY}", "Content-Type": "application/json"}

class OANDAClient:
    def __init__(self):
        self.base_url   = OANDA_BASE_URL
        self.account_id = OANDA_ACCOUNT_ID
        self._session   = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(headers=HEADERS)
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.close()

    async def _get(self, url, params=None):
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, url, payload):
        async with self._session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_candles(self, instrument, granularity="H1", count=500, from_dt=None, to_dt=None):
        url    = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {"granularity": granularity, "price": "M"}
        if from_dt and to_dt:
            params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params["to"]   = to_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            params["count"] = min(count, 5000)
        data    = await self._get(url, params)
        candles = []
        for c in data.get("candles", []):
            if c.get("complete", True):
                mid = c.get("mid", {})
                candles.append({
                    "time":   c["time"],
                    "open":   float(mid.get("o", 0)),
                    "high":   float(mid.get("h", 0)),
                    "low":    float(mid.get("l", 0)),
                    "close":  float(mid.get("c", 0)),
                    "volume": int(c.get("volume", 0)),
                })
        return candles

    async def get_latest_price(self, instrument):
        url    = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        data   = await self._get(url, {"instruments": instrument})
        prices = data.get("prices", [{}])[0]
        bid    = float(prices.get("bids", [{}])[0].get("price", 0))
        ask    = float(prices.get("asks", [{}])[0].get("price", 0))
        return {"instrument": instrument, "bid": bid, "ask": ask, "mid": round((bid+ask)/2, 5)}

    async def get_multi_prices(self, instruments=None):
        pairs  = instruments or DEFAULT_PAIRS
        url    = f"{self.base_url}/v3/accounts/{self.account_id}/pricing"
        data   = await self._get(url, {"instruments": ",".join(pairs)})
        result = []
        for p in data.get("prices", []):
            bid = float(p.get("bids", [{}])[0].get("price", 0))
            ask = float(p.get("asks", [{}])[0].get("price", 0))
            result.append({
                "instrument":  p.get("instrument"),
                "bid":         bid, "ask": ask,
                "mid":         round((bid+ask)/2, 5),
                "spread_pips": round((ask-bid)*10000, 2),
            })
        return result

    async def get_account_summary(self):
        url  = f"{self.base_url}/v3/accounts/{self.account_id}/summary"
        data = await self._get(url)
        acct = data.get("account", {})
        return {
            "balance":          float(acct.get("balance", 0)),
            "nav":              float(acct.get("NAV", 0)),
            "unrealized_pnl":   float(acct.get("unrealizedPL", 0)),
            "margin_used":      float(acct.get("marginUsed", 0)),
            "margin_available": float(acct.get("marginAvailable", 0)),
            "open_trade_count": int(acct.get("openTradeCount", 0)),
        }

    async def get_open_trades(self):
        url  = f"{self.base_url}/v3/accounts/{self.account_id}/openTrades"
        data = await self._get(url)
        return data.get("trades", [])

    async def place_market_order(self, instrument, units, stop_loss_pips=None, take_profit_pips=None):
        url   = f"{self.base_url}/v3/accounts/{self.account_id}/orders"
        order = {"order": {"type": "MARKET", "instrument": instrument, "units": str(units)}}
        if stop_loss_pips:
            price     = await self.get_latest_price(instrument)
            direction = 1 if units > 0 else -1
            pip_size  = 0.0001 if "JPY" not in instrument else 0.01
            sl_price  = price["mid"] - direction * stop_loss_pips * pip_size
            order["order"]["stopLossOnFill"] = {"price": f"{sl_price:.5f}"}
        return await self._post(url, order)

    async def close_trade(self, trade_id):
        url = f"{self.base_url}/v3/accounts/{self.account_id}/trades/{trade_id}/close"
        async with self._session.put(url) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_historical_data(self, instrument, granularity, days=365):
        all_candles = []
        to_dt   = datetime.utcnow()
        from_dt = to_dt - timedelta(days=days)
        batch   = {"M1":.35,"M5":1.7,"M15":5,"M30":10,"H1":20,"H4":83,"D":500}
        chunk   = timedelta(days=batch.get(granularity, 20))
        current = from_dt
        while current < to_dt:
            end = min(current + chunk, to_dt)
            try:
                candles = await self.get_candles(instrument, granularity, from_dt=current, to_dt=end)
                all_candles.extend(candles)
            except Exception as e:
                logger.warning(f"Candle fetch error {instrument} {current}: {e}")
            current = end
            await asyncio.sleep(0.2)
        logger.info(f"Fetched {len(all_candles)} candles for {instrument} ({granularity}, {days}d)")
        return all_candles
