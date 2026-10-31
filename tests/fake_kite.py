"""A fake `KiteConnect` object on top of the fake market (`oracle.FakeHL`), for testing `KiteBroker`.

It validates the request shapes the real SDK validates (GTT legs need transaction_type / quantity / order_type /
product / price; single GTTs take exactly one trigger value) and models Zerodha's semantics that matter here:
regular orders are DAY orders (`expire_day_orders()` plays the 15:30 expiry), GTTs persist, and a SELL single-GTT
whose trigger is below the creation price is a stop while one above it is a take-profit.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


class FakeKite:
    def __init__(self, hl, symbol="AAA", exchange="NSE", token=4242):
        self.hl, self.symbol, self.exchange, self.token = hl, symbol, exchange, token
        self.kind: dict = {}              # hl oid -> "day" | "gtt"
        self.gtt_last: dict = {}          # hl oid -> last_price given at creation
        self.finished: dict = {}          # order_id -> status for orders that left the book
        self.calls: list = []

    # ---- market data ----
    def instruments(self, exchange=None):
        self.calls.append(("instruments", exchange))
        return [{"tradingsymbol": self.symbol, "exchange": self.exchange, "segment": self.exchange,
                 "instrument_type": "EQ", "instrument_token": self.token},
                {"tradingsymbol": self.symbol + "BEES", "exchange": self.exchange, "segment": self.exchange,
                 "instrument_type": "EQ", "instrument_token": 1}]

    def ltp(self, *instruments):
        return {f"{self.exchange}:{self.symbol}": {"instrument_token": self.token, "last_price": self.hl.price}}

    def historical_data(self, instrument_token, from_date, to_date, interval, continuous=False, oi=False):
        self.calls.append(("historical_data", instrument_token, interval))
        assert instrument_token == self.token
        src = self.hl.candles[{"day": "1d", "60minute": "1h"}[interval]]
        rows = []
        for c in src:
            d = datetime.fromtimestamp(c["t"] / 1000, tz=timezone.utc).astimezone(IST)
            if interval == "day":
                d = d.replace(hour=0, minute=0, second=0, microsecond=0)
            rows.append({"date": d, "open": c["o"], "high": c["h"], "low": c["l"], "close": c["c"], "volume": c["v"]})
        return rows

    # ---- orders ----
    def place_order(self, variety, exchange, tradingsymbol, transaction_type, quantity, product, order_type,
                    price=None, validity=None, trigger_price=None, **kw):
        self.calls.append(("place_order", dict(variety=variety, exchange=exchange, tradingsymbol=tradingsymbol,
                                               transaction_type=transaction_type, quantity=quantity, product=product,
                                               order_type=order_type, price=price, validity=validity)))
        assert variety == "regular" and exchange == self.exchange and tradingsymbol == self.symbol
        assert isinstance(quantity, int) and quantity > 0 and product in ("CNC", "MIS", "NRML")
        assert order_type in ("LIMIT", "MARKET", "SL", "SL-M") and (order_type == "MARKET" or price is not None)
        buy = transaction_type == "BUY"
        r = self.hl.order(f"{self.exchange}:{self.symbol}", buy, quantity, price, {"limit": {"tif": "Gtc"}},
                          reduce_only=not buy)
        st = r["response"]["data"]["statuses"][0]
        if "filled" in st:
            oid = self.hl._oid                      # the order id FakeHL just minted
            self.finished[str(oid)] = ("COMPLETE", quantity)
            return str(oid)
        oid = st["resting"]["oid"]
        if validity == "IOC":                       # immediate-or-cancel: whatever did not fill is cancelled
            self.hl.orders.pop(oid, None)
            self.finished[str(oid)] = ("CANCELLED", 0)
        else:
            self.kind[oid] = "day"
        return str(oid)

    def cancel_order(self, variety, order_id, parent_order_id=None):
        self.calls.append(("cancel_order", variety, order_id))
        self.hl.cancel(None, int(order_id))
        self.finished[str(order_id)] = ("CANCELLED", 0)
        return order_id

    def orders(self):
        rows = []
        for oid, o in self.hl.orders.items():
            if self.kind.get(oid) == "day":
                rows.append({"order_id": str(oid), "tradingsymbol": self.symbol, "exchange": self.exchange,
                             "transaction_type": "BUY" if o.is_buy else "SELL", "order_type": "LIMIT",
                             "status": "OPEN", "price": o.px, "trigger_price": 0, "quantity": int(o.sz),
                             "pending_quantity": int(o.sz), "filled_quantity": 0, "product": "CNC",
                             "validity": "DAY"})
        for oid, (status, q) in self.finished.items():
            rows.append({"order_id": oid, "tradingsymbol": self.symbol, "exchange": self.exchange,
                         "transaction_type": "SELL", "order_type": "LIMIT", "status": status, "price": 0,
                         "trigger_price": 0, "quantity": q, "pending_quantity": 0, "filled_quantity": q,
                         "product": "CNC"})
        return rows

    # ---- GTT ----
    def place_gtt(self, trigger_type, tradingsymbol, exchange, trigger_values, last_price, orders):
        self.calls.append(("place_gtt", dict(trigger_type=trigger_type, trigger_values=trigger_values,
                                             last_price=last_price, orders=orders)))
        assert trigger_type in ("single", "two-leg")
        assert trigger_type == "two-leg" or len(trigger_values) == 1
        for leg in orders:
            for req in ("transaction_type", "quantity", "order_type", "product", "price"):
                assert req in leg, f"`{req}` missing inside orders"
        leg, trig = orders[0], float(trigger_values[0])
        assert leg["transaction_type"] == "SELL"
        coin = f"{exchange}:{tradingsymbol}"
        if trig < last_price:       # stop: when LTP falls to the trigger, a limit sell at leg['price'] goes in
            r = self.hl.order(coin, False, leg["quantity"], leg["price"],
                              {"trigger": {"triggerPx": trig, "isMarket": True, "tpsl": "sl"}}, reduce_only=True)
        else:                       # take-profit: a limit sell that rests until price rises to it
            r = self.hl.order(coin, False, leg["quantity"], leg["price"], {"limit": {"tif": "Gtc"}}, reduce_only=True)
        oid = r["response"]["data"]["statuses"][0]["resting"]["oid"]
        self.kind[oid] = "gtt"
        self.gtt_last[oid] = float(last_price)
        return {"trigger_id": oid}

    def get_gtts(self):
        out = []
        for oid, o in self.hl.orders.items():
            if self.kind.get(oid) != "gtt":
                continue
            trig = o.trigger_px if o.is_trigger else o.px
            out.append({"id": oid, "type": "single", "status": "active",
                        "condition": {"exchange": self.exchange, "tradingsymbol": self.symbol,
                                      "trigger_values": [trig], "last_price": self.gtt_last[oid]},
                        "orders": [{"transaction_type": "SELL", "quantity": int(o.sz), "price": o.px,
                                    "order_type": "LIMIT", "product": "CNC"}]})
        return out

    def delete_gtt(self, trigger_id):
        self.calls.append(("delete_gtt", trigger_id))
        self.hl.cancel(None, int(trigger_id))

    # ---- portfolio ----
    def holdings(self):
        h = self.hl
        if h.pos_size <= 0:
            return []
        return [{"tradingsymbol": self.symbol, "exchange": self.exchange, "quantity": int(h.pos_size),
                 "t1_quantity": 0, "average_price": h.pos_entry, "product": "CNC"}]

    def positions(self):
        return {"net": [], "day": []}

    # ---- test controls ----
    def expire_day_orders(self):
        """The 15:30 expiry of every regular (DAY) order; GTTs survive."""
        for oid in [o for o, k in self.kind.items() if k == "day"]:
            self.hl.orders.pop(oid, None)
            self.kind.pop(oid, None)
