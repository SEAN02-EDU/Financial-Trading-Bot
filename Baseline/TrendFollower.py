import random
import numpy as np
import pandas as pd
import yfinance as yf
from prettytable import PrettyTable

class Market:
    def __init__(
        self,
        tickers=["TSM", "TSLA", "AMZN"],
        period: str  = None,
        start:  str  = None,
        end:    str  = None,
        interval: str = "1d",
    ):
        """
        If start & end are provided, fetch by date range; otherwise use period.
        """
        self.tickers      = tickers
        self.period       = period
        self.start        = start
        self.end          = end
        self.interval     = interval
        self.history      = {}
        self.dates        = {}
        self.current_step = -1
        self.price        = 0.0

        for t in self.tickers:
            closes, dates = self.get_stock_prices(t)
            self.history[t] = closes
            self.dates[t]   = dates

        first = self.tickers[0]
        if self.history[first]:
            self.price = self.history[first][0]

    def get_stock_prices(self, ticker: str):
        stock = yf.Ticker(ticker)
        if self.start and self.end:
            data = stock.history(start=self.start, end=self.end, interval=self.interval)
        elif self.period:
            data = stock.history(period=self.period, interval=self.interval)
        else:
            raise ValueError("Must specify either a period or both start & end dates")

        if data.empty:
            print(f"Error: no data for {ticker}")
            return [], []

        return data["Close"].tolist(), data.index.strftime("%Y-%m-%d").tolist()

    def update_price(self):
        t    = self.tickers[0]
        hist = self.history.get(t, [])
        if not hist:
            return
        if self.current_step + 1 < len(hist):
            self.current_step += 1
        self.price = hist[self.current_step]
        date = self.dates[t][self.current_step]
        print(f"[{t}] Step {self.current_step + 1}: price = {self.price:.2f} on {date}")

class TrendFollowerAgent:
    def __init__(self, name="TrendFollower", initial_balance=1000.0):
        self.name     = name
        self.balance  = initial_balance
        self.holdings = 0

    def decide(self, market: Market):
        t      = market.tickers[0]
        step   = market.current_step
        # on the very first step we have no previous price, so we just hold
        if step == 0:
            return "hold"
        prev_price = market.history[t][step - 1]
        cur_price  = market.price
        return "buy" if cur_price > prev_price else "sell"

    def trade(self, market: Market):
        action = self.decide(market)
        price  = market.price
        print(f"{self.name} decides to {action} at price {price:.2f}")
        if action == "buy" and self.balance >= price:
            self.holdings += 1
            self.balance  -= price
        elif action == "sell" and self.holdings > 0:
            self.holdings -= 1
            self.balance  += price

    def net_worth(self, market: Market):
        # cash + value of holdings
        return self.balance + self.holdings * market.price

if __name__ == "__main__":
    tickers = ["TSM", "TSLA", "AMZN"]
    summary = {}

    for ticker in tickers:
        print(f"\n=== Simulation for {ticker} ===")
        market = Market(
            tickers=[ticker],
            start="2022-07-05",
            end="2022-09-29",
            interval="1d"
        )
        agent = TrendFollowerAgent()

        records = []
        # step through every bar
        for _ in range(len(market.history[ticker])):
            market.update_price()
            rec = {
                "Step": market.current_step + 1,
                "Price": market.price,
            }
            # decide & trade
            action = agent.decide(market)
            agent.trade(market)
            rec[agent.name] = action
            records.append(rec)

        # final net worth
        final_row = {
            "Step": "FINAL",
            "Price": market.price,
            agent.name: f"${agent.net_worth(market):.2f}",
        }
        records.append(final_row)

        df = pd.DataFrame(records).set_index("Step")
        summary[ticker] = df

    # Combine per-ticker tables side by side
    combined = pd.concat(summary, axis=1)

    # Flatten MultiIndex column names
    flat = combined.copy()
    flat.columns = [f"{tick}_{col}" for tick, col in combined.columns]
    flat = flat.reset_index().rename(columns={"index": "Step"})

    # Round all the Price columns to 2 decimals
    price_cols = [c for c in flat.columns if c.endswith("_Price")]
    for c in price_cols:
        flat[c] = flat[c].map(lambda x: f"{x:.2f}")

    # Build and print a PrettyTable with fixed max widths
    pt = PrettyTable()
    pt.field_names = flat.columns.tolist()
    for col in pt.field_names:
        pt.max_width[col] = 12
    for _, row in flat.iterrows():
        pt.add_row(row.tolist())

    print("\n=== Step-by-Step Summary ===\n")
    print(pt)
