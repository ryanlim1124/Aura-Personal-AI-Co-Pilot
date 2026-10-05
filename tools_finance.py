"""
tools_finance.py - finance analysis tools (free data from Yahoo Finance).

Needs: pip install yfinance pandas numpy

Yahoo data can be delayed, adjusted or occasionally wrong. These tools are for
learning and research, not personal financial advice.
"""
import json
import math

import pandas as pd
from langchain_core.tools import tool

VALID_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
TRADING_DAYS = 252


# ------------------------------------------------------------------ helpers
def _prices(ticker: str, period: str = "1y") -> pd.Series:
    """Download adjusted daily closing prices as a Series indexed by date."""
    import yfinance as yf

    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {sorted(VALID_PERIODS)}")
    df = yf.Ticker(ticker.strip().upper()).history(period=period, auto_adjust=True)
    prices = df["Close"].dropna()
    if len(prices) < 5:
        raise ValueError(f"not enough price data for '{ticker}' (check the ticker symbol)")
    idx = pd.to_datetime(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    prices.index = idx.normalize()  # same date format across exchanges
    return prices


def _stats(prices: pd.Series, risk_free: float = 0.0) -> dict:
    """All the numbers you usually want from a price series."""
    returns = prices.pct_change().dropna()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    total = prices.iloc[-1] / prices.iloc[0] - 1
    vol = returns.std() * math.sqrt(TRADING_DAYS)
    drawdown = prices / prices.cummax() - 1
    trough = drawdown.idxmin()
    peak = prices.loc[:trough].idxmax()
    sharpe = (returns.mean() * TRADING_DAYS - risk_free) / vol if vol > 1e-12 else None
    return {
        "start_date": str(prices.index[0].date()),
        "end_date": str(prices.index[-1].date()),
        "start_price": round(float(prices.iloc[0]), 2),
        "end_price": round(float(prices.iloc[-1]), 2),
        "period_high": round(float(prices.max()), 2),
        "period_low": round(float(prices.min()), 2),
        "total_return_pct": round(float(total) * 100, 2),
        # Annualising less than a year of data is misleading, so skip it.
        "annualized_return_pct": round(((1 + float(total)) ** (1 / years) - 1) * 100, 2)
        if years >= 1
        else None,
        "annualized_volatility_pct": round(float(vol) * 100, 2),
        "max_drawdown_pct": round(float(drawdown.min()) * 100, 2),
        "max_drawdown_peak_date": str(peak.date()),
        "max_drawdown_trough_date": str(trough.date()),
        "sharpe_ratio": round(float(sharpe), 2) if sharpe is not None else None,
        "best_day_pct": round(float(returns.max()) * 100, 2),
        "worst_day_pct": round(float(returns.min()) * 100, 2),
    }


def _beta(prices: pd.Series, benchmark: pd.Series):
    both = pd.concat([prices.pct_change(), benchmark.pct_change()], axis=1, join="inner").dropna()
    if len(both) < 20:
        return None
    var = both.iloc[:, 1].var()
    return round(float(both.iloc[:, 0].cov(both.iloc[:, 1]) / var), 2) if var else None


def _pick(d: dict, keys: list) -> dict:
    return {k: d[k] for k in keys}


# -------------------------------------------------------------------- tools
@tool
def price_history(ticker: str, period: str = "1y") -> str:
    """Summarise how a stock or ETF's price moved: start/end price, total and
    annualised return, high/low, best and worst day. period is one of
    1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max."""
    try:
        s = _stats(_prices(ticker, period))
    except Exception as e:
        return f"Could not analyse {ticker}: {e}"
    keys = ["start_date", "end_date", "start_price", "end_price", "period_high",
            "period_low", "total_return_pct", "annualized_return_pct",
            "best_day_pct", "worst_day_pct"]
    return json.dumps({"ticker": ticker.upper(), "period": period, **_pick(s, keys)}, indent=2)


@tool
def risk_metrics(ticker: str, period: str = "1y", risk_free_rate: float = 0.0,
                 benchmark: str = "SPY") -> str:
    """Risk profile of a stock or ETF: annualised volatility, maximum drawdown
    (with peak and trough dates), Sharpe ratio, and beta versus a benchmark
    (default SPY). risk_free_rate is an annual decimal such as 0.04 (default 0).
    period is one of 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max."""
    try:
        prices = _prices(ticker, period)
        s = _stats(prices, risk_free_rate)
    except Exception as e:
        return f"Could not analyse {ticker}: {e}"
    try:
        beta = _beta(prices, _prices(benchmark, period))
    except Exception:
        beta = None
    keys = ["annualized_volatility_pct", "max_drawdown_pct", "max_drawdown_peak_date",
            "max_drawdown_trough_date", "sharpe_ratio"]
    out = {"ticker": ticker.upper(), "period": period, **_pick(s, keys),
           f"beta_vs_{benchmark.upper()}": beta, "risk_free_rate_used": risk_free_rate}
    return json.dumps(out, indent=2)


@tool
def compare_stocks(tickers: str, period: str = "1y") -> str:
    """Compare 2-6 stocks/ETFs side by side (return, volatility, max drawdown,
    Sharpe) plus how correlated their daily moves are. tickers is comma
    separated, e.g. 'AAPL,MSFT,VOO'."""
    names = [t.strip().upper() for t in tickers.split(",") if t.strip()][:6]
    if len(names) < 2:
        return "Give at least two tickers separated by commas."
    table, returns, errors = {}, {}, []
    for t in names:
        try:
            p = _prices(t, period)
            s = _stats(p)
            table[t] = _pick(s, ["total_return_pct", "annualized_volatility_pct",
                                 "max_drawdown_pct", "sharpe_ratio"])
            returns[t] = p.pct_change()
        except Exception as e:
            errors.append(f"{t}: {e}")
    out = {"period": period, "stats": table}
    if len(returns) >= 2:
        corr = pd.concat(returns, axis=1, join="inner").dropna().corr().round(2)
        out["correlation_of_daily_returns"] = corr.to_dict()
    if errors:
        out["errors"] = errors
    return json.dumps(out, indent=2)


@tool
def portfolio_risk(holdings: str, period: str = "1y") -> str:
    """Analyse a portfolio of stocks/ETFs. holdings is 'TICKER:shares' pairs,
    comma separated, e.g. 'AAPL:10,MSFT:5,VOO:2'. Returns current weights,
    portfolio volatility, max drawdown, how much diversification helped, and
    each holding's share of total risk. Assumes the current weights were held
    throughout; mixed currencies are NOT converted."""
    try:
        shares = {}
        for pair in holdings.split(","):
            t, n = pair.split(":")
            shares[t.strip().upper()] = float(n)
        if len(shares) < 2:
            return "Give at least two holdings, like 'AAPL:10,MSFT:5'."
    except ValueError:
        return "Format holdings like 'AAPL:10,MSFT:5' (ticker:shares)."

    prices, errors = {}, []
    for t in shares:
        try:
            prices[t] = _prices(t, period)
        except Exception as e:
            errors.append(f"{t}: {e}")
    if len(prices) < 2:
        return f"Not enough usable data. {errors}"

    values = {t: shares[t] * float(p.iloc[-1]) for t, p in prices.items()}
    total = sum(values.values())
    weights = {t: v / total for t, v in values.items()}
    rets = pd.concat({t: p.pct_change() for t, p in prices.items()}, axis=1, join="inner").dropna()
    if len(rets) < 20:
        return "Not enough overlapping price history for these tickers."
    port = sum(rets[t] * w for t, w in weights.items())
    index = (1 + port).cumprod()
    port_vol = float(port.std() * math.sqrt(TRADING_DAYS))
    avg_vol = sum(weights[t] * float(rets[t].std() * math.sqrt(TRADING_DAYS)) for t in weights)
    var_p = port.var()
    contrib = {t: round(float(weights[t] * rets[t].cov(port) / var_p) * 100, 1) for t in weights}
    out = {
        "period": period,
        "total_value": round(total, 2),
        "weights_pct": {t: round(w * 100, 1) for t, w in weights.items()},
        "portfolio_volatility_pct": round(port_vol * 100, 2),
        "weighted_avg_stand_alone_volatility_pct": round(avg_vol * 100, 2),
        "diversification_benefit_pct_points": round((avg_vol - port_vol) * 100, 2),
        "max_drawdown_pct": round(float((index / index.cummax() - 1).min()) * 100, 2),
        "share_of_total_risk_pct": contrib,
    }
    if errors:
        out["errors"] = errors
    return json.dumps(out, indent=2)


@tool
def company_fundamentals(ticker: str) -> str:
    """Key fundamentals for a listed company: revenue, growth, margins, return
    on equity, debt, cash flow, valuation multiples. Use for questions about how
    healthy or expensive a company is."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker.strip().upper()).info
    except Exception as e:
        return f"Could not fetch data for {ticker}: {e}"

    def pct(key):
        v = info.get(key)
        return round(v * 100, 2) if isinstance(v, (int, float)) else None

    fields = {
        "name": info.get("shortName"),
        "revenue": info.get("totalRevenue"),
        "revenue_growth_pct": pct("revenueGrowth"),
        "gross_margin_pct": pct("grossMargins"),
        "operating_margin_pct": pct("operatingMargins"),
        "net_margin_pct": pct("profitMargins"),
        "return_on_equity_pct": pct("returnOnEquity"),
        "return_on_assets_pct": pct("returnOnAssets"),
        "debt_to_equity_pct": info.get("debtToEquity"),  # Yahoo gives this as a percentage
        "free_cash_flow": info.get("freeCashflow"),
        "operating_cash_flow": info.get("operatingCashflow"),
        "trailing_eps": info.get("trailingEps"),
        "price_to_book": info.get("priceToBook"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        "peg_ratio": info.get("pegRatio") or info.get("trailingPegRatio"),
        "beta": info.get("beta"),
    }
    fields = {k: v for k, v in fields.items() if v is not None}
    if len(fields) <= 1:
        return f"No fundamentals found for '{ticker}'. It may be an ETF or a wrong ticker."
    return json.dumps(fields, indent=2)


@tool
def investment_projection(initial: float, monthly: float, annual_return_pct: float,
                          years: float, inflation_pct: float = 0.0) -> str:
    """Project what regular investing could grow to: a starting amount plus a
    monthly contribution, at an ASSUMED constant annual return, compounded
    monthly. Shows money put in, final value, and value in today's money if
    inflation_pct is given. An illustration of compounding, not a forecast."""
    n = int(round(years * 12))
    r = annual_return_pct / 100 / 12
    if abs(r) < 1e-12:
        final = initial + monthly * n
    else:
        final = initial * (1 + r) ** n + monthly * (((1 + r) ** n - 1) / r)
    put_in = initial + monthly * n
    out = {
        "total_contributed": round(put_in, 2),
        "final_value": round(final, 2),
        "growth_from_returns": round(final - put_in, 2),
        "assumptions": "constant return, contributions at each month end, no fees or tax",
    }
    if inflation_pct:
        out["final_value_in_todays_money"] = round(final / (1 + inflation_pct / 100) ** years, 2)
    return json.dumps(out, indent=2)


finance_tools = [price_history, risk_metrics, compare_stocks, portfolio_risk,
                 company_fundamentals, investment_projection]