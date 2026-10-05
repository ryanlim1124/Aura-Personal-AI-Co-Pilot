"""
tools.py - the abilities your agent can use.

The docstring under each @tool is what the AI reads to decide WHEN to use it,
so write docstrings like instructions to a smart intern.

Install what these need (all free, no API keys):
    pip install ddgs yfinance langchain-core
"""
import ast
import json
import operator
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# Your learning journal lives next to this file.
NOTES_FILE = Path(__file__).parent / "learning_notes.md"


# ---------------------------------------------------------------- research
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or anything you are unsure
    about. Returns the top results with titles, links and snippets."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)


@tool
def wikipedia_summary(topic: str) -> str:
    """Get a short, reliable summary of a concept, company, person or technology
    from Wikipedia. Best for definitions and background before going deeper."""
    title = urllib.parse.quote(topic.strip().replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    req = urllib.request.Request(url, headers={"User-Agent": "personal-ai-agent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        link = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"{data.get('title')}: {data.get('extract')}\n{link}"
    except Exception as e:
        return f"No Wikipedia page found for '{topic}' ({e}). Try web_search instead."


# ----------------------------------------------------------------- finance
@tool
def stock_snapshot(ticker: str) -> str:
    """Get a quick snapshot of a listed stock or ETF: price, market cap, P/E,
    52-week range, sector. Use the Yahoo Finance ticker, e.g. AAPL, MSFT, VOO."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker.strip().upper()).info
    except Exception as e:
        return f"Could not fetch data for {ticker}: {e}"

    fields = {
        "name": info.get("shortName"),
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "dividend_yield": info.get("dividendYield"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return f"No data found for '{ticker}'. Check the ticker symbol."
    return json.dumps(fields, indent=2)


# Safe maths: only numbers and + - * / ** % are allowed (never plain eval()).
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Do exact maths. Use this for ANY calculation (compound interest, returns,
    percentages, ratios) instead of doing arithmetic yourself.
    Example: '10000 * (1 + 0.07) ** 10'"""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Could not calculate '{expression}': {e}"


# --------------------------------------------------------------- assistant
@tool
def current_datetime() -> str:
    """Get today's date and the current time. Use whenever the answer depends on
    'today', 'this week', or how recent something is."""
    return datetime.now().strftime("%A, %d %B %Y, %H:%M")


@tool
def save_note(note: str) -> str:
    """Save something worth remembering (a lesson learned, a summary, a to-do)
    to the user's learning journal."""
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"- [{datetime.now():%Y-%m-%d %H:%M}] {note}\n")
    return "Note saved."


@tool
def read_notes() -> str:
    """Read back the user's saved learning notes. Use this to review what they
    have learned or to quiz them on it."""
    if not NOTES_FILE.exists():
        return "No notes saved yet."
    return NOTES_FILE.read_text(encoding="utf-8")[-4000:]


# main.py imports this list and hands it to the agent.
all_tools = [
    web_search,
    wikipedia_summary,
    stock_snapshot,
    calculator,
    current_datetime,
    save_note,
    read_notes,
]