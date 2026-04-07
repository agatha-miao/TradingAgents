from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .akshare_cn import (
    get_stock_data_akshare,
    get_indicator_akshare,
    get_fundamentals_akshare,
    get_balance_sheet_akshare,
    get_cashflow_akshare,
    get_income_statement_akshare,
    get_news_akshare,
    get_global_news_akshare,
    get_insider_transactions_akshare,
)

# Configuration and routing logic
from .config import get_config


def _normalize_symbol_for_vendor(symbol: str, vendor: str) -> str:
    """Normalize A-share symbols for non-AKShare vendors when possible."""
    if symbol is None:
        return symbol
    s = str(symbol).strip()
    low = s.lower()

    if vendor not in {"yfinance", "alpha_vantage"}:
        return s

    # Already normalized for Yahoo/most global feeds
    if "." in s:
        return s

    code = None
    market = None
    if low.startswith(("sh", "sz", "bj")) and len(low) >= 8:
        market = low[:2]
        code = low[2:8]
    elif s.isdigit() and len(s) == 6:
        code = s
        if s.startswith(("6", "9")):
            market = "sh"
        elif s.startswith(("0", "2", "3")):
            market = "sz"
        elif s.startswith(("4", "8")):
            market = "bj"

    if not code or not market:
        return s

    suffix_map = {"sh": ".SS", "sz": ".SZ", "bj": ".BJ"}
    return f"{code}{suffix_map[market]}"


def _yf_stock(symbol, start_date, end_date):
    return get_YFin_data_online(_normalize_symbol_for_vendor(symbol, "yfinance"), start_date, end_date)


def _yf_indicator(symbol, indicator, curr_date, look_back_days):
    return get_stock_stats_indicators_window(
        _normalize_symbol_for_vendor(symbol, "yfinance"),
        indicator,
        curr_date,
        look_back_days,
    )


def _yf_fundamentals(ticker, curr_date=None):
    return get_yfinance_fundamentals(_normalize_symbol_for_vendor(ticker, "yfinance"), curr_date)


def _yf_balance_sheet(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_balance_sheet(_normalize_symbol_for_vendor(ticker, "yfinance"), freq, curr_date)


def _yf_cashflow(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_cashflow(_normalize_symbol_for_vendor(ticker, "yfinance"), freq, curr_date)


def _yf_income_statement(ticker, freq="quarterly", curr_date=None):
    return get_yfinance_income_statement(_normalize_symbol_for_vendor(ticker, "yfinance"), freq, curr_date)


def _yf_news(ticker, start_date=None, end_date=None):
    return get_news_yfinance(_normalize_symbol_for_vendor(ticker, "yfinance"), start_date, end_date)


def _yf_insider(ticker):
    return get_yfinance_insider_transactions(_normalize_symbol_for_vendor(ticker, "yfinance"))


def _av_stock(symbol, start_date, end_date):
    return get_alpha_vantage_stock(_normalize_symbol_for_vendor(symbol, "alpha_vantage"), start_date, end_date)


def _av_indicator(symbol, indicator, curr_date, look_back_days):
    return get_alpha_vantage_indicator(
        _normalize_symbol_for_vendor(symbol, "alpha_vantage"),
        indicator,
        curr_date,
        look_back_days,
    )


def _av_fundamentals(ticker, curr_date=None):
    return get_alpha_vantage_fundamentals(_normalize_symbol_for_vendor(ticker, "alpha_vantage"), curr_date)


def _av_balance_sheet(ticker, freq="quarterly", curr_date=None):
    return get_alpha_vantage_balance_sheet(_normalize_symbol_for_vendor(ticker, "alpha_vantage"), freq, curr_date)


def _av_cashflow(ticker, freq="quarterly", curr_date=None):
    return get_alpha_vantage_cashflow(_normalize_symbol_for_vendor(ticker, "alpha_vantage"), freq, curr_date)


def _av_income_statement(ticker, freq="quarterly", curr_date=None):
    return get_alpha_vantage_income_statement(_normalize_symbol_for_vendor(ticker, "alpha_vantage"), freq, curr_date)


def _av_news(ticker, start_date=None, end_date=None):
    return get_alpha_vantage_news(_normalize_symbol_for_vendor(ticker, "alpha_vantage"), start_date, end_date)


def _av_insider(ticker):
    return get_alpha_vantage_insider_transactions(_normalize_symbol_for_vendor(ticker, "alpha_vantage"))

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "akshare",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": _av_stock,
        "yfinance": _yf_stock,
        "akshare": get_stock_data_akshare,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": _av_indicator,
        "yfinance": _yf_indicator,
        "akshare": get_indicator_akshare,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": _av_fundamentals,
        "yfinance": _yf_fundamentals,
        "akshare": get_fundamentals_akshare,
    },
    "get_balance_sheet": {
        "alpha_vantage": _av_balance_sheet,
        "yfinance": _yf_balance_sheet,
        "akshare": get_balance_sheet_akshare,
    },
    "get_cashflow": {
        "alpha_vantage": _av_cashflow,
        "yfinance": _yf_cashflow,
        "akshare": get_cashflow_akshare,
    },
    "get_income_statement": {
        "alpha_vantage": _av_income_statement,
        "yfinance": _yf_income_statement,
        "akshare": get_income_statement_akshare,
    },
    # news_data
    "get_news": {
        "alpha_vantage": _av_news,
        "yfinance": _yf_news,
        "akshare": get_news_akshare,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "akshare": get_global_news_akshare,
    },
    "get_insider_transactions": {
        "alpha_vantage": _av_insider,
        "yfinance": _yf_insider,
        "akshare": get_insider_transactions_akshare,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    def _is_error_like(result) -> bool:
        if not isinstance(result, str):
            return False
        txt = result.strip().lower()
        error_prefixes = (
            "error",
            "no ",
            "unable",
            "runtimeerror",
            "n/a",
            "akshare adapter currently supports",
        )
        return txt.startswith(error_prefixes)

    last_error = None
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            result = impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

        if _is_error_like(result):
            last_error = RuntimeError(f"{vendor} returned error-like response: {str(result)[:200]}")
            continue
        return result

    if last_error:
        raise RuntimeError(f"No available vendor for '{method}': {last_error}")
    raise RuntimeError(f"No available vendor for '{method}'")
