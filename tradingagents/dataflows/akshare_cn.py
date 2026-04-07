from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

import pandas as pd


def _load_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            "AKShare is not installed. Run `pip install akshare` first."
        ) from exc
    return ak


def _normalize_a_share_symbol(symbol: str) -> str:
    s = symbol.strip().lower()
    if s.startswith(("sh", "sz", "bj")) and len(s) >= 8:
        return s
    if s.isdigit() and len(s) == 6:
        if s.startswith(("6", "9")):
            return f"sh{s}"
        if s.startswith(("0", "2", "3")):
            return f"sz{s}"
        if s.startswith(("4", "8")):
            return f"bj{s}"
    return s


def _extract_code(symbol: str) -> str:
    norm = _normalize_a_share_symbol(symbol)
    if len(norm) >= 8 and norm[:2] in {"sh", "sz", "bj"}:
        return norm[2:]
    return norm


def _to_em_symbol(symbol: str) -> str:
    """Convert A-share symbol to Eastmoney style, e.g. SH600519."""
    norm = _normalize_a_share_symbol(symbol)
    if len(norm) >= 8 and norm[:2] in {"sh", "sz", "bj"}:
        return f"{norm[:2].upper()}{norm[2:]}"
    code = _extract_code(symbol)
    if code.startswith(("6", "9")):
        return f"SH{code}"
    if code.startswith(("0", "2", "3")):
        return f"SZ{code}"
    if code.startswith(("4", "8")):
        return f"BJ{code}"
    return code.upper()


def _filter_by_date(df: pd.DataFrame, start_date: str = None, end_date: str = None, curr_date: str = None) -> pd.DataFrame:
    """Filter dataframe by first matching date-like column."""
    if df is None or df.empty:
        return df

    date_cols = [
        "REPORT_DATE",
        "REPORT_DATE_NAME",
        "报告期",
        "日期",
        "公告日期",
        "发布时间",
        "时间",
        "date",
        "Date",
    ]
    col = next((c for c in date_cols if c in df.columns), None)
    if not col:
        return df

    tmp = df.copy()
    parsed = pd.to_datetime(tmp[col], errors="coerce")
    mask = parsed.notna()
    if start_date:
        mask &= parsed >= pd.to_datetime(start_date)
    if end_date:
        mask &= parsed <= pd.to_datetime(end_date)
    if curr_date:
        mask &= parsed <= pd.to_datetime(curr_date)
    return tmp[mask]


def _format_table(title: str, df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return f"## {title}\nNo data available.\n"
    clipped = df.head(max_rows)
    return (
        f"## {title}\n"
        f"Rows: {len(df)} (showing first {len(clipped)})\n\n"
        f"{clipped.to_csv(index=False)}"
    )


def _apply_statement_frequency(df: pd.DataFrame, freq: str = "quarterly") -> pd.DataFrame:
    """Filter statement rows by frequency preference when possible."""
    if df is None or df.empty:
        return df

    freq_norm = (freq or "quarterly").strip().lower()
    if freq_norm not in {"quarterly", "annual"}:
        return df

    out = df.copy()

    # Text-based filters (works for many CN statement datasets)
    text_cols = ["REPORT_DATE_NAME", "报告期", "报告类型"]
    if freq_norm == "annual":
        for col in text_cols:
            if col in out.columns:
                mask = out[col].astype(str).str.contains("年报|年度|12-31", na=False)
                if mask.any():
                    out = out[mask]
                    return out

    # Date-based filters as fallback
    date_cols = ["REPORT_DATE", "报告期", "日期", "Date"]
    col = next((c for c in date_cols if c in out.columns), None)
    if not col:
        return out

    parsed = pd.to_datetime(out[col], errors="coerce")
    valid = parsed.notna()
    if not valid.any():
        return out
    out = out[valid].copy()
    parsed = parsed[valid]

    if freq_norm == "annual":
        return out[(parsed.dt.month == 12) & (parsed.dt.day == 31)]

    # quarterly: keep quarter-end rows
    return out[
        ((parsed.dt.month == 3) & (parsed.dt.day == 31))
        | ((parsed.dt.month == 6) & (parsed.dt.day == 30))
        | ((parsed.dt.month == 9) & (parsed.dt.day == 30))
        | ((parsed.dt.month == 12) & (parsed.dt.day == 31))
    ]


def get_stock_data_akshare(
    symbol: Annotated[str, "A-share symbol, e.g. 600519 or sh600519"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):
    ak = _load_akshare()
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    code = _extract_code(symbol)
    start = start_date.replace("-", "")
    end = end_date.replace("-", "")

    try:
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
    except Exception as exc:
        return f"Error fetching A-share stock data for {symbol}: {exc}"

    if df is None or df.empty:
        return f"No A-share data found for symbol '{symbol}' between {start_date} and {end_date}"

    rename_map = {
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "ChangePct",
        "涨跌额": "Change",
        "换手率": "Turnover",
    }
    out = df.rename(columns=rename_map)
    header = (
        f"# A-share stock data for {symbol} ({code}) from {start_date} to {end_date}\n"
        f"# Total records: {len(out)}\n"
        f"# Data source: AKShare stock_zh_a_hist\n\n"
    )
    return header + out.to_csv(index=False)


def get_indicator_akshare(
    symbol: Annotated[str, "A-share symbol"],
    indicator: Annotated[str, "technical indicator"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    end_date = curr_date
    start_date = (datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=look_back_days * 3)).strftime("%Y-%m-%d")
    raw = get_stock_data_akshare(symbol, start_date, end_date)
    if raw.startswith("No A-share") or raw.startswith("Error"):
        return raw

    csv_part = "\n".join(raw.splitlines()[4:])
    df = pd.read_csv(pd.io.common.StringIO(csv_part))
    if df.empty or "Close" not in df.columns:
        return f"Unable to compute indicator {indicator} for {symbol}"

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df.get("High"), errors="coerce") if "High" in df.columns else None
    low = pd.to_numeric(df.get("Low"), errors="coerce") if "Low" in df.columns else None
    volume = pd.to_numeric(df.get("Volume"), errors="coerce") if "Volume" in df.columns else None

    indicator_lower = indicator.lower()
    if indicator_lower == "rsi":
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        series = 100 - (100 / (1 + rs))
    elif indicator_lower in {"close_10_ema", "ema10"}:
        series = close.ewm(span=10, adjust=False).mean()
    elif indicator_lower in {"close_50_sma", "sma50"}:
        series = close.rolling(50).mean()
    elif indicator_lower in {"close_200_sma", "sma200"}:
        series = close.rolling(200).mean()
    elif indicator_lower == "macd":
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        series = ema12 - ema26
    elif indicator_lower == "macds":
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        series = macd_line.ewm(span=9, adjust=False).mean()
    elif indicator_lower == "macdh":
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        series = macd_line - signal_line
    elif indicator_lower == "boll":
        series = close.rolling(20).mean()
    elif indicator_lower == "boll_ub":
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        series = ma20 + 2 * std20
    elif indicator_lower == "boll_lb":
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        series = ma20 - 2 * std20
    elif indicator_lower == "atr":
        if high is None or low is None:
            return f"Unable to compute ATR for {symbol}: High/Low columns are missing."
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        series = tr.rolling(14).mean()
    elif indicator_lower == "vwma":
        if volume is None:
            return f"Unable to compute VWMA for {symbol}: Volume column is missing."
        pv = close * volume
        series = pv.rolling(20).sum() / volume.rolling(20).sum()
    else:
        return (
            "AKShare adapter currently supports indicators: "
            "rsi, close_10_ema, close_50_sma, close_200_sma, "
            "macd, macds, macdh, boll, boll_ub, boll_lb, atr, vwma. "
            f"Requested: {indicator}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["indicator"] = series
    window_df = df.dropna(subset=["Date"]).tail(look_back_days)
    lines = [f"{d}: {v}" for d, v in zip(window_df["Date"], window_df["indicator"])]
    return f"## {indicator} values for {symbol} up to {curr_date}:\n\n" + "\n".join(lines)


def get_fundamentals_akshare(
    ticker: Annotated[str, "A-share symbol"],
    curr_date: Annotated[str, "current date"] = None,
):
    ak = _load_akshare()
    code = _extract_code(ticker)

    sections = []

    try:
        info_df = ak.stock_individual_info_em(symbol=code)
        sections.append(_format_table(f"Basic info for {ticker}", info_df, max_rows=50))
    except Exception as exc:
        sections.append(f"## Basic info for {ticker}\nError: {exc}\n")

    try:
        biz_df = ak.stock_zyjs_ths(symbol=code)
        sections.append(_format_table(f"Business profile for {ticker}", biz_df, max_rows=30))
    except Exception as exc:
        sections.append(f"## Business profile for {ticker}\nError: {exc}\n")

    try:
        abstract_df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        abstract_df = _filter_by_date(abstract_df, curr_date=curr_date)
        sections.append(_format_table(f"Financial abstract for {ticker}", abstract_df, max_rows=30))
    except Exception as exc:
        sections.append(f"## Financial abstract for {ticker}\nError: {exc}\n")

    try:
        start_year = "1900"
        if curr_date:
            start_year = str(max(1900, pd.to_datetime(curr_date).year - 5))
        fi_df = ak.stock_financial_analysis_indicator(symbol=code, start_year=start_year)
        fi_df = _filter_by_date(fi_df, curr_date=curr_date)
        sections.append(_format_table(f"Financial indicators for {ticker}", fi_df, max_rows=30))
    except Exception as exc:
        sections.append(f"## Financial indicators for {ticker}\nError: {exc}\n")

    return "\n\n".join(sections)


def get_balance_sheet_akshare(
    ticker: Annotated[str, "A-share symbol"],
    freq: Annotated[str, "quarterly/annual"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    ak = _load_akshare()
    em_symbol = _to_em_symbol(ticker)
    try:
        df = ak.stock_balance_sheet_by_report_em(symbol=em_symbol)
        df = _apply_statement_frequency(df, freq=freq)
        df = _filter_by_date(df, curr_date=curr_date)
        return _format_table(f"Balance sheet ({freq}) for {ticker}", df, max_rows=40)
    except Exception as exc:
        return f"Error retrieving AKShare balance sheet for {ticker}: {exc}"


def get_cashflow_akshare(
    ticker: Annotated[str, "A-share symbol"],
    freq: Annotated[str, "quarterly/annual"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    ak = _load_akshare()
    em_symbol = _to_em_symbol(ticker)
    try:
        df = ak.stock_cash_flow_sheet_by_report_em(symbol=em_symbol)
        df = _apply_statement_frequency(df, freq=freq)
        df = _filter_by_date(df, curr_date=curr_date)
        return _format_table(f"Cashflow ({freq}) for {ticker}", df, max_rows=40)
    except Exception as exc:
        return f"Error retrieving AKShare cashflow for {ticker}: {exc}"


def get_income_statement_akshare(
    ticker: Annotated[str, "A-share symbol"],
    freq: Annotated[str, "quarterly/annual"] = "quarterly",
    curr_date: Annotated[str, "current date"] = None,
):
    ak = _load_akshare()
    em_symbol = _to_em_symbol(ticker)
    try:
        df = ak.stock_profit_sheet_by_report_em(symbol=em_symbol)
        df = _apply_statement_frequency(df, freq=freq)
        df = _filter_by_date(df, curr_date=curr_date)
        return _format_table(f"Income statement ({freq}) for {ticker}", df, max_rows=40)
    except Exception as exc:
        return f"Error retrieving AKShare income statement for {ticker}: {exc}"


def get_news_akshare(
    ticker: Annotated[str, "A-share symbol"],
    start_date: Annotated[str, "Start date"] = None,
    end_date: Annotated[str, "End date"] = None,
):
    ak = _load_akshare()
    code = _extract_code(ticker)

    sections = []

    try:
        stock_news_df = ak.stock_news_em(symbol=code)
        stock_news_df = _filter_by_date(stock_news_df, start_date=start_date, end_date=end_date)
        sections.append(_format_table(f"Stock news for {ticker}", stock_news_df, max_rows=30))
    except Exception as exc:
        sections.append(f"## Stock news for {ticker}\nError: {exc}\n")

    # Announcements are useful as high-signal company events
    try:
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = pd.Timestamp.now().normalize()
        if start_date:
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = end_dt - pd.Timedelta(days=7)
        if start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        date_range = pd.date_range(start=start_dt, end=end_dt, freq="D")
        # Avoid extremely long loops against announcement endpoint.
        if len(date_range) > 31:
            date_range = date_range[-31:]

        notice_frames = []
        for d in date_range:
            try:
                day_df = ak.stock_notice_report(symbol="全部", date=d.strftime("%Y%m%d"))
                if day_df is not None and not day_df.empty:
                    notice_frames.append(day_df)
            except Exception:
                continue

        if notice_frames:
            notice_df = pd.concat(notice_frames, ignore_index=True).drop_duplicates()
        else:
            notice_df = pd.DataFrame()

        code_cols = [c for c in ["代码", "证券代码", "股票代码"] if c in notice_df.columns]
        if code_cols and not notice_df.empty:
            notice_df = notice_df[
                notice_df[code_cols[0]].astype(str).str.contains(code, na=False)
            ]
        notice_df = _filter_by_date(
            notice_df, start_date=start_dt.strftime("%Y-%m-%d"), end_date=end_dt.strftime("%Y-%m-%d")
        )
        sections.append(_format_table(f"Announcements for {ticker}", notice_df, max_rows=30))
    except Exception as exc:
        sections.append(f"## Announcements for {ticker}\nError: {exc}\n")

    return "\n\n".join(sections)


def get_global_news_akshare(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"] = None,
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
):
    ak = _load_akshare()
    end_dt = pd.to_datetime(curr_date) if curr_date else pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days=look_back_days)

    try:
        df = ak.stock_info_global_em()
    except Exception:
        # Fallback provider
        try:
            df = ak.stock_info_global_cls(symbol="全部")
        except Exception as exc:
            return f"Error retrieving AKShare global news: {exc}"

    df = _filter_by_date(
        df,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
    )
    if limit and limit > 0:
        df = df.head(limit)
    return _format_table("Global market news", df, max_rows=limit or 20)


def get_insider_transactions_akshare(
    ticker: Annotated[str, "A-share symbol"],
):
    ak = _load_akshare()
    code = _extract_code(ticker)
    try:
        df = ak.stock_management_change_ths(symbol=code)
        return _format_table(f"Management holding changes for {ticker}", df, max_rows=30)
    except Exception as exc:
        return f"Error retrieving management holding changes for {ticker}: {exc}"
