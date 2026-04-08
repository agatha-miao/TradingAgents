"""
Daily Scan -> Deep Dive -> Report & Email pipeline.

Supabase integration entry points are intentionally stubbed so you can
wire your own DB credentials/schema after local testing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import signal
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients import create_llm_client


load_dotenv()


@contextlib.contextmanager
def _time_limit(seconds: int):
    """Raise TimeoutError if code block exceeds `seconds` (Unix only)."""
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handle_timeout(signum, frame):  # type: ignore[unused-argument]
        raise TimeoutError(f"operation timed out after {seconds}s")

    prev_handler = signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prev_handler)


@dataclass
class ScanResult:
    ticker: str
    technical_score: float
    fundamental_score: float
    total_score: float
    rationale: str
    company_id: int | None = None
    ts_code: str | None = None
    trigger_tags: List[str] | None = None
    themed_news: List[Dict[str, str]] | None = None


@dataclass
class UniverseItem:
    ticker: str
    company_id: int | None = None
    ts_code: str | None = None
    name: str | None = None
    trigger_tags: List[str] | None = None
    business: str | None = None


def _ticker_to_code(ticker: str) -> str:
    s = (ticker or "").strip()
    if "." in s:
        return s.split(".")[0]
    low = s.lower()
    if low.startswith(("sh", "sz", "bj")) and len(low) >= 8:
        return low[2:8]
    return s


def _code_to_ts_code(code: str) -> str:
    c = str(code).strip()
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return c


def _extract_last_float(text: str) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _score_technical(ticker: str, trade_date: str) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    rsi_text = route_to_vendor("get_indicators", ticker, "rsi", trade_date, 30)
    rsi = _extract_last_float(rsi_text)
    if rsi is not None:
        if rsi < 30:
            score += 1.0
            reasons.append(f"RSI={rsi:.2f} (oversold)")
        elif rsi > 70:
            score -= 1.0
            reasons.append(f"RSI={rsi:.2f} (overbought)")
        else:
            reasons.append(f"RSI={rsi:.2f} (neutral)")

    macd_text = route_to_vendor("get_indicators", ticker, "macd", trade_date, 30)
    macd = _extract_last_float(macd_text)
    if macd is not None:
        if macd > 0:
            score += 1.0
            reasons.append(f"MACD={macd:.4f} (>0)")
        else:
            score -= 1.0
            reasons.append(f"MACD={macd:.4f} (<=0)")

    return score, reasons


def _score_fundamentals(ticker: str, trade_date: str) -> Tuple[float, List[str]]:
    text = route_to_vendor("get_fundamentals", ticker, trade_date)
    s = (text or "").lower()
    score = 0.0
    reasons: List[str] = []

    positive_terms = ["增长", "improve", "increase", "盈利", "profit", "现金流", "buyback", "回购"]
    negative_terms = ["风险", "decline", "decrease", "亏损", "lawsuit", "减持", "warning"]

    pos = sum(term in s for term in positive_terms)
    neg = sum(term in s for term in negative_terms)
    score += (pos - neg) * 0.25
    reasons.append(f"fundamental_terms: +{pos}/-{neg}")
    return score, reasons


def _web_search_news_rss(query: str, days: int = 7, limit: int = 5) -> List[Dict[str, str]]:
    q = quote_plus(f"{query} when:{max(1, days)}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
    with urlopen(url, timeout=10) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    items = []
    for item in root.findall(".//item")[: max(1, limit)]:
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pub_date": (item.findtext("pubDate") or "").strip(),
            }
        )
    return items


def _score_theme_news(item: UniverseItem, trade_date: str) -> Tuple[float, List[str], List[Dict[str, str]]]:
    themes = (item.trigger_tags or [])[:4]
    company_term = item.name or item.ts_code or item.ticker
    queries = [f"{company_term} {t}" for t in themes] if themes else [company_term]
    dt = datetime.strptime(trade_date, "%Y-%m-%d")
    start_date = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    score = 0.0
    reasons: List[str] = []
    all_news: List[Dict[str, str]] = []
    seen_links = set()

    positive_terms = ("中标", "订单", "增长", "创新高", "突破", "合作", "回购", "增持")
    negative_terms = ("减持", "诉讼", "下滑", "亏损", "风险", "处罚", "违约")

    for q in queries:
        # 先使用数据源新闻，补充使用主题web搜索
        vendor_text = str(route_to_vendor("get_news", item.ticker, start_date, trade_date) or "")
        if vendor_text:
            reasons.append(f"vendor_news[{q}]: yes")
        try:
            web_news = _web_search_news_rss(q, days=7, limit=3)
        except Exception as exc:
            reasons.append(f"web_news[{q}]: error({exc})")
            web_news = []
        for n in web_news:
            link = n.get("link", "")
            if not link or link in seen_links:
                continue
            seen_links.add(link)
            themed = {"query": q, **n}
            all_news.append(themed)
            title = n.get("title", "")
            score += 0.15 * sum(k in title for k in positive_terms)
            score -= 0.15 * sum(k in title for k in negative_terms)

    reasons.append(f"theme_news_hits: {len(all_news)}")
    if themes:
        reasons.append(f"themes: {', '.join(themes)}")
    return score, reasons, all_news


def _contains_enough_chinese(text: str, min_ratio: float = 0.60) -> bool:
    if not text:
        return False
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters == 0:
        return True
    return (cjk / letters) >= min_ratio


def _polish_chinese_report(report_text: str, ticker: str, config: Dict[str, Any]) -> str:
    """
    对最终中文报告做语言质量检查：
    1) 中文占比检查
    2) 尝试用 quick-think LLM 做中文润色（保留事实，不改结论）
    """
    needs_polish = not _contains_enough_chinese(report_text, min_ratio=0.6)
    provider = config.get("llm_provider", "openai")
    model = config.get("quick_think_llm") or config.get("deep_think_llm")
    base_url = config.get("backend_url")

    try:
        client = create_llm_client(provider=provider, model=model, base_url=base_url)
        llm = client.get_llm()
        prompt = (
            "你是A股研究报告的中文审校助手。"
            "请对下面报告做中文质量检查与润色，要求：\n"
            "1) 必须输出中文；2) 语句通顺、专业；3) 保留原始事实和结论，不新增编造信息；\n"
            "4) 保留原文中的关键数字、公司名、代码与证据链接；\n"
            "5) 输出结构：先给“审校结论（1-2句）”，再给“润色后报告”正文。\n\n"
            f"股票: {ticker}\n"
            f"原始报告:\n{report_text}\n"
        )
        resp = llm.invoke(prompt)
        polished = str(getattr(resp, "content", "") or "").strip()
        if polished:
            return polished
    except Exception:
        # 无可用LLM时走兜底策略
        pass

    if needs_polish:
        return "审校结论：检测到原文中文占比偏低，建议配置LLM后启用自动润色。\n\n润色后报告：\n" + report_text
    return "审校结论：语言基本通顺，符合中文输出要求。\n\n润色后报告：\n" + report_text


def run_light_scan(universe_items: List[UniverseItem], trade_date: str) -> List[ScanResult]:
    results: List[ScanResult] = []
    total_items = len(universe_items)
    for idx, item in enumerate(universe_items, 1):
        ticker = item.ticker
        started = time.monotonic()
        print(f"[LightScan] ({idx}/{total_items}) start ticker={ticker}", flush=True)
        try:
            ts, tr = _score_technical(ticker, trade_date)
            fs, fr = _score_fundamentals(ticker, trade_date)
            ns, nr, news = _score_theme_news(item, trade_date)
            total = ts * 0.6 + fs * 0.25 + ns * 0.15
            results.append(
                ScanResult(
                    ticker=ticker,
                    technical_score=ts,
                    fundamental_score=fs,
                    total_score=total,
                    rationale="; ".join(tr + fr + nr),
                    company_id=item.company_id,
                    ts_code=item.ts_code,
                    trigger_tags=item.trigger_tags or [],
                    themed_news=news,
                )
            )
            elapsed = time.monotonic() - started
            print(
                f"[LightScan] ({idx}/{total_items}) done ticker={ticker} score={total:.3f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        except Exception as exc:
            results.append(
                ScanResult(
                    ticker=ticker,
                    technical_score=-999.0,
                    fundamental_score=-999.0,
                    total_score=-999.0,
                    rationale=f"scan_error: {exc}",
                    company_id=item.company_id,
                    ts_code=item.ts_code,
                    trigger_tags=item.trigger_tags or [],
                    themed_news=[],
                )
            )
            elapsed = time.monotonic() - started
            print(
                f"[LightScan] ({idx}/{total_items}) error ticker={ticker} elapsed={elapsed:.1f}s err={exc}",
                flush=True,
            )
    return sorted(results, key=lambda x: x.total_score, reverse=True)


def run_deep_dive(top_tickers: List[str], trade_date: str, config: Dict) -> Dict[str, str]:
    # Keep fundamentals/news in deep dive; technical remains weighted in scan stage.
    ta = TradingAgentsGraph(
        debug=False,
        config=config,
        selected_analysts=["market", "news", "fundamentals"],
    )
    reports: Dict[str, str] = {}
    timeout_seconds = int(os.getenv("DEEP_DIVE_TIMEOUT_SECONDS", "1200"))
    total_items = len(top_tickers)
    for idx, t in enumerate(top_tickers, 1):
        started = time.monotonic()
        print(f"[DeepDive] ({idx}/{total_items}) start ticker={t}", flush=True)
        try:
            with _time_limit(timeout_seconds):
                _, decision = ta.propagate(t, trade_date)
            reports[t] = _polish_chinese_report(str(decision), t, config)
            elapsed = time.monotonic() - started
            print(f"[DeepDive] ({idx}/{total_items}) done ticker={t} elapsed={elapsed:.1f}s", flush=True)
        except TimeoutError:
            elapsed = time.monotonic() - started
            reports[t] = (
                "审校结论：本次深度分析超时，已跳过该标的。\n\n"
                "润色后报告：\n"
                f"- ticker: {t}\n"
                f"- trade_date: {trade_date}\n"
                f"- status: timeout_after_{timeout_seconds}s"
            )
            print(
                f"[DeepDive] ({idx}/{total_items}) timeout ticker={t} elapsed={elapsed:.1f}s "
                f"timeout={timeout_seconds}s",
                flush=True,
            )
        except Exception as exc:
            elapsed = time.monotonic() - started
            reports[t] = (
                "审校结论：本次深度分析异常，已跳过该标的。\n\n"
                "润色后报告：\n"
                f"- ticker: {t}\n"
                f"- trade_date: {trade_date}\n"
                f"- status: error\n"
                f"- reason: {exc}"
            )
            print(f"[DeepDive] ({idx}/{total_items}) error ticker={t} elapsed={elapsed:.1f}s err={exc}", flush=True)
    return reports


def render_reports(
    scan_results: List[ScanResult],
    deep_reports: Dict[str, str],
    out_dir: Path,
    deep_count: int,
) -> Tuple[Path, Dict[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    details: Dict[str, Path] = {}

    for ticker, content in deep_reports.items():
        detail_path = out_dir / f"{ticker}.html"
        detail_html = f"""
        <html><body>
        <h1>{ticker} 深度分析</h1>
        <h2>200字结论摘要</h2>
        <p>{content[:220]}</p>
        <h2>详细分析（技术面 + 基本面 + 证据链）</h2>
        <pre>{content}</pre>
        </body></html>
        """
        detail_path.write_text(detail_html, encoding="utf-8")
        details[ticker] = detail_path

    winners = scan_results[:deep_count]
    rows = []
    for idx, r in enumerate(winners, 1):
        link = details[r.ticker].name if r.ticker in details else "#"
        rows.append(
            f"<tr><td>{idx}</td><td><a href='{link}'>{r.ticker}</a></td>"
            f"<td>{r.total_score:.3f}</td><td>{r.rationale}</td></tr>"
        )

    overview = out_dir / "overview.html"
    overview_html = f"""
    <html><body>
    <h1>每日扫描结果</h1>
    <h2>胜选股票（Top {deep_count}）</h2>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr><th>#</th><th>Ticker</th><th>Score</th><th>Reason</th></tr>
      {''.join(rows)}
    </table>
    </body></html>
    """
    overview.write_text(overview_html, encoding="utf-8")
    return overview, details


def send_email(overview_path: Path, details: Dict[str, Path], recipients: List[str]) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("SMTP_SENDER", user or "")

    if not host or not user or not pwd or not recipients:
        print("Email skipped: SMTP config or recipients missing.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Stock Scan Report - {datetime.now().strftime('%Y-%m-%d')}"
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)

    html = overview_path.read_text(encoding="utf-8")
    detail_links = "".join([f"<li>{p.name}</li>" for p in details.values()])
    html += f"<hr/><p>Detail files generated:</p><ul>{detail_links}</ul>"
    msg.attach(MIMEText(html, "html", "utf-8"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, pwd)
        server.sendmail(sender, recipients, msg.as_string())
    print(f"Email sent to: {recipients}")


def load_universe_from_csv(path: str) -> List[UniverseItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Universe CSV not found: {path}")
    rows = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Support CSV with header "ticker"
    if rows and rows[0].lower() in {"ticker", "symbol"}:
        rows = rows[1:]
    return [UniverseItem(ticker=r) for r in rows]


def load_universe_from_supabase() -> List[UniverseItem]:
    """
    Supabase entrypoint (to be wired with your DB credentials/schema).
    Required env vars to implement later:
      - SUPABASE_URL
      - SUPABASE_SERVICE_ROLE_KEY
      - SUPABASE_UNIVERSE_TABLE (default suggestion: stock_universe)
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    table = os.getenv("SUPABASE_UNIVERSE_TABLE", "companies")
    market = os.getenv("SUPABASE_MARKET", "CN_A")
    ticker_field = os.getenv("SUPABASE_TICKER_FIELD", "code")  # code or ts_code

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")

    try:
        from supabase import create_client
    except Exception as exc:
        raise RuntimeError("Please install supabase client: pip install supabase") from exc

    sb = create_client(url, key)
    query = sb.table(table).select("id,code,ts_code,name,market,groups,sectors,business")
    if market:
        query = query.eq("market", market)
    rows = query.execute().data or []

    items: List[UniverseItem] = []
    for r in rows:
        raw_ticker = (r.get("ts_code") if ticker_field == "ts_code" else r.get("code")) or ""
        ticker = _ticker_to_code(str(raw_ticker))
        if not ticker:
            continue
        tags = []
        for key in ("groups", "sectors"):
            val = r.get(key)
            if isinstance(val, list):
                tags.extend([str(v) for v in val if v])
        items.append(
            UniverseItem(
                ticker=ticker,
                company_id=r.get("id"),
                ts_code=r.get("ts_code"),
                name=r.get("name"),
                trigger_tags=list(dict.fromkeys(tags)) if tags else [],
                business=r.get("business"),
            )
        )

    if not items:
        raise RuntimeError(f"No tickers loaded from Supabase table '{table}'")
    return items


def write_results_to_supabase(
    scan_results: List[ScanResult],
    overview_path: Path,
    detail_paths: Dict[str, Path],
    top_k_scan: int = 20,
    top_k_deep: int = 5,
) -> None:
    """
    Supabase write-back entrypoint (to be wired after your schema is finalized).
    Suggested fields:
      ticker, trade_date, technical_score, fundamental_score, total_score,
      rationale, overview_url, detail_url, sent_at
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    table = os.getenv("SUPABASE_SCAN_RESULTS_TABLE", "scan_results")
    if not url or not key:
        print("Supabase write-back skipped: missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY")
        return

    try:
        from supabase import create_client
    except Exception:
        print("Supabase write-back skipped: install with `pip install supabase`.")
        return

    sb = create_client(url, key)
    trade_date = overview_path.parent.name
    now_iso = datetime.utcnow().isoformat()
    # Generate incremental run_id
    latest = sb.table(table).select("run_id").order("run_id", desc=True).limit(1).execute().data or []
    run_id = (latest[0].get("run_id", 0) if latest else 0) + 1

    rows = []
    ranked = scan_results[:top_k_scan]
    for rank, r in enumerate(ranked, 1):
        code = _ticker_to_code(r.ticker)
        status = "deep_dive" if rank <= top_k_deep else "watch"
        reasons = r.rationale[:500]
        lightweight_summary = "进入重点深挖名单" if status == "deep_dive" else "作为可比公司继续观察"
        rows.append(
            {
                "run_id": run_id,
                "company_id": r.company_id,
                "selection_score": r.total_score,
                "trigger_tags": r.trigger_tags or [],
                "status": status,
                "rank_no": rank,
                "reasons": reasons,
                "lightweight_summary": lightweight_summary,
                "raw_output": {
                    "ticker": r.ticker,
                    "code": code,
                    "ts_code": r.ts_code or _code_to_ts_code(code),
                    "technical_score": r.technical_score,
                    "fundamental_score": r.fundamental_score,
                    "selection_score": r.total_score,
                    "theme_news": r.themed_news or [],
                    "overview_url": str(overview_path),
                    "detail_url": str(detail_paths.get(r.ticker, "")),
                    "trade_date": trade_date,
                    "created_at": now_iso,
                },
            }
        )
    try:
        sb.table(table).insert(rows).execute()
        print(f"Supabase write-back done: {len(rows)} rows -> {table} (run_id={run_id})")
    except Exception as exc:
        print(f"Supabase write-back failed on table '{table}': {exc}")


def main():
    parser = argparse.ArgumentParser(description="Daily scan + deep dive + email pipeline")
    parser.add_argument("--trade-date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top-k-scan", type=int, default=20)
    parser.add_argument("--top-k-deep", type=int, default=5)
    parser.add_argument("--universe-source", choices=["csv", "supabase"], default="csv")
    parser.add_argument("--universe-csv", type=str, default="stock_universe.csv")
    parser.add_argument("--out-dir", type=str, default="daily_reports")
    parser.add_argument("--recipients", type=str, default=os.getenv("REPORT_RECIPIENTS", ""))
    args = parser.parse_args()

    if args.universe_source == "supabase":
        universe_items = load_universe_from_supabase()
    else:
        universe_items = load_universe_from_csv(args.universe_csv)

    config = DEFAULT_CONFIG.copy()
    config["output_language"] = "Chinese"

    scan_results = run_light_scan(universe_items, args.trade_date)
    top_scan = scan_results[: args.top_k_scan]
    top_tickers = [r.ticker for r in top_scan[: args.top_k_deep]]
    deep_reports = run_deep_dive(top_tickers, args.trade_date, config)

    out_dir = Path(args.out_dir) / args.trade_date
    overview_path, detail_paths = render_reports(top_scan, deep_reports, out_dir, args.top_k_deep)

    # Optional: send email
    recipients = [e.strip() for e in args.recipients.split(",") if e.strip()]
    send_email(overview_path, detail_paths, recipients)

    # Optional: write back (entrypoint reserved)
    write_results_to_supabase(
        scan_results,
        overview_path,
        detail_paths,
        top_k_scan=args.top_k_scan,
        top_k_deep=args.top_k_deep,
    )

    # Persist machine-readable scan output
    out_json = out_dir / "scan_results.json"
    out_json.write_text(json.dumps([r.__dict__ for r in scan_results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Overview: {overview_path}")


if __name__ == "__main__":
    main()
