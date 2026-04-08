from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from supabase import create_client


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_supabase_client():
    url = _require_env("SUPABASE_URL")
    key = _require_env("SUPABASE_SERVICE_ROLE_KEY")
    return create_client(url, key)


def _normalize_ticker(value: str) -> str:
    return (value or "").strip().upper()


def _pick_item_rows(rows: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    target = _normalize_ticker(ticker)
    if not target:
        return rows
    selected: list[dict[str, Any]] = []
    for row in rows:
        values = [
            str(row.get("ticker", "")),
            str(row.get("code", "")),
            str(row.get("ts_code", "")),
            str(row.get("symbol", "")),
        ]
        normalized_values = {_normalize_ticker(v) for v in values if v and v != "None"}
        if target in normalized_values:
            selected.append(row)
    return selected


def _build_qa_context(report: dict[str, Any], item_rows: list[dict[str, Any]], ticker: str) -> str:
    ticker_text = _normalize_ticker(ticker) or "未指定"
    lines = [
        "你是投研问答助手。必须基于给定上下文回答，不允许编造。",
        f"【当前问题股票】{ticker_text}",
        f"【报告标题】{report.get('title', '')}",
        f"【报告日期】{report.get('report_date', '')}",
        f"【报告摘要】{report.get('summary', '')}",
        "【可引用条目】",
    ]
    if not item_rows:
        lines.append("无可引用条目。")
    for idx, row in enumerate(item_rows[:5], 1):
        lines.append(f"- 条目{idx}: {row}")
    return "\n".join(lines)


def _ask_deepseek(context: str, question: str) -> str:
    api_key = _require_env("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    timeout_seconds = float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
    url = f"{base_url.rstrip('/')}/chat/completions"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": context},
                {"role": "user", "content": question},
            ],
        },
        timeout=timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek response missing choices.")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        raise RuntimeError("DeepSeek response missing content.")
    return str(content)


class QAAskRequest(BaseModel):
    report_id: int = Field(..., ge=1)
    question: str = Field(..., min_length=3)
    ticker: str = Field(..., min_length=1)


class SimRunRequest(BaseModel):
    report_id: int | None = Field(default=None, ge=1)
    top_n: int = Field(default=3, ge=1, le=20)
    order_amount: float = Field(default=10000, gt=0)
    cooldown_days: int = Field(default=3, ge=0, le=60)


app = FastAPI(title="TradingAgents API", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def home_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TradingAgents 报告中心</title>
  <style>
    :root {
      --bg: #f6f8fc;
      --card: #ffffffcc;
      --text: #0f172a;
      --muted: #64748b;
      --brand: #2563eb;
      --brand2: #7c3aed;
      --line: #e2e8f0;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: radial-gradient(1200px 600px at 0% 0%, #dbeafe 0%, transparent 50%),
                  radial-gradient(900px 500px at 100% 0%, #ede9fe 0%, transparent 45%),
                  var(--bg);
    }
    .container { max-width: 1120px; margin: 0 auto; padding: 24px 16px 40px; }
    .hero {
      background: linear-gradient(135deg, #1d4ed8, #7c3aed);
      color: #fff;
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 16px 40px rgba(37, 99, 235, 0.25);
    }
    .hero h1 { margin: 0 0 10px; font-size: clamp(24px, 4vw, 40px); }
    .hero p { margin: 0; opacity: .95; line-height: 1.7; }
    .actions { margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; }
    .btn {
      display: inline-flex; align-items: center; justify-content: center;
      border-radius: 999px; padding: 10px 16px; text-decoration: none; font-weight: 600;
    }
    .btn-primary { background: #fff; color: #1e40af; }
    .btn-ghost { background: #ffffff22; color: #fff; border: 1px solid #ffffff44; }
    .cards { margin-top: 18px; display: grid; gap: 12px; grid-template-columns: repeat(3, 1fr); }
    .card {
      background: var(--card);
      backdrop-filter: blur(6px);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }
    .muted { color: var(--muted); font-size: 14px; }
    @media (max-width: 900px) { .cards { grid-template-columns: 1fr; } .hero { padding: 20px; border-radius: 16px; } }
  </style>
</head>
<body>
  <div class="container">
    <section class="hero">
      <h1>TradingAgents 报告与模拟中枢</h1>
      <p>单用户专业版界面：查看日报、按股票上下文提问、并将 Agent 结论自动转为模拟交易（含交易前审核）。</p>
      <div class="actions">
        <a class="btn btn-primary" href="/reports">进入报告中心</a>
        <a class="btn btn-ghost" href="/api/sim/orders">查看模拟订单 JSON</a>
      </div>
    </section>
    <section class="cards">
      <div class="card"><strong>上下文问答</strong><div class="muted">必须先选报告中的股票，再向 DeepSeek 发问。</div></div>
      <div class="card"><strong>自动下单</strong><div class="muted">根据 Agent 结论选股，自动生成 BUY 候选单。</div></div>
      <div class="card"><strong>预交易审核</strong><div class="muted">cooldown 阻止短期频繁交易同一标的。</div></div>
    </section>
  </div>
</body>
</html>
"""


@app.get("/reports", response_class=HTMLResponse)
def reports_page() -> str:
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>报告列表</title>
  <style>
    :root { --bg:#f8fafc; --text:#0f172a; --muted:#64748b; --line:#e2e8f0; --card:#fff; --brand:#2563eb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    .container { max-width: 1120px; margin: 0 auto; padding: 20px 14px 36px; }
    .top { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
    .title { font-size: clamp(22px, 3vw, 30px); margin:0; }
    .card { border:1px solid var(--line); border-radius:14px; padding:14px; margin-bottom:10px; background:var(--card); box-shadow:0 6px 18px rgba(15,23,42,.04); }
    .muted { color: var(--muted); font-size: 13px; line-height:1.5; }
    .badge { display:inline-flex; font-size:12px; color:#1d4ed8; background:#dbeafe; border-radius:999px; padding:4px 8px; }
    a { color: var(--brand); text-decoration: none; font-weight: 600; }
  </style>
</head>
<body>
  <div class="container">
    <div class="top">
      <h1 class="title">报告列表</h1>
      <a href="/">返回首页</a>
    </div>
    <div id="list">加载中...</div>
  </div>
  <script>
    (async () => {
      const res = await fetch('/api/reports?limit=50');
      const data = await res.json();
      const box = document.getElementById('list');
      const rows = data.reports || [];
      if (!rows.length) { box.innerHTML = '<p>暂无报告</p>'; return; }
      box.innerHTML = rows.map(r => `
        <div class="card">
          <div class="badge">日报</div>
          <div><b>${r.title || '(无标题)'}</b></div>
          <div class="muted">report_id=${r.id} | 日期=${r.report_date || ''} | 渠道=${r.channel || ''}</div>
          <div style="margin-top:6px;"><a href="/reports/${r.id}/view">查看详情与问答 →</a></div>
        </div>
      `).join('');
    })();
  </script>
</body>
</html>
"""


@app.get("/reports/{report_id}/view", response_class=HTMLResponse)
def report_detail_page(report_id: int) -> str:
    return f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>报告详情</title>
  <style>
    :root {{ --bg:#f8fafc; --card:#fff; --line:#e2e8f0; --text:#0f172a; --muted:#64748b; --brand:#2563eb; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px 14px 32px; }}
    .row {{ display: grid; grid-template-columns: 1.45fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px; margin-bottom: 10px; background: var(--card); box-shadow:0 6px 18px rgba(15,23,42,.04); }}
    .muted {{ color: var(--muted); font-size: 13px; line-height: 1.6; }}
    textarea {{ width: 100%; min-height: 110px; border-radius: 10px; border:1px solid var(--line); padding:10px; }}
    select, button {{ padding: 9px 10px; border-radius: 10px; border:1px solid var(--line); }}
    button {{ background: #2563eb; color: #fff; border-color: #2563eb; font-weight: 600; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
    .top {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; margin-bottom:8px; }}
    .title {{ margin: 0; font-size: clamp(20px, 3vw, 28px); }}
    @media (max-width: 900px) {{
      .row {{ grid-template-columns: 1fr; }}
      .container {{ padding: 14px 10px 22px; }}
      .card {{ border-radius: 12px; }}
    }}
  </style>
</head>
<body>
  <div class="container">
  <div class="top">
    <h1 class="title">报告详情（ID={report_id}）</h1>
    <a href="/reports">← 返回列表</a>
  </div>
  <div class="row">
    <div>
      <div class="card"><h3>报告概览</h3><pre id="report">加载中...</pre></div>
      <div class="card"><h3>条目（选择某个股票用于问答）</h3><div id="items">加载中...</div></div>
    </div>
    <div>
      <div class="card">
        <h3>DeepSeek 上下文问答</h3>
        <div class="muted">不是从零回答：会引用你选择股票在该报告中的条目做回答。</div>
        <p><label>股票：</label><select id="ticker"></select></p>
        <p><textarea id="q" placeholder="例如：结合这份报告，这只票未来5个交易日主要风险是什么？"></textarea></p>
        <p><button id="ask">提问</button></p>
        <div class="muted" id="meta"></div>
        <pre id="answer"></pre>
      </div>
    </div>
  </div>
  </div>

  <script>
    const reportId = {report_id};
    let itemRows = [];
    const tickerSel = document.getElementById('ticker');
    const reportBox = document.getElementById('report');
    const itemsBox = document.getElementById('items');
    const answerBox = document.getElementById('answer');
    const metaBox = document.getElementById('meta');

    function rowTicker(r) {{
      return (r.ticker || r.ts_code || r.code || r.symbol || '').toString();
    }}

    (async () => {{
      const [reportRes, itemsRes] = await Promise.all([
        fetch(`/api/reports/${{reportId}}`),
        fetch(`/api/reports/${{reportId}}/items?limit=200`),
      ]);
      const reportData = await reportRes.json();
      const itemsData = await itemsRes.json();
      reportBox.textContent = JSON.stringify(reportData.report || {{}}, null, 2);
      itemRows = itemsData.items || [];
      const tickers = [...new Set(itemRows.map(rowTicker).filter(Boolean))];
      tickerSel.innerHTML = tickers.map(t => `<option value="${{t}}">${{t}}</option>`).join('');
      itemsBox.innerHTML = itemRows.map(r => `
        <div class="card">
          <div><b>${{rowTicker(r) || '(未知代码)'}}</b></div>
          <div class="muted">${{(r.reason || r.reasons || r.summary || '').toString().slice(0, 180)}}</div>
        </div>
      `).join('') || '<p>该报告暂无条目。</p>';
    }})();

    document.getElementById('ask').addEventListener('click', async () => {{
      answerBox.textContent = '思考中...';
      metaBox.textContent = '';
      const question = document.getElementById('q').value.trim();
      const ticker = tickerSel.value;
      if (!ticker) {{
        answerBox.textContent = '请先选择股票。';
        return;
      }}
      if (!question) {{
        answerBox.textContent = '请输入问题。';
        return;
      }}
      const resp = await fetch('/api/qa/ask', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ report_id: reportId, ticker, question }})
      }});
      const data = await resp.json();
      if (!resp.ok) {{
        answerBox.textContent = `请求失败: ${{data.detail || 'unknown error'}}`;
        return;
      }}
      metaBox.textContent = `引用条目数: ${{data.citations_count}}`;
      answerBox.textContent = data.answer || '';
    }});
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _parse_report_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return datetime.utcnow().date()
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return datetime.utcnow().date()


def _extract_score(row: dict[str, Any]) -> float:
    for key in ("selection_score", "total_score", "score", "rank_score"):
        value = row.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return 0.0


def _extract_signal_rows(item_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    ranked = sorted(item_rows, key=_extract_score, reverse=True)
    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in ranked:
        ticker = _normalize_ticker(
            str(row.get("ticker") or row.get("ts_code") or row.get("code") or row.get("symbol") or "")
        )
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        signals.append({"ticker": ticker, "score": _extract_score(row), "source_row": row})
        if len(signals) >= top_n:
            break
    return signals


def _passes_trade_audit(
    ticker: str,
    report_day: date,
    recent_orders: list[dict[str, Any]],
    cooldown_days: int,
) -> tuple[bool, str]:
    for order in recent_orders:
        order_ticker = _normalize_ticker(str(order.get("ticker", "")))
        if order_ticker != ticker:
            continue
        created_raw = order.get("created_at") or order.get("order_date")
        if not created_raw:
            continue
        try:
            created_day = _parse_report_date(created_raw)
        except Exception:
            continue
        if (report_day - created_day).days < cooldown_days:
            return False, f"cooldown_block: last order on {created_day.isoformat()}"
    return True, "approved"


@app.get("/api/reports")
def list_reports(limit: int = 20) -> dict[str, Any]:
    sb = _get_supabase_client()
    limit = max(1, min(limit, 100))
    data = (
        sb.table("output_reports")
        .select("*")
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return {"reports": data.data or []}


@app.get("/api/reports/{report_id}")
def get_report(report_id: int) -> dict[str, Any]:
    sb = _get_supabase_client()
    data = sb.table("output_reports").select("*").eq("id", report_id).limit(1).execute().data or []
    if not data:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"report": data[0]}


@app.get("/api/reports/{report_id}/items")
def get_report_items(report_id: int, limit: int = 100) -> dict[str, Any]:
    sb = _get_supabase_client()
    limit = max(1, min(limit, 500))
    data = (
        sb.table("output_report_items")
        .select("*")
        .eq("report_id", report_id)
        .limit(limit)
        .execute()
    )
    return {"items": data.data or []}


@app.get("/api/tickers/{ticker}/latest")
def get_ticker_latest(ticker: str, limit: int = 20) -> dict[str, Any]:
    sb = _get_supabase_client()
    limit = max(1, min(limit, 100))
    rows = (
        sb.table("output_report_items")
        .select("*")
        .order("report_id", desc=True)
        .limit(limit * 5)
        .execute()
        .data
        or []
    )
    items = _pick_item_rows(rows, ticker)[:limit]
    return {"ticker": ticker, "items": items}


@app.post("/api/qa/ask")
def ask_with_context(payload: QAAskRequest) -> dict[str, Any]:
    sb = _get_supabase_client()
    report_rows = (
        sb.table("output_reports").select("*").eq("id", payload.report_id).limit(1).execute().data or []
    )
    if not report_rows:
        raise HTTPException(status_code=404, detail=f"Report {payload.report_id} not found")
    report = report_rows[0]
    item_rows = (
        sb.table("output_report_items")
        .select("*")
        .eq("report_id", payload.report_id)
        .limit(200)
        .execute()
        .data
        or []
    )
    selected_rows = _pick_item_rows(item_rows, payload.ticker)
    if not selected_rows:
        raise HTTPException(
            status_code=404,
            detail=f"No context rows found for ticker '{payload.ticker}' in report {payload.report_id}",
        )
    context = _build_qa_context(report, selected_rows, payload.ticker)
    try:
        answer = _ask_deepseek(context=context, question=payload.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"DeepSeek request failed: {exc}") from exc
    return {
        "report_id": payload.report_id,
        "ticker": payload.ticker,
        "question": payload.question,
        "citations_count": len(selected_rows),
        "answer": answer,
    }


@app.get("/api/sim/orders")
def list_sim_orders(limit: int = 100) -> dict[str, Any]:
    sb = _get_supabase_client()
    limit = max(1, min(limit, 500))
    rows = (
        sb.table("sim_orders")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
        .data
        or []
    )
    return {"orders": rows}


@app.post("/api/sim/run-daily")
def run_daily_sim(payload: SimRunRequest) -> dict[str, Any]:
    sb = _get_supabase_client()
    if payload.report_id is None:
        latest = (
            sb.table("output_reports")
            .select("*")
            .order("report_date", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not latest:
            raise HTTPException(status_code=404, detail="No reports available")
        report = latest[0]
    else:
        rows = sb.table("output_reports").select("*").eq("id", payload.report_id).limit(1).execute().data or []
        if not rows:
            raise HTTPException(status_code=404, detail=f"Report {payload.report_id} not found")
        report = rows[0]

    report_id = int(report["id"])
    report_day = _parse_report_date(report.get("report_date"))
    item_rows = (
        sb.table("output_report_items")
        .select("*")
        .eq("report_id", report_id)
        .limit(500)
        .execute()
        .data
        or []
    )
    if not item_rows:
        raise HTTPException(status_code=404, detail=f"No output_report_items for report {report_id}")

    recent_orders = (
        sb.table("sim_orders")
        .select("*")
        .gte("created_at", (report_day - timedelta(days=payload.cooldown_days + 5)).isoformat())
        .limit(2000)
        .execute()
        .data
        or []
    )
    signals = _extract_signal_rows(item_rows, payload.top_n)
    accepted_orders: list[dict[str, Any]] = []
    rejected_orders: list[dict[str, Any]] = []
    for signal in signals:
        ticker = signal["ticker"]
        passed, reason = _passes_trade_audit(
            ticker=ticker,
            report_day=report_day,
            recent_orders=recent_orders,
            cooldown_days=payload.cooldown_days,
        )
        order_row = {
            "report_id": report_id,
            "order_date": report_day.isoformat(),
            "ticker": ticker,
            "side": "BUY",
            "order_amount": payload.order_amount,
            "signal_score": signal["score"],
            "audit_status": "approved" if passed else "rejected",
            "audit_reason": reason,
        }
        if passed:
            accepted_orders.append(order_row)
        else:
            rejected_orders.append(order_row)

    if accepted_orders:
        sb.table("sim_orders").insert(accepted_orders).execute()
    if rejected_orders:
        sb.table("sim_order_audits").insert(rejected_orders).execute()

    return {
        "report_id": report_id,
        "report_date": report_day.isoformat(),
        "signals": signals,
        "accepted_count": len(accepted_orders),
        "rejected_count": len(rejected_orders),
        "accepted_orders": accepted_orders,
        "rejected_orders": rejected_orders,
    }


def main() -> None:
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("tradingagents.api.server:app", host=host, port=port, reload=False)
