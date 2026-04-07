from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.config import get_config
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    return route_to_vendor("get_news", ticker, start_date, end_date)

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor.
    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back (default 7)
        limit (int): Maximum number of articles to return (default 5)
    Returns:
        str: A formatted string containing global news data
    """
    return route_to_vendor("get_global_news", curr_date, look_back_days, limit)

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    return route_to_vendor("get_insider_transactions", ticker)


@tool
def web_search_news(
    query: Annotated[str, "Search query, e.g. company name + topic"],
    days: Annotated[int, "Recency window in days"] = 7,
    limit: Annotated[int, "Maximum number of results"] = 5,
    preferred_domains_csv: Annotated[str, "Optional comma-separated preferred domains"] = "",
) -> str:
    """
    Search public web news via Google News RSS.
    Use this when vendor news tools return insufficient evidence.
    Prioritizes domains from `preferred_domains_csv` and config `trusted_news_domains`.
    """

    def _fetch_items(raw_query: str):
        q = quote_plus(raw_query)
        url = f"https://news.google.com/rss/search?q={q}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
        with urlopen(url, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        return root.findall(".//item")

    try:
        cfg = get_config()
        cfg_domains = cfg.get("trusted_news_domains", []) or []
        arg_domains = [d.strip() for d in preferred_domains_csv.split(",") if d.strip()]
        preferred_domains = []
        for d in arg_domains + list(cfg_domains):
            if d not in preferred_domains:
                preferred_domains.append(d)

        collected = []
        seen_links = set()
        base_query = f"{query} when:{max(1, days)}d"

        # 1) Preferred-domain search first
        for domain in preferred_domains:
            for item in _fetch_items(f"{base_query} site:{domain}"):
                link = (item.findtext("link") or "").strip()
                if link and link not in seen_links:
                    seen_links.add(link)
                    collected.append((item, domain))
                if len(collected) >= max(1, limit):
                    break
            if len(collected) >= max(1, limit):
                break

        # 2) Fallback to broad query if still insufficient
        if len(collected) < max(1, limit):
            for item in _fetch_items(base_query):
                link = (item.findtext("link") or "").strip()
                if link and link not in seen_links:
                    seen_links.add(link)
                    collected.append((item, None))
                if len(collected) >= max(1, limit):
                    break

        if not collected:
            return f"No web news results for query: {query}"

        lines = [f"## Web news results for: {query}"]
        if preferred_domains:
            lines.append(f"Preferred domains: {', '.join(preferred_domains)}")
        for i, (item, preferred_domain) in enumerate(collected[: max(1, limit)], 1):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            source = ""
            source_node = item.find("{http://search.yahoo.com/mrss/}source")
            if source_node is not None and source_node.text:
                source = source_node.text.strip()
            preferred_tag = f" | preferred:{preferred_domain}" if preferred_domain else ""
            lines.append(f"{i}. [{title}]({link}) | {source} | {pub_date}{preferred_tag}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error searching web news for '{query}': {exc}"
