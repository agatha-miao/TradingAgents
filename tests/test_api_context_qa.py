import unittest
from datetime import date

from tradingagents.api.server import (
    _build_qa_context,
    _extract_signal_rows,
    _passes_trade_audit,
    _pick_item_rows,
)


class TestAPIContextQA(unittest.TestCase):
    def test_pick_item_rows_matches_ticker_code_and_ts_code(self):
        rows = [
            {"ticker": "688256.SH", "code": "688256", "ts_code": "688256.SH"},
            {"ticker": "300308.SZ", "code": "300308", "ts_code": "300308.SZ"},
        ]
        by_ticker = _pick_item_rows(rows, "688256.SH")
        by_code = _pick_item_rows(rows, "688256")
        by_ts_code = _pick_item_rows(rows, "300308.SZ")
        self.assertEqual(len(by_ticker), 1)
        self.assertEqual(len(by_code), 1)
        self.assertEqual(len(by_ts_code), 1)
        self.assertEqual(by_ticker[0]["code"], "688256")
        self.assertEqual(by_ts_code[0]["code"], "300308")

    def test_build_qa_context_contains_report_summary_and_item_rows(self):
        report = {
            "title": "每日AI投研简报",
            "report_date": "2026-04-08",
            "summary": "今日聚焦算力链。",
        }
        item_rows = [{"ticker": "688256.SH", "conclusion": "景气度上行"}]
        context = _build_qa_context(report, item_rows, "688256.SH")
        self.assertIn("每日AI投研简报", context)
        self.assertIn("今日聚焦算力链", context)
        self.assertIn("688256.SH", context)
        self.assertIn("景气度上行", context)

    def test_extract_signal_rows_deduplicates_by_ticker_and_sorts_by_score(self):
        rows = [
            {"ticker": "AAA", "selection_score": 1.2},
            {"ticker": "BBB", "selection_score": 3.4},
            {"ticker": "AAA", "selection_score": 2.0},
        ]
        out = _extract_signal_rows(rows, top_n=2)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["ticker"], "BBB")
        self.assertEqual(out[1]["ticker"], "AAA")

    def test_passes_trade_audit_blocks_recent_order(self):
        recent_orders = [{"ticker": "688256.SH", "created_at": "2026-04-07T09:30:00"}]
        ok, reason = _passes_trade_audit(
            ticker="688256.SH",
            report_day=date(2026, 4, 8),
            recent_orders=recent_orders,
            cooldown_days=3,
        )
        self.assertFalse(ok)
        self.assertIn("cooldown_block", reason)

    def test_passes_trade_audit_allows_old_order(self):
        recent_orders = [{"ticker": "688256.SH", "created_at": "2026-03-01T09:30:00"}]
        ok, reason = _passes_trade_audit(
            ticker="688256.SH",
            report_day=date(2026, 4, 8),
            recent_orders=recent_orders,
            cooldown_days=3,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "approved")


if __name__ == "__main__":
    unittest.main()
