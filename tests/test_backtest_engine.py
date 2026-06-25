# ====================================================================
# 回测引擎开关单元测试 (test_backtest_engine.py)
# 对应 README 阶段 8「历史回测」及自定义回测脚本 bt_custom.py 的
# 「自动开仓系统」开关（enable_open）。
#
# 重点锁定 enable_open=False 的行为契约：
#   - 引擎仍逐日推进，照常计算并记录 Regime / position_scale（供可视化）；
#   - 但不生成任何买入开仓单，全程从空仓起步 → 无成交、净值恒为初始资金。
# 同时验证默认 enable_open=True 时能正常产生成交（向后兼容）。
#
# 用离线「合成行情」构造样本，避免依赖网络与真实数据源。
# ====================================================================

import unittest

from src.backtest.engine import BacktestEngine
from src.data.fetcher import generate_synthetic_daily
from src.data.processor import DataProcessor

# 测试用初始资金与标的、区间（合成数据，离线可复现）。
INIT = 1_000_000.0
TS_CODE = "TEST.SH"
START = "20210101"
END = "20221231"


def _make_processed_df():
    """生成已清洗并含 MA/ATR/Z 指标的合成行情（约两年、数百行）。"""
    raw = generate_synthetic_daily(TS_CODE, START, END)
    return DataProcessor().process(raw)


class TestEnableOpenOff(unittest.TestCase):
    """enable_open=False：仅算因子 / Regime，不开仓、无成交。"""

    def setUp(self):
        self.df = _make_processed_df()
        # 关闭自动开仓：保留 Regime 计算，关闭风控自动交易。
        bt = BacktestEngine(
            TS_CODE, self.df, INIT, k=1.0,
            enable_regime=True, enable_risk=False, enable_open=False,
        )
        self.result = bt.run()

    def test_no_fills(self):
        """关闭自动开仓后不应产生任何成交。"""
        self.assertEqual(len(self.result["fills"]), 0)

    def test_equity_constant_at_capital(self):
        """全程无交易 → 净值恒等于初始资金。"""
        equity = self.result["equity_curve"]
        self.assertGreater(len(equity), 0)
        self.assertAlmostEqual(equity.iloc[0], INIT)
        self.assertAlmostEqual(equity.iloc[-1], INIT)
        # 任一交易日净值都不偏离初始资金。
        self.assertTrue((equity == INIT).all())

    def test_daily_log_records_regime(self):
        """每日快照仍记录 regime / position_scale（供可视化数据准备）。"""
        log = self.result["daily_log"]
        self.assertIn("regime", log.columns)
        self.assertIn("position_scale", log.columns)
        # 应有逐日记录，且 regime 取值落在合法档位 {0,1,2}。
        self.assertGreater(len(log), 0)
        self.assertTrue(set(log["regime"].unique()).issubset({0, 1, 2}))
        # position_scale 与 regime 一一对应（绿1.0/黄0.5/红0.0）。
        self.assertTrue(
            set(log["position_scale"].unique()).issubset({0.0, 0.5, 1.0}))

    def test_no_holdings(self):
        """从空仓起步且不开仓 → 任一日持仓层数均为 0。"""
        log = self.result["daily_log"]
        self.assertEqual(int(log["holding_layers"].max()), 0)


class TestEnableOpenOn(unittest.TestCase):
    """默认 enable_open=True：正常开仓回测应产生成交（向后兼容）。"""

    def setUp(self):
        self.df = _make_processed_df()
        bt = BacktestEngine(TS_CODE, self.df, INIT, k=1.0)
        self.result = bt.run()

    def test_produces_fills(self):
        """默认配置下应至少有一笔成交。"""
        self.assertGreater(len(self.result["fills"]), 0)

    def test_daily_log_has_regime_columns(self):
        """每日快照同样记录 regime / position_scale。"""
        log = self.result["daily_log"]
        self.assertIn("regime", log.columns)
        self.assertIn("position_scale", log.columns)


if __name__ == "__main__":
    unittest.main()
