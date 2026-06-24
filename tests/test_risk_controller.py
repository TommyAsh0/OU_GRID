# ====================================================================
# 风控系统单元测试 (test_risk_controller.py)
# 对应 README 阶段 7「风控系统」
#
# 逐级验证四级风控的触发条件与优先级：
#   层级1 网格级、层级2 标的级、层级3 策略级、层级4 账户级，
# 以及冷却 / 暂停 / 跨年重置等状态机行为。
# 用最小的 Position 桩对象隔离风控逻辑（不依赖真实持仓实现细节）。
# ====================================================================

import unittest

import pandas as pd

from src.grid.position import GridPosition
from src.risk.controller import (
    ACTION_CLOSE_ALL,
    ACTION_CLOSE_LAYER,
    ACTION_CLOSE_SYMBOL,
    ACTION_NONE,
    ACTION_REDUCE_HALF,
    ACTION_STOP_YEAR,
    RiskController,
)

# 初始资金（多数用例以 100 万为基准）。
INIT = 1_000_000.0


class TestAccountLevel(unittest.TestCase):
    """层级4：账户级年度止损（README 7.1，最高优先级）。"""

    def setUp(self):
        self.rc = RiskController(INIT)
        self.rc.roll_year(pd.Timestamp("2023-01-03"))  # 设定年度基准

    def test_none_when_within_limit(self):
        """年度亏损未达 -25% → 无动作。"""
        # 亏损 -10%。
        self.assertEqual(self.rc.check_account_level(900_000), ACTION_NONE)

    def test_stop_year_when_exceeds(self):
        """年度亏损 ≤ -25% → 当年停止策略。"""
        # 亏损 -26%。
        self.assertEqual(self.rc.check_account_level(740_000), ACTION_STOP_YEAR)
        # 触发后即使净值回升，当年仍保持停止。
        self.assertEqual(self.rc.check_account_level(950_000), ACTION_STOP_YEAR)

    def test_year_reset_clears_stop(self):
        """跨年后年度止损标记应被重置。"""
        self.rc.check_account_level(740_000)  # 触发当年停止
        self.assertTrue(self.rc.year_stopped)
        # 进入新一年（峰值需更新以便重置基准）。
        self.rc.update_equity_peak(950_000)
        self.rc.roll_year(pd.Timestamp("2024-01-02"))
        self.assertFalse(self.rc.year_stopped)


class TestStrategyLevel(unittest.TestCase):
    """层级3：策略级回撤（README 7.1）。"""

    def setUp(self):
        self.rc = RiskController(INIT)
        self.date = pd.Timestamp("2023-06-01")

    def test_none_when_small_drawdown(self):
        """回撤 < 15% → 无动作。"""
        # 峰值 100 万，回撤 -10%。
        self.assertEqual(
            self.rc.check_strategy_level(900_000, self.date), ACTION_NONE)

    def test_reduce_half_at_15pct(self):
        """回撤 ≤ -15%（未到 -20%）→ 仓位减半。"""
        # 回撤 -16%。
        self.assertEqual(
            self.rc.check_strategy_level(840_000, self.date), ACTION_REDUCE_HALF)

    def test_close_all_at_20pct(self):
        """回撤 ≤ -20% → 全部清仓并暂停。"""
        # 回撤 -21%。
        self.assertEqual(
            self.rc.check_strategy_level(790_000, self.date), ACTION_CLOSE_ALL)
        # 清仓后应进入策略暂停期。
        self.assertTrue(self.rc.strategy_paused(self.date))

    def test_drawdown_uses_peak(self):
        """回撤以净值峰值为基准，而非初始资金。"""
        self.rc.update_equity_peak(1_200_000)  # 峰值抬升到 120 万
        # 净值 1_020_000 相对峰值回撤 = -15%，应减半。
        self.assertEqual(
            self.rc.check_strategy_level(1_020_000, self.date),
            ACTION_REDUCE_HALF)


class TestSymbolLevel(unittest.TestCase):
    """层级2：标的级浮亏止损（README 7.1）。"""

    def setUp(self):
        self.rc = RiskController(INIT)
        self.pos = GridPosition("TEST", n_layers=5)

    def test_none_when_empty(self):
        """空仓标的 → 无动作。"""
        self.assertEqual(
            self.rc.check_symbol_level(self.pos, price=100.0), ACTION_NONE)

    def test_none_when_small_loss(self):
        """浮亏 < 15% → 无动作。"""
        self.pos.apply_buy(layer=1, price=100.0, quantity=1000)
        # 现价 95 → 浮亏 -5%。
        self.assertEqual(
            self.rc.check_symbol_level(self.pos, price=95.0), ACTION_NONE)

    def test_close_symbol_at_15pct(self):
        """浮亏 ≤ -15% → 清该标的。"""
        self.pos.apply_buy(layer=1, price=100.0, quantity=1000)
        # 现价 84 → 浮亏 -16%（相对投入 10 万）。
        self.assertEqual(
            self.rc.check_symbol_level(self.pos, price=84.0),
            ACTION_CLOSE_SYMBOL)

    def test_cooldown_blocks_reentry(self):
        """清仓后开启冷却期，期内 in_cooldown 为真。"""
        d = pd.Timestamp("2023-06-01")
        self.rc.start_cooldown("TEST", d)
        # 冷却期内（次日）应仍在冷却。
        self.assertTrue(self.rc.in_cooldown("TEST", d + pd.Timedelta(days=1)))
        # 冷却期满后（cooldown_days+1 天）应解除。
        after = d + pd.Timedelta(days=self.rc.cooldown_days + 1)
        self.assertFalse(self.rc.in_cooldown("TEST", after))


class TestLayerLevel(unittest.TestCase):
    """层级1：网格级单层止损（README 7.1）。"""

    def setUp(self):
        self.rc = RiskController(INIT)
        self.pos = GridPosition("TEST", n_layers=5)

    def test_none_when_layer_empty(self):
        """该层未持仓 → 无动作。"""
        self.assertEqual(
            self.rc.check_layer_level(self.pos, layer=1, price=100.0),
            ACTION_NONE)

    def test_none_when_small_layer_loss(self):
        """单层浮亏 < 15% → 无动作。"""
        self.pos.apply_buy(layer=1, price=100.0, quantity=1000)
        self.assertEqual(
            self.rc.check_layer_level(self.pos, layer=1, price=95.0),
            ACTION_NONE)

    def test_close_layer_at_15pct(self):
        """单层 (现价−建仓价)/建仓价 ≤ -15% → 平该层。"""
        self.pos.apply_buy(layer=1, price=100.0, quantity=1000)
        # 现价 84 → 层亏 -16%。
        self.assertEqual(
            self.rc.check_layer_level(self.pos, layer=1, price=84.0),
            ACTION_CLOSE_LAYER)


class TestStateMachine(unittest.TestCase):
    """冷却 / 暂停 / 峰值等状态维护。"""

    def test_equity_peak_monotonic(self):
        """净值峰值只增不减。"""
        rc = RiskController(INIT)
        rc.update_equity_peak(1_100_000)
        rc.update_equity_peak(1_050_000)  # 回落不应降低峰值
        self.assertEqual(rc.equity_peak, 1_100_000)

    def test_strategy_pause_expires(self):
        """策略暂停期满后 strategy_paused 转为假。"""
        rc = RiskController(INIT)
        d = pd.Timestamp("2023-06-01")
        rc.check_strategy_level(790_000, d)  # 触发清仓+暂停
        self.assertTrue(rc.strategy_paused(d))
        after = d + pd.Timedelta(days=rc.cooldown_days + 1)
        self.assertFalse(rc.strategy_paused(after))


if __name__ == "__main__":
    unittest.main()
