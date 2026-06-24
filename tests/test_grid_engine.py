# ====================================================================
# 网格引擎单元测试 (test_grid_engine.py)
# 对应 README 阶段 5「网格引擎设计」
#
# 覆盖网格价格计算、仓位方案权重、下单数量取整、信号生成
# （空仓挂买 / 持仓挂卖）以及网格深度校验（D = n×K ≤ 2.5×s）。
# ====================================================================

import unittest

from src.grid.engine import GridEngine
from src.grid.order import OrderSide
from src.grid.position import GridPosition


class TestGridPrices(unittest.TestCase):
    """grid_prices 价格计算测试。"""

    def setUp(self):
        # 5 层、K=1.0、等权，资金 100 万。
        self.eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                              position_mode="equal")

    def test_buy_prices_descend(self):
        """买入价应自上而下递减：买入价_i = MA − i×K×ATR。"""
        prices = self.eng.grid_prices(ma=100.0, atr=2.0)
        # 第 1 层 = 100 − 1×1×2 = 98；第 2 层 = 96；……
        self.assertAlmostEqual(prices[1]["buy"], 98.0)
        self.assertAlmostEqual(prices[2]["buy"], 96.0)
        self.assertAlmostEqual(prices[5]["buy"], 90.0)

    def test_sell_equals_buy_plus_spacing(self):
        """卖出价 = 买入价 + K×ATR（回到上一层价位止盈）。"""
        prices = self.eng.grid_prices(ma=100.0, atr=2.0)
        for i in range(1, 6):
            self.assertAlmostEqual(
                prices[i]["sell"], prices[i]["buy"] + 1.0 * 2.0)


class TestLayerWeights(unittest.TestCase):
    """仓位方案权重测试（README 5.3）。"""

    def test_equal_weights(self):
        """等权方案：每层权重 = 1/n，且总和为 1。"""
        eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                         position_mode="equal")
        for i in range(1, 6):
            self.assertAlmostEqual(eng.layer_weights[i], 0.2)
        self.assertAlmostEqual(sum(eng.layer_weights.values()), 1.0)

    def test_linear_weights(self):
        """线性方案：权重比 1:1.5:2:2.5:3 归一化后为 0.1…0.3，总和为 1。"""
        eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                         position_mode="linear")
        expected = {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3}
        for i in range(1, 6):
            self.assertAlmostEqual(eng.layer_weights[i], expected[i])
        self.assertAlmostEqual(sum(eng.layer_weights.values()), 1.0)


class TestComputeQuantity(unittest.TestCase):
    """compute_quantity 下单数量取整测试。"""

    def setUp(self):
        self.eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                              position_mode="equal")

    def test_rounds_to_lot(self):
        """股数应向下取整到 100 股（A 股一手）的整数倍。"""
        # 层资金 = 20 万；买价 100 → 2000 股（恰为整百）。
        qty = self.eng.compute_quantity(layer=1, buy_price=100.0)
        self.assertEqual(qty % 100, 0)
        self.assertEqual(qty, 2000)

    def test_non_integer_lot_floored(self):
        """非整百股数应向下取整。"""
        # 层资金 20 万；买价 33 → 6060.6 股 → 取整 6000 股。
        qty = self.eng.compute_quantity(layer=1, buy_price=33.0)
        self.assertEqual(qty, 6000)

    def test_zero_when_insufficient(self):
        """单层资金不足一手时返回 0。"""
        small = GridEngine("TEST", 100.0, k=1.0, n_layers=5,
                           position_mode="equal")
        # 层资金 = 20 元；买价 100 → 不足 100 股 → 0。
        self.assertEqual(small.compute_quantity(layer=1, buy_price=100.0), 0)

    def test_zero_on_nonpositive_price(self):
        """买价非正时返回 0（避免除零 / 负数股）。"""
        self.assertEqual(self.eng.compute_quantity(layer=1, buy_price=0.0), 0)
        self.assertEqual(self.eng.compute_quantity(layer=1, buy_price=-5.0), 0)


class TestGenerateOrders(unittest.TestCase):
    """generate_orders 信号生成测试（README 5.4）。"""

    def setUp(self):
        self.eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                              position_mode="equal")
        self.pos = GridPosition("TEST", n_layers=5)

    def test_all_buys_when_empty(self):
        """完全空仓 → 每层都挂买入单。"""
        orders = self.eng.generate_orders(ma=100.0, atr=2.0, position=self.pos)
        self.assertEqual(len(orders), 5)
        self.assertTrue(all(o.side == OrderSide.BUY for o in orders))

    def test_sell_when_holding(self):
        """某层持仓 → 该层改挂卖出单（数量为持仓量）。"""
        self.pos.apply_buy(layer=2, price=96.0, quantity=2000)
        orders = self.eng.generate_orders(ma=100.0, atr=2.0, position=self.pos)
        # 第 2 层应为卖单，其余为买单。
        sells = [o for o in orders if o.side == OrderSide.SELL]
        self.assertEqual(len(sells), 1)
        self.assertEqual(sells[0].layer, 2)
        self.assertEqual(sells[0].quantity, 2000)

    def test_no_orders_on_invalid_indicator(self):
        """指标非法（ATR/MA ≤ 0 或 NaN）→ 不产生任何信号。"""
        self.assertEqual(
            self.eng.generate_orders(ma=100.0, atr=0.0, position=self.pos), [])
        self.assertEqual(
            self.eng.generate_orders(ma=0.0, atr=2.0, position=self.pos), [])
        self.assertEqual(
            self.eng.generate_orders(
                ma=float("nan"), atr=2.0, position=self.pos), [])

    def test_skips_negative_buy_price(self):
        """极深层买价为负时跳过该层（不挂买单）。"""
        # MA=5, ATR=2 → 第 3 层买价 = 5 − 3×2 = −1 < 0，应被跳过。
        orders = self.eng.generate_orders(ma=5.0, atr=2.0, position=self.pos)
        layers_with_buy = [o.layer for o in orders if o.side == OrderSide.BUY]
        self.assertNotIn(3, layers_with_buy)
        self.assertNotIn(4, layers_with_buy)
        self.assertNotIn(5, layers_with_buy)


class TestValidateDepth(unittest.TestCase):
    """validate_depth 网格深度校验测试（README 5.2）。"""

    def setUp(self):
        # D = n×K = 5×1.0 = 5.0。
        self.eng = GridEngine("TEST", 1_000_000, k=1.0, n_layers=5,
                              position_mode="equal")

    def test_pass_when_shallow_enough(self):
        """D ≤ 2.5×s 时通过：s=3 → 2.5×3=7.5 ≥ 5 → True。"""
        self.assertTrue(self.eng.validate_depth(steady_std=3.0))

    def test_fail_when_too_deep(self):
        """D > 2.5×s 时不通过：s=1.5 → 2.5×1.5=3.75 < 5 → False。"""
        self.assertFalse(self.eng.validate_depth(steady_std=1.5))

    def test_fail_on_nonpositive_std(self):
        """稳态标准差非正 / 非法 → 直接不通过。"""
        self.assertFalse(self.eng.validate_depth(steady_std=0.0))
        self.assertFalse(self.eng.validate_depth(steady_std=-1.0))


if __name__ == "__main__":
    unittest.main()
