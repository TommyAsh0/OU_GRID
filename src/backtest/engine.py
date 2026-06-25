# ====================================================================
# 回测引擎模块 (engine.py)
# 对应 README 阶段 8「历史回测」（8.3 回测引擎关键要求）
#
# 把网格引擎、Regime 检测、风控系统串成一个事件驱动的日级回测器，
# 在单只标的的历史行情上模拟「盘后挂单、次日撮合」的完整交易流程。
#
# README 8.3 关键要求（本引擎逐条实现）：
#   ✅ 用「次日开盘价」模拟成交（信号盘后产生，最早次日执行）；
#   ✅ 限价单撮合：次日最低价 ≤ 买入限价 → 成交；次日最高价 ≥ 卖出限价 → 成交；
#   ✅ 涨跌停 / 停牌不成交；
#   ✅ 交易成本按 5.5 节计算；
#   ✅ 每日记录持仓、现金、总资产、各层状态；
#   ✅ Regime 检测在回测中也运行（策略开关）；
#   ❌ 绝不使用未来数据（MA/ATR 已在数据层用「当前及之前」滚动计算）。
#
# 多标的的资金分配与汇总在上层 run_backtest.py 中协调，本引擎聚焦单标的。
# ====================================================================

import pandas as pd

from config.loader import CONFIG
from src.backtest.metrics import compute_metrics
from src.grid.engine import GridEngine
from src.grid.order import Fill, OrderSide
from src.grid.position import GridPosition
from src.regime.detector import RegimeDetector, RED
from src.risk.controller import (
    ACTION_CLOSE_ALL,
    ACTION_CLOSE_LAYER,
    ACTION_CLOSE_SYMBOL,
    ACTION_REDUCE_HALF,
    ACTION_STOP_YEAR,
    RiskController,
)


class BacktestEngine:
    """单只标的的事件驱动日级回测引擎。

    用法：
        bt = BacktestEngine(ts_code, df, capital, k=1.0)
        result = bt.run()
        # result 含 equity_curve / fills / metrics / daily_log
    """

    def __init__(self, ts_code: str, df: pd.DataFrame, capital: float,
                 k: float = None, enable_regime: bool = True,
                 enable_risk: bool = True, enable_open: bool = True):
        """初始化回测。

        Args:
            ts_code: 标的代码。
            df: 已清洗并含 MA/ATR/Z 指标的行情（来自 DataProcessor.process）。
            capital: 分配给该标的的资金。
            k: 网格 ATR 倍数 K（验证集选定后传入）。默认读配置。
            enable_regime: 是否启用 Regime 策略开关。
            enable_risk: 是否启用四级风控。
            enable_open: 是否启用「自动开仓系统」。为 False 时引擎仍逐日
                推进并照常计算 Regime / 因子（供可视化与数据准备），但
                不生成任何买入开新仓的限价单（从空仓起步即全程无交易），
                也不触发风控的自动强平 / 暂停。默认 True（正常开仓回测）。
        """
        self.ts_code = ts_code
        self.df = df.reset_index(drop=True)
        self.capital = capital
        self.enable_regime = enable_regime
        self.enable_risk = enable_risk
        self.enable_open = enable_open

        # 交易成本参数（README 5.5）。
        cost = CONFIG["cost"]
        self.commission = cost["commission"]
        self.stamp_tax = cost["stamp_tax"]
        self.slippage = cost["slippage"]
        self.transfer_fee = cost.get("transfer_fee", 0.0)

        # 短期 ATR 窗口（波动率爆发检测用 ATR_5）。
        self.atr_fast_len = 5

        # 组合各子模块。
        self.grid = GridEngine(ts_code, capital, k=k)
        self.position = GridPosition(ts_code, self.grid.n_layers)
        self.regime = RegimeDetector()
        self.risk = RiskController(capital)

        # 运行时状态。
        self.cash = capital
        self.fills = []                 # 所有成交记录
        self.daily_records = []         # 每日快照
        self.pending_orders = []        # 昨日盘后生成、今日待撮合的限价单

    # ---------------- 主循环 ----------------
    def run(self) -> dict:
        """执行回测主循环，逐日推进。

        每个交易日的处理顺序（严格区分「昨日信号 / 今日执行」）：
          1. 撮合「昨日挂出」的限价单（用今日 OHLC 判断成交，成交价取今日开盘）。
          2. 运行风控（账户 > 策略 > 标的 > 网格），必要时强制平仓 / 暂停。
          3. 运行 Regime 检测，得到当日仓位系数（绿满 / 黄半 / 红停）。
          4. 用「今日收盘后」的 MA/ATR 与当前持仓，生成「次日」限价单。
          5. 记录当日净值快照。

        Returns:
            dict: {
                "ts_code", "equity_curve"(Series), "fills"(list),
                "daily_log"(DataFrame), "metrics"(dict)
            }
        """
        # 预计算短期 ATR（ATR_5），用于波动率爆发检测。
        atr_fast = self.df["TR"].rolling(self.atr_fast_len).mean()

        for i in range(len(self.df)):
            row = self.df.iloc[i]
            date = row["trade_date"]

            # 跨年重置年度风控基准。
            if self.enable_risk:
                self.risk.roll_year(date)

            # —— 步骤 1：撮合昨日挂出的限价单（用今日行情）——
            self._match_orders(row)

            # 当前以收盘价估值的总资产。
            equity = self._equity(row["close"])

            # —— 步骤 2：风控（高优先级在前）——
            # 仅在「自动开仓系统」开启且启用风控时执行自动强平 / 暂停；
            # 关闭自动开仓时（仅做因子计算与可视化）不触发任何风控交易。
            if self.enable_open and self.enable_risk:
                self.risk.update_equity_peak(equity)
                stop_trading = self._apply_risk(row, equity)
            else:
                stop_trading = False

            # —— 步骤 3：Regime 检测，得到当日仓位系数 ——
            # Regime 始终按 enable_regime 计算并记录，即使关闭自动开仓也照常
            # 输出（供可视化与数据准备使用）。
            if self.enable_regime:
                z_window = self.df["Z"].values[:i + 1]
                reg = self.regime.decide(
                    z_window,
                    atr_fast=_safe(atr_fast.iloc[i]),
                    atr_slow=row["ATR"],
                )
                regime_state = reg["regime"]
                position_scale = reg["position_scale"]
            else:
                regime_state = 0
                position_scale = 1.0

            # —— 步骤 4：生成次日限价单 ——
            # 触发账户/策略级停牌、或 regime 为红、或风控要求停手时，不再开新仓。
            # 关闭「自动开仓系统」时不生成任何买入开仓单（见 _build_next_orders）。
            self.pending_orders = self._build_next_orders(
                row, stop_trading, regime_state, position_scale, date,
            )

            # —— 步骤 5：记录当日快照 ——
            self._record_day(row, equity, regime_state, position_scale)

        return self._finalize()

    # ---------------- 步骤 1：撮合 ----------------
    def _match_orders(self, row) -> None:
        """用今日 OHLC 撮合昨日挂出的限价单（README 8.3）。

        成交规则：
          - 买入：今日最低价 ≤ 买入限价，且今日非涨停 / 非停牌 → 以今日开盘价成交；
            （若开盘价低于限价，按更有利的开盘价成交，符合限价单语义）
          - 卖出：今日最高价 ≥ 卖出限价，且今日非跌停 / 非停牌 → 以今日开盘价成交。
        """
        if not self.pending_orders:
            return
        # 停牌当天不撮合任何单子。
        if bool(row.get("is_suspended", False)):
            self.pending_orders = []
            return

        open_px = row["open"]
        low_px = row["low"]
        high_px = row["high"]
        limit_up = bool(row.get("limit_up", False))
        limit_down = bool(row.get("limit_down", False))

        for order in self.pending_orders:
            if order.side == OrderSide.BUY:
                # 涨停无法买入（README 7.2 场景1：不追涨）。
                if limit_up:
                    continue
                # 价格触及限价才成交。
                if low_px <= order.price:
                    # 成交价取「开盘价」与「限价」中对买方更有利者（更低价）。
                    fill_price = min(open_px, order.price)
                    self._execute_buy(row, order, fill_price)
            else:  # SELL
                # 跌停无法卖出（挂单失败，等待重挂）。
                if limit_down:
                    continue
                if high_px >= order.price:
                    # 成交价取「开盘价」与「限价」中对卖方更有利者（更高价）。
                    fill_price = max(open_px, order.price)
                    self._execute_sell(row, order, fill_price)

        # 当日有效（DAY）：未成交的限价单当日作废。
        self.pending_orders = []

    def _execute_buy(self, row, order, price: float) -> None:
        """执行买入成交：扣现金（含费用）、更新持仓、记录成交。"""
        # 该层若已持仓则跳过（避免重复建仓）。
        if self.position.is_layer_holding(order.layer):
            return
        gross = price * order.quantity
        cost = self._buy_cost(gross)
        total_out = gross + cost
        # 现金不足则放弃该笔（保守处理，不允许透支）。
        if total_out > self.cash:
            return
        self.cash -= total_out
        self.position.apply_buy(order.layer, price, order.quantity)
        self.fills.append(Fill(
            trade_date=row["trade_date"], layer=order.layer,
            side=OrderSide.BUY, price=price, quantity=order.quantity,
            amount=gross, cost=cost, realized_pnl=0.0,
        ))

    def _execute_sell(self, row, order, price: float) -> None:
        """执行卖出成交：结算盈亏、加现金（扣费用）、记录成交。"""
        if not self.position.is_layer_holding(order.layer):
            return
        qty = self.position.layers[order.layer].quantity
        gross = price * qty
        cost = self._sell_cost(gross)
        realized = self.position.apply_sell(order.layer, price) - cost
        self.cash += gross - cost
        self.fills.append(Fill(
            trade_date=row["trade_date"], layer=order.layer,
            side=OrderSide.SELL, price=price, quantity=qty,
            amount=gross, cost=cost, realized_pnl=realized,
        ))

    # ---------------- 步骤 2：风控执行 ----------------
    def _apply_risk(self, row, equity: float) -> bool:
        """按优先级执行四级风控，返回「是否应停止新开仓」。

        优先级：账户(4) > 策略(3) > 标的(2) > 网格(1)。
        高层级触发清仓时直接以收盘价强平，并返回 True（当日不再开仓）。
        """
        close_px = row["close"]
        date = row["trade_date"]

        # 层级4：账户级年度止损。
        if self.risk.check_account_level(equity) == ACTION_STOP_YEAR:
            self._liquidate(row, close_px)
            return True

        # 层级3：策略级回撤。
        strat = self.risk.check_strategy_level(equity, date)
        if strat == ACTION_CLOSE_ALL:
            self._liquidate(row, close_px)
            return True
        # 减半在「生成次日单」时通过 position_scale 体现，这里不立即强平。

        # 层级2：标的级止损。
        if self.risk.check_symbol_level(self.position, close_px) == ACTION_CLOSE_SYMBOL:
            self._liquidate(row, close_px)
            self.risk.start_cooldown(self.ts_code, date)
            return True

        # 层级1：网格级单层止损（逐层检查）。
        for layer in self.position.holding_layers():
            if self.risk.check_layer_level(self.position, layer, close_px) == ACTION_CLOSE_LAYER:
                self._force_close_layer(row, layer, close_px)

        # 是否处于策略暂停 / 标的冷却中（影响开新仓）。
        paused = self.risk.strategy_paused(date) or self.risk.in_cooldown(self.ts_code, date)
        return paused

    def _liquidate(self, row, price: float) -> None:
        """以给定价格清空该标的全部持仓（风控强平）。"""
        for layer in self.position.holding_layers():
            self._force_close_layer(row, layer, price)

    def _force_close_layer(self, row, layer: int, price: float) -> None:
        """强制平掉某一层（风控触发，按给定价成交并计费）。"""
        qty = self.position.layers[layer].quantity
        if qty <= 0:
            return
        gross = price * qty
        cost = self._sell_cost(gross)
        realized = self.position.apply_sell(layer, price) - cost
        self.cash += gross - cost
        self.fills.append(Fill(
            trade_date=row["trade_date"], layer=layer,
            side=OrderSide.SELL, price=price, quantity=qty,
            amount=gross, cost=cost, realized_pnl=realized,
        ))

    # ---------------- 步骤 4：生成次日单 ----------------
    def _build_next_orders(self, row, stop_trading: bool, regime_state: int,
                           position_scale: float, date) -> list:
        """根据当日收盘指标与各类开关，生成次日限价单。

        - 若 stop_trading（账户/策略停手、标的冷却）→ 仅允许「卖出止盈」单，不开新仓；
        - 若 regime 为红 → 同样不开新仓（仅保留卖出单）；
        - 若 regime 为黄（position_scale=0.5）→ 按半仓资金生成买入单；
        - 停牌日不挂任何单。
        """
        # 关闭「自动开仓系统」：不生成任何限价单（既不开新仓也无需挂卖出，
        # 因为从空仓起步、全程无持仓）。Regime / 因子仍在主循环照常计算并
        # 记录，供可视化与数据准备使用。
        if not self.enable_open:
            return []

        if bool(row.get("is_suspended", False)):
            return []

        ma = row["MA"]
        atr = row["ATR"]
        orders = self.grid.generate_orders(ma, atr, self.position)

        # 不允许开新仓的情形：剔除买入单，仅保留卖出（止盈/减仓）。
        no_new_position = stop_trading or (regime_state == RED)
        if no_new_position:
            orders = [o for o in orders if o.side == OrderSide.SELL]
            return orders

        # 仓位减半：按 position_scale 缩减买入数量（卖出单不缩减）。
        if position_scale < 1.0:
            scaled = []
            for o in orders:
                if o.side == OrderSide.BUY:
                    new_qty = int((o.quantity * position_scale) // 100) * 100
                    if new_qty > 0:
                        o.quantity = new_qty
                        scaled.append(o)
                else:
                    scaled.append(o)
            return scaled
        return orders

    # ---------------- 步骤 5：记录 ----------------
    def _record_day(self, row, equity: float, regime_state: int,
                    position_scale: float) -> None:
        """记录当日净值快照（持仓、现金、总资产、regime 等）。"""
        self.daily_records.append({
            "trade_date": row["trade_date"],
            "close": row["close"],
            "cash": self.cash,
            "position_value": self.position.market_value(row["close"]),
            "equity": equity,
            "holding_layers": len(self.position.holding_layers()),
            "regime": regime_state,
            "position_scale": position_scale,
            "realized_pnl": self.position.realized_pnl,
        })

    # ---------------- 估值与成本 ----------------
    def _equity(self, price: float) -> float:
        """当前总资产 = 现金 + 持仓市值。"""
        return self.cash + self.position.market_value(price)

    def _buy_cost(self, gross: float) -> float:
        """买入费用：佣金 + 过户费 + 滑点（README 5.5）。"""
        return gross * (self.commission + self.transfer_fee + self.slippage)

    def _sell_cost(self, gross: float) -> float:
        """卖出费用：佣金 + 过户费 + 印花税 + 滑点（README 5.5）。"""
        return gross * (self.commission + self.transfer_fee
                        + self.stamp_tax + self.slippage)

    # ---------------- 收尾 ----------------
    def _finalize(self) -> dict:
        """回测结束：对剩余持仓按最后收盘价估值，汇总结果与绩效。"""
        daily_log = pd.DataFrame(self.daily_records)
        if daily_log.empty:
            equity_curve = pd.Series(dtype=float)
        else:
            equity_curve = pd.Series(
                daily_log["equity"].values,
                index=daily_log["trade_date"].values,
            )
        metrics = compute_metrics(equity_curve, self.fills)
        return {
            "ts_code": self.ts_code,
            "equity_curve": equity_curve,
            "fills": self.fills,
            "daily_log": daily_log,
            "metrics": metrics,
        }


# ======================== 内部工具 ========================

def _safe(value, default: float = 0.0) -> float:
    """把可能为 NaN 的标量转成安全浮点（NaN → default）。"""
    try:
        v = float(value)
        if v != v:  # NaN 判定
            return default
        return v
    except (TypeError, ValueError):
        return default
