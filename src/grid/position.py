# ====================================================================
# 网格持仓模块 (position.py)
# 对应 README 阶段 5.3「仓位方案」、7.1「风控层级」、7.3「最大持仓限制」
#
# 职责：跟踪单只标的「逐层网格」的持仓状态。
#   - 每一层独立记录：是否持仓、持仓数量、买入均价；
#   - 提供层级买入/卖出的状态更新；
#   - 计算单标的的持仓市值、浮动盈亏、累计已实现盈亏，
#     供风控（标的级 / 策略级止损）与回测净值核算使用。
#
# 设计上「一只标的对应一个 GridPosition」，多标的由上层用字典管理。
# ====================================================================


class LayerState:
    """单个网格层的持仓状态。

    Attributes:
        is_holding: 该层当前是否持仓。
        quantity: 持仓数量（股）。
        entry_price: 建仓均价（买入成交价）。
    """

    def __init__(self):
        self.is_holding = False
        self.quantity = 0
        self.entry_price = 0.0

    def buy(self, price: float, quantity: int) -> None:
        """记录该层买入成交（网格每层至多持有一笔，故直接覆盖）。"""
        self.is_holding = True
        self.quantity = quantity
        self.entry_price = price

    def sell(self) -> None:
        """记录该层卖出成交，状态清空回到「空仓」。"""
        self.is_holding = False
        self.quantity = 0
        self.entry_price = 0.0


class GridPosition:
    """单只标的的网格持仓集合（管理 n 个 LayerState）。

    统一提供「按层买卖」「按当前价估值」「统计盈亏」等接口，
    上层（网格引擎 / 回测引擎 / 风控）无需关心逐层细节。
    """

    def __init__(self, ts_code: str, n_layers: int):
        """初始化某标的的逐层持仓。

        Args:
            ts_code: 标的代码。
            n_layers: 网格层数 n。
        """
        self.ts_code = ts_code
        self.n_layers = n_layers
        # 层号从 1 到 n，每层一个 LayerState。
        self.layers = {i: LayerState() for i in range(1, n_layers + 1)}
        # 该标的累计已实现盈亏（卖出了结时累加）。
        self.realized_pnl = 0.0
        # 该标的累计投入成本（用于标的级止损的分母）。
        self.invested_capital = 0.0

    # ---------------- 状态查询 ----------------
    def is_layer_holding(self, layer: int) -> bool:
        """某层是否持仓。"""
        return self.layers[layer].is_holding

    def holding_layers(self) -> list:
        """返回当前持仓的层号列表。"""
        return [i for i, st in self.layers.items() if st.is_holding]

    def total_quantity(self) -> int:
        """该标的当前总持仓股数（各层之和）。"""
        return sum(st.quantity for st in self.layers.values())

    def is_empty(self) -> bool:
        """是否完全空仓（所有层都未持仓）。"""
        return self.total_quantity() == 0

    # ---------------- 估值与盈亏 ----------------
    def market_value(self, price: float) -> float:
        """按给定价格计算当前持仓市值。"""
        return self.total_quantity() * price

    def cost_basis(self) -> float:
        """当前持仓的总成本（各层 数量×建仓价 之和）。"""
        return sum(st.quantity * st.entry_price for st in self.layers.values())

    def unrealized_pnl(self, price: float) -> float:
        """当前持仓的浮动盈亏（市值 − 成本）。"""
        return self.market_value(price) - self.cost_basis()

    def floating_return(self, price: float) -> float:
        """标的级浮动收益率（浮亏 / 投入资金），用于标的级止损。

        分母用「累计投入资金」而非当前成本，更贴近 README 7.1
        「单标的总浮亏 > 标的投入资金的 15%」的定义。
        投入资金为 0（尚未建仓）时返回 0。
        """
        if self.invested_capital <= 0:
            return 0.0
        return self.unrealized_pnl(price) / self.invested_capital

    # ---------------- 成交更新 ----------------
    def apply_buy(self, layer: int, price: float, quantity: int) -> None:
        """登记某层买入成交，并累加投入资金。"""
        self.layers[layer].buy(price, quantity)
        self.invested_capital += price * quantity

    def apply_sell(self, layer: int, price: float) -> float:
        """登记某层卖出成交，结算并返回该层已实现盈亏。

        Args:
            layer: 卖出的层号。
            price: 卖出成交价。

        Returns:
            float: 该层实现盈亏 = (卖价 − 建仓价) × 数量。
        """
        st = self.layers[layer]
        pnl = (price - st.entry_price) * st.quantity
        self.realized_pnl += pnl
        st.sell()
        return pnl

    def clear_all(self, price: float) -> float:
        """以给定价格清空所有持仓层（风控强平 / regime 紧急止损用）。

        Args:
            price: 清仓成交价。

        Returns:
            float: 本次清仓合计实现的盈亏。
        """
        total = 0.0
        for layer in self.holding_layers():
            total += self.apply_sell(layer, price)
        return total
