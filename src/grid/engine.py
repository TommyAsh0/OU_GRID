# ====================================================================
# 网格引擎模块 (engine.py)
# 对应 README 阶段 5「网格引擎设计」（5.1 结构 / 5.2 深度校验 /
# 5.3 仓位方案 / 5.4 交易信号逻辑）
#
# 职责：
#   1. 根据当日 MA、ATR 与层数 n、ATR 倍数 K，计算各层的买入价 / 卖出价；
#   2. 按「当前各层持仓状态」生成次日的限价单列表（空仓挂买、持仓挂卖）；
#   3. 提供网格深度校验（D = n×K ≤ 2.5×s）与每层下单数量计算。
#
# 本模块只负责「生成信号」，不负责撮合成交（撮合在回测引擎中），
# 严格遵循 README 8.3「信号盘后产生、次日执行、不使用未来数据」。
# ====================================================================

from config.loader import CONFIG
from src.grid.order import Order, OrderSide


class GridEngine:
    """单只标的的网格信号引擎。

    给定每日的 MA / ATR 与持仓状态，产出次日限价单。一个标的对应
    一个 GridEngine 实例（K 在验证集选定后固定下来）。
    """

    def __init__(self, ts_code: str, capital_per_symbol: float,
                 k: float = None, n_layers: int = None,
                 position_mode: str = None):
        """初始化网格引擎。

        Args:
            ts_code: 标的代码。
            capital_per_symbol: 分配给该标的的总资金（用于计算每层股数）。
            k: ATR 倍数 K（网格间距 = K×ATR）。默认读配置 grid.K。
            n_layers: 网格层数 n。默认读配置 grid.n_layers。
            position_mode: 仓位方案，"equal"（等权）或 "linear"（线性递增）。
                           默认读配置 grid.position_mode。
        """
        grid_cfg = CONFIG["grid"]
        self.ts_code = ts_code
        self.capital_per_symbol = capital_per_symbol
        self.k = k if k is not None else grid_cfg["K"]
        self.n_layers = n_layers if n_layers is not None else grid_cfg["n_layers"]
        self.position_mode = position_mode or grid_cfg["position_mode"]
        self.max_depth_sigma = grid_cfg["max_depth_sigma"]

        # 预计算各层资金权重（与价格无关，构造时确定一次即可）。
        self.layer_weights = self._compute_layer_weights()

    # ---------------- 仓位方案 ----------------
    def _compute_layer_weights(self) -> dict:
        """按仓位方案计算各层资金权重（归一化，和为 1）。

        README 5.3：
          方案 A「equal」：每层等权 = 1/n。
          方案 B「linear」：权重比 1 : 1.5 : 2 : ... 线性递增后归一化。
        """
        n = self.n_layers
        if self.position_mode == "linear":
            # 线性递增：第 i 层原始权重 = 1 + 0.5×(i-1)，再归一化。
            raw = [1.0 + 0.5 * (i - 1) for i in range(1, n + 1)]
        else:
            # 等权（默认）。
            raw = [1.0 for _ in range(1, n + 1)]
        total = sum(raw)
        # 层号 → 归一化权重。
        return {i + 1: raw[i] / total for i in range(n)}

    def compute_quantity(self, layer: int, buy_price: float) -> int:
        """计算某层的下单股数（A 股按 100 股「一手」向下取整）。

        Args:
            layer: 网格层号。
            buy_price: 该层买入价（用于把层资金换算成股数）。

        Returns:
            int: 下单股数（100 的整数倍；资金不足一手时为 0）。
        """
        layer_capital = self.capital_per_symbol * self.layer_weights[layer]
        if buy_price <= 0:
            return 0
        raw_shares = layer_capital / buy_price
        # 向下取整到 100 股的整数倍（A 股最小交易单位）。
        lots = int(raw_shares // 100)
        return lots * 100

    # ---------------- 网格价格 ----------------
    def grid_prices(self, ma: float, atr: float) -> dict:
        """计算各层的买入价与卖出价（README 5.1）。

        买入价_i = MA − i×K×ATR
        卖出价_i = 买入价_i + K×ATR（回到上一层价格时止盈）

        Args:
            ma: 当日 MA_60。
            atr: 当日 ATR_20。

        Returns:
            dict: {层号: {"buy": 买入价, "sell": 卖出价}}（价格保留 2 位小数）。
        """
        spacing = self.k * atr
        prices = {}
        for i in range(1, self.n_layers + 1):
            buy = ma - i * spacing
            sell = buy + spacing
            prices[i] = {"buy": round(buy, 2), "sell": round(sell, 2)}
        return prices

    # ---------------- 信号生成 ----------------
    def generate_orders(self, ma: float, atr: float, position) -> list:
        """根据当日指标与持仓状态生成次日限价单（README 5.4）。

        规则：
          - 某层「空仓」→ 挂买入限价单（compute_quantity 决定数量）；
          - 某层「持仓」→ 挂卖出限价单（数量为该层现有持仓）。
          - 数量为 0（资金不足一手）或价格非正的层跳过。

        Args:
            ma: 当日 MA_60。
            atr: 当日 ATR_20。
            position: 该标的的 GridPosition（提供逐层持仓状态）。

        Returns:
            list[Order]: 次日待挂的限价单列表。
        """
        # 指标非法（NaN / 非正）时不产生任何信号。
        if not (atr > 0) or not (ma > 0):
            return []

        prices = self.grid_prices(ma, atr)
        orders = []
        for i in range(1, self.n_layers + 1):
            buy_price = prices[i]["buy"]
            sell_price = prices[i]["sell"]

            if position.is_layer_holding(i):
                # 已持仓 → 挂卖出单（止盈）。
                qty = position.layers[i].quantity
                if qty > 0 and sell_price > 0:
                    orders.append(Order(
                        layer=i, side=OrderSide.SELL,
                        price=sell_price, quantity=qty,
                    ))
            else:
                # 空仓 → 挂买入单（逢跌建仓）。
                # 买入价为负（极端深层）则跳过该层。
                if buy_price <= 0:
                    continue
                qty = self.compute_quantity(i, buy_price)
                if qty > 0:
                    orders.append(Order(
                        layer=i, side=OrderSide.BUY,
                        price=buy_price, quantity=qty,
                    ))
        return orders

    # ---------------- 网格深度校验 ----------------
    def validate_depth(self, steady_std: float) -> bool:
        """校验网格总深度是否在稳态波动范围内（README 5.2）。

        D = n × K（Z 空间总深度），要求 D ≤ max_depth_sigma × s。
        s 为 OU 稳态标准差。s 非正 / 非法时视为不通过。

        Args:
            steady_std: OU 稳态标准差 s。

        Returns:
            bool: True 表示深度合理，可用当前 (n, K) 运行该标的。
        """
        if not (steady_std > 0):
            return False
        depth = self.n_layers * self.k
        return depth <= self.max_depth_sigma * steady_std
