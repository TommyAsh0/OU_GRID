# ====================================================================
# 风控系统模块 (controller.py)
# 对应 README 阶段 7「风控系统」（7.1 风控层级 / 7.3 最大持仓限制）
# 以及阶段 0.3「绝对止损线——不可协商」。
#
# 实现四级风控，执行优先级：层级4 > 层级3 > 层级2 > 层级1。
#   层级1 网格级：单层浮亏 > 15% → 平该层。
#   层级2 标的级：单标的总浮亏 > 投入资金 15% → 清该标的，冷却 30 日。
#   层级3 策略级：总回撤 > 15% → 全仓减半；> 20% → 全清，暂停 30 日。
#   层级4 账户级：年度亏损 > 25% → 当年停止策略。
#
# 本模块只「判定」该触发何种风控动作，具体的平仓成交由回测引擎执行，
# 这样风控逻辑保持纯粹、可单元测试，与撮合解耦。
# ====================================================================

from config.loader import CONFIG

# 风控动作类型（供回测引擎据此执行对应操作）。
ACTION_NONE = "none"                  # 无需动作
ACTION_CLOSE_LAYER = "close_layer"    # 层级1：平掉指定层
ACTION_CLOSE_SYMBOL = "close_symbol"  # 层级2：清掉指定标的并冷却
ACTION_REDUCE_HALF = "reduce_half"    # 层级3a：全部仓位减半
ACTION_CLOSE_ALL = "close_all"        # 层级3b：全部清仓并暂停
ACTION_STOP_YEAR = "stop_year"        # 层级4：当年停止策略


class RiskController:
    """四级风控判定器。

    维护策略级 / 账户级的状态（峰值净值、暂停截止日、冷却中的标的、
    当年是否已停），并按优先级给出当日应执行的风控动作。
    """

    def __init__(self, initial_capital: float):
        """初始化风控状态。

        Args:
            initial_capital: 初始资金（用作年度盈亏与净值峰值的基准）。
        """
        risk = CONFIG["risk"]
        self.single_stop = risk["single_stop_loss"]        # 标的级止损线（-0.15）
        self.strategy_reduce = risk["strategy_reduce"]     # 策略减半线（-0.15）
        self.strategy_stop = risk["strategy_stop"]         # 策略清仓线（-0.20）
        self.annual_stop = risk["annual_stop"]             # 年度止损线（-0.25）
        self.cooldown_days = risk["cooldown_days"]         # 清仓冷却天数
        # 网格级单层止损线（README 7.1：层仓位 × 15%）。
        self.layer_stop = self.single_stop

        self.initial_capital = initial_capital
        # 策略净值峰值（计算回撤用），初值为初始资金。
        self.equity_peak = initial_capital
        # 冷却中的标的：{ts_code: 冷却到期日}。
        self.cooldown_until = {}
        # 当前年度的起始净值与年份（年度亏损判定用）。
        self.year_start_equity = initial_capital
        self.current_year = None
        # 当年是否已因年度止损而停。
        self.year_stopped = False
        # 策略级暂停截止日（None 表示未暂停）。
        self.strategy_paused_until = None

    # ---------------- 每日状态维护 ----------------
    def update_equity_peak(self, equity: float) -> None:
        """更新净值峰值（用于策略级回撤计算）。"""
        if equity > self.equity_peak:
            self.equity_peak = equity

    def roll_year(self, trade_date) -> None:
        """跨年时重置年度基准（年度亏损按自然年统计）。

        Args:
            trade_date: 当前交易日（需有 .year 属性，如 pandas.Timestamp）。
        """
        year = trade_date.year
        if self.current_year is None:
            self.current_year = year
            self.year_start_equity = self.equity_peak
        elif year != self.current_year:
            # 进入新的一年：重置年度基准与「当年已停」标记。
            self.current_year = year
            self.year_start_equity = self.equity_peak
            self.year_stopped = False

    # ---------------- 冷却 / 暂停查询 ----------------
    def in_cooldown(self, ts_code: str, trade_date) -> bool:
        """某标的是否处于清仓后的冷却期（冷却期内不开新仓）。"""
        until = self.cooldown_until.get(ts_code)
        return until is not None and trade_date < until

    def strategy_paused(self, trade_date) -> bool:
        """策略是否处于清仓后的暂停期。"""
        return (self.strategy_paused_until is not None
                and trade_date < self.strategy_paused_until)

    def start_cooldown(self, ts_code: str, trade_date) -> None:
        """对某标的开启 cooldown_days 天的冷却。"""
        self.cooldown_until[ts_code] = trade_date + _days(self.cooldown_days)

    # ---------------- 层级4：账户级 ----------------
    def check_account_level(self, equity: float) -> str:
        """层级4：年度亏损是否超限（README 7.1，最高优先级）。

        年度收益率 = equity / year_start_equity − 1，低于 annual_stop 则当年停止。
        """
        if self.year_stopped:
            return ACTION_STOP_YEAR
        if self.year_start_equity > 0:
            year_return = equity / self.year_start_equity - 1.0
            if year_return <= self.annual_stop:
                self.year_stopped = True
                return ACTION_STOP_YEAR
        return ACTION_NONE

    # ---------------- 层级3：策略级 ----------------
    def check_strategy_level(self, equity: float, trade_date) -> str:
        """层级3：策略总回撤判定（README 7.1）。

        回撤 = equity / peak − 1：
          ≤ strategy_stop(-0.20)   → 全部清仓并暂停 cooldown_days 天；
          ≤ strategy_reduce(-0.15) → 全部仓位减半。
        """
        if self.equity_peak <= 0:
            return ACTION_NONE
        drawdown = equity / self.equity_peak - 1.0
        if drawdown <= self.strategy_stop:
            # 触发清仓线：设置策略暂停截止日。
            self.strategy_paused_until = trade_date + _days(self.cooldown_days)
            return ACTION_CLOSE_ALL
        if drawdown <= self.strategy_reduce:
            return ACTION_REDUCE_HALF
        return ACTION_NONE

    # ---------------- 层级2：标的级 ----------------
    def check_symbol_level(self, position, price: float) -> str:
        """层级2：单标的浮亏是否超限（README 7.1）。

        浮动收益率 ≤ single_stop(-0.15) → 该标的清仓并进入冷却。
        """
        if position.is_empty():
            return ACTION_NONE
        if position.floating_return(price) <= self.single_stop:
            return ACTION_CLOSE_SYMBOL
        return ACTION_NONE

    # ---------------- 层级1：网格级 ----------------
    def check_layer_level(self, position, layer: int, price: float) -> str:
        """层级1：单层浮亏是否超限（README 7.1）。

        (现价 − 建仓价)/建仓价 ≤ layer_stop(-0.15) → 平掉该层。
        """
        if not position.is_layer_holding(layer):
            return ACTION_NONE
        st = position.layers[layer]
        if st.entry_price <= 0:
            return ACTION_NONE
        layer_return = (price - st.entry_price) / st.entry_price
        if layer_return <= self.layer_stop:
            return ACTION_CLOSE_LAYER
        return ACTION_NONE


# ======================== 内部工具 ========================

def _days(n: int):
    """返回 n 天的时间增量（pandas 时间戳可直接相加）。"""
    import pandas as pd
    return pd.Timedelta(days=n)
