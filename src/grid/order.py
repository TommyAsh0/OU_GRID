# ====================================================================
# 订单与成交模块 (order.py)
# 对应 README 阶段 5.4「交易信号逻辑」与 8.3「回测引擎关键要求」
#
# 定义网格引擎产出的「限价单」以及回测撮合后产生的「成交记录」两个
# 轻量数据结构。它们是网格引擎、回测引擎、风控之间传递信息的载体。
#
# 设计原则：纯数据容器，不含业务逻辑，便于各模块解耦与单元测试。
# ====================================================================

from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    """订单方向。"""

    BUY = "BUY"     # 买入（网格下方挂单）
    SELL = "SELL"   # 卖出（持仓层的获利了结）


@dataclass
class Order:
    """一张网格限价单（盘后生成，次日生效）。

    Attributes:
        layer: 网格层号（1=最浅，n=最深）。
        side: 买卖方向（OrderSide）。
        price: 限价（已按 2 位小数取整）。
        quantity: 委托数量（股；A 股通常为 100 的整数倍）。
        order_type: 订单类型，固定为 "LIMIT"（限价单）。
        valid: 有效期，固定为 "DAY"（当日有效）。
    """

    layer: int
    side: OrderSide
    price: float
    quantity: int
    order_type: str = "LIMIT"
    valid: str = "DAY"


@dataclass
class Fill:
    """一笔成交记录（回测撮合或实盘回报后生成）。

    Attributes:
        trade_date: 成交日期。
        layer: 对应网格层号。
        side: 买卖方向。
        price: 实际成交价（回测中通常取次日开盘价或限价）。
        quantity: 成交数量（股）。
        amount: 成交金额（price × quantity，不含费用）。
        cost: 该笔交易的总费用（佣金 + 印花税 + 过户费 + 滑点）。
        realized_pnl: 该笔成交实现的盈亏（仅卖出成交时非零）。
    """

    trade_date: object
    layer: int
    side: OrderSide
    price: float
    quantity: int
    amount: float
    cost: float
    realized_pnl: float = 0.0
