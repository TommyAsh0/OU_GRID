# ====================================================================
# 绩效评估模块 (metrics.py)
# 对应 README 阶段 8.4「绩效评估」
#
# 输入回测产生的「每日净值序列」与「成交记录」，输出一整套绩效与风险
# 指标：年化收益、最大回撤、Calmar、Sharpe、胜率、盈亏比、CVaR 等。
#
# 所有公式严格按 README 8.4 定义，便于与文档逐条对照核验。
# ====================================================================

import numpy as np
import pandas as pd

from config.loader import CONFIG


def compute_metrics(equity_curve, fills) -> dict:
    """根据净值曲线与成交记录计算全套绩效指标（README 8.4）。

    Args:
        equity_curve: 每日总资产序列（pandas.Series，索引为日期）或可转为其的序列。
        fills: 成交记录列表（src.grid.order.Fill），用于胜率 / 盈亏比统计。

    Returns:
        dict: 包含以下键的绩效字典：
            annual_return, max_drawdown, calmar, sharpe, win_rate,
            profit_loss_ratio, total_return, volatility,
            longest_drawdown_days, cvar_95, n_trades, n_win, n_loss。
    """
    bt = CONFIG["backtest"]
    rf = bt["risk_free_rate"]
    periods = bt["trading_days_per_year"]

    equity = pd.Series(equity_curve, dtype=float).dropna()
    # 净值点不足，无法计算 → 返回全 NaN 的结果（上层据此判定无效）。
    if len(equity) < 2:
        return _empty_metrics()

    # ---- 收益类 ----
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    n_days = len(equity)
    # 年化收益率 = (终值/初值)^(252/天数) − 1。
    annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (periods / n_days) - 1.0

    # 日收益率序列（用于年化波动率与 Sharpe）。
    daily_ret = equity.pct_change().dropna()
    volatility = daily_ret.std(ddof=1) * np.sqrt(periods) if len(daily_ret) > 1 else 0.0

    # Sharpe = (年化收益 − 无风险) / 年化波动率。
    sharpe = (annual_return - rf) / volatility if volatility > 1e-12 else 0.0

    # ---- 回撤类 ----
    max_dd, longest_dd = _drawdown_stats(equity)
    # Calmar = 年化收益 / |最大回撤|。
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

    # ---- 风险类：95% CVaR（日收益最差 5% 的平均损失，取正数表示）。----
    cvar_95 = _cvar(daily_ret, level=0.95)

    # ---- 交易类：胜率与盈亏比（仅统计卖出了结的成交）----
    win_stats = _trade_stats(fills)

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "longest_drawdown_days": int(longest_dd),
        "cvar_95": float(cvar_95),
        "win_rate": float(win_stats["win_rate"]),
        "profit_loss_ratio": float(win_stats["profit_loss_ratio"]),
        "n_trades": int(win_stats["n_trades"]),
        "n_win": int(win_stats["n_win"]),
        "n_loss": int(win_stats["n_loss"]),
    }


# ======================== 内部工具 ========================

def _drawdown_stats(equity: pd.Series):
    """计算最大回撤幅度与最长回撤持续天数。

    Returns:
        (max_drawdown, longest_drawdown_days)
        max_drawdown 为负数（如 -0.18 表示最大回撤 18%）。
    """
    # 历史峰值（截至每个时点的最大净值）。
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = drawdown.min()

    # 最长回撤天数：连续处于「低于峰值」状态的最长长度。
    longest = 0
    current = 0
    for dd in drawdown.values:
        if dd < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(max_dd), int(longest)


def _cvar(daily_ret: pd.Series, level: float = 0.95) -> float:
    """计算条件风险价值 CVaR（最差 (1−level) 分位的平均损失，返回正数）。"""
    if len(daily_ret) < 2:
        return 0.0
    # 最差 5% 的分位点（VaR 阈值）。
    var_threshold = daily_ret.quantile(1.0 - level)
    tail = daily_ret[daily_ret <= var_threshold]
    if len(tail) == 0:
        return 0.0
    # 取负号让「损失」表示为正值，便于阅读。
    return float(-tail.mean())


def _trade_stats(fills) -> dict:
    """从成交记录统计胜率与盈亏比（只看卖出了结的实现盈亏）。"""
    from src.grid.order import OrderSide

    realized = [
        f.realized_pnl for f in fills
        if f.side == OrderSide.SELL
    ]
    n_trades = len(realized)
    if n_trades == 0:
        return {"win_rate": 0.0, "profit_loss_ratio": 0.0,
                "n_trades": 0, "n_win": 0, "n_loss": 0}

    wins = [p for p in realized if p > 0]
    losses = [p for p in realized if p < 0]
    n_win = len(wins)
    n_loss = len(losses)

    win_rate = n_win / n_trades
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    # 盈亏比 = 平均盈利 / 平均亏损；无亏损时记为 inf。
    pl_ratio = avg_win / avg_loss if avg_loss > 1e-12 else float("inf")

    return {
        "win_rate": win_rate,
        "profit_loss_ratio": pl_ratio,
        "n_trades": n_trades,
        "n_win": n_win,
        "n_loss": n_loss,
    }


def _empty_metrics() -> dict:
    """净值点不足时返回的占位结果（全部为 0 / NaN）。"""
    nan = float("nan")
    return {
        "total_return": nan, "annual_return": nan, "volatility": nan,
        "sharpe": nan, "max_drawdown": nan, "calmar": nan,
        "longest_drawdown_days": 0, "cvar_95": nan,
        "win_rate": 0.0, "profit_loss_ratio": 0.0,
        "n_trades": 0, "n_win": 0, "n_loss": 0,
    }
