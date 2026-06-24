# ====================================================================
# 回测可视化模块 (visualizer.py)
# 对应 README 阶段 8「历史回测」的结果可视化
#
# 职责：把一次回测的结果画成一张「可交互的 HTML 图表」。
#   - 主图：K 线（蜡烛图）+ 每个买入 / 卖出成交点；
#   - 副图：当天「盘前可见」的因子 / 指标值（每日滚动计算）。
#
# ★核心原则——「盘前可见、无前视」★
#   真实交易中，t 日开盘前你只能看到「t-1 日收盘」为止的数据，
#   因此 t 日盘前用来决策的因子（Z、滚动 κ、半衰期、Regime 等）
#   必须只用「截至 t-1 日」的历史滚动算出。
#   本模块据此把每日因子整体「向后移一位」（shift(1)）对齐到
#   「可观测日」：图上 t 日副图显示的，就是 t 日盘前你真实能看到的值。
#
# 为什么单独逐日滚动重算，而不直接用整段指标？
#   直接对整段历史一次性算指标，会在某些统计量（如滚动 Hurst / κ）上
#   隐含「用到未来」的风险，也不符合「每天盘前重算一次」的真实流程。
#   这里用一个朴素的 for 循环逐日推进、每日只喂入当日及之前的数据，
#   逻辑直白、易于核对（不追求速度，只追求正确与可读）。
#
# 绘图库：plotly（生成自包含、可缩放、可悬浮查看数值的交互式 HTML）。
# ====================================================================

import numpy as np
import pandas as pd

from config.loader import CONFIG
from src.grid.order import OrderSide
from src.regime.detector import (
    GREEN,
    YELLOW,
    RED,
    RegimeDetector,
)
from src.screening.ou_estimator import estimate_ou

# plotly 为可视化专用依赖；缺失时给出友好提示而非在导入期直接崩溃。
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover - 仅在未安装 plotly 时触发
    go = None
    make_subplots = None

# Regime 状态 → 中文名 / 颜色（副图背景着色与图例用）。
REGIME_NAME = {GREEN: "绿·正常", YELLOW: "黄·减半", RED: "红·暂停"}
REGIME_COLOR = {
    GREEN: "rgba(0, 176, 80, 0.10)",    # 浅绿
    YELLOW: "rgba(255, 192, 0, 0.12)",  # 浅黄
    RED: "rgba(255, 0, 0, 0.10)",       # 浅红
}


# ====================================================================
# 一、盘前滚动因子计算（每日只用 t-1 及之前数据，模拟真实盘前）
# ====================================================================

def compute_premarket_factors(df: pd.DataFrame) -> pd.DataFrame:
    """逐日滚动计算「盘前可见」的因子 / 指标。

    模拟真实流程：站在每个交易日 t 的「盘前」，只用「截至 t-1 日收盘」
    的历史数据，重算当天用于决策的因子。最后把结果对齐到「可观测日」
    （即 t 日那一行，存放的是 t 日盘前、由 t-1 及之前算出的值）。

    计算的因子（均为盘前可见）：
      - z_premarket   : 最近一个收盘日的标准化偏离度 Z（网格核心驱动量）。
      - kappa_roll    : 滚动 OU 回复速度 κ（窗口取配置 ou_estimation.window）。
      - half_life_roll: 由 κ 换算的半衰期 HL = ln2 / κ。
      - regime        : Regime 状态（0 绿 / 1 黄 / 2 红）。
      - position_scale: Regime 给出的目标仓位系数（1.0 / 0.5 / 0.0）。

    Args:
        df: 已含 MA/ATR/Z/TR 指标的单标的行情（来自 DataProcessor.process），
            需按交易日升序排列。

    Returns:
        pd.DataFrame: 与输入等长、索引对齐的因子表，列见上。
            第 t 行 = 第 t 日「盘前」可见的因子值（由 t-1 及之前算出）。
            暖机期（数据不足）对应行为 NaN / 中性值。
    """
    df = df.reset_index(drop=True)
    n = len(df)

    ou_window = CONFIG["ou_estimation"]["window"]  # 滚动 OU 估计窗口
    atr_fast_len = 5                               # 短期 ATR（波动率爆发用）

    # 短期 ATR_5（与回测引擎一致，用于 Regime 的波动率检测）。
    atr_fast = df["TR"].rolling(atr_fast_len).mean()

    # 用一个独立的 Regime 检测器逐日推进（其内部维护恢复状态机）。
    detector = RegimeDetector()

    # ---- 逐日滚动：第 i 日「收盘后」算出的因子，代表「第 i+1 日盘前可见」 ----
    # 我们先按「收盘日」算出每日因子，最后整体 shift(1) 对齐到盘前可观测日。
    z_close = []          # 第 i 日收盘的 Z
    kappa_close = []      # 截至第 i 日的滚动 κ
    hl_close = []         # 截至第 i 日的滚动半衰期
    regime_close = []     # 第 i 日收盘后的 Regime 状态
    scale_close = []      # 第 i 日收盘后的目标仓位系数

    for i in range(n):
        # 当日及之前的 Z 序列（绝不使用 i 之后的数据）。
        z_hist = df["Z"].values[: i + 1]
        z_close.append(z_hist[-1])

        # 滚动 OU：取最近 ou_window 个 Z 估计 κ；不足或无效则记 NaN。
        seg = z_hist[-ou_window:] if len(z_hist) >= ou_window else z_hist
        params = estimate_ou(seg)
        if params.valid:
            kappa_close.append(params.kappa)
            hl_close.append(params.half_life)
        else:
            kappa_close.append(np.nan)
            hl_close.append(np.nan)

        # Regime 决策：同样只喂入当日及之前的数据。
        reg = detector.decide(
            z_hist,
            atr_fast=_safe(atr_fast.iloc[i]),
            atr_slow=_safe(df["ATR"].iloc[i]),
        )
        regime_close.append(reg["regime"])
        scale_close.append(reg["position_scale"])

    factors_close = pd.DataFrame({
        "z_premarket": z_close,
        "kappa_roll": kappa_close,
        "half_life_roll": hl_close,
        "regime": regime_close,
        "position_scale": scale_close,
    })

    # ★关键的「盘前对齐」：第 i 日收盘算出的值，最早在第 i+1 日盘前可见。
    #   因此整体下移一行（shift 1）：第 t 行 = 第 t 日盘前可见（= t-1 收盘算出）。
    factors = factors_close.shift(1)
    # Regime 首行无历史，用「绿·正常」中性值占位（避免 NaN 影响着色）。
    factors["regime"] = factors["regime"].fillna(GREEN).astype(int)
    factors["position_scale"] = factors["position_scale"].fillna(1.0)

    # 附上交易日，便于与行情对齐绘图。
    factors.insert(0, "trade_date", df["trade_date"].values)
    return factors


# ====================================================================
# 二、构建交互式图表（主图 K 线 + 买卖点；副图盘前因子）
# ====================================================================

def build_figure(df: pd.DataFrame, fills: list, factors: pd.DataFrame,
                 ts_code: str):
    """用 plotly 构建「主图 K 线 + 买卖点 / 副图盘前因子」的交互图。

    布局：上下两块（共享 x 轴 = 交易日）。
      主图（row 1）：
        - 蜡烛图（开高低收）；
        - MA 均线（网格中轴）；
        - 买入成交点（▲ 红）、卖出成交点（▼ 绿），悬浮显示层号 / 价格。
      副图（row 2）：
        - 盘前可见的 Z（主因子，左轴）；
        - 盘前 Regime 状态用背景色带标注（绿 / 黄 / 红）。
      （κ / 半衰期等因子放入悬浮信息，避免副图过于拥挤；如需可单独成图。）

    Args:
        df: 单标的行情（含 open/high/low/close/MA/trade_date），升序。
        fills: 回测成交列表（Fill 对象，含 trade_date/side/price/layer）。
        factors: compute_premarket_factors 的输出（盘前对齐后的因子）。
        ts_code: 标的代码（用于标题）。

    Returns:
        plotly.graph_objects.Figure: 可保存为 HTML 的图表对象。

    Raises:
        RuntimeError: 未安装 plotly 时抛出，提示安装方式。
    """
    if go is None or make_subplots is None:
        raise RuntimeError(
            "未检测到 plotly，无法生成 HTML 图表。请先安装：pip install plotly"
        )

    dates = df["trade_date"]

    # 上下两行子图，主图占 70% 高度，副图 30%，共享 x 轴。
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
        subplot_titles=(f"{ts_code} K线与买卖点", "盘前可见因子（t 日显示 t-1 数据）"),
    )

    # ---------------- 主图：K 线蜡烛图 ----------------
    fig.add_trace(
        go.Candlestick(
            x=dates, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"],
            name="K线",
            increasing_line_color="#d62728",   # A 股习惯：涨红
            decreasing_line_color="#2ca02c",   # 跌绿
        ),
        row=1, col=1,
    )

    # 主图：MA 均线（网格中轴），帮助理解买卖点相对中轴的位置。
    if "MA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates, y=df["MA"], name="MA(网格中轴)",
                line=dict(color="#1f77b4", width=1.2),
                opacity=0.9,
            ),
            row=1, col=1,
        )

    # ---------------- 主图：买入 / 卖出成交点 ----------------
    buys = [f for f in fills if f.side == OrderSide.BUY]
    sells = [f for f in fills if f.side == OrderSide.SELL]

    if buys:
        fig.add_trace(
            go.Scatter(
                x=[f.trade_date for f in buys],
                y=[f.price for f in buys],
                mode="markers", name="买入",
                marker=dict(symbol="triangle-up", size=11,
                            color="#d62728",
                            line=dict(width=1, color="#7f0000")),
                # 悬浮显示「第几层 / 成交价 / 数量」。
                customdata=[[f.layer, f.quantity] for f in buys],
                hovertemplate=("买入<br>日期=%{x|%Y-%m-%d}<br>"
                               "价格=%{y:.2f}<br>层=%{customdata[0]}<br>"
                               "数量=%{customdata[1]}<extra></extra>"),
            ),
            row=1, col=1,
        )

    if sells:
        fig.add_trace(
            go.Scatter(
                x=[f.trade_date for f in sells],
                y=[f.price for f in sells],
                mode="markers", name="卖出",
                marker=dict(symbol="triangle-down", size=11,
                            color="#2ca02c",
                            line=dict(width=1, color="#0b3d0b")),
                customdata=[[f.layer, f.realized_pnl] for f in sells],
                hovertemplate=("卖出<br>日期=%{x|%Y-%m-%d}<br>"
                               "价格=%{y:.2f}<br>层=%{customdata[0]}<br>"
                               "实现盈亏=%{customdata[1]:.0f}<extra></extra>"),
            ),
            row=1, col=1,
        )

    # ---------------- 副图：盘前可见因子（Z 主因子） ----------------
    # 悬浮里附带 κ / 半衰期 / Regime 名称，便于逐日核对盘前全貌。
    regime_names = factors["regime"].map(REGIME_NAME).fillna("-")
    custom = np.column_stack([
        factors["kappa_roll"].values,
        factors["half_life_roll"].values,
        regime_names.values,
    ])
    fig.add_trace(
        go.Scatter(
            x=factors["trade_date"], y=factors["z_premarket"],
            name="Z(盘前偏离度)",
            line=dict(color="#9467bd", width=1.3),
            customdata=custom,
            hovertemplate=("日期=%{x|%Y-%m-%d}<br>Z=%{y:.2f}<br>"
                           "κ=%{customdata[0]:.3f}<br>"
                           "半衰期=%{customdata[1]:.1f}<br>"
                           "Regime=%{customdata[2]}<extra></extra>"),
        ),
        row=2, col=1,
    )
    # 副图：Z=0 参考线（价格回到 MA 中轴）。
    fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"),
                  row=2, col=1)

    # ---------------- 副图背景：按盘前 Regime 状态着色 ----------------
    _shade_regime(fig, factors)

    # ---------------- 全局布局 ----------------
    fig.update_layout(
        title=dict(text=f"OU 网格回测可视化 · {ts_code}", x=0.5),
        height=820, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=40),
        # 关闭主图自带的区间滑块，避免与副图争抢空间。
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="Z 偏离度", row=2, col=1)
    fig.update_xaxes(title_text="交易日", row=2, col=1)
    return fig


def save_html(fig, out_path: str) -> str:
    """把图表保存为自包含的交互式 HTML 文件。

    Args:
        fig: build_figure 返回的 plotly 图表。
        out_path: 输出 HTML 路径。

    Returns:
        str: 实际写入的文件路径。
    """
    # include_plotlyjs="cdn"：HTML 体积小，打开时从 CDN 加载 plotly.js；
    # 若需完全离线可改为 True（把 plotly.js 内联进 HTML）。
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)
    return out_path


# ======================== 内部工具 ========================

def _shade_regime(fig, factors: pd.DataFrame) -> None:
    """在副图上按「盘前 Regime 状态」绘制背景色带。

    把连续相同状态的日期段合并为一个矩形（vrect），减少图元数量、
    让黄 / 红的暂停区间一目了然（绿色为正常态，默认不额外着色）。
    """
    dates = pd.to_datetime(factors["trade_date"]).values
    states = factors["regime"].values
    n = len(states)
    if n == 0:
        return

    start = 0
    for i in range(1, n + 1):
        # 到达结尾，或状态发生切换 → 收束当前连续段 [start, i-1]。
        if i == n or states[i] != states[start]:
            state = int(states[start])
            # 只给黄 / 红着色（绿色保持留白，突出异常区间）。
            if state in (YELLOW, RED):
                x0 = dates[start]
                x1 = dates[i - 1]
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor=REGIME_COLOR[state], line_width=0,
                    layer="below", row=2, col=1,
                )
            start = i


def _safe(value, default: float = 0.0) -> float:
    """把可能为 NaN 的标量转成安全浮点（NaN → default）。"""
    try:
        v = float(value)
        if v != v:  # NaN 判定
            return default
        return v
    except (TypeError, ValueError):
        return default
