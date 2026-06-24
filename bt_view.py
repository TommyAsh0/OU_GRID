# ====================================================================
# 回测可视化入口脚本 (bt_view.py)
# 对应 README 阶段 8「历史回测」的「结果可视化」补充
#
# 用途：
#   把「单只标的」的一次回测结果画成一张可交互的 HTML 图表，方便
#   直观核对策略行为：主图看 K 线与每个买卖点，副图看「每天盘前
#   可见」的因子 / 指标值（每日滚动重算，t 日显示 t-1 日数据）。
#
# 数据来源：本地数据（DataFetcher 优先读 data/raw/ 缓存；缓存缺失
#   且无法联网时，自动用合成数据兜底，保证离线也能出图）。
#
# 处理流程（与 run_backtest.py 的单标的链路保持一致）：
#   1. 取数    : DataFetcher.get_daily 读本地行情。
#   2. 指标    : DataProcessor.process 计算 MA/ATR/Z 并清洗。
#   3. 回测    : BacktestEngine.run 产出成交列表 fills（买卖点来源）。
#   4. 盘前因子: visualizer.compute_premarket_factors 逐日滚动重算
#                （shift(1) 对齐到「盘前可观测日」，t 日显示 t-1 数据）。
#   5. 绘图    : visualizer.build_figure 画「K 线+买卖点 / 盘前因子」。
#   6. 保存    : visualizer.save_html 写出自包含 HTML 到 data/results/。
#
# 运行方式：
#   python bt_view.py                 # 用默认标的（股票池第一只）
#   python bt_view.py 600519.SH       # 指定标的代码
#   python bt_view.py 600519.SH 1.0   # 再指定网格 K 值（ATR 倍数）
#
# 说明：本脚本只做「单标的可视化」，不参与筛选 / 选 K / 组合汇总；
#       其结果与 run_backtest.py 的回测引擎完全同源，仅多了画图步骤。
# ====================================================================

import os
import sys
import warnings

from config.loader import CONFIG, resolve_path
from config.stock_pool_config import get_stock_pool
from src.backtest.engine import BacktestEngine
from src.backtest.visualizer import (
    build_figure,
    compute_premarket_factors,
    save_html,
)
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor

warnings.filterwarnings("ignore")


def make_view(ts_code: str, k: float = None, verbose: bool = True) -> str:
    """为单只标的生成回测可视化 HTML，并返回输出文件路径。

    Args:
        ts_code: 标的代码，如 "600519.SH"。
        k: 网格 ATR 倍数 K；None 则用配置默认值 grid.K。
        verbose: 是否打印过程信息。

    Returns:
        str: 生成的 HTML 文件路径；数据不足无法出图时返回空字符串。
    """
    data_cfg = CONFIG["data"]
    start = data_cfg["start_date"]
    end = data_cfg["end_date"]
    capital = CONFIG["backtest"]["initial_capital"]

    # —— 1) 取数（本地缓存优先，缺失则兜底）——
    if verbose:
        print(f"[1/6] 读取本地行情 {ts_code} ({start} ~ {end}) ...")
    fetcher = DataFetcher()
    raw = fetcher.get_daily(ts_code, start, end)

    # —— 2) 清洗 + 指标计算 ——
    if verbose:
        print("[2/6] 清洗数据并计算 MA/ATR/Z 指标 ...")
    processor = DataProcessor()
    df = processor.process(raw)
    if df.empty or len(df) < 30:
        print(f"       数据不足（有效 {len(df)} 行），无法出图。")
        return ""

    # —— 3) 回测，得到成交列表（买卖点来源）——
    if verbose:
        print("[3/6] 运行回测引擎，产出买卖成交点 ...")
    bt = BacktestEngine(ts_code, df, capital, k=k)
    result = bt.run()
    fills = result["fills"]
    if verbose:
        n_buy = sum(1 for f in fills if f.side.value == "BUY")
        n_sell = len(fills) - n_buy
        print(f"       成交：买入 {n_buy} 笔 / 卖出 {n_sell} 笔 "
              f"（共 {len(fills)} 笔）")

    # —— 4) 逐日滚动计算「盘前可见」因子（t 日显示 t-1 数据）——
    if verbose:
        print("[4/6] 逐日滚动计算盘前因子（无前视，模拟真实盘前）...")
    factors = compute_premarket_factors(df)

    # —— 5) 构建交互图（主图 K 线+买卖点 / 副图盘前因子）——
    if verbose:
        print("[5/6] 构建 K 线与因子交互图 ...")
    fig = build_figure(df, fills, factors, ts_code)

    # —— 6) 保存 HTML ——
    results_dir = resolve_path(data_cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{ts_code}_view.html")
    save_html(fig, out_path)
    if verbose:
        print(f"[6/6] 已保存可视化图表：{out_path}")
    return out_path


def _parse_args(argv: list):
    """解析命令行参数：[ts_code] [k]。

    - 无参数         → 用股票池第一只标的、配置默认 K；
    - 仅 ts_code     → 指定标的、配置默认 K；
    - ts_code + k    → 同时指定标的与网格 K 值。
    """
    ts_code = None
    k = None
    if len(argv) >= 2:
        ts_code = argv[1]
    if len(argv) >= 3:
        try:
            k = float(argv[2])
        except ValueError:
            print(f"忽略非法的 K 参数：{argv[2]}（将使用配置默认 K）")
    if ts_code is None:
        pool = get_stock_pool()
        ts_code = pool[0]
    return ts_code, k


if __name__ == "__main__":
    code, k_val = _parse_args(sys.argv)
    make_view(code, k=k_val)
