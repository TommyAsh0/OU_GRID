# ====================================================================
# 自定义多标的回测入口脚本 (bt_custom.py)
# 对应 README 阶段 8「历史回测」的「自定义回测」补充
#
# 用途：
#   面向「指定标的 + 指定区间」的轻量回测。与 run_backtest.py 不同，
#   本脚本【跳过】训练段筛选与验证段选 K 的 Walk-Forward 流程，直接对
#   用户给定的一组标的、在给定时间区间上回测，便于针对自选标的快速验证。
#
# 满足的需求：
#   1. 自定义标的：兼容多个股票（逗号分隔或多次 --code 传入）。
#   2. 自定义开始时间：--start（缺省回退配置 data.start_date）。
#   3. 自定义结束时间（可选）：--end（缺省回退配置 data.end_date）。
#   4. 是否使用「自动开仓系统」：--auto-open / --no-auto-open。
#      开启 → 按之前版本的 Regime 进行开仓与四级风控（自动平仓 / 暂停）。
#      关闭 → 不自动开仓、不触发风控交易；但 Regime / 因子 / 指标仍逐日
#             照常计算并保留，仅用于可视化与数据准备。
#   5. 网格参数沿用之前的模型设置（config/settings.yaml 的 grid.* 等），
#      本脚本不另立网格参数。
#
# 处理流程（与 bt_view.py / run_backtest.py 的单标的链路完全同源）：
#   对每个标的：
#     1. 取数    : DataFetcher.get_daily 读本地行情（缺失则合成兜底）。
#     2. 指标    : DataProcessor.process 计算 MA/ATR/Z 并清洗。
#     3. 回测    : BacktestEngine.run（按 auto_open 决定是否自动开仓）。
#     4. 盘前因子: visualizer.compute_premarket_factors 逐日滚动重算。
#     5. 绘图    : visualizer.build_figure / save_html 输出交互式 HTML。
#     6. 落盘    : 每标的绩效 / 成交 CSV，外加一张多标的绩效汇总 CSV。
#
# 运行方式：
#   python bt_custom.py --code 600519.SH
#   python bt_custom.py --code 600519.SH,000858.SZ --start 2021-01-01
#   python bt_custom.py --code 600519.SH --start 2021-01-01 --end 2023-12-31
#   python bt_custom.py --code 600519.SH --no-auto-open   # 仅算因子、不开仓
#   python bt_custom.py --code 600519.SH --k 1.2 --no-view # 指定 K、不出图
#
# 说明：沙箱无法访问 Tushare 时会自动使用合成数据，整条流程仍可完整跑通，
#       但结果仅用于演示系统正确性，不能用于真实投资决策。
# ====================================================================

import argparse
import os
import warnings

import pandas as pd

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


def run_symbol(ts_code: str, start: str, end: str, capital: float,
               auto_open: bool = True, k: float = None,
               make_html: bool = True, verbose: bool = True) -> dict:
    """对单只标的执行一次自定义区间回测，并（可选）输出可视化 HTML。

    Args:
        ts_code: 标的代码，如 "600519.SH"。
        start: 回测开始日期（"YYYY-MM-DD" 或 "YYYYMMDD"）。
        end: 回测结束日期（同上）。
        capital: 分配给该标的的资金。
        auto_open: 是否启用「自动开仓系统」（Regime 开仓 + 四级风控）。
            关闭时仅计算因子 / Regime 用于可视化，不产生任何成交。
        k: 网格 ATR 倍数 K；None 则沿用配置默认值 grid.K。
        make_html: 是否生成可视化 HTML（含盘前因子 / Regime）。
        verbose: 是否打印过程信息。

    Returns:
        dict: {
            "ts_code", "result"(回测结果或 None), "html"(HTML 路径或 ""),
            "skipped"(bool, 数据不足时为 True)
        }
    """
    if verbose:
        print(f"\n>>> 标的 {ts_code}（{start} ~ {end}）")

    # —— 1) 取数（本地缓存优先，缺失则合成兜底）——
    fetcher = DataFetcher()
    raw = fetcher.get_daily(ts_code, start, end)

    # —— 2) 清洗 + 指标计算 ——
    processor = DataProcessor()
    df = processor.process(raw)
    if df.empty or len(df) < 30:
        if verbose:
            print(f"    数据不足（有效 {len(df)} 行），跳过该标的。")
        return {"ts_code": ts_code, "result": None, "html": "", "skipped": True}

    # —— 3) 回测（auto_open 控制是否自动开仓 / 风控）——
    # 关闭自动开仓时仍保留 Regime 计算（enable_regime=True），以便因子 /
    # Regime 进入每日快照与可视化；同时关闭风控自动交易（enable_risk=False）。
    bt = BacktestEngine(
        ts_code, df, capital, k=k,
        enable_regime=True,
        enable_risk=auto_open,
        enable_open=auto_open,
    )
    result = bt.run()
    fills = result["fills"]
    if verbose:
        n_buy = sum(1 for f in fills if f.side.value == "BUY")
        n_sell = len(fills) - n_buy
        m = result["metrics"]
        mode = "自动开仓" if auto_open else "仅因子(不开仓)"
        print(f"    模式={mode}；成交：买 {n_buy} / 卖 {n_sell} "
              f"（共 {len(fills)} 笔）")
        print(f"    年化={_pct(m['annual_return'])} "
              f"最大回撤={_pct(m['max_drawdown'])} "
              f"Calmar={_num(m['calmar'])} Sharpe={_num(m['sharpe'])}")

    # —— 4) & 5) 盘前因子 + 可视化 HTML（含 Regime 背景着色）——
    html_path = ""
    if make_html:
        factors = compute_premarket_factors(df)
        fig = build_figure(df, fills, factors, ts_code)
        results_dir = resolve_path(CONFIG["data"]["results_dir"])
        os.makedirs(results_dir, exist_ok=True)
        suffix = "auto" if auto_open else "factors"
        html_path = os.path.join(results_dir, f"{ts_code}_custom_{suffix}.html")
        save_html(fig, html_path)
        if verbose:
            print(f"    可视化已保存：{html_path}")

    return {"ts_code": ts_code, "result": result, "html": html_path,
            "skipped": False}


def run_custom(codes, start: str = None, end: str = None,
               auto_open: bool = True, k: float = None,
               make_html: bool = True, verbose: bool = True) -> dict:
    """对一组自定义标的在指定区间上逐个回测，并汇总落盘。

    Args:
        codes: 标的代码列表；为空 / None 时回退配置股票池。
        start: 开始日期；None 回退配置 data.start_date。
        end: 结束日期（可选）；None 回退配置 data.end_date。
        auto_open: 是否启用自动开仓系统（见 run_symbol）。
        k: 网格 ATR 倍数 K；None 沿用配置默认。
        make_html: 是否生成每标的可视化 HTML。
        verbose: 是否打印过程信息与汇总报告。

    Returns:
        dict: {
            "results": {code: 回测结果},
            "summary": DataFrame,   # 各标的绩效汇总
            "skipped": list,        # 数据不足被跳过的标的
        }
    """
    data_cfg = CONFIG["data"]
    # 自定义起止时间：缺省回退配置区间（结束时间为可选项）。
    start = start or data_cfg["start_date"]
    end = end or data_cfg["end_date"]
    capital = CONFIG["backtest"]["initial_capital"]

    pool = list(codes) if codes else get_stock_pool()

    if verbose:
        mode = "开启" if auto_open else "关闭（仅因子 / 可视化）"
        print("=" * 70)
        print("自定义多标的回测")
        print(f"  标的数        : {len(pool)}")
        print(f"  区间          : {start} ~ {end}")
        print(f"  自动开仓系统  : {mode}")
        print(f"  网格 K        : {'配置默认' if k is None else k}（其余网格参数沿用配置）")
        print("=" * 70)

    results = {}
    skipped = []
    for code in pool:
        out = run_symbol(code, start, end, capital,
                         auto_open=auto_open, k=k,
                         make_html=make_html, verbose=verbose)
        if out["skipped"]:
            skipped.append(code)
            continue
        results[code] = out["result"]

    summary = _build_summary(results, k)

    if verbose:
        _print_summary(summary, skipped)
    _save_outputs(results, summary, auto_open)

    return {"results": results, "summary": summary, "skipped": skipped}


# ======================== 汇总与落盘 ========================

def _build_summary(results: dict, k) -> pd.DataFrame:
    """把各标的绩效汇总成一张表（每行一只标的）。"""
    rows = []
    for code, res in results.items():
        m = res["metrics"]
        rows.append({
            "ts_code": code,
            "K": (CONFIG["grid"]["K"] if k is None else k),
            "total_return": m["total_return"],
            "annual_return": m["annual_return"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "win_rate": m["win_rate"],
            "profit_loss_ratio": m["profit_loss_ratio"],
            "n_trades": m["n_trades"],
        })
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame, skipped: list) -> None:
    """在终端打印各标的绩效汇总。"""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 70)
    print("各标的回测绩效汇总")
    print("=" * 70)
    if summary.empty:
        print("  （无有效标的结果）")
    else:
        show = summary.copy()
        for c in ["total_return", "annual_return", "max_drawdown",
                  "calmar", "sharpe", "win_rate", "profit_loss_ratio"]:
            show[c] = show[c].map(lambda v: f"{v:.4g}")
        print(show.to_string(index=False))
    if skipped:
        print(f"\n  数据不足被跳过：{skipped}")
    print("=" * 70 + "\n")


def _save_outputs(results: dict, summary: pd.DataFrame, auto_open: bool) -> None:
    """把绩效汇总、各标的成交明细与每日快照落盘到 data/results/。"""
    results_dir = resolve_path(CONFIG["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    tag = "auto" if auto_open else "factors"

    # 1) 绩效汇总。
    if not summary.empty:
        summary.to_csv(
            os.path.join(results_dir, f"custom_summary_{tag}.csv"),
            index=False,
        )

    # 2) 各标的成交明细（合并）。
    fill_rows = []
    for code, res in results.items():
        for f in res["fills"]:
            fill_rows.append({
                "ts_code": code, "trade_date": f.trade_date,
                "layer": f.layer, "side": f.side.value,
                "price": f.price, "quantity": f.quantity,
                "amount": f.amount, "cost": f.cost,
                "realized_pnl": f.realized_pnl,
            })
    if fill_rows:
        pd.DataFrame(fill_rows).to_csv(
            os.path.join(results_dir, f"custom_fills_{tag}.csv"), index=False)

    # 3) 各标的每日快照（含 regime / position_scale，便于数据准备与可视化）。
    for code, res in results.items():
        daily = res["daily_log"]
        if daily is not None and not daily.empty:
            daily.to_csv(
                os.path.join(results_dir, f"custom_daily_{code}_{tag}.csv"),
                index=False,
            )


# ======================== 工具与命令行 ========================

def _pct(v) -> str:
    """把比率格式化为百分比字符串（NaN → '-'）。"""
    try:
        if v != v:  # NaN
            return "-"
        return f"{v:.2%}"
    except (TypeError, ValueError):
        return "-"


def _num(v) -> str:
    """把数值格式化为 3 位有效数字字符串（NaN → '-'）。"""
    try:
        if v != v:  # NaN
            return "-"
        return f"{v:.3f}"
    except (TypeError, ValueError):
        return "-"


def _parse_codes(raw) -> list:
    """把 --code 参数解析为标的列表（支持逗号分隔与多次传入）。"""
    if not raw:
        return []
    codes = []
    for item in raw:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                codes.append(part)
    return codes


def _build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="自定义多标的回测（自定义标的 / 区间 / 是否自动开仓）。",
    )
    parser.add_argument(
        "--code", action="append", default=None,
        help="标的代码，可逗号分隔或多次传入；缺省用配置股票池。"
             "例如 --code 600519.SH,000858.SZ",
    )
    parser.add_argument(
        "--start", default=None,
        help="回测开始日期 YYYY-MM-DD；缺省回退配置 data.start_date。",
    )
    parser.add_argument(
        "--end", default=None,
        help="回测结束日期 YYYY-MM-DD（可选）；缺省回退配置 data.end_date。",
    )
    parser.add_argument(
        "--k", type=float, default=None,
        help="网格 ATR 倍数 K；缺省沿用配置 grid.K（其余网格参数始终沿用配置）。",
    )
    # 自动开仓系统开关：默认开启；--no-auto-open 关闭。
    parser.add_argument(
        "--auto-open", dest="auto_open", action="store_true", default=True,
        help="启用自动开仓系统（按 Regime 开仓 + 四级风控）。默认开启。",
    )
    parser.add_argument(
        "--no-auto-open", dest="auto_open", action="store_false",
        help="关闭自动开仓系统：仅计算因子 / Regime 用于可视化，不产生成交。",
    )
    parser.add_argument(
        "--no-view", dest="make_html", action="store_false", default=True,
        help="不生成可视化 HTML（仅跑回测并输出绩效 / 成交 CSV）。",
    )
    return parser


def main(argv=None) -> dict:
    """命令行入口：解析参数并执行自定义回测。"""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    codes = _parse_codes(args.code)
    return run_custom(
        codes,
        start=args.start,
        end=args.end,
        auto_open=args.auto_open,
        k=args.k,
        make_html=args.make_html,
    )


if __name__ == "__main__":
    main()
