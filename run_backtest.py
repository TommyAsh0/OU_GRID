# ====================================================================
# 回测主程序 (run_backtest.py)
# 对应 README 阶段 8「历史回测」（8.1 数据划分 / 8.2 K 值选择 / 8.4 评估）
#
# 这是整个回测系统的「总调度」，把各模块串成一条端到端流水线：
#
#   1. 取数        : DataFetcher 拉取候选池行情（在线 Tushare / 离线合成兜底）。
#   2. 划分窗口    : 按训练 / 验证 / 测试三段切分（Walk-Forward 的单轮）。
#   3. 标的筛选    : 在训练段上用 Screener 选出合格标的（ADF/KPSS/Hurst/OU/HL/CV）。
#   4. K 值选择    : 在验证段上对每只标的逐个 K 回测，按 Calmar 选最优 K（README 8.2）。
#   5. 测试评估    : 在测试段上用选定 K 回测，汇总绩效（README 8.4）。
#   6. 输出报告    : 打印筛选明细、每标的测试绩效、组合汇总，并落盘 CSV。
#
# 运行方式：
#   python run_backtest.py
#
# 说明：沙箱无法访问 Tushare 时会自动使用合成数据，整条流程仍可完整跑通，
#       但结果仅用于演示系统正确性，不能用于真实投资决策。
# ====================================================================

import os
import warnings

import pandas as pd

from config.loader import CONFIG, resolve_path
from config.stock_pool_config import get_stock_pool
from src.backtest.engine import BacktestEngine
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.screening.screener import Screener

warnings.filterwarnings("ignore")


def slice_by_date(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """按日期区间 [start, end) 截取行情子段（左闭右开）。"""
    mask = (df["trade_date"] >= start) & (df["trade_date"] < end)
    return df[mask].reset_index(drop=True)


def split_windows() -> dict:
    """根据配置的起止日期与训练/验证/测试年数，划分三段时间窗口。

    采用单轮 Walk-Forward（README 8.1）：
        训练段 → 筛选标的；验证段 → 选 K；测试段 → 最终评估。

    Returns:
        dict: {"train": (s,e), "validation": (s,e), "test": (s,e)}，
              各值为 (起始, 结束) 的 pandas.Timestamp 对。
    """
    bt = CONFIG["backtest"]
    data_cfg = CONFIG["data"]
    start = pd.Timestamp(data_cfg["start_date"])

    train_end = start + pd.DateOffset(years=bt["train_years"])
    val_end = train_end + pd.DateOffset(years=bt["validation_years"])
    test_end = val_end + pd.DateOffset(years=bt["test_years"])

    return {
        "train": (start, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, test_end),
    }


def select_best_k(ts_code: str, val_df: pd.DataFrame, capital: float):
    """在验证段上为单只标的选择最优 K（README 8.2）。

    对每个候选 K 回测，主指标 Calmar 最高者胜出；
    Calmar 接近（差异 < 10%）时偏好更大的 K（更宽网格更鲁棒）。
    约束 Sharpe > 0.5（验证段较短时作为软约束，若全不满足则退而取 Calmar 最高）。

    Args:
        ts_code: 标的代码。
        val_df: 验证段行情（含指标）。
        capital: 分配给该标的的资金。

    Returns:
        (best_k, table): 最优 K 及各 K 的绩效明细 DataFrame。
    """
    candidates = CONFIG["grid"]["K_candidates"]
    rows = []
    for k in candidates:
        bt = BacktestEngine(ts_code, val_df, capital, k=k)
        m = bt.run()["metrics"]
        rows.append({
            "K": k,
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "win_rate": m["win_rate"],
            "annual_return": m["annual_return"],
            "max_drawdown": m["max_drawdown"],
        })
    table = pd.DataFrame(rows)

    # 优先在满足 Sharpe > 0.5 的候选里挑 Calmar 最高；若无则放宽。
    eligible = table[table["sharpe"] > 0.5]
    pool = eligible if not eligible.empty else table
    best_calmar = pool["calmar"].max()
    # Calmar 在最高值 90% 以上视为「接近」，其中取 K 最大者。
    close = pool[pool["calmar"] >= best_calmar * 0.9]
    best_k = float(close.sort_values("K", ascending=False).iloc[0]["K"])
    return best_k, table


def run_pipeline(stock_pool=None, verbose: bool = True) -> dict:
    """执行完整的「筛选 → 选 K → 测试」回测流水线。

    Args:
        stock_pool: 候选标的列表；None 则用 config 默认股票池。
        verbose: 是否打印过程信息。

    Returns:
        dict: {
            "screen_detail": DataFrame,   # 训练段筛选明细
            "passed": list,               # 合格标的
            "k_selection": dict,          # {code: best_k}
            "test_results": dict,         # {code: 测试段回测结果}
            "portfolio": dict,            # 组合层面汇总
        }
    """
    data_cfg = CONFIG["data"]
    pool = stock_pool or get_stock_pool()
    windows = split_windows()

    # —— 1) 取数（整段，后续再按窗口切片）——
    fetcher = DataFetcher()
    full_start = data_cfg["start_date"]
    full_end = data_cfg["end_date"]
    if verbose:
        print(f"[1/5] 获取 {len(pool)} 只候选标的行情 "
              f"({full_start} ~ {full_end}) ...")
    raw = fetcher.get_many(pool, full_start, full_end)

    # —— 2) 训练段筛选 ——
    if verbose:
        s, e = windows["train"]
        print(f"[2/5] 训练段筛选标的 ({s.date()} ~ {e.date()}) ...")
    train_raw = {c: slice_by_date(df, *windows["train"]) for c, df in raw.items()}
    screener = Screener()
    screen_out = screener.screen(train_raw)
    passed = screen_out["passed"]
    if verbose:
        print(f"       合格标的（{len(passed)}）：{passed}")

    if not passed:
        if verbose:
            print("       没有标的通过筛选，流程结束。")
        return {
            "screen_detail": screen_out["detail"], "passed": [],
            "k_selection": {}, "test_results": {}, "portfolio": {},
        }

    # 资金分配：在「策略可用资金」内对合格标的等额分配，遵守单标的上限。
    capital_per_symbol = _capital_per_symbol(len(passed))

    processor = DataProcessor()
    k_selection = {}
    test_results = {}

    # —— 3) & 4) 逐只标的：验证段选 K → 测试段评估 ——
    if verbose:
        vs, ve = windows["validation"]
        ts, te = windows["test"]
        print(f"[3/5] 验证段选择 K ({vs.date()} ~ {ve.date()}) ...")
        print(f"[4/5] 测试段评估 ({ts.date()} ~ {te.date()}) ...")
    for code in passed:
        full_df = processor.process(raw[code])
        val_df = slice_by_date(full_df, *windows["validation"])
        test_df = slice_by_date(full_df, *windows["test"])
        # 验证 / 测试段数据过短则跳过该标的。
        if len(val_df) < 30 or len(test_df) < 30:
            continue

        best_k, k_table = select_best_k(code, val_df, capital_per_symbol)
        k_selection[code] = {"best_k": best_k, "table": k_table}

        # 用选定 K 在测试段回测（不再回看验证集调参）。
        bt = BacktestEngine(code, test_df, capital_per_symbol, k=best_k)
        test_results[code] = bt.run()

    # —— 5) 组合汇总 ——
    portfolio = aggregate_portfolio(test_results, capital_per_symbol)

    if verbose:
        print("[5/5] 汇总绩效并输出报告 ...")
        _print_report(screen_out["detail"], k_selection, test_results, portfolio)
        _save_results(screen_out["detail"], test_results, portfolio)

    return {
        "screen_detail": screen_out["detail"],
        "passed": passed,
        "k_selection": k_selection,
        "test_results": test_results,
        "portfolio": portfolio,
    }


def aggregate_portfolio(test_results: dict, capital_per_symbol: float) -> dict:
    """把各标的测试段净值合并为组合净值并计算组合绩效。

    组合净值 = 各标的净值按日对齐后求和（等额资金、独立运行）。
    缺失日（某标的停牌 / 区间不一致）用前向填充，再用初始资金回填。

    Args:
        test_results: {code: 单标的回测结果}。
        capital_per_symbol: 单标的初始资金。

    Returns:
        dict: {"equity_curve": Series, "metrics": dict, "n_symbols": int}。
    """
    from src.backtest.metrics import compute_metrics

    if not test_results:
        return {"equity_curve": pd.Series(dtype=float), "metrics": {}, "n_symbols": 0}

    # 收集各标的净值曲线，按日期对齐成一张表。
    curves = {}
    all_fills = []
    for code, res in test_results.items():
        curve = res["equity_curve"]
        if not curve.empty:
            curves[code] = curve
            all_fills.extend(res["fills"])

    if not curves:
        return {"equity_curve": pd.Series(dtype=float), "metrics": {}, "n_symbols": 0}

    combined = pd.DataFrame(curves).sort_index()
    # 各标的初始资金已知：缺失日先前向填充，期初缺失回填为该标的初始资金。
    combined = combined.ffill().fillna(capital_per_symbol)
    portfolio_equity = combined.sum(axis=1)

    metrics = compute_metrics(portfolio_equity, all_fills)
    return {
        "equity_curve": portfolio_equity,
        "metrics": metrics,
        "n_symbols": len(curves),
    }


# ======================== 内部工具 ========================

def _capital_per_symbol(n_passed: int) -> float:
    """计算分配给每只标的的资金。

    可用于网格的资金 = 初始资金 ×(1 − 现金储备)，再对合格标的等额分配；
    同时不超过单标的最大占比（max_single_weight × 初始资金）。
    """
    init_cap = CONFIG["backtest"]["initial_capital"]
    risk = CONFIG["risk"]
    investable = init_cap * (1.0 - risk["cash_reserve"])
    per = investable / max(n_passed, 1)
    cap_limit = init_cap * risk["max_single_weight"]
    return min(per, cap_limit)


def _print_report(detail, k_selection, test_results, portfolio) -> None:
    """在终端打印筛选明细、各标的测试绩效与组合汇总。"""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n" + "=" * 70)
    print("筛选明细（训练段）")
    print("=" * 70)
    cols = ["ts_code", "n_days", "adf_pvalue", "kpss_pvalue", "hurst",
            "half_life", "kappa_cv", "passed", "reject_reason"]
    show = detail[cols].copy()
    for c in ["adf_pvalue", "kpss_pvalue", "hurst", "half_life", "kappa_cv"]:
        show[c] = show[c].map(lambda v: f"{v:.4g}")
    print(show.to_string(index=False))

    print("\n" + "=" * 70)
    print("各标的测试段绩效")
    print("=" * 70)
    rows = []
    for code, res in test_results.items():
        m = res["metrics"]
        rows.append({
            "ts_code": code,
            "K": k_selection.get(code, {}).get("best_k", float("nan")),
            "annual_return": m["annual_return"],
            "max_drawdown": m["max_drawdown"],
            "calmar": m["calmar"],
            "sharpe": m["sharpe"],
            "win_rate": m["win_rate"],
            "pl_ratio": m["profit_loss_ratio"],
            "n_trades": m["n_trades"],
        })
    if rows:
        perf = pd.DataFrame(rows)
        for c in ["annual_return", "max_drawdown", "calmar", "sharpe",
                  "win_rate", "pl_ratio"]:
            perf[c] = perf[c].map(lambda v: f"{v:.4g}")
        print(perf.to_string(index=False))

    print("\n" + "=" * 70)
    print("组合汇总（测试段）")
    print("=" * 70)
    pm = portfolio.get("metrics", {})
    if pm:
        print(f"  纳入标的数      : {portfolio['n_symbols']}")
        print(f"  累计收益率      : {pm['total_return']:.2%}")
        print(f"  年化收益率      : {pm['annual_return']:.2%}")
        print(f"  最大回撤        : {pm['max_drawdown']:.2%}")
        print(f"  Calmar 比率     : {pm['calmar']:.3f}")
        print(f"  Sharpe 比率     : {pm['sharpe']:.3f}")
        print(f"  胜率            : {pm['win_rate']:.2%}")
        print(f"  盈亏比          : {pm['profit_loss_ratio']:.3f}")
        print(f"  最长回撤天数    : {pm['longest_drawdown_days']}")
        print(f"  95% CVaR(日)    : {pm['cvar_95']:.4%}")
        print(f"  总成交笔数      : {pm['n_trades']}")
    print("=" * 70 + "\n")


def _save_results(detail, test_results, portfolio) -> None:
    """把筛选明细、组合净值、各标的成交落盘到 data/results/。"""
    results_dir = resolve_path(CONFIG["data"]["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    detail.to_csv(os.path.join(results_dir, "screening_detail.csv"), index=False)

    pe = portfolio.get("equity_curve")
    if pe is not None and not pe.empty:
        pe.to_frame("equity").to_csv(
            os.path.join(results_dir, "portfolio_equity.csv"))

    # 各标的成交明细合并落盘。
    fill_rows = []
    for code, res in test_results.items():
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
            os.path.join(results_dir, "test_fills.csv"), index=False)


if __name__ == "__main__":
    run_pipeline()
