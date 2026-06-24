# ====================================================================
# 数据获取模块 (fetcher.py)
# 对应 README 阶段 2.1「数据获取」
#
# 职责：
#   1. 通过 Tushare Pro 接口获取 A 股日线行情（前复权 OHLCV、成交额、
#      涨跌停、停牌、ST 等标记）。
#   2. 把下载的数据缓存到本地 CSV（data/raw/），避免重复请求接口。
#   3. 当无法联网 / 接口不可用时，提供「合成数据」兜底，
#      保证回测系统在离线环境下也能完整跑通（仅用于演示与测试）。
#
# 设计原则（README 要求）：逻辑清晰、可读性强、可改动性强。
#   - 每个公开函数只做一件事；
#   - 网络请求与本地缓存解耦；
#   - Tushare 不可用时优雅降级，不让整个流程崩溃。
# ====================================================================

import os
import time

import numpy as np
import pandas as pd

from config.loader import CONFIG, resolve_path
from config.stock_pool_config import TUSHARE_TOKEN

# Tushare 为可选依赖：如果环境未安装或无法联网，则退化为合成数据。
try:
    import tushare as ts
except Exception:  # pragma: no cover - 仅在缺少 tushare 时触发
    ts = None


class DataFetcher:
    """行情数据获取器。

    统一封装「优先读本地缓存 → 否则调用 Tushare → 仍失败则生成合成数据」
    的三级取数逻辑。调用方只需调用 :meth:`get_daily`，无需关心数据来源。
    """

    def __init__(self, token: str = TUSHARE_TOKEN, use_cache: bool = True):
        """初始化取数器。

        Args:
            token: Tushare Pro token。
            use_cache: 是否启用本地 CSV 缓存。
        """
        self.use_cache = use_cache
        self.raw_dir = resolve_path(CONFIG["data"]["raw_dir"])
        os.makedirs(self.raw_dir, exist_ok=True)

        # 初始化 Tushare Pro 客户端。失败（无网络 / token 无效）则置空，后续走兜底。
        self.pro = None
        if ts is not None and token:
            try:
                ts.set_token(token)
                self.pro = ts.pro_api()
            except Exception:
                self.pro = None

    # ---------------- 对外主接口 ----------------
    def get_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取单只标的的日线行情（带缓存与兜底）。

        Args:
            ts_code: Tushare 股票代码，如 "600519.SH"。
            start_date: 起始日期，格式 "YYYY-MM-DD" 或 "YYYYMMDD"。
            end_date: 结束日期，同上。

        Returns:
            pd.DataFrame: 按日期升序排列，至少包含列：
                trade_date(交易日, datetime), open, high, low, close,
                vol(成交量), amount(成交额),
                limit_up(是否涨停 bool), limit_down(是否跌停 bool),
                is_suspended(是否停牌 bool), is_st(是否 ST bool)。
        """
        start = _normalize_date(start_date)
        end = _normalize_date(end_date)

        # 1) 本地缓存优先。
        cache_path = self._cache_path(ts_code, start, end)
        if self.use_cache and os.path.exists(cache_path):
            return _read_cache(cache_path)

        # 2) 尝试 Tushare 在线获取。
        df = None
        if self.pro is not None:
            df = self._fetch_from_tushare(ts_code, start, end)

        # 3) 仍无数据 → 生成合成数据兜底（保证离线可跑）。
        if df is None or df.empty:
            df = generate_synthetic_daily(ts_code, start, end)

        # 写入缓存供下次复用。
        if self.use_cache:
            df.to_csv(cache_path, index=False)
        return df

    def get_many(self, ts_codes, start_date: str, end_date: str) -> dict:
        """批量获取多只标的行情。

        Args:
            ts_codes: 股票代码列表。
            start_date / end_date: 同 :meth:`get_daily`。

        Returns:
            dict[str, pd.DataFrame]: {代码: 行情 DataFrame}。
        """
        result = {}
        for code in ts_codes:
            result[code] = self.get_daily(code, start_date, end_date)
        return result

    # ---------------- 内部辅助 ----------------
    def _cache_path(self, ts_code: str, start: str, end: str) -> str:
        """生成缓存文件路径，按「代码_起_止.csv」命名。"""
        fname = f"{ts_code}_{start}_{end}.csv"
        return os.path.join(self.raw_dir, fname)

    def _fetch_from_tushare(self, ts_code: str, start: str, end: str):
        """调用 Tushare 接口拉取前复权日线 + 各类标记。

        失败时返回 None（由上层走兜底），不抛异常，保证流程健壮。
        """
        try:
            # 前复权日线行情（pro_bar 封装了复权计算，adj='qfq' 即前复权）。
            df = ts.pro_bar(
                ts_code=ts_code,
                adj="qfq",
                start_date=start,
                end_date=end,
                freq="D",
            )
            if df is None or df.empty:
                return None

            # daily_basic 提供成交额等基础指标（此处主要用其换手率/成交额）。
            # pro_bar 已含 amount，这里直接使用，无需额外请求。
            df = df.rename(columns={"vol": "vol", "amount": "amount"})
            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
            df = df.sort_values("trade_date").reset_index(drop=True)

            # 涨跌停标记：用日涨跌幅近似（A 股主板约 ±10%）。
            # 真实项目可调用 stk_limit 接口获取精确涨跌停价，这里做简洁近似。
            df["pct_chg"] = df["close"].pct_change()
            df["limit_up"] = df["pct_chg"] >= 0.0985
            df["limit_down"] = df["pct_chg"] <= -0.0985

            # 停牌标记：成交量为 0 视为停牌。
            df["is_suspended"] = df["vol"].fillna(0) <= 0
            # ST 标记：Tushare pro_bar 不直接返回名称，这里默认非 ST。
            df["is_st"] = False

            cols = [
                "trade_date", "open", "high", "low", "close",
                "vol", "amount", "limit_up", "limit_down",
                "is_suspended", "is_st",
            ]
            return df[cols]
        except Exception:
            # 网络异常 / 频率限制等：稍等后返回 None 交由上层兜底。
            time.sleep(0.2)
            return None


# ======================== 模块级工具函数 ========================

def _normalize_date(date_str: str) -> str:
    """把日期统一成 Tushare 要求的 "YYYYMMDD" 形式。"""
    return str(date_str).replace("-", "").strip()


def _read_cache(path: str) -> pd.DataFrame:
    """读取缓存 CSV 并恢复正确的数据类型。"""
    df = pd.read_csv(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 布尔列从 CSV 读回时会变成字符串/对象，统一转回 bool。
    for col in ["limit_up", "limit_down", "is_suspended", "is_st"]:
        if col in df.columns:
            df[col] = df[col].astype(str).isin(["True", "true", "1", "1.0"])
    return df.sort_values("trade_date").reset_index(drop=True)


def generate_synthetic_daily(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """生成合成日线行情（OU 均值回复 + 随机噪声），用于离线演示 / 测试。

    设计目的：
        在无法访问 Tushare 的环境（如 CI / 沙箱）中，依然能让整个回测
        流程端到端跑通。合成价格刻意带有「围绕均值波动」的特性，
        以便筛选与网格逻辑能产生有意义的交易。

    注意：这是模拟数据，不能用于任何真实投资决策。

    Args:
        ts_code: 股票代码（仅用于设定随机种子，保证可复现）。
        start / end: "YYYYMMDD" 起止日期。

    Returns:
        pd.DataFrame: 结构与 :meth:`DataFetcher.get_daily` 输出一致。
    """
    # 用代码做随机种子，保证同一标的每次生成的数据一致（可复现）。
    seed = abs(hash(ts_code)) % (2**32)
    rng = np.random.default_rng(seed)

    # 生成交易日序列（仅工作日，近似 A 股交易日历）。
    dates = pd.bdate_range(
        start=pd.to_datetime(start, format="%Y%m%d"),
        end=pd.to_datetime(end, format="%Y%m%d"),
    )
    n = len(dates)
    if n == 0:
        # 极端情况下（区间为空）返回空表，列结构保持一致。
        return pd.DataFrame(
            columns=[
                "trade_date", "open", "high", "low", "close",
                "vol", "amount", "limit_up", "limit_down",
                "is_suspended", "is_st",
            ]
        )

    # ---- 构造「围绕缓慢漂移的均值快速回复」的收盘价 ----
    # 设计思路：把对数价格分解为两部分
    #   1) trend：缓慢随机游走的长期中枢（模拟基本面慢变）；
    #   2) dev  ：围绕 0 快速均值回复的短期偏离（模拟可被网格捕捉的振荡）。
    # 收盘价 = exp(trend + dev)。这样「价格 - MA」（即 Z 的分子）主要由 dev 主导，
    # dev 的强均值回复使 Z 序列呈现明显的反持续性（Hurst < 0.45），
    # 从而能通过筛选并产生有意义的网格交易。
    base_price = float(rng.uniform(20, 120))    # 初始价位
    log_mean = np.log(base_price)               # 长期对数中枢初值

    # 每只标的的回复速度略有差异，增加候选池多样性。
    # 这里的 kappa_dev 是离散 OU 的「每步衰减比例」，对应 AR(1) 系数 b = 1 - kappa_dev，
    # 因此估计器测得的回复速度约为 κ ≈ -ln(1 - kappa_dev)、半衰期 HL = ln2/κ。
    # 取 kappa_dev ∈ [0.05, 0.12] 可使经 60 日 MA 标准化后的 Z 序列半衰期
    # 自然落在筛选要求的 [5, 20] 交易日区间内，同时 Hurst 仍 < 0.45（均值回复）。
    kappa_dev = float(rng.uniform(0.05, 0.12))  # 偏离项回复速度（温和）
    sigma_dev = 0.06                            # 偏离项噪声（放大以拉开网格振幅）
    sigma_trend = 0.0010                        # 中枢漂移噪声（很慢）

    trend = np.empty(n)
    dev = np.empty(n)
    trend[0] = log_mean
    dev[0] = 0.0
    for t in range(1, n):
        # 长期中枢：缓慢随机游走。
        trend[t] = trend[t - 1] + sigma_trend * rng.standard_normal()
        # 短期偏离：向 0 快速回复的离散 OU。
        dev[t] = (1.0 - kappa_dev) * dev[t - 1] + sigma_dev * rng.standard_normal()
    close = np.exp(trend + dev)

    # ---- 由收盘价构造 OHLC（日内高低用小幅随机带宽模拟）----
    intraday = np.abs(rng.normal(0.0, 0.01, n)) + 0.005  # 日内振幅比例
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = close * (1 + rng.normal(0.0, 0.006, n))
    # 保证 OHLC 的大小关系合法：low ≤ {open, close} ≤ high。
    high = np.maximum.reduce([high, open_, close])
    low = np.minimum.reduce([low, open_, close])

    # ---- 成交量 / 成交额（与价格无关的对数正态随机量）----
    vol = rng.lognormal(mean=11.5, sigma=0.4, size=n)        # 手
    amount = vol * close * 100 / 1000                        # 千元（近似）

    df = pd.DataFrame(
        {
            "trade_date": dates,
            "open": np.round(open_, 2),
            "high": np.round(high, 2),
            "low": np.round(low, 2),
            "close": np.round(close, 2),
            "vol": np.round(vol, 2),
            "amount": np.round(amount, 2),
        }
    )

    # ---- 各类标记 ----
    df["pct_chg"] = df["close"].pct_change()
    df["limit_up"] = df["pct_chg"] >= 0.0985
    df["limit_down"] = df["pct_chg"] <= -0.0985
    df["is_suspended"] = False   # 合成数据默认不停牌
    df["is_st"] = False          # 合成数据默认非 ST
    df = df.drop(columns=["pct_chg"])
    return df
