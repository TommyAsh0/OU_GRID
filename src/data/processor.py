# ====================================================================
# 数据清洗与指标计算模块 (processor.py)
# 对应 README 阶段 2.2「数据清洗规则」与 2.3「指标计算」
#
# 职责：
#   1. 清洗原始行情：处理停牌、ST、次新股、异常涨跌幅、长期停牌等。
#   2. 计算核心指标：MA(均线)、ATR(真实波幅)、Z(标准化偏离度)。
#
# 这些指标是后续所有环节的基础：
#   - Z 序列    → 用于平稳性检验与 OU 参数估计；
#   - MA / ATR  → 用于构建网格价格（中轴与间距）。
# ====================================================================

import numpy as np
import pandas as pd

from config.loader import CONFIG


class DataProcessor:
    """行情清洗与指标计算器。

    输入单只标的的原始日线 DataFrame（来自 :class:`DataFetcher`），
    输出清洗完毕、并附加 MA/ATR/Z 指标的 DataFrame。
    """

    def __init__(self, ma_length: int = None, atr_length: int = None):
        """初始化处理器。

        Args:
            ma_length: MA 均线周期，默认读取配置 indicators.ma_length。
            atr_length: ATR 周期，默认读取配置 indicators.atr_length。
        """
        ind = CONFIG["indicators"]
        self.ma_length = ma_length or ind["ma_length"]
        self.atr_length = atr_length or ind["atr_length"]
        self.min_listed_days = CONFIG["screening"]["min_listed_days"]

    # ---------------- 对外主接口 ----------------
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行完整的「清洗 + 指标计算」流水线。

        Args:
            df: 原始行情（需含 open/high/low/close/vol 及各标记列）。

        Returns:
            pd.DataFrame: 清洗后并附加 MA/ATR/Z 列的行情；
                          若数据不合格（如次新股）则返回空 DataFrame。
        """
        df = self.clean(df)
        if df.empty:
            return df
        df = self.compute_indicators(df)
        return df

    # ---------------- 步骤 1：清洗 ----------------
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 README 2.2 的规则清洗行情数据。

        规则：
          1. 次新股过滤：总交易日不足 min_listed_days → 整只剔除。
          2. ST 期间：ST 标记为真的行整体剔除。
          3. 停牌日：收盘价前向填充，成交量 / 成交额置 0。
          4. 长期停牌：连续停牌超过 20 日的区间整段剔除。
          5. 异常值：单日涨跌幅超 ±11%（且非涨跌停）标记为异常并剔除。
        """
        df = df.copy().sort_values("trade_date").reset_index(drop=True)

        # 规则 1：次新股（上市/可用交易日不足）整只放弃。
        if len(df) < self.min_listed_days:
            return df.iloc[0:0]  # 返回空表

        # 规则 2：剔除 ST 期间的数据。
        if "is_st" in df.columns:
            df = df[~df["is_st"].astype(bool)].reset_index(drop=True)

        # 规则 4：标记并剔除「连续停牌 > 20 日」的长停区间。
        if "is_suspended" in df.columns:
            df = self._drop_long_suspensions(df, max_consecutive=20)

        # 规则 3：停牌日收盘价前向填充，量 / 额置 0。
        if "is_suspended" in df.columns:
            suspended = df["is_suspended"].astype(bool)
            df.loc[suspended, ["vol", "amount"]] = 0.0
            # 收盘价前向填充（停牌当天价格沿用上一交易日）。
            df["close"] = df["close"].ffill()
            # 停牌日缺失的 OHLC 也用收盘价补齐，避免后续计算出现 NaN。
            for col in ["open", "high", "low"]:
                df[col] = df[col].fillna(df["close"])

        # 规则 5：剔除异常涨跌幅（非涨跌停但单日涨跌超 ±11%）。
        df["pct_chg"] = df["close"].pct_change()
        is_limit = df.get("limit_up", False) | df.get("limit_down", False)
        abnormal = (df["pct_chg"].abs() > 0.11) & (~is_limit)
        df = df[~abnormal].reset_index(drop=True)
        df = df.drop(columns=["pct_chg"])

        return df

    def _drop_long_suspensions(self, df: pd.DataFrame, max_consecutive: int) -> pd.DataFrame:
        """剔除连续停牌天数超过阈值的区间。

        实现：对 is_suspended 做「连续段」分组，统计每段长度，
        删除长度 > max_consecutive 的停牌段。
        """
        suspended = df["is_suspended"].astype(bool).values
        # 给每个「连续相同值」的段落分配一个组号。
        group_id = (pd.Series(suspended) != pd.Series(suspended).shift()).cumsum()
        keep_mask = np.ones(len(df), dtype=bool)
        for _, idx in pd.Series(range(len(df))).groupby(group_id.values):
            rows = idx.values
            # 该段是停牌段且长度超限 → 标记删除。
            if suspended[rows[0]] and len(rows) > max_consecutive:
                keep_mask[rows] = False
        return df[keep_mask].reset_index(drop=True)

    # ---------------- 步骤 2：指标计算 ----------------
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 MA、ATR、Z 三个核心指标（README 2.3）。

        公式：
          MA  = close 的 ma_length 日简单移动平均（网格中轴）。
          TR  = max(high-low, |high-prev_close|, |low-prev_close|)。
          ATR = TR 的 atr_length 日移动平均（网格间距基准）。
          Z   = (close - MA) / ATR（标准化偏离度，OU 建模对象）。

        同时剔除 ATR 过小（接近 0）的行，避免 Z 计算时除零放大噪声。
        """
        df = df.copy()

        # 1) 移动平均（网格中轴）。
        df["MA"] = df["close"].rolling(self.ma_length).mean()

        # 2) 真实波幅 TR 与平均真实波幅 ATR。
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],                  # 当日高低差
                (df["high"] - prev_close).abs(),         # 高 - 昨收
                (df["low"] - prev_close).abs(),          # 低 - 昨收
            ],
            axis=1,
        ).max(axis=1)
        df["TR"] = tr
        df["ATR"] = df["TR"].rolling(self.atr_length).mean()

        # 3) 标准化偏离度 Z（OU 过程建模对象）。
        df["Z"] = (df["close"] - df["MA"]) / df["ATR"]

        # 4) 剔除 ATR 为 0 / 极小值的行（避免除零）。
        #    仅在存在有效 ATR 时计算分位数阈值，避免全 NaN 报错。
        valid_atr = df["ATR"].dropna()
        if not valid_atr.empty:
            atr_floor = valid_atr.quantile(0.01)
            df = df[(df["ATR"].notna()) & (df["ATR"] > max(atr_floor, 1e-8))]

        # 丢弃指标暖机期产生的 NaN 行（前 ma_length / atr_length 天）。
        df = df.dropna(subset=["MA", "ATR", "Z"]).reset_index(drop=True)
        return df
