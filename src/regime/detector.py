# ====================================================================
# Regime 检测模块 (detector.py)
# 对应 README 阶段 6「Regime 检测与策略开关」
#
# 网格交易本质是逆势操作，一旦市场进入强趋势（κ 失效）就会越买越套。
# 本模块在「κ 失效之前」给出降仓 / 暂停信号，是策略的安全开关。
#
# 三重检测（README 6.2），每日对 Z / 价格序列计算：
#   1. 滚动 Hurst 指数（窗口 60）   —— 判断均值回复 / 趋势。
#   2. 滚动 ADF 检验（窗口 120）    —— 判断平稳性是否仍显著。
#   3. 波动率爆发（ATR_5 / ATR_60） —— 判断是否出现异常放量波动。
#
# 综合决策（README 6.3）：对三者取「最严格」信号，并用连续 5 日全绿
# 作为从暂停状态恢复的条件，避免频繁切换。
# ====================================================================

import numpy as np

from config.loader import CONFIG
from src.screening.stat_tests import adf_test, hurst_exponent

# 三档 regime 状态（数值越大越严格，便于取 max）。
GREEN = 0    # 🟢 正常运行（满仓网格）
YELLOW = 1   # 🟡 仓位减半（不开新仓，保留已有持仓）
RED = 2      # 🔴 暂停网格（不开新仓，已有持仓设紧急止损）

# 各状态对应的「目标仓位系数」：绿=满仓，黄=半仓，红=不新增。
POSITION_SCALE = {GREEN: 1.0, YELLOW: 0.5, RED: 0.0}


class RegimeDetector:
    """单只标的的 regime（市场状态）检测器。

    维护一个「连续正常天数」计数器以实现恢复逻辑：处于暂停（RED/YELLOW）
    后，需连续 recover_days 个交易日全绿才恢复到正常运行。
    """

    def __init__(self):
        """从配置读取三重检测的窗口与阈值。"""
        reg = CONFIG["regime"]
        self.hurst_window = reg["hurst_window"]
        self.hurst_trend = reg["hurst_trend_threshold"]   # > 此值 → 趋势(红)
        self.hurst_mr = reg["hurst_mr_threshold"]         # < 此值 → 均值回复(绿)
        self.adf_window = reg["adf_window"]
        self.adf_reject = reg["adf_reject_threshold"]     # > 此值 → 平稳消失(红)
        self.vol_spike = reg["vol_spike_ratio"]           # ≥ 此值 → 波动爆发(红)
        self.vol_warn = reg["vol_warn_ratio"]             # ≥ 此值 → 波动放大(黄)
        self.recover_days = reg["recover_days"]

        # 状态机：当前是否处于「暂停/降级」中，以及连续全绿天数。
        self.suspended = False
        self.consecutive_green = 0

    # ---------------- 三个独立检测器 ----------------
    def detect_hurst(self, z_window) -> int:
        """检测器 1：滚动 Hurst（README 6.2）。

        H < mr → 绿；mr ≤ H ≤ trend → 黄；H > trend → 红。
        数据不足时返回绿（不因样本不足而误触发）。
        """
        z = np.asarray(z_window, dtype=float)
        if len(z) < self.hurst_window:
            return GREEN
        h = hurst_exponent(z[-self.hurst_window:])
        if h > self.hurst_trend:
            return RED
        if h >= self.hurst_mr:
            return YELLOW
        return GREEN

    def detect_adf(self, z_window) -> int:
        """检测器 2：滚动 ADF（README 6.2）。

        p < 0.05 → 绿；0.05 ≤ p ≤ adf_reject → 黄；p > adf_reject → 红。
        数据不足时返回绿。
        """
        z = np.asarray(z_window, dtype=float)
        if len(z) < self.adf_window:
            return GREEN
        p = adf_test(z[-self.adf_window:])
        if p > self.adf_reject:
            return RED
        if p >= 0.05:
            return YELLOW
        return GREEN

    def detect_volatility(self, atr_fast: float, atr_slow: float) -> int:
        """检测器 3：波动率爆发（README 6.2）。

        ratio = ATR_5 / ATR_60：
          < warn → 绿；warn ≤ ratio < spike → 黄；≥ spike → 红。

        Args:
            atr_fast: 短期 ATR（如 5 日）。
            atr_slow: 长期 ATR（如 60 日）。
        """
        if not (atr_slow > 0):
            return GREEN
        ratio = atr_fast / atr_slow
        if ratio >= self.vol_spike:
            return RED
        if ratio >= self.vol_warn:
            return YELLOW
        return GREEN

    # ---------------- 综合决策 ----------------
    def decide(self, z_window, atr_fast: float, atr_slow: float) -> dict:
        """综合三重检测，给出当日 regime 与目标仓位系数（README 6.3）。

        步骤：
          1. 三个检测器各出一档信号，取最严格（max）作为「原始信号」。
          2. 应用恢复逻辑：
             - 原始信号为绿：连续全绿天数 +1；累计达 recover_days 则解除暂停。
             - 原始信号非绿：清零计数；若为红则进入暂停状态。
          3. 暂停状态未解除前，即使当日转绿也维持暂停（仅当满足连续条件才恢复）。

        Args:
            z_window: 截至当日的 Z 序列（用于 Hurst / ADF 滚动窗口）。
            atr_fast / atr_slow: 短期 / 长期 ATR（用于波动率检测）。

        Returns:
            dict: {
                "regime": int,            # 最终状态（GREEN/YELLOW/RED）
                "position_scale": float,  # 目标仓位系数（1.0/0.5/0.0）
                "raw_signal": int,        # 未经恢复逻辑的当日原始信号
                "hurst_signal": int,
                "adf_signal": int,
                "vol_signal": int,
            }
        """
        h_sig = self.detect_hurst(z_window)
        a_sig = self.detect_adf(z_window)
        v_sig = self.detect_volatility(atr_fast, atr_slow)
        raw = max(h_sig, a_sig, v_sig)  # 取最严格信号

        if raw == GREEN:
            self.consecutive_green += 1
            # 连续足够多天全绿 → 解除暂停。
            if self.suspended and self.consecutive_green >= self.recover_days:
                self.suspended = False
        else:
            # 任何非绿信号都打断连续全绿计数。
            self.consecutive_green = 0
            if raw == RED:
                self.suspended = True

        # 最终状态：仍处暂停中则至少为红（强制不开新仓）；否则用原始信号。
        final = RED if self.suspended else raw
        return {
            "regime": final,
            "position_scale": POSITION_SCALE[final],
            "raw_signal": raw,
            "hurst_signal": h_sig,
            "adf_signal": a_sig,
            "vol_signal": v_sig,
        }
