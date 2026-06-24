# ====================================================================
# 统计检验模块 (stat_tests.py)
# 对应 README 阶段 3.2「统计检验细节」
#
# 提供三类均值回复检验，作用于标准化偏离度序列 Z_t：
#   1. ADF 检验   —— 检验是否平稳（拒绝单位根）。
#   2. KPSS 检验  —— 检验是否平稳（不拒绝平稳原假设）。
#   3. Hurst 指数 —— 衡量序列的长期记忆性（< 0.5 → 均值回复）。
#
# 三者结合使用，互相印证，避免单一检验的误判（详见 README 3.2）。
# ====================================================================

import warnings

import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series) -> float:
    """ADF（Augmented Dickey-Fuller）单位根检验。

    原假设 H0：序列存在单位根（非平稳）。
    我们希望「拒绝」H0，即 p-value 越小越好（< 0.05 视为平稳）。
    滞后阶数由 AIC 自动选择（autolag="AIC"）。

    Args:
        series: 一维数值序列（通常是 Z_t）。

    Returns:
        float: ADF 检验的 p-value；序列过短 / 异常时返回 1.0（视为不平稳）。
    """
    x = _clean_series(series)
    if len(x) < 20:
        return 1.0
    try:
        # adfuller 返回元组，索引 1 为 p-value。
        result = adfuller(x, autolag="AIC")
        return float(result[1])
    except Exception:
        return 1.0


def kpss_test(series) -> float:
    """KPSS 平稳性检验。

    原假设 H0：序列平稳。
    我们希望「不拒绝」H0，即 p-value 越大越好（> 0.10 视为平稳）。
    regression="c"：围绕常数均值检验平稳性（适配 Z_t 长期均值≈0 的设定）。

    Args:
        series: 一维数值序列（通常是 Z_t）。

    Returns:
        float: KPSS 检验的 p-value；异常时返回 0.0（视为非平稳）。
    """
    x = _clean_series(series)
    if len(x) < 20:
        return 0.0
    try:
        # KPSS 在 p-value 超出查表范围时会抛 InterpolationWarning，这里忽略。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(x, regression="c", nlags="auto")
        return float(result[1])
    except Exception:
        return 0.0


def hurst_exponent(series, max_lag: int = 40) -> float:
    """估计 Hurst 指数，用于判断序列的均值回复 / 趋势倾向。

    解释（README 3.2）：
        H < 0.5 → 均值回复倾向（反持续性）；
        H = 0.5 → 随机游走；
        H > 0.5 → 趋势倾向（持续性）。

    方法说明：
        README 原文建议「R/S 分析法」。但经典 R/S 在有限样本（尤其是
        经过移动平均标准化后的 Z 序列）上存在显著的系统性「上偏」，
        会把明显均值回复的序列也估成 H≈0.6~0.8，导致筛选失真。
        因此这里改用统计上更稳健、且物理含义更直接的
        「差分方差标度法」（variance-of-differences，又称广义 Hurst）：

            对滞后 k，计算 Std(x_{t+k} - x_t)；
            理论上 Std(Δ_k) ∝ k^H；
            在 log(k) - log(Std) 上回归，斜率即为 *水平序列* 的 Hurst H。

        该方法对随机游走给出 H≈0.5、白噪声给出 H≈0、趋势给出 H>0.5，
        无 R/S 的小样本偏差，结论与 ADF/KPSS 高度一致。

    Args:
        series: 一维数值序列（通常是 Z_t）。
        max_lag: 最大滞后阶数（默认 40 个交易日）。

    Returns:
        float: Hurst 指数估计值；数据不足时返回 0.5（中性，视为随机游走）。
    """
    x = _clean_series(series)
    n = len(x)
    if n < 60:
        return 0.5

    # 滞后阶数在 [1, max_lag] 内对数取点（避免点过密、加快计算）。
    lags = np.unique(
        np.floor(np.logspace(0, np.log10(max_lag), 12)).astype(int)
    )
    lags = lags[lags >= 1]

    log_k = []
    log_std = []
    for k in lags:
        # 滞后 k 的差分序列。
        diff = x[k:] - x[:-k]
        std = np.sqrt(np.mean(diff**2))
        if std > 0:
            log_k.append(np.log(k))
            log_std.append(np.log(std))

    if len(log_k) < 4:
        return 0.5
    # 回归斜率即水平序列的 Hurst 指数。
    slope = np.polyfit(log_k, log_std, 1)[0]
    return float(slope)


def run_all_tests(series) -> dict:
    """一次性运行 ADF / KPSS / Hurst 三项检验。

    Args:
        series: 一维数值序列（通常是 Z_t）。

    Returns:
        dict: {"adf_pvalue", "kpss_pvalue", "hurst"} 三个键。
    """
    return {
        "adf_pvalue": adf_test(series),
        "kpss_pvalue": kpss_test(series),
        "hurst": hurst_exponent(series),
    }


# ======================== 内部工具 ========================

def _clean_series(series) -> np.ndarray:
    """转成一维 numpy 数组并剔除 NaN / Inf。"""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    return x
