# ====================================================================
# OU 参数估计模块 (ou_estimator.py)
# 对应 README 阶段 4「OU 参数估计与校验」及 3.3「稳定性检验」
#
# 核心思想：
#   把标准化偏离度 Z_t 建模为离散 OU 过程，通过一阶自回归
#   (AR(1)) 的 OLS 回归估计回复速度 κ、长期均值 θ、波动率 σ，
#   并据此推导半衰期 HL 与稳态标准差 s。
#
#   离散 OU 回归模型：  Z_{t+1} = a + b * Z_t + ε_t
#   参数转换（Δt = 1）：
#       κ      = -ln(b)
#       θ      = a / (1 - b)
#       σ_ou   = std(ε) * sqrt(-2 ln(b) / (1 - b²))
#       HL     = ln(2) / κ              （半衰期）
#       s      = σ_ou / sqrt(2κ)        （稳态标准差）
# ====================================================================

import numpy as np

from config.loader import CONFIG


class OUParams:
    """OU 参数估计结果的容器。

    Attributes:
        b: AR(1) 回归系数。
        a: AR(1) 截距。
        kappa: 回复速度 κ。
        theta: 长期均值 θ。
        sigma_ou: OU 过程波动率 σ_ou。
        half_life: 半衰期 HL（交易日）。
        steady_std: 稳态标准差 s。
        valid: 本次估计是否有效（b ∈ (0,1) 等条件）。
    """

    def __init__(self, b, a, kappa, theta, sigma_ou, half_life, steady_std, valid):
        self.b = b
        self.a = a
        self.kappa = kappa
        self.theta = theta
        self.sigma_ou = sigma_ou
        self.half_life = half_life
        self.steady_std = steady_std
        self.valid = valid

    def as_dict(self) -> dict:
        """转成普通字典，便于打印 / 存表。"""
        return {
            "b": self.b,
            "a": self.a,
            "kappa": self.kappa,
            "theta": self.theta,
            "sigma_ou": self.sigma_ou,
            "half_life": self.half_life,
            "steady_std": self.steady_std,
            "valid": self.valid,
        }


def estimate_ou(z_series) -> OUParams:
    """对单段 Z_t 序列做一次 OU 参数估计（OLS 回归）。

    步骤：
      1. 构造回归对：X = Z_t（去掉最后一个），Y = Z_{t+1}（去掉第一个）。
      2. 用最小二乘求 b, a。
      3. 按公式换算 κ, θ, σ_ou, HL, s。
      4. 校验有效性：b ∈ (0, 1)（README 4.2 陷阱1）。

    Args:
        z_series: 一维 Z_t 序列。

    Returns:
        OUParams: 估计结果；数据不足或非法时 valid=False。
    """
    z = np.asarray(z_series, dtype=float)
    z = z[np.isfinite(z)]
    # 至少需要 ~30 个点回归才有意义。
    if len(z) < 30:
        return _invalid_params()

    # 构造 AR(1) 回归的自变量 X 与因变量 Y。
    x = z[:-1]
    y = z[1:]

    # OLS：对 [1, x] 做最小二乘，得到截距 a 与斜率 b。
    # np.polyfit(x, y, 1) 返回 [斜率, 截距]。
    b, a = np.polyfit(x, y, 1)

    # README 4.2 陷阱1：b ∉ (0,1) → OU 模型不适用，本次估计无效。
    if not (0.0 < b < 1.0):
        return _invalid_params(b=b, a=a)

    # 参数换算。
    kappa = -np.log(b)                 # 回复速度
    theta = a / (1.0 - b)              # 长期均值
    residuals = y - (a + b * x)        # 回归残差 ε_t
    sigma_eps = residuals.std(ddof=2)  # 残差标准差（自由度修正）
    # σ_ou = σ_ε * sqrt(-2 ln(b) / (1 - b²))
    sigma_ou = sigma_eps * np.sqrt(-2.0 * np.log(b) / (1.0 - b**2))
    half_life = np.log(2.0) / kappa    # 半衰期
    steady_std = sigma_ou / np.sqrt(2.0 * kappa)  # 稳态标准差

    return OUParams(
        b=float(b),
        a=float(a),
        kappa=float(kappa),
        theta=float(theta),
        sigma_ou=float(sigma_ou),
        half_life=float(half_life),
        steady_std=float(steady_std),
        valid=True,
    )


def rolling_kappa(z_series, window: int = None, step: int = None):
    """滚动窗口估计 κ 序列（README 3.3 稳定性检验的基础）。

    在 Z_t 上以固定窗口、固定步长滑动，每个窗口独立估计一个 κ，
    得到一组 {κ_1, κ_2, ...}，用于评估回复速度是否随时间稳定。

    Args:
        z_series: 一维 Z_t 序列。
        window: 滚动窗口长度，默认读配置 ou_estimation.window。
        step: 滚动步长，默认读配置 ou_estimation.step。

    Returns:
        list[float]: 各窗口估计出的有效 κ 值（无效窗口被跳过）。
    """
    ou_cfg = CONFIG["ou_estimation"]
    window = window or ou_cfg["window"]
    step = step or ou_cfg["step"]

    z = np.asarray(z_series, dtype=float)
    kappas = []
    # 从头到尾滑动窗口。
    for start in range(0, len(z) - window + 1, step):
        seg = z[start:start + window]
        params = estimate_ou(seg)
        if params.valid:
            kappas.append(params.kappa)
    return kappas


def kappa_stability(z_series, window: int = None, step: int = None) -> dict:
    """评估 κ 的稳定性（README 3.3）。

    计算：
      - CV(κ) = std(κ) / mean(κ)（变异系数，越小越稳定）；
      - 最近 2 个窗口的 κ 是否都为正（当前是否处于均值回复状态）。

    Args:
        z_series: 一维 Z_t 序列。
        window / step: 同 :func:`rolling_kappa`。

    Returns:
        dict: {
            "kappa_cv": float,            # κ 变异系数（无有效估计时为 inf）
            "recent_positive": bool,      # 最近 2 窗 κ 是否都为正
            "n_windows": int,             # 有效窗口数
            "mean_kappa": float,          # κ 均值
        }
    """
    kappas = rolling_kappa(z_series, window, step)
    if len(kappas) == 0:
        return {
            "kappa_cv": float("inf"),
            "recent_positive": False,
            "n_windows": 0,
            "mean_kappa": float("nan"),
        }

    arr = np.asarray(kappas, dtype=float)
    mean_k = arr.mean()
    std_k = arr.std(ddof=1) if len(arr) > 1 else 0.0
    # 变异系数；均值≈0 时设为 inf（视为不稳定）。
    cv = std_k / mean_k if abs(mean_k) > 1e-9 else float("inf")
    # 最近 2 个窗口的 κ 是否都为正。
    recent = arr[-2:]
    recent_positive = bool(np.all(recent > 0))

    return {
        "kappa_cv": float(cv),
        "recent_positive": recent_positive,
        "n_windows": int(len(arr)),
        "mean_kappa": float(mean_k),
    }


# ======================== 内部工具 ========================

def _invalid_params(b=float("nan"), a=float("nan")) -> OUParams:
    """构造一个标记为无效的 OUParams（其余字段为 NaN）。"""
    nan = float("nan")
    return OUParams(
        b=b, a=a, kappa=nan, theta=nan, sigma_ou=nan,
        half_life=nan, steady_std=nan, valid=False,
    )
