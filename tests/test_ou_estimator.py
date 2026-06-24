# ====================================================================
# OU 参数估计单元测试 (test_ou_estimator.py)
# 对应 README 阶段 4「OU 参数估计与校验」
#
# 用「已知参数」的合成 OU 序列验证估计器能否还原参数，并覆盖
# README 4.2 列出的关键陷阱（b ∉ (0,1)、数据不足等）。
# ====================================================================

import unittest

import numpy as np

from src.screening.ou_estimator import (
    estimate_ou,
    kappa_stability,
    rolling_kappa,
)


def make_ou_series(n, b, a=0.0, sigma=1.0, seed=0):
    """构造已知 AR(1)/OU 参数的序列：Z_{t+1} = a + b·Z_t + N(0,σ)。"""
    rng = np.random.default_rng(seed)
    z = np.empty(n)
    z[0] = 0.0
    for t in range(1, n):
        z[t] = a + b * z[t - 1] + sigma * rng.standard_normal()
    return z


class TestEstimateOU(unittest.TestCase):
    """estimate_ou 的核心行为测试。"""

    def test_recovers_known_b(self):
        """对已知 b=0.8 的序列，估计的 b 应接近真值。"""
        z = make_ou_series(3000, b=0.8, seed=1)
        params = estimate_ou(z)
        self.assertTrue(params.valid)
        # 大样本下 OLS 应较准确（容差 0.05）。
        self.assertAlmostEqual(params.b, 0.8, delta=0.05)

    def test_kappa_and_half_life_positive(self):
        """均值回复序列的 κ 与半衰期应为正且有限。"""
        z = make_ou_series(3000, b=0.85, seed=2)
        params = estimate_ou(z)
        self.assertGreater(params.kappa, 0.0)
        self.assertGreater(params.half_life, 0.0)
        # κ = -ln(b)，半衰期 = ln2/κ，二者应自洽。
        expected_hl = np.log(2.0) / params.kappa
        self.assertAlmostEqual(params.half_life, expected_hl, places=6)

    def test_half_life_formula(self):
        """半衰期理论值 ln2 / (-ln b) 应与估计接近。"""
        b_true = 0.9
        z = make_ou_series(5000, b=b_true, seed=3)
        params = estimate_ou(z)
        theo_hl = np.log(2.0) / (-np.log(b_true))
        # 允许估计误差（半衰期对 b 较敏感）。
        self.assertAlmostEqual(params.half_life, theo_hl, delta=2.0)

    def test_invalid_when_b_out_of_range(self):
        """b ∉ (0,1)（发散序列）→ 估计标记为无效（README 4.2 陷阱1）。"""
        # 构造爆炸式 AR(1)：b=1.02 > 1，回归应估出 b≥1 并判无效。
        z = make_ou_series(2000, b=1.02, sigma=0.1, seed=4)
        params = estimate_ou(z)
        self.assertFalse(params.valid)

    def test_invalid_when_too_short(self):
        """样本过短（< 30）→ 无效。"""
        params = estimate_ou(np.arange(10.0))
        self.assertFalse(params.valid)

    def test_handles_nan(self):
        """序列含 NaN 时应自动剔除后再估计，不报错。"""
        z = make_ou_series(1000, b=0.8, seed=5)
        z[::50] = np.nan
        params = estimate_ou(z)
        # 仍应得到有效估计。
        self.assertTrue(params.valid)


class TestKappaStability(unittest.TestCase):
    """rolling_kappa 与 kappa_stability 的测试。"""

    def test_rolling_kappa_returns_list(self):
        """滚动估计应返回若干有效 κ。"""
        z = make_ou_series(1200, b=0.85, seed=6)
        kappas = rolling_kappa(z, window=250, step=50)
        self.assertIsInstance(kappas, list)
        self.assertGreater(len(kappas), 0)
        self.assertTrue(all(k > 0 for k in kappas))

    def test_stability_fields(self):
        """稳定性结果应包含约定字段且类型正确。"""
        z = make_ou_series(1200, b=0.85, seed=7)
        stab = kappa_stability(z, window=250, step=50)
        for key in ["kappa_cv", "recent_positive", "n_windows", "mean_kappa"]:
            self.assertIn(key, stab)
        self.assertIsInstance(stab["recent_positive"], bool)
        self.assertGreaterEqual(stab["kappa_cv"], 0.0)

    def test_stability_empty_on_short(self):
        """数据不足以形成任何窗口时，CV 为 inf、窗口数为 0。"""
        z = make_ou_series(100, b=0.85, seed=8)
        stab = kappa_stability(z, window=250, step=50)
        self.assertEqual(stab["n_windows"], 0)
        self.assertEqual(stab["kappa_cv"], float("inf"))


if __name__ == "__main__":
    unittest.main()
