# ====================================================================
# 标的筛选主逻辑 (screener.py)
# 对应 README 阶段 3「标的筛选系统」（3.1 筛选流程）
#
# 把前面各模块串成一条完整的筛选流水线：
#
#   候选池
#     → 流动性过滤（日均成交额 ≥ 1亿）
#     → 数据完整性过滤（上市 ≥ 250 日）
#     → 均值回复检验（ADF + KPSS + Hurst）
#     → OU 参数估计与稳定性检验（κ, HL, CV(κ)）
#     → 半衰期过滤（5 ≤ HL ≤ 20）
#     → 合格标的池
#     → 按 κ 稳定性（CV 升序）排序，取 Top N
#
# 输出：一份每只候选标的的检验明细表 + 最终合格标的列表。
# ====================================================================

import pandas as pd

from config.loader import CONFIG
from src.data.processor import DataProcessor
from src.screening import stat_tests
from src.screening.ou_estimator import estimate_ou, kappa_stability


class Screener:
    """标的筛选器。

    给定「代码 → 原始行情」字典，逐只标的执行全套筛选，
    返回明细表与合格标的列表。
    """

    def __init__(self):
        """从配置读取所有筛选阈值。"""
        scr = CONFIG["screening"]
        self.adf_threshold = scr["adf_pvalue"]       # ADF p 上界
        self.kpss_threshold = scr["kpss_pvalue"]     # KPSS p 下界
        self.hurst_upper = scr["hurst_upper"]        # Hurst 上界
        self.hl_min = scr["half_life_min"]           # 半衰期下界
        self.hl_max = scr["half_life_max"]           # 半衰期上界
        self.kappa_cv_max = scr["kappa_cv_max"]      # κ CV 上界
        self.min_turnover = scr["min_turnover"]      # 最小日均成交额
        self.max_active = CONFIG["risk"]["max_active_symbols"]  # 最终取 Top N
        self.processor = DataProcessor()

    # ---------------- 对外主接口 ----------------
    def screen(self, raw_data: dict) -> dict:
        """对一批标的执行完整筛选流程。

        Args:
            raw_data: {ts_code: 原始行情 DataFrame}。

        Returns:
            dict: {
                "detail": pd.DataFrame,   # 每只标的的逐项检验结果
                "passed": list[str],      # 最终合格并排序后的标的代码
            }
        """
        records = []
        for code, df in raw_data.items():
            records.append(self._evaluate_one(code, df))

        detail = pd.DataFrame(records)
        # 选出通过全部条件的标的，并按 κ 稳定性（CV 升序，越稳越靠前）排序。
        passed_df = detail[detail["passed"]].copy()
        passed_df = passed_df.sort_values("kappa_cv", ascending=True)
        passed = passed_df["ts_code"].head(self.max_active).tolist()

        return {"detail": detail, "passed": passed}

    # ---------------- 单只标的评估 ----------------
    def _evaluate_one(self, code: str, raw_df: pd.DataFrame) -> dict:
        """对单只标的执行全部过滤步骤，返回一行检验结果。

        采用「短路」策略：任一硬性条件不满足就提前返回，
        既符合 README 的漏斗式筛选，也避免无谓计算。
        """
        # 结果模板：默认全部 NaN / False，逐步填充。
        result = {
            "ts_code": code,
            "avg_turnover": float("nan"),
            "n_days": 0,
            "adf_pvalue": float("nan"),
            "kpss_pvalue": float("nan"),
            "hurst": float("nan"),
            "kappa": float("nan"),
            "half_life": float("nan"),
            "kappa_cv": float("inf"),
            "recent_positive": False,
            "passed": False,
            "reject_reason": "",
        }

        # 步骤 0：清洗 + 指标计算。
        df = self.processor.process(raw_df)
        result["n_days"] = len(df)
        if df.empty:
            result["reject_reason"] = "数据不足/次新股"
            return result

        # 步骤 1：流动性过滤（日均成交额 ≥ 阈值）。
        #   Tushare amount 单位为千元，这里换算为元后比较。
        avg_turnover = df["amount"].mean() * 1000.0
        result["avg_turnover"] = avg_turnover
        if avg_turnover < self.min_turnover:
            result["reject_reason"] = "流动性不足"
            return result

        # 步骤 2：均值回复检验（ADF + KPSS + Hurst），作用于 Z 序列。
        z = df["Z"].values
        tests = stat_tests.run_all_tests(z)
        result["adf_pvalue"] = tests["adf_pvalue"]
        result["kpss_pvalue"] = tests["kpss_pvalue"]
        result["hurst"] = tests["hurst"]

        adf_ok = tests["adf_pvalue"] < self.adf_threshold     # 平稳
        # KPSS 的 p-value 被 statsmodels 截断在 [0.01, 0.10]：当真实 p ≥ 0.10
        # 时统一返回 0.10。因此「不拒绝平稳」对应 p ≥ 阈值（用 >= 而非 >）。
        kpss_ok = tests["kpss_pvalue"] >= self.kpss_threshold  # 平稳
        hurst_ok = tests["hurst"] < self.hurst_upper          # 均值回复
        if not (adf_ok and kpss_ok and hurst_ok):
            result["reject_reason"] = "均值回复检验未通过"
            return result

        # 步骤 3：OU 参数估计（全样本）+ 半衰期过滤。
        ou = estimate_ou(z)
        if not ou.valid:
            result["reject_reason"] = "OU 估计无效"
            return result
        result["kappa"] = ou.kappa
        result["half_life"] = ou.half_life
        if not (self.hl_min <= ou.half_life <= self.hl_max):
            result["reject_reason"] = "半衰期超范围"
            return result

        # 步骤 4：κ 稳定性检验（滚动窗口 CV + 最近窗口为正）。
        stab = kappa_stability(z)
        result["kappa_cv"] = stab["kappa_cv"]
        result["recent_positive"] = stab["recent_positive"]
        if stab["kappa_cv"] >= self.kappa_cv_max:
            result["reject_reason"] = "κ 不稳定 (CV 过大)"
            return result
        if not stab["recent_positive"]:
            result["reject_reason"] = "最近窗口 κ 非正"
            return result

        # 全部通过。
        result["passed"] = True
        result["reject_reason"] = "通过"
        return result
