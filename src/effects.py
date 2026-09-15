"""リフト率（対数係数）を、報告で使う「増分件数」に変換する。

役員会で必要なのは「+2.8%」ではなく「+◯◯件」。
    反実仮想 = 実績 / exp(beta)
    増分     = 実績 - 反実仮想 = 実績 * (1 - exp(-beta))
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def incremental_counts(
    panel: pd.DataFrame,
    beta: float,
    ci: tuple[float, float] | None = None,
    metric: str = config.PRIMARY,
) -> dict:
    """配信群の介入期実績から増分件数を逆算する。"""
    actual = float(
        panel.loc[(panel.treat == 1) & (panel.post == 1), metric].sum()
    )
    counterfactual = actual / np.exp(beta)
    result = {
        "actual": actual,
        "counterfactual": counterfactual,
        "incremental": actual - counterfactual,
        "lift_pct": (np.exp(beta) - 1) * 100,
    }
    if ci is not None:
        lo, hi = ci
        result["incremental_low"] = actual * (1 - np.exp(-lo))
        result["incremental_high"] = actual * (1 - np.exp(-hi))
    return result


def extrapolate_nationwide(
    panel: pd.DataFrame, beta: float, metric: str = config.PRIMARY
) -> dict:
    """全国配信していた場合の月間増分の見込み。

    「非配信だった15県にも同じ率で効く」という強い仮定に立つ。
    地域特性が異なれば成り立たないため、断定せず幅で示すこと。
    """
    post = panel[panel.post == 1]
    treat_actual = float(post.loc[post.treat == 1, metric].sum())
    control_actual = float(post.loc[post.treat == 0, metric].sum())

    treat_inc = treat_actual * (1 - np.exp(-beta))
    # 非配信群は「効果を受けていない実績」なので、こちらは実績に率を掛けて増分を出す
    control_inc = control_actual * (np.exp(beta) - 1)
    return {
        "treated_incremental": treat_inc,
        "untreated_expected_incremental": control_inc,
        "nationwide_incremental": treat_inc + control_inc,
        "assumption": "非配信15県にも配信32県と同率の効果が生じると仮定",
    }


def cpa_table(
    incremental: float, budgets: list[float] | None = None
) -> pd.DataFrame:
    """消化金額が未受領のため、増分CPAの感度分析表を作る。

    費用が判明し次第、担当者がこの表から自分で読み取れるようにする。
        増分CPA = 消化金額 / 増分エントリー数
    """
    budgets = budgets or [1e6, 3e6, 5e6, 1e7, 2e7, 3e7, 5e7]
    return pd.DataFrame(
        {
            "消化金額（円）": [f"{b:,.0f}" for b in budgets],
            "増分エントリー数": [round(incremental)] * len(budgets),
            "増分CPA（円）": [round(b / incremental) for b in budgets],
        }
    )
