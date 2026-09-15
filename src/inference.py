"""検定と区間推定。

クラスタ数が47（規模の偏りを考慮した実効クラスタ数は約21）と少なく、
漸近正規近似が効きにくい。そこで分布仮定に依存しない2つの方法を併用する。

1. 並べ替え検定（randomization inference）
   都道府県への配信/非配信の割当をランダムに振り直し、DID統計量の帰無分布を作る。
   「都道府県の分け方の偶然だけで、観測された差は作れるか」を直接評価する。

2. クラスタブートストラップ
   都道府県を単位に復元抽出し、増分件数の信頼区間を得る。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _did_from_arrays(
    pre: np.ndarray, post: np.ndarray, treat: np.ndarray, weighted: bool
) -> float:
    """県別の事前・介入期日平均からDID（対数）を計算する閉形式。

    PPML（県FE+日付FE）の beta と一致することを models.check_fe_redundancy で検証済み。
    並べ替え検定では1万回の再計算が要るため、回帰を回さずこの形で高速に評価する。
    """
    t, c = treat == 1, treat == 0
    if weighted:
        return float(
            (np.log(post[t].sum()) - np.log(pre[t].sum()))
            - (np.log(post[c].sum()) - np.log(pre[c].sum()))
        )
    return float(np.mean(np.log(post[t] / pre[t])) - np.mean(np.log(post[c] / pre[c])))


def permutation_test(
    summary: pd.DataFrame,
    weighted: bool = False,
    n_perm: int = config.N_PERMUTATIONS,
    seed: int = config.SEED,
) -> dict:
    """配信/非配信の割当を振り直して帰無分布を作り、両側p値を返す。

    配信県数（32）と非配信県数（15）は実際の設計どおりに固定する。
    """
    df = summary.dropna(subset=["dlog"])
    pre = df["pre_mean"].to_numpy(float)
    post = df["post_mean"].to_numpy(float)
    treat = df["treat"].to_numpy(int)

    observed = _did_from_arrays(pre, post, treat, weighted)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = _did_from_arrays(pre, post, rng.permutation(treat), weighted)

    n_extreme = int((np.abs(null) >= abs(observed)).sum())
    # 観測された割当自身を帰無分布の1つとして数える（p値が0にならないようにする）
    p_two_sided = (n_extreme + 1) / (n_perm + 1)
    return {
        "observed_beta": observed,
        "observed_lift_pct": (np.exp(observed) - 1) * 100,
        "n_permutations": n_perm,
        "n_as_extreme": n_extreme,
        "p_value": p_two_sided,
        "null": null,
        "weighted": weighted,
    }


def cluster_bootstrap(
    summary: pd.DataFrame,
    weighted: bool = False,
    n_boot: int = config.N_BOOTSTRAP,
    seed: int = config.SEED,
) -> dict:
    """都道府県を単位に復元抽出し、DID（対数）の信頼区間を得る。

    群ごとに層化して抽出し、配信32県・非配信15県の構成を保つ。
    """
    df = summary.dropna(subset=["dlog"])
    idx_t = np.flatnonzero(df["treat"].to_numpy() == 1)
    idx_c = np.flatnonzero(df["treat"].to_numpy() == 0)
    pre = df["pre_mean"].to_numpy(float)
    post = df["post_mean"].to_numpy(float)
    treat = df["treat"].to_numpy(int)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate(
            [rng.choice(idx_t, idx_t.size, replace=True),
             rng.choice(idx_c, idx_c.size, replace=True)]
        )
        draws[i] = _did_from_arrays(pre[pick], post[pick], treat[pick], weighted)

    lo, hi = np.percentile(draws, [100 * config.ALPHA / 2, 100 * (1 - config.ALPHA / 2)])
    point = _did_from_arrays(pre, post, treat, weighted)
    return {
        "beta": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "lift_pct": (np.exp(point) - 1) * 100,
        "lift_ci": ((np.exp(lo) - 1) * 100, (np.exp(hi) - 1) * 100),
        "draws": draws,
        "weighted": weighted,
    }


def separation_check(summary: pd.DataFrame) -> dict:
    """2群の県別変化率の分布が重なっているかを調べる。

    完全に分離していれば、並べ替え検定の結果を待たずとも
    「偶然では起こりにくい」ことが一目で伝わる。
    """
    df = summary.dropna(subset=["dlog"])
    t = df.loc[df.treat == 1, "ratio"]
    c = df.loc[df.treat == 0, "ratio"]
    return {
        "treat_min_pct": (t.min() - 1) * 100,
        "treat_max_pct": (t.max() - 1) * 100,
        "control_min_pct": (c.min() - 1) * 100,
        "control_max_pct": (c.max() - 1) * 100,
        "fully_separated": bool(t.min() > c.max()),
        "n_overlap": int(((t.values[:, None] <= c.values[None, :])).sum()),
    }


def placebo_time_test(
    panel: pd.DataFrame,
    metric: str = config.PRIMARY,
    window_days: int = 31,
    min_pre_days: int = 31,
) -> dict:
    """介入がない期間に「偽の配信期間」を置き、どの程度の見かけの効果が出るかを測る。

    県クラスタだけで標準誤差を出すと、ある週に両群を非対称に揺らす全国的なショック
    （群×時点の共通ショック）が誤差として数えられず、区間が過小評価される。
    事前期間だけで同じ推定を繰り返せば、その揺らぎの大きさを実測できる。
    """
    daily = (
        panel.pivot_table(index="date", columns="treat", values=metric, aggfunc="sum")
        .rename(columns={0: "control", 1: "treat"})
        .sort_index()
    )
    pre = daily.loc[: pd.Timestamp(config.PRE_END)]
    dates = pre.index

    rows = []
    for start_i in range(min_pre_days, len(dates) - window_days + 1):
        fake_post = dates[start_i : start_i + window_days]
        fake_pre = dates[:start_i]
        b = (
            np.log(pre.loc[fake_post, "treat"].mean()) - np.log(pre.loc[fake_pre, "treat"].mean())
        ) - (
            np.log(pre.loc[fake_post, "control"].mean()) - np.log(pre.loc[fake_pre, "control"].mean())
        )
        rows.append({"fake_post_start": fake_post[0], "beta": b, "lift_pct": (np.exp(b) - 1) * 100})

    placebo = pd.DataFrame(rows)
    real = (
        np.log(daily.loc[pd.Timestamp(config.POST_START):, "treat"].mean())
        - np.log(pre["treat"].mean())
    ) - (
        np.log(daily.loc[pd.Timestamp(config.POST_START):, "control"].mean())
        - np.log(pre["control"].mean())
    )
    sd = float(placebo["beta"].std(ddof=1))
    return {
        "placebo": placebo,
        "placebo_sd_pct": (np.exp(sd) - 1) * 100,
        "placebo_min_pct": float(placebo["lift_pct"].min()),
        "placebo_max_pct": float(placebo["lift_pct"].max()),
        "real_beta": float(real),
        "real_lift_pct": (np.exp(real) - 1) * 100,
        "z_vs_placebo": float(real / sd) if sd > 0 else np.nan,
        "n_windows": len(placebo),
    }


def headline_estimate(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    beta: float,
    metric: str = config.PRIMARY,
) -> dict:
    """報告に使う代表値を、最も保守的な誤差評価と組み合わせて返す。

    誤差には2つの源泉があり、片方だけでは区間が狭くなりすぎる。

      (a) どの県が配信対象になったか（設計上のばらつき）
          → 並べ替え検定／県クラスタで捉えられる。本データでは非常に小さい
      (b) ある時期に両群を非対称に揺らす全国的なショック
          → 県クラスタでは捉えられない。事前期間のプラセボ推定で実測する

    (b) は (a) より1桁大きい。したがって報告する信頼区間は (b) を基準に置く。
    p値は割当の無作為化に基づく並べ替え検定の結果を用いる。
    """
    placebo = placebo_time_test(panel, metric)
    perm = permutation_test(summary, weighted=True)

    sd_log = float(placebo["placebo"]["beta"].std(ddof=1))
    z = 1.959963985
    lo, hi = beta - z * sd_log, beta + z * sd_log

    return {
        "beta": beta,
        "lift_pct": (np.exp(beta) - 1) * 100,
        "ci_low_pct": (np.exp(lo) - 1) * 100,
        "ci_high_pct": (np.exp(hi) - 1) * 100,
        "ci_low_log": lo,
        "ci_high_log": hi,
        "se_source": "事前期間プラセボ（群×時点ショックを含む）",
        "se_log": sd_log,
        "p_value": perm["p_value"],
        "p_source": f"並べ替え検定 {perm['n_permutations']:,}回（割当の無作為化）",
        "placebo_range_pct": (placebo["placebo_min_pct"], placebo["placebo_max_pct"]),
        "z_vs_placebo": placebo["z_vs_placebo"],
    }
