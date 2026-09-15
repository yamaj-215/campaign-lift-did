"""DIDを統計モデルとして推定する。

主分析は PPML（ポアソン疑似最尤）。
    E[Y_it] = exp(alpha_i + gamma_t + beta * treat_i * post_t)
被説明変数がカウントで、県の規模が30倍の幅を持つため乗法モデルを採る。
PPMLはデータが実際にポアソン分布に従っているかに関わらず beta を一致推定するため、
分布の仮定を誤るリスクを負わない（標準誤差は県クラスタロバストで別途担保する）。

beta はリフト率の対数。exp(beta) - 1 が「配信によって何%増えたか」。

県FE・日付FEは、バランスドパネルでは点推定に対して冗長である
（県ごとの対数変化率を取る操作と数学的に同じ）。それでもモデルとして推定するのは
  - 標準誤差・信頼区間・検定を統一した枠組みで出すため
  - イベントスタディやセグメント別へ拡張するため
  - 重み付け（県1票か規模加重か）を明示的に選ぶため
であり、この冗長性自体を assert で検証している（check_fe_redundancy）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config


@dataclass
class DidResult:
    """DID推定の結果。リフト率は対数係数から変換して保持する。"""

    name: str
    beta: float
    se: float
    pvalue: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_clusters: int | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def lift(self) -> float:
        """リフト率（%）。exp(beta) - 1。"""
        return (np.exp(self.beta) - 1.0) * 100.0

    @property
    def lift_ci(self) -> tuple[float, float]:
        return (
            (np.exp(self.ci_low) - 1.0) * 100.0,
            (np.exp(self.ci_high) - 1.0) * 100.0,
        )

    def as_row(self) -> dict:
        lo, hi = self.lift_ci
        return {
            "モデル": self.name,
            "beta(log)": self.beta,
            "SE": self.se,
            "リフト率%": self.lift,
            "CI下限%": lo,
            "CI上限%": hi,
            "p値": self.pvalue,
            "観測数": self.n_obs,
            "クラスタ数": self.n_clusters,
            "備考": self.note,
        }


def _from_glm(name: str, res, term: str = "did", note: str = "") -> DidResult:
    ci = res.conf_int(alpha=config.ALPHA).loc[term]
    groups = getattr(res.model.data, "cluster_groups", None)
    n_clusters = None if groups is None else int(pd.Series(groups).nunique())
    return DidResult(
        name=name,
        beta=float(res.params[term]),
        se=float(res.bse[term]),
        pvalue=float(res.pvalues[term]),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        n_obs=int(res.nobs),
        n_clusters=n_clusters,
        note=note,
    )


def _cluster_kwds(panel: pd.DataFrame) -> dict:
    """県クラスタロバスト標準誤差の指定。

    同じ県の日次データは独立ではない（同一の配信判断の結果）。
    行数を自由度として扱うとp値を桁違いに過小評価するため、必ずクラスタを指定する。
    """
    return {"cov_type": "cluster", "cov_kwds": {"groups": panel[config.CLUSTER].values}}


# ---------------------------------------------------------------- 主分析

def fit_ppml(panel: pd.DataFrame, metric: str = config.PRIMARY) -> DidResult:
    """主分析：PPML（県FE + 日付FE + did）、県クラスタロバストSE。"""
    res = smf.glm(
        f"{metric} ~ did + C(pref) + C(date)",
        data=panel,
        family=sm.families.Poisson(),
    ).fit(**_cluster_kwds(panel))
    out = _from_glm("PPML（県FE+日付FE）", res, note="主分析")
    out.n_clusters = panel[config.CLUSTER].nunique()
    out.extra["result"] = res
    return out


# ---------------------------------------------------------------- 頑健性

def fit_ppml_no_datefe(panel: pd.DataFrame, metric: str = config.PRIMARY) -> DidResult:
    """日付FEを外したPPML。バランスドパネルでは主分析と一致するはず。"""
    res = smf.glm(
        f"{metric} ~ did + post + C(pref)",
        data=panel,
        family=sm.families.Poisson(),
    ).fit(**_cluster_kwds(panel))
    out = _from_glm("PPML（県FEのみ）", res, note="日付FEの冗長性確認")
    out.n_clusters = panel[config.CLUSTER].nunique()
    return out


def dispersion_diagnostics(panel: pd.DataFrame, metric: str = config.PRIMARY) -> dict:
    """固定効果を入れた後に過分散が残っているかを調べる。

    生データの分散/平均は県の規模差を反映して大きく出るが、それは県FEが吸収する分。
    モデル選択の根拠になるのはFE適用後の残差分散であり、こちらを見る必要がある。
    alpha は Cameron-Trivedi の補助回帰による NB2 の分散パラメータ推定値
    （((y-mu)^2 - y) / mu = alpha * mu）。alpha <= 0 なら過分散はない。
    """
    pois = smf.glm(
        f"{metric} ~ did + C(pref) + C(date)", data=panel, family=sm.families.Poisson()
    ).fit()
    mu = pois.fittedvalues.to_numpy(float)
    y = panel[metric].to_numpy(float)

    pearson = float(((y - mu) ** 2 / mu).sum() / pois.df_resid)
    aux_y = ((y - mu) ** 2 - y) / mu
    alpha = float(sm.OLS(aux_y, mu).fit().params[0])

    raw_ratio = float(y.var(ddof=1) / y.mean())
    return {
        "raw_var_mean_ratio": raw_ratio,
        "pearson_dispersion": pearson,
        "nb2_alpha": alpha,
        "overdispersed": pearson > 1.0,
    }


def fit_negbin(panel: pd.DataFrame, metric: str = config.PRIMARY,
               alpha: float | None = None) -> DidResult:
    """負の二項回帰。分布仮定を変えても結論が動かないことの確認。

    alpha を省略した場合は Cameron-Trivedi の補助回帰で推定する。
    決め打ちの alpha（例 1.0）は実態と乖離して大きな重み付けの歪みを生むため使わない。
    """
    if alpha is None:
        alpha = dispersion_diagnostics(panel, metric)["nb2_alpha"]
    alpha = max(alpha, 1e-8)  # GLMは正の alpha を要求する。0近傍ならポアソンに一致する

    res = smf.glm(
        f"{metric} ~ did + C(pref) + C(date)",
        data=panel,
        family=sm.families.NegativeBinomial(alpha=alpha),
    ).fit(**_cluster_kwds(panel))
    out = _from_glm(
        "負の二項（県FE+日付FE）", res, note=f"分布仮定の頑健性 alpha={alpha:.2e}"
    )
    out.n_clusters = panel[config.CLUSTER].nunique()
    out.extra["alpha"] = alpha
    return out


def fit_ols_log(panel: pd.DataFrame, metric: str = config.PRIMARY) -> DidResult:
    """log(Y)のOLS（二元固定効果）。0を含むと使えないため本データ限定の参考値。"""
    df = panel.copy()
    if (df[metric] <= 0).any():
        raise ValueError("0以下の値があるため log-OLS は使えない。PPMLを使うこと")
    df["log_y"] = np.log(df[metric])
    res = smf.ols("log_y ~ did + C(pref) + C(date)", data=df).fit(**_cluster_kwds(df))
    out = _from_glm("log-OLS（県FE+日付FE）", res, note="関数形の頑健性")
    out.n_clusters = df[config.CLUSTER].nunique()
    return out


def fit_prefecture_level(
    summary: pd.DataFrame, weighted: bool = False
) -> DidResult:
    """県単位モデル：dlog_i = a + beta * treat_i + e_i（n=47）。

    推論単位（都道府県）と分析単位が一致するため擬似反復の危険がない。
    weighted=True で事前期の規模を重みにすると、PPML（規模加重）と整合する。
    weighted=False は県1票。両者が近ければ、効果が県の規模によらず一様である証拠。
    """
    df = summary.dropna(subset=["dlog"]).copy()
    if weighted:
        res = smf.wls("dlog ~ treat", data=df, weights=df["pre_total"]).fit(cov_type="HC3")
        name, note = "県単位WLS（規模加重）", "n=47。規模加重"
    else:
        res = smf.ols("dlog ~ treat", data=df).fit(cov_type="HC3")
        name, note = "県単位OLS（県1票）", "n=47。不均一分散ロバスト(HC3)"

    ci = res.conf_int(alpha=config.ALPHA).loc["treat"]
    return DidResult(
        name=name,
        beta=float(res.params["treat"]),
        se=float(res.bse["treat"]),
        pvalue=float(res.pvalues["treat"]),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        n_obs=int(res.nobs),
        n_clusters=int(res.nobs),
        note=note,
    )


# ---------------------------------------------------------------- イベントスタディ

def add_event_time(panel: pd.DataFrame) -> pd.DataFrame:
    """配信開始日を基点に7日刻みの相対週（event time）を付ける。

    カレンダー週（月曜起点など）で切ると、8/1をまたぐ週に事前期と介入期の日が混在し、
    その週の係数が両者の平均になってしまう。基点を配信開始日に置いて混在を避ける。
    week = 0 が 8/1〜8/7、week = -1 が 7/25〜7/31。
    """
    df = panel.copy()
    offset = (df["date"] - pd.Timestamp(config.POST_START)).dt.days
    df["event_week"] = np.floor(offset / 7).astype(int)
    return df


def fit_event_study(panel: pd.DataFrame, metric: str = config.PRIMARY) -> pd.DataFrame:
    """相対週ごとの効果 beta_k を推定する（基準＝配信直前の週 k=-1）。

    介入前の beta_k が0周辺に並べば平行トレンドの根拠になり、
    介入と同時に立ち上がれば「効果が出たタイミングが配信開始と一致する」証拠になる。

    交互作用ダミーは式に頼らず明示的に作る。patsyの交互作用記法は
    主効果と重複して設計行列がランク落ちしやすいため。
    """
    df = add_event_time(panel)
    weeks = sorted(df["event_week"].unique())
    base_week = -1  # 配信直前の週を基準に置く
    if base_week not in weeks:
        raise ValueError("基準週（k=-1）がデータに存在しない")

    term_of: dict[int, str] = {}
    for k in weeks:
        if k == base_week:
            continue
        col = f"ev_{'m' if k < 0 else 'p'}{abs(k)}"
        df[col] = (df["treat"] * (df["event_week"] == k)).astype(float)
        term_of[k] = col

    formula = f"{metric} ~ C(pref) + C(date) + " + " + ".join(term_of.values())
    res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(**_cluster_kwds(df))

    rows = []
    for k in weeks:
        if k == base_week:
            rows.append({"event_week": k, "beta": 0.0, "se": 0.0, "ci_low": 0.0, "ci_high": 0.0})
            continue
        term = term_of[k]
        ci = res.conf_int(alpha=config.ALPHA).loc[term]
        rows.append(
            {
                "event_week": k,
                "beta": float(res.params[term]),
                "se": float(res.bse[term]),
                "ci_low": float(ci[0]),
                "ci_high": float(ci[1]),
            }
        )

    out = pd.DataFrame(rows).sort_values("event_week").reset_index(drop=True)
    for src, dst in [("beta", "lift_pct"), ("ci_low", "lift_low"), ("ci_high", "lift_high")]:
        out[dst] = (np.exp(out[src]) - 1) * 100
    out["is_post"] = out["event_week"] >= 0

    span = (
        df.groupby("event_week")["date"]
        .agg(date_from="min", date_to="max", n_days="nunique")
        .reset_index()
    )
    out = out.merge(span, on="event_week", how="left")
    # 端の不完全な週（7日未満）は他の週と比較できないため印を付ける
    out["full_week"] = out["n_days"] == 7
    return out


# ---------------------------------------------------------------- 検証

def check_fe_redundancy(panel: pd.DataFrame, metric: str = config.PRIMARY,
                        tol: float = 1e-6) -> dict:
    """「県FE・日付FEは点推定に対して冗長」という主張をデータ上で検証する。

    バランスドパネルでは、PPML(FE付き) の beta は
    群合計の対数比の差（＝集計しただけのDID）と一致するはず。
    一致しなければ前提が崩れているので、黙って進めず気づけるようにする。
    """
    agg = panel.pivot_table(index="treat", columns="post", values=metric, aggfunc="mean")
    closed_form = float(
        (np.log(agg.loc[1, 1]) - np.log(agg.loc[1, 0]))
        - (np.log(agg.loc[0, 1]) - np.log(agg.loc[0, 0]))
    )
    ppml = fit_ppml(panel, metric).beta
    return {
        "closed_form_beta": closed_form,
        "ppml_beta": ppml,
        "abs_diff": abs(closed_form - ppml),
        "matches": abs(closed_form - ppml) < tol,
    }
