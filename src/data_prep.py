"""生ログの読み込み・結合・検証・パネル化。

生ログは「エントリーが1件以上発生した組み合わせ」しか行を持たない疎なデータで、
0件のセルは行として存在しない。県×日へ集約する際に必ず0埋めを行う。
埋め忘れると小規模県の水準を過大評価するため、行数を検証して落ちるようにしている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

COLMAP = {
    "日付": "date",
    "Prefecture": "pref",
    "都道府県": "pref_ja",
    "性別": "gender",
    "年代": "age",
    "前職種": "job",
    "エントリー数": "entries",
    "面談実施数": "interviews",
}


class DataValidationError(AssertionError):
    """前処理の前提が崩れたときに送出する。黙って進めない。"""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise DataValidationError(message)


def load_raw() -> pd.DataFrame:
    """生ログを読み込み、群割当を結合する。

    BOM付きUTF-8のため encoding='utf-8-sig' が必須（指定しないと先頭列名が壊れる）。
    """
    log = pd.read_csv(config.LOG_CSV, encoding="utf-8-sig").rename(columns=COLMAP)
    assign = pd.read_csv(config.ASSIGN_CSV).rename(columns={"Prefecture": "pref"})

    _check(
        len(log) == config.EXPECTED_RAW_ROWS,
        f"生ログの行数が想定と異なる: {len(log)} != {config.EXPECTED_RAW_ROWS}",
    )
    _check(not log.isna().any().any(), "生ログに欠損値が含まれている")
    _check(
        ((assign["Test"] + assign["Control"]) == 1).all(),
        "群割当が排他的でない（TestとControlの両方または どちらでもない県がある）",
    )
    _check(
        set(log["pref"]) == set(assign["pref"]),
        "ログと群割当で都道府県の表記が一致しない",
    )

    log["date"] = pd.to_datetime(log["date"])
    merged = log.merge(assign[["pref", "Test"]], on="pref", how="left")
    _check(
        len(merged) == len(log),
        f"群割当の結合で行数が変化した: {len(merged)} != {len(log)}",
    )
    merged = merged.rename(columns={"Test": "treat"})

    _check(
        merged.loc[merged.treat == 1, "pref"].nunique() == config.EXPECTED_N_TREAT,
        "配信群の県数が想定と異なる",
    )
    _check(
        merged.loc[merged.treat == 0, "pref"].nunique() == config.EXPECTED_N_CONTROL,
        "非配信群の県数が想定と異なる",
    )
    return merged


def build_panel(raw: pd.DataFrame | None = None, by: list[str] | None = None) -> pd.DataFrame:
    """県×日（+任意の属性）のバランスドパネルを作る。0件セルは0で埋める。

    Parameters
    ----------
    by : 追加の分割軸（例 ["gender"]）。省略時は県×日のみ。
    """
    raw = load_raw() if raw is None else raw
    keys = ["pref", "date"] + (by or [])

    agg = raw.groupby(keys, as_index=False)[["entries", "interviews"]].sum()

    # --- 0埋め：取りうるキーの直積で貼り直す ---
    levels = [sorted(raw[k].unique()) for k in keys]
    full = pd.MultiIndex.from_product(levels, names=keys)
    panel = (
        agg.set_index(keys)
        .reindex(full, fill_value=0)
        .reset_index()
    )

    expected_rows = 1
    for lv in levels:
        expected_rows *= len(lv)
    _check(
        len(panel) == expected_rows,
        f"0埋め後の行数が直積と一致しない: {len(panel)} != {expected_rows}",
    )
    _check(
        panel["entries"].sum() == raw["entries"].sum(),
        "0埋めの前後でエントリー総数が変化した",
    )

    assign = raw[["pref", "treat"]].drop_duplicates()
    panel = panel.merge(assign, on="pref", how="left")

    panel["post"] = (panel["date"] >= pd.Timestamp(config.POST_START)).astype(int)
    panel["did"] = panel["treat"] * panel["post"]
    panel["dow"] = panel["date"].dt.dayofweek
    panel["group"] = panel["treat"].map({1: "配信群", 0: "非配信群"})

    _check(
        panel["pref"].nunique() == config.EXPECTED_N_PREF,
        "県数が47でない",
    )
    _check(
        panel["date"].nunique() == config.EXPECTED_N_DAYS,
        f"日数が{config.EXPECTED_N_DAYS}でない",
    )
    _check(
        panel.groupby("pref")["date"].nunique().nunique() == 1,
        "パネルがバランスしていない（県によって観測日数が異なる）",
    )
    return panel


def prefecture_summary(panel: pd.DataFrame, metric: str = config.PRIMARY) -> pd.DataFrame:
    """県単位に集約し、事前期／介入期の日平均と対数変化率を返す。

    対数変化率 dlog は、県固定効果を差分で除去した後の被説明変数にあたる
    （対数を取れば「県ごとに変化率を見る」ことと「県固定効果を入れる」ことは同じ操作）。
    pre_total は規模加重の推定に使う重み。
    """
    means = (
        panel.pivot_table(index=["pref", "treat"], columns="post", values=metric, aggfunc="mean")
        .rename(columns={0: "pre_mean", 1: "post_mean"})
        .reset_index()
    )
    means.columns.name = None

    totals = (
        panel.pivot_table(index="pref", columns="post", values=metric, aggfunc="sum")
        .rename(columns={0: "pre_total", 1: "post_total"})
        .reset_index()
    )
    totals.columns.name = None

    out = means.merge(totals, on="pref", how="left")
    _check(
        (out["pre_mean"] > 0).all(),
        "事前期の平均が0以下の県がある。対数変換できないため個別対応が必要",
    )
    out["ratio"] = out["post_mean"] / out["pre_mean"]
    out["dlog"] = np.log(out["ratio"])
    return out.sort_values(["treat", "pref"]).reset_index(drop=True)
