"""DID分析パイプラインの実行。

    python run_did.py

出力は output/tables/ に CSV で保存し、要点は標準出力に出す。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import config
from src import data_prep, effects, inference, models

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    config.TABLES.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 0. 前処理
    hr("0. データ前処理と検証")
    raw = data_prep.load_raw()
    panel = data_prep.build_panel(raw)
    summary = data_prep.prefecture_summary(panel, config.PRIMARY)
    print(f"生ログ          : {len(raw):,} 行")
    print(f"県×日パネル     : {len(panel):,} 行（{panel.pref.nunique()}県 × {panel.date.nunique()}日、バランス済）")
    print(f"エントリー総数  : {panel.entries.sum():,} 件（0埋め前後で一致を検証済み）")
    print("※ 県×日の粒度では全セルが1件以上。0埋めが効くのは属性別に分割したとき。")

    # ---------------------------------------------------------------- 1. 前提検証
    hr("1. モデルの前提検証")
    red = models.check_fe_redundancy(panel)
    print("[固定効果の冗長性]")
    print(f"  PPML(県FE+日付FE) beta : {red['ppml_beta']:.10f}")
    print(f"  集計のみの閉形式  beta : {red['closed_form_beta']:.10f}")
    print(f"  差 {red['abs_diff']:.1e} → 一致: {red['matches']}")
    print("  バランスドパネルのため固定効果は点推定に対して冗長（対数変化率を取るのと同じ操作）。")
    print("  回帰の価値は点推定ではなく、推論の枠組みと拡張性にある。")

    disp = models.dispersion_diagnostics(panel)
    print("\n[分散構造]")
    print(f"  分散/平均（生データ）        : {disp['raw_var_mean_ratio']:.1f}  ※大半は県の規模差")
    print(f"  Pearson分散（固定効果適用後）: {disp['pearson_dispersion']:.3f} → 過分散: {disp['overdispersed']}")
    print(f"  NB2のalpha推定値             : {disp['nb2_alpha']:.2e}（0以下＝過分散なし）")
    print("  固定効果適用後に過分散はないため、主分析はPPMLで妥当。負の二項は不要。")

    # ---------------------------------------------------------------- 2. 推定
    hr("2. DIDの推定（主要指標＝エントリー数）")
    fits = [
        models.fit_ppml(panel),
        models.fit_ppml_no_datefe(panel),
        models.fit_negbin(panel),
        models.fit_ols_log(panel),
        models.fit_prefecture_level(summary, weighted=False),
        models.fit_prefecture_level(summary, weighted=True),
    ]
    table = pd.DataFrame([f.as_row() for f in fits])
    print(
        table[["モデル", "リフト率%", "CI下限%", "CI上限%", "クラスタ数", "備考"]]
        .round({"リフト率%": 3, "CI下限%": 3, "CI上限%": 3})
        .to_string(index=False)
    )
    table.to_csv(config.TABLES / "did_models.csv", index=False, encoding="utf-8-sig")
    print("\n※ ここに出ている信頼区間は県クラスタのみを考慮したもので、後述のとおり狭すぎる。")
    print("   log-OLSが低めに出るのは、日次の対数平均が小規模県を相対的に重く扱うため。")

    primary = fits[0]

    # ---------------------------------------------------------------- 3. 検定
    hr("3. 検定：割当の無作為化に基づく並べ替え検定")
    sep = inference.separation_check(summary)
    print(f"配信群32県の変化率   : {sep['treat_min_pct']:+.2f}% 〜 {sep['treat_max_pct']:+.2f}%")
    print(f"非配信群15県の変化率 : {sep['control_min_pct']:+.2f}% 〜 {sep['control_max_pct']:+.2f}%")
    print(f"分布の重なり         : {sep['n_overlap']} 組 → 完全分離: {sep['fully_separated']}")

    perm = inference.permutation_test(summary, weighted=True)
    null_lo = (np.exp(perm["null"].min()) - 1) * 100
    null_hi = (np.exp(perm["null"].max()) - 1) * 100
    print(f"\n並べ替え検定 {perm['n_permutations']:,} 回（配信32県・非配信15県の構成を保って再割当）")
    print(f"  観測されたリフト率 : {perm['observed_lift_pct']:+.3f}%")
    print(f"  帰無分布の範囲     : {null_lo:+.2f}% 〜 {null_hi:+.2f}%")
    print(f"  同等以上の差の回数 : {perm['n_as_extreme']} 回 / {perm['n_permutations']:,} 回")
    print(f"  両側p値            : p < {perm['p_value']:.4f}")
    print("  ※ 1回も出なければ (0+1)/(10,000+1) が機械的に出るだけで、等号では書けない下限値。")
    print("  ※ この検定が答えるのは「県の分け方の偶然で説明できるか」のみ。配信前の月でも同じ値が出る。")

    # ---------------------------------------------------------------- 4. 誤差評価
    hr("4. 誤差評価：県クラスタだけでは足りない")
    print("[PPMLの分散推定を指定別に比較]")
    print("  PPMLは疑似最尤法なので素のMLE分散は無効。サンドイッチ型が必須。")
    for label, kw in [
        ("素のMLE分散（無効）", {}),
        ("Huber-White (HC0)", {"cov_type": "HC0"}),
        ("県クラスタ・サンドイッチ", {"cov_type": "cluster",
                                      "cov_kwds": {"groups": panel["pref"].values}}),
    ]:
        import statsmodels.api as _sm, statsmodels.formula.api as _smf
        _r = _smf.glm(f"{config.PRIMARY} ~ did + C(pref) + C(date)", data=panel,
                      family=_sm.families.Poisson()).fit(**kw)
        print(f"  {label:26s} SE {_r.bse['did']:.5f}  z {_r.params['did']/_r.bse['did']:6.1f}")
    print("  点推定は3つとも同じ。クラスタSEが最も小さいのは、日次の撹乱が県内で打ち消し合うため。")
    print("  つまりクラスタSEは「県どうしの違い」を正しく測っているが、")
    print("  「群全体がキャンペーン以外の理由で動きうる幅」は測っていない。z=64はその帰結。\n")

    placebo = inference.placebo_time_test(panel)
    print("事前期間に「偽の配信期間（31日）」を置いて同じ推定を繰り返した結果：")
    print(f"  プラセボ窓数     : {placebo['n_windows']}")
    print(f"  見かけの効果     : {placebo['placebo_min_pct']:+.3f}% 〜 {placebo['placebo_max_pct']:+.3f}%")
    print(f"  標準偏差         : {placebo['placebo_sd_pct']:.3f} pt")
    print(f"  実際の効果       : {placebo['real_lift_pct']:+.3f}%  → プラセボsd比 z = {placebo['z_vs_placebo']:.2f}")
    print("\n介入がない期間でも最大 +2.1% の見かけの効果が生じる。")
    print("これは「ある時期に両群を非対称に揺らす全国的なショック（群×時点の共通ショック）」で、")
    print("県クラスタでは誤差として数えられない。したがって報告する区間はこちらを基準に置く。")

    head = inference.headline_estimate(panel, summary, primary.beta)
    print(f"\n県クラスタのみの区間 : [{primary.lift_ci[0]:+.2f}%, {primary.lift_ci[1]:+.2f}%]  ← 狭すぎる")
    print(f"プラセボ基準の区間   : [{head['ci_low_pct']:+.2f}%, {head['ci_high_pct']:+.2f}%]  ← 採用")

    # ---------------------------------------------------------------- 5. 結論
    hr("5. 結論：増分エントリー数")
    inc = effects.incremental_counts(
        panel, primary.beta, (head["ci_low_log"], head["ci_high_log"])
    )
    print(f"配信群8月の実績    : {inc['actual']:,.0f} 件")
    print(f"配信しなかった場合 : {inc['counterfactual']:,.0f} 件（反実仮想）")
    print(f"\n  増分エントリー数 : {inc['incremental']:+,.0f} 件"
          f"  95%信頼区間 [{inc['incremental_low']:+,.0f}, {inc['incremental_high']:+,.0f}] 件")
    print(f"  リフト率         : {head['lift_pct']:+.2f}%"
          f"  95%信頼区間 [{head['ci_low_pct']:+.2f}%, {head['ci_high_pct']:+.2f}%]")
    print(f"  p値              : {head['p_value']:.4f}（{head['p_source']}）")
    print(f"  誤差の出所       : {head['se_source']}")
    print(f"  参考             : {head['perm_note']}")

    nat = effects.extrapolate_nationwide(panel, primary.beta)
    print(f"\n全国配信時の月間増分見込み : {nat['nationwide_incremental']:+,.0f} 件")
    print(f"  前提: {nat['assumption']}")

    # ---------------------------------------------------------------- 6. 副次指標
    hr("6. 副次指標：面談実施数（獲得の質）")
    sec = models.fit_ppml(panel, config.SECONDARY)
    sec_head = inference.headline_estimate(panel, summary, sec.beta, config.SECONDARY)
    print(f"面談実施数のリフト率 : {sec_head['lift_pct']:+.2f}%"
          f"  95%CI [{sec_head['ci_low_pct']:+.2f}%, {sec_head['ci_high_pct']:+.2f}%]")
    print(f"エントリーのリフト率 : {head['lift_pct']:+.2f}%")
    print("→ ほぼ同幅。エントリー増がそのまま面談増につながっており、質の劣化は見られない。")

    rate = (
        panel.groupby(["group", "post"])[[config.PRIMARY, config.SECONDARY]]
        .sum()
        .assign(面談実施率=lambda d: d[config.SECONDARY] / d[config.PRIMARY])
    )
    print("\n面談実施率（post: 0=事前期, 1=8月）")
    print(rate.round(4).to_string())

    # ---------------------------------------------------------------- 7. イベントスタディ
    hr("7. イベントスタディ：効果の立ち上がりと平行トレンド")
    es = models.fit_event_study(panel)
    full_pre = es[(~es.is_post) & es.full_week]
    band = 1.96 * full_pre["lift_pct"].std(ddof=1)
    print(f"配信前（完全週 {len(full_pre)} 週）の効果のばらつき : sd {full_pre['lift_pct'].std(ddof=1):.3f} pt")
    print(f"→ 参照帯（±1.96sd）: ±{band:.2f} pt。この幅に収まる変動はノイズとみなす。\n")

    es_out = es.assign(
        期間=es.apply(lambda r: f"{r.date_from:%m/%d}-{r.date_to:%m/%d}", axis=1),
        区分=np.where(es.is_post, "配信期間", "配信前"),
        帯外=np.where(es.full_week & (es.lift_pct.abs() > band), "*", ""),
    )
    print(
        es_out[["event_week", "期間", "n_days", "区分", "lift_pct", "帯外"]]
        .rename(columns={"event_week": "相対週", "n_days": "日数", "lift_pct": "効果%"})
        .round(3)
        .to_string(index=False)
    )
    es.to_csv(config.TABLES / "event_study.csv", index=False, encoding="utf-8-sig")
    print(f"\n配信前の効果 : 平均 {full_pre.lift_pct.mean():+.3f}%（完全週のみ）")
    print(f"配信後の効果 : 平均 {es[es.is_post].lift_pct.mean():+.3f}%")

    # ---------------------------------------------------------------- 8. セグメント別
    hr("8. セグメント別の効果（探索的）")
    seg = models.segment_effects(raw)
    seg.to_csv(config.TABLES / "segment_effects.csv", index=False, encoding="utf-8-sig")
    print(
        seg.sort_values(["分類", "リフト率%"], ascending=[True, False])
        [["分類", "セグメント", "8月構成比%", "リフト率%", "CI下限%", "CI上限%",
          "z", "p値", "q値(BH)", "有意(q<0.05)", "増分件数"]]
        .round({"8月構成比%": 1, "リフト率%": 2, "CI下限%": 2, "CI上限%": 2,
                "z": 2, "p値": 4, "q値(BH)": 4, "増分件数": 0})
        .to_string(index=False)
    )
    print("\n誤差はセグメントごとに週ブロック・ブートストラップで測り直している。")
    print("16回の探索的な比較なので、BH法でFDRを調整したq値で判定する。")
    print(f"調整後も0と区別できるのは {int(seg['有意(q<0.05)'].sum())} セグメント。")
    print("\n各次元の増分件数の合計（全体+726件と一致するはず）:")
    print(seg.groupby("分類")["増分件数"].sum().round(0).to_string())

    cross = models.segment_cross_effects(raw)
    cross.to_csv(config.TABLES / "segment_cross_effects.csv", index=False, encoding="utf-8-sig")
    print("\n[補助] 性別 × 年代")
    print(cross.round({"リフト率%": 2, "CI下限%": 2, "CI上限%": 2, "z": 2,
                       "p値": 4, "q値(BH)": 4, "配信群8月実績": 0}).to_string(index=False))

    # ---------------------------------------------------------------- 9. CPA
    hr("9. 増分CPA感度分析（消化金額が未受領のため算定式のみ提示）")
    print("増分CPA = 消化金額 ÷ 増分エントリー数\n")
    print(effects.cpa_table(inc["incremental"]).to_string(index=False))

    summary.to_csv(config.TABLES / "prefecture_summary.csv", index=False, encoding="utf-8-sig")
    placebo["placebo"].to_csv(config.TABLES / "placebo_time.csv", index=False, encoding="utf-8-sig")
    print(f"\n出力先: {config.TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
