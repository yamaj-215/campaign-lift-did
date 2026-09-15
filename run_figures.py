"""図の生成。

    python run_figures.py

output/figures/ に PNG を出力する。run_did.py と同じ推定を経由するため、
資料に載せる図と本文の数字が食い違わない。
"""

from __future__ import annotations

import sys

import config
from src import data_prep, inference, models, viz


def main() -> int:
    viz.setup_style()

    panel = data_prep.build_panel()
    summary = data_prep.prefecture_summary(panel, config.PRIMARY)
    master = data_prep.load_master()

    beta = models.fit_ppml(panel).beta
    event_study = models.fit_event_study(panel)
    perm = inference.permutation_test(summary, weighted=True)
    placebo = inference.placebo_time_test(panel)
    head = inference.headline_estimate(panel, summary, beta)
    monthly = viz.monthly_change(panel)
    weekly = viz.weekly_change(panel)

    config.TABLES.mkdir(parents=True, exist_ok=True)
    monthly.merge(master, on="pref", how="left").to_csv(
        config.TABLES / "monthly_change.csv", index=False, encoding="utf-8-sig")
    weekly.merge(master, on="pref", how="left").to_csv(
        config.TABLES / "weekly_change.csv", index=False, encoding="utf-8-sig")

    paths = [
        viz.plot_trend(panel),
        viz.plot_event_study(event_study, head),
        viz.plot_prefecture_dots(summary, master),
        viz.plot_inference(perm, placebo, head),
        viz.plot_monthly_change(monthly, master),
        viz.plot_weekly_change(weekly, master),
    ]
    for p in paths:
        print(f"saved: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
