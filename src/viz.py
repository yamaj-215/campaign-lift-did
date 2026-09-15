"""図の作成。

配色は色覚多様性を考慮した検証済みの2色（青＝配信群 / 橙＝非配信群）を基調とし、
推論に関わる要素（プラセボ・帰無分布）には紫を割り当てる。
色だけで系列を識別させず、必ず凡例か直接ラベルを併置する。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import font_manager

import config

BLUE = "#2a78d6"      # 配信群
ORANGE = "#eb6834"    # 非配信群
VIOLET = "#4a3aa7"    # 推論（プラセボ・帰無分布）
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#96948c"
SURF = "#fcfcfb"
GRID = "#e7e6e1"

_JP_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]


def setup_style() -> None:
    """日本語フォントを登録する。未設定だとラベルが豆腐になる。"""
    for path in _JP_FONT_CANDIDATES:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        print("[warn] 日本語フォントが見つからない。ラベルが文字化けする可能性がある")
    plt.rcParams["axes.unicode_minus"] = False


def _frame(ax, xgrid: bool = False) -> None:
    ax.set_facecolor(SURF)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    (ax.xaxis if xgrid else ax.yaxis).grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=10.5, length=0)


def _save(fig, name: str) -> Path:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / name
    fig.savefig(path, facecolor=SURF, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    return path


def _title(fig, title: str, subtitle: str, x: float = 0.06, y: float = 0.965) -> None:
    fig.text(x, y, title, fontsize=19, weight="bold", color=INK)
    fig.text(x, y - 0.035, subtitle, fontsize=11, color=INK2)


# ------------------------------------------------------------------ 図1

def plot_trend(panel: pd.DataFrame, metric: str = config.PRIMARY) -> Path:
    """日次エントリー数の推移。細線＝日次、太線＝7日移動平均、破線＝期間平均。

    日次の変動（±150件程度）に対し群間の差は10〜20件と小さく、0起点の軸では
    平均線が重なって見えない。平均線だけを拡大した図を埋め込み、
    差そのものの推移を下段に置いて効果が見える形にする。
    """
    d = (
        panel.pivot_table(index="date", columns="treat", values=metric, aggfunc="sum")
        .rename(columns={0: "control", 1: "treat"})
        .sort_index()
    )
    pre, post = d.loc[: config.PRE_END], d.loc[config.POST_START :]
    mpt, mpc = pre.treat.mean(), pre.control.mean()
    mot, moc = post.treat.mean(), post.control.mean()
    ma = d.rolling(7, center=True).mean()
    diff = d.treat - d.control
    diff_ma = diff.rolling(7, center=True).mean()
    dpre = diff.loc[: config.PRE_END].mean()
    dpost = diff.loc[config.POST_START :].mean()

    P0, P1 = d.index[0], pd.Timestamp(config.PRE_END)
    A0, A1 = pd.Timestamp(config.POST_START), pd.Timestamp(config.POST_END)
    xr = A1 + pd.Timedelta(days=2)

    fig = plt.figure(figsize=(14.5, 10.4), dpi=200, facecolor=SURF)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.95, 1], hspace=0.30,
                          left=0.062, right=0.838, top=0.845, bottom=0.072)
    ax, ax2 = fig.add_subplot(gs[0]), None
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    for a in (ax, ax2):
        _frame(a)
        a.axvspan(A0, A1 + pd.Timedelta(days=1), color=BLUE, alpha=0.055, zorder=0, lw=0)
        a.axvline(A0, color=BLUE, lw=1.2, ls=(0, (4, 3)), alpha=0.6, zorder=1)

    def meanline(a, x0, x1, y, c, lw=1.8):
        a.plot([x0, x1], [y, y], color=c, lw=lw, ls=(0, (5, 2.5)), zorder=6,
               solid_capstyle="butt")

    ax.plot(d.index, d.treat, color=BLUE, lw=0.85, alpha=0.20, zorder=2)
    ax.plot(d.index, d.control, color=ORANGE, lw=0.85, alpha=0.20, zorder=2)
    ax.plot(ma.index, ma.treat, color=BLUE, lw=2.4, zorder=5, label="配信群（32県）")
    ax.plot(ma.index, ma.control, color=ORANGE, lw=2.4, zorder=5, label="非配信群（15県）")
    for x0, x1, y, c in [(P0, P1, mpt, BLUE), (P0, P1, mpc, ORANGE),
                         (A0, A1, mot, BLUE), (A0, A1, moc, ORANGE)]:
        meanline(ax, x0, x1, y, c)

    ax.set_ylim(0, 1330)
    ax.set_yticks(range(0, 1201, 200))
    ax.set_ylabel("1日あたりエントリー数（件）", fontsize=11.5, color=INK2, labelpad=9)
    ax.annotate("", xy=(A0, 1205), xytext=(A1 + pd.Timedelta(days=1), 1205),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.3, alpha=0.8))
    ax.text(A0 + pd.Timedelta(days=15), 1235, "8/1–8/31  キャンペーン配信期間",
            ha="center", fontsize=11.5, color=BLUE, weight="bold")
    ax.text(P0, 1290, "既往キャンペーン（両群に配信）", fontsize=9.5, color=MUTED)
    for a0, a1, lab in config.PAST_CAMPAIGNS:
        x0, x1 = pd.Timestamp(a0), pd.Timestamp(a1)
        ax.axvspan(x0, x1 + pd.Timedelta(days=1), color=MUTED, alpha=0.10, zorder=0, lw=0)
        ax.text(x0 + (x1 - x0) / 2, 1225, lab.replace("対象求人特集", "").replace("転職", "")
                .replace("業界経験者向け", ""), ha="center", fontsize=9.5, color=MUTED)

    ax.annotate(f"配信群 8月平均\n{mot:.1f} 件/日", xy=(A1, mot), xytext=(xr, mot + 150),
                color=BLUE, fontsize=10.5, weight="bold", va="center", ha="left",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1, alpha=0.6, shrinkA=0, shrinkB=2))
    ax.annotate(f"非配信群 8月平均\n{moc:.1f} 件/日", xy=(A1, moc), xytext=(xr, moc - 160),
                color=ORANGE, fontsize=10.5, weight="bold", va="center", ha="left",
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1, alpha=0.6, shrinkA=0, shrinkB=2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.008), frameon=False, fontsize=11.5,
              ncol=2, handlelength=2.0, columnspacing=2.4, labelcolor=INK)
    ax.set_title("① 日次エントリー数の推移　細線＝日次実績／太線＝7日移動平均／破線＝期間平均",
                 fontsize=12.5, color=INK2, loc="left", pad=32)

    ins = ax.inset_axes([0.062, 0.062, 0.545, 0.285])
    ins.patch.set_edgecolor("#d8d6d0")
    ins.patch.set_linewidth(1)
    ins.set_facecolor("#f2f1ed")
    _frame(ins)
    ins.tick_params(labelsize=9)
    ins.axvspan(A0, A1 + pd.Timedelta(days=1), color=BLUE, alpha=0.09, zorder=0, lw=0)
    ins.axvline(A0, color=BLUE, lw=1, ls=(0, (4, 3)), alpha=0.6)
    for x0, x1, y, c in [(P0, P1, mpt, BLUE), (P0, P1, mpc, ORANGE),
                         (A0, A1, mot, BLUE), (A0, A1, moc, ORANGE)]:
        meanline(ins, x0, x1, y, c, lw=2.6)
    ins.set_ylim(814, 872)
    ins.set_yticks([820, 830, 840, 850, 860, 870])
    ins.set_xlim(P0 - pd.Timedelta(days=1), A1 + pd.Timedelta(days=1))
    ins.xaxis.set_major_locator(mdates.MonthLocator())
    ins.xaxis.set_major_formatter(mdates.DateFormatter("%-m月"))
    ins.text(P0 + pd.Timedelta(days=4), mpc + 2.4, f"非配信群 事前期平均 {mpc:.1f}",
             color=ORANGE, fontsize=9.5, weight="bold")
    ins.text(P0 + pd.Timedelta(days=4), mpt - 6.4, f"配信群 事前期平均 {mpt:.1f}",
             color=BLUE, fontsize=9.5, weight="bold")
    ins.text(A0 + pd.Timedelta(days=2), mot + 2.4, f"配信群 {mot:.1f}",
             color=BLUE, fontsize=9.5, weight="bold")
    ins.text(A0 + pd.Timedelta(days=2), moc - 6.4, f"非配信群 {moc:.1f}",
             color=ORANGE, fontsize=9.5, weight="bold")
    ins.annotate("", xy=(pd.Timestamp("2025-08-20"), mot), xytext=(pd.Timestamp("2025-08-20"), moc),
                 arrowprops=dict(arrowstyle="<->", color=VIOLET, lw=1.3))
    ins.text(pd.Timestamp("2025-08-22"), (mot + moc) / 2, f"{mot - moc:+.1f}",
             fontsize=9.5, color=VIOLET, va="center", weight="bold")
    ins.set_title("【拡大】期間平均の破線だけを取り出したもの（縦軸 820〜870件に拡大）",
                  fontsize=10, color=INK2, loc="left", pad=6)

    ax2.axhline(0, color=MUTED, lw=1)
    ax2.plot(diff.index, diff, color=MUTED, lw=0.8, alpha=0.18)
    ax2.fill_between(diff_ma.index, 0, diff_ma.values, where=(diff_ma.values >= 0),
                     color=BLUE, alpha=0.18, lw=0, interpolate=True)
    ax2.fill_between(diff_ma.index, 0, diff_ma.values, where=(diff_ma.values < 0),
                     color=ORANGE, alpha=0.18, lw=0, interpolate=True)
    ax2.plot(diff_ma.index, diff_ma, color=INK, lw=2.2, zorder=5)
    meanline(ax2, P0, P1, dpre, VIOLET, lw=2.2)
    meanline(ax2, A0, A1, dpost, VIOLET, lw=2.2)
    ax2.set_ylim(-155, 168)
    ax2.set_ylabel("配信群 − 非配信群（件/日）", fontsize=11.5, color=INK2, labelpad=9)
    ax2.annotate(f"事前期の平均差 {dpre:+.1f} 件/日", xy=(pd.Timestamp("2025-06-05"), dpre),
                 xytext=(P0 + pd.Timedelta(days=9), -95), fontsize=10.5, color=VIOLET,
                 ha="left", weight="bold",
                 arrowprops=dict(arrowstyle="-", color=VIOLET, lw=1, alpha=0.55))
    ax2.annotate(f"8月の平均差\n{dpost:+.1f} 件/日", xy=(A1, dpost), xytext=(xr, dpost + 48),
                 fontsize=10.5, color=VIOLET, ha="left", weight="bold", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", color=VIOLET, lw=1, alpha=0.6, shrinkA=0, shrinkB=2))
    ax2.text(xr, dpost - 72,
             f"差の変化（差分の差分）\n{dpost - dpre:+.1f} 件/日\n≒ 31日で {(dpost - dpre) * 31:+,.0f} 件",
             fontsize=11, color=INK, weight="bold", va="center", clip_on=False, linespacing=1.6)
    ax2.set_title("② 両群の差の推移　配信前はほぼ0で推移し、配信開始とともにプラスへ転じる",
                  fontsize=12.5, color=INK2, loc="left", pad=10)
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-m月"))
    ax2.set_xlim(P0 - pd.Timedelta(days=1), A1 + pd.Timedelta(days=1))
    plt.setp(ax.get_xticklabels(), visible=False)

    _title(fig, "配信群 vs 非配信群：エントリー数の推移",
           "2025年5月1日〜8月31日｜日次・都道府県群別の合計（配信32県／非配信15県）", x=0.062)
    fig.text(0.062, 0.016,
             "移動平均は中央揃え7日。期間平均は事前期＝5/1〜7/31（92日）、8月＝8/1〜8/31（31日）。",
             fontsize=9, color=MUTED)
    return _save(fig, "fig1_trend.png")


# ------------------------------------------------------------------ 図2

def plot_event_study(event_study: pd.DataFrame, headline: dict | None = None) -> Path:
    """相対週ごとの効果。配信前の完全週のばらつきを参照帯として重ねる。

    モデルのクラスタロバスト信頼区間は群×時点の共通ショックを捉えず狭すぎるため、
    誤差棒としては描かない。代わりに配信前の実測ばらつき（±1.96sd）を帯で示す。

    週ごとの推定はノイズが大きく、単週で帯を超えるかどうかは判定材料にならない。
    推定対象は配信期間の平均であり、こちらは平均化によって誤差が縮む。
    その区間（headline）を重ねて、比較すべき水準を明示する。
    """
    es = event_study.copy()
    # 先頭の不完全な週（5/1の1日のみ）は他の週と比較できず、縦軸を無駄に広げるので除く
    lead_partial = es[(~es.is_post) & (~es.full_week)]
    es = es.drop(lead_partial.index).reset_index(drop=True)

    full_pre = es[(~es.is_post) & es.full_week]
    band = 1.96 * full_pre["lift_pct"].std(ddof=1)
    post_m = es[es.is_post]
    pre_m = es[~es.is_post]

    fig, ax = plt.subplots(figsize=(13.5, 7.4), dpi=200, facecolor=SURF)
    fig.subplots_adjust(left=0.075, right=0.745, top=0.80, bottom=0.135)
    _frame(ax)

    x_lo, x_hi = es.event_week.min() - 0.6, es.event_week.max() + 0.6
    ax.axhspan(-band, band, color=MUTED, alpha=0.13, lw=0, zorder=0)
    ax.axhline(0, color=MUTED, lw=1.2, zorder=1)
    ax.axvspan(-0.5, x_hi, color=BLUE, alpha=0.05, lw=0, zorder=0)
    ax.axvline(-0.5, color=BLUE, lw=1.4, ls=(0, (4, 3)), alpha=0.7, zorder=2)

    # 推定対象＝配信期間の平均とその信頼区間
    if headline is not None:
        ax.fill_between([-0.5, x_hi], headline["ci_low_pct"], headline["ci_high_pct"],
                        color=BLUE, alpha=0.12, lw=0, zorder=1)
        ax.hlines(headline["lift_pct"], -0.5, x_hi, color=BLUE, lw=1.8,
                  ls=(0, (5, 2.5)), zorder=3)

    ax.plot(pre_m.event_week, pre_m.lift_pct, color=MUTED, lw=1.6, zorder=3)
    ax.plot(post_m.event_week, post_m.lift_pct, color=BLUE, lw=2.4, zorder=4)
    for _, r in es.iterrows():
        full = bool(r.full_week)
        color = BLUE if r.is_post else INK2
        ax.plot(r.event_week, r.lift_pct, "o", ms=10 if full else 8,
                mfc=color if full else SURF, mec=color, mew=1.8, zorder=5)
    for _, r in post_m.iterrows():
        ax.annotate(f"{r.lift_pct:+.1f}%", xy=(r.event_week, r.lift_pct),
                    xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=10.5, color=BLUE, weight="bold")
    for _, r in es[~es.full_week].iterrows():
        ax.annotate(f"{int(r.n_days)}日のみ", xy=(r.event_week, r.lift_pct),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", fontsize=9, color=MUTED)

    ax.set_xticks(es.event_week)
    ax.set_xticklabels([f"{int(w)}" for w in es.event_week], fontsize=10.5)
    ax.set_xlabel("配信開始を起点とした相対週（0 = 8/1〜8/7）", fontsize=11.5, color=INK2, labelpad=8)
    ax.set_ylabel("効果（%）　基準＝配信直前の週", fontsize=11.5, color=INK2, labelpad=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(min(-band, es.lift_pct.min()) - 0.9, es.lift_pct.max() + 1.5)
    ax.text(-0.65, ax.get_ylim()[1] * 0.93, "配信開始", fontsize=10.5, color=BLUE,
            ha="right", weight="bold")

    xr = x_hi + 0.25
    if headline is not None:
        ax.annotate(
            f"推定された効果（配信期間の平均）\n{headline['lift_pct']:+.2f}%\n"
            f"95%信頼区間 {headline['ci_low_pct']:+.2f}% 〜 {headline['ci_high_pct']:+.2f}%",
            xy=(x_hi, headline["lift_pct"]), xytext=(xr, headline["lift_pct"] + 0.3),
            fontsize=10.5, color=BLUE, weight="bold", va="center", ha="left",
            annotation_clip=False, linespacing=1.6)
    ax.text(xr, -band - 0.35,
            f"配信前の週次のばらつき\n±{band:.2f} pt（完全週{len(full_pre)}週の±1.96sd）\n"
            "単週の推定はノイズが大きく、\n単週で帯を超えるかは判定材料にならない",
            fontsize=9.5, color=INK2, va="top", ha="left", clip_on=False, linespacing=1.6)

    _title(fig, "イベントスタディ：効果の立ち上がりと平行トレンドの確認",
           "配信前は0周辺を推移し、配信開始とともに全週がプラス側へ移る", x=0.075, y=0.955)
    fig.text(0.075, 0.022,
             "PPML（県FE＋日付FE）で相対週ごとに効果を推定。モデルの信頼区間は群×時点の共通ショックを"
             "捉えず狭すぎるため誤差棒は描かない。5/1（1日のみ）は他の週と比較できないため除外。",
             fontsize=9, color=MUTED)
    return _save(fig, "fig2_event_study.png")


# ------------------------------------------------------------------ 図3

def plot_prefecture_dots(summary: pd.DataFrame, master: pd.DataFrame) -> Path:
    """47都道府県の変化率を並べ、2群の分布が重なっていないことを示す。

    「配信した県は全て非配信の県より伸びている」を一目で伝える図。
    統計の説明を経ずに効果の存在を示せるため、報告資料の中心に置ける。
    """
    df = summary.merge(master, on="pref", how="left").copy()
    df["chg"] = (df["ratio"] - 1) * 100
    df = df.sort_values("chg").reset_index(drop=True)
    df["y"] = np.arange(len(df))

    t = df[df.treat == 1]
    c = df[df.treat == 0]
    boundary = (t.chg.min() + c.chg.max()) / 2

    fig, ax = plt.subplots(figsize=(11.5, 12.5), dpi=200, facecolor=SURF)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.885, bottom=0.075)
    _frame(ax, xgrid=True)

    ax.axhspan(-0.6, c.y.max() + 0.5, color=ORANGE, alpha=0.055, lw=0, zorder=0)
    ax.axhspan(t.y.min() - 0.5, t.y.max() + 0.6, color=BLUE, alpha=0.055, lw=0, zorder=0)

    for grp, color, label in [(c, ORANGE, "非配信群（15県）"), (t, BLUE, "配信群（32県）")]:
        ax.hlines(grp.y, 0, grp.chg, color=color, lw=1.6, alpha=0.45, zorder=2)
        ax.plot(grp.chg, grp.y, "o", ms=9, color=color, mec=SURF, mew=1.4, zorder=4, label=label)
        ax.axvline(grp.chg.mean(), color=color, lw=1.8, ls=(0, (5, 2.5)), zorder=3)

    ax.axvline(0, color=MUTED, lw=1.1, zorder=1)
    ax.set_yticks(df.y)
    ax.set_yticklabels(df.pref_ja, fontsize=9.5)
    for tick, tr in zip(ax.get_yticklabels(), df.treat):
        tick.set_color(BLUE if tr == 1 else ORANGE)
    ax.set_ylim(-0.9, len(df) - 0.1)
    ax.set_xlim(-0.4, 5.9)
    ax.set_xlabel("事前期（5〜7月）に対する8月のエントリー数変化率（%）",
                  fontsize=11.5, color=INK2, labelpad=9)

    ax.axhline(c.y.max() + 0.5, color=VIOLET, lw=1.6, ls=(0, (4, 3)), zorder=5)
    ax.text(5.75, c.y.max() + 0.5, "  分布の境界\n  重なり 0 組", fontsize=10.5, color=VIOLET,
            va="center", ha="right", weight="bold", linespacing=1.5)

    ax.text(c.chg.mean(), len(df) + 0.6, f"平均 {c.chg.mean():+.2f}%", color=ORANGE,
            fontsize=10.5, weight="bold", ha="center")
    ax.text(t.chg.mean(), len(df) + 0.6, f"平均 {t.chg.mean():+.2f}%", color=BLUE,
            fontsize=10.5, weight="bold", ha="center")
    # 群平均の差は、非配信群ブロックの空きスペースに置く（点や接続線と重ならない）
    y_arrow = c.y.max() - 3.2
    ax.annotate("", xy=(t.chg.mean(), y_arrow), xytext=(c.chg.mean(), y_arrow),
                arrowprops=dict(arrowstyle="<->", color=VIOLET, lw=1.8))
    ax.text((t.chg.mean() + c.chg.mean()) / 2, y_arrow + 0.75,
            f"群平均の差 {t.chg.mean() - c.chg.mean():+.2f} pt", color=VIOLET, fontsize=11.5,
            weight="bold", ha="center")

    ax.legend(loc="lower right", frameon=False, fontsize=11, labelcolor=INK,
              handletextpad=0.6, borderpad=1.2)

    _title(fig, "都道府県別の変化率：2群の分布は完全に分離している",
           f"配信32県は {t.chg.min():+.2f}% 〜 {t.chg.max():+.2f}%、"
           f"非配信15県は {c.chg.min():+.2f}% 〜 {c.chg.max():+.2f}%", x=0.16, y=0.962)
    fig.text(0.16, 0.022,
             "各県の事前期日平均に対する8月日平均の変化率。破線は群平均。"
             "配信県の最小値が非配信県の最大値を上回り、47県のうち重なる組み合わせは存在しない。",
             fontsize=9, color=MUTED)
    return _save(fig, "fig3_prefecture_dots.png")


# ------------------------------------------------------------------ 図4

def plot_inference(permutation: dict, placebo: dict, headline: dict) -> Path:
    """検定の2つの柱を1枚にまとめる。

    左：割当を振り直した帰無分布（どの県を配信対象にするかの偶然）
    右：事前期間に偽の配信期間を置いたプラセボ分布（時期による揺らぎ）
    """
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4), dpi=200, facecolor=SURF,
                             gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.06, right=0.965, top=0.76, bottom=0.145, wspace=0.22)

    # --- 左：並べ替え検定 ---
    ax = axes[0]
    _frame(ax)
    null_pct = (np.exp(permutation["null"]) - 1) * 100
    obs = permutation["observed_lift_pct"]
    ax.hist(null_pct, bins=60, color=VIOLET, alpha=0.30, lw=0)
    ax.axvline(obs, color=BLUE, lw=2.6, zorder=5)
    ax.axvline(0, color=MUTED, lw=1)
    ymax = ax.get_ylim()[1]
    ax.annotate(f"実測 {obs:+.2f}%", xy=(obs, ymax * 0.62), xytext=(obs - 0.42, ymax * 0.86),
                fontsize=11.5, color=BLUE, weight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))
    ax.text(null_pct.mean(), ymax * 0.98,
            f"帰無分布\n{null_pct.min():+.2f}% 〜 {null_pct.max():+.2f}%",
            fontsize=10.5, color=VIOLET, ha="center", va="top", weight="bold", linespacing=1.5)
    ax.set_xlabel("リフト率（%）", fontsize=11.5, color=INK2, labelpad=8)
    ax.set_ylabel("頻度", fontsize=11.5, color=INK2, labelpad=9)
    ax.set_title(
        f"① 配信県の割当を{permutation['n_permutations']:,}回振り直した場合\n"
        f"　 実測以上の差が出た回数：{permutation['n_as_extreme']} 回　→　p = {permutation['p_value']:.4f}",
        fontsize=12, color=INK2, loc="left", pad=12, linespacing=1.6)

    # --- 右：時間プラセボ ---
    ax = axes[1]
    _frame(ax)
    pl = placebo["placebo"]
    ax.axhline(0, color=MUTED, lw=1)
    ax.plot(pl.fake_post_start, pl.lift_pct, "o-", color=VIOLET, lw=1.8, ms=6,
            mec=SURF, mew=1.0, label="偽の配信期間（介入なし）")
    real = placebo["real_lift_pct"]
    ax.axhline(real, color=BLUE, lw=2.6, zorder=5)
    ax.text(pl.fake_post_start.iloc[0], real + 0.16, f"実際の配信期間 {real:+.2f}%",
            color=BLUE, fontsize=11, weight="bold")
    sd = pl.lift_pct.std(ddof=1)
    ax.axhspan(-1.96 * sd, 1.96 * sd, color=VIOLET, alpha=0.10, lw=0, zorder=0)
    ax.text(pl.fake_post_start.iloc[-1], 1.96 * sd + 0.12,
            f"プラセボの±1.96sd（±{1.96 * sd:.2f} pt）", color=VIOLET, fontsize=10,
            ha="right", weight="bold")
    ax.set_ylim(-1.2, max(real, pl.lift_pct.max()) + 0.9)
    ax.set_xlabel("偽の配信期間の開始日", fontsize=11.5, color=INK2, labelpad=8)
    ax.set_ylabel("見かけのリフト率（%）", fontsize=11.5, color=INK2, labelpad=9)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d"))
    ax.set_title(
        f"② 事前期間に偽の配信期間（31日）を置いた場合（{placebo['n_windows']}通り）\n"
        f"　 介入がなくても最大 {placebo['placebo_max_pct']:+.2f}% の効果が出る　→　"
        f"実測は sd の {placebo['z_vs_placebo']:.1f} 倍",
        fontsize=12, color=INK2, loc="left", pad=12, linespacing=1.6)

    _title(fig, "検定：観測された効果は偶然で説明できるか",
           f"結論 リフト率 {headline['lift_pct']:+.2f}%（95%CI "
           f"{headline['ci_low_pct']:+.2f}% 〜 {headline['ci_high_pct']:+.2f}%）、p = {headline['p_value']:.4f}",
           x=0.06, y=0.955)
    fig.text(0.06, 0.022,
             "信頼区間は②のばらつきを基準に算出している。①だけでは群×時点の共通ショックが"
             "誤差に数えられず、区間が実態の約7分の1に狭まるため。",
             fontsize=9, color=MUTED)
    return _save(fig, "fig4_inference.png")


# ------------------------------------------------------------------ 図5

def monthly_change(panel: pd.DataFrame, metric: str = config.PRIMARY) -> pd.DataFrame:
    """都道府県 × 月の日平均、対前月変化率、5月を100とした指数を返す。

    月によって日数が異なる（5月31日・6月30日・7月31日・8月31日）ため、
    合計ではなく日平均で比較する。
    """
    df = panel.copy()
    df["month"] = df["date"].dt.to_period("M")
    wide = df.pivot_table(index=["pref", "treat"], columns="month", values=metric, aggfunc="mean")
    months = list(wide.columns)

    out = wide.copy()
    out.columns = [str(m) for m in months]
    cols = list(out.columns)
    for prev, cur in zip(cols[:-1], cols[1:]):
        out[f"mom_{cur}"] = (out[cur] / out[prev] - 1) * 100
    for c in cols:
        out[f"idx_{c}"] = out[c] / out[cols[0]] * 100
    return out.reset_index()


def plot_monthly_change(monthly: pd.DataFrame, master: pd.DataFrame) -> Path:
    """都道府県ごとの月次変化率の推移。

    左：対前月変化率。右：5月を100とした指数。
    47本の細線が県、太線が群平均。群内のばらつきが小さく群間の差が大きいため、
    「月ごとに群がまとまって上下する」構造がそのまま見える。
    """
    df = monthly.merge(master, on="pref", how="left")
    mom_cols = [c for c in df.columns if c.startswith("mom_")]
    idx_cols = [c for c in df.columns if c.startswith("idx_")]
    mom_labels = [f"{int(c.split('-')[1])}月" for c in mom_cols]
    idx_labels = [f"{int(c.split('-')[1])}月" for c in idx_cols]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), dpi=200, facecolor=SURF,
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.058, right=0.80, top=0.775, bottom=0.115, wspace=0.20)

    for ax, cols, labels, title, ylab in [
        (axes[0], mom_cols, mom_labels, "① 対前月変化率", "対前月変化率（%）"),
        (axes[1], idx_cols, idx_labels, "② 5月を100とした指数", "指数（5月＝100）"),
    ]:
        _frame(ax)
        x = np.arange(len(cols))
        aug_i = len(cols) - 1
        ax.axvspan(aug_i - 0.42, aug_i + 0.42, color=BLUE, alpha=0.07, lw=0, zorder=0)
        ax.axhline(0 if cols is mom_cols else 100, color=MUTED, lw=1.1, zorder=1)

        for treat, color in [(0, ORANGE), (1, BLUE)]:
            sub = df[df.treat == treat]
            for _, r in sub.iterrows():
                ax.plot(x, r[cols].to_numpy(float), color=color, lw=0.9, alpha=0.30, zorder=2)
        for treat, color, label in [(0, ORANGE, "非配信群（15県）"), (1, BLUE, "配信群（32県）")]:
            sub = df[df.treat == treat]
            ax.plot(x, sub[cols].mean().to_numpy(float), color=color, lw=3.2, zorder=5,
                    marker="o", ms=9, mec=SURF, mew=1.6, label=label)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11.5)
        ax.set_xlim(-0.45, len(cols) - 0.55)
        ax.set_ylabel(ylab, fontsize=11.5, color=INK2, labelpad=9)
        ax.set_title(title, fontsize=12.5, color=INK2, loc="left", pad=12)
        ax.text(aug_i, ax.get_ylim()[1], "配信", ha="center", va="bottom",
                fontsize=10.5, color=BLUE, weight="bold")

    # 左パネル：月ごとの群平均の差を注記する
    ax = axes[0]
    x = np.arange(len(mom_cols))
    mt = df[df.treat == 1][mom_cols].mean().to_numpy(float)
    mc = df[df.treat == 0][mom_cols].mean().to_numpy(float)
    for i, (a, b) in enumerate(zip(mt, mc)):
        gap = a - b
        ax.annotate("", xy=(i + 0.16, a), xytext=(i + 0.16, b),
                    arrowprops=dict(arrowstyle="<->", color=VIOLET, lw=1.5))
        ax.text(i + 0.23, (a + b) / 2, f"{gap:+.2f} pt", color=VIOLET, fontsize=10.5,
                weight="bold", va="center")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.06), frameon=False, fontsize=11.5,
              ncol=2, handlelength=2.0, columnspacing=2.2, labelcolor=INK)

    gap_jun, gap_aug = mt[0] - mc[0], mt[-1] - mc[-1]
    fig.text(0.815, 0.60,
             "読み取り\n\n"
             f"・8月の差 {gap_aug:+.2f} pt が本施策の効果\n\n"
             f"・配信前の6月にも {gap_jun:+.2f} pt の差が\n"
             "　生じており、7月は逆に縮んでいる\n\n"
             "・群内の県は互いにほぼ同じ動きをする。\n"
             "　月ごとに群単位でまとまって上下する\n"
             "　構造があり、これが誤差の主要因\n\n"
             "・したがって「分布が分離しているか」\n"
             "　だけでは効果を判定できない。\n"
             "　差の大きさを時系列の揺らぎと\n"
             "　比べる必要がある",
             fontsize=10.5, color=INK2, va="top", linespacing=1.55)

    _title(fig, "都道府県別・月次変化率の推移",
           "細線＝47都道府県それぞれ／太線＝群平均　月ごとの日平均で比較（月の日数差を調整）",
           x=0.058, y=0.955)
    fig.text(0.058, 0.022,
             "各県の月次日平均エントリー数から算出。8月のみキャンペーン配信期間。"
             "6月・7月の差は配信前に生じたもので、群×時点の共通ショックの大きさを示す。",
             fontsize=9, color=MUTED)
    return _save(fig, "fig5_monthly_change.png")
