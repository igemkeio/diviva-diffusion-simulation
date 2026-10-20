"""
Notion の model-2 ページ「cycle 2 / 結果」に貼る図を作る。

    python figures_notion.py          # 6 枚（約 2 分）
    python figures_notion.py 5        # 図5 だけ

結果の 6 項目に 1 枚ずつ対応する。番号は Notion ページの結果の番号と一致している。

    c2fig1_energy.png       1. 結合強度比の問題は解決した
    c2fig2_frap.png         2. FRAP の問題も 3 桁改善した
    c2fig3_two_enrich.png   3. 極濃縮は 2 つあり、同じ数字ではない
    c2fig4_timescale.png    4. 局在が立つのにほぼ一世代かかる
    c2fig5_bistability.png  5. p > 1 なら初期の偏りが残る
    c2fig6_sensitivity.png  6. 効くパラメータは rho_n と p の 2 つだけ

=============================================================================
  ★ PopZ の図は入れない
=============================================================================
`figures2.py` の fig6（持続摂動と利得）と fig7(b)（PopZ パルスのラッチ試験）は
**PopZ の作用そのもの**なので、Notion ページの方針に従って cycle 4
（PopZ の導入、手法の比較）へ送る。cycle 2 に残すのは

    「成長の次数 p > 1 ならクラスター場は双安定で、初期の偏りが残る」

という **モデル自体の性質** までで、これは PopZ を一切使わずに示せる。
図5 がその PopZ 抜きの版である（`figures2.py` の fig7 は PopZ を使っている）。

1〜4 と 6 は `figures2.py` の出力をそのまま使う（再計算しない）。
作り直すのは図5 だけ。配色・体裁は `figures2.py` と共通。
"""

# === repo 構成に合わせたパス解決（figure_code/ から実行する）=========================
#   code/ のモデル本体を import し、図は ../figures/ に出す。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_sys.path.insert(0, _os.path.join(_ROOT, "code"))
_FIGS = _os.path.join(_ROOT, "figures"); _os.makedirs(_FIGS, exist_ok=True)
_os.chdir(_FIGS)
# ==============================================================================


import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cluster as K
from cluster import variant

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

# figures2.py と同じ配色（P 型 D 型色覚で検証済み。緑・青緑・紫は使わない）
MONO = "#1f4e79"      # 膜上の単量体 / 単安定
CLU = "#e8a33d"       # クラスター / 双安定（★ 直接ラベル必須）
GREY = "#8a8a8a"

# 流用する図（figures2.py の出力）
REUSE = {
    1: ("fig3_energy.png", "c2fig1_energy.png"),
    2: ("fig5_frap.png", "c2fig2_frap.png"),
    3: ("fig1_cluster_steady.png", "c2fig3_two_enrich.png"),
    4: ("fig2_timescale.png", "c2fig4_timescale.png"),
    6: ("fig4_sensitivity.png", "c2fig6_sensitivity.png"),
}


def reuse(n):
    src, dst = REUSE[n]
    shutil.copyfile(src, dst)
    print(f"  -> {dst}   ({src} を流用)")


def fig5():
    """p > 1 なら初期の偏りが残る — ★ PopZ を使わずに双安定を示す。

    (a) p を振ったときの非対称枝（分岐図）
    (b) 片極にクラスターの種を置いただけの初期条件からの時間変化。
        PopZ も摂動も無い。p=0.5 では偏りが消え、p=1.2 では育って残る。

    ⚠️ 種の大きさ（seed）は結果を変える。双安定な系には臨界の種があり、
       それより小さいと対称な枝に吸い込まれる。p=1.2 の臨界の種は 2 なので、
       余裕を見て 3 を使う。「種 1 通りで双安定でない」と結論しないこと。
    """
    ps = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5]
    rows = K.bistability_scan(ps, n=101)
    asym = np.array([r[2] for r in rows])
    bist = np.array([bool(r[3]) for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0),
                           gridspec_kw={"width_ratios": [1, 1.15]})

    # ---------------------------------------------------------------- (a)
    ax[0].axvspan(0.3, 1.0, color="#f2f2f2", lw=0)
    ax[0].text(0.67, 2e2, "monostable\n(a head start is erased)",
               fontsize=9, color="#555", ha="center")
    ax[0].text(1.33, 1.0e2, "bistable\n(a head start is kept)",
               fontsize=9, color="#a86c14", ha="center")
    ax[0].plot(ps, np.maximum(asym, 1.0), "o-", color=CLU, lw=2.4, ms=7)
    for x, y, b in zip(ps, asym, bist):
        if b:
            ax[0].text(x + 0.035, max(y, 1) * 0.45, f"{y:,.0f}$\\times$",
                       fontsize=8.5, color="#a86c14", ha="left")
    ax[0].axvline(1.0, color="k", lw=1.2, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_ylim(0.5, 3e5)
    ax[0].set_xlim(0.35, 1.6)
    ax[0].set_xlabel("$p$   (cluster growth exponent)")
    ax[0].set_ylabel("asymmetry of the asymmetric branch")
    ax[0].set_title("(a) $p$ = 1 is the boundary")

    # ---------------------------------------------------------------- (b)
    #   片極にクラスターの種を置くだけ。PopZ は登場しない。
    for p, col in [(0.5, MONO), (1.2, CLU)]:
        par = variant(p=p)
        r = K.solve(101, par, T=1e5, ic="seed_pole", seed=3.0, n_out=140)
        g = r["g"]
        nl = (r["Nt"] * g["A_capL"][:, None]).sum(0)
        nr = (r["Nt"] * g["A_capR"][:, None]).sum(0)
        a = nl / np.maximum(nr, 1e-300)
        m = r["t"] > 0
        ax[1].plot(r["t"][m], a[m], color=col, lw=2.4, label=f"$p$ = {p}")

    ax[1].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[1].text(6e4, 1.35, "symmetric", fontsize=8.5, color="#555", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1e-1, 1e5)
    ax[1].set_ylim(0.5, 1e5)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("pole-to-pole asymmetry of clusters")
    ax[1].legend(fontsize=9, frameon=False, loc="center left")
    ax[1].set_title("(b) Same head start, opposite fate  (no PopZ)")

    fig.tight_layout()
    fig.savefig("c2fig5_bistability.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> c2fig5_bistability.png")


def main(which=None):
    want = [int(x) for x in which] if which else [1, 2, 3, 4, 5, 6]
    for n in want:
        print(f"図{n} ...")
        if n == 5:
            fig5()
        elif n in REUSE:
            reuse(n)
        else:
            print(f"  ★ 図{n} は無い")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
