"""
Cycle 2 の図をすべて生成する。

    python figures2.py          # 全部（約 6 分）
    python figures2.py 3 5      # 図3 と図5 だけ

ファイル名の番号は REPORT_CYCLE2.md / WIKI_CYCLE2.md の図番号と一致している。
すべて cluster.Par の既定値から計算するので、パラメータを変えたら図も変わる。
Cycle 1 の `figures.py` と同じ体裁・配色を使う。

配色について
------------
Cycle 1 の色をそのまま引き継ぎつつ、Cycle 2 で増えた「クラスター」の場に
色を 1 つ割り当てる必要があった。**緑・青緑は使えない**: 細胞質の
赤 #c1666b と並べると P 型・D 型色覚での色差が ΔE 1.1〜5.5 しかなく、
区別できなくなる（検証: dataviz skill の validate_palette.js）。
紫 #7a3b8f も膜の紺 #1f4e79 と ΔE 4.1（protan）で近すぎた。
採用した 紺 / 琥珀 / 赤 の 3 色は全ペアで CVD 合格（最悪 ΔE 11.1）。
琥珀は背景とのコントラストが 2.1:1 と低いので、**必ず直接ラベルを添える**。
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


import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import brentq

import model as M
import cluster as K
from cluster import Par, variant

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

CAP = "#d9e8f5"       # 極キャップの網掛け
MONO = "#1f4e79"      # 膜上の単量体
CLU = "#e8a33d"       # クラスター（★ 背景とのコントラストが低い。直接ラベル必須）
CYT = "#c1666b"       # 細胞質
GREY = "#8a8a8a"      # 参照・解析解・Cycle 1
TRANS = "#7a3b8f"     # 移植した値
GEN = 1500.0          # s, 一世代
FRAP_WT, FRAP_DMINJ = 128.64, 50.9
R_BSUB, L_BSUB = 0.425, 2.8


def _save(fig, name):
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}", flush=True)


def _cell_outline(ax, R, L, npts=2000):
    """細胞の輪郭を描いて極キャップを塗る。"""
    zf = np.linspace(0, L, npts)
    rr = np.full_like(zf, R)
    m = zf < R
    rr[m] = np.sqrt(np.maximum(R**2 - (zf[m] - R)**2, 0))
    m = zf > L - R
    rr[m] = np.sqrt(np.maximum(R**2 - (zf[m] - (L - R))**2, 0))
    ax.fill_between(zf, -rr, rr, color="#f2f2f2", lw=0)
    ax.plot(zf, rr, color="k", lw=1.2)
    ax.plot(zf, -rr, color="k", lw=1.2)
    for lo, hi in [(0, R), (L - R, L)]:
        s = (zf >= lo) & (zf <= hi)
        ax.fill_between(zf[s], -rr[s], rr[s], color=CAP, lw=0)
    return zf, rr


def _half(t, y, lvl=0.5):
    y = (np.asarray(y, float) - y[0]) / (y[-1] - y[0])
    k = int(np.argmax(y >= lvl))
    if k == 0:
        return float(t[0])
    f = (lvl - y[k - 1]) / max(y[k] - y[k - 1], 1e-300)
    return float(t[k - 1] + f * (t[k] - t[k - 1]))


# ------------------------------------------------------------------- 図1
def fig1():
    """曲率がどこに入るか / 定常の 2 つの膜プロファイル / 2 つの極濃縮。"""
    r = K.steady(201, Par)
    g = r["g"]
    z, R, L = g["z"], Par.R, Par.L
    fig, ax = plt.subplots(1, 3, figsize=(12.8, 3.5))

    # (a) 曲率は k_off_n に入る。両方とも「側壁の何倍か」に揃えて 1 軸で描く
    #     （2 つの y 軸は使わない）
    kn = r["k_off_n"]
    mid = len(z) // 2
    life = (np.log(2) / kn) / (np.log(2) / kn[mid])     # 側壁を 1 とした寿命
    wm = g["w"] / g["w"][mid]                           # 側壁を 1 とした結合能
    ax[0].axvspan(0, R, color=CAP)
    ax[0].axvspan(L - R, L, color=CAP)
    ax[0].plot(z, life, color=CLU, lw=2.6)
    ax[0].plot(z, wm, color=MONO, lw=2.2)
    ax[0].text(L / 2, life.max() * 0.93, "cluster lifetime  $1/k_{off,n}(z)$",
               color="#a86c14", ha="center", fontsize=9)
    ax[0].text(L / 2, 1.0 - 0.22, "monomer binding  $w(z)$   (flat: $\\rho$ = 1)",
               color=MONO, ha="center", fontsize=9)
    ax[0].text(R / 2, life.max() * 1.06, "pole\ncap", ha="center", fontsize=8,
               color="#2b6ca3", va="top")
    ax[0].set_xlim(0, L)
    ax[0].set_ylim(0.0, life.max() * 1.30)
    ax[0].set_xlabel("z (µm)")
    ax[0].set_ylabel("relative to the sidewall")
    ax[0].set_title("(a) Curvature acts on cluster lifetime,\nnot on binding")

    # (b) 膜の 2 成分（直接ラベル。凡例は置かない）
    ax[1].axvspan(0, R, color=CAP)
    ax[1].axvspan(L - R, L, color=CAP)
    ax[1].plot(z, r["M"], color=MONO, lw=2.2)
    ax[1].plot(z, r["Nf"], color=CLU, lw=2.6)
    ymax = max(r["M"].max(), r["Nf"].max())
    ax[1].text(L / 2, r["M"][mid] + ymax * 0.05, "monomer  $M(z)$", color=MONO,
               fontsize=9, ha="center")
    ax[1].text(L / 2, r["Nf"][mid] + ymax * 0.05, "cluster  $N(z)$",
               color="#a86c14", fontsize=9, ha="center")
    ax[1].annotate("", xy=(R * 0.5, r["Nf"][0]), xytext=(R * 0.5, r["Nf"][mid]),
                   arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
    ax[1].text(R * 0.58, 0.5 * (r["Nf"][0] + r["Nf"][mid]),
               f"{r['Nf'][0]/r['Nf'][mid]:.1f}$\\times$", fontsize=9,
               color=GREY, va="center")
    ax[1].set_xlabel("z (µm)")
    ax[1].set_ylabel("surface density  (molecules/µm²)")
    ax[1].set_xlim(0, L)
    ax[1].set_ylim(0, ymax * 1.25)
    ax[1].set_title("(b) Only the cluster field is polar")

    # (c) 2 つの極濃縮
    vals = [r["enr_clu"], r["enrichment"]]
    labs = ["cluster field\n(the polar scaffold)",
            "total membrane\n(what a fluorophore shows)"]
    bars = ax[2].bar([0, 1], vals, color=[CLU, MONO], width=.55)
    ax[2].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[2].text(-0.45, 0.88, "no enrichment", fontsize=8.5, color="#555")
    e1 = M.solve(201, M.Par, T=20.0)["enrichment"]
    ax[2].axhline(e1, color=GREY, lw=1.3, ls="--")
    ax[2].text(1.48, e1 + 0.07, f"Cycle 1 (total): {e1:.2f}$\\times$",
               fontsize=8.5, color=GREY, ha="right")
    for rect, v in zip(bars, vals):
        ax[2].text(rect.get_x() + rect.get_width() / 2, v + 0.07,
                   f"{v:.2f}$\\times$", ha="center", fontsize=11,
                   fontweight="bold")
    ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(labs, fontsize=8.2)
    ax[2].set_ylabel("polar enrichment")
    ax[2].set_ylim(0, 3.3)
    ax[2].set_xlim(-0.55, 1.55)
    ax[2].set_title("(c) The two are not the same number")

    fig.tight_layout()
    _save(fig, "fig1_cluster_steady.png")


# ------------------------------------------------------------------- 図2
def fig2():
    """クラスター場の時空間分布 / 立ち上がりが世代時間と同程度であること。"""
    T = 1e5
    r = K.solve(101, Par, T=T, ic="cytoplasm", n_out=200)
    g = r["g"]
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.0),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    z, t = g["z"], r["t"]
    m = t > 1e-2
    pcm = ax[0].pcolormesh(z, t[m], r["Nt"].T[m], shading="auto", cmap="magma")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-2, T)
    ax[0].set_xlabel("z (µm)")
    ax[0].set_ylabel("time (s)")
    ax[0].set_title("(a) Cluster density $N(z,t)$ from expression start")
    ax[0].axhline(GEN, color="w", lw=1.2, ls="--", alpha=.9)
    ax[0].text(Par.L * 0.5, GEN * 1.4, "one generation", color="w", fontsize=8.5,
               ha="center", va="bottom")
    fig.colorbar(pcm, ax=ax[0], label="N  (molecules/µm²)", pad=0.02)

    e, tt = r["enr_clu_t"], r["t"]
    ok = np.isfinite(e) & (tt > 0)
    e, tt = e[ok], tt[ok]
    ess = e[-1]
    ax[1].plot(tt, e, color=CLU, lw=2.6)
    ax[1].text(6e3, 2.62, "cluster enrichment\n(Cycle 2)", color="#a86c14",
               fontsize=9, ha="center")

    r1 = M.solve(201, M.Par, T=3600.0, ic="cytoplasm", n_out=300)
    e1, t1 = r1["enr_t"], r1["t"]
    ok1 = np.isfinite(e1) & (t1 > 0)
    ax[1].plot(t1[ok1], e1[ok1], color=GREY, lw=1.8, ls="--")
    ax[1].text(1.3e-3, 3.10, "Cycle 1 (monomer only):\nsettles in 0.01 s",
               color=GREY, fontsize=8.5, va="top")

    ax[1].axvline(GEN, color="k", lw=1.0, ls="--")
    ax[1].text(GEN * 1.25, 0.96, "one generation", color="k", fontsize=8.5,
               rotation=90, ha="left", va="bottom")
    k50 = int(np.argmax(np.abs(e - ess) / ess < 0.5))
    k90 = int(np.argmax(np.abs(e - ess) / ess < 0.1))
    ax[1].plot(tt[k50], e[k50], "o", color=CYT, ms=6)
    ax[1].text(tt[k50] * 0.55, e[k50] - 0.06, f"50% at {tt[k50]:.0f} s",
               fontsize=8.5, color=CYT, ha="right", va="top")
    ax[1].plot(tt[k90], e[k90], "o", color=CYT, ms=6)
    ax[1].text(tt[k90] * 0.55, e[k90] + 0.06, f"90% at {tt[k90]:.0f} s",
               fontsize=8.5, color=CYT, ha="right", va="bottom")
    ax[1].set_xscale("log")
    ax[1].set_xlim(1e-3, T)
    ax[1].set_ylim(0.9, 3.2)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("polar enrichment")
    ax[1].set_title("(b) Localization is no longer quasi-static")

    fig.tight_layout()
    _save(fig, "fig2_timescale.png")


# ------------------------------------------------------------------- 図3
def fig3():
    """rho_n の校正 / エネルギーのギャップが閉じたこと。"""
    rhos = np.linspace(1.0, 6.0, 22)
    frac = np.array([K.steady(101, variant(rho_n=v))["clu_pole_frac"]
                     for v in rhos])
    dG = M.monomer_curvature_energy()

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    _, a = M.analytic()
    ax[0].plot(rhos, frac, color=CLU, lw=2.6)
    ax[0].text(3.6, 0.80, "cluster molecules\nat the poles", color="#a86c14",
               fontsize=9)
    ax[0].axhline(0.66, color=CYT, lw=1.3, ls=":")
    ax[0].text(1.05, 0.675, "measured 66% (PALM)", color=CYT, fontsize=8.5)
    ax[0].axhline(a, color=GREY, lw=1.2, ls="--")
    ax[0].text(1.05, a + 0.012, f"pole area fraction {a:.3f}  (no preference)",
               color=GREY, fontsize=8.5)
    ax[0].plot(Par.rho_n, 0.66, "o", color=CYT, ms=7)
    ax[0].annotate(f"$\\rho_n$ = {Par.rho_n}\n$\\ln\\rho_n$ = "
                   f"{np.log(Par.rho_n):.2f} $k_BT$",
                   xy=(Par.rho_n, 0.66), xytext=(Par.rho_n + 0.7, 0.45),
                   fontsize=9, color=CYT,
                   arrowprops=dict(arrowstyle="->", color=CYT, lw=1))
    ax[0].set_xlabel(r"$\rho_n$   (pole / sidewall cluster lifetime ratio)")
    ax[0].set_ylabel("fraction of cluster molecules at the poles")
    ax[0].set_xlim(1, 6)
    ax[0].set_ylim(0.2, 1.0)
    ax[0].set_title("(a) Calibration of the cluster curvature preference")

    e_c1 = np.log(M.Par.rho)
    e_c2 = np.log(Par.rho_n)
    Q = e_c2 / dG
    vals = [dG, e_c2, e_c1]
    cols = [GREY, CLU, TRANS]
    labs = ["one monomer\ncan provide",
            "Cycle 2 asks\nof a cluster",
            "Cycle 1 asked of\none monomer"]
    notes = ["", f"= {Q:.0f} monomers\n(measured mean: 47)",
             f"= {e_c1/dG:.0f} monomers\nin one molecule"]
    bars = ax[1].bar([0, 1, 2], vals, color=cols, width=.58)
    ax[1].set_yscale("log")
    ax[1].set_ylim(3e-3, 300)
    ax[1].set_xlim(-0.6, 2.6)
    ax[1].set_xticks([0, 1, 2])
    ax[1].set_xticklabels(labs, fontsize=8.0)
    ax[1].set_ylabel("curvature energy difference  ($k_BT$)")
    ax[1].set_title("(b) The gap a cluster can close")
    for i, (rect, v, nt) in enumerate(zip(bars, vals, notes)):
        x = rect.get_x() + rect.get_width() / 2
        ax[1].text(x, v * 1.35, f"{v:.3g} $k_BT$", ha="center", fontsize=9,
                   fontweight="bold" if i else "normal")
        if nt:
            ax[1].text(x, v * 3.6, nt, ha="center", fontsize=8.2,
                       color="#a86c14" if i == 1 else TRANS)
    ax[1].axhline(dG * 47, color=CYT, lw=1.3, ls=":")
    ax[1].text(-0.55, dG * 47 * 1.25,
               "what a 47-molecule cluster\ncan provide", fontsize=8.2,
               color=CYT, ha="left")
    ax[1].text(1.0, 150, "a cluster reaches it;\na monomer cannot",
               fontsize=9, ha="center", color="#333")

    fig.tight_layout()
    _save(fig, "fig3_energy.png")


# ------------------------------------------------------------------- 図4
def fig4():
    """パラメータ感度。効くのは rho_n と p だけであることを示す。"""
    specs = [(r"$\rho_n$  cluster curvature preference", "rho_n",
              [Par.rho_n / 2, Par.rho_n, Par.rho_n * 2], "x2"),
             (r"$p$  growth exponent  (no data)", "p", [0.3, 0.5, 0.8], "0.3-0.8"),
             (r"$k_{nuc}$  nucleation  (no data)", "k_nuc",
              [1e-6, 1e-5, 1e-4], "x100"),
             (r"$f_{clu}$  cluster fraction", "f_clu", [0.10, 0.198, 0.50],
              "meas. range"),
             (r"$k_{off,n}$  cluster turnover", "k_off_n",
              [0.0054, 0.0115, 0.0136], "meas. range"),
             (r"$f_{nm}$  escape branch  (undetermined)", "f_nm",
              [0.0, 0.5, 1.0], "0-1"),
             (r"$D_m$  membrane diffusion", "D_m", [0.01, 0.1, 1.0], "x100"),
             (r"$N$  expression level", "N", [169, 1690, 169000], "x1000")]
    rows = []
    for lab, name, vals, note in specs:
        es = []
        for v in vals:
            kw = {name: v}
            if name == "f_clu":
                kw = {"f_clu": v, "f_mono": Par.f_mono + Par.f_clu - v}
            es.append(K.steady(101, variant(**kw))["enr_clu"])
        es = np.array(es)
        rows.append((lab, es.min(), es.max(), note))
    rows.sort(key=lambda r: r[2] - r[1])
    base = K.steady(101, Par)["enr_clu"]

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    for i, (lab, lo, hi, note) in enumerate(rows):
        col = CLU if hi - lo > 0.5 else (MONO if hi - lo > 0.05 else GREY)
        ax.plot([lo, hi], [i, i], color=col, lw=7,
                solid_capstyle="round", alpha=.85, zorder=3)
        tail = "   (no effect)" if hi - lo < 0.01 else ""
        ax.text(hi + 0.03, i, note + tail, va="center", fontsize=8, color=GREY)
    ax.axvline(base, color="k", lw=1.0, ls=":", zorder=0)
    ax.text(base, -1.05, "baseline", fontsize=8.5, ha="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_xlabel("cluster polar enrichment")
    ax.set_xlim(1.1, 4.1)
    ax.set_ylim(-1.4, len(rows) - 0.4)
    ax.set_title("The only two levers are the two quantities nobody has measured:\n"
                 "the cluster's curvature preference and the growth exponent",
                 pad=12)

    fig.tight_layout()
    _save(fig, "fig4_sensitivity.png")


# ------------------------------------------------------------------- 図5
def fig5():
    """FRAP が二相になること / Cycle 1 の 3 桁のずれが縮んだこと。"""
    rb = K.solve(101, Par, T=3e3, ic="bleach", n_out=300)
    cap = rb["g"]["A_capL"]
    mono = (rb["Mt"] * cap[:, None]).sum(0)
    clu = (rb["Nt"] * cap[:, None]).sum(0)
    tot = mono + clu
    rec = (tot - tot[0]) / (tot[-1] - tot[0])
    t_fast, t_slow = _half(rb["t"], mono), _half(rb["t"], clu)
    amp_c = (clu[-1] - clu[0]) / (tot[-1] - tot[0])

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9),
                           gridspec_kw={"width_ratios": [1.2, 1]})

    t = rb["t"]
    m = t > 1e-3
    ax[0].plot(t[m], rec[m], color="#333333", lw=2.6)
    ax[0].text(1.5e3, 0.97, "total membrane signal", fontsize=8.5,
               color="#333333", ha="right", va="top")
    ax[0].axvspan(FRAP_DMINJ, FRAP_WT, color=CYT, alpha=.15, lw=0)
    ax[0].text(np.sqrt(FRAP_DMINJ * FRAP_WT), 0.06,
               "measured\n$T_{1/2}$ 51-129 s", color=CYT, fontsize=8.5,
               ha="center")
    ax[0].axhline(1 - amp_c, color=GREY, lw=1.0, ls=":")
    ax[0].annotate(f"fast phase  $T_{{1/2}}$ = {t_fast*1000:.0f} ms\n"
                   f"({100*(1-amp_c):.0f}% of the signal: monomers)",
                   xy=(t_fast, 0.5 * (1 - amp_c)), xytext=(3e-3, 0.62),
                   fontsize=8.5, color=MONO,
                   arrowprops=dict(arrowstyle="->", color=MONO, lw=1))
    ax[0].annotate(f"slow phase  $T_{{1/2}}$ = {t_slow:.0f} s\n"
                   f"({100*amp_c:.0f}%: clusters)",
                   xy=(t_slow, 1 - amp_c / 2), xytext=(0.3, 0.30),
                   fontsize=8.5, color="#a86c14",
                   arrowprops=dict(arrowstyle="->", color=CLU, lw=1))
    ax[0].set_xscale("log")
    ax[0].set_xlim(1e-3, 3e3)
    ax[0].set_ylim(0, 1.05)
    ax[0].set_xlabel("time after bleach (s)")
    ax[0].set_ylabel("fractional recovery")
    ax[0].set_title("(a) Recovery is biphasic - do not fit one exponential")

    items = [("Cycle 1\n(monomer only)", 0.0989, GREY),
             ("Cycle 2\nfast phase", t_fast, MONO),
             ("measured\n(single-exp fit)", FRAP_WT, CYT),
             ("Cycle 2\nslow phase", t_slow, CLU)]
    for i, (lab, v, c) in enumerate(items):
        ax[1].barh(i, v, color=c, height=.55)
        ax[1].text(v * 1.35, i, f"{v*1000:.0f} ms" if v < 1 else f"{v:.0f} s",
                   va="center", fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.03, 4e3)
    ax[1].set_yticks(range(len(items)))
    ax[1].set_yticklabels([i[0] for i in items], fontsize=8.2)
    ax[1].set_xlabel("half-recovery time (s)")
    ax[1].set_title(f"(b) The 3-order gap closed to {t_slow/FRAP_WT:.1f}$\\times$"
                    " - and flipped sign")

    fig.tight_layout()
    _save(fig, "fig5_frap.png")


# ------------------------------------------------------------------- 図6
def fig6():
    """持続的な片極ブロックへの応答。Cycle 1 との比較と、利得が 1 を超えること。"""
    th = np.linspace(0.0, 0.95, 20)
    c2 = np.array([1.0 / K.sustained_asymmetry(float(x), n=101) if x > 0 else 1.0
                   for x in th])
    c1 = np.array([M.asymmetry(float(x)) if x > 0 else 1.0 for x in th])
    ideal = 1.0 / (1.0 - th)

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    ax[0].plot(th, ideal, color=GREY, lw=1.6, ls="--",
               label=r"local equilibrium  $1/(1-\theta)$")
    ax[0].plot(th, c1, color=TRANS, lw=2.0, label="Cycle 1  (monomer only)")
    ax[0].plot(th, c2, color=CLU, lw=2.6, label="Cycle 2  (with clusters)")
    for y in (2, 5, 10):
        ax[0].axhline(y, color="#eee", lw=1, zorder=0)
    for y, col, fn in [(10, TRANS, lambda x: M.asymmetry(x)),
                       (10, CLU, lambda x: 1.0 / K.sustained_asymmetry(x, n=101))]:
        try:
            tk = brentq(lambda x: fn(x) - y, 0.05, 0.95, xtol=1e-3)
            ax[0].plot(tk, y, "o", color=col, ms=6)
            ax[0].text(tk, y * 1.32, f"$\\theta$={tk:.2f}", fontsize=8.5,
                       color=col, ha="center")
        except ValueError:
            pass
    ax[0].set_xlabel(r"$\theta$   (fraction of the polar membrane blocked by PopZ)")
    ax[0].set_ylabel("pole-to-pole asymmetry")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, 40)
    ax[0].set_xlim(0, 0.95)
    ax[0].set_yticks([1, 2, 5, 10, 20])
    ax[0].set_yticklabels(["1", "2", "5", "10", "20"])
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    ax[0].set_title("(a) Clusters need much less PopZ coverage")

    g2, g1 = c2 / ideal, c1 / ideal
    ax[1].plot(th, g1, color=TRANS, lw=2.0)
    ax[1].plot(th, g2, color=CLU, lw=2.6)
    ax[1].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[1].fill_between(th, g2, 1.0, where=(g2 > 1), color=CLU, alpha=.18)
    ax[1].text(0.06, 1.02, "gain = 1  (no amplification)", fontsize=8.5)
    ax[1].text(0.30, g2.max() * 0.97, "Cycle 2: amplifies", color="#a86c14",
               fontsize=9)
    ax[1].text(0.30, 0.80, "Cycle 1: never above 1", color=TRANS, fontsize=9)
    ax[1].set_xlabel(r"$\theta$")
    ax[1].set_ylabel("output asymmetry / imposed asymmetry")
    ax[1].set_ylim(0.4, max(1.3, g2.max() * 1.12))
    ax[1].set_xlim(0, 0.95)
    ax[1].set_title("(b) The output can now exceed the input")

    fig.tight_layout()
    _save(fig, "fig6_sustained.png")


# ------------------------------------------------------------------- 図7
def fig7():
    """成長の次数 p が対称性の運命を決めること（一過的な PopZ が効くかどうか）。"""
    ps = [0.5, 0.8, 1.0, 1.1, 1.2, 1.5]
    rows = K.bistability_scan(ps, n=101)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0),
                           gridspec_kw={"width_ratios": [1, 1.2]})

    asym = np.array([r[2] for r in rows])
    bist = np.array([bool(r[3]) for r in rows])
    ax[0].axvspan(0.3, 1.0, color="#f2f2f2", lw=0)
    ax[0].text(0.67, 2e2, "monostable\n(PopZ must be\nheld on)", fontsize=9,
               color="#555", ha="center")
    ax[0].text(1.33, 1.2e2, "bistable\n(a pulse latches)", fontsize=9,
               color="#a86c14", ha="center")
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

    # (b) ラッチ試験の時間変化
    for p, col, ls in [(0.5, MONO, "-"), (1.2, CLU, "-")]:
        par = variant(p=p)
        g = K.geometry2(101, par)
        block = 1.0 - 0.9 * (g["A_capL"] / g["A"])
        held = K.solve(101, par, T=600.0, ic="uniform", n_out=60, block=block)
        y = np.concatenate([held["C"], held["M"], held["Nf"]])
        rel = K.solve(101, par, T=1e5, n_out=120, y0=y)
        aL = (held["Nt"] * g["A_capL"][:, None]).sum(0)
        aR = (held["Nt"] * g["A_capR"][:, None]).sum(0)
        bL = (rel["Nt"] * g["A_capL"][:, None]).sum(0)
        bR = (rel["Nt"] * g["A_capR"][:, None]).sum(0)
        t = np.concatenate([held["t"], 600.0 + rel["t"][1:]])
        a = np.concatenate([aR / np.maximum(aL, 1e-300),
                            (bR / np.maximum(bL, 1e-300))[1:]])
        ax[1].plot(t, a, color=col, lw=2.4, ls=ls, label=f"$p$ = {p}")
    ax[1].axvline(600.0, color="k", lw=1.2, ls="--")
    ax[1].text(600 * 1.15, 1.6e4, "PopZ removed", fontsize=8.5, rotation=90,
               va="top")
    ax[1].axvspan(1e0, 600.0, color=CAP, alpha=.5, lw=0)
    ax[1].text(60, 3e4, "PopZ on\n($\\theta$ = 0.9)", fontsize=8.5,
               color="#2b6ca3", ha="center")
    ax[1].axhline(1.0, color="k", lw=1.0, ls=":")
    ax[1].text(3e4, 1.35, "symmetric", fontsize=8.5, color="#555", ha="right")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlim(1e0, 1e5)
    ax[1].set_ylim(0.5, 1e5)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("pole-to-pole asymmetry")
    ax[1].legend(fontsize=9, frameon=False, loc="center left")
    ax[1].set_title("(b) A 600 s pulse: erased at $p$ = 0.5, latched at $p$ = 1.2")

    fig.tight_layout()
    _save(fig, "fig7_bistability.png")


# ------------------------------------------------------------------- 図8
def fig8():
    """サイクル全体の見取り図（計算は要約のみ）。"""
    dG = M.monomer_curvature_energy()
    r = K.steady(201, Par)
    Q, _ = K.required_cluster_size(Par.rho_n)
    rb = K.solve(101, Par, T=3e3, ic="bleach", n_out=300)
    cap = rb["g"]["A_capL"]
    t_slow = _half(rb["t"], (rb["Nt"] * cap[:, None]).sum(0))
    th10 = brentq(lambda x: 1.0 / K.sustained_asymmetry(x, n=101) - 10,
                  0.05, 0.95, xtol=1e-3)

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0.4, 6.8)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc, ec, tcol="k"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.35, title, ha="center", fontsize=10.5,
                fontweight="bold", color=tcol)
        for i, s in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.85 - i * 0.31, s, fontsize=8.8,
                    color="#333")

    box(0.2, 3.5, 3.35, 3.05, "CYCLE 1  (done)",
        ["cytoplasm  ⇌  membrane monomer",
         "curvature enters binding $\\rho$", "",
         f"• needs {np.log(M.Par.rho)/dG:.0f}$\\times$ the monomer energy",
         "• FRAP 1300$\\times$ too fast",
         "• gain $\\leq$ 1: no amplification",
         "• a pulse is erased in 5 s",
         "• expression level: no effect"],
        "#f4f0f7", TRANS, TRANS)

    box(4.25, 3.5, 3.35, 3.05, "CYCLE 2  (done)",
        ["+ immobile cluster field $N(z)$",
         "curvature enters $k_{off,n}$, not $k_{on}$", "",
         f"• needs {Q:.0f} molecules/cluster (meas. 47)",
         f"• FRAP biphasic, slow phase {t_slow:.0f} s",
         "• gain up to 1.8: amplifies",
         f"• 10$\\times$ needs $\\theta$ = {th10:.2f} (was 0.94)",
         "• $p$ > 1 ⇒ a pulse latches"],
        "#fdf3e3", CLU, "#a86c14")

    box(8.30, 3.5, 3.35, 3.05, "CYCLE 3  (next)",
        ["S1 occupancy: $k_{on}(1-\\theta)$",
         "S2 cleavage: $k_{off,n}$ + $k_{cl}$", "",
         "• which strategy wins?",
         "• induction timing now matters",
         "   (localization takes ~1 generation)",
         "• % unipolar cells vs wet"],
        "#f8eef0", CYT, "#8f3d43")

    for x0 in (3.62, 7.67):
        ax.add_patch(FancyArrowPatch((x0, 4.9), (x0 + 0.52, 4.9),
                                     arrowstyle="-|>", mutation_scale=18,
                                     lw=2, color="#555"))
    ax.text(3.70, 5.1, "why?", fontsize=9, style="italic", color="#555")
    ax.text(7.75, 5.1, "how?", fontsize=9, style="italic", color="#555")

    ax.add_patch(FancyBboxPatch((0.2, 0.6), 11.45, 2.4,
                                boxstyle="round,pad=0.12",
                                fc="#f7f7f7", ec="#bbb", lw=1.2))
    ax.text(0.45, 2.78, "What cycle 2 established", fontsize=10.5,
            fontweight="bold")
    msgs = [
        f"1.  Clustering closes the energy gap: {Q:.0f} molecules carry what one "
        f"monomer could not ({np.log(M.Par.rho)/dG:.0f}$\\times$ short).",
        "2.  Localization now takes ~1 generation, not 0.01 s  →  "
        "induction timing becomes a real design variable.",
        f"3.  The scaffold enriches {r['enr_clu']:.1f}$\\times$, but the total "
        f"membrane signal only {r['enrichment']:.1f}$\\times$  →  do not compare "
        "the wrong number to microscopy.",
        f"4.  A sustained block now amplifies (gain up to 1.8): 10$\\times$ "
        f"asymmetry needs PopZ over {th10*100:.0f}% of the pole, not 94%.",
        "5.  Whether a PopZ pulse latches depends entirely on $p$, which nobody "
        "has measured  →  that is the next experiment.",
    ]
    for i, s in enumerate(msgs):
        ax.text(0.45, 2.40 - i * 0.355, s, fontsize=9.0, color="#222")

    _save(fig, "fig8_roadmap.png")


ALL = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}


if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or sorted(ALL)
    for k in want:
        print(f"fig{k} ...", flush=True)
        ALL[k]()
    print("done")
