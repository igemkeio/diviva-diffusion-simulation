"""
Cycle 1（単量体モデル）の Wiki 用の図をすべて生成する。

    python figures_cycle1.py            # 全部
    python figures_cycle1.py A C        # 指定したものだけ

出力（すべて model.Par の既定値から計算する）:
    figA_geometry.png     大腸菌ジオメトリの定義と、膜（弧長）基準で切ると壊れる話
    figB_fields.png       現在の場（C と M、結合する項と拡散する項）
    figC_kymograph.png    膜密度・細胞質密度の時空間図（横軸 z・縦軸 時間）
    figC_cell3d.gif       スフェロシリンダー表面の色が時間変化する GIF
    figD_scan.png         パラメータスキャン（どれが局在を決めるか）
    figE_energy.png       要求エネルギーと単量体が出せるエネルギーのギャップ
    figF_frap.png         FRAP の時間スケール（モデル 96 ms 対 実測 50〜129 s）
    figG_ceiling.png      極濃縮 vs rho の飽和と、天井 1/a
    figH_asymmetry.png    片極初期条件からの緩和（非対称は保持されない）
    figI_total.png        膜と細胞質の合計（顕微鏡が見る線プロファイル）
    figJ_cell3d_cut.png   断面を切り開いた 3D（定常状態）※予備
    figJ_cell3d_cut.gif   同、時間変化する GIF ※予備
    figK_total_sideview.png  横から見た 2D 細胞＋線プロファイル（total で着色）
    figK_total_sideview.gif  同、時間変化する GIF
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import brentq

import model as M
from model import Par, variant

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})

CAP, MEM, CYT, ACC, GREY = "#d9e8f5", "#1f4e79", "#c1666b", "#e8a33d", "#8a8a8a"
BAD, TRANS = "#b03a2e", "#7a3b8f"
GEN = 1500.0
LNRHO_BSUB, R_BSUB, L_BSUB = 2.042, 0.425, 2.8
RHO_TR = float(np.exp(LNRHO_BSUB * R_BSUB / Par.R))


def _save(fig, name):
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}", flush=True)


def outline(z, par=Par):
    """z における細胞半径 r(z)。"""
    R, L = par.R, par.L
    r = np.full_like(z, R, dtype=float)
    m = z < R
    r[m] = np.sqrt(np.maximum(R**2 - (z[m] - R)**2, 0))
    m = z > L - R
    r[m] = np.sqrt(np.maximum(R**2 - (z[m] - (L - R))**2, 0))
    return r


# =============================================================== 図A 幾何
def figA():
    R, L = Par.R, Par.L
    Lcyl = L - 2 * R
    g = M.geometry(401)
    A, V = g["A"].sum(), g["V"].sum()
    a = g["A_cap"].sum() / A

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.28)

    # ---- (a) 幾何の定義
    ax = fig.add_subplot(gs[0, :2])
    zf = np.linspace(0, L, 2000)
    rr = outline(zf)
    ax.fill_between(zf, -rr, rr, color="#f4f4f4", lw=0)
    for lo, hi in [(0, R), (L - R, L)]:
        s = (zf >= lo) & (zf <= hi)
        ax.fill_between(zf[s], -rr[s], rr[s], color=CAP, lw=0)
    ax.plot(zf, rr, color="k", lw=1.6)
    ax.plot(zf, -rr, color="k", lw=1.6)
    for x in (R, L - R):
        ax.plot([x, x], [-R, R], color=GREY, lw=1.0, ls="--")

    ax.annotate("", xy=(0, -0.56), xytext=(L, -0.56),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
    ax.text(L / 2, -0.54, f"$L$ = {L} µm", ha="center", fontsize=10, va="bottom")
    ax.annotate("", xy=(R, 0.52), xytext=(L - R, 0.52),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.0))
    ax.text(L / 2, 0.55, f"$L_{{cyl}} = L-2R$ = {Lcyl:.1f} µm",
            ha="center", fontsize=8.5, color=GREY, va="bottom")
    ax.annotate("", xy=(L / 2, 0), xytext=(L / 2, R),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.1))
    ax.text(L / 2 + 0.05, R / 2, f"$R$ = {R} µm", fontsize=10, va="center")

    ax.annotate(r"pole cap:  $H = 1/R$", xy=(0.10, -0.33), xytext=(0.00, -0.78),
                fontsize=9.5, color="#2b6ca3",
                arrowprops=dict(arrowstyle="->", color="#2b6ca3", lw=1.1))
    ax.annotate(r"sidewall:  $H = 1/(2R)$", xy=(2.0, -0.40), xytext=(1.95, -0.78),
                fontsize=9.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))
    ax.set_xlim(-0.12, L + 0.12)
    ax.set_ylim(-1.00, 0.80)
    ax.set_xlabel("z (µm)")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("(a) The cell is a spherocylinder; only $R$ and $L$ are free", pad=8)

    # ---- (b) 導かれる量
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    ax.text(0.0, 1.0, "Derived quantities", fontsize=11, fontweight="bold",
            va="top")
    lines = [
        (r"$A = 2\pi R L_{cyl} + 4\pi R^2$", f"= {A:.3f} µm²", 0.17),
        (r"$V = \pi R^2 L_{cyl} + \frac{4}{3}\pi R^3$", f"= {V:.3f} µm³", 0.19),
        (r"$a = 4\pi R^2 / A = 2R/L$", f"= {a:.4f}", 0.17),
        (r"$1/a = L/2R$", f"= {1/a:.3f}", 0.17),
    ]
    y = 0.88
    for eq, val, step in lines:
        ax.text(0.0, y, eq, fontsize=11.5, va="top")
        ax.text(0.06, y - 0.085, val, fontsize=10, color=MEM, va="top")
        y -= step
    ax.text(0.0, y + 0.01,
            "$a$ is the one geometric number\nthat every conclusion depends on;\n"
            "$1/a$ is the ceiling on enrichment.",
            fontsize=9, color=BAD, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # ---- (c) z で切る vs 弧長で切る
    for col, mode in enumerate(["z", "s"]):
        ax = fig.add_subplot(gs[1, col])
        n = 34
        if mode == "z":
            ze = np.linspace(0, L, n + 1)
            title = "(c) Slicing along the axis $z$  —  correct"
            col_e = MEM
        else:
            # 弧長で等間隔 -> z に写す
            zs = np.linspace(0, L, 4000)
            rs = outline(zs)
            dsdz = np.sqrt(1 + np.gradient(rs, zs) ** 2)
            s = np.concatenate([[0], np.cumsum(np.gradient(zs) * dsdz)[1:]])
            ze = np.interp(np.linspace(0, s[-1], n + 1), s, zs)
            title = "(d) Slicing along the arc length $s$  —  collapses"
            col_e = BAD
        ax.fill_between(zf, -rr, rr, color="#f7f7f7", lw=0)
        ax.plot(zf, rr, color="k", lw=1.2)
        ax.plot(zf, -rr, color="k", lw=1.2)
        for x in ze:
            rx = outline(np.array([x]))[0]
            ax.plot([x, x], [-rx, rx], color=col_e, lw=0.9, alpha=.85)
        ax.set_xlim(-0.08, L + 0.08)
        ax.set_ylim(-0.62, 0.62)
        ax.set_xlabel("z (µm)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=10, color=col_e, pad=6)
        if mode == "s":
            ax.annotate("slices pile up at the tips\n$\\Rightarrow$ their volume collapses",
                        xy=(0.06, 0.30), xytext=(0.70, 0.46), fontsize=8.5,
                        color=BAD,
                        arrowprops=dict(arrowstyle="->", color=BAD, lw=1))
        else:
            ax.text(L / 2, 0.46, "equal $\\Delta z$ everywhere",
                    ha="center", fontsize=8.5, color=GREY)

    # ---- (e) 先端セルの体積のスケーリング
    ax = fig.add_subplot(gs[1, 2])
    ns = np.array([51, 101, 201, 401, 801])
    dz = L / ns
    Vz = np.pi * (R * dz**2 - dz**3 / 3)
    S = np.pi * R + Lcyl
    ds = S / ns
    th = ds / R
    Vs = np.pi * R**3 * (th / 2 - np.sin(2 * th) / 4)
    Vmid = np.pi * R**2 * dz
    ax.loglog(ns, Vmid, "o-", color=GREY, lw=1.4, ms=4, label="mid-cell (either)")
    ax.loglog(ns, Vz, "o-", color=MEM, lw=2.0, ms=5,
              label=r"tip cell, axis $z$  $\propto \Delta z^2$")
    ax.loglog(ns, Vs, "o-", color=BAD, lw=2.0, ms=5,
              label=r"tip cell, arc $s$  $\propto \Delta s^3$")
    for i in (1, 3):
        ax.annotate(f"{Vz[i]/Vs[i]:.0f}×", xy=(ns[i], np.sqrt(Vz[i] * Vs[i])),
                    fontsize=9, color=BAD, ha="center")
    ax.set_xlabel("grid points $n$")
    ax.set_ylabel("cell volume (µm³)")
    ax.set_title("(e) The tip cell collapses one\norder faster with arc length",
                 fontsize=9.5, pad=6)
    ax.legend(fontsize=7.6, frameon=False, loc="lower left")
    ax.grid(alpha=.25, which="both", lw=.5)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns], fontsize=8)
    ax.minorticks_off()

    _save(fig, "figA_geometry.png")


# =============================================================== 図B 場
def figB():
    r = M.solve(401, T=20.0, ic="uniform")
    g = r["g"]
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # 2 つの箱
    ax.add_patch(FancyBboxPatch((0.5, 2.85), 5.4, 1.55,
                                boxstyle="round,pad=0.12", fc="#fbeced",
                                ec=CYT, lw=1.6))
    ax.text(3.2, 4.12, "cytoplasm   $C(z,t)$   [molecules/µm³]",
            ha="center", fontsize=11, fontweight="bold", color="#8f3d43")
    ax.text(3.2, 3.62, r"$\partial_t C = D_c \nabla^2 C \;-\; J\,\dfrac{A}{V}$",
            ha="center", fontsize=13)
    ax.text(3.2, 3.08, f"$D_c$ = {Par.D_c} µm²/s   (E. coli cytoplasm)",
            ha="center", fontsize=8.6, color="#777")

    ax.add_patch(FancyBboxPatch((0.5, 0.45), 5.4, 1.55,
                                boxstyle="round,pad=0.12", fc="#eaf2f9",
                                ec=MEM, lw=1.6))
    ax.text(3.2, 1.72, "membrane monomer   $M(z,t)$   [molecules/µm²]",
            ha="center", fontsize=11, fontweight="bold", color=MEM)
    ax.text(3.2, 1.22, r"$\partial_t M = D_m \nabla^2 M \;+\; J$",
            ha="center", fontsize=13)
    ax.text(3.2, 0.68, f"$D_m$ = {Par.D_m} µm²/s   (SPT, slow population)",
            ha="center", fontsize=8.6, color="#777")

    ax.add_patch(FancyArrowPatch((2.5, 2.80), (2.5, 2.05), arrowstyle="-|>",
                                 mutation_scale=16, lw=2, color=ACC))
    ax.add_patch(FancyArrowPatch((3.9, 2.05), (3.9, 2.80), arrowstyle="-|>",
                                 mutation_scale=16, lw=2, color=GREY))
    ax.text(2.35, 2.42, r"$k_{on}(z)\,C$", ha="right", fontsize=10, color="#a86c14")
    ax.text(4.05, 2.42, r"$k_{off}\,M$", ha="left", fontsize=10, color="#555")
    ax.text(3.2, 2.42, r"$J$", ha="center", fontsize=11, style="italic")

    # 右側: 注記
    notes = [
        (r"$J = k_{on}(z)\,C - k_{off}\,M$   [molecules/(µm²·s)]", "k"),
        ("", "k"),
        (r"$A/V$ appears ONLY in the $C$ equation —", BAD),
        ("     it converts a surface flux into a volume rate.", BAD),
        ("", "k"),
        (r"Curvature enters through $k_{on}(z) = k_{on}^0\,w(z)$,", "k"),
        (r"     $w = \rho$ on the caps and 1 on the sidewall.", "k"),
        ("", "k"),
        (r"$\int k_{on}\,dA = k_{off}V(1-f_{cyt})/f_{cyt}$ is fixed;", MEM),
        (r"     $\rho$ only redistributes it.", MEM),
        ("", "k"),
        ("No cluster field. No synthesis, no dilution:", "#777"),
        ("     the system is closed, so mass conservation", "#777"),
        ("     can be used as the validation backbone.", "#777"),
    ]
    y = 4.35
    for s, c in notes:
        if s:
            ax.text(6.35, y, s, fontsize=9.3, color=c, va="top")
        y -= 0.30

    ax.text(0.5, 4.95, "The two fields of Cycle 1", fontsize=12.5,
            fontweight="bold")
    _save(fig, "figB_fields.png")


# ======================================================= 図C キモグラフ
def figC_kymo():
    T = 20.0
    r = M.solve(201, T=T, ic="cytoplasm", n_out=400)
    g = r["g"]
    z, t = g["z"], r["t"]
    m = t > 1e-4
    R, L = Par.R, Par.L

    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.2),
                           gridspec_kw={"width_ratios": [1, 1, 0.85]})

    pcm = ax[0].pcolormesh(z, t[m], r["Mt"].T[m], shading="auto", cmap="magma")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("z (µm)")
    ax[0].set_ylabel("time (s)")
    ax[0].set_title("(a) membrane  $M(z,t)$")
    fig.colorbar(pcm, ax=ax[0], label="molecules/µm²", pad=0.02)

    pcm = ax[1].pcolormesh(z, t[m], r["Ct"].T[m], shading="auto", cmap="viridis")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("z (µm)")
    ax[1].set_title("(b) cytoplasm  $C(z,t)$")
    fig.colorbar(pcm, ax=ax[1], label="molecules/µm³", pad=0.02)

    for a_ in ax[:2]:
        for x in (R, L - R):
            a_.axvline(x, color="w", lw=0.9, ls="--", alpha=.6)
        a_.axhline(0.94, color="w", lw=1.2, ls="-", alpha=.9)
        a_.text(L / 2, 1.15, "settled (within 1%) at 0.94 s", color="w",
                fontsize=8, ha="center")

    e, tt = r["enr_t"], r["t"]
    ok = np.isfinite(e) & (tt > 0)
    ax[2].plot(tt[ok], e[ok], color=MEM, lw=2.2)
    ax[2].axhline(e[ok][-1], color=GREY, ls=":", lw=1.1)
    ax[2].text(1e-4, e[ok][-1] * 1.02, f"steady {e[ok][-1]:.2f}×",
               fontsize=9, color=GREY)
    ax[2].axhline(1 / (g["A_cap"].sum() / g["A"].sum()), color=BAD, ls="--", lw=1.1)
    ax[2].text(1e-4, 3.80, "ceiling $1/a$ = 3.75", fontsize=9, color=BAD)
    ax[2].axvline(0.94, color=ACC, lw=1.2)
    ax[2].text(1.15, 1.55, "settled\n0.94 s", fontsize=8.5, color="#a86c14")
    ax[2].set_xscale("log")
    ax[2].set_xlim(1e-4, T)
    ax[2].set_ylim(0.9, 4.1)
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("polar enrichment")
    ax[2].set_title("(c) overshoot, then settle")
    ax[2].annotate("local equilibrium first (2.83×)",
                   xy=(2e-3, 2.83), xytext=(1.3e-4, 3.30), fontsize=8.3,
                   color="#555",
                   arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    ax[2].annotate("membrane diffusion overshoots\ndown to 2.02×, then recovers",
                   xy=(0.11, 2.02), xytext=(0.012, 1.15), fontsize=8.3,
                   color="#555",
                   arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    fig.tight_layout()
    _save(fig, "figC_kymograph.png")


# ============================================================ 図C GIF
def figC_gif():
    T = 20.0
    r = M.solve(121, T=T, ic="cytoplasm", n_out=160)
    g = r["g"]
    z, t, Mt = g["z"], r["t"], r["Mt"]
    R, L = Par.R, Par.L

    keep = np.unique(np.searchsorted(
        t, np.logspace(np.log10(max(t[1], 1e-4)), np.log10(T), 48)))
    keep = keep[keep < len(t)]
    vmax = Mt.max() * 1.02
    cmap = plt.get_cmap("magma")

    nth = 44
    th = np.linspace(0, 2 * np.pi, nth)
    rz = outline(z)
    X = np.outer(z, np.ones(nth))
    Y = np.outer(rz, np.cos(th))
    Z = np.outer(rz, np.sin(th))

    fig = plt.figure(figsize=(7.4, 3.1))
    ax = fig.add_axes([-0.04, -0.10, 0.86, 1.14], projection="3d")
    ax.set_box_aspect((L, 2 * R * 1.5, 2 * R * 1.5), zoom=1.18)
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()
    ax.set_xlim(-0.10, L + 0.10)
    ax.set_ylim(-R * 1.25, R * 1.25)
    ax.set_zlim(-R * 1.25, R * 1.25)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(0, vmax))
    cax = fig.add_axes([0.86, 0.22, 0.022, 0.56])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("membrane density $M$  (molecules/µm²)", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    title = fig.text(0.44, 0.92, "", fontsize=11, ha="center",
                     family="monospace")
    surf = [None]

    def draw(k):
        i = keep[k]
        if surf[0] is not None:
            surf[0].remove()
        facec = cmap(np.clip(Mt[:, i] / vmax, 0, 1))
        C = np.repeat(facec[:, None, :], nth, axis=1)
        surf[0] = ax.plot_surface(X, Y, Z, facecolors=C, rstride=1, cstride=1,
                                  linewidth=0, antialiased=False, shade=False)
        e = r["enr_t"][i]
        title.set_text(f"t = {t[i]:8.4f} s     polar enrichment = {e:.2f}x"
                       if np.isfinite(e) else f"t = {t[i]:8.4f} s")
        return (surf[0],)

    anim = FuncAnimation(fig, draw, frames=len(keep), interval=160, blit=False)
    anim.save("figC_cell3d.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)
    print("  -> figC_cell3d.gif", flush=True)


# ====================================================== 図D スキャン
def figD():
    base = M.enrichment()
    specs = [(r"$D_m$   membrane diffusion", "D_m", [0.01, 0.1, 1.0], "×10"),
             (r"$\rho$   curvature preference", "rho",
              [Par.rho / 2, Par.rho, Par.rho * 2], "×2"),
             (r"$D_c$   cytoplasmic diffusion", "D_c", [0.3, 3.0, 30.0], "×10"),
             (r"$k_{off}$   dissociation", "k_off", [3, 9, 27], "×3"),
             (r"$f_{cyt}$   cytoplasmic fraction", "f_cyt",
              [0.157, 0.30, 0.50], "measured range"),
             (r"$N$   expression level", "N", [169, 1690, 169000], "×1000")]
    rows = []
    for lab, name, vals, note in specs:
        e = np.array([M.solve(101, variant(**{name: v}), T=20.0)["enrichment"]
                      for v in vals])
        rows.append((lab, e.min(), e.max(), note, e))
    rows.sort(key=lambda q: q[2] - q[1])

    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    for i, (lab, lo, hi, note, e) in enumerate(rows):
        span = hi - lo
        col = ACC if span > 0.6 else (MEM if span > 0.1 else GREY)
        ax[0].plot([lo, hi], [i, i], color=col, lw=9,
                   solid_capstyle="round", alpha=.85)
        ax[0].plot(base, i, "|", color="k", ms=13, mew=1.6)
        tail = "   ← dead lever" if span < 0.02 else ""
        ax[0].text(hi + 0.05, i, f"{note}{tail}", va="center", fontsize=8.4,
                   color=BAD if tail else GREY)
    ax[0].axvline(base, color="k", lw=1.0, ls=":")
    ax[0].text(base, -0.95, f"baseline {base:.2f}", fontsize=8.5, ha="center")
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([q[0] for q in rows], fontsize=9.5)
    ax[0].set_xlabel("polar enrichment")
    ax[0].set_xlim(1.45, 3.25)
    ax[0].set_ylim(-1.3, len(rows) - 0.4)
    ax[0].set_title("(a) Range of the answer when each parameter is swept", pad=10)

    # 連続スキャン: D_m と rho
    dms = np.logspace(-2, 0, 13)
    e_dm = [M.solve(101, variant(D_m=v), T=20.0)["enrichment"] for v in dms]
    rhos = np.logspace(np.log10(1.5), np.log10(40), 13)
    e_rho = [M.enrichment(v) for v in rhos]
    ns = np.logspace(2, 5.3, 9)
    e_N = [M.solve(101, variant(N=v), T=20.0)["enrichment"] for v in ns]

    ax2 = ax[1]
    ax2.semilogx(dms / Par.D_m, e_dm, "o-", color=ACC, lw=2.2, ms=4.5,
                 label=r"$D_m$ (relative to default)")
    ax2.semilogx(rhos / Par.rho, e_rho, "s-", color=MEM, lw=2.2, ms=4.5,
                 label=r"$\rho$ (relative to default)")
    ax2.semilogx(ns / Par.N, e_N, "^-", color=GREY, lw=2.2, ms=4.5,
                 label=r"$N$ (relative to default)")
    _, a = M.analytic()
    ax2.axhline(1 / a, color=BAD, ls="--", lw=1.2)
    ax2.text(1.2e-2, 1 / a + 0.05, "ceiling $1/a$ = 3.75", fontsize=8.5, color=BAD)
    ax2.axvline(1.0, color="k", lw=0.9, ls=":")
    ax2.set_xlabel("parameter / default value")
    ax2.set_ylabel("polar enrichment")
    ax2.set_ylim(1.0, 4.1)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax2.set_title("(b) $N$ is exactly flat — the model is linear in it", pad=10)

    fig.tight_layout()
    _save(fig, "figD_scan.png")


# ======================================================= 図E エネルギー
def figE():
    e_mono = M.monomer_curvature_energy()
    need_tr = np.log(RHO_TR)
    need_fit = np.log(brentq(lambda v: M.enrichment(v) - 2.2, 1.01, 60, xtol=1e-4))

    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2),
                           gridspec_kw={"width_ratios": [1, 1.15]})

    vals = [e_mono, need_fit, need_tr]
    labs = ["what one monomer\ncan gain from the\ncurvature difference",
            "required by the model\n(fit in $E.\\ coli$ shape)",
            "required by the model\n(transferred from\n$B.\\ subtilis$)"]
    cols = [GREY, ACC, TRANS]
    bars = ax[0].bar(range(3), vals, color=cols, width=.6)
    ax[0].set_yscale("log")
    ax[0].set_ylim(3e-3, 60)
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(labs, fontsize=7.8)
    ax[0].set_ylabel("binding free energy difference  ($k_BT$)")
    for rect, v in zip(bars, vals):
        ax[0].text(rect.get_x() + rect.get_width() / 2, v * 1.3, f"{v:.3g}",
                   ha="center", fontsize=9.5)
    ax[0].annotate("", xy=(1.0, need_fit), xytext=(1.0, e_mono),
                   arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax[0].text(1.08, np.sqrt(need_fit * e_mono),
               f"{need_fit/e_mono:.0f}–{need_tr/e_mono:.0f}×",
               fontsize=14, fontweight="bold")
    ax[0].set_title("(a) Two orders of magnitude short", pad=8)

    lps = np.logspace(np.log10(20), 3, 60)
    dg = np.array([M.monomer_curvature_energy(lp=v) for v in lps])
    ax[1].loglog(lps, need_tr / dg, color=TRANS, lw=2.4,
                 label="transferred ($\\ln\\rho$ = 2.17)")
    ax[1].loglog(lps, need_fit / dg, color=ACC, lw=2.0, ls="--",
                 label="fit in $E.\\ coli$ ($\\ln\\rho$ = 1.79)")
    ax[1].axhline(1, color="k", lw=1.2)
    ax[1].fill_between(lps, 1, 1e4, color="#f6dede", alpha=.45, zorder=0)
    ax[1].text(25, 2.2, "model demands more than a monomer has",
               fontsize=9, color=BAD)
    for v in (100,):
        ax[1].axvline(v, color=GREY, ls=":", lw=1.1)
        ax[1].plot(v, need_tr / M.monomer_curvature_energy(lp=v), "o",
                   color=TRANS, ms=7)
        ax[1].text(v * 1.1, need_tr / M.monomer_curvature_energy(lp=v) * 1.2,
                   f"$\\ell_p$ = {v} nm\n{need_tr/M.monomer_curvature_energy(lp=v):.0f}×",
                   fontsize=9, color=TRANS)
    ax[1].set_xlabel("persistence length $\\ell_p$ (nm)   — not measured for DivIVA")
    ax[1].set_ylabel("required / available  energy")
    ax[1].set_ylim(0.5, 3e3)
    ax[1].legend(fontsize=8.5, frameon=False, loc="upper right")
    ax[1].set_title("(b) The gap survives a decade of uncertainty in $\\ell_p$",
                    pad=8)
    ax[1].grid(alpha=.25, which="both", lw=.5)

    fig.tight_layout()
    _save(fig, "figE_energy.png")


# ============================================================ 図F FRAP
def figF():
    """FRAP。半回復時間は『漂白した極のシグナルが自身の平坦値まで戻る時間』で測る。

    ⚠️ 閉じた系なので、漂白した分子は戻ってこない。極のシグナルは漂白前の
       77.7% までしか回復しない。T1/2 はその平坦値を 1 として測ること。
       （極濃縮の回復時間 95.8 ms は比をとった別の量で、FRAP とは比べられない）
    """
    rb = M.solve(201, T=60.0, ic="bleach", n_out=400)
    g = rb["g"]
    left = g["z"] < Par.R
    sig = (rb["Mt"] * g["A_cap"][:, None])[left].sum(0)
    t = rb["t"]
    s0, sinf = sig[0], sig[-1]
    rec = (sig - s0) / (sinf - s0)
    k = np.argmax(rec >= 0.5)
    t12 = float(np.interp(0.5, [rec[k - 1], rec[k]], [t[k - 1], t[k]]))
    base = M.solve(201, T=5.0, ic="uniform", n_out=3)
    pre = (base["M"] * g["A_cap"])[left].sum()

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.9),
                           gridspec_kw={"width_ratios": [1.15, 1]})

    ok = t > 0
    ax[0].semilogx(t[ok], rec[ok], color=MEM, lw=2.4,
                   label="model (monomer only)")
    for T12, c, lab in [(50.9, GREY, "measured $\\Delta minJ$   51 s"),
                        (128.6, CYT, "measured wild type   129 s")]:
        tg = np.logspace(-4, 4, 300)
        ax[0].semilogx(tg, 1 - np.exp(-np.log(2) * tg / T12), color=c, lw=2.0,
                       ls="--", label=lab)
    ax[0].axvline(t12, color=ACC, lw=1.2)
    ax[0].text(t12 * 1.3, 0.06, f"model $T_{{1/2}}$\n{t12*1000:.0f} ms",
               fontsize=8.5, color=ACC)
    ax[0].axhline(0.5, color="k", lw=0.8, ls=":")
    ax[0].set_xlim(1e-4, 1e4)
    ax[0].set_ylim(0, 1.08)
    ax[0].set_xlabel("time after bleaching (s)")
    ax[0].set_ylabel("recovery  (1 = own plateau)")
    ax[0].legend(fontsize=8.3, frameon=False, loc="upper left")
    ax[0].set_title("(a) Recovery curves do not overlap at all", pad=8)
    ax[0].text(1.5e-4, 0.62,
               f"the cell is closed, so the pole only\n"
               f"returns to {sinf/pre:.0%} of its pre-bleach signal",
               fontsize=7.8, color="#777")

    items = [("model\n(monomer only)", t12, MEM),
             ("measured\n$\\Delta minJ$", 50.9, GREY),
             ("measured\nwild type", 128.6, CYT)]
    for i, (lab, v, c) in enumerate(items):
        ax[1].barh(i, v, color=c, height=.55)
        ax[1].text(v * 1.4, i, f"{v:.3g} s" if v > 1 else f"{v*1000:.0f} ms",
                   va="center", fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlim(0.03, 3e4)
    ax[1].set_yticks(range(3))
    ax[1].set_yticklabels([q[0] for q in items], fontsize=8.5)
    ax[1].set_xlabel("half-recovery time (s)")
    ax[1].annotate("", xy=(128.6, 2.45), xytext=(t12, 2.45),
                   arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax[1].text(np.sqrt(128.6 * t12), 2.62,
               f"{50.9/t12:,.0f}-{128.6/t12:,.0f}$\\times$", ha="center",
               fontsize=11, fontweight="bold")
    ax[1].set_ylim(-0.6, 2.95)
    ax[1].set_title("(b) Two to three orders of magnitude", pad=8)

    fig.tight_layout()
    _save(fig, "figF_frap.png")


# ========================================================= 図G 天井
def figG():
    _, a = M.analytic()
    rhos = np.logspace(0, 3, 26)
    num = np.array([M.enrichment(v) for v in rhos])
    ana = rhos / (rhos * a + 1 - a)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.semilogx(rhos, ana, color=GREY, lw=1.8, ls="--",
                label="local equilibrium (no membrane diffusion)")
    ax.semilogx(rhos, num, color=MEM, lw=2.6, label="full model")
    ax.axhline(1 / a, color=BAD, lw=1.4, ls=":")
    ax.text(1.1, 1 / a + 0.07, f"ceiling  $1/a = L/2R$ = {1/a:.2f}",
            fontsize=9.5, color=BAD)
    ax.axhline(2.2, color=CYT, lw=1.1, ls=":")
    ax.text(1.1, 2.26, "$B.\\ subtilis$ measured 2.2×", fontsize=9, color=CYT)
    e_tr = M.enrichment(RHO_TR)
    ax.plot(RHO_TR, e_tr, "s", color=TRANS, ms=8)
    ax.annotate(f"$\\rho$ = {RHO_TR:.2f}\n$\\Rightarrow$ predicts {e_tr:.2f}×",
                xy=(RHO_TR, e_tr), xytext=(24, 1.75), fontsize=9, color=TRANS,
                arrowprops=dict(arrowstyle="->", color=TRANS, lw=1.1))
    ax.fill_between(rhos, num, ana, color=ACC, alpha=.18)
    ax.annotate("lost to membrane diffusion\n$\\ell_{diff}=\\sqrt{D_m/k_{off}}$ = 0.105 µm",
                xy=(120, (M.enrichment(120) + 120 / (120 * a + 1 - a)) / 2),
                xytext=(3.0, 3.45), fontsize=8.8, color="#a86c14",
                arrowprops=dict(arrowstyle="->", color="#a86c14", lw=1))
    ax.set_xlabel(r"$\rho$   (pole / sidewall binding affinity ratio)")
    ax.set_ylabel("polar enrichment")
    ax.set_ylim(1, 4.1)
    ax.set_xlim(1, 1e3)
    ax.legend(fontsize=8.6, frameon=False, loc="lower right")
    ax.set_title("Even infinite curvature preference cannot beat the geometry",
                 pad=10)
    _save(fig, "figG_ceiling.png")


# ====================================================== 図H 非対称の緩和
def figH():
    t, asym = M.pole_asymmetry_trace(T=20.0, n_out=300)
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.8),
                           gridspec_kw={"width_ratios": [1, 1.05]})

    r = M.solve(201, T=20.0, ic="one_pole", n_out=60)
    g = r["g"]
    idx = [1, 6, 12, 25, 59]
    cols = plt.get_cmap("viridis")(np.linspace(0, .85, len(idx)))
    for c, i in zip(cols, idx):
        ax[0].plot(g["z"], r["Mt"][:, i], color=c, lw=1.8,
                   label=f"t = {r['t'][i]:.3g} s")
    for x in (Par.R, Par.L - Par.R):
        ax[0].axvline(x, color=GREY, lw=0.8, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, 8e3)
    ax[0].set_xlabel("z (µm)")
    ax[0].set_ylabel("M (molecules/µm²)")
    ax[0].legend(fontsize=7.8, frameon=False, loc="upper right")
    ax[0].set_title("(a) Start with everything on the left pole", pad=8)

    ax[1].plot(t, asym, color=ACC, lw=2.4)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xscale("symlog", linthresh=1e-3)
    ax[1].set_xlim(0, 20)
    ax[1].set_ylim(-0.08, 1.08)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("$(M_L-M_R)/(M_L+M_R)$")
    k = np.argmax(np.abs(asym) < 0.01)
    ax[1].plot(t[k], asym[k], "o", color=BAD, ms=7)
    ax[1].annotate(f"below 1% at {t[k]:.1f} s\nfinal {asym[-1]:.0e}\n"
                   "$\\Rightarrow$ the steady state is unique\n     and symmetric",
                   xy=(t[k], asym[k]), xytext=(3e-3, 0.42), fontsize=8.6,
                   color=BAD, arrowprops=dict(arrowstyle="->", color=BAD, lw=1))
    ax[1].set_title("(b) A one-sided bias is not retained", pad=8)

    fig.tight_layout()
    _save(fig, "figH_asymmetry.png")


# =============================================== 図I 膜+細胞質の合計
def figI():
    """顕微鏡が見るのは膜と細胞質の合計。断面を積分した線プロファイルを出す。

    各スライスの分子数は  M*A + C*V  なので、長さあたりでは
        lambda(z) = (M*A + C*V) / dz     [分子/µm]
    極では断面積が小さくなるので、密度が高くても線プロファイルは
    先端で落ちる。Wet の line profile と比べられるのはこちら。
    """
    r = M.solve(401, T=20.0, ic="uniform")
    g = r["g"]
    z, V, A, dz = g["z"], g["V"], g["A"], g["dz"]
    nM, nC = r["M"] * A, r["C"] * V
    cap = (z < Par.R) | (z > Par.L - Par.R)
    a = g["A_cap"].sum() / A.sum()

    # 線プロファイルは細かい格子の上で解析的に作る（セル平均の段差を出さないため）
    #   lambda(z) = C(z)*pi r(z)^2  +  M(z)*2 pi r(z) * ds/dz      [分子/µm]
    #
    # ★ 膜の帯は軸方向に一定である。球冠では r*ds/dz = r*(R/r) = R、
    #   円筒では r*ds/dz = R なので、どちらでも 2*pi*R になる
    #   （アルキメデスの帽子箱定理）。数値微分すると先端で発散するので
    #   この解析値を使うこと。
    zf = np.linspace(0, Par.L, 4000)
    rf = outline(zf)
    Mf = np.interp(zf, z, r["M"])
    Cf = np.interp(zf, z, r["C"])
    lam_c = Cf * np.pi * rf ** 2
    lam_m = Mf * 2 * np.pi * Par.R
    lam = lam_c + lam_m
    midf = int(np.argmin(np.abs(zf - Par.L / 2)))
    mid = int(np.argmin(np.abs(z - Par.L / 2)))

    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9),
                           gridspec_kw={"width_ratios": [1, 1.15, 0.8]})

    # (a) 密度（単位が違うので twin 軸）
    ax[0].axvspan(0, Par.R, color=CAP)
    ax[0].axvspan(Par.L - Par.R, Par.L, color=CAP)
    ax[0].plot(z, r["M"], color=MEM, lw=2.2)
    ax[0].set_ylabel("M  (molecules/µm²)", color=MEM)
    ax[0].tick_params(axis="y", colors=MEM)
    ax[0].set_ylim(0, r["M"].max() * 1.25)
    ax2 = ax[0].twinx()
    ax2.plot(z, r["C"], color=CYT, lw=2.2)
    ax2.set_ylabel("C  (molecules/µm³)", color=CYT)
    ax2.tick_params(axis="y", colors=CYT)
    lo, hi = r["C"].min(), r["C"].max()
    ax2.set_ylim(lo - 2.0 * (hi - lo), hi + 0.3 * (hi - lo))
    ax2.spines["top"].set_visible(False)
    ax[0].set_xlabel("z (µm)")
    ax[0].set_xlim(0, Par.L)
    ax[0].set_title("(a) The two densities\n(different units — cannot be added)",
                    fontsize=10)
    ax[0].plot([], [], color=MEM, lw=2.2, label="membrane $M$  (left axis)")
    ax[0].plot([], [], color=CYT, lw=2.2,
               label="cytoplasm $C$  (right axis)")
    ax[0].legend(fontsize=8.2, frameon=False, loc="upper center")

    # (b) 線プロファイル（積み上げ）
    ax[1].axvspan(0, Par.R, color=CAP, zorder=0)
    ax[1].axvspan(Par.L - Par.R, Par.L, color=CAP, zorder=0)
    ax[1].fill_between(zf, 0, lam_c, color=CYT, alpha=.55, label="cytoplasm")
    ax[1].fill_between(zf, lam_c, lam, color=MEM, alpha=.70, label="membrane")
    ax[1].plot(zf, lam, color="k", lw=2.0, label="total  (what a microscope sees)")
    ax[1].axhline(lam[midf], color=GREY, ls=":", lw=1.1)
    ax[1].annotate(f"peak / mid-cell = {lam.max()/lam[midf]:.2f}",
                   xy=(zf[int(lam.argmax())], lam.max()),
                   xytext=(Par.L * 0.30, lam.max() * 1.07), fontsize=9.5,
                   arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    ax[1].annotate("only the cytoplasmic part tapers:\n"
                   "the membrane band is $2\\pi R$ per µm\n"
                   "of length everywhere (hat-box theorem)",
                   xy=(0.10, lam_c[130] * 0.5), xytext=(0.52, lam.max() * 0.60),
                   fontsize=8.2, color="#555",
                   arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    ax[1].set_xlabel("z (µm)")
    ax[1].set_ylabel(r"molecules per µm of cell length")
    ax[1].set_xlim(0, Par.L)
    ax[1].set_ylim(0, lam.max() * 1.28)
    ax[1].legend(fontsize=8.3, frameon=False, loc="lower center")
    ax[1].set_title("(b) Membrane + cytoplasm, integrated over the cross-section",
                    fontsize=10)

    # (c) 極にいる割合
    fr = [nM[cap].sum() / nM.sum(), nC[cap].sum() / nC.sum(),
          (nM + nC)[cap].sum() / (nM + nC).sum()]
    labs = ["membrane\nonly", "cytoplasm\nonly", "TOTAL"]
    cols = [MEM, CYT, "#333333"]
    bars = ax[2].bar(range(3), fr, color=cols, width=.6)
    ax[2].axhline(a, color=BAD, ls="--", lw=1.3)
    ax[2].text(2.46, a + 0.012, "no enrichment\n($a$ = 0.267)", fontsize=8,
               color=BAD, ha="right")
    for rect, v in zip(bars, fr):
        ax[2].text(rect.get_x() + rect.get_width() / 2, v + 0.015,
                   f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(labs, fontsize=8.8)
    ax[2].set_ylabel("fraction of molecules in the polar caps")
    ax[2].set_ylim(0, 0.78)
    ax[2].set_title("(c) Half of all DivIVA\nsits in the caps", fontsize=10)

    fig.tight_layout()
    _save(fig, "figI_total.png")


# ================================= 図J 断面を切り開いた 3D（膜＋細胞質）
def _cutaway_meshes(z):
    """半分に切り開いた細胞の、外表面と切断面のメッシュを返す。

    外表面は theta を 0..pi の半周だけ描き、切断面（y=0 の平面）を
    露出させる。切断面は z-x 平面上の |x| <= r(z) の領域。
    """
    rz = outline(z)
    th = np.linspace(np.pi, 2 * np.pi, 40)         # 下半分だけ残す
    # ★ 残すのは下半分。上から見下ろしたときに y=0 の切断面が見えるようにする。
    #   上半分を残すと切断面が surface の裏に隠れて何も見えない。
    Xs = np.outer(z, np.ones_like(th))
    Ys = np.outer(rz, np.cos(th))
    Zs = np.outer(rz, np.sin(th))

    u = np.linspace(-1, 1, 24)                     # 切断面（y=0）
    Xc = np.outer(z, np.ones_like(u))
    Yc = np.outer(rz, u)
    Zc = np.zeros_like(Xc)
    return (Xs, Ys, Zs), (Xc, Yc, Zc), th, u


def figJ(static_only=False):
    """断面を切り開いた 3D。外表面の色が膜 M、切断面の色が細胞質 C。

    膜（面密度）と細胞質（体積濃度）は単位が違うので、
    1 本のカラーバーにまとめてはいけない。必ず 2 本に分ける。
    """
    T = 20.0
    r = M.solve(121, T=T, ic="cytoplasm", n_out=160)
    g = r["g"]
    z, t, Mt, Ct = g["z"], r["t"], r["Mt"], r["Ct"]
    R, L = Par.R, Par.L
    (Xs, Ys, Zs), (Xc, Yc, Zc), th, u = _cutaway_meshes(z)

    cmM, cmC = plt.get_cmap("magma"), plt.get_cmap("viridis")
    vmaxM, vmaxC = Mt.max(), Ct.max()

    def build(i, fig, ax, vmaxM=None, vmaxC=None, vminC=0.0):
        vmaxM = vmaxM or Mt.max()
        vmaxC = vmaxC or Ct.max()
        fM = cmM(np.clip(Mt[:, i] / vmaxM, 0, 1))
        fC = cmC(np.clip((Ct[:, i] - vminC) / (vmaxC - vminC), 0, 1))
        a = ax.plot_surface(Xs, Ys, Zs, facecolors=np.repeat(fM[:, None, :],
                            len(th), axis=1), rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
        b = ax.plot_surface(Xc, Yc, Zc, facecolors=np.repeat(fC[:, None, :],
                            len(u), axis=1), rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
        return a, b

    def setup(figsize=(8.0, 3.3), vmaxM=vmaxM, vmaxC=vmaxC, vminC=0.0):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([-0.06, -0.10, 0.80, 1.14], projection="3d")
        ax.set_box_aspect((L, 2 * R * 1.5, 2 * R * 1.5), zoom=1.18)
        ax.view_init(elev=24, azim=-62)
        ax.set_axis_off()
        ax.set_xlim(-0.10, L + 0.10)
        ax.set_ylim(-R * 1.25, R * 1.25)
        ax.set_zlim(-R * 1.25, R * 1.25)
        c1 = fig.add_axes([0.80, 0.56, 0.020, 0.34])
        c2 = fig.add_axes([0.80, 0.12, 0.020, 0.34])
        cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmM,
                           norm=plt.Normalize(0, vmaxM)), cax=c1)
        cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmC,
                           norm=plt.Normalize(vminC, vmaxC)), cax=c2)
        cb1.set_label("membrane $M$\n(molecules/µm²)", fontsize=7.5)
        cb2.set_label("cytoplasm $C$\n(molecules/µm³)", fontsize=7.5)
        for cb in (cb1, cb2):
            cb.ax.tick_params(labelsize=6.5)
        return fig, ax

    # --- 定常状態の静止画（定常の値域に合わせて色を振り直す）
    vM1 = Mt[:, -1].max() * 1.02
    # 細胞質は勾配が 14.8% しかないので、0 から描くと一様に見えてしまう。
    # 定常の静止画だけは自分の値域に拡大する（カラーバーにそう書く）。
    cmin, cmax = Ct[:, -1].min(), Ct[:, -1].max()
    pad = (cmax - cmin) * 0.12
    fig, ax = setup(vmaxM=vM1, vmaxC=cmax + pad, vminC=cmin - pad)
    build(len(t) - 1, fig, ax, vmaxM=vM1, vmaxC=cmax + pad, vminC=cmin - pad)
    fig.text(0.40, 0.93, "Steady state  —  cut open along the long axis",
             fontsize=10.5, ha="center")
    fig.text(0.40, 0.04,
             "surface = membrane monomer,   cut face = cytoplasm",
             fontsize=8.5, ha="center", color="#444")
    fig.text(0.40, -0.01,
             f"the cytoplasm scale is stretched to its own range "
             f"({cmin:.0f}-{cmax:.0f}, a {1-cmin/cmax:.1%} gradient) "
             f"so that it is visible at all",
             fontsize=7.6, ha="center", color="#888")
    _save(fig, "figJ_cell3d_cut.png")
    if static_only:
        return

    # --- GIF
    keep = np.unique(np.searchsorted(
        t, np.logspace(np.log10(max(t[1], 1e-4)), np.log10(T), 48)))
    keep = keep[keep < len(t)]
    fig, ax = setup()
    title = fig.text(0.40, 0.92, "", fontsize=10.5, ha="center",
                     family="monospace")
    arts = []

    def draw(k):
        i = keep[k]
        for a in arts:
            a.remove()
        arts.clear()
        arts.extend(build(i, fig, ax))
        e = r["enr_t"][i]
        title.set_text(f"t = {t[i]:8.4f} s     polar enrichment = {e:.2f}x"
                       if np.isfinite(e) else f"t = {t[i]:8.4f} s")
        return tuple(arts)

    anim = FuncAnimation(fig, draw, frames=len(keep), interval=160, blit=False)
    anim.save("figJ_cell3d_cut.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)
    print("  -> figJ_cell3d_cut.gif", flush=True)


# ============== 図K 横から見た 2D 細胞（total で着色）＋ 線プロファイル
def _lambda_of(Mz, Cz, z, zf, rf):
    """線プロファイル lambda(z) [分子/µm] を細かい格子の上で作る。

        lambda(z) = C(z)*pi r(z)^2  +  M(z)*2 pi R

    膜の帯は軸方向にどこでも 2 pi R で一定（アルキメデスの帽子箱定理）。
    r*ds/dz は球冠で r*(R/r)=R、円筒でも R になるため。
    数値微分で ds/dz を作ると先端で発散するので、この解析値を使うこと。
    """
    return (np.interp(zf, z, Cz) * np.pi * rf ** 2
            + np.interp(zf, z, Mz) * 2 * np.pi * Par.R)


def figK(static_only=False):
    """横から見た 2D の細胞と、その真下に線プロファイルを並べる。

    色は total（膜＋細胞質）だけ。各 z で断面を積分した量なので、
    細胞の幅方向には一様に塗る。**顕微鏡画像のシミュレーションではない**
    （点像分布関数も、視線方向の膜の重なりも入っていない）。
    上の帯の色と下の曲線（total）は同じ数字である。
    下段は細胞質（赤）と膜（青）の積み上げにしてあり、
    発現開始からの時間でこの内訳がひっくり返るのが見える。
    """
    T = 20.0
    r = M.solve(201, T=T, ic="cytoplasm", n_out=200)
    g = r["g"]
    z, t, Mt, Ct = g["z"], r["t"], r["Mt"], r["Ct"]
    R, L = Par.R, Par.L

    zf = np.linspace(0, L, 1200)
    rf = outline(zf)
    # 内訳も持っておく（下段を細胞質／膜で積み上げるため）
    lamC_t = np.stack([np.interp(zf, z, Ct[:, i]) * np.pi * rf ** 2
                       for i in range(len(t))], axis=1)
    lamM_t = np.stack([np.interp(zf, z, Mt[:, i]) * 2 * np.pi * Par.R
                       for i in range(len(t))], axis=1)
    lam_t = lamC_t + lamM_t
    vmax = lam_t.max() * 1.02
    cmap = plt.get_cmap("magma")

    # 細胞の外側をマスクする（シルエットの形に塗るため）
    xg = np.linspace(-R * 1.05, R * 1.05, 160)
    ZZ, XX = np.meshgrid(zf, xg)
    inside = np.abs(XX) <= np.interp(zf, zf, rf)[None, :]

    def setup():
        fig, ax = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.5],
                                            "hspace": 0.12})
        ax[0].set_xlim(0, L)
        ax[0].set_ylim(-R * 1.15, R * 1.15)
        ax[0].set_aspect("equal")
        ax[0].axis("off")
        ax[0].plot(zf, rf, color="k", lw=1.3, zorder=5)
        ax[0].plot(zf, -rf, color="k", lw=1.3, zorder=5)
        ax[1].set_xlim(0, L)
        ax[1].set_ylim(0, vmax)
        ax[1].set_xlabel("z (µm)")
        ax[1].set_ylabel("molecules per µm\nof cell length")
        for x in (R, L - R):
            ax[1].axvline(x, color=GREY, lw=0.8, ls="--")
        # 凡例（内訳の色分け）。中身は paint() が描き替えるので、
        # 凡例用のダミーをここで一度だけ作る。
        ax[1].fill_between([], [], color=CYT, alpha=.60, label="cytoplasm")
        ax[1].fill_between([], [], color=MEM, alpha=.70, label="membrane")
        ax[1].plot([], [], color="k", lw=2.0, label="total")
        ax[1].legend(fontsize=8.2, frameon=False, loc="upper center",
                     ncol=3, bbox_to_anchor=(0.5, 1.02))
        fig.subplots_adjust(bottom=0.20, top=0.90, left=0.11, right=0.855)
        cax = fig.add_axes([0.915, 0.22, 0.020, 0.50])
        cb = fig.colorbar(plt.cm.ScalarMappable(
            cmap=cmap, norm=plt.Normalize(0, vmax)), cax=cax)
        cb.set_label("total  (molecules per µm)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        return fig, ax

    def paint(i, ax, arts):
        for a in arts:
            a.remove()
        arts.clear()
        img = np.where(inside, np.tile(lam_t[:, i], (len(xg), 1)), np.nan)
        arts.append(ax[0].imshow(img, extent=[0, L, -R * 1.05, R * 1.05],
                                 origin="lower", aspect="equal", cmap=cmap,
                                 vmin=0, vmax=vmax, interpolation="bilinear",
                                 zorder=1))
        # 下段は内訳の積み上げ。細胞質を下、膜を上に重ねる。
        arts.append(ax[1].fill_between(zf, 0, lamC_t[:, i], color=CYT,
                                       alpha=.60))
        arts.append(ax[1].fill_between(zf, lamC_t[:, i], lam_t[:, i],
                                       color=MEM, alpha=.70))
        arts.extend(ax[1].plot(zf, lam_t[:, i], color="k", lw=2.0))
        # いまの内訳を数字でも出す（時間でひっくり返るのが要点）
        fm = lamM_t[:, i].sum() / max(lam_t[:, i].sum(), 1e-30)
        arts.append(ax[1].text(0.015, 0.76,
                               f"membrane {fm:5.1%}   cytoplasm {1-fm:5.1%}",
                               transform=ax[1].transAxes, fontsize=8.5,
                               family="monospace", color="#333"))
        return arts

    # --- 定常の静止画
    fig, ax = setup()
    arts = []
    paint(len(t) - 1, ax, arts)
    mid = int(np.argmin(np.abs(zf - L / 2)))
    lam = lam_t[:, -1]
    ax[1].annotate(f"peak / mid-cell = {lam.max()/lam[mid]:.2f}",
                   xy=(zf[int(lam.argmax())], lam.max()),
                   xytext=(L * 0.30, lam.max() * 0.80), fontsize=9.5,
                   arrowprops=dict(arrowstyle="->", color="#555", lw=1))
    fig.text(0.5, 0.955, "Steady state  —  total signal (membrane + cytoplasm)",
             fontsize=11, ha="center")
    fig.text(0.5, 0.035,
             "the colour is the cross-section total at each z, painted "
             "uniformly across the width — not a simulated micrograph",
             fontsize=7.6, ha="center", color="#888")
    _save(fig, "figK_total_sideview.png")
    if static_only:
        return

    # --- GIF
    keep = np.unique(np.searchsorted(
        t, np.logspace(np.log10(max(t[1], 1e-4)), np.log10(T), 48)))
    keep = keep[keep < len(t)]
    fig, ax = setup()
    arts = []
    title = fig.text(0.5, 0.955, "", fontsize=11, ha="center",
                     family="monospace")

    def draw(k):
        i = keep[k]
        paint(i, ax, arts)
        e = r["enr_t"][i]
        title.set_text(f"t = {t[i]:8.4f} s     polar enrichment = {e:.2f}x"
                       if np.isfinite(e) else f"t = {t[i]:8.4f} s")
        return tuple(arts)

    anim = FuncAnimation(fig, draw, frames=len(keep), interval=160, blit=False)
    anim.save("figK_total_sideview.gif", writer=PillowWriter(fps=6), dpi=110)
    plt.close(fig)
    print("  -> figK_total_sideview.gif", flush=True)


ALL = {"A": figA, "B": figB, "C": lambda: (figC_kymo(), figC_gif()),
       "D": figD, "E": figE, "F": figF, "G": figG, "H": figH, "I": figI, "J": figJ, "K": figK}

if __name__ == "__main__":
    want = [s.upper() for s in sys.argv[1:]] or list(ALL)
    for k in want:
        print(f"fig{k} ...", flush=True)
        ALL[k]()
    print("done")
