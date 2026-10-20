"""Cycle 3: S1 空間占有で左右比 5 倍になる条件（theta = 0.659）の時間発展。

Cycle 1 の figK / Cycle 2 の c2figK と同じ体裁・配色。
PopZ は最初から左極を覆っている。DivIVA の発現開始（全分子が細胞質）から追う。
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


import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import popz as P
from popz import Par

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})
CAP, MEM, CYT, CLU, GREY = "#d9e8f5", "#1f4e79", "#c1666b", "#e8a33d", "#8a8a8a"
POPZ = "#7a3b8f"
R, L = Par.R, Par.L
NG, T, THETA = 201, 4000.0, 0.659

def outline(zz):
    rr = np.full_like(zz, R)
    m = zz < R;     rr[m] = np.sqrt(np.maximum(R**2 - (zz[m] - R)**2, 0))
    m = zz > L - R; rr[m] = np.sqrt(np.maximum(R**2 - (zz[m] - (L - R))**2, 0))
    return rr

# PopZ は最初から居る。全分子が細胞質の状態（発現開始）から追う。
par = P.variant(theta=THETA)
r = P.solve(NG, par=par, T=T, ic="cytoplasm", n_out=200)
g, z, t, dz = r["g"], r["g"]["z"], r["t"], r["g"]["dz"]

zf = np.linspace(0, L, 1200)
rf = outline(zf)
lamC = np.stack([np.interp(zf, z, r["Ct"][:, i] * g["V"] / dz) for i in range(len(t))], 1)
lamM = np.stack([np.interp(zf, z, r["Mt"][:, i] * g["A"] / dz) for i in range(len(t))], 1)
lamN = np.stack([np.interp(zf, z, r["Nt"][:, i] * g["A"] / dz) for i in range(len(t))], 1)
lam = lamC + lamM + lamN
vmax = lam.max() * 1.02
ytop = lam.max() * 1.20
cmap = plt.get_cmap("magma")

uf = np.interp(zf, z, r["u"])            # 占有の強さ theta*phi(z)
zpz = zf[uf > 0.5 * max(uf.max(), 1e-12)]
pz_lo, pz_hi = (zpz.min(), zpz.max()) if len(zpz) else (0.0, 0.0)

xg = np.linspace(-R * 1.05, R * 1.05, 160)
ZZ, XX = np.meshgrid(zf, xg)
inside = np.abs(XX) <= rf[None, :]

def setup():
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 5.0), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})
    ax[0].set_xlim(0, L); ax[0].set_ylim(-R * 1.45, R * 1.15)
    ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[0].plot(zf, rf, color="k", lw=1.3, zorder=5)
    ax[0].plot(zf, -rf, color="k", lw=1.3, zorder=5)
    # PopZ が覆っている範囲
    ax[0].plot([pz_lo, pz_hi], [-R * 1.28] * 2, color=POPZ, lw=5,
               solid_capstyle="butt", zorder=6)
    ax[0].text(pz_hi + 0.04, -R * 1.28, "PopZ  ($\\theta$ = %.2f)" % THETA,
               color=POPZ, fontsize=8.5, va="center")
    ax[1].set_xlim(0, L); ax[1].set_ylim(0, ytop)
    ax[1].set_xlabel("z (µm)")
    ax[1].set_ylabel("molecules per µm\nof cell length")
    ax[1].axvspan(pz_lo, pz_hi, color=POPZ, alpha=0.10, lw=0, zorder=0)
    for x in (R, L - R):
        ax[1].axvline(x, color=GREY, lw=0.8, ls="--")
    ax[1].fill_between([], [], color=CYT, alpha=.60, label="cytoplasm")
    ax[1].fill_between([], [], color=MEM, alpha=.70, label="membrane monomer")
    ax[1].fill_between([], [], color=CLU, alpha=.85, label="cluster")
    ax[1].plot([], [], color="k", lw=2.0, label="total")
    ax[1].legend(fontsize=8.2, frameon=False, loc="upper center", ncol=4,
                 bbox_to_anchor=(0.5, 1.13))
    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.11, right=0.855)
    cax = fig.add_axes([0.915, 0.22, 0.020, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax)),
                      cax=cax)
    cb.set_label("total  (molecules per µm)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig, ax

def paint(i, ax, arts):
    for a in arts:
        a.remove()
    arts.clear()
    img = np.where(inside, np.tile(lam[:, i], (len(xg), 1)), np.nan)
    arts.append(ax[0].imshow(img, extent=[0, L, -R * 1.05, R * 1.05],
                             origin="lower", aspect="equal", cmap=cmap,
                             vmin=0, vmax=vmax, interpolation="bilinear", zorder=1))
    arts.append(ax[1].fill_between(zf, 0, lamC[:, i], color=CYT, alpha=.60))
    arts.append(ax[1].fill_between(zf, lamC[:, i], lamC[:, i] + lamM[:, i],
                                   color=MEM, alpha=.70))
    arts.append(ax[1].fill_between(zf, lamC[:, i] + lamM[:, i], lam[:, i],
                                   color=CLU, alpha=.85))
    arts.extend(ax[1].plot(zf, lam[:, i], color="k", lw=2.0))
    nL_ = (r["Nt"][:, i] * g["A_capL"]).sum()
    nR_ = (r["Nt"][:, i] * g["A_capR"]).sum()
    arts.append(ax[1].text(0.015, 0.055,
                           "scaffold   blocked pole %6.1f    free pole %6.1f  molecules"
                           % (nL_, nR_), transform=ax[1].transAxes,
                           fontsize=8.2, family="monospace", color="#333"))
    return arts

# ---- 最終状態の静止画 ----
fig, ax = setup(); arts = []
paint(len(t) - 1, ax, arts)
fig.text(0.5, 0.955, "S1 spatial occupancy  —  steady state at  $\\theta$ = %.2f "
         "(free / blocked pole = %.1f×)" % (THETA, r["lr_t"][-1]),
         fontsize=11, ha="center")
fig.text(0.5, 0.035, "the colour is the cross-section total at each z, painted "
         "uniformly across the width — not a simulated micrograph",
         fontsize=7.6, ha="center", color="#888")
fig.savefig("c3figK_occupancy.png", bbox_inches="tight"); plt.close(fig)
print("  -> c3figK_occupancy.png")

# ---- GIF ----
keep = np.unique(np.searchsorted(t, np.logspace(np.log10(max(t[1], 1.0)),
                                                np.log10(T), 56)))
keep = keep[keep < len(t)]
keep = np.concatenate(([0], keep))
fig, ax = setup(); arts = []
title = fig.text(0.5, 0.955, "", fontsize=11, ha="center", family="monospace")
fig.text(0.5, 0.035, "the colour is the cross-section total at each z, painted "
         "uniformly across the width — not a simulated micrograph",
         fontsize=7.6, ha="center", color="#888")

def draw(k):
    i = keep[k]
    paint(i, ax, arts)
    nR_ = (r["Nt"][:, i] * g["A_capR"]).sum()
    if nR_ > 5.0:
        title.set_text("t = %7.1f s    free / blocked pole = %5.2fx"
                       % (t[i], r["lr_t"][i]))
    else:
        title.set_text("t = %7.1f s    (clusters not formed yet)" % t[i])
    return arts

anim = FuncAnimation(fig, draw, frames=len(keep), blit=False)
anim.save("c3figK_occupancy.gif", writer=PillowWriter(fps=6))
plt.close(fig)
print("  -> c3figK_occupancy.gif (%d frames)  final ratio %.2f  total/N %.4f"
      % (len(keep), r["lr_t"][-1], r["mass_frac"]))
