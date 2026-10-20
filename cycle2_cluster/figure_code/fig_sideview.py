"""Cycle 2: side view (total) + 3-way stacked breakdown. PNG and GIF.

Cycle 1 の figK_total_sideview と同じ体裁・配色。上段の帯は total を
magma で塗り、下段は細胞質（赤）/ 膜上単量体（紺）/ クラスター（琥珀）の
積み上げ。クラスターの層が育つのが Cycle 1 との違いである。
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
import cluster as K
from cluster import Par

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
})
CAP, MEM, CYT, CLU, GREY = "#d9e8f5", "#1f4e79", "#c1666b", "#e8a33d", "#8a8a8a"
R, L = Par.R, Par.L
T, NGRID, NOUT = 4000.0, 201, 200

def outline(zz):
    rr = np.full_like(zz, R)
    m = zz < R;     rr[m] = np.sqrt(np.maximum(R**2 - (zz[m] - R)**2, 0))
    m = zz > L - R; rr[m] = np.sqrt(np.maximum(R**2 - (zz[m] - (L - R))**2, 0))
    return rr

r = K.solve(NGRID, T=T, ic="cytoplasm", n_out=NOUT)
g, z, t = r["g"], r["g"]["z"], r["t"]
dz = g["dz"]
zf = np.linspace(0, L, 1200)
rf = outline(zf)
# 線密度 [分子/µm]: 細胞質は断面積、膜は帯 2πR（キャップでも一定）
lamC = np.stack([np.interp(zf, z, r["Ct"][:, i] * g["V"] / dz)
                 for i in range(len(t))], axis=1)
lamM = np.stack([np.interp(zf, z, r["Mt"][:, i] * g["A"] / dz)
                 for i in range(len(t))], axis=1)
lamN = np.stack([np.interp(zf, z, r["Nt"][:, i] * g["A"] / dz)
                 for i in range(len(t))], axis=1)
lam = lamC + lamM + lamN
vmax = lam.max() * 1.02
ytop = lam.max() * 1.20
cmap = plt.get_cmap("magma")

xg = np.linspace(-R * 1.05, R * 1.05, 160)
ZZ, XX = np.meshgrid(zf, xg)
inside = np.abs(XX) <= rf[None, :]

def setup():
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 4.8), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.5],
                                        "hspace": 0.12})
    ax[0].set_xlim(0, L); ax[0].set_ylim(-R * 1.15, R * 1.15)
    ax[0].set_aspect("equal"); ax[0].axis("off")
    ax[0].plot(zf, rf, color="k", lw=1.3, zorder=5)
    ax[0].plot(zf, -rf, color="k", lw=1.3, zorder=5)
    ax[1].set_xlim(0, L); ax[1].set_ylim(0, ytop)
    ax[1].set_xlabel("z (µm)")
    ax[1].set_ylabel("molecules per µm\nof cell length")
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
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap,
                                            norm=plt.Normalize(0, vmax)), cax=cax)
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
                             vmin=0, vmax=vmax, interpolation="bilinear",
                             zorder=1))
    arts.append(ax[1].fill_between(zf, 0, lamC[:, i], color=CYT, alpha=.60))
    arts.append(ax[1].fill_between(zf, lamC[:, i], lamC[:, i] + lamM[:, i],
                                   color=MEM, alpha=.70))
    arts.append(ax[1].fill_between(zf, lamC[:, i] + lamM[:, i], lam[:, i],
                                   color=CLU, alpha=.85))
    arts.extend(ax[1].plot(zf, lam[:, i], color="k", lw=2.0))
    tot = max(lam[:, i].sum(), 1e-30)
    arts.append(ax[1].text(0.015, 0.055,
                           "cytoplasm %5.1f%%   monomer %5.1f%%   cluster %5.1f%%"
                           % (100 * lamC[:, i].sum() / tot,
                              100 * lamM[:, i].sum() / tot,
                              100 * lamN[:, i].sum() / tot),
                           transform=ax[1].transAxes, fontsize=8.2,
                           family="monospace", color="#333"))
    return arts

# ---- 定常の静止画 ----
fig, ax = setup(); arts = []
paint(len(t) - 1, ax, arts)
mid = int(np.argmin(np.abs(zf - L / 2)))
lamS = lam[:, -1]
ax[1].annotate("peak / mid-cell = %.2f" % (lamS.max() / lamS[mid]),
               xy=(zf[int(lamS.argmax())], lamS.max()),
               xytext=(L * 0.34, lamS.max() * 0.86), fontsize=9.5,
               arrowprops=dict(arrowstyle="->", color="#555", lw=1))
fig.text(0.5, 0.955, "Steady state  —  total signal (cytoplasm + membrane)",
         fontsize=11, ha="center")
fig.text(0.5, 0.035,
         "the colour is the cross-section total at each z, painted uniformly "
         "across the width — not a simulated micrograph", fontsize=7.6,
         ha="center", color="#888")
fig.savefig("c2figK_total_sideview.png", bbox_inches="tight")
plt.close(fig)
print("  -> c2figK_total_sideview.png")

# ---- GIF ----
keep = np.unique(np.searchsorted(
    t, np.logspace(np.log10(max(t[1], 1e-2)), np.log10(T), 56)))
keep = keep[keep < len(t)]
fig, ax = setup(); arts = []
title = fig.text(0.5, 0.955, "", fontsize=11, ha="center", family="monospace")
fig.text(0.5, 0.035,
         "the colour is the cross-section total at each z, painted uniformly "
         "across the width — not a simulated micrograph", fontsize=7.6,
         ha="center", color="#888")

def draw(k):
    i = keep[k]
    paint(i, ax, arts)
    e = r["enr_clu_t"][i]
    title.set_text("t = %7.1f s   cluster enrichment = %.2fx" % (t[i], e)
                   if np.isfinite(e) else "t = %7.1f s" % t[i])
    return arts

anim = FuncAnimation(fig, draw, frames=len(keep), blit=False)
anim.save("c2figK_total_sideview.gif", writer=PillowWriter(fps=6))
plt.close(fig)
print("  -> c2figK_total_sideview.gif  (%d frames)" % len(keep))
