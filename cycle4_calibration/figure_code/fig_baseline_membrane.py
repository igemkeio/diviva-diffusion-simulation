# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/figure_code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
_os.makedirs(FIGS, exist_ok=True)
if _HERE not in _sys.path: _sys.path.insert(0, _os.path.join(ROOT, 'code'))        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Why a free fluorescent protein is the wrong baseline for a membrane-bound protein.

Both species are perfectly uniform - nothing is localised anywhere.  They differ only in
whether the molecules fill the volume or sit on the surface.
Convention: line width 5 px (matching anl/final), tip window z 0-R/L = 0.1441 (cap boundary),
L = 2.7767 um, PSF 0.870 um.
"""
import sys, csv
import numpy as np, forward as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, L, W, PSF = 0.40, 2.7767, 5, 0.870
WIDE = _os.path.join(DATA,'cells_final_w5.csv')
OUT = _os.path.join(FIGS,'cal_fig_baseline.png')
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8882'
C0, C50, C100 = '#2a78d6', '#1baf7a', '#eb6834'

Lc = L - 2*R
V = np.pi*R*R*Lc + 4/3*np.pi*R**3
A = 2*np.pi*R*Lc + 4*np.pi*R*R
zm = np.linspace(0, L, 2001); one = np.ones_like(zm)

def prof(f):
    pr = F.line_profile(C=(1-f)/V*one, M=f/A*one, z_model=zm, R=R, L=L,
                        psf_fwhm_um=PSF, line_w_px=W)
    v, z = pr['val'], pr['z']
    k = max(2, int(round(len(v)*F.TAIL)))
    S = np.clip(v - np.median(np.r_[v[:k], v[-k:]]), 0, None)
    return z, S/np.median(S[(z >= .35) & (z <= .65)])

def em(z, S):
    e = np.r_[S[(z >= 0) & (z <= .1441)], S[(z >= .8559) & (z <= 1)]]
    return e.mean()/S[(z >= .40) & (z <= .60)].mean()

# --- measured null (same line width, w5)
rows = list(csv.DictReader(open(WIDE, encoding='utf-8')))
pc = [c for c in rows[0] if c.startswith('div_z')]
zs = np.array([float(c[5:]) for c in pc])
def flt(v):
    try: return float(v)
    except: return np.nan
sel = [r for r in rows if r['condition'] == 'mCherry_null']
P = np.array([[flt(r[c]) for c in pc] for r in sel])
P = P[np.isfinite(P).all(axis=1)]
n0 = np.median(P[:, (zs >= .35) & (zs <= .65)], axis=1)
P = P[n0 > 0]/n0[n0 > 0][:, None]
m_null = np.median(P, axis=0); n_null = len(P)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=200,
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
fig.patch.set_facecolor(SURF)

# ---------- (a) the bare projection weights, no blur ----------
a1.set_facecolor(SURF)
zz = np.linspace(0, L, 1200)
r, dr = F.radius(zz, R, L)
vol = 2*r                                  # volume-distributed: chord length
sur = 2*np.sqrt(1 + dr**2)                 # surface-distributed: 2*sigma*R/r(z)
mid = (zz > .4*L) & (zz < .6*L)
a1.axvspan(0, .1441, color='#efeee9', zorder=0); a1.axvspan(.8559, 1, color='#efeee9', zorder=0)
a1.plot(zz/L, vol/vol[mid].mean(), color=C0, lw=2.4, label='volume  $\\propto 2r(z)$')
a1.plot(zz/L, np.clip(sur/sur[mid].mean(), 0, 4), color=C100, lw=2.4,
        label='surface  $\\propto 2\\sigma R/r(z)$')
a1.axhline(1, color='#c9c8c2', lw=1.0, zorder=1)
a1.set_ylim(0, 2.6); a1.set_xlim(0, 1)
a1.set_xlabel('position along the cell  (z / L)', fontsize=9.5, color=INK2)
a1.set_ylabel('projected weight / mid-cell', fontsize=9.5, color=INK2)
a1.set_title('(a) why the two differ — before any blur', fontsize=10.5, color=INK, loc='left', pad=8)
a1.legend(frameon=False, fontsize=9, loc='upper center', labelcolor=INK2, handlelength=1.8)
a1.annotate('toward the pole the cell thins,\nso the volume term falls', (.07, .45),
            fontsize=8.3, color=C0, ha='left')
a1.annotate('…while the membrane tilts into\nthe line of sight, so it rises', (.30, 1.75),
            fontsize=8.3, color=C100, ha='left')

# ---------- (b) the resulting profiles ----------
a2.set_facecolor(SURF)
a2.axvspan(0, .1441, color='#efeee9', zorder=0); a2.axvspan(.8559, 1, color='#efeee9', zorder=0)
a2.plot(zs, m_null, color=MUTED, lw=1.7, ls=(0, (4, 2)), zorder=2,
        label='measured free mCherry (n=%d)' % n_null)
for f, c, lab in ((0.0, C0, 'membrane 0%  (= free FP)'),
                  (0.5, C50, 'membrane 50%'),
                  (1.0, C100, 'membrane 100%')):
    z, S = prof(f)
    a2.plot(z, S, color=c, lw=2.3, zorder=3, label='%s   end/mid %.3f' % (lab, em(z, S)))
a2.set_xlim(-.03, 1.03); a2.set_ylim(0, 1.15)
a2.set_xlabel('position along the cell  (z / L)', fontsize=9.5, color=INK2)
a2.set_ylabel('intensity / mid-cell', fontsize=9.5, color=INK2)
a2.set_title('(b) none of these is localised — yet the tips differ',
             fontsize=10.5, color=INK, loc='left', pad=8)
a2.legend(frameon=False, fontsize=8.7, loc='lower center', labelcolor=INK2, handlelength=1.8)
a2.text(.072, 1.07, 'tip window', fontsize=8, color=MUTED, ha='center')

for ax in (a1, a2):
    ax.tick_params(colors=INK2, labelsize=9)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color('#d6d5cf')

fig.text(.012, -.045,
         'Line width 5 px, tip window z 0–0.144 (= R/L, the cap), cell 2.78 × 0.80 µm, effective PSF 0.870 µm. '
         'Panel (a) is the central-line projection weight with no blur; panel (b) is the full '
         'forward model (projection → PSF → pixels), each curve normalised to mid-cell.',
         fontsize=8.2, color=MUTED, ha='left', va='top')
plt.tight_layout()
plt.savefig(OUT, facecolor=SURF, bbox_inches='tight')
print('wrote', OUT)
for f in (0.0, 0.5, 1.0):
    z, S = prof(f); print('  membrane %3.0f%% -> end/mid %.4f' % (100*f, em(z, S)))
