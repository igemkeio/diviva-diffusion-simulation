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
"""What a UNIFORM membrane distribution looks like when the cytoplasmic null is used as the baseline.

Both species are uniform (no localisation at all).  The only difference is where the
molecules sit: throughout the volume (free FP) vs on the surface (membrane-bound).
"""
import sys, csv
import numpy as np, forward as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, L, W = 0.40, 2.7767, 5
PSF_MEAS, PSF_IDEAL = 0.870, 0.0
FIN = _os.path.join(DATA,'cells_final_w5.csv')
OUT = _os.path.join(FIGS,'cal_fig7_membrane_baseline.png')
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8882'
C_CYT, C_MEM, C_MIX = '#2a78d6', '#eb6834', '#1baf7a'

Lc = L - 2*R
V = np.pi*R*R*Lc + 4/3*np.pi*R**3
A = 2*np.pi*R*Lc + 4*np.pi*R*R
zm = np.linspace(0, L, 2001); one = np.ones_like(zm)

def prof(kind, psf, w=W, f=None):
    kw = dict(C=one/V) if kind == 'cyt' else \
         dict(M=one/A) if kind == 'mem' else \
         dict(C=(1-f)*one/V, M=f*one/A)
    pr = F.line_profile(z_model=zm, R=R, L=L, psf_fwhm_um=psf, line_w_px=w, **kw)
    v, z = pr['val'], pr['z']
    k = max(2, int(round(len(v)*F.TAIL)))
    S = np.clip(v - np.median(np.r_[v[:k], v[-k:]]), 0, None)
    return z, S/np.median(S[(z >= .35) & (z <= .65)])

def em(z, S, hi=.20):
    return np.r_[S[(z >= 0) & (z <= hi)], S[(z >= 1-hi) & (z <= 1)]].mean() \
         / S[(z >= .40) & (z <= .60)].mean()

z, Sc = prof('cyt', PSF_MEAS)
_, Sm = prof('mem', PSF_MEAS)
_, Sx = prof('mix', PSF_MEAS, f=0.70)
z0, Sc0 = prof('cyt', PSF_IDEAL, w=3)
_,  Sm0 = prof('mem', PSF_IDEAL, w=3)

rows = list(csv.DictReader(open(FIN, encoding='utf-8')))
cols = [c for c in rows[0] if c.startswith('div_z')]
zs = np.array([float(c[5:]) for c in cols])
def fl(v):
    try: return float(v)
    except: return np.nan
def meas(cond):
    sel = [r for r in rows if r['condition'] == cond]
    P = np.array([[fl(r[c]) for c in cols] for r in sel])
    P = P[np.isfinite(P).all(axis=1)]
    n = np.median(P[:, (zs >= .35) & (zs <= .65)], axis=1)
    P = P[n > 0]/n[n > 0][:, None]
    return np.median(P, axis=0), len(P)
m_null, n_null = meas('mCherry_null')
m_div,  n_div  = meas('DivIVA_mCherry')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=200)
fig.patch.set_facecolor(SURF)
for ax in (a1, a2):
    ax.set_facecolor(SURF)
    ax.axvspan(0, .20, color='#efeee9', zorder=0)
    ax.axvspan(.80, 1., color='#efeee9', zorder=0)
    ax.set_xlim(-.03, 1.03); ax.tick_params(colors=INK2, labelsize=9)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color('#d6d5cf')
    ax.set_xlabel('position along the cell  (z / L)', fontsize=9.5, color=INK2)

a1.plot(zs, m_null, color=MUTED, lw=1.6, ls=(0, (4, 2)), zorder=2,
        label='measured free mCherry (n=%d)' % n_null)
a1.plot(z, Sc, color=C_CYT, lw=2.2, zorder=3, label='uniform cytoplasm')
a1.plot(z, Sx, color=C_MIX, lw=2.2, zorder=3, label='uniform, 70% membrane')
a1.plot(z, Sm, color=C_MEM, lw=2.2, zorder=3, label='uniform membrane')
a1.set_ylim(0, 1.12); a1.set_ylabel('intensity / mid-cell', fontsize=9.5, color=INK2)
a1.set_title('Two uniform distributions are not the same profile', fontsize=10.5,
             color=INK, loc='left', pad=8)
a1.legend(frameon=False, fontsize=8.6, loc='lower center', labelcolor=INK2, handlelength=1.8)
a1.annotate('end/mid %.3f' % em(z, Sm), (0.025, Sm[np.argmin(abs(z-0.025))]),
            textcoords='offset points', xytext=(9, 13), fontsize=8.4, color=C_MEM)
a1.annotate('end/mid %.3f' % em(z, Sc), (0.025, Sc[np.argmin(abs(z-0.025))]),
            textcoords='offset points', xytext=(9, -17), fontsize=8.4, color=C_CYT)
a1.annotate('%.2fx apart, with no localisation anywhere'
            % (em(z, Sm)/em(z, Sc)), (0.50, 0.17), ha='center',
            fontsize=8.8, color=INK2)

rat = np.where(Sc > .15, Sm/np.maximum(Sc, 1e-9), np.nan)
ratx = np.where(Sc > .15, Sx/np.maximum(Sc, 1e-9), np.nan)
rat0 = np.where(Sc0 > .15, Sm0/np.maximum(Sc0, 1e-9), np.nan)
a2.axhline(1.0, color='#c9c8c2', lw=1.1, zorder=1)
a2.plot(z0, rat0, color=C_MEM, lw=1.5, ls=(0, (3, 2)), zorder=3,
        label='uniform membrane, no blur (PSF 0)')
a2.plot(z, rat, color=C_MEM, lw=2.4, zorder=4, label='uniform membrane, PSF 0.87 µm')
a2.plot(z, ratx, color=C_MIX, lw=2.2, zorder=4, label='uniform, 70% membrane, PSF 0.87 µm')
rm = np.where(m_null > .15, m_div/np.maximum(m_null, 1e-9), np.nan)
a2.plot(zs, rm, color=INK2, lw=1.8, ls=(0, (1, 1.6)), zorder=5,
        label='measured DivIVA / measured null')
a2.set_ylim(.8, 2.6); a2.set_ylabel('apparent enrichment\n(÷ the cytoplasmic reference)',
                                    fontsize=9.5, color=INK2)
a2.set_title('Using a cytoplasmic null as the baseline invents enrichment',
             fontsize=10.5, color=INK, loc='left', pad=8)
a2.legend(frameon=False, fontsize=8.4, loc='upper center', labelcolor=INK2, handlelength=2.0)
a2.text(.10, .84, 'pole window', fontsize=8, color=MUTED, ha='center')

fig.text(.012, -.055,
         'Both simulated species are perfectly uniform — neither is localised anywhere. They differ only in '
         'where the molecules sit:\nthroughout the volume (cytoplasm) or on the surface (membrane). '
         'Line width 5 px, cell 2.78 × 0.80 µm.  '
         'End/mid of the measured pair: DivIVA 0.803, null 0.578 (ratio 1.39).',
         fontsize=8.3, color=MUTED, ha='left', va='top')
plt.tight_layout()
plt.savefig(OUT, facecolor=SURF, bbox_inches='tight')
print('wrote', OUT)
print('  end/mid  cyt %.4f | mix70 %.4f (%.2fx) | mem %.4f (%.2fx)'
      % (em(z, Sc), em(z, Sx), em(z, Sx)/em(z, Sc), em(z, Sm), em(z, Sm)/em(z, Sc)))
