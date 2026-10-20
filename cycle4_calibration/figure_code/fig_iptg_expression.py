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
"""IPTG dose vs DivIVA-mCherry expression (peak fluorescence above background).

Valid as a cross-field comparison because all 5 IPTG fields were acquired with
identical settings: 2000 ms exposure, 560 nm, 100% LED, same day within 25 min.
Source: anl/selection/pool_features.csv (pre-selection population + `selected` flag).
"""
import csv, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = _os.path.join(DATA,'pool_features.csv')
OUT = _os.path.join(FIGS,'cal_fig5_iptg_expression.png')

# validated categorical slots 1 and 2 (dataviz reference palette, light mode)
C_POP, C_SEL = '#2a78d6', '#eb6834'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8882'
SURF = '#fcfcfb'

def fl(v):
    try: return float(v)
    except: return np.nan

rows = [r for r in csv.DictReader(open(SRC, encoding='utf-8'))
        if r['condition'] == 'DivIVA_mCherry']
doses, pop, sel = [], [], []
for rp in '12345':
    sub = [r for r in rows if r['rep'] == rp]
    a = np.array([fl(r['fl_peak']) for r in sub])
    a = a[np.isfinite(a)]
    b = np.array([fl(r['fl_peak']) for r in sub
                  if r['selected'].strip().lower() in ('true', '1')])
    b = b[np.isfinite(b)]
    doses.append(10.0 ** (int(rp) - 1)); pop.append(a); sel.append(b)
doses = np.array(doses)

fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

def draw(xs, data, color, label, off):
    x = xs * (10.0 ** off)
    med = np.array([np.median(d) for d in data])
    for xi, d in zip(x, data):
        lo, hi = np.percentile(d, [10, 90])
        q1, q3 = np.percentile(d, [25, 75])
        ax.plot([xi, xi], [lo, hi], color=color, lw=1.0, alpha=.55,
                solid_capstyle='round', zorder=2)
        ax.plot([xi, xi], [q1, q3], color=color, lw=3.2, alpha=.85,
                solid_capstyle='round', zorder=3)
    ax.plot(x, med, color=color, lw=2.0, zorder=4, label=label)
    ax.plot(x, med, 'o', ms=8, color=color, mec=SURF, mew=2.0, zorder=5)
    return x, med

ax.grid(axis='y', color='#e3e2dd', lw=.8, zorder=0)
ax.set_axisbelow(True)
xp, mp = draw(doses, pop, C_POP, 'All cells detected (population)', -0.035)
xs_, ms_ = draw(doses, sel, C_SEL, 'Hand-selected cells (used in the analysis)', +0.035)

# selective direct labels: the anchor and the maximum of the population curve
imax = int(np.argmax(mp))
for i, off in ((0, (-12, 1)), (imax, (-13, -17))):
    ax.annotate('%.0f' % mp[i], (xp[i], mp[i]), textcoords='offset points',
                xytext=off, ha='right', fontsize=9, color=INK2)
ax.annotate('peak at %s nM, not at the\nhighest dose (%.2fx overall)'
            % (('%.0f' % doses[imax]), mp.max()/mp.min()),
            (xp[imax], mp[imax]), textcoords='axes fraction', xytext=(0.47, 0.04),
            fontsize=9, color=INK2, ha='left',
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=.9,
                            connectionstyle='arc3,rad=-.3'))

ax.set_yscale('log')
ax.set_yticks([100, 150, 200, 300, 400, 600, 800])
ax.set_yticklabels(['100', '150', '200', '300', '400', '600', '800'])
ax.minorticks_off()
ax.set_xscale('log')
ax.set_xticks(doses)
ax.set_xticklabels(['1', '10', '100', '1000', '10000'])
ax.set_xlim(doses[0] / 2.6, doses[-1] * 2.6)
ax.set_xlabel('IPTG (nM, log scale)', color=INK2, fontsize=10)
ax.set_ylabel('Peak fluorescence above background (ADU)', color=INK2, fontsize=10)
ax.set_title('DivIVA-mCherry expression vs IPTG  —  comparable across fields\n'
             '(identical settings: 2000 ms, 560 nm, 100% LED, same day)',
             color=INK, fontsize=11.5, loc='left', pad=12)
ax.tick_params(colors=INK2, labelsize=9.5)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'):
    ax.spines[sp].set_color('#d6d5cf')

ns = '   '.join('%d/%d' % (len(s), len(p)) for s, p in zip(sel, pop))
ax.annotate('bars: 25–75% (thick) and 10–90% (thin) of cells', (0, 0),
            xycoords='axes fraction', xytext=(0, -44), textcoords='offset points',
            fontsize=8.5, color=MUTED)
ax.annotate('selected / detected per dose:   ' + ns, (0, 0), xycoords='axes fraction',
            xytext=(0, -57), textcoords='offset points', fontsize=8.5, color=MUTED)

leg = ax.legend(frameon=False, fontsize=9.5, loc='upper left',
                labelcolor=INK2, handlelength=1.8)
plt.tight_layout()
plt.savefig(OUT, facecolor=SURF, bbox_inches='tight')
print('wrote', OUT)

print()
print('%-8s %6s %8s %8s %8s %8s' % ('IPTG', 'n_pop', 'med_pop', 'p25', 'p75', 'med_sel'))
for dz, p, s in zip(doses, pop, sel):
    print('%-8.0f %6d %8.1f %8.1f %8.1f %8.1f'
          % (dz, len(p), np.median(p), *np.percentile(p, [25, 75]), np.median(s)))
print()
print('population median: max/min = %.2f  (max at %.0f nM)'
      % (max(np.median(p) for p in pop)/min(np.median(p) for p in pop),
         doses[int(np.argmax([np.median(p) for p in pop]))]))
