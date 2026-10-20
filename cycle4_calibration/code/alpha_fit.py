# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
import sys, csv
import numpy as np

CSV = _os.path.join(DATA,'cells_integrated.csv')

rows = list(csv.DictReader(open(CSV, encoding='utf-8')))
def f(v):
    try: return float(v)
    except: return np.nan

def load(cond, use_shape_only=True):
    out = []
    for r in rows:
        if r['condition'] != cond: continue
        if use_shape_only and r['use_shape'].strip().lower() not in ('true','1'): continue
        C, E, X = f(r['mid_mean_adu']), f(r['end_mean_adu']), f(r['expr_side_adu'])
        if not np.isfinite(C) or not np.isfinite(E) or not np.isfinite(X): continue
        if C <= 0 or X <= 0: continue
        out.append((r['rep'], C, E, X, f(r['cell_len_um'])))
    return out

def alpha(cond, b, nq=5, use_shape_only=True, verbose=False):
    """Reproduce Q2: B = E - b*C ; quintiles of X within rep ;
       fit log B = a*log C + per-rep intercept."""
    d = load(cond, use_shape_only)
    reps = sorted(set(x[0] for x in d))
    lx, ly, rid = [], [], []
    for ri, rp in enumerate(reps):
        sub = [x for x in d if x[0] == rp]
        X = np.array([x[3] for x in sub])
        order = np.argsort(X)
        bins = np.array_split(order, nq)
        for bb in bins:
            if len(bb) == 0: continue
            Cs = np.array([sub[i][1] for i in bb])
            Es = np.array([sub[i][2] for i in bb])
            Bs = Es - b*Cs
            Bm, Cm = np.median(Bs), np.median(Cs)
            if Bm <= 0 or Cm <= 0: continue
            lx.append(np.log(Cm)); ly.append(np.log(Bm)); rid.append(ri)
    if len(lx) < 3: return np.nan, np.nan, 0, len(d)
    lx = np.array(lx); ly = np.array(ly); rid = np.array(rid)
    # design: slope + one intercept per rep
    D = np.zeros((len(lx), 1+len(reps)))
    D[:, 0] = lx
    for i, ri in enumerate(rid): D[i, 1+ri] = 1.0
    coef, res, rank, sv = np.linalg.lstsq(D, ly, rcond=None)
    pred = D @ coef
    dof = max(1, len(lx) - rank)
    s2 = ((ly-pred)**2).sum()/dof
    cov = s2*np.linalg.pinv(D.T @ D)
    return coef[0], np.sqrt(cov[0,0]), len(lx), len(d)

print("=== reproduce wet's published alpha (Q2) ===")
for cond, b, ref in [('DivIVA_mCherry', 0.521, '1.055 +- 0.089'),
                     ('DivIVA_GFP',     0.571, '0.96  +- 0.56'),
                     ('mDiv_gPopZ',     0.521, '0.99  +- 0.10/0.11'),
                     ('gDiv_mPopZ',     0.571, '0.98  +- 0.23')]:
    a, se, nb, nc = alpha(cond, b)
    print("  %-16s b=%.3f  alpha=%.3f +- %.3f   (bins %d, cells %d)   report: %s"
          % (cond, b, a, se, nb, nc, ref))

print()
print("=== cell length medians (for the forward-model baseline) ===")
for cond in ('DivIVA_mCherry','DivIVA_GFP'):
    d = load(cond)
    L = np.array([x[4] for x in d]); L = L[np.isfinite(L)]
    print("  %-16s n=%3d  median L = %.4f um" % (cond, len(L), np.median(L)))
