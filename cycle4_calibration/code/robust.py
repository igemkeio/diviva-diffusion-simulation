# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0,'/tmp/claude-0/-home-claude/4e18436e-3aed-52d9-9db2-d45c28c0591d/scratchpad')
import numpy as np
import pooled_p as M   # reuses d, alpha, model_em, bnull, p_of, L
d, alpha, p_of = M.d, M.alpha, M.p_of
L = M.L
bs = [('null', M.bnull), ('mem0%', M.model_em(0.0,L,0.870)), ('mem70%', M.model_em(0.7,L,0.870))]
Ls = np.array([x[4] for x in d])

print()
print("=== robustness of the pooled alpha (DivIVA_mCherry, 5 fields, same exposure) ===")
print(" L band      nq   " + "   ".join("%-14s" % b[0] for b in bs))
allres = []
for band in [(0,100),(10,90),(20,80),(30,70),(40,60)]:
    lo,hi = np.percentile(Ls[np.isfinite(Ls)], band)
    dn = [x for x in d if np.isfinite(x[4]) and lo<=x[4]<=hi]
    for nq in (6,10,15):
        cells = []
        for lab,b in bs:
            a,s,n = alpha(dn, b, nq=nq, pooled=True)
            cells.append("%.3f±%.3f" % (a,s) if np.isfinite(a) else "   undef    ")
            if np.isfinite(a): allres.append(a)
        print(" %3d-%3d%%  %2d   %s" % (band[0],band[1],nq,"   ".join(cells)))

a = np.array(allres)
print()
print("=== summary over all %d variants ===" % len(a))
print("  alpha: min %.3f  median %.3f  max %.3f" % (a.min(), np.median(a), a.max()))
pv = [p_of(x) for x in a]
pv = [x for x in pv if np.isfinite(x)]
print("  implied p: min %.2f  median %.2f  max %.2f" % (min(pv), np.median(pv), max(pv)))
print("  alpha needed for p=0.5 : 1.886  -> %d of %d variants reach it"
      % (sum(x>=1.886 for x in a), len(a)))
print("  alpha needed for p>1   : 3.100  -> %d of %d variants reach it"
      % (sum(x>=3.100 for x in a), len(a)))
