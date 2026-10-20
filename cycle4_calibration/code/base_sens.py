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
import numpy as np
import forward as F

R, L = 0.4, 2.7767
PSF, W = 0.87, 3

Lcyl = L - 2*R
V = np.pi*R*R*Lcyl + 4.0/3.0*np.pi*R**3
A = 2*np.pi*R*Lcyl + 4*np.pi*R*R
print("R=%.3f L=%.4f  V=%.4f um^3  A=%.4f um^2" % (R, L, V, A))

zm = np.linspace(0, L, 2001)
one = np.ones_like(zm)

def baseline(f, psf=PSF, w=W):
    """f = fraction of molecules bound to the membrane (uniform, no localization)."""
    C = (1.0-f)/V * one
    M = f/A * one
    m = F.metrics_of(C=C, M=M, z_model=zm, R=R, L=L,
                     psf_fwhm_um=psf, line_w_px=w)
    return m["end_mid"]

print()
print("--- pure components (check against REPORT_CYCLE5 s3) ---")
print("  uniform cytoplasm  end_mid = %.4f   (report 0.5430)" % baseline(0.0))
print("  uniform membrane   end_mid = %.4f   (report 0.6939)" % baseline(1.0))
print("  mixture f=0.70     end_mid = %.4f   (report 0.6432)" % baseline(0.70))

print()
print("--- sensitivity to the membrane-bound fraction f ---")
print(" f(membrane)  baseline   measured excess 0.7852-base   as %% of wet's +0.264")
for f in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    b = baseline(f)
    exc = 0.7852 - b
    print("   %4.2f       %.4f     %+.4f                      %5.1f%%"
          % (f, b, exc, 100*exc/0.264))

print()
print("--- same, at PSF 0.70 and 1.05 (the sensitivity range) ---")
for psf in (0.70, 0.87, 1.05):
    row = [baseline(f, psf=psf) for f in (0.0,0.5,0.7,1.0)]
    print("  PSF %.2f : f=0 %.4f | f=0.5 %.4f | f=0.7 %.4f | f=1 %.4f"
          % (psf, *row))

print()
print("--- line width dependence at f=0.7 (the effect is width-driven) ---")
for w in (1,3,5,7,9,11):
    print("  w=%2d px : cyt %.4f | mem %.4f | mix0.7 %.4f"
          % (w, baseline(0.0,w=w), baseline(1.0,w=w), baseline(0.7,w=w)))
