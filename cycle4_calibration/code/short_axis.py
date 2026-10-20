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
Lcyl = L - 2*R
V = np.pi*R*R*Lcyl + 4.0/3.0*np.pi*R**3
A = 2*np.pi*R*Lcyl + 4*np.pi*R*R
PX = F.PX_UM

def transverse(f, psf, nz_fine=12, oversample=8):
    """Profile across the SHORT axis at mid-cell. Returns (x_um, value)."""
    # fine pixel grid in x, oversampled so the shape is not pixel-limited
    px = PX/oversample
    nx = int(np.ceil((2*R + 6*psf)/px/2))*2 + 1
    xe = (np.arange(nx+1) - nx/2.0)*px
    # z grid: a band around mid-cell, wide enough for the PSF
    nzp = int(np.ceil((6*psf + 1.0)/px/2))*2 + 1
    zc = L/2
    zpos = zc + (np.arange(nzp) - (nzp-1)/2.0)*px
    sub = (np.arange(nz_fine)+0.5)/nz_fine - 0.5
    zf = (zpos[:, None] + sub[None, :]*px).ravel()
    inside = (zf >= 0) & (zf <= L)
    C = np.zeros_like(zf); M = np.zeros_like(zf)
    C[inside] = (1.0-f)/V
    M[inside] = f/A
    img = F.project(zf, xe, R, L, c_of_z=C, s_of_z=M)
    img = img.reshape(nzp, nz_fine, nx).mean(axis=1)
    sig = psf/2.3548/px
    k = F._gauss_kernel(sig)
    if len(k) > 1:
        pad = len(k)//2
        for ax in (0, 1):
            img = np.apply_along_axis(
                lambda v: np.convolve(np.r_[np.zeros(pad), v, np.zeros(pad)],
                                      k, mode="same")[pad:-pad], ax, img)
    cut = img[nzp//2, :]
    xc = 0.5*(xe[:-1]+xe[1:])
    return xc, cut

print("short-axis profile at mid-cell: is the membrane shell visible as a dip?")
print("contrast = 1 - center/max  (0 = single peak, >0 = dip in the middle)\n")
print("  PSF[um]   f=0 (cyt)   f=0.5      f=0.7      f=1 (mem)   diam/PSF")
for psf in (0.20, 0.30, 0.35, 0.50, 0.70, 0.87, 1.05):
    row = []
    for f in (0.0, 0.5, 0.7, 1.0):
        x, v = transverse(f, psf)
        c = v[len(v)//2]
        row.append(1.0 - c/v.max())
    print("   %.2f     %+.4f    %+.4f    %+.4f    %+.4f     %.2f"
          % (psf, *row, 2*R/psf))

print("\nNOTE: measured effective PSF = 0.87 um, cell diameter 2R = %.2f um" % (2*R))
