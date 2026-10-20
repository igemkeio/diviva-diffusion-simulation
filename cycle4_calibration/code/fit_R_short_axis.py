# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Measure the cell radius R from the SHORT-axis profile of the non-localizing controls.

Along the long axis R and the PSF are degenerate.  Across the short axis the cell is a
cylinder, so the profile is the chord 2*sqrt(R^2-y^2) convolved with the PSF, and
variances add:  sigma_obs^2 = (R/(2*sqrt2))^2 + sigma_PSF^2.
Fitting both axes together pins R and the PSF separately.
"""
import csv, numpy as np, tifffile
from scipy import ndimage, optimize

INT = _os.path.join(DATA,'cells_integrated.csv')
SRC = {'mCherry_null': (_os.path.join(REPO,'data','1010_FPnull','mche','mche4.tif'), '1'),
       'GFP_null':     (_os.path.join(REPO,'data','1010_FPnull','g','gg.tif'),       '1')}
PX = 0.1625
HALF = 1.30          # um, half-height of the transverse cut
BAND = 0.15          # longitudinal half-window (fraction of L) around mid-cell

def fl(v):
    try: return float(v)
    except: return np.nan

rows = list(csv.DictReader(open(INT, encoding='utf-8')))

def transverse_of_cell(tif, r):
    """rotate the cell horizontal, return the mid-cell transverse profile"""
    sl = int(fl(r['fl_slice'])) - 1
    img = tif[sl].astype(float)
    x1, y1, x2, y2 = (fl(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
    dx, dy = x2 - x1, y2 - y1
    Ln = np.hypot(dx, dy)
    t = fl(r['center_um']) / PX
    cx, cy = x1 + dx/Ln*t, y1 + dy/Ln*t
    ny, nx = img.shape
    img = ndimage.shift(img, (ny/2 - cy, nx/2 - cx), order=1, mode='nearest')
    img = ndimage.rotate(img, np.degrees(np.arctan2(dy, dx)), reshape=False,
                         order=1, mode='nearest')
    L = fl(r['cell_len_um'])
    hw = max(1, int(round(BAND * L / PX)))          # columns around mid-cell
    hh = int(round(HALF / PX))
    band = img[ny//2 - hh: ny//2 + hh + 1, nx//2 - hw: nx//2 + hw + 1]
    return np.median(band, axis=1)                  # median over the longitudinal window

def model_transverse(y, R, psf_fwhm, amp, bg, y0):
    """chord of a cylinder, convolved with a Gaussian, sampled on the pixel grid"""
    fine = np.arange(-3.0, 3.0, PX/16)
    ch = np.where(np.abs(fine) < R, 2*np.sqrt(np.maximum(R**2 - fine**2, 0)), 0.0)
    sig = psf_fwhm/2.3548/(PX/16)
    rad = max(1, int(np.ceil(4*sig)))
    g = np.exp(-0.5*((np.arange(-rad, rad+1))/sig)**2); g /= g.sum()
    conv = np.convolve(ch, g, mode='same')
    conv = conv/conv.max()
    return bg + amp*np.interp(y - y0, fine, conv)

out = {}
for cond, (path, rep) in SRC.items():
    tif = tifffile.imread(path)
    cs = [r for r in rows if r['condition'] == cond and r['rep'] == rep
          and r['use_shape'].strip().lower() in ('true', '1')
          and np.isfinite(fl(r['x1'])) and np.isfinite(fl(r['center_um']))
          and np.isfinite(fl(r['cell_len_um']))]
    profs = []
    for r in cs:
        p = transverse_of_cell(tif, r)
        tail = np.r_[p[:4], p[-4:]]
        p = p - np.median(tail)
        if p.max() <= 0: continue
        profs.append(p/p.max())
    P = np.array(profs)
    med = np.median(P, axis=0)
    y = (np.arange(len(med)) - (len(med)-1)/2)*PX
    p0 = [0.40, 0.87, 1.0, 0.0, 0.0]
    fit, cov = optimize.curve_fit(model_transverse, y, med, p0=p0,
                                  bounds=([0.15, 0.10, .3, -.3, -.4],
                                          [0.90, 2.00, 3., .3, .4]), maxfev=40000)
    perr = np.sqrt(np.diag(cov))
    resid = med - model_transverse(y, *fit)
    # moment-based cross-check
    w = np.clip(med - fit[3], 0, None)
    sig_obs = np.sqrt((w*(y - fit[4])**2).sum()/w.sum())
    out[cond] = dict(n=len(P), R=fit[0], dR=perr[0], psf=fit[1], dpsf=perr[1],
                     rms=float(np.sqrt((resid**2).mean())), sig=sig_obs, y=y, med=med, fit=fit)
    print('%-13s n=%3d   R = %.3f +- %.3f um   PSF = %.3f +- %.3f um   RMS %.4f'
          % (cond, len(P), fit[0], perr[0], fit[1], perr[1], out[cond]['rms']))
    print('              observed short-axis sigma = %.4f um' % sig_obs)
    print('              implied by the fit:  chord %.4f + psf %.4f -> %.4f'
          % (fit[0]/(2*np.sqrt(2)), fit[1]/2.3548,
             np.hypot(fit[0]/(2*np.sqrt(2)), fit[1]/2.3548)))

print()
print('=== comparison with the long-axis calibration (R fixed at 0.40) ===')
print('   long-axis fit gave  PSF 0.870 (mCherry) / 0.892 (GFP),  R assumed 0.40')
for c in out:
    print('   %-13s short-axis: R %.3f, PSF %.3f' % (c, out[c]['R'], out[c]['psf']))
np.save(_os.path.join(ROOT, '_shortaxis.npy'),
        np.array([[out[c]['y'], out[c]['med']] for c in ('mCherry_null', 'GFP_null')],
                 dtype=object),
        allow_pickle=True)
import json
json.dump({c: {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
               for k, v in out[c].items()
               if k in ('n', 'R', 'dR', 'psf', 'dpsf', 'rms', 'sig')}
           for c in out},
          open(_os.path.join(ROOT, '_shortaxis.json'), 'w'), indent=1)
