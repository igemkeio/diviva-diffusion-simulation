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
"""Model convolution: simulated cell images vs real E. coli crops.

Left  : free mCherry (null)      -> model = uniform cytoplasm
Right : DivIVA-mCherry (10 uM)   -> model = cluster steady state (Cycle 5 parameters)

Real crops: data/1010_*/..., rotated so the long axis is horizontal.
Model: cluster.steady -> forward.project (+PSF, +background, +noise), same pixel size.
"""
import sys, csv
import numpy as np, tifffile
from scipy import ndimage
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import forward as F, cluster as CL

INT = _os.path.join(DATA,'cells_integrated.csv')
FIN = _os.path.join(DATA,'cells_final_w5.csv')
# 生 tif の場所: Igem数値解析/data/ を優先し、無ければ同梱コピー（wiki/Cycle4(calibration)/data/）
_RAW = _os.path.join(REPO, 'data')
if not _os.path.isdir(_os.path.join(_RAW, '1010_FPnull')):
    _RAW = _os.path.join(ROOT, 'data')
TIF = {'mCherry_null':  (_os.path.join(_RAW,'1010_FPnull','mche','mche4.tif'), '1'),
       'DivIVA_mCherry':(_os.path.join(_RAW,'1010_mc_Div','5','5aa.tif'),      '5')}
PX, PSF, R = F.PX_UM, 0.870, 0.40
POLE_WIN = 0.144      # = R/L (0.40/2.78) 半球キャップが終わる位置。dry 標準の窓
NCELL = 3
OUT = _os.path.join(FIGS,'cal_fig6_model_vs_image.png')
SURF, INK, INK2, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8882'
C_MEAS, C_MODEL = '#2a78d6', '#eb6834'

def fl(v):
    try: return float(v)
    except: return np.nan

rows_int = list(csv.DictReader(open(INT, encoding='utf-8')))
rows_fin = list(csv.DictReader(open(FIN, encoding='utf-8')))
pcols = [c for c in rows_fin[0] if c.startswith('div_z')]
zs = np.array([float(c[5:]) for c in pcols])

# 図に出す DivIVA の細胞（cells_integrated.csv の cell_id、rep 5）。
# SNR 順だと局在が見えている細胞が選ばれないので、明示的に指定する。
#   #221  pole/mid 1.55  sym 0.93  SNR 29  L 2.89 µm
#   #139  pole/mid 1.25  sym 0.77  SNR 14  L 2.95 µm
#   #284  pole/mid 1.17  sym 0.91  SNR 19  L 4.07 µm
DIV_CELL_IDS = ['221', '139', '284']

def pick(cond, rep, n=NCELL, Ltarget=None):
    if cond == 'DivIVA_mCherry':
        byid = {r['cell_id']: r for r in rows_int
                if r['condition'] == cond and r['rep'] == rep}
        return [byid[w] for w in DIV_CELL_IDS]
    """good-SNR cells whose length is close to the one the model was solved at"""
    cs = [r for r in rows_int
          if r['condition'] == cond and r['rep'] == rep
          and r['use_shape'].strip().lower() in ('true', '1')
          and np.isfinite(fl(r['snr'])) and np.isfinite(fl(r['x1']))
          and np.isfinite(fl(r['center_um'])) and np.isfinite(fl(r['cell_len_um']))]
    cs.sort(key=lambda r: -fl(r['snr']))
    cs = cs[:max(n * 6, 12)]                       # brightest, then closest in length
    if Ltarget is not None:
        cs.sort(key=lambda r: abs(fl(r['cell_len_um']) - Ltarget))
    return cs[:n]

def crop(tif, r, half_um=2.2, h_um=1.6):
    """rotate the cell to horizontal, centred on the registered cell centre.

    fl_slice is 1-based.  The fluorescence ROI line runs past both ends of the
    cell, so its midpoint is NOT the cell centre: `center_um` gives the distance
    along that line from (x1,y1) to the centre (METHODS_AND_RESULTS section 2).
    """
    sl = int(float(r['fl_slice'])) - 1
    img = tif[sl].astype(float)
    x1, y1, x2, y2 = (fl(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
    dx, dy = x2 - x1, y2 - y1
    Ln = np.hypot(dx, dy)
    t = fl(r['center_um']) / PX
    cx, cy = x1 + dx / Ln * t, y1 + dy / Ln * t
    ny, nx = img.shape
    img = ndimage.shift(img, (ny / 2 - cy, nx / 2 - cx), order=1, mode='nearest')
    ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    img = ndimage.rotate(img, ang, reshape=False, order=1, mode='nearest')
    hw, hh = int(half_um / PX), int(h_um / 2 / PX)
    return img[ny // 2 - hh: ny // 2 + hh + 1, nx // 2 - hw: nx // 2 + hw + 1]

def model_image(C, M, N, zm, L, bg, noise, peak, half_um=2.2, h_um=1.6, nz_fine=12):
    """projection -> PSF -> pixels, with measured background and noise added"""
    npx = int(round(2 * half_um / PX)) | 1
    pos = L / 2 + (np.arange(npx) - (npx - 1) / 2.0) * PX
    sub = (np.arange(nz_fine) + 0.5) / nz_fine - 0.5
    zf = (pos[:, None] + sub[None, :] * PX).ravel()
    inside = (zf >= 0) & (zf <= L)
    def ip(f):
        v = np.zeros_like(zf)
        if f is not None: v[inside] = np.interp(zf[inside], zm, f)
        return v
    nx = int(round(h_um / PX)) | 1
    xe = (np.arange(nx + 1) - nx / 2.0) * PX
    mem = ip(M) + ip(N)
    im = F.project(zf, xe, R, L, c_of_z=ip(C), s_of_z=mem)
    im = im.reshape(npx, nz_fine, nx).mean(axis=1).T          # (x, z)
    k = F._gauss_kernel(PSF / 2.3548 / PX)
    if len(k) > 1:
        pad = len(k) // 2
        for ax in (0, 1):
            im = np.apply_along_axis(
                lambda v: np.convolve(np.r_[np.zeros(pad), v, np.zeros(pad)],
                                      k, mode='same')[pad:-pad], ax, im)
    im = im / im.max() * peak                                  # match measured peak
    rng = np.random.default_rng(0)
    return im + bg + rng.normal(0, noise, im.shape)

def meas_profile(cond):
    sel = [r for r in rows_fin if r['condition'] == cond]
    P = np.array([[fl(r[c]) for c in pcols] for r in sel])
    P = P[np.isfinite(P).all(axis=1)]
    nrm = np.median(P[:, (zs >= .35) & (zs <= .65)], axis=1)
    P = P[nrm > 0] / nrm[nrm > 0][:, None]
    return np.median(P, axis=0), np.percentile(P, [25, 75], axis=0)

def stats(cond):
    sel = [r for r in rows_int if r['condition'] == cond
           and r['use_shape'].strip().lower() in ('true', '1')]
    g = lambda k: np.nanmedian([fl(r[k]) for r in sel])
    return g('bg'), g('noise'), g('cell_len_um')

# ---------------- model fields ----------------
par = CL.variant(p=0.17, rho_n=4.1812, L=2.7767, R=0.40)
st = CL.steady(n=201, par=par)
zm_div = np.linspace(0, par.L, 201)
C_div, M_div, N_div = st['C'], st['M'], st['Nf']

fig = plt.figure(figsize=(11.8, 6.8), dpi=200)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 0.62, 3.0],
                      hspace=.42, wspace=.14,
                      left=.105, right=.985, top=.885, bottom=.175)

def disp(a):
    """per-cell display scaling -> 0..1 (grayscale, linear, 2-99.5 percentile)"""
    f = a[np.isfinite(a)]
    lo, hi = np.percentile(f, 2), np.percentile(f, 99.5)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)

def strip_of(arrs):
    return np.concatenate([np.pad(disp(a), ((0, 0), (2, 2)),
                                  constant_values=np.nan) for a in arrs], axis=1)

panels = [
    ('mCherry_null',   'Free mCherry  (non-localizing control)', 'uniform cytoplasm', None),
    ('DivIVA_mCherry', 'DivIVA-mCherry  (10 µM IPTG)',           'cluster steady state', 'div'),
]

for col, (cond, title, mlabel, kind) in enumerate(panels):
    tif = tifffile.imread(TIF[cond][0])
    cells = pick(cond, TIF[cond][1], Ltarget=(par.L if kind == 'div' else None))
    bands = [crop(tif, r) for r in cells]
    bg, noise, Lmed = stats(cond)

    axr = fig.add_subplot(gs[0, col]); axr.set_facecolor(SURF)
    axr.imshow(strip_of(bands), cmap='gray', vmin=0, vmax=1,
               interpolation='nearest', aspect='equal')
    axr.set_title(title, fontsize=10.5, color=INK, loc='left', pad=7)
    if col == 0:
        bb = axr.get_position()
        fig.text(.022, (bb.y0 + bb.y1) / 2, 'measured', fontsize=9.5, color=INK2,
                 va='center', ha='left')
    axr.set_xticks([]); axr.set_yticks([])
    for sp in axr.spines.values(): sp.set_visible(False)
    if col == 0:   # scale bar: 1 um
        n = bands[0].shape[1] + 4
        axr.plot([4, 4 + 1.0 / PX], [bands[0].shape[0] - 2.5] * 2,
                 color='white', lw=2.4, solid_capstyle='butt')
        axr.text(4, bands[0].shape[0] - 4.5, '1 µm', color='white', fontsize=8, va='bottom')

    axm = fig.add_subplot(gs[1, col]); axm.set_facecolor(SURF)
    if kind == 'div':
        sims = [model_image(C_div, M_div, N_div, zm_div, par.L, bg, noise,
                            np.nanpercentile(b, 99.5)) for b in bands]
    else:
        zmn = np.linspace(0, Lmed, 201); one = np.ones_like(zmn)
        sims = [model_image(one, None, None, zmn, Lmed, bg, noise,
                            np.nanpercentile(b, 99.5)) for b in bands]
    axm.imshow(strip_of(sims), cmap='gray', vmin=0, vmax=1,
               interpolation='nearest', aspect='equal')
    axm.set_xticks([]); axm.set_yticks([])
    for sp in axm.spines.values(): sp.set_visible(False)
    axm.set_title('model: ' + mlabel, fontsize=9.2, color=MUTED, loc='left', pad=4)
    if col == 0:
        bb = axm.get_position()
        fig.text(.022, (bb.y0 + bb.y1) / 2, 'simulated', fontsize=9.5, color=INK2,
                 va='center', ha='left')

    axp = fig.add_subplot(gs[2, col]); axp.set_facecolor(SURF)
    med, iqr = meas_profile(cond)
    axp.axvspan(0, POLE_WIN, color='#efeee9', zorder=0)
    axp.axvspan(1 - POLE_WIN, 1., color='#efeee9', zorder=0)
    axp.fill_between(zs, iqr[0], iqr[1], color=C_MEAS, alpha=.16, lw=0, zorder=1)
    axp.plot(zs, med, color=C_MEAS, lw=2.0, zorder=3,
             label='measured (median of %d cells, IQR shaded)'
                   % sum(r['condition'] == cond for r in rows_fin))
    if kind == 'div':
        pr = F.line_profile(C=C_div, M=M_div, N=N_div, z_model=zm_div, R=R,
                            L=par.L, psf_fwhm_um=PSF, line_w_px=5)
    else:
        zmn = np.linspace(0, Lmed, 201)
        pr = F.line_profile(C=np.ones(201), z_model=zmn, R=R, L=Lmed,
                            psf_fwhm_um=PSF, line_w_px=5)
    v, zz = pr['val'], pr['z']
    kk = max(2, int(round(len(v) * F.TAIL)))
    v = np.clip(v - np.median(np.r_[v[:kk], v[-kk:]]), 0, None)
    v = v / np.median(v[(zz >= .35) & (zz <= .65)])
    axp.plot(zz, v, color=C_MODEL, lw=2.0, zorder=4, label='model (forward)')
    if kind == 'div':
        # same total amount, same 30/70 cytoplasm/membrane split, but NO localization
        Lc = par.L - 2 * R
        Vc = np.pi * R * R * Lc + 4 / 3 * np.pi * R ** 3
        Ac = 2 * np.pi * R * Lc + 4 * np.pi * R * R
        one = np.ones_like(zm_div)
        pr0 = F.line_profile(C=.30 / Vc * one, M=.70 / Ac * one, z_model=zm_div,
                             R=R, L=par.L, psf_fwhm_um=PSF, line_w_px=5)
        v0, z0_ = pr0['val'], pr0['z']
        kk0 = max(2, int(round(len(v0) * F.TAIL)))
        v0 = np.clip(v0 - np.median(np.r_[v0[:kk0], v0[-kk0:]]), 0, None)
        v0 = v0 / np.median(v0[(z0_ >= .35) & (z0_ <= .65)])
        axp.plot(z0_, v0, color=MUTED, lw=1.6, ls=(0, (4, 2.4)), zorder=2,
                 label='model with NO localization (same 30/70 split)')
        axp.annotate('', xy=(.10, np.interp(.10, zz, v)),
                     xytext=(.10, np.interp(.10, z0_, v0)),
                     arrowprops=dict(arrowstyle='<->', color='#b4532a', lw=1.3))
        axp.text(.125, (np.interp(.10, zz, v) + np.interp(.10, z0_, v0)) / 2,
                 'this gap is\nthe evidence', fontsize=8.4, color='#b4532a', va='center')
    axp.set_xlim(-.08, 1.08); axp.set_ylim(0, 1.42)
    axp.set_xlabel('position along the cell  (z / L)', fontsize=9.5, color=INK2)
    if col == 0:
        axp.set_ylabel('intensity / mid-cell', fontsize=9.5, color=INK2)
    axp.tick_params(colors=INK2, labelsize=9)
    for sp in ('top', 'right'): axp.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): axp.spines[sp].set_color('#d6d5cf')
    axp.legend(frameon=False, fontsize=8.3, loc='lower center',
               labelcolor=INK2, handlelength=1.9)
    axp.text(POLE_WIN / 2, 1.33, 'pole window', fontsize=8, color=MUTED,
             ha='center')

fig.text(.012, .945,
         'Model convolution — images simulated from the model, beside the real cells',
         fontsize=12.5, color=INK, ha='left')
fig.text(.012, .012,
         'Simulated = model distribution → projection through the spherocylinder → PSF 0.87 µm → '
         'pixels (0.1625 µm) → measured background and noise added.\n'
         'Grayscale, linear; each cell is contrast-scaled individually (2–99.5 percentile). '
         'The same scaling rule is used for measured and simulated; no other processing beyond '
         'rotation and centring.\n'
         'Note the left panel is a CALIBRATION (the PSF was fitted to these null profiles); only the '
         'right panel is a prediction. The overall arch shape comes from the cell geometry and is '
         'present even without localization — the dashed line.',
         fontsize=8.2, color=MUTED, ha='left', va='bottom')
plt.savefig(OUT, facecolor=SURF, bbox_inches='tight')
print('wrote', OUT)
