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
import numpy as np, forward as F

rows = list(csv.DictReader(open(_os.path.join(DATA,'cells_integrated.csv'), encoding='utf-8')))
def fl(v):
    try: return float(v)
    except: return np.nan

R = 0.4
def base(f, L, psf, w=3):
    Lc = L-2*R
    V = np.pi*R*R*Lc + 4/3*np.pi*R**3
    A = 2*np.pi*R*Lc + 4*np.pi*R*R
    zm = np.linspace(0, L, 2001); one = np.ones_like(zm)
    return F.metrics_of(C=(1-f)/V*one, M=f/A*one, z_model=zm, R=R, L=L,
                        psf_fwhm_um=psf, line_w_px=w)["end_mid"]

PSF = {'mCherry_null':0.870, 'DivIVA_mCherry':0.870,
       'GFP_null':0.892, 'DivIVA_GFP':0.892}

print("condition          n   median L   measured    model     model     model")
print("                           [um]   end_mid     cyt f=0   f=0.7     mem f=1")
for cond in ('mCherry_null','DivIVA_mCherry','GFP_null','DivIVA_GFP'):
    sel = [r for r in rows if r['condition']==cond
           and r['use_shape'].strip().lower() in ('true','1')]
    L  = np.array([fl(r['cell_len_um']) for r in sel]); L = L[np.isfinite(L)]
    em = np.array([fl(r['end_mid'])     for r in sel]); em = em[np.isfinite(em)]
    Lm = float(np.median(L))
    psf = PSF[cond]
    print("%-16s %4d   %.4f    %.4f      %.4f    %.4f    %.4f"
          % (cond, len(sel), Lm, np.median(em),
             base(0.0, Lm, psf), base(0.7, Lm, psf), base(1.0, Lm, psf)))

print()
print("KEY: excess over each candidate baseline, at the condition's OWN length")
for cond in ('DivIVA_mCherry','DivIVA_GFP'):
    sel = [r for r in rows if r['condition']==cond
           and r['use_shape'].strip().lower() in ('true','1')]
    L  = np.array([fl(r['cell_len_um']) for r in sel]); L = L[np.isfinite(L)]
    em = np.array([fl(r['end_mid'])     for r in sel]); em = em[np.isfinite(em)]
    Lm, e = float(np.median(L)), float(np.median(em))
    psf = PSF[cond]
    nullname = 'mCherry_null' if 'mCherry' in cond else 'GFP_null'
    nsel = [r for r in rows if r['condition']==nullname
            and r['use_shape'].strip().lower() in ('true','1')]
    nem = np.median([fl(r['end_mid']) for r in nsel if np.isfinite(fl(r['end_mid']))])
    print("  %-16s measured %.4f" % (cond, e))
    print("      vs measured same-colour null  %.4f  -> %+.4f" % (nem, e-nem))
    print("      vs model f=0.7 (own L)        %.4f  -> %+.4f" % (base(0.7,Lm,psf), e-base(0.7,Lm,psf)))
    print("      vs model f=0.5 (own L)        %.4f  -> %+.4f" % (base(0.5,Lm,psf), e-base(0.5,Lm,psf)))
    print("      vs model f=0   (own L)        %.4f  -> %+.4f" % (base(0.0,Lm,psf), e-base(0.0,Lm,psf)))

print()
print("Length-matched check: GFP baseline recomputed at the GFP_null length")
gl = np.median([fl(r['cell_len_um']) for r in rows
                if r['condition']=='GFP_null' and r['use_shape'].strip().lower() in ('true','1')
                and np.isfinite(fl(r['cell_len_um']))])
print("  GFP_null L=%.4f : model cyt %.4f  vs measured GFP_null %.4f  (residual %+.4f)"
      % (gl, base(0.0, gl, 0.892),
         np.median([fl(r['end_mid']) for r in rows if r['condition']=='GFP_null'
                    and r['use_shape'].strip().lower() in ('true','1')
                    and np.isfinite(fl(r['end_mid']))]),
         np.median([fl(r['end_mid']) for r in rows if r['condition']=='GFP_null'
                    and r['use_shape'].strip().lower() in ('true','1')
                    and np.isfinite(fl(r['end_mid']))]) - base(0.0, gl, 0.892)))
ml = np.median([fl(r['cell_len_um']) for r in rows
                if r['condition']=='mCherry_null' and r['use_shape'].strip().lower() in ('true','1')
                and np.isfinite(fl(r['cell_len_um']))])
mm = np.median([fl(r['end_mid']) for r in rows if r['condition']=='mCherry_null'
                and r['use_shape'].strip().lower() in ('true','1')
                and np.isfinite(fl(r['end_mid']))])
print("  mCherry_null L=%.4f : model cyt %.4f  vs measured %.4f  (residual %+.4f)"
      % (ml, base(0.0, ml, 0.870), mm, mm - base(0.0, ml, 0.870)))
