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

FIN = _os.path.join(DATA,'cells_final_w5.csv')
INT = _os.path.join(DATA,'cells_integrated.csv')

# ---- final's window convention (final/README.md: w5, pole band z 0-0.20)
ENDS = [(0.00, 0.20), (0.80, 1.00)]
MID  = (0.40, 0.60)
R, W = 0.4, 5
PSF = {'mCherry_null':0.870, 'DivIVA_mCherry':0.870,
       'GFP_null':0.892, 'DivIVA_GFP':0.892}

def fl(v):
    try: return float(v)
    except: return np.nan

# ---------- model baseline with final's windows and line width ----------
def model_em(f, L, psf, w=W):
    Lc = L-2*R
    V = np.pi*R*R*Lc + 4/3*np.pi*R**3
    A = 2*np.pi*R*Lc + 4*np.pi*R*R
    zm = np.linspace(0, L, 2001); one = np.ones_like(zm)
    pr = F.line_profile(C=(1-f)/V*one, M=f/A*one, z_model=zm, R=R, L=L,
                        psf_fwhm_um=psf, line_w_px=w)
    v, z = pr["val"], pr["z"]
    k = max(2, int(round(len(v)*F.TAIL)))
    bg = np.median(np.r_[v[:k], v[-k:]])
    S = np.clip(v-bg, 0, None)
    # resample onto the SAME 0.02 z-grid the csv profiles use, so the model
    # baseline and the measured E/C go through identical sampling
    ZG = np.round(np.arange(-0.10, 1.1001, 0.02), 2)
    G = np.interp(ZG, z, S)
    e = np.r_[G[(ZG>=ENDS[0][0])&(ZG<=ENDS[0][1])], G[(ZG>=ENDS[1][0])&(ZG<=ENDS[1][1])]]
    m = G[(ZG>=MID[0])&(ZG<=MID[1])]
    return e.mean()/m.mean()

# ---------- read final, rebuild E and C from the stored profile ----------
rows = list(csv.DictReader(open(FIN, encoding='utf-8')))
hdr = rows[0].keys()
zs, cols = [], []
for c in hdr:
    if c.startswith('div_z'):
        zs.append(float(c[len('div_z'):])); cols.append(c)
zs = np.array(zs)
mE = ((zs>=0)&(zs<=0.20)) | ((zs>=0.80)&(zs<=1.00))
mM = (zs>=0.40)&(zs<=0.60)

X = {}
for r in csv.DictReader(open(INT, encoding='utf-8')):
    X[r['cell_uid']] = fl(r['expr_side_adu'])

def load(cond):
    out = []
    for r in rows:
        if r['condition'] != cond: continue
        v = np.array([fl(r[c]) for c in cols])
        if not np.isfinite(v).all(): continue
        S = np.clip(v, 0, None)          # already background-subtracted [ADU]
        E, C = S[mE].mean(), S[mM].mean()
        x = X.get(r['cell_uid'], np.nan)
        if C <= 0 or not np.isfinite(x) or x <= 0: continue
        out.append((r['rep'], C, E, x, fl(r['cell_len_um']), fl(r['end_mid'])))
    return out

def alpha(cond, b, nq=5):
    d = load(cond)
    reps = sorted(set(x[0] for x in d))
    lx, ly, rid = [], [], []
    for ri, rp in enumerate(reps):
        sub = [x for x in d if x[0]==rp]
        order = np.argsort([x[3] for x in sub])
        for bb in np.array_split(order, nq):
            if len(bb)==0: continue
            Cs = np.array([sub[i][1] for i in bb]); Es = np.array([sub[i][2] for i in bb])
            Bm, Cm = np.median(Es-b*Cs), np.median(Cs)
            if Bm<=0 or Cm<=0: continue
            lx.append(np.log(Cm)); ly.append(np.log(Bm)); rid.append(ri)
    if len(lx) < 3: return np.nan, np.nan, len(lx), len(d)
    lx=np.array(lx); ly=np.array(ly); rid=np.array(rid)
    D=np.zeros((len(lx),1+len(reps))); D[:,0]=lx
    for i,ri in enumerate(rid): D[i,1+ri]=1.0
    coef,_,rank,_ = np.linalg.lstsq(D,ly,rcond=None)
    pred=D@coef; dof=max(1,len(lx)-rank)
    cov=((ly-pred)**2).sum()/dof*np.linalg.pinv(D.T@D)
    return coef[0], np.sqrt(cov[0,0]), len(lx), len(d)

print("=== final (w5, ends z 0-0.20): measured vs model baselines, own L ===")
print("condition          n   med L   measured  my calc   model f=0  f=0.5  f=0.7  f=1")
for cond in ('mCherry_null','DivIVA_mCherry','GFP_null','DivIVA_GFP'):
    d = load(cond)
    if not d: 
        print(cond, 'no rows'); continue
    L = float(np.median([x[4] for x in d]))
    em_file = float(np.median([x[5] for x in d]))
    em_mine = float(np.median([x[2]/x[1] for x in d]))
    psf = PSF[cond]
    print("%-16s %4d  %.3f   %.4f   %.4f    %.4f   %.4f %.4f %.4f"
          % (cond, len(d), L, em_file, em_mine,
             model_em(0.0,L,psf), model_em(0.5,L,psf),
             model_em(0.7,L,psf), model_em(1.0,L,psf)))

P   = np.array([0.00,0.10,0.20,0.35,0.50,0.70])
ALP = np.array([1.000,1.107,1.235,1.489,1.886,3.142])
def p_of(a):
    if not np.isfinite(a) or a<=ALP[0]: return 0.0 if np.isfinite(a) else np.nan
    if a>=ALP[-1]: return np.nan
    return float(np.interp(a,ALP,P))

print()
print("=== alpha on final data ===")
for cond in ('DivIVA_mCherry','DivIVA_GFP'):
    d = load(cond)
    L = float(np.median([x[4] for x in d])); psf = PSF[cond]
    nullname = 'mCherry_null' if 'mCherry' in cond else 'GFP_null'
    nd = load(nullname)
    bnull = float(np.median([x[2]/x[1] for x in nd]))
    cands = [('measured same-colour null', bnull),
             ('model f=0   (own L)', model_em(0.0,L,psf)),
             ('model f=0.7 (own L)', model_em(0.7,L,psf))]
    print(" %s   (%d cells, %d field(s))" % (cond, len(d), len(set(x[0] for x in d))))
    for lab, b in cands:
        a, se, nb, nc = alpha(cond, b)
        pv = p_of(a)
        print("   b=%.4f %-26s alpha=%s  -> p %s   (bins %d)"
              % (b, lab,
                 ('%.3f +- %.3f' % (a,se)) if np.isfinite(a) else '   undefined  ',
                 ('%.2f' % pv) if np.isfinite(pv) else ' -', nb))
    a,se,_,_ = alpha(cond, bnull)
    if np.isfinite(a):
        print("   -> p>1 needs alpha>=3.1 : %.1f sigma away" % ((3.1-a)/se))
