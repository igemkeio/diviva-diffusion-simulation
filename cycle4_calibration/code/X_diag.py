# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
REPO  = _os.path.dirname(ROOT)
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""Is the 'cell total' breakage a binning artefact of the 0.02-resampled csv,
   or genuine pixel-sharing between the binning variable and B/C?
   Decisive test: bin by an integral that EXCLUDES the pole and mid bands."""
import csv
import numpy as np
FIN=_os.path.join(DATA,'cells_final_w5.csv')
INT=_os.path.join(DATA,'cells_integrated.csv')
rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1.0)); mM=(zs>=.40)&(zs<=.60)
mIn=(zs>=0)&(zs<=1)
mGAP=((zs>.20)&(zs<.40))|((zs>.60)&(zs<.80))        # neither pole nor mid
def fl(v):
    try: return float(v)
    except: return np.nan
SIDE,TOT={},{}
for r in csv.DictReader(open(INT,encoding='utf-8')):
    SIDE[r['cell_uid']]=fl(r['expr_side_adu']); TOT[r['cell_uid']]=fl(r['total'])
d=[]
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    v=np.array([fl(r[c]) for c in cols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None); C=S[mM].mean(); E=S[mE].mean()
    if C<=0: continue
    d.append(dict(C=C,E=E,
        side=SIDE.get(r['cell_uid'],np.nan),
        tot_sum=S[mIn].sum(),                       # the one used before (shares E and C)
        tot_trap=np.trapezoid(S[mIn],zs[mIn]),      # same data, different quadrature
        gap=S[mGAP].mean(),                         # SAME csv, same grid, but no shared pixels
        roi_total=TOT.get(r['cell_uid'],np.nan)))   # pipeline's own 2-D integral
def alpha(key,b,nq=10):
    sub=[t for t in d if np.isfinite(t[key]) and t[key]>0]
    order=np.argsort([t[key] for t in sub]); lx,ly=[],[]
    for bb in np.array_split(order,nq):
        if len(bb)==0: continue
        Cs=np.array([sub[i]['C'] for i in bb]); Es=np.array([sub[i]['E'] for i in bb])
        Bm,Cm=np.median(Es-b*Cs),np.median(Cs)
        if Bm<=0 or Cm<=0: continue
        lx.append(np.log(Cm)); ly.append(np.log(Bm))
    if len(lx)<3: return np.nan,np.nan,len(lx),len(sub)
    lx=np.array(lx); ly=np.array(ly); A=np.c_[lx,np.ones(len(lx))]
    co,*_=np.linalg.lstsq(A,ly,rcond=None); res=ly-A@co
    s2=(res**2).sum()/max(len(lx)-2,1)
    return co[0],np.sqrt(s2*np.linalg.inv(A.T@A)[0,0]),len(lx),len(sub)
print('n =',len(d))
print()
print('ビン分けに使う量                       画素を E/C と共有?  alpha (下駄=実測null 0.5779)')
for key,lab,share in (('side','±3 px の平行線（外側）','しない'),
                      ('roi_total','パイプラインの 2 次元積分  ','する'),
                      ('tot_sum','プロファイルの総和（単純和）','する'),
                      ('tot_trap','プロファイルの総和（台形則）','する'),
                      ('gap','z 0.2-0.4 / 0.6-0.8 の平均  ','しない'),
                      ('C','C そのもの              ','する(C と同一)')):
    a,se,nb,ns=alpha(key,0.5779)
    print('  %-30s  %-8s  %s  (%d bins, n=%d)'%(lab,share,
          ('%.3f ± %.3f'%(a,se)) if np.isfinite(a) else '   undefined ',nb,ns))
print()
print('=== 相関（log） ===')
import itertools
v={k:np.array([t[k] for t in d]) for k in ('C','side','gap','tot_sum','roi_total')}
ok=np.ones(len(d),bool)
for k in v: ok &= np.isfinite(v[k])&(v[k]>0)
for k in ('side','gap','tot_sum','roi_total'):
    print('   rho(log %-10s, log C) = %+.3f'%(k,np.corrcoef(np.log(v[k][ok]),np.log(v['C'][ok]))[0,1]))
