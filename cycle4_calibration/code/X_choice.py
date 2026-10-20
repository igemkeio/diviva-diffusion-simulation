# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Does the choice of the brightness proxy X change alpha?
   (a) side lines  (b) the cell's own background-subtracted integral  (c) C itself"""
import sys, csv
import numpy as np, forward as F

FIN=_os.path.join(DATA,'cells_final_w5.csv')
INT=_os.path.join(DATA,'cells_integrated.csv')
rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1.0)); mM=(zs>=.40)&(zs<=.60); mIn=(zs>=0)&(zs<=1)
def fl(v):
    try: return float(v)
    except: return np.nan
SIDE={r['cell_uid']:fl(r['expr_side_adu']) for r in csv.DictReader(open(INT,encoding='utf-8'))}

d=[]
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    v=np.array([fl(r[c]) for c in cols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None)
    C=S[mM].mean(); E=S[mE].mean(); tot=S[mIn].sum()
    x=SIDE.get(r['cell_uid'],np.nan)
    if C<=0 or tot<=0: continue
    d.append((r['rep'],C,E,x,tot,fl(r['cell_len_um'])))
print('n =',len(d),' (X が欠けている細胞:',sum(1 for t in d if not np.isfinite(t[3])),')')

C=np.array([t[1] for t in d]); tot=np.array([t[4] for t in d])
X=np.array([t[3] for t in d]); ok=np.isfinite(X)&(X>0)
print('X<=0 の細胞:',int((np.isfinite(X)&(X<=0)).sum()))
print()
print('=== 候補どうしの相関（log, n=%d） ===' % ok.sum())
print('   rho(log X_side, log C)      = %+.3f' % np.corrcoef(np.log(X[ok]),np.log(C[ok]))[0,1])
print('   rho(log total,  log C)      = %+.3f   <- 自分自身を含むので高い' % np.corrcoef(np.log(tot),np.log(C))[0,1])
print('   rho(log X_side, log total)  = %+.3f' % np.corrcoef(np.log(X[ok]),np.log(tot[ok]))[0,1])

def alpha(dd, b, key, nq=10):
    sub=[t for t in dd if np.isfinite(t[key])]
    order=np.argsort([t[key] for t in sub])
    lx,ly=[],[]
    for bb in np.array_split(order,nq):
        if len(bb)==0: continue
        Cs=np.array([sub[i][1] for i in bb]); Es=np.array([sub[i][2] for i in bb])
        Bm,Cm=np.median(Es-b*Cs),np.median(Cs)
        if Bm<=0 or Cm<=0: continue
        lx.append(np.log(Cm)); ly.append(np.log(Bm))
    if len(lx)<3: return np.nan,np.nan,len(lx)
    globals()['_last']=(np.exp(lx),np.exp(ly))
    lx=np.array(lx); ly=np.array(ly)
    A=np.c_[lx,np.ones(len(lx))]
    co,*_=np.linalg.lstsq(A,ly,rcond=None)
    res=ly-A@co; s2=(res**2).sum()/max(len(lx)-2,1)
    return co[0], np.sqrt(s2*np.linalg.inv(A.T@A)[0,0]), len(lx)

print()
print('=== プールした alpha を、ビン分けに使う量だけ変えて比べる ===')
print('   下駄 b        ビン分けの量            alpha')
for lab,b in (('実測 null  ',0.5779),('モデル膜0% ',0.6143),('モデル膜70%',0.7019)):
    line='   %s' % lab
    res=[]
    for knm,key in (('side lines',3),('cell total',4),('C itself  ',1)):
        a,se,n=alpha(d,b,key)
        res.append('%-11s %.3f±%.3f (%d bins)' % (knm,a,se,n))
    print('   %s  |  %s' % (lab,'  |  '.join(res)))
