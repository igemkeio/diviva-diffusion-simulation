# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
DATA  = _os.path.join(ROOT, 'data')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""alpha fitted separately in each of the 5 IPTG fields (DivIVA-mCherry),
   instead of pooling. Binning by the +-3 px parallel lines."""
import csv
import numpy as np
FIN=_os.path.join(DATA,'cells_final_w5.csv')
INT=_os.path.join(DATA,'cells_integrated.csv')
rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1.0)); mM=(zs>=.40)&(zs<=.60)
def fl(v):
    try: return float(v)
    except: return np.nan
SIDE={r['cell_uid']:fl(r['expr_side_adu']) for r in csv.DictReader(open(INT,encoding='utf-8'))}
d={}
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    v=np.array([fl(r[c]) for c in cols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None); c=S[mM].mean(); e=S[mE].mean(); x=SIDE.get(r['cell_uid'],np.nan)
    if c<=0 or not np.isfinite(x) or x<=0: continue
    d.setdefault(r['rep'],[]).append((c,e,x))
def fit(sub,b,nq):
    sub=sorted(sub,key=lambda t:t[2]); lx,ly=[],[]
    for g in np.array_split(np.arange(len(sub)),nq):
        if len(g)==0: continue
        Cs=np.array([sub[i][0] for i in g]); Es=np.array([sub[i][1] for i in g])
        Bm,Cm=np.median(Es-b*Cs),np.median(Cs)
        if Bm<=0 or Cm<=0: continue
        lx.append(np.log(Cm)); ly.append(np.log(Bm))
    if len(lx)<3: return np.nan,np.nan,len(lx)
    lx=np.array(lx);ly=np.array(ly);A=np.c_[lx,np.ones(len(lx))]
    co,*_=np.linalg.lstsq(A,ly,rcond=None); res=ly-A@co
    s2=(res**2).sum()/max(len(lx)-2,1)
    return co[0],np.sqrt(s2*np.linalg.inv(A.T@A)[0,0]),len(lx)
def p_of(a): return 1-1/a if (np.isfinite(a) and a>1) else (0.0 if np.isfinite(a) else np.nan)
BS=[('実測 null',0.5779),('モデル膜0%',0.6377),('モデル膜70%',0.7227)]
print('=== 5 条件を別々に当てはめた α（階級 5、±3 px 平行線で並べる）===')
print(' IPTG      n   明るさの幅(90/10)   %s'%('  '.join('%-18s'%l for l,_ in BS)))
for rp in sorted(d):
    sub=d[rp]; xs=np.array([t[2] for t in sub])
    cells=['%s'%(('%.3f ± %.3f'%fit(sub,b,5)[:2]) if np.isfinite(fit(sub,b,5)[0]) else '   undefined  ') for _,b in BS]
    print(' 10^%d nM  %3d        %.2f 倍        %s'
          %(int(rp)-1,len(sub),np.percentile(xs,90)/np.percentile(xs,10),'  '.join('%-18s'%c for c in cells)))
allsub=[t for rp in d for t in d[rp]]
xs=np.array([t[2] for t in allsub])
cells=['%.3f ± %.3f'%fit(allsub,b,10)[:2] for _,b in BS]
print(' ------- プール %3d        %.2f 倍        %s'
      %(len(allsub),np.percentile(xs,90)/np.percentile(xs,10),'  '.join('%-18s'%c for c in cells)))
print()
print('=== p に直す（下駄 = 実測 null）===')
vals=[]
for rp in sorted(d):
    a,se,nb=fit(d[rp],0.5779,5); vals.append(a)
    print('  10^%d nM   α %.3f ± %.3f   → p %.2f   (階級 %d)'%(int(rp)-1,a,se,p_of(a),nb))
v=np.array([x for x in vals if np.isfinite(x)])
print('  5 条件の広がり: α %.3f 〜 %.3f (中央値 %.3f)  → p %.2f 〜 %.2f'
      %(v.min(),v.max(),np.median(v),p_of(v.min()),p_of(v.max())))
a,se,_=fit(allsub,0.5779,10)
print('  プール:        α %.3f ± %.3f   → p %.2f'%(a,se,p_of(a)))
print()
print('=== 5 条件の α の散らばりは、誤差で説明できるか ===')
ses=[fit(d[rp],0.5779,5)[1] for rp in sorted(d)]
ses=np.array([s for s,a in zip(ses,vals) if np.isfinite(a)])
w=1/ses**2; wm=(v*w).sum()/w.sum()
chi2=((v-wm)**2*w).sum(); dof=len(v)-1
print('  重み付き平均 α = %.3f,  chi2 = %.2f / dof %d = %.2f'%(wm,chi2,dof,chi2/dof))
print('  → chi2/dof ≈ 1 なら「ばらつきは誤差の範囲」、>> 1 なら「条件ごとに本当に違う」')
