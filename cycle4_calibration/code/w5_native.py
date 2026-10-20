# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""w5 / cap window, both sides on the PIXEL grid (the pipeline's own convention).
   The w5 csv publishes poleA/poleB at z 0-0.20 only, so the cap-window value is
   obtained by a per-cell ratio taken on the csv's own 0.02 profile grid."""
import sys, csv
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par); zc=np.linspace(0,par.L,201)
def em(kw,w,e,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()

P=_os.path.join(DATA,'cells_final_w5.csv')
rows=list(csv.DictReader(open(P,encoding='utf-8')))
pc=[c for c in rows[0] if c.startswith('div_z')]; zs=np.array([float(c[5:]) for c in pc])
def fl(v):
    try: return float(v)
    except: return np.nan
def meas(cond):
    pub=[];est=[]
    m20=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1)); mcap=((zs>=0)&(zs<=CAP))|((zs>=1-CAP)&(zs<=1))
    mM=(zs>=.40)&(zs<=.60)
    for r in rows:
        if r['condition']!=cond: continue
        a,b=fl(r['poleA_mid']),fl(r['poleB_mid'])
        if not (np.isfinite(a) and np.isfinite(b)): continue
        v=np.array([fl(r[c]) for c in pc])
        if not np.isfinite(v).all(): continue
        S=np.clip(v,0,None); mid=S[mM].mean()
        if mid<=0: continue
        g20,gc=S[m20].mean()/mid,S[mcap].mean()/mid
        if g20<=0: continue
        pub.append((a+b)/2); est.append((a+b)/2*gc/g20)
    return float(np.median(pub)),float(np.median(est)),len(pub)

print('dry 標準 = 線幅 5 px / 先端 z 0-%.4f (= R/L)  PSF 0.870  L 2.7767  R 0.40'%CAP)
print()
print('  膜結合率  下駄      細胞質基準の見かけの比')
for f in (0.0,0.5,0.7,1.0):
    b=em(dict(C=(1-f)/V*one,M=f/A*one),5,CAP)
    if f==0.0: b0=b
    print('    %3.0f%%    %.4f    %.3f'%(100*f,b,b/b0))
b70=em(dict(C=0.2957/V*one,M=0.7043/A*one),5,CAP)
mo =em(dict(C=st['C'],M=st['M'],N=st['Nf']),5,CAP,Lx=par.L,zx=zc)
print()
for cond in ('DivIVA_mCherry','mCherry_null','DivIVA_GFP','GFP_null'):
    p20,pcap,n=meas(cond)
    print('  実測 %-15s  z0-0.20 %.4f (公表値)   z0-0.144 %.4f (比で換算)  n=%d'%(cond,p20,pcap,n))
me=meas('DivIVA_mCherry')[1]
print()
print('  下駄（膜 70.43%%）      %.4f'%b70)
print('  モデル（p=0.17 定常）  %.4f   超過 %+.4f'%(mo,mo-b70))
print('  実測 DivIVA-mCherry    %.4f   超過 %+.4f'%(me,me-b70))
print('  一致度 %.2f 倍'%((me-b70)/(mo-b70)))
print()
print('=== 参考: 線幅 5 px で窓を変えたとき ===')
for e,lab in ((CAP,'0.144 (cap)'),(0.20,'0.20 (wet)')):
    b=em(dict(C=0.2957/V*one,M=0.7043/A*one),5,e); m=em(dict(C=st['C'],M=st['M'],N=st['Nf']),5,e,Lx=par.L,zx=zc)
    mm=meas('DivIVA_mCherry')[1 if e==CAP else 0]
    print('  z 0-%-12s 下駄 %.4f  モデル %.4f  実測 %.4f  超過 %+.4f/%+.4f  一致度 %.2f 倍'
          %(lab,b,m,mm,m-b,mm-b,(mm-b)/(m-b)))
