# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Standardise the dry side on the CAP window z 0 - R/L = 0.1441."""
import sys, csv
import numpy as np, forward as F, cluster as CL

R,L=0.40,2.7767
CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
zc=np.linspace(0,par.L,201)
print('cap window  z 0 - %.4f   (= R/L, R=%.2f L=%.4f)' % (CAP,R,L))

def model_em(kw,w,e,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,
                      psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    e2=np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]]
    return e2.mean()/S[(z>=.40)&(z<=.60)].mean()

PATHS={3:_os.path.join(DATA,'cells_profiles_wide_w3.csv'),
       5:_os.path.join(DATA,'cells_final_w5.csv')}
def meas_em(cond,w,e):
    """per-cell pooled end_mid, then the median across cells (what the pipeline reports)"""
    rows=list(csv.DictReader(open(PATHS[w],encoding='utf-8')))
    pc=[c for c in rows[0] if c.startswith('div_z')]
    zs=np.array([float(c[5:]) for c in pc])
    mE=((zs>=0)&(zs<=e))|((zs>=1-e)&(zs<=1)); mM=(zs>=.40)&(zs<=.60)
    def fl(v):
        try: return float(v)
        except: return np.nan
    out=[]
    for r in rows:
        if r['condition']!=cond: continue
        v=np.array([fl(r[c]) for c in pc])
        if not np.isfinite(v).all(): continue
        S=np.clip(v,0,None); m=S[mM].mean()
        if m<=0: continue
        out.append(S[mE].mean()/m)
    return float(np.median(out)), len(out)

print()
print('=== キャップ窓（z 0-0.1441）での標準値 ===')
print(' 線幅  下駄(膜70%)  モデル   実測DivIVA  実測null   モデル超過  実測超過  一致度')
for w in (3,5):
    b=model_em(dict(C=0.2957/V*one,M=0.7043/A*one),w,CAP)
    mo=model_em(dict(C=st['C'],M=st['M'],N=st['Nf']),w,CAP,Lx=par.L,zx=zc)
    me,n=meas_em('DivIVA_mCherry',w,CAP)
    nu,nn=meas_em('mCherry_null',w,CAP)
    print('  %d px   %.4f     %.4f    %.4f(n=%d)  %.4f    %+.4f   %+.4f   %.2f 倍'
          % (w,b,mo,me,n,nu,mo-b,me-b,(me-b)/(mo-b)))

print()
print('=== 膜結合率ごとの下駄（キャップ窓）===')
for w in (3,5):
    print('  線幅 %d px:' % w, end='')
    for f in (0.0,0.5,0.7,1.0):
        print('  膜%3.0f%% %.4f' % (100*f, model_em(dict(C=(1-f)/V*one,M=f/A*one),w,CAP)), end='')
    print()

print()
print('=== 参考: 窓を変えたときの一致度（線幅 3 px）===')
for e,lab in ((0.1441,'キャップ 0.1441'),(0.15,'0.15 (Cycle 5)'),(0.20,'0.20 (wet 側)')):
    b=model_em(dict(C=0.2957/V*one,M=0.7043/A*one),3,e)
    mo=model_em(dict(C=st['C'],M=st['M'],N=st['Nf']),3,e,Lx=par.L,zx=zc)
    me,_=meas_em('DivIVA_mCherry',3,e)
    print('  z 0-%-16s 下駄 %.4f  モデル %.4f  実測 %.4f  一致度 %.2f 倍'
          % (lab,b,mo,me,(me-b)/(mo-b)))
