# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
def prof(kw,w,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    return z,np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
def em(z,S,e): return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()
def base(f,w,e):
    z,S=prof(dict(C=(1-f)/V*one,M=f/A*one),w); return em(z,S,e)
def alpha_at(p,w,e):
    bse=CL.variant(p=p,rho_n=4.1812,L=L,R=R); kg=CL._kg(bse); b0=base(0.7043,w,e)
    Cs,Bs=[],[]
    for mult in (0.25,0.5,1.0,2.0,4.0,8.0):
        par=CL.derive(bse,N=1690.0*mult,k_grow=kg); st=CL.steady(n=201,par=par)
        z,S=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),w,Lx=par.L,zx=np.linspace(0,par.L,201))
        E=np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean(); Cm=S[(z>=.40)&(z<=.60)].mean()
        Cs.append(Cm); Bs.append(E-b0*Cm)
    Cs=np.array(Cs);Bs=np.array(Bs);ok=Bs>0
    if ok.sum()<3: return np.nan
    M=np.c_[np.log(Cs[ok]),np.ones(ok.sum())]
    co,*_=np.linalg.lstsq(M,np.log(Bs[ok]),rcond=None); return co[0]
print('    p     1/(1-p)   w5/cap0.144   w5/0.20    w3/0.15')
for p in (0.10,0.20,0.35,0.50):
    print('   %.2f     %.3f      %.3f        %.3f      %.3f'%(p,1/(1-p),alpha_at(p,5,CAP),alpha_at(p,5,0.20),alpha_at(p,3,0.15)))
