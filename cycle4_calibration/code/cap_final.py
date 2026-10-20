# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Cap window z 0-R/L as the dry-side standard. Canonical measured values
   (0.7852 / 0.5211) come from the per-cell native pixel grid, which is the
   same treatment the model gets - so they are the ones to use."""
import sys
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
zc=np.linspace(0,par.L,201)
def em(kw,w,e,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,
                      psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()
MEAS, NULLM = 0.7852, 0.5211
print('キャップ窓 z 0-%.4f (=R/L)   線幅 3 px   PSF 0.870 um   L %.4f R %.2f'%(CAP,L,R))
print()
print('  膜結合率   下駄(cap)   下駄(0.15)   差')
for f in (0.0,0.5,0.7,1.0):
    a=em(dict(C=(1-f)/V*one,M=f/A*one),3,CAP); b=em(dict(C=(1-f)/V*one,M=f/A*one),3,0.15)
    print('   %3.0f%%      %.4f      %.4f     %+.5f'%(100*f,a,b,a-b))
mo=em(dict(C=st['C'],M=st['M'],N=st['Nf']),3,CAP,Lx=par.L,zx=zc)
b70=em(dict(C=0.2957/V*one,M=0.7043/A*one),3,CAP)
print()
print('  モデル(p=0.17 定常)  %.4f   超過 %+.4f'%(mo,mo-b70))
print('  実測 DivIVA-mCherry  %.4f   超過 %+.4f'%(MEAS,MEAS-b70))
print('  実測 mCherry_null    %.4f'%NULLM)
print('  一致度  %.4f / %.4f = %.2f 倍'%(MEAS-b70,mo-b70,(MEAS-b70)/(mo-b70)))
