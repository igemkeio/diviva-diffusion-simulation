# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Does the choice of pole window change the conclusions?
   Cycle 5 used w3 / z 0-0.15 ; the wet side (anl/final) uses w5 / z 0-0.20.
"""
import numpy as np, forward as F, cluster as CL

R,L=0.40,2.7767
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)

def em_of(S,z,e):     # e = tip window half-extent
    v=np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]]
    return v.mean()/S[(z>=.40)&(z<=.60)].mean()

def prof(kw,psf,w):
    pr=F.line_profile(z_model=zm,R=R,L=L,psf_fwhm_um=psf,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    return z,np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)

def base(f,psf,w,e):
    z,S=prof(dict(C=(1-f)/V*one,M=f/A*one),psf,w); return em_of(S,z,e)

print("=== 下駄は窓でどれだけ変わるか（PSF 0.870, L 2.7767） ===")
print("   膜結合率    w3 / z0-0.15      w5 / z0-0.20")
for f in (0.0,0.5,0.7,1.0):
    print("     %3.0f%%        %.4f            %.4f" % (100*f, base(f,0.870,3,0.15), base(f,0.870,5,0.20)))
print()
print("   実測 null      0.5211            0.5768")
print("   実測 DivIVA    0.7852            0.8031")
print()
for lab,w,e,meas in (('w3 / z0-0.15',3,0.15,0.7852),('w5 / z0-0.20',5,0.20,0.8031)):
    b=base(0.70,0.870,w,e)
    print("   %s : 下駄 %.4f  実測 %.4f  ->  局在の超過 %+.4f" % (lab,b,meas,meas-b))

print()
print("=== alpha(p) の対応表は窓で変わるか（モデルの発現量を振って測定） ===")
def alpha_at(p,w,e):
    bse=CL.variant(p=p,rho_n=4.1812,L=L,R=R); kg=CL._kg(bse)
    b0=base(0.70,0.870,w,e)
    Cs,Bs=[],[]
    for mult in (0.25,0.5,1.0,2.0,4.0,8.0):
        par=CL.derive(bse,N=1690.0*mult,k_grow=kg)
        st=CL.steady(n=201,par=par)
        # note: model fields are on their own grid
        pr=F.line_profile(C=st['C'],M=st['M'],N=st['Nf'],z_model=np.linspace(0,par.L,201),
                          R=R,L=par.L,psf_fwhm_um=0.870,line_w_px=w)
        v,zz=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
        S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
        E=np.r_[S[(zz>=0)&(zz<=e)],S[(zz>=1-e)&(zz<=1)]].mean()
        Cm=S[(zz>=.40)&(zz<=.60)].mean()
        Cs.append(Cm); Bs.append(E-b0*Cm)
    Cs=np.array(Cs);Bs=np.array(Bs);ok=Bs>0
    if ok.sum()<3: return np.nan
    Am=np.c_[np.log(Cs[ok]),np.ones(ok.sum())]
    co,*_=np.linalg.lstsq(Am,np.log(Bs[ok]),rcond=None)
    return co[0]
print("    p      1/(1-p)   w3/z0-0.15   w5/z0-0.20")
for p in (0.10,0.20,0.35,0.50):
    print("   %.2f      %.3f      %.3f        %.3f" % (p,1/(1-p),alpha_at(p,3,0.15),alpha_at(p,5,0.20)))
