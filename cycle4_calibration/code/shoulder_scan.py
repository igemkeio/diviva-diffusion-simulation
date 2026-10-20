# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
import sys
import numpy as np, forward as F, cluster as CL

def prof(kind, psf, w, L=2.7767, R=0.40, f=None, fields=None):
    Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
    zm=np.linspace(0,L,2001); one=np.ones_like(zm)
    if fields is not None:
        kw=dict(C=fields[0],M=fields[1],N=fields[2]); zm=fields[3]
    elif kind=='cyt': kw=dict(C=one/V)
    elif kind=='mem': kw=dict(M=one/A)
    else:             kw=dict(C=(1-f)*one/V, M=f*one/A)
    nx=int(np.ceil((2*R+6*psf)/F.PX_UM/2))*2+1
    if w>nx: return None,None
    pr=F.line_profile(z_model=zm,R=R,L=L,psf_fwhm_um=psf,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']
    k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return z, S/np.median(S[(z>=.35)&(z<=.65)])

def shoulder(z,S):
    if z is None: return np.nan
    d=np.minimum(z,1-z); m=(d>=0.05)&(d<=0.35)
    return S[m].max()

print("=== shoulder height (max of the profile outside mid-cell, mid-cell = 1) ===")
print("    > 1.000 means the profile rises above mid-cell before falling\n")
print("  PSF[um]    w=1     w=3     w=5     w=9     <- UNIFORM MEMBRANE (no localisation)")
for psf in (0.0,0.15,0.25,0.35,0.50,0.65,0.87,1.05):
    row=[]
    for w in (1,3,5,9):
        z,S=prof('mem',psf,w); row.append(shoulder(z,S))
    print("   %.2f   " % psf + "  ".join(("%6.3f"%v if np.isfinite(v) else "    - ") for v in row))

print()
print("  PSF[um]    w=1     w=3     w=5     w=9     <- UNIFORM CYTOPLASM")
for psf in (0.0,0.35,0.87):
    row=[shoulder(*prof('cyt',psf,w)) for w in (1,3,5,9)]
    print("   %.2f   " % psf + "  ".join(("%6.3f"%v if np.isfinite(v) else "    - ") for v in row))

print()
print("=== at the measured conditions (PSF 0.87, w 5): how much membrane is needed? ===")
for f in (0.0,0.3,0.5,0.7,0.9,1.0):
    z,S=prof('mix',0.87,5,f=f); print("   membrane fraction %.1f -> shoulder %.3f" % (f, shoulder(z,S)))

print()
print("=== cell geometry (uniform membrane, PSF 0.87, w 5) ===")
for L in (2.2,2.5,2.78,3.2,4.0,5.0):
    z,S=prof('mem',0.87,5,L=L); print("   L=%.2f um -> shoulder %.3f" % (L, shoulder(z,S)))
for Rv in (0.30,0.35,0.40,0.45,0.50):
    z,S=prof('mem',0.87,5,R=Rv); print("   R=%.2f um -> shoulder %.3f" % (Rv, shoulder(z,S)))

print()
print("=== with real polar clusters (Cycle 5 steady state) ===")
par=CL.variant(p=0.17,rho_n=4.1812,L=2.7767,R=0.40); st=CL.steady(n=201,par=par)
zc=np.linspace(0,par.L,201)
for psf in (0.35,0.87):
    for w in (3,5):
        z,S=prof(None,psf,w,fields=(st['C'],st['M'],st['Nf'],zc))
        print("   PSF %.2f w%d -> shoulder %.3f" % (psf,w,shoulder(z,S)))

print()
print("=== threshold: below which PSF does a uniform membrane alone make a shoulder? ===")
for w in (3,5):
    lo,hi=0.0,1.2
    for _ in range(40):
        mid=(lo+hi)/2
        z,S=prof('mem',mid,w)
        if shoulder(z,S)>1.0: lo=mid
        else: hi=mid
    print("   line width %d px : uniform membrane shows a shoulder for PSF < %.2f um" % (w,(lo+hi)/2))
