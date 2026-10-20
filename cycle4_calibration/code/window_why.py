# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Why does moving the tip window change the verdict so much?"""
import sys, csv
import numpy as np, forward as F, cluster as CL

R,L=0.40,2.7767
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
zc=np.linspace(0,par.L,201)

def prof(kw,w,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return z,S/np.median(S[(z>=.35)&(z<=.65)])

def meas(cond,path,w):
    rows=list(csv.DictReader(open(path,encoding='utf-8')))
    pc=[c for c in rows[0] if c.startswith('div_z')]
    zs=np.array([float(c[5:]) for c in pc])
    def fl(v):
        try: return float(v)
        except: return np.nan
    sel=[r for r in rows if r['condition']==cond]
    P=np.array([[fl(r[c]) for c in pc] for r in sel]); P=P[np.isfinite(P).all(axis=1)]
    n=np.median(P[:,(zs>=.35)&(zs<=.65)],axis=1); P=P[n>0]/n[n>0][:,None]
    return zs,np.median(P,axis=0)

W3=_os.path.join(DATA,'cells_profiles_wide_w3.csv')
W5=_os.path.join(DATA,'cells_final_w5.csv')

def em(z,S,e):
    v=np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]]
    return v.mean()/S[(z>=.40)&(z<=.60)].mean()

print("=== 窓と線幅を別々に動かす ===")
print("  設定              下駄    モデル  実測    モデル超過 実測超過  一致度")
for w,e,path in ((3,0.15,W3),(3,0.20,W3),(5,0.15,W5),(5,0.20,W5)):
    zb,Sb=prof(dict(C=0.2957/V*one,M=0.7043/A*one),w)
    zmo,Smo=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),w,Lx=par.L,zx=zc)
    zme,Sme=meas('DivIVA_mCherry',path,w)
    b,mo,me=em(zb,Sb,e),em(zmo,Smo,e),em(zme,Sme,e)
    print("  w%d / z0-%.2f      %.4f  %.4f  %.4f   %+.4f   %+.4f   %.2f"
          % (w,e,b,mo,me,mo-b,me-b,(me-b)/(mo-b)))

print()
print("=== 超過がどこにあるか（下駄からの差を z ごとに、w3）===")
zb,Sb=prof(dict(C=0.2957/V*one,M=0.7043/A*one),3)
zmo,Smo=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),3,Lx=par.L,zx=zc)
zme,Sme=meas('DivIVA_mCherry',W3,3)
gb=np.interp(zme,zb,Sb); gm=np.interp(zme,zmo,Smo)
d_mo=gm-gb; d_me=Sme-gb
print("   z の帯        モデルの超過   実測の超過    実測/モデル")
for lo,hi in ((0,0.05),(0.05,0.10),(0.10,0.15),(0.15,0.20),(0.20,0.25)):
    m=((zme>=lo)&(zme<hi))|((zme>1-hi)&(zme<=1-lo))
    a,b2=d_mo[m].mean(),d_me[m].mean()
    print("   %.2f - %.2f      %+.4f        %+.4f       %.2f" % (lo,hi,a,b2,b2/a if a!=0 else np.nan))
