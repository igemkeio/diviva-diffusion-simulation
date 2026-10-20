# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Dry-side standard: line width 5 px (matching anl/final), tip window z 0-R/L.
   Everything - model and measurement - is put on the SAME 0.02 z-grid that the
   published profile csv uses, so the two are treated identically."""
import sys, csv
import numpy as np, forward as F, cluster as CL

R,L=0.40,2.7767; CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
ZG=np.round(np.arange(-0.10,1.1001,0.02),2)          # the csv's grid
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
zc=np.linspace(0,par.L,201)

def prof(kw,w,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,
                      psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    return z,np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)

def em_native(z,S,e):
    return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()
def em_grid(z,S,e):
    G=np.interp(ZG,z,S); mE=((ZG>=0)&(ZG<=e))|((ZG>=1-e)&(ZG<=1)); mM=(ZG>=.40)&(ZG<=.60)
    return G[mE].mean()/G[mM].mean()

PATH={3:_os.path.join(DATA,'cells_profiles_wide_w3.csv'),
      5:_os.path.join(DATA,'cells_final_w5.csv')}
def meas(cond,w,e):
    rows=list(csv.DictReader(open(PATH[w],encoding='utf-8')))
    pc=[c for c in rows[0] if c.startswith('div_z')]; zs=np.array([float(c[5:]) for c in pc])
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
        if m>0: out.append(S[mE].mean()/m)
    return float(np.median(out)),len(out)

print('=== モデルを「画素格子」と「csv の 0.02 格子」で測ったときの差 ===')
print('  線幅 窓       画素格子   0.02格子    差')
for w in (3,5):
    for e,lab in ((CAP,'cap 0.1441'),(0.20,'0.20     ')):
        z,S=prof(dict(C=0.2957/V*one,M=0.7043/A*one),w)
        print('   %d  %s  %.4f    %.4f   %+.4f'%(w,lab,em_native(z,S,e),em_grid(z,S,e),em_grid(z,S,e)-em_native(z,S,e)))
print()
print('=== dry 標準 = 線幅 5 px / 窓 z 0-0.1441 （すべて 0.02 格子）===')
base={}
for f in (0.0,0.5,0.7,1.0):
    z,S=prof(dict(C=(1-f)/V*one,M=f/A*one),5); base[f]=em_grid(z,S,CAP)
for f in (0.0,0.5,0.7,1.0):
    print('   膜 %3.0f%%  下駄 %.4f   細胞質基準の見かけの比 %.3f'%(100*f,base[f],base[f]/base[0.0]))
z,S=prof(dict(C=0.2957/V*one,M=0.7043/A*one),5); b70=em_grid(z,S,CAP)
z,S=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),5,Lx=par.L,zx=zc); mo=em_grid(z,S,CAP)
me,nme=meas('DivIVA_mCherry',5,CAP); nu,nnu=meas('mCherry_null',5,CAP)
print()
print('   下駄（膜 70.43%%）      %.4f'%b70)
print('   モデル（p=0.17 定常）  %.4f   超過 %+.4f'%(mo,mo-b70))
print('   実測 DivIVA-mCherry    %.4f (n=%d)   超過 %+.4f'%(me,nme,me-b70))
print('   実測 mCherry_null      %.4f (n=%d)'%(nu,nnu))
print('   一致度  %.4f / %.4f = %.2f 倍'%(me-b70,mo-b70,(me-b70)/(mo-b70)))
