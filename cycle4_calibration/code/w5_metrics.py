# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""pole_mid and f20 at line width 5 px (model vs measurement, same definitions)."""
import sys, csv
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L
Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par); zc=np.linspace(0,par.L,201)
def prof(kw,w,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    return z,np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
def pole_mid(z,S):
    p=np.r_[S[(z>=.15)&(z<=.35)],S[(z>=.65)&(z<=.85)]]
    return p.mean()/S[(z>=.40)&(z<=.60)].mean()
def f20(z,S):
    m=(z>=0)&(z<=1); s=np.sort(S[m])[::-1]
    k=max(1,int(round(.20*len(s))))
    return s[:k].sum()/s.sum()
P=_os.path.join(DATA,'cells_final_w5.csv')
rows=[r for r in csv.DictReader(open(P,encoding='utf-8')) if r['condition']=='DivIVA_mCherry']
pc=[c for c in rows[0] if c.startswith('div_z')]; zs=np.array([float(c[5:]) for c in pc])
def fl(v):
    try: return float(v)
    except: return np.nan
pm=[];f2=[]
for r in rows:
    v=np.array([fl(r[c]) for c in pc])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None)
    if S[(zs>=.40)&(zs<=.60)].mean()<=0: continue
    pm.append(pole_mid(zs,S)); f2.append(f20(zs,S))
zM,SM=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),5,Lx=par.L,zx=zc)
zMg=np.round(np.arange(-.10,1.1001,.02),2); SMg=np.interp(zMg,zM,SM)
print('線幅 5 px                  モデル(画素格子)  モデル(0.02格子)   実測(0.02格子, n=%d)'%len(pm))
print('  pole_mid (0.15-0.35)       %.4f           %.4f            %.4f'%(pole_mid(zM,SM),pole_mid(zMg,SMg),np.median(pm)))
print('  f20                        %.4f           %.4f            %.4f'%(f20(zM,SM),f20(zMg,SMg),np.median(f2)))
