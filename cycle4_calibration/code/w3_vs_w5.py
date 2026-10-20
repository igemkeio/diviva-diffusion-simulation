# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Apples-to-apples: pixel-grid model vs published-native measurement
   (converted between windows by the per-cell grid ratio), for BOTH line widths."""
import sys, csv
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L; Lc=L-2*R
V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par); zc=np.linspace(0,par.L,201)
def em(kw,w,e,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=w,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()
SRC={3:(_os.path.join(DATA,'cells_profiles_wide_w3.csv'),'end_mid',0.15),
     5:(_os.path.join(DATA,'cells_final_w5.csv'),None,0.20)}
def fl(v):
    try: return float(v)
    except: return np.nan
def meas(w,cond,e):
    """published native value at its own window, scaled by the per-cell grid ratio to window e"""
    path,col,e0=SRC[w]
    rows=list(csv.DictReader(open(path,encoding='utf-8')))
    pc=[c for c in rows[0] if c.startswith('div_z')]; zs=np.array([float(c[5:]) for c in pc])
    def mask(x): return ((zs>=0)&(zs<=x))|((zs>=1-x)&(zs<=1))
    m0,me_,mM=mask(e0),mask(e),(zs>=.40)&(zs<=.60)
    out=[]
    for r in rows:
        if r['condition']!=cond: continue
        nat = fl(r[col]) if col else (fl(r['poleA_mid'])+fl(r['poleB_mid']))/2
        if not np.isfinite(nat): continue
        v=np.array([fl(r[c]) for c in pc])
        if not np.isfinite(v).all(): continue
        S=np.clip(v,0,None); mid=S[mM].mean()
        if mid<=0: continue
        g0,ge=S[m0].mean()/mid,S[me_].mean()/mid
        if g0<=0: continue
        out.append(nat*ge/g0)
    return float(np.median(out)),len(out)
print('同じ扱い（モデル=画素格子 / 実測=公表値を窓の比で換算）')
print('  線幅 窓         下駄    モデル  実測    モデル超過 実測超過  一致度')
for w in (3,5):
    for e,lab in ((CAP,'0.144(cap)'),(0.20,'0.20 (wet)')):
        b=em(dict(C=0.2957/V*one,M=0.7043/A*one),w,e)
        mo=em(dict(C=st['C'],M=st['M'],N=st['Nf']),w,e,Lx=par.L,zx=zc)
        me,n=meas(w,'DivIVA_mCherry',e)
        print('   %d  %s  %.4f  %.4f  %.4f  %+.4f   %+.4f   %.2f 倍 (n=%d)'%(w,lab,b,mo,me,mo-b,me-b,(me-b)/(mo-b),n))
