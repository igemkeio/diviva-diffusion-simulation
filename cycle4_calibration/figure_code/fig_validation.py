# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
_os.makedirs(FIGS, exist_ok=True)
if _HERE not in _sys.path: _sys.path.insert(0, _os.path.join(ROOT, 'code'))
# ---------------------------------------------------------------------------
"""Section 5 figure: the steady state at the calibrated values, and what the
microscope would see.
(a) the three fields at steady state - where the localisation actually lives
(b) through the forward model, against the measured profile
Values: R 0.40, L 2.7767, p 0.17, rho_n 4.1812, PSF 0.870, line width 5 px.
"""
import csv
import numpy as np, forward as F, cluster as CL
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
FIN=_os.path.join(DATA,'cells_final_w5.csv')
OUT=_os.path.join(FIGS,'cal_fig2_validation.png')
SURF,INK,INK2,MUTED='#fcfcfb','#0b0b0b','#52514e','#8a8882'
CYT,MON,CLU,TOT,PRE='#2a78d6','#1baf7a','#eb6834','#0b0b0b','#9b6fd4'
R,L,PSF,W=0.40,2.7767,0.870,5
par=CL.variant(p=0.17,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
g=st['g']; z=g['z']/L
V,A=g['V'],g['A']
nC,nM,nN=st['C']*V, st['M']*A, st['Nf']*A      # molecules per grid cell
tot=nC+nM+nN
print('f_cyt %.3f  f_mono %.3f  f_clu %.3f   極にあるクラスター %.1f%%   膜の極濃縮 %.3f  クラスターの極濃縮 %.3f'
      %(st['f_cyt'],st['f_mono'],st['f_clu'],100*st['clu_pole_frac'],st['enrichment'],st['enr_clu']))
Lc=L-2*R; Vt=np.pi*R*R*Lc+4/3*np.pi*R**3; At=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
def prof(kw,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=PSF,line_w_px=W,**kw)
    v,zz=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return zz,S/np.median(S[(zz>=.35)&(zz<=.65)])
zmod,Smod=prof(dict(C=st['C'],M=st['M'],N=st['Nf']),Lx=par.L,zx=np.linspace(0,par.L,201))
# pre-calibration model: p 0.5, rho_n 2.370, L 3.0 um (Cycle 2/4), same imaging
pre=CL.variant(p=0.50,rho_n=2.3696,L=3.0,R=R); sp=CL.steady(n=201,par=pre)
zpre,Spre=prof(dict(C=sp['C'],M=sp['M'],N=sp['Nf']),Lx=pre.L,zx=np.linspace(0,pre.L,201))
zbas,Sbas=prof(dict(C=0.2957/Vt*one,M=0.7043/At*one))
rows=[r for r in csv.DictReader(open(FIN,encoding='utf-8')) if r['condition']=='DivIVA_mCherry']
pc=[c for c in rows[0] if c.startswith('div_z')]; zs=np.array([float(c[5:]) for c in pc])
def fl(v):
    try: return float(v)
    except: return np.nan
P=np.array([[fl(r[c]) for c in pc] for r in rows]); P=P[np.isfinite(P).all(axis=1)]
n0=np.median(P[:,(zs>=.35)&(zs<=.65)],axis=1); P=P[n0>0]/n0[n0>0][:,None]
mes=np.median(P,axis=0); q1=np.percentile(P,25,axis=0); q3=np.percentile(P,75,axis=0)

fig,(a1,a2)=plt.subplots(1,2,figsize=(11.4,4.5),dpi=200)
fig.patch.set_facecolor(SURF)
for a in (a1,a2):
    a.set_facecolor(SURF)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    for sp in ('left','bottom'): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2,labelsize=9)
    a.axvspan(0,0.1441,color='#efeee9',zorder=0); a.axvspan(0.8559,1,color='#efeee9',zorder=0)

a1.plot(z,tot,color=TOT,lw=2.2,zorder=5,label='all DivIVA')
a1.plot(z,nC,color=CYT,lw=1.8,label='cytoplasm C   (%.0f%%)'%(100*st['f_cyt']))
a1.plot(z,nM,color=MON,lw=1.8,label='membrane monomer M   (%.0f%%)'%(100*st['f_mono']))
a1.plot(z,nN,color=CLU,lw=1.8,label='cluster N   (%.0f%%)'%(100*st['f_clu']))
a1.set_xlim(-.02,1.02); a1.set_ylim(0,13.5)
a1.set_xlabel('position along the cell  (z / L)',fontsize=10,color=INK2)
a1.set_ylabel('molecules per grid cell',fontsize=10,color=INK2)
a1.set_title('(a) steady state — only the clusters are localised',fontsize=11.5,color=INK,loc='left',pad=10)
a1.legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper center',ncol=1,bbox_to_anchor=(.5,1.0))


a2.fill_between(zs,q1,q3,color=MUTED,alpha=.18,lw=0,label='measured  25–75%')
a2.plot(zs,mes,color=INK,lw=2.2,zorder=5,label='measured  median  (n=%d)'%len(P))
a2.plot(zpre,Spre,':',color=PRE,lw=1.9,zorder=4,label='model, before calibration  (L = 3.0 µm, p = 0.5)')
a2.plot(zmod,Smod,color=CLU,lw=2.0,zorder=4,label='model, after calibration  (L = 2.78 µm, p = 0.17)')
a2.plot(zbas,Sbas,'--',color=MON,lw=1.6,zorder=3,label='no localisation  (membrane 70%)')
a2.set_xlim(-.02,1.02); a2.set_ylim(0,1.25)
a2.set_xlabel('position along the cell  (z / L)',fontsize=10,color=INK2)
a2.set_ylabel('intensity / mid-cell',fontsize=10,color=INK2)
a2.set_title('(b) through the forward model, against the data',fontsize=11.5,color=INK,loc='left',pad=10)
a2.legend(frameon=False,fontsize=8.2,labelcolor=INK2,loc='lower center')
a2.text(.072,1.17,'tip window',fontsize=8,color=MUTED,ha='center')

def em(zz,S,e=0.1441):
    return np.r_[S[(zz>=0)&(zz<=e)],S[(zz>=1-e)&(zz<=1)]].mean()/S[(zz>=.40)&(zz<=.60)].mean()
print('end_mid  model %.4f  pre-cal %.4f  baseline %.4f'%(em(zmod,Smod),em(zpre,Spre),em(zbas,Sbas)))
fig.text(.012,-.030,
 'R = 0.40 µm, L = 2.78 µm, p = 0.17, ρ$_n$ = 4.181, effective PSF 0.870 µm, line width 5 px, tip window z 0–0.144. '
 'Panel (a): C and M come out flat to 3×10$^{-6}$ — all of the localisation is in N, which steps at z = R/L = 0.144 because the curvature term ρ$_n$ steps at the cap boundary; 69% of cluster molecules sit in the caps.\n'
 'Panel (b): all curves are normalised to mid-cell; the shaded bands are the tip window. The two model curves differ ONLY through the cell length: re-deriving ρ$_n$ for the new p leaves the steady state unchanged.',
 fontsize=8,color=MUTED,ha='left')
fig.tight_layout(rect=[0,.05,1,1])
fig.savefig(OUT,facecolor=SURF,bbox_inches='tight')
print('wrote',OUT)
