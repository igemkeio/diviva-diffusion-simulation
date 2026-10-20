# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/figure_code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
_os.makedirs(FIGS, exist_ok=True)
if _HERE not in _sys.path: _sys.path.insert(0, _os.path.join(ROOT, 'code'))        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Section 4 figure: what was actually plotted to get alpha.
(a) per-cell X (independent brightness) vs B (pole excess)  - the binning variable
(b) the fit itself: log C vs log B on the 10 pooled bins, slope = alpha
Convention: anl/final window (line width 5 px, tip z 0-0.20), pooled 5 IPTG fields.
"""
import csv
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIN=_os.path.join(DATA,'cells_final_w5.csv')
INT=_os.path.join(DATA,'cells_integrated.csv')
OUT=_os.path.join(FIGS,'cal_fig_alpha.png')
SURF,INK,INK2,MUTED='#fcfcfb','#0b0b0b','#52514e','#8a8882'
ACC,ACC2,GREY='#2a78d6','#eb6834','#c9c7c1'
B_NULL=0.5779                      # measured same-colour null, same grid (4-3)

def fl(v):
    try: return float(v)
    except: return np.nan
rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1.0)); mM=(zs>=.40)&(zs<=.60)
X={r['cell_uid']:fl(r['expr_side_adu']) for r in csv.DictReader(open(INT,encoding='utf-8'))}
C_,E_,Xv=[],[],[]
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    v=np.array([fl(r[c]) for c in cols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None); C=S[mM].mean(); E=S[mE].mean(); x=X.get(r['cell_uid'],np.nan)
    if C<=0 or not np.isfinite(x) or x<=0: continue
    C_.append(C); E_.append(E); Xv.append(x)
C_=np.array(C_); E_=np.array(E_); Xv=np.array(Xv)
B_=E_-B_NULL*C_
n=len(C_)

# ---- 10 bins by X (the independent variable) ----
order=np.argsort(Xv); bins=np.array_split(order,10)
bx=np.array([np.median(Xv[b]) for b in bins])
bB=np.array([np.median(B_[b]) for b in bins])
bC=np.array([np.median(C_[b]) for b in bins])
ok=bB>0
A=np.c_[np.log(bC[ok]),np.ones(ok.sum())]
co,_,_,_=np.linalg.lstsq(A,np.log(bB[ok]),rcond=None)
pred=A@co; dof=max(1,ok.sum()-2)
cov=((np.log(bB[ok])-pred)**2).sum()/dof*np.linalg.pinv(A.T@A)
al,se=co[0],np.sqrt(cov[0,0])
pv=1-1/al if al>1 else 0.0

fig,(a1,a2)=plt.subplots(1,2,figsize=(11.2,4.5),dpi=200)
fig.patch.set_facecolor(SURF)
for a in (a1,a2):
    a.set_facecolor(SURF)
    for sp in ('top','right'): a.spines[sp].set_visible(False)
    for sp in ('left','bottom'): a.spines[sp].set_color(MUTED)
    a.tick_params(colors=INK2, labelsize=9)

# ---------- (a) X vs B, per cell ----------
pos=B_>0
a1.scatter(Xv[pos],B_[pos],s=7,color=ACC,alpha=.22,linewidths=0,label='cell  (B > 0)')
a1.scatter(Xv[~pos],np.full((~pos).sum(),1.0),s=7,color=GREY,alpha=.5,linewidths=0,
           marker='v',label='cell  (B ≤ 0, drawn at 1)')
a1.plot(bx,bB,'o-',color=ACC2,ms=6,lw=1.8,zorder=5,label='median of 10 bins')
a1.set_xscale('log'); a1.set_yscale('log')
a1.set_xlabel('X — brightness from the ±3 px parallel lines  [ADU]',fontsize=10,color=INK2)
a1.set_ylabel('B = E − b·C   pole excess  [ADU]',fontsize=10,color=INK2)
a1.set_title('(a) what X is for: it only sets the bins',fontsize=11.5,color=INK,loc='left',pad=10)
a1.legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper left')
a1.text(.97,.04,'n = %d cells, 5 IPTG fields pooled\nb = %.4f (measured null)'%(n,B_NULL),
        transform=a1.transAxes,ha='right',va='bottom',fontsize=8,color=MUTED)

# ---------- (b) the fit ----------
a2.plot(bC[ok],bB[ok],'o',color=ACC2,ms=7,zorder=5,label='10 bins (median)')
xs=np.linspace(np.log(bC[ok].min()),np.log(bC[ok].max()),50)
a2.plot(np.exp(xs),np.exp(co[1]+co[0]*xs),'-',color=INK,lw=1.8,zorder=4,
        label='fit   α = %.3f ± %.3f'%(al,se))
a2.plot(np.exp(xs),np.exp(np.log(bB[ok]).mean()-1.0*np.log(bC[ok]).mean()+1.0*xs),'--',
        color=MUTED,lw=1.4,zorder=3,label='α = 1  (p = 0, no self-help)')
a2.plot(np.exp(xs),np.exp(np.log(bB[ok]).mean()-1.886*np.log(bC[ok]).mean()+1.886*xs),':',
        color=ACC,lw=1.6,zorder=3,label='α = 1.886  (p = 0.5)')
a2.set_xscale('log'); a2.set_yscale('log')
a2.set_xlabel('C — mid-cell intensity  [ADU]   (∝ cytoplasmic concentration)',fontsize=10,color=INK2)
a2.set_ylabel('B — pole excess  [ADU]',fontsize=10,color=INK2)
a2.set_title('(b) the slope is what we are after:  α = 1/(1−p)',fontsize=11.5,color=INK,loc='left',pad=10)
a2.legend(frameon=False,fontsize=8.5,labelcolor=INK2,loc='upper left')
a2.text(.97,.04,'α = %.3f  →  p = %.2f\nα = 1.886 is %.0fσ away'%(al,pv,(1.886-al)/se),
        transform=a2.transAxes,ha='right',va='bottom',fontsize=9,color=INK)

fig.text(.012,-.015,
 'DivIVA-mCherry, anl/final window (line width 5 px, tip z 0–0.20), 5 IPTG fields pooled (identical exposure). '
 'X is measured on lines ±3 px outside the cell, so it shares no pixels with B or C; it is used only to form the bins.',
 fontsize=8,color=MUTED,ha='left')
fig.tight_layout(rect=[0,.03,1,1])
fig.savefig(OUT,facecolor=SURF,bbox_inches='tight')
print('wrote',OUT)
print('alpha = %.4f +- %.4f  -> p = %.3f   (bins used %d/10, n=%d)'%(al,se,pv,ok.sum(),n))
print('B<=0 cells:',(~pos).sum())
