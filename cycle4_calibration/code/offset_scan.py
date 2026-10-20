# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Why +-3 px?  Scan the offset of the parallel side lines and watch two things:
   (a) does X still track the cell brightness?   (b) does the fitted alpha stay stable?
"""
import csv, numpy as np, tifffile
from scipy.ndimage import map_coordinates

INT=_os.path.join(DATA,'cells_integrated.csv')
FIN=_os.path.join(DATA,'cells_final_w5.csv')
TIF={'1':'1/1aa','2':'2/2aa','3':'3/3aa','4':'4/4a','5':'5/5aa'}
ROOT=_os.path.join(REPO,'data','1010_mc_Div','')
PX=0.1625
def fl(v):
    try: return float(v)
    except: return np.nan

def profile(img,x1,y1,x2,y2,w=1):
    dx,dy=x2-x1,y2-y1; L=np.hypot(dx,dy); n=int(round(L))
    t=np.linspace(0,1,n); px,py=-dy/L,dx/L
    offs=np.arange(w)-(w-1)/2
    return np.mean([map_coordinates(img,[y1+dy*t+py*o, x1+dx*t+px*o],order=1,mode='nearest')
                    for o in offs],axis=0)

rows=list(csv.DictReader(open(INT,encoding='utf-8')))
fin={r['cell_uid']:r for r in csv.DictReader(open(FIN,encoding='utf-8'))}
fcols=[c for c in next(iter(fin.values())) if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in fcols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1)); mM=(zs>=.40)&(zs<=.60)

stacks={k:tifffile.imread(ROOT+v+'.tif') for k,v in TIF.items()}
cells=[]
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    if r['use_shape'].strip().lower() not in ('true','1'): continue
    u=r['cell_uid']
    if u not in fin: continue
    v=np.array([fl(fin[u][c]) for c in fcols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None); C=S[mM].mean(); E=S[mE].mean()
    if C<=0: continue
    cells.append((r,C,E))
print('cells:',len(cells))

def X_at(offset):
    out=[]
    for r,C,E in cells:
        img=stacks[r['rep']][int(fl(r['fl_slice']))-1].astype(float)
        x1,y1,x2,y2=(fl(r[k]) for k in ('x1','y1','x2','y2'))
        L=np.hypot(x2-x1,y2-y1); px,py=-(y2-y1)/L,(x2-x1)/L
        bg=fl(r['bg'])
        vals=[]
        for o in (-offset,offset):
            p=profile(img,x1+o*px,y1+o*py,x2+o*px,y2+o*py,w=1)
            n=len(p); z=(np.arange(n)*L/(n-1)-fl(r['z0_px']))/fl(r['L_px'])
            m=(z>=0.2)&(z<=0.8)
            if m.sum()==0: vals.append(np.nan); continue
            vals.append(p[m].mean())
        out.append(np.mean(vals)-bg)
    return np.array(out)

C=np.array([c[1] for c in cells]); E=np.array([c[2] for c in cells])
rep=np.array([c[0]['rep'] for c in cells])

def alpha(X,b,nq=10):
    ok=np.isfinite(X)
    order=np.argsort(X[ok]); idx=np.where(ok)[0][order]
    lx,ly=[],[]
    for bb in np.array_split(idx,nq):
        if len(bb)==0: continue
        Bm=np.median(E[bb]-b*C[bb]); Cm=np.median(C[bb])
        if Bm<=0 or Cm<=0: continue
        lx.append(np.log(Cm)); ly.append(np.log(Bm))
    if len(lx)<3: return np.nan,len(lx),np.nan
    A=np.c_[lx,np.ones(len(lx))]
    co,*_=np.linalg.lstsq(A,np.array(ly),rcond=None)
    return co[0],len(lx),np.exp(min(ly))

print()
print(' offset   rho(logX,logC)   SNR of X    alpha (b=null)  bins  faintest-bin B')
print(' [px]                      (med/MAD)')
for off in (1,2,3,4,5,6,8):
    X=X_at(off)
    ok=np.isfinite(X)&(X>0)
    rho=np.corrcoef(np.log(X[ok]),np.log(C[ok]))[0,1]
    snr=np.median(X[ok])/(1.4826*np.median(np.abs(X[ok]-np.median(X[ok]))))
    a,nb,bmin=alpha(X,0.5779)
    print('  %+d     %+.3f           %6.2f      %6.3f         %2d    %7.2f'
          % (off,rho,snr,a,nb,bmin))

print()
print(' reference: binning by C itself ->  alpha %.3f' % alpha(C,0.5779)[0])
print(' reference: the cell total      ->  (see X_choice.py)')
print()
print(' note: the main profile band is 3 px wide (offsets -1,0,+1),')
print('       and the cell half-width is R/PX = %.1f px' % (0.40/PX))
