# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
import sys, csv
import numpy as np, forward as F

FIN = _os.path.join(DATA,'cells_final_w5.csv')
INT = _os.path.join(DATA,'cells_integrated.csv')
ENDS = [(0.00,0.20),(0.80,1.00)]; MID = (0.40,0.60)
R, W = 0.4, 5

def fl(v):
    try: return float(v)
    except: return np.nan

def model_em(f, L, psf, w=W):
    Lc=L-2*R; V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
    zm=np.linspace(0,L,2001); one=np.ones_like(zm)
    pr=F.line_profile(C=(1-f)/V*one, M=f/A*one, z_model=zm, R=R, L=L,
                      psf_fwhm_um=psf, line_w_px=w)
    v,z=pr["val"],pr["z"]; k=max(2,int(round(len(v)*F.TAIL)))
    bg=np.median(np.r_[v[:k],v[-k:]]); S=np.clip(v-bg,0,None)
    e=np.r_[S[(z>=0)&(z<=.20)], S[(z>=.80)&(z<=1.0)]]; m=S[(z>=.40)&(z<=.60)]
    return e.mean()/m.mean()

rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mE=((zs>=0)&(zs<=.20))|((zs>=.80)&(zs<=1.0)); mM=(zs>=.40)&(zs<=.60)
X={r['cell_uid']:fl(r['expr_side_adu']) for r in csv.DictReader(open(INT,encoding='utf-8'))}

def load(cond):
    out=[]
    for r in rows:
        if r['condition']!=cond: continue
        v=np.array([fl(r[c]) for c in cols])
        if not np.isfinite(v).all(): continue
        S=np.clip(v,0,None); C=S[mM].mean(); E=S[mE].mean()
        x=X.get(r['cell_uid'],np.nan)
        if C<=0 or not np.isfinite(x) or x<=0: continue
        out.append((r['rep'],C,E,x,fl(r['cell_len_um'])))
    return out

d = load('DivIVA_mCherry')
print("DivIVA_mCherry  n=%d, fields=%d" % (len(d), len(set(x[0] for x in d))))
print()
print("=== brightness range: within-field vs pooled across the 5 IPTG fields ===")
print("  (exposure 2000 ms / 560 nm / 100%% identical in all 5 -> ADU comparable)")
for rp in sorted(set(x[0] for x in d)):
    xs=np.array([x[3] for x in d if x[0]==rp])
    print("   rep %s (IPTG 10^%s nM)  n=%3d  median X %7.1f  90/10 ratio %.2f"
          % (rp, int(rp)-1, len(xs), np.median(xs),
             np.percentile(xs,90)/np.percentile(xs,10)))
allx=np.array([x[3] for x in d])
print("   ---- pooled           n=%3d  median X %7.1f  90/10 ratio %.2f   (99/1: %.2f, max/min: %.2f)"
      % (len(allx), np.median(allx),
         np.percentile(allx,90)/np.percentile(allx,10),
         np.percentile(allx,99)/np.percentile(allx,1),
         allx.max()/allx.min()))

def alpha(d, b, nq=5, pooled=False):
    if pooled:
        groups=[('all', d)]
    else:
        groups=[(rp,[x for x in d if x[0]==rp]) for rp in sorted(set(x[0] for x in d))]
    lx,ly,gid=[],[],[]
    for gi,(_,sub) in enumerate(groups):
        order=np.argsort([x[3] for x in sub])
        for bb in np.array_split(order, nq if not pooled else 10):
            if len(bb)==0: continue
            Cs=np.array([sub[i][1] for i in bb]); Es=np.array([sub[i][2] for i in bb])
            Bm,Cm=np.median(Es-b*Cs), np.median(Cs)
            if Bm<=0 or Cm<=0: continue
            lx.append(np.log(Cm)); ly.append(np.log(Bm)); gid.append(gi)
    if len(lx)<3: return np.nan,np.nan,len(lx)
    lx=np.array(lx);ly=np.array(ly);gid=np.array(gid)
    ng=len(set(gid.tolist()))
    D=np.zeros((len(lx),1+ng)); D[:,0]=lx
    remap={g:i for i,g in enumerate(sorted(set(gid.tolist())))}
    for i,g in enumerate(gid): D[i,1+remap[g]]=1.0
    coef,_,rank,_=np.linalg.lstsq(D,ly,rcond=None)
    pred=D@coef; dof=max(1,len(lx)-rank)
    cov=((ly-pred)**2).sum()/dof*np.linalg.pinv(D.T@D)
    return coef[0], np.sqrt(cov[0,0]), len(lx)

L=float(np.median([x[4] for x in d]))
nd=load('mCherry_null'); bnull=float(np.median([x[2]/x[1] for x in nd]))
P=np.array([0.00,0.10,0.20,0.35,0.50,0.70]); ALP=np.array([1.000,1.107,1.235,1.489,1.886,3.142])
def p_of(a):
    if not np.isfinite(a): return np.nan
    if a<=ALP[0]: return 0.0
    if a>=ALP[-1]: return np.nan
    return float(np.interp(a,ALP,P))

print()
print("=== alpha: per-field intercepts (old) vs single pooled intercept (now allowed) ===")
for lab,b in [('measured null      ',bnull),
              ('model membrane 0%%  ',model_em(0.0,L,0.870)),
              ('model membrane 70%%ved',model_em(0.7,L,0.870))]:
    a1,s1,n1=alpha(d,b,pooled=False)
    a2,s2,n2=alpha(d,b,pooled=True)
    print("  b=%.4f %s per-field: %.3f +- %.3f (%2d bins) | pooled: %.3f +- %.3f (%2d bins)  -> p %s / %s"
          % (b,lab,a1,s1,n1,a2,s2,n2,
             ('%.2f'%p_of(a1)) if np.isfinite(p_of(a1)) else '-',
             ('%.2f'%p_of(a2)) if np.isfinite(p_of(a2)) else '-'))

print()
print("=== confound check: does cell length track IPTG / brightness? ===")
for rp in sorted(set(x[0] for x in d)):
    sub=[x for x in d if x[0]==rp]
    Ls=np.array([x[4] for x in sub]); Xs=np.array([x[3] for x in sub])
    print("   rep %s  median L %.3f um   median X %6.1f" % (rp, np.median(Ls), np.median(Xs)))
Ls=np.array([x[4] for x in d]); Xs=np.array([x[3] for x in d])
ok=np.isfinite(Ls)&np.isfinite(Xs)
print("   pooled Spearman rho(log X, L) = %.3f"
      % np.corrcoef(np.argsort(np.argsort(np.log(Xs[ok]))),
                    np.argsort(np.argsort(Ls[ok])))[0,1])

print()
print("=== pooled fit restricted to a narrow cell-length band (removes the L confound) ===")
lo,hi=np.percentile(Ls[ok],[30,70])
dn=[x for x in d if np.isfinite(x[4]) and lo<=x[4]<=hi]
print("   band %.2f-%.2f um, n=%d" % (lo,hi,len(dn)))
for lab,b in [('measured null   ',bnull),
              ('model membrane 0%',model_em(0.0,L,0.870)),
              ('model memb. 70%  ',model_em(0.7,L,0.870))]:
    a,s,n=alpha(dn,b,pooled=True)
    print("     b=%.4f %s alpha=%.3f +- %.3f (%d bins) -> p %s"
          % (b,lab,a,s,n,('%.2f'%p_of(a)) if np.isfinite(p_of(a)) else '-'))

print()
print("=== how strongly is p=0.5 excluded by the pooled fit? ===")
a,s,_=alpha(d,bnull,pooled=True)
print("   alpha=%.3f +- %.3f :  p=0.5 (alpha 1.886) is %.1f sigma away;  p>1 (alpha 3.1) is %.1f sigma"
      % (a,s,(1.886-a)/s,(3.1-a)/s))
