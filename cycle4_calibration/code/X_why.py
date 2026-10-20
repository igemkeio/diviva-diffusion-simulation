# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
ROOT  = _os.path.dirname(_HERE)
DATA  = _os.path.join(ROOT, 'data')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""Why sort the cells by the parallel lines and not by the cell's own fluorescence?

Test 1 (real data, two poles): the two poles of one cell are the same cell but
  different pixels. Sorting by pole A and measuring the excess at pole B shares
  no pole pixels; sorting by pole B and measuring at pole B shares all of them.
Test 2 (synthetic, known answer alpha = 1): which sorting variable recovers it?
"""
import csv
import numpy as np
rng=np.random.default_rng(0)
FIN=_os.path.join(DATA,'cells_final_w5.csv')
INT=_os.path.join(DATA,'cells_integrated.csv')
rows=list(csv.DictReader(open(FIN,encoding='utf-8')))
cols=[c for c in rows[0] if c.startswith('div_z')]
zs=np.array([float(c[5:]) for c in cols])
mA=(zs>=0)&(zs<=.20); mB=(zs>=.80)&(zs<=1.0); mM=(zs>=.40)&(zs<=.60)
def fl(v):
    try: return float(v)
    except: return np.nan
SIDE={r['cell_uid']:fl(r['expr_side_adu']) for r in csv.DictReader(open(INT,encoding='utf-8'))}
C,SA,SB,X=[],[],[],[]
for r in rows:
    if r['condition']!='DivIVA_mCherry': continue
    v=np.array([fl(r[c]) for c in cols])
    if not np.isfinite(v).all(): continue
    S=np.clip(v,0,None); c=S[mM].mean(); x=SIDE.get(r['cell_uid'],np.nan)
    if c<=0 or not np.isfinite(x) or x<=0: continue
    C.append(c); SA.append(S[mA].mean()); SB.append(S[mB].mean()); X.append(x)
C=np.array(C); SA=np.array(SA); SB=np.array(SB); X=np.array(X); b=0.5779
def alpha(sortv,Cv,Ev,bb=None,nq=10):
    bb=b if bb is None else bb
    o=np.argsort(sortv); lx,ly=[],[]
    for g in np.array_split(o,nq):
        Bm=np.median(Ev[g]-bb*Cv[g]); Cm=np.median(Cv[g])
        if Bm<=0 or Cm<=0: continue
        lx.append(np.log(Cm)); ly.append(np.log(Bm))
    if len(lx)<3: return np.nan,len(lx)
    lx=np.array(lx); ly=np.array(ly); A=np.c_[lx,np.ones(len(lx))]
    return np.linalg.lstsq(A,ly,rcond=None)[0][0],len(lx)

print('=== テスト 1: 実データ。同じ細胞の 2 つの極で切り分ける（n = %d）==='%len(C))
print('  並べ替えに使う極   超過を測る極   極の画素を共有   alpha')
for sv,sl,ev,el in ((SA,'極 A',SB,'極 B'),(SB,'極 B',SA,'極 A'),
                    (SA,'極 A',SA,'極 A'),(SB,'極 B',SB,'極 B')):
    a,n=alpha(sv,C,ev)
    print('      %s            %s         %s        %.3f  (%d 区画)'
          %(sl,el,'しない' if sl!=el else '**する**',a,n))
a,_=alpha(X,C,(SA+SB)/2); print('   ±3 px 平行線      両極の平均     しない        %.3f'%a)
a,_=alpha((SA+SB)/2,C,(SA+SB)/2); print('   両極の平均        両極の平均     **する**      %.3f'%a)

print()
print('=== テスト 2: 人工データ。真の答えは alpha = 1.000 ===')
print('  実測の C を「真の明るさ」とし、E_true = (k + b)*c としてから')
print('  E と C に独立なノイズを載せる。並べ替えの量を 3 通り試す。')
k=np.median(((SA+SB)/2-b*C)/C)
print('  k = %.4f'%k)
print()
print('  E・C のノイズ   平行線（独立、2 倍ノイズが多い）   細胞の総和（E と C を含む）   C そのもの')
for frac in (0.025,0.05,0.10,0.20):
    s=frac*np.median(C); A_s,A_t,A_c=[],[],[]
    for _ in range(300):
        Co=np.clip(C+rng.normal(0,s,len(C)),1e-9,None)
        Eo=np.clip((k+b)*C+rng.normal(0,s,len(C)),1e-9,None)
        tot=2*Eo+3*Co
        side=np.clip(C+rng.normal(0,2*s,len(C)),1e-9,None)
        A_s.append(alpha(side,Co,Eo)[0]); A_t.append(alpha(tot,Co,Eo)[0]); A_c.append(alpha(Co,Co,Eo)[0])
    print('      %4.1f%%            %.3f                      %.3f               %.3f'
          %(100*frac,np.nanmean(A_s),np.nanmean(A_t),np.nanmean(A_c)))
print()
print('  参考: 光子ノイズだけなら C の 2.4%（画素ノイズ 14.6 ADU, C 中央値 145 ADU）。')
print('        両極の食い違いから見積もる実効的なばらつきは 10-20% で、こちらが支配的。')
