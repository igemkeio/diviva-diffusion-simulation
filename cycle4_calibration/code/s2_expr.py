# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""S2 (cluster-cleavable): how much extra expression is needed to keep the same
   scaffold as S1 at the same left/right ratio?  k_grow fixed as a molecular constant."""
import numpy as np, popz
from cluster import _kg, derive
CASES=[('較正前',dict(R=0.40,L=3.0,   p=0.50,rho_n=2.3696),210,0.504,0.004297),
       ('較正後',dict(R=0.40,L=2.7767,p=0.17,rho_n=4.1812),236,0.702,0.008688)]
for lab,base,n,th3,kc3 in CASES:
    P0=popz.variant(popz_mode='cap',theta=0.0,k_cut=0.0,**base)
    kg=_kg(P0)                                   # 基準状態で校正した k_grow（分子定数として固定）
    s1=popz.steady(n,popz.variant(popz_mode='cap',theta=th3,k_cut=0.0,**base))
    print('\n== %s =='%lab)
    print('  S1 (theta %.3f, 左右比 3 倍): 良い極の足場 %.1f 分子'%(th3,s1['scaffold_R']))
    lo,hi=1.0,8.0
    for _ in range(18):                          # 二分法で発現量を探す
        m=(lo+hi)/2
        P=popz.variant(popz_mode='cap',theta=0.0,k_cut=kc3,cut_target='MN',
                       **derive(type('X',(),base),N=1690.0*m,k_grow=kg).__dict__ if False else {})
        P=popz.variant(popz_mode='cap',theta=0.0,k_cut=kc3,cut_target='MN',
                       N=1690.0*m,k_grow=kg,**base)
        r=popz.steady(n,P)
        if r['scaffold_R']<s1['scaffold_R']: lo=m
        else: hi=m
    m=(lo+hi)/2
    P=popz.variant(popz_mode='cap',theta=0.0,k_cut=kc3,cut_target='MN',N=1690.0*m,k_grow=kg,**base)
    r=popz.steady(n,P)
    tot=r['mass_frac']*1690.0*m
    print('  S2 で同じ足場にするには 発現量 ×%.2f（%.0f 分子）'%(m,1690*m))
    print('     左右比 %.2f 倍  足場 %.1f 分子  総量 %.0f 分子（%.0f%% = %.0f 分子を常時分解）'
          %(r['lr_clu'],r['scaffold_R'],tot,100*(1-r['mass_frac']),1690*m-tot))
    print('     必要なプロテアーゼ処理量 %.2f 分子/s'%r['cut_flux'])
