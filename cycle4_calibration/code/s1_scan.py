# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""S1: required coverage theta, before vs after calibration."""
import numpy as np, popz
PRE  = dict(R=0.40,L=3.0,   p=0.50,rho_n=2.3696)   # 較正前（Cycle 2/4 既定）
POST = dict(R=0.40,L=2.7767,p=0.17,rho_n=4.1812)   # 較正後
NPRE,NPOST=210,236
def run(base,n,lab,**kw):
    P=popz.variant(popz_mode='cap',theta=0.0,k_cut=0.0,**base,**kw)
    b=popz.steady(n,P)
    print('\n== %s =='%lab)
    print('  基準（PopZ なし）: クラスター極濃縮 %.3f  極割合 %.3f  f_clu %.3f  片極の足場 %.1f 分子'
          %(b['enr_clu'],b['clu_pole_frac'],b['f_clu'],b['scaffold_R']))
    print('  目標 左右比   必要 theta   良い極の足場[分子]  基準比   総量/N')
    out={}
    for tgt in (2.0,3.0,5.0,10.0):
        th=popz.theta_for_ratio(tgt,P,n=n)
        if not np.isfinite(th):
            print('   %5.0f 倍      届かない'%tgt); out[tgt]=np.nan; continue
        r=popz.steady(n,popz.variant(popz_mode='cap',theta=th,k_cut=0.0,**base,**kw))
        print('   %5.0f 倍      %.3f        %7.1f            %.2f    %.4f'
              %(tgt,th,r['scaffold_R'],r['scaffold_R']/b['scaffold_R'],r['mass_frac']))
        out[tgt]=th
    return b,out
run(PRE ,NPRE ,'較正前  L 3.0 µm, p 0.5,  rho_n 2.370  (n=%d)'%NPRE)
run(POST,NPOST,'較正後  L 2.78 µm, p 0.17, rho_n 4.181  (n=%d)'%NPOST)
