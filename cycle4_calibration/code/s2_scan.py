# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
"""S2: local degradation, before vs after calibration."""
import numpy as np, popz
PRE =dict(R=0.40,L=3.0,   p=0.50,rho_n=2.3696); NPRE=210
POST=dict(R=0.40,L=2.7767,p=0.17,rho_n=4.1812); NPOST=236
def scan(base,n,lab,target):
    P0=popz.variant(popz_mode='cap',theta=0.0,k_cut=0.0,**base)
    b=popz.steady(n,P0)
    print('\n== %s   切る相手 = %s =='%(lab,target))
    print('  基準（PopZ なし）: 片極の足場 %.1f 分子'%b['scaffold_R'])
    print('  目標 左右比   必要 k_cut[1/s]  良い極の足場  基準比   総量/N   切断流量[分子/s]')
    for tgt in (2.0,3.0,5.0):
        P=popz.variant(popz_mode='cap',theta=0.0,cut_target=target,**base)
        kc=popz.kcut_for_ratio(tgt,P,n=n)
        if not np.isfinite(kc):
            print('   %5.0f 倍      届かない'%tgt); continue
        r=popz.steady(n,popz.variant(popz_mode='cap',theta=0.0,k_cut=kc,cut_target=target,**base))
        print('   %5.0f 倍      %9.4g      %7.2f      %.3f   %.4f    %.3f'
              %(tgt,kc,r['scaffold_R'],r['scaffold_R']/b['scaffold_R'],r['mass_frac'],r['cut_flux']))
for tgt in ('M','MN'):
    scan(PRE ,NPRE ,'較正前  L 3.0, p 0.5,  rho_n 2.370',tgt)
    scan(POST,NPOST,'較正後  L 2.78, p 0.17, rho_n 4.181',tgt)
