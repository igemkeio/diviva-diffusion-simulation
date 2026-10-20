# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)
# ---------------------------------------------------------------------------
import numpy as np, popz
POST=dict(R=0.40,L=2.7767,p=0.17,rho_n=4.1812)
PRE =dict(R=0.40,L=3.0,p=0.50,rho_n=2.3696)
print(' theta    較正前 左右比 / 足場     較正後 左右比 / 足場')
for th in (0.90,0.95,0.98,0.995):
    a=popz.steady(210,popz.variant(popz_mode='cap',theta=th,k_cut=0.0,**PRE))
    b=popz.steady(236,popz.variant(popz_mode='cap',theta=th,k_cut=0.0,**POST))
    print('  %.3f    %8.2f 倍 / %5.1f       %8.2f 倍 / %5.1f'
          %(th,a['lr_clu'],a['scaffold_R'],b['lr_clu'],b['scaffold_R']))
print()
print('=== PopZ の必要分子数（極キャップ 2piR^2 = %.3f um^2、30 nm^2/分子で単層）==='%(2*np.pi*0.4**2))
cap=2*np.pi*0.4**2; full=cap*1e6/30.0
print('   極キャップを埋め尽くす量 = %.0f 分子。膜タンパクの面密度 5e4/um^2 なら極に %.0f 分子が居る'%(full,cap*5e4))
for lab,ths in (('較正前',[(2,0.350),(3,0.504),(5,0.659),(10,0.818)]),
                ('較正後',[(2,0.517),(3,0.702),(5,0.861)])):
    print('  %s:'%lab, '  '.join('%d倍 theta %.3f -> %5.0f 分子 (極の膜の %.0f%%)'
          %(t,th,th*full,100*th*full/(cap*5e4)) for t,th in ths))
