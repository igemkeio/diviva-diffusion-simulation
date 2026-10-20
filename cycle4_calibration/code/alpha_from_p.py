# --- paths (resolved relative to this file; edit nothing) -------------------
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))        # calibration/code
ROOT  = _os.path.dirname(_HERE)                              # calibration
REPO  = _os.path.dirname(ROOT)                               # Igem数値解析
DATA  = _os.path.join(ROOT, 'data')
FIGS  = _os.path.join(ROOT, 'figures')
if _HERE not in _sys.path: _sys.path.insert(0, _HERE)        # forward.py, cluster.py
# ---------------------------------------------------------------------------
"""Where does alpha = 1/(1-p) come from, and does the full model obey it?"""
import numpy as np, cluster as CL, forward as F

print("=== 解析的な導出 ===")
print("  クラスターの成長:  G = (k_nuc + k_grow * N^p) * M      <- cluster.py の実装")
print("  クラスターの崩壊:  k_off_n(z) * N")
print("  単量体の出入り  :  k_on * C - k_off * M     ->  M ∝ C")
print()
print("  定常 (k_nuc を無視):  k_grow * N^p * M = k_off_n * N")
print("                     ->  N^(1-p) = k_grow * M / k_off_n")
print("                     ->  N = (k_grow * M / k_off_n)^(1/(1-p))  ∝  C^(1/(1-p))")
print()
print("  極の超過 B ∝ N,  細胞質 C ∝ C  なので   B ∝ C^alpha,  alpha = 1/(1-p)")
print("                                          逆に           p = 1 - 1/alpha")
print()
print("=== 公表されている表と、解析式 1/(1-p) の比較 ===")
tab = {0.00:1.000, 0.10:1.107, 0.20:1.235, 0.35:1.489, 0.50:1.886, 0.70:3.142}
print("   p      表の alpha   1/(1-p)    ずれ      表から戻した p = 1-1/alpha")
for p,a in tab.items():
    print("  %.2f      %.3f      %.3f     %+.1f%%        %.3f" % (p,a,1/(1-p),100*(a/(1/(1-p))-1),1-1/a))

print()
print("=== 実測の alpha を解析式で p に戻す ===")
for lab,a,se in (('下駄 0.5211 (実測 null)',1.055,0.089),
                 ('下駄 0.6432 (膜 70%)  ',1.196,0.272),
                 ('5 濃度プール・全細胞   ',1.076,0.061)):
    p=1-1/a
    print("   %s  alpha %.3f -> p = 1 - 1/alpha = %.3f" % (lab,a,p))
print()
print("   p = 0.5 には alpha = 1/(1-0.5) = 2.000 が要る")
print("   p = 0.7 には alpha = 3.333")
print("   p -> 1  には alpha -> 無限大   <- どんな有限の alpha も p < 1 を意味する")
print("   -> プールの alpha 1.076 +- 0.061 に対し、p=0.5 の 2.000 は %.1f sigma" % ((2.0-1.076)/0.061))

print()
print("=== 完全なモデルでも成り立つか（発現量 N を振って alpha を測る） ===")
for p in (0.20,0.50):
    base=CL.variant(p=p, rho_n=4.1812, L=2.7767, R=0.40)
    kg=CL._kg(base)                      # k_grow は分子定数として固定（設計判断 [D4]）
    zs=None; Cs=[]; Bs=[]
    for mult in (0.25,0.5,1.0,2.0,4.0,8.0):
        par=CL.derive(base, N=1690.0*mult, k_grow=kg)
        st=CL.steady(n=201,par=par)
        zm=np.linspace(0,par.L,201)
        pr=F.line_profile(C=st['C'],M=st['M'],N=st['Nf'],z_model=zm,
                          R=0.40,L=par.L,psf_fwhm_um=0.870,line_w_px=3)
        v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
        S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
        E=np.r_[S[(z>=0)&(z<=.15)],S[(z>=.85)&(z<=1)]].mean()
        Cm=S[(z>=.40)&(z<=.60)].mean()
        Cs.append(Cm); Bs.append(E-0.6432*Cm)
    Cs=np.array(Cs); Bs=np.array(Bs); ok=Bs>0
    A=np.c_[np.log(Cs[ok]),np.ones(ok.sum())]
    co,*_=np.linalg.lstsq(A,np.log(Bs[ok]),rcond=None)
    print("   p=%.2f : 数値で測った alpha = %.3f   （解析式 %.3f、公表表 %.3f）"
          % (p,co[0],1/(1-p),tab[p]))
