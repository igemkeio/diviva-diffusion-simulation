"""p を変えるときは rho_n の再校正が要る（results5_out.txt の表）。
   前回の走査は rho_n を 4.1812 に固定していたので無効。やり直す。"""
import os as _os, sys as _sys
_HERE=_os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0,_HERE)
import numpy as np, forward as F, cluster as CL
R,L=0.40,2.7767; CAP=R/L; Lc=L-2*R
V=np.pi*R*R*Lc+4/3*np.pi*R**3; A=2*np.pi*R*Lc+4*np.pi*R*R
zm=np.linspace(0,L,2001); one=np.ones_like(zm)
def em(kw,e=CAP,Lx=L,zx=None):
    pr=F.line_profile(z_model=(zm if zx is None else zx),R=R,L=Lx,psf_fwhm_um=0.870,line_w_px=5,**kw)
    v,z=pr['val'],pr['z']; k=max(2,int(round(len(v)*F.TAIL)))
    S=np.clip(v-np.median(np.r_[v[:k],v[-k:]]),0,None)
    return np.r_[S[(z>=0)&(z<=e)],S[(z>=1-e)&(z<=1)]].mean()/S[(z>=.40)&(z<=.60)].mean()
b70=em(dict(C=0.2957/V*one,M=0.7043/A*one)); MEAS=0.7539
print('下駄 %.4f   実測 %.4f（超過 %+.4f）\n'%(b70,MEAS,MEAS-b70))
# results5_out.txt §5: p ごとに 66% 目標へ再校正した rho_n（B.sub→E.coli 移植後）
PAIRS=[(0.05,5.1441,175),(0.17,4.1812,153),(0.35,3.0662,120),(0.50,2.3696,92)]
print('★ 正しい走査: p と rho_n をセットで動かす')
print('    p    rho_n    Q 分子   モデル end_mid   超過      一致度')
for p,rn,Q in PAIRS:
    par=CL.variant(p=p,rho_n=rn,L=L,R=R); st=CL.steady(n=201,par=par)
    mo=em(dict(C=st['C'],M=st['M'],N=st['Nf']),Lx=par.L,zx=np.linspace(0,par.L,201))
    print('  %.2f   %.4f    %3d      %.4f        %+.4f    %.2f 倍%s'
          %(p,rn,Q,mo,mo-b70,(MEAS-b70)/(mo-b70),'   <- 5 節で使用' if p==0.17 else ''))
print()
print('⚠️ 参考（誤り）: rho_n を 4.1812 に固定したまま p だけ動かした場合')
print('    p    モデル end_mid   一致度')
for p in (0.05,0.17,0.35,0.50):
    par=CL.variant(p=p,rho_n=4.1812,L=L,R=R); st=CL.steady(n=201,par=par)
    mo=em(dict(C=st['C'],M=st['M'],N=st['Nf']),Lx=par.L,zx=np.linspace(0,par.L,201))
    print('  %.2f      %.4f        %.2f 倍'%(p,mo,(MEAS-b70)/(mo-b70)))
