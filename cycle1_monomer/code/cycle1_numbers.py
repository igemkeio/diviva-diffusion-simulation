"""DBTL（Cycle 1）に載せる数値だけを再計算する。"""
import numpy as np
from scipy.optimize import brentq
import model as M
from model import Par, variant

LNRHO_BSUB, R_BSUB, L_BSUB = 2.042, 0.425, 2.8


def h(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


h("0. 検証")
ok = M.validate(verbose=True)
print("validate ->", ok)

h("1. 幾何")
g = M.geometry(401)
a = g["A_cap"].sum() / g["A"].sum()
print(f"  A = {g['A'].sum():.4f} µm²,  V = {g['V'].sum():.4f} µm³,  a = {a:.4f}")
print(f"  1/a (天井) = {1/a:.4f}")

h("2. 定常")
r = M.solve(401, T=20.0, ic="uniform")
print(f"  極濃縮      = {r['enrichment']:.4f}")
print(f"  細胞質割合  = {r['f_cyt']:.4f}")
print(f"  細胞質の勾配= {r['cyto_var']:.4f}")
print(f"  極にいる割合= {r['enrichment']*a:.4f}")
r1 = M.solve(101, variant(rho=1.0), T=20.0)
print(f"  rho=1 の勾配= {r1['cyto_var']:.3e}")
print(f"  k_on(側壁)  = {r['k_on'][len(r['k_on'])//2]:.4f} µm/s")
print(f"  ∫k_on dA    = {(r['k_on']*r['g']['A']).sum():.4f} µm³/s")
ana, _ = M.analytic()
print(f"  局所平衡の解析解 = {ana:.4f}")

h("3. 時間経過")
rc = M.solve(201, T=3600.0, ic="cytoplasm", n_out=300)
e, t = rc["enr_t"], rc["t"]
ok2 = np.isfinite(e) & (t > 0)
e, t = e[ok2], t[ok2]
k = np.argmax(np.abs(e - e[-1]) / e[-1] < 0.01)
print(f"  定常の1%以内に入る時刻 = {t[k]:.4f} s  (世代1500s の 1/{1500/t[k]:.0f})")
print(f"  t->0 の極濃縮 = {e[0]:.4f},  定常 = {e[-1]:.4f}")

h("4. エネルギー")
em = M.monomer_curvature_energy()
print(f"  単量体の曲率ミスマッチ = {em:.5f} kT   (たわみ {30**2/(8*Par.R*1000):.3f} nm)")
for target in (2.2,):
    x = brentq(lambda v: M.enrichment(v) - target, 1.01, 60, xtol=1e-4)
    print(f"  大腸菌形状で極濃縮{target}に直接合わせる: rho={x:.3f}, ln rho={np.log(x):.4f} kT"
          f" -> {np.log(x)/em:.0f} 倍")
rho_tr = np.exp(LNRHO_BSUB * R_BSUB / Par.R)
print(f"  B.subtilis から移植: rho={rho_tr:.3f}, ln rho={np.log(rho_tr):.4f} kT"
      f" -> {np.log(rho_tr)/em:.0f} 倍")
print(f"  その rho での大腸菌の極濃縮【予測】= {M.enrichment(rho_tr):.4f}")
print("  lp 感度:")
for lp in (30, 100, 300, 1000):
    e_lp = M.monomer_curvature_energy(lp=lp)
    print(f"    lp={lp:5d} nm -> dG={e_lp:.5f} kT, ギャップ {np.log(rho_tr)/e_lp:.0f} 倍")

h("5. 感度")
base = M.enrichment()
print(f"  baseline(n=101) = {base:.4f}")
for name, vals in [("rho", [Par.rho / 2, Par.rho, Par.rho * 2]),
                   ("D_m", [0.01, 0.1, 1.0]),
                   ("D_c", [0.3, 3.0, 30.0]),
                   ("k_off", [3, 9, 27]),
                   ("f_cyt", [0.157, 0.30, 0.50]),
                   ("N", [169, 1690, 169000])]:
    es = [M.solve(101, variant(**{name: v}), T=20.0)["enrichment"] for v in vals]
    print(f"  {name:6} " + " | ".join(f"{v:g}->{x:.4f}" for v, x in zip(vals, es))
          + f"    幅 {max(es)-min(es):.4f}")

h("6. FRAP と対称性の回復")
rb = M.solve(101, T=20.0, ic="bleach")
tb = M.rise_time(rb, frac=0.5)
print(f"  半回復時間 = {tb*1000:.1f} ms   実測 50.9〜128.6 s")
print(f"  ずれ = {50.9/tb:.0f} 〜 {128.6/tb:.0f} 倍")
t_as, asym = M.pole_asymmetry_trace()
k = np.argmax(np.abs(asym) < 0.01)
print(f"  片極初期条件の左右差が1%を切る時刻 = {t_as[k]:.3f} s")
print(f"  最終の左右差 = {asym[-1]:.2e}")

h("7. 持続摂動（PopZ の空間占有）")
print(f"  {'目標の左右比':>12} {'必要なθ':>10} {'素朴な予想':>11} {'利得':>8}")
for target in (2, 3, 5, 10):
    th = brentq(lambda x: M.asymmetry(x) - target, 0.01, 0.985, xtol=1e-4)
    print(f"  {target:12d} {th:10.3f} {1-1/target:11.3f} {(1-th)*target:8.3f}")

print("\n完了")
