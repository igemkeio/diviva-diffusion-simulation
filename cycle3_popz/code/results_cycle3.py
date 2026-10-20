"""
Cycle 3 の数値 — S1 空間占有 と S2 局所分解の比較

    python results_cycle3.py            # 全部
    python results_cycle3.py 3 6        # 節を指定

⚠️ この出力が唯一の正である。報告書と食い違ったら報告書のほうが古い。
⚠️ 走らせる前に `python popz.py`（検証）を通すこと。
"""
import sys
import numpy as np
import popz
import cluster
from cluster import derive

P = popz.Par
NG = 210          # 極境界がセル境界に乗る格子（設計判断 [P12]）
_B = {}


def base(par=P):
    """基準状態（PopZ なし）をキャッシュして返す。"""
    k = (par.p, par.mu, par.N, par.block_mode)
    if k not in _B:
        r = popz.steady(NG, derive(par, theta=0.0, k_cut=0.0))
        _B[k] = (r, np.concatenate([r["C"], r["M"], r["Nf"]]))
    return _B[k]


def run(par, y0=None):
    r0, w = base(par)
    return popz.steady(NG, par, y0=y0 if y0 is not None else w)


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# -----------------------------------------------------------------------------
def s1_geometry():
    hdr("1. 幾何と PopZ の占有プロファイル")
    g = cluster.geometry2(NG, P)
    print(f"  R={P.R} µm, L={P.L} µm, 膜面積 A={g['A'].sum():.4f} µm², "
          f"体積 V={g['V'].sum():.4f} µm³")
    print(f"  極キャップ面積比 a = {g['A_cap'].sum()/g['A'].sum():.4f} "
          f"(解析解 2R/L = {2*P.R/P.L:.4f})")
    print(f"  片極キャップ面積 = {g['A_capL'].sum():.4f} µm²")
    print(f"\n  PopZ 凝集体の半径 r_popz → 極キャップの被覆率（設計判断 [P5]）")
    print(f"  {'r_popz [nm]':>12} {'被覆率(数値)':>14} {'解析解':>10} {'実効 theta/theta':>18}")
    for rp in (0.15, 0.20, 0.25, 0.30, 0.35, P.R*np.pi/2):
        pz = popz.popz_profile(NG, derive(P, popz_mode="arc", r_popz=rp, theta=1.0))
        ana = popz.popz_cap_cover_analytic(derive(P, r_popz=rp))
        lab = f"{rp*1000:.0f}" + ("(=Rπ/2)" if abs(rp-P.R*np.pi/2) < 1e-9 else "")
        print(f"  {lab:>12} {pz['cap_cover']:>14.4f} "
              f"{ana if np.isfinite(ana) else float('nan'):>10.4f} {pz['theta_eff']:>18.4f}")
    print("\n  ⚠️ theta と r_popz は独立でない。両方を上げても実効被覆率は")
    print("     キャップ被覆率 × theta を超えない。以降の theta は")
    print("     popz_mode='cap'（キャップ全面を覆う）での値である。")


def s2_baseline():
    hdr("2. 基準状態（PopZ なし）と Cycle 2 からの補正")
    print("  --- 格子を揃える効果（設計判断 [P12]、mu=0 で Cycle 2 と同条件）---")
    print(f"  {'n':>5} {'境界整合':>10} {'enr_clu':>10} {'極割合':>9} {'膜全体':>9}")
    for n in (101, 201, 401):
        r = popz.steady(n, derive(P, mu=0.0))
        print(f"  {n:>5} {'×':>10} {r['enr_clu']:>10.4f} {r['clu_pole_frac']:>9.4f} "
              f"{r['enrichment']:>9.4f}")
    for n in (105, 210, 420):
        r = popz.steady(n, derive(P, mu=0.0))
        print(f"  {n:>5} {'○':>10} {r['enr_clu']:>10.4f} {r['clu_pole_frac']:>9.4f} "
              f"{r['enrichment']:>9.4f}")
    print("  → 揃えると n に依らず一定。Cycle 2 の『幅 0.88%』は按分の揺れだった。")

    print("\n  --- 希釈 mu の効果（n=210、設計判断 [P2]）---")
    a = popz.steady(NG, derive(P, mu=0.0))
    b = popz.steady(NG, derive(P))
    print(f"  {'':>22} {'mu=0':>10} {'mu=ln2/1500':>13} {'差':>8}")
    for k, lab in (("enr_clu", "クラスター極濃縮"), ("clu_pole_frac", "極割合"),
                   ("enrichment", "膜全体の極濃縮"), ("f_clu", "f_clu")):
        print(f"  {lab:>22} {a[k]:>10.4f} {b[k]:>13.4f} "
              f"{abs(b[k]-a[k])/a[k]:>7.2%}")
    print("  ⚠️ popz.py の当初の記述『1% 未満』は誤り。実測 3.11%。")

    print(f"\n  --- Cycle 3 の既定での基準状態（mu>0, n=210）---")
    print(f"  クラスター極濃縮 {b['enr_clu']:.4f} / 膜全体 {b['enrichment']:.4f} "
          f"/ 極割合 {b['clu_pole_frac']:.4f}")
    print(f"  3 相分配 cyt/mono/clu = {b['f_cyt']:.4f}/{b['f_mono']:.4f}/{b['f_clu']:.4f}")
    print(f"  ★ 片極のクラスター分子数（足場の基準値）= {b['scaffold_R']:.2f} 分子")
    print(f"  総量/N = {b['mass_frac']:.8f}（分解なしなら厳密に 1。設計判断 [P2]）")
    print(f"  合成速度 mu*N = {P.mu*P.N:.4f} 分子/s  ★S2 の構造的な上限")


def s3_s1():
    hdr("3. S1 空間占有 — theta スキャンと規約の影響")
    r0, _ = base()
    convs = (("redistribute", False, "再配分・成長阻害なし（Cycle 2 の規約）"),
             ("lose", False, "容量喪失・成長阻害なし"),
             ("redistribute", True, "再配分・成長も阻害"),
             ("lose", True, "容量喪失・成長も阻害"))
    ths = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98)
    for bm, bg, lab in convs:
        print(f"\n  --- {lab} ---")
        print(f"  {'theta':>7} {'左右比':>9} {'局所平衡':>9} {'利得':>7} "
              f"{'足場 n_R':>10} {'基準比':>8} {'総量/N':>8}")
        for th in ths:
            r = run(derive(P, theta=th, block_mode=bm, block_grow=bg))
            loc = 1.0/(1.0-th)
            print(f"  {th:>7} {r['lr_clu']:>9.3f} {loc:>9.3f} "
                  f"{r['lr_clu']/loc:>7.3f} {r['scaffold_R']:>10.2f} "
                  f"{r['scaffold_R']/r0['scaffold_R']:>8.3f} {r['mass_frac']:>8.4f}")
    print("\n  ★ S1 は総量を一切削らない（総量/N = 1）。さらに塞いだ極の分子が")
    print("    良い極へ移るので、足場は基準より**増える**。")

    print("\n  --- 目標の左右比に必要な被覆率 theta（判定基準 1）---")
    print(f"  {'目標':>6} " + " ".join(f"{b+('+g' if g else ''):>16}" for b, g, _ in convs))
    for tg in (2, 3, 5, 10):
        row = []
        for bm, bg, _ in convs:
            th = popz.theta_for_ratio(tg, derive(P, block_mode=bm, block_grow=bg), n=NG)
            row.append(f"{th:>16.3f}" if np.isfinite(th) else f"{'届かない':>16}")
        print(f"  {tg:>5}倍 " + " ".join(row))
    print("  ⚠️ Cycle 2 の公表値（35% / 50% / 65% / 81%）は mu=0・n=201 のもの。")


def s4_s2():
    hdr("4. S2 局所分解 — k_cut スキャンと構造的な上限")
    r0, _ = base()
    muN = P.mu*P.N
    print(f"  ★ 定常では ∫r_deg dA ≤ mu*N = {muN:.4f} 分子/s（合成より速くは切れない）")
    print(f"    したがって 総量/N = 1 - ∫r_deg dA/(mu*N) が厳密に成り立つ。")
    for tgt, lab in (("M", "単量体のみを切る（既定、設計判断 [P9]）"),
                     ("MN", "クラスターも切る")):
        print(f"\n  --- {lab} ---")
        print(f"  {'k_cut':>8} {'左右比':>10} {'総量/N':>8} {'1-D/muN':>9} "
              f"{'足場 n_R':>10} {'基準比':>8} {'処理量 D':>9}")
        for kc in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            r = run(derive(P, theta=0.0, k_cut=kc, cut_target=tgt))
            print(f"  {kc:>8} {r['lr_clu']:>10.3f} {r['mass_frac']:>8.4f} "
                  f"{1-r['cut_flux']/muN:>9.4f} {r['scaffold_R']:>10.3f} "
                  f"{r['scaffold_R']/r0['scaffold_R']:>8.4f} {r['cut_flux']:>9.4f}")
    print("\n  --- なぜ単量体だけを切ると左右比が立たないか（時間尺度）---")
    print(f"  細胞質の均一化   L²/D_c = {P.L**2/P.D_c:>7.2f} s")
    print(f"  膜単量体の均一化 L²/D_m = {P.L**2/P.D_m:>7.2f} s")
    print(f"  クラスターの拡散        D_n = 0 → **動かない**")
    print(f"  切断 k_cut=0.01 の時定数 = {1/0.01:>7.0f} s（拡散より遅い→大域的に枯れる）")
    print("  ★ 単量体は切る前に細胞全体へ均される。クラスターは動かないので")
    print("    切断が局所に効く。S2 が機能するかは**クラスターを切れるか**で決まる。")

    print("\n  --- 目標の左右比に必要な k_cut（判定基準 1）---")
    print(f"  {'目標':>6} {'M のみ':>12} {'そのとき総量':>13} "
          f"{'M+N':>12} {'そのとき総量':>13}")
    for tg in (2, 3, 5, 10):
        row = []
        for tgt in ("M", "MN"):
            kc = popz.kcut_for_ratio(tg, derive(P, theta=0.0, cut_target=tgt), n=NG)
            if np.isfinite(kc):
                r = run(derive(P, theta=0.0, k_cut=kc, cut_target=tgt))
                row += [f"{kc:>12.4g}", f"{r['mass_frac']:>13.4f}"]
            else:
                row += [f"{'届かない':>12}", f"{'-':>13}"]
        print(f"  {tg:>5}倍 " + " ".join(row))


def s5_match():
    hdr("5. ★ 同じ左右比で比べる（判定基準 1 と 2 を同時に見る）")
    r0, _ = base()
    print(f"  基準（PopZ なし）の片極クラスター分子数 = {r0['scaffold_R']:.2f} 分子\n")
    print(f"  {'目標':>5} {'戦略':>14} {'必要な入力':>16} {'左右比':>8} "
          f"{'足場 n_R':>10} {'基準比':>8} {'総量/N':>8} {'処理量':>8}")
    for tg in (2, 3, 5, 10):
        th = popz.theta_for_ratio(tg, P, n=NG)
        if np.isfinite(th):
            r = run(derive(P, theta=th))
            print(f"  {tg:>4}倍 {'S1 占有':>14} {'theta = '+f'{th:.3f}':>16} "
                  f"{r['lr_clu']:>8.3f} {r['scaffold_R']:>10.2f} "
                  f"{r['scaffold_R']/r0['scaffold_R']:>8.3f} {r['mass_frac']:>8.4f} "
                  f"{r['cut_flux']:>8.3f}")
        for tgt, lab in (("M", "S2 分解(M)"), ("MN", "S2 分解(M+N)")):
            kc = popz.kcut_for_ratio(tg, derive(P, theta=0.0, cut_target=tgt), n=NG)
            if np.isfinite(kc):
                r = run(derive(P, theta=0.0, k_cut=kc, cut_target=tgt))
                print(f"  {'':>5} {lab:>14} {'k_cut = '+f'{kc:.4g}':>16} "
                      f"{r['lr_clu']:>8.3f} {r['scaffold_R']:>10.3f} "
                      f"{r['scaffold_R']/r0['scaffold_R']:>8.4f} {r['mass_frac']:>8.4f} "
                      f"{r['cut_flux']:>8.3f}")
            else:
                print(f"  {'':>5} {lab:>14} {'届かない':>16}")
    print("\n  ★ 判定基準 2（足場として使える量が残るか）で勝敗が決まる。")
    print("    S1 は足場が増え、S2 は削って買う。左右比だけで比べないこと。")


def s6_p():
    hdr("6. 成長の次数 p への依存（結論が反転しうる唯一の量）")
    for p in (0.5, 1.2):
        pp = derive(P, p=p)
        r0, _ = base(pp)
        print(f"\n  --- p = {p} （基準の足場 {r0['scaffold_R']:.2f} 分子、"
              f"極濃縮 {r0['enr_clu']:.3f}）---")
        print(f"  {'入力':>18} {'左右比':>11} {'足場 n_R':>10} {'基準比':>8} {'総量/N':>8}")
        for th in (0.5, 0.8, 0.9):
            r = run(derive(pp, theta=th))
            print(f"  {'S1 theta='+f'{th}':>18} {r['lr_clu']:>11.3f} "
                  f"{r['scaffold_R']:>10.2f} {r['scaffold_R']/r0['scaffold_R']:>8.3f} "
                  f"{r['mass_frac']:>8.4f}")
        for kc in (0.003, 0.01, 0.05):
            r = run(derive(pp, theta=0.0, k_cut=kc, cut_target="MN"))
            print(f"  {'S2(M+N) k='+f'{kc}':>18} {r['lr_clu']:>11.3f} "
                  f"{r['scaffold_R']:>10.3f} {r['scaffold_R']/r0['scaffold_R']:>8.4f} "
                  f"{r['mass_frac']:>8.4f}")


def s7_latch():
    hdr("7. ★ 誘導の順序とラッチ（一過的な誘導で足りるか）")
    print("  ★ 初期条件が本質的である。2 つの順序を比べる:")
    print("     ic='cytoplasm'   = PopZ が先、DivIVA を後から発現（本プロジェクトの基本戦略）")
    print("     ic='steady_free' = DivIVA を先に局在させてから PopZ をかける（逆順）")
    print("  PopZ を t_on = 600 s かけてから外し、十分待った後を見る。")
    for ic, lab in (("cytoplasm", "PopZ が先（DivIVA は素から）"),
                    ("steady_free", "DivIVA が先（既に両極局在）")):
        print(f"\n  ===== {lab} =====")
        for p in (0.5, 1.2):
            print(f"\n  --- p = {p} ---")
            print(f"  {'入力':>20} {'外す直前':>11} {'最終':>12} {'最終の足場':>11} "
                  f"{'f_clu':>8} {'判定':>8}")
            for l2, kw in (("S1 theta=0.5", dict(theta=0.5)),
                           ("S1 theta=0.9", dict(theta=0.9)),
                           ("S2(M+N) k=0.01", dict(theta=0.0, k_cut=0.01,
                                                   cut_target="MN")),
                           ("S2(M+N) k=0.05", dict(theta=0.0, k_cut=0.05,
                                                   cut_target="MN"))):
                pp = derive(P, p=p, **kw)
                a, b, r2 = popz.latch(pp, n=NG, t_on=600.0, T_after=1e5, ic=ic)
                print(f"  {l2:>20} {a:>11.4g} {b:>12.4g} {r2['scaffold_R']:>11.2f} "
                      f"{r2['f_clu']:>8.4f} {'残る' if b > 1.5 else '戻る':>8}")
    print("\n  ★ p=1.2 で PopZ が先だと DivIVA はクラスターを作れない（核形成の壁）。")
    print("    ラッチが効くのは DivIVA を先に局在させた場合だけである。")
    print("    Cycle 2 の『600 s でラッチ 2.4e4』は後者の条件の結果だった。")


def s7b_nucleation():
    hdr("7b. ★ 核形成の壁 — p>1 では素から発現するとクラスター相に入れない")
    print("  PopZ を一切かけず、初期条件だけを変えて定常を探す。")
    print(f"  {'p':>5} {'素から f_clu':>14} {'素から足場':>11} "
          f"{'uniform f_clu':>15} {'uniform 足場':>13} {'判定':>12}")
    for p in (0.5, 0.8, 1.0, 1.1, 1.2, 1.5):
        q = derive(P, p=p)
        a = popz.steady(NG, q, ic="cytoplasm")
        b = popz.steady(NG, q, ic="uniform")
        same = abs(a["f_clu"] - b["f_clu"]) / max(b["f_clu"], 1e-30) < 1e-3
        print(f"  {p:>5} {a['f_clu']:>14.4f} {a['scaffold_R']:>11.2f} "
              f"{b['f_clu']:>15.4f} {b['scaffold_R']:>13.2f} "
              f"{('一意' if same else '★二つの相'):>12}")
    print("\n  ★ p<=1 では経路に依らない（定常が一意）。p>1 では素からだと")
    print("    クラスター相に入れず、足場がどちらの極にもできない。")
    print("    これは PopZ の有無とは無関係な、DivIVA 側だけの性質である。")


def s8_lead():
    hdr("8. 誘導の先行時間（Cycle 2 の『局在に一世代』がここに効く）")
    print("  t_pre = PopZ をかける前に DivIVA を走らせる時間（素から出発）。")
    print("  t_pre = 0 が『PopZ が先』、t_pre が大きいほど『DivIVA が先』に近づく。")
    gen = np.log(2)/P.mu
    for p in (0.5, 1.2):
        print(f"\n  --- p = {p}（世代時間 {gen:.0f} s）---")
        print(f"  {'t_pre [s]':>10} {'世代比':>8} {'S1 th=0.9 最終':>15} {'足場':>9} "
              f"{'S2(M+N) k=0.01 最終':>21} {'足場':>9}")
        for tp in (0.0, 750.0, 1500.0, 3000.0, 6000.0, 15000.0):
            o = []
            for kw in (dict(theta=0.9),
                       dict(theta=0.0, k_cut=0.01, cut_target="MN")):
                pp = derive(P, p=p, **kw)
                _, b, r2 = popz.latch(pp, n=NG, t_pre=tp, t_on=600.0,
                                      T_after=1e5, ic="cytoplasm")
                o += [b, r2["scaffold_R"]]
            print(f"  {tp:>10.0f} {tp/gen:>8.2f} {o[0]:>15.4g} {o[1]:>9.2f} "
                  f"{o[2]:>21.4g} {o[3]:>9.2f}")
    print("\n  ⚠️ p=1.2 では t_pre をいくら伸ばしても素からはクラスターが立たないので")
    print("     ラッチに入れない（7b の核形成の壁）。")


def s9_expr():
    hdr("9. DivIVA 発現量 N（大腸菌の異種発現で 10〜100 倍になりうる）")
    print("  ⚠️ k_grow をクラスター割合に校正し続ける流儀なので、")
    print("     N はスケーリング対称性で**厳密に効かない**（Cycle 2 V11）。")
    print("     この表はそれを確認するためのもので、N の効果を示すものではない。")
    print(f"\n  {'N':>9} {'左右比(S1 th=0.8)':>19} {'足場 n_R':>10} "
          f"{'足場/N':>9} {'処理量上限 mu*N':>16}")
    for Nv in (169.0, 1690.0, 16900.0, 169000.0):
        pp = derive(P, N=Nv)
        r = popz.steady(NG, derive(pp, theta=0.8))
        print(f"  {Nv:>9.0f} {r['lr_clu']:>19.4f} {r['scaffold_R']:>10.2f} "
              f"{r['scaffold_R']/Nv:>9.5f} {pp.mu*Nv:>16.4f}")
    print("\n  ★ 左右比と足場/N は N に依らない。一方で S2 の処理量上限 mu*N は")
    print("    N に比例するので、**発現量を上げると必要なプロテアーゼ量も比例して増える**。")



def s10_nuc_constraint():
    hdr("10. ★ p > 1 は k_nuc と一緒にしか決まらない")
    print("  実際の細胞では DivIVA は発現後に de novo で両極へ局在する。")
    print("  モデルがそれを再現できない (p, k_nuc) の組は棄却できる。")

    def ok(par, ic="uniform"):
        try:
            return popz.steady(NG, par, ic=ic)
        except Exception:
            return None

    lo, hi = 1e-5, 1.0
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if ok(derive(P, k_nuc=mid)) is not None:
            lo = mid
        else:
            hi = mid
    print(f"\n  --- 上限: 校正できる k_nuc ≈ {lo:.3e} "
          f"（既定 {P.k_nuc:.0e} の {lo/P.k_nuc:.0f} 倍）---")
    print("  これ以上だと核形成だけで目標 f_clu を超え、k_grow>0 の解が無くなる。")
    print("  ★ k_nuc は実測が無くても、校正によって上から押さえられている。")

    kns = [k for k in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3) if k < lo * 0.98]
    print(f"\n  --- 下限: 素から（ic='cytoplasm'）クラスター相に入れるか ---")
    print(f"  {'p':>5} " + " ".join(f"{k:>10.0e}" for k in kns))
    for p in (1.0, 1.1, 1.2, 1.3):
        row = []
        for kn in kns:
            r = ok(derive(P, p=p, k_nuc=kn), ic="cytoplasm")
            row.append(f"{r['f_clu']:>10.4f}" if r else f"{'校正不能':>10}")
        print(f"  {p:>5} " + " ".join(row))
    print("  値は到達した f_clu。0.2 前後 = 入れた / 0.00x = 核形成の壁に阻まれた。")
    print(f"\n  ★ p > 1 を主張するなら k_nuc ≳ 1e-3 が必要。校正の上限 {lo:.1e} と")
    print("    合わせると、p>1 は k_nuc を幅 4 倍ほどの窓に押し込む。")
    print("    既定の k_nuc = 1e-5 では p > 1 は de novo 局在と両立しない。")


def s11_p15():
    hdr("11. p = 1.5 のクラスター相は希釈で消える（Cycle 2 の訂正）")
    print("  Cycle 2 は mu=0 で p=1.5 に非対称枝（左右比 46299）を報告している。")
    print(f"  {'mu':>16} {'p':>5} " + " ".join(f"{'種'+f'{s:.0e}':>14}" for s in (10., 100., 1000.)))
    for mu, lab in ((0.0, "0 (Cycle 2)"), (P.mu, "ln2/1500 (C4)")):
        for p in (1.2, 1.5):
            row = []
            for sd in (10., 100., 1000.):
                r = popz.steady(NG, derive(P, p=p, mu=mu), ic="seed_pole", seed=sd)
                row.append(f"{r['f_clu']:>14.4f}")
            print(f"  {lab:>16} {p:>5} " + " ".join(row))
    print("  値は到達した f_clu。")
    print("\n  ★ mu>0 では p=1.5 にクラスター相が無い（どんな種からも 0.0006）。")
    print("    希釈が双安定の窓を p ≈ 1.1〜1.2 に狭める。")



def s12_absolute():
    hdr("12. ★ 良い極の絶対量で比べる")
    r0, y0 = base()
    CLU = 47.0      # 実測の平均クラスターサイズ（PARAMETERS.md §3）
    print(f"  基準（PopZ なし）の片極クラスター分子数 = {r0['scaffold_R']:.2f} 分子"
          f"（N = {P.N:.0f}）")
    print(f"  実測の平均クラスターサイズ = {CLU:.0f} 分子／クラスター数 15 個/細胞")

    print("\n  --- 絶対量は発現量 N に厳密比例するか（前提の確認）---")
    print(f"  {'N':>9} {'左右比':>9} {'足場':>11} {'足場/N':>10} {'処理量':>9} {'処理量/N':>11}")
    for Nv in (169.0, 1690.0, 16900.0):
        r = popz.steady(NG, derive(P, N=Nv, theta=0.0, k_cut=0.01,
                                   cut_target="MN"))
        print(f"  {Nv:>9.0f} {r['lr_clu']:>9.4f} {r['scaffold_R']:>11.3f} "
              f"{r['scaffold_R']/Nv:>10.6f} {r['cut_flux']:>9.4f} "
              f"{r['cut_flux']/Nv:>11.3e}")
    print("  → 一定。したがって絶対量は N=1690 の値 × (N/1690) で読める。")

    print(f"\n  --- 同じ左右比での良い極の絶対量（N = {P.N:.0f}）---")
    print(f"  {'目標':>5} {'戦略':>14} {'入力':>16} {'足場[分子]':>11} "
          f"{'クラスター換算':>14} {'処理量':>8}")
    keep = {}
    for tg in (2, 3, 5, 10):
        th = popz.theta_for_ratio(tg, P, n=NG)
        r = run(derive(P, theta=th))
        keep[("S1", tg)] = r
        print(f"  {tg:>4}倍 {'S1 占有':>14} {'θ='+f'{th:.3f}':>16} "
              f"{r['scaffold_R']:>11.2f} {r['scaffold_R']/CLU:>14.2f} "
              f"{r['cut_flux']:>8.3f}")
        for tgt, lab in (("MN", "S2 分解(M+N)"), ("M", "S2 分解(M)")):
            kc = popz.kcut_for_ratio(tg, derive(P, theta=0.0, cut_target=tgt), n=NG)
            if not np.isfinite(kc):
                continue
            r = run(derive(P, theta=0.0, k_cut=kc, cut_target=tgt))
            keep[(tgt, tg)] = r
            print(f"  {'':>5} {lab:>14} {'k='+f'{kc:.4g}':>16} "
                  f"{r['scaffold_R']:>11.2f} {r['scaffold_R']/CLU:>14.2f} "
                  f"{r['cut_flux']:>8.3f}")

    print(f"\n  --- ★ S2 が平均クラスター 1 個分（{CLU:.0f} 分子）を保てる上限 ---")
    from scipy.optimize import brentq

    def f(logk):
        r = run(derive(P, theta=0.0, k_cut=10 ** logk, cut_target="MN"))
        return r["scaffold_R"] - CLU

    lk = brentq(f, -3.5, -1.0, rtol=1e-5)
    r = run(derive(P, theta=0.0, k_cut=10 ** lk, cut_target="MN"))
    print(f"  k_cut = {10**lk:.4g} /s で足場 = {r['scaffold_R']:.1f} 分子、"
          f"そのときの左右比 = {r['lr_clu']:.3f} 倍")
    print(f"  → S2 は左右比 {r['lr_clu']:.1f} 倍を超えると"
          f"良い極にクラスター 1 個分も残らない")

    print(f"\n  --- 足場を基準の {r0['scaffold_R']:.0f} 分子に揃えるのに必要な発現量 ---")
    print(f"  {'目標':>5} {'戦略':>14} {'必要な N':>11} {'実測比':>8} {'処理量':>10}")
    for tg in (2, 5, 10):
        for key, lab in (("S1", "S1 占有"), ("MN", "S2 分解(M+N)")):
            r = keep[(key, tg)]
            sN = r["scaffold_R"] / P.N
            Nn = r0["scaffold_R"] / sN
            print(f"  {tg:>4}倍 {lab:>14} {Nn:>11.0f} {Nn/P.N:>7.2f}倍 "
                  f"{r['cut_flux']/P.N*Nn:>9.2f}/s")

    print("\n  --- 発現量で埋め合わせた S2 の代償（左右比 10 倍）---")
    r = keep[("MN", 10)]
    lam = r0["scaffold_R"] / r["scaffold_R"]
    Nn = P.N * lam
    rv = popz.steady(NG, derive(P, N=Nn, theta=0.0,
                                k_cut=popz.kcut_for_ratio(
                                    10, derive(P, theta=0.0, cut_target="MN"),
                                    n=NG), cut_target="MN"))
    print(f"  N = {Nn:.0f}（実測の {lam:.2f} 倍）で検算: "
          f"足場 {rv['scaffold_R']:.2f} 分子、左右比 {rv['lr_clu']:.3f}、"
          f"処理量 {rv['cut_flux']:.2f} 分子/s")
    print(f"  常時分解される DivIVA = {Nn*(1-rv['mass_frac']):.0f} 分子/細胞"
          f"（総量の {1-rv['mass_frac']:.0%}）")
    print("  ⚠️ ClpXP の細胞全体の処理能力の推定は「数分子/s」"
          "（PARAMETERS.md §5.1、未確認）。ここで初めて効いてくる。")
    r1 = keep[("S1", 10)]
    print(f"\n  ★ 対して S1 は N = {r0['scaffold_R']/(r1['scaffold_R']/P.N):.0f}"
          f"（実測より少ない）で同じ足場を達成し、処理量は 0。")


SECTIONS = [s1_geometry, s2_baseline, s3_s1, s4_s2, s5_match, s6_p, s7_latch,
            s7b_nucleation, s8_lead, s9_expr,
            s10_nuc_constraint, s11_p15, s12_absolute]

if __name__ == "__main__":
    want = [int(a) for a in sys.argv[1:]] or list(range(1, len(SECTIONS)+1))
    for i in want:
        SECTIONS[i-1]()
    print()
