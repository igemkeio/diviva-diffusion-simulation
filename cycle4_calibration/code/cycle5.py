"""Cycle 5 の結果をすべて再計算して印字する。

    python cycle5.py            # 全部（約 10 分）
    python cycle5.py 1 2        # 節だけ

⚠️ 本計算の前に `python model.py && python cluster.py && python forward.py` を通すこと。
⚠️ REPORT_CYCLE5.md と食い違ったら、この出力が唯一の正。

=============================================================================
  Cycle 5 の設計判断
=============================================================================

[D1] ★ Wet と比べる量は「観測される線プロファイルの指標」であって、
     モデル内部の面密度比ではない。両者は 4〜6 倍ちがう（§6）。
     Cycle 4 までの「左右比」はすべて内部の量なので、wet の数字と
     直接比べてはいけなかった。前向きモデル（forward.py）を通してから比べる。

[D2] ★ 撮像パラメータ（PSF）は wet の null に合わせて決める。光学の理論値は使わない。
     実測の null プロファイルを再現するには FWHM ≈ 0.87 µm が要る。
     NA 0.75 の回折限界 0.35〜0.42 µm の 2 倍で、原因は未特定
     （焦点ずれ / 対物の違い / 画素校正）。metadata が無いので確かめられない。
     ⚠️ 原因が何であれ、**同じ撮像で撮った DivIVA にも同じぼけが掛かる**ので、
       null で較正した値を使うのが正しい。ただし R と PSF は長軸プロファイル
       だけでは分離できない（縮退。§2）。R は物理値に固定して PSF に吸わせる。

[D3] ★ 「局在しない DivIVA の下駄」は細胞質 null ではない。
     DivIVA は 7 割が膜なので、一様な膜タンパクの投影（極で 2σR/r に増える）が
     効く。細胞質 null を基準にすると局在を 1.7 倍に過大評価する（§3）。
     wet にこの対照は無いので、モデルから出すしかない。

[D4] ★ 発現量依存から p を出すときは k_grow を分子定数として固定する。
     f_clu に校正し直す流儀はスケーリング対称性で N が厳密に効かなくなり、
     どんな p でも α = 1 になってしまう（V11）。同じ画像の中の細胞どうしは
     同じ分子を持っているので、固定するのが物理的に正しい。

[D5] 共局在（wet の R = 0.90）はモデルに入れない。
     原因が 3 つ候補にあり対照が無い（Q7 D4）。パラメータに化けさせると
     アーティファクトをモデルに焼き込むことになる。§7 で「モデルの範囲外」
     として扱い、覆る条件だけ書く。
"""

import sys
import numpy as np

import cluster as K
import forward as F
import wetdata as W

# =============================================================================
#  Cycle 5 の既定（すべて wet 由来か、wet で較正した値）
# =============================================================================
R_PHYS = 0.40          # µm  細胞半径。★ null からは PSF と縮退して決まらない（§2）
L_DIV = 2.7767         # µm  DivIVA_mCherry の実測中央値
PSF = 0.87             # µm  null に合わせた実効 PSF FWHM（設計判断 [D2]）
WLINE = 3              # px  wet の線幅
P_FIT = 0.17           # 成長の次数。実測 α から（§4）
RHO_N5 = 4.1812        # p=0.17 で 66% に再校正し、1/R² で移植（§5）
NG = 101

P5 = K.variant(R=R_PHYS, L=L_DIV, p=P_FIT, rho_n=RHO_N5)
IMG = dict(R=R_PHYS, L=L_DIV, psf_fwhm_um=PSF, line_w_px=WLINE)

# ★ Cycle 5 は cluster.py に 1 つだけ追加を要求する: steady()/solve()/_assemble() の
#   grow_block 引数（PopZ がクラスターの成長も妨げる規約。popz.py [P8] 相当）。
#   当て方は REPORT_CYCLE5.md 付録 A。未適用でも走るが、帯の片側が出せない。
HAS_GROW = "grow_block" in K.steady.__code__.co_varnames
if not HAS_GROW:
    print("⚠️ cluster.py に grow_block が無い（REPORT_CYCLE5.md 付録 A）。"
          "「成長も妨げる」列は飛ばす")


def _steady(n, par, block=None, grow=False, **kw):
    """grow_block の有無を吸収する。未適用なら grow=True は None を返す。"""
    if grow:
        if not HAS_GROW:
            return None
        return K.steady(n, par, block=block, grow_block=block, **kw)
    return K.steady(n, par, block=block, **kw)


def _block_of(theta, n, par):
    """左極の占有（面積重み。model.py 設計判断 [8]）。cluster.py に無ければ自前で作る。"""
    if hasattr(K, "block_of"):
        return K.block_of(theta, n, par)
    g = K.geometry2(n, par)
    return 1.0 - theta * (g["A_capL"] / g["A"])

# wet の実測値（anl/results/*。出典は各 Q*_REPORT.md）
OBS = dict(
    div_mch=dict(n=471, L=2.7767, snr=10.51, end_mid=0.7852, pole_mid=0.9659, f20=0.2615),
    mch_null=dict(n=117, L=2.5825, snr=43.17, end_mid=0.5211, pole_mid=0.8639, f20=0.2665),
    gfp_null=dict(n=109, L=2.4549, snr=96.82, end_mid=0.5713, pole_mid=0.8832, f20=0.2491),
    div_gfp=dict(n=79, L=2.9992, snr=20.07, end_mid=0.6324, pole_mid=0.9370, f20=0.2582),
)
ALPHA_WET = (1.055, 0.089)      # wet の下駄（細胞質 null）での α
ALPHA_FIX = (1.196, 0.272)      # 下駄を補正した α（§4）
R_SIGNED = (0.90, 0.86, 0.92)   # gDiv_mPopZ の符号付き左右比と 95%CI


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def _prof(res, par=None, **kw):
    par = par or P5
    o = dict(IMG); o.update(kw)
    o["R"], o["L"] = par.R, par.L
    return F.line_profile(C=res["C"], M=res["M"], N=res["Nf"],
                          z_model=res["g"]["z"], **o)


def _bands(v, z):
    k = max(2, int(round(len(v) * 0.12)))
    bg = np.median(np.r_[v[:k], v[-k:]])
    s = np.clip(v - bg, 0, None)
    return s


def obs_ratio(res, par=None, windows="half", **kw):
    """観測される左右比（無摂動の極 / 塞いだ極）。block は左極に掛ける規約。"""
    pr = _prof(res, par, **kw)
    s = _bands(pr["val"], pr["z"])
    z = pr["z"]
    if windows == "half":
        a, b = (0, 0.35), (0.65, 1.0)
    elif windows == "end":
        a, b = (0, 0.15), (0.85, 1.0)
    else:
        raise ValueError(windows)
    IA = s[(z >= a[0]) & (z <= a[1])].mean()
    IB = s[(z >= b[0]) & (z <= b[1])].mean()
    return IB / max(IA, 1e-12)


def flat_baseline(par=None, **kw):
    """局在しない DivIVA 相当（3 相分配のまま一様）の指標（設計判断 [D3]）。"""
    par = par or P5
    fc, fm, fn = K.fractions(K.Par)
    g = K.geometry2(NG, par)
    one = np.ones(NG)
    o = dict(IMG)
    o.update(kw)
    o["R"], o["L"] = par.R, par.L
    return F.metrics_of(C=one * fc * par.N / g["V"].sum(),
                        M=one * (fm + fn) * par.N / g["A"].sum(),
                        z_model=g["z"], **o)


# =============================================================================
def s1_calib():
    hdr("1. 撮像の較正 — 実測の null にモデルの一様細胞質を合わせる")
    from scipy.optimize import minimize_scalar
    pr, ci = W.load()

    def mp(zg, R, L, fw):
        zm = np.linspace(0, L, 1201)
        p = F.line_profile(C=np.ones_like(zm), z_model=zm, R=R, L=L,
                           psf_fwhm_um=fw, line_w_px=WLINE, nz_fine=12)
        v, z = p["val"], p["z"]
        s = _bands(v, z)
        return np.interp(zg, z, s / s[(z >= .40) & (z <= .60)].mean())

    print(f"  {'条件':13} {'n':>4} {'L(µm)':>7} {'R 固定':>7} {'PSF FWHM':>9} {'RMS':>7}")
    out = {}
    for cond, key in (("mCherry_null", "mch_null"), ("GFP_null", "gfp_null")):
        L = OBS[key]["L"]
        m = W.mean_profile(pr, ci, cond)
        zg, ob = m["z"], m["med"]
        ok = np.isfinite(ob)
        for R in (0.35, 0.40, 0.45):
            f = lambda fw: float(np.sum((mp(zg[ok], R, L, fw) - ob[ok]) ** 2))
            r = minimize_scalar(f, bounds=(0.05, 1.6), method="bounded",
                                options=dict(xatol=1e-4))
            out[(cond, R)] = r.x
            print(f"  {cond:13} {m['n']:4d} {L:7.3f} {R:7.2f} {r.x:9.4f}"
                  f" {np.sqrt(r.fun/ok.sum()):7.4f}")
    print("\n  ⚠️ R と PSF は長軸プロファイルだけでは分離できない（設計判断 [D2]）。")
    print("     どちらに寄せても null は同じくらい再現できるので、R は物理値に固定し")
    print("     PSF に吸わせる。回折限界は GFP 0.35 / mCherry 0.42 µm なので、")
    print("     実効 0.87 µm はその 2 倍。焦点ずれか対物か画素校正かは metadata 待ち。")
    return out


def s2_baseline():
    hdr("2. ★ 局在しない『下駄』— 細胞質 null は膜タンパクの基準にならない")
    cyt = F.uniform_metrics("cyt", **IMG)
    mem = F.uniform_metrics("mem", **IMG)
    flat = flat_baseline()
    print(f"  {'':40} {'end_mid':>8} {'pole_mid':>9} {'f20':>7}")
    for lab, m in (("一様な細胞質（＝遊離 FP）", cyt),
                   ("一様な膜タンパク", mem),
                   ("★局在しない DivIVA 相当（30%細胞質+70%膜）", flat)):
        print(f"  {lab:40} {m['end_mid']:8.4f} {m['pole_mid']:9.4f} {m['f20']:7.4f}")
    o = OBS["mch_null"]
    print(f"  {'実測 mCherry_null':40} {o['end_mid']:8.4f} {o['pole_mid']:9.4f}"
          f" {o['f20']:7.4f}")
    print("\n  ★ 細い線（幅 3 px）で中心だけ見ると、一様な膜タンパクでも極が明るい。")
    print("    投影密度が中心線で 2σ·R/r(z) に増えるため（forward.py [F2]）。")
    print(f"    下駄: 細胞質 {cyt['end_mid']:.4f} → DivIVA 相当 {flat['end_mid']:.4f}"
          f"（{flat['end_mid']/cyt['end_mid']:.2f} 倍）")
    for w in (1, 3, 5, 9, 15):
        a = F.uniform_metrics("cyt", R=R_PHYS, L=L_DIV, psf_fwhm_um=PSF, line_w_px=w)
        b = F.uniform_metrics("mem", R=R_PHYS, L=L_DIV, psf_fwhm_um=PSF, line_w_px=w)
        print(f"      線幅 {w:2d} px: 細胞質 {a['end_mid']:.4f} / 膜 {b['end_mid']:.4f}")
    return flat


def s3_validate(flat):
    hdr("3. ★ モデルの検証 — DivIVA 単体の予言 vs 実測")
    r = K.steady(201, P5)
    m = F.metrics_of(C=r["C"], M=r["M"], N=r["Nf"], z_model=r["g"]["z"], **IMG)
    o = OBS["div_mch"]
    print(f"  モデル定常: 極濃縮(膜) {r['enrichment']:.4f} / クラスター {r['enr_clu']:.4f}"
          f" / 極割合 {r['clu_pole_frac']:.4f}")
    print(f"\n  {'':26} {'モデル':>8} {'実測':>8} {'差':>8}")
    for k in ("end_mid", "pole_mid", "f20"):
        print(f"  {k:26} {m[k]:8.4f} {o[k]:8.4f} {m[k]-o[k]:+8.4f}")
    print(f"\n  ★ 下駄からの超過（局在の大きさ）")
    print(f"    モデル {m['end_mid']:.4f} − {flat['end_mid']:.4f} = "
          f"{m['end_mid']-flat['end_mid']:+.4f}")
    print(f"    実測   {o['end_mid']:.4f} − {flat['end_mid']:.4f} = "
          f"{o['end_mid']-flat['end_mid']:+.4f}")
    print(f"    → 一致度 {(o['end_mid']-flat['end_mid'])/(m['end_mid']-flat['end_mid']):.2f} 倍")
    cyt = F.uniform_metrics("cyt", **IMG)
    print(f"    （wet の下駄 {cyt['end_mid']:.4f} で測ると超過は "
          f"{o['end_mid']-cyt['end_mid']:+.4f} と 1.7 倍に見える）")
    print("\n  ⚠️ PSF と R への感度")
    for fw in (0.70, 0.87, 1.05):
        for RR in (0.35, 0.40, 0.45):
            mm = F.metrics_of(C=r["C"], M=r["M"], N=r["Nf"], z_model=r["g"]["z"],
                              R=RR, L=L_DIV, psf_fwhm_um=fw, line_w_px=WLINE)
            fl = flat_baseline(K.derive(P5, R=RR), psf_fwhm_um=fw)
            print(f"    PSF {fw:.2f} / R {RR:.2f}: モデル end_mid {mm['end_mid']:.4f}"
                  f"  下駄 {fl['end_mid']:.4f}  超過 {mm['end_mid']-fl['end_mid']:+.4f}")
    return r


def s4_p():
    hdr("4. ★ 成長の次数 p — 実測の発現量依存から決める")
    import pandas as pd
    q = pd.read_csv(W.os.path.join(W.WET, "results", "q2_cells.csv"))

    def fit_p(df, b, nq=5):
        rows = []
        for rep, g in df.groupby("rep"):
            g = g.copy()
            g["B"] = g.E - b * g.C
            g["q"] = pd.qcut(g.X, nq, labels=False, duplicates="drop")
            for _, gg in g.groupby("q"):
                rows.append((rep, np.median(gg.C), np.median(gg.B)))
        d = pd.DataFrame(rows, columns=["rep", "C", "B"])
        d = d[(d.B > 0) & (d.C > 0)]
        if len(d) < 4:
            return np.nan, np.nan, len(d)
        reps = sorted(d.rep.unique())
        X = np.column_stack([np.log(d.C)] + [(d.rep == r).astype(float) for r in reps])
        y = np.log(d.B)
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        res = y - X @ c
        s2 = (res ** 2).sum() / max(len(d) - X.shape[1], 1)
        return c[0], np.sqrt(s2 * np.linalg.pinv(X.T @ X)[0, 0]), len(d)

    d = q[q.condition == "DivIVA_mCherry"]
    cyt = F.uniform_metrics("cyt", **IMG)["end_mid"]
    fl = flat_baseline()["end_mid"]
    print("  実測の指数 α = dlogB/dlogC（B = 先端 − b×中央、下駄 b を振る）")
    print(f"  {'b':>8} {'由来':34} {'α':>7} {'±se':>6}")
    for b, lab in ((0.5211, "wet が使った値＝遊離 mCherry の実測"),
                   (cyt, "モデル: 一様な細胞質"),
                   (fl, "★モデル: 局在しない DivIVA 相当"),
                   (F.uniform_metrics("mem", **IMG)["end_mid"], "モデル: 一様な膜")):
        a, se, nb = fit_p(d, b)
        print(f"  {b:8.4f} {lab:34} {a:7.3f} {se:6.3f}")

    print("\n  モデルで α を計算する（k_grow は分子定数に固定。設計判断 [D4]）")
    print(f"  {'p':>5} {'α':>7} {'f_clu の範囲':>18} {'end_mid の範囲':>16}")
    facs = np.array([1 / 2.3, 1 / 1.5, 1.0, 1.5, 2.3])
    tab = []
    for p in (0.0, 0.1, 0.2, 0.35, 0.5, 0.7):
        base = K.variant(R=R_PHYS, L=L_DIV, p=p)
        kg = K._kg(base)
        Bs, Cs, fcl, em = [], [], [], []
        for f in facs:
            par = K.variant(R=R_PHYS, L=L_DIV, p=p, N=1690.0 * f, k_grow=kg)
            rr = K.steady(151, par)
            pp = _prof(rr, par)
            s = _bands(pp["val"], pp["z"])
            z = pp["z"]
            E = np.r_[s[(z >= 0) & (z <= .15)], s[(z >= .85) & (z <= 1)]].mean()
            C = s[(z >= .40) & (z <= .60)].mean()
            Bs.append(E - fl * C)
            Cs.append(C)
            fcl.append(rr["f_clu"])
            em.append(E / C)
        Bs, Cs = np.array(Bs), np.array(Cs)
        ok = Bs > 0
        a = np.polyfit(np.log(Cs[ok]), np.log(Bs[ok]), 1)[0] if ok.sum() >= 3 else np.nan
        tab.append((p, a))
        print(f"  {p:5.2f} {a:7.3f} {min(fcl):8.3f}–{max(fcl):<8.3f}"
              f" {min(em):7.3f}–{max(em):<7.3f}")
    arr = np.array(tab)
    o = np.argsort(arr[:, 1])
    for tgt, lab in ((ALPHA_FIX[0], "下駄を補正"), (ALPHA_WET[0], "wet の下駄")):
        print(f"  → α = {tgt:.3f}（{lab}）に合う p ≈ "
              f"{np.interp(tgt, arr[o,1], arr[o,0]):.3f}")
    print(f"  ★ 既定値 p = 0.5 は α = {dict(tab)[0.5]:.2f} を予言する。"
          f"実測 {ALPHA_FIX[0]:.2f}±{ALPHA_FIX[1]:.2f} とは "
          f"{(dict(tab)[0.5]-ALPHA_FIX[0])/ALPHA_FIX[1]:.1f}σ 離れている")
    return tab


def s5_rhon():
    hdr("5. p を変えると rho_n の再校正が要る")
    print(f"  {'p':>5} {'rho_n(B.sub,66%)':>17} {'→E.coli':>9} {'ln rho_n':>9} {'Q 分子':>8}"
          f" {'実測平均 47 との比':>18}")
    for p in (0.5, 0.35, 0.17, 0.05):
        bs = K.variant(R=0.425, L=2.8, p=p)
        rn = K.calibrate_rho_n(0.66, n=101, par=bs, lo=1.0, hi=60.0)
        rn_ec = np.exp(np.log(rn) * (0.425 / 0.400) ** 2)
        Q, _ = K.required_cluster_size(rn_ec, K.variant(R=0.4))
        print(f"  {p:5.2f} {rn:17.4f} {rn_ec:9.4f} {np.log(rn_ec):9.4f} {Q:8.0f}"
              f" {Q/47:18.1f}")
    print("\n  ★ p を下げると必要な曲率が増える（協同的な増幅が無いぶんを、")
    print("    もともとの偏りで稼ぐ必要がある）。Cycle 2 の「Q = 92 分子は実測の")
    print("    分布に収まる」という主張は、p = 0.17 では Q = 153 に上がる。")

    print("\n  --- 双安定性（ラッチが成り立つか）---")
    for p in (0.05, 0.17, 0.35, 0.5, 1.2):
        sv, a = K.critical_seed(K.derive(P5, p=p), n=71)
        print(f"    p={p:4.2f}: 臨界の種 {sv!s:>6}  "
              f"{'双安定（一過的な誘導で足りる）' if np.isfinite(sv) else '単安定（PopZ は出し続ける）'}")
    print("    ★ 実測の p = 0.17 は単安定。Cycle 2/4 の『p>1 ならラッチ』の枝は消える。")


def s6_theta():
    hdr("6. ★ θ スキャン — 内部の左右比と、観測される左右比")
    print(f"  {'θ':>5} | {'クラスター':>10} {'膜':>8} | {'観測(half窓)':>12}"
          f" {'観測(end窓)':>12} | {'足場 n_R':>9}")
    rows = []
    for th in (0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        b = _block_of(th, NG, P5)
        r = _steady(NG, P5, block=b)
        g = r["g"]
        clu = (r["Nf"] * g["A_capR"]).sum() / max((r["Nf"] * g["A_capL"]).sum(), 1e-300)
        mem = ((r["M"] + r["Nf"]) * g["A_capR"]).sum() / \
              max(((r["M"] + r["Nf"]) * g["A_capL"]).sum(), 1e-300)
        rows.append((th, clu, mem, obs_ratio(r), obs_ratio(r, windows="end"),
                     (r["Nf"] * g["A_capR"]).sum()))
        print(f"  {th:5.2f} | {clu:10.3f} {mem:8.3f} | {rows[-1][3]:12.3f}"
              f" {rows[-1][4]:12.3f} | {rows[-1][5]:9.2f}")

    print("\n  --- θ=0.9 で、内部の 6 倍が観測ではどこまで潰れるか ---")
    r = _steady(NG, P5, block=_block_of(0.9, NG, P5))
    g = r["g"]
    print(f"    内部のクラスター左右比                    "
          f"{(r['Nf']*g['A_capR']).sum()/(r['Nf']*g['A_capL']).sum():8.3f}")
    # ⚠️ 「クラスターだけ」は C も M も消すこと。C を消し忘れて
    #    細胞質込みの値（1.52）を「クラスターだけ」と誤って出したことがある。
    for lab, drop, psf in (("クラスターだけ・PSF 無し", "CM", 0.0),
                           ("クラスターだけ・PSF 0.87", "CM", PSF),
                           ("膜（単量体+クラスター）・PSF 無し", "C", 0.0),
                           ("膜（単量体+クラスター）・PSF 0.87", "C", PSF),
                           ("全部・PSF 無し", "", 0.0),
                           ("全部・PSF 0.35（回折限界）", "", 0.35),
                           ("全部・PSF 0.87（実測相当）", "", PSF)):
        rr = dict(C=r["C"], M=r["M"], Nf=r["Nf"], g=r["g"])
        if "C" in drop:
            rr["C"] = np.zeros_like(r["C"])
        if "M" in drop:
            rr["M"] = np.zeros_like(r["M"])
        print(f"    {lab:34} {obs_ratio(rr, psf_fwhm_um=psf):8.3f}"
              f"   （end 窓 {obs_ratio(rr, windows='end', psf_fwhm_um=psf):.3f}）")

    print("\n  --- 観測できる左右比の天井（θ=0.98）---")
    r98 = _steady(NG, P5, block=_block_of(0.98, NG, P5))
    for fw in (0.0, 0.20, 0.35, 0.50, 0.70, PSF):
        print(f"    PSF {fw:4.2f} µm: 上限 {obs_ratio(r98, psf_fwhm_um=fw):.3f}"
              f"（end 窓なら {obs_ratio(r98, windows='end', psf_fwhm_um=fw):.3f}）")
    return rows


def s7_required():
    hdr("7. ★ Wet へ渡す表 — 目標の『観測される』左右比に必要な θ")
    from scipy.optimize import brentq
    print(f"  {'目標 観測 R':>12} {'θ（結合のみ）':>14} {'θ（成長も妨げる）':>18}")
    for tgt in (1.05, 1.10, 1.20, 1.30, 1.40, 1.50, 2.00):
        row = []
        for grow in (False, True):
            if grow and not HAS_GROW:
                row.append("未適用")
                continue

            def f(th, grow=grow):
                return obs_ratio(_steady(NG, P5, block=_block_of(th, NG, P5),
                                         grow=grow)) - tgt
            try:
                row.append("届かない" if f(0.999) < 0
                           else f"{brentq(f, 0.001, 0.999, xtol=2e-3):.3f}")
            except Exception:
                row.append("—")
        print(f"  {tgt:12.2f} {row[0]:>14} {row[1]:>18}")
    print("\n  ⚠️ Cycle 4 §7 の表（2 倍に θ=0.17〜0.35 など）は**内部のクラスター比**。")
    print("     観測量では 2 倍はどの θ でも届かない。表を差し替えること。")
    print(f"\n  実測: gDiv_mPopZ の符号付き左右比 R = {R_SIGNED[0]}"
          f" [{R_SIGNED[1]}, {R_SIGNED[2]}]（n=343）")
    print("     → モデルの範囲（1.00〜1.54）の外。θ では表現できない（設計判断 [D5]）")


def main():
    which = set(sys.argv[1:])
    run = lambda k: (not which) or (k in which)
    if run("1"):
        s1_calib()
    flat = s2_baseline() if run("2") or run("3") or run("4") else None
    if run("3"):
        s3_validate(flat)
    if run("4"):
        s4_p()
    if run("5"):
        s5_rhon()
    if run("6"):
        s6_theta()
    if run("7"):
        s7_required()
    print("\n" + "=" * 78)
    print("完了。REPORT_CYCLE5.md の数値と突き合わせること。")
    print("=" * 78)


if __name__ == "__main__":
    main()
