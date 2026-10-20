"""
REPORT_CYCLE2.md / WIKI_CYCLE2.md に書いた数値をすべて再計算して印字する。

    python results2.py

レポートの記述と食い違ったら、レポートのほうが古い。
新しい数字を貼り直すこと（この出力が唯一の正）。
実行時間は約 4 分。

Cycle 1 の `results.py` に対応する。節番号は REPORT_CYCLE2.md に合わせてある。
"""

import numpy as np
from scipy.optimize import brentq

import model as M
import cluster as K
from cluster import Par, variant, derive

R_BSUB, L_BSUB = 0.425, 2.8
GEN = 1500.0                     # s, 一世代
FRAP_WT, FRAP_DMINJ = 128.64, 50.9   # s, Feddersen 2021 Tab.2
CYCLE1_FRAP = 0.0989             # s, Cycle 1 の半回復時間（REPORT §4.5）


def h(s):
    print(f"\n{'='*72}\n{s}\n{'='*72}")


def _half(t, y, lvl=0.5):
    """y を 0→1 に正規化して lvl に達する時刻を線形補間で返す。"""
    y = (np.asarray(y, float) - y[0]) / (y[-1] - y[0])
    k = int(np.argmax(y >= lvl))
    if k == 0:
        return float(t[0])
    f = (lvl - y[k - 1]) / max(y[k] - y[k - 1], 1e-300)
    return float(t[k - 1] + f * (t[k] - t[k - 1]))


def main():
    # ---------------------------------------------------------------- 検証
    h("0. 検証（これが落ちたら以下は無意味）")
    if not K.validate(verbose=True):
        raise SystemExit("validation failed")

    # ---------------------------------------------------------------- 幾何
    h("1. 幾何とモデル構成（REPORT §3）")
    g = K.geometry2(401)
    a = g["A_cap"].sum() / g["A"].sum()
    print(f"  R, L              = {Par.R}, {Par.L} µm  （Cycle 1 と同一）")
    print(f"  膜面積 A          = {g['A'].sum():.3f} µm²")
    print(f"  細胞体積 V        = {g['V'].sum():.3f} µm³")
    print(f"  極キャップ比 a    = {a:.4f}")
    print(f"  単量体の曲率重み  側壁 {g['w'][200]:.4f} / 極 {g['w'][0]:.4f}"
          f"   （rho = {Par.rho}、曲率に無関心）")
    print(f"  クラスターの重み  側壁 {g['w_n'][200]:.4f} / 極 {g['w_n'][0]:.4f}"
          f"   （rho_n = {Par.rho_n}）")
    kn = K.k_off_n_of(g, Par)
    print(f"  k_off_n(z)        極 {kn[0]:.5f} /s（寿命 {np.log(2)/kn[0]:.0f} s）"
          f" / 側壁 {kn[200]:.5f} /s（寿命 {np.log(2)/kn[200]:.0f} s）")
    print(f"  anchor            {Par.k_off_n_anchor}"
          f"  （実測 {Par.k_off_n} /s = T1/2 {np.log(2)/Par.k_off_n:.0f} s をどこに置くか）")

    # -------------------------------------------------------------- 定常
    h("2. 定常状態（REPORT §4.1）")
    r = K.steady(201, Par)
    print(f"  rho_n             = {Par.rho_n}")
    print(f"  クラスターの極濃縮 = {r['enr_clu']:.4f}")
    print(f"  極にいるクラスター分子の割合 = {r['clu_pole_frac']:.4f}"
          f"   （校正の目標 0.66）")
    print(f"  膜全体（単量体+クラスター）の極濃縮 = {r['enrichment']:.4f}")
    print(f"  ★ 蛍光で見えるのは後者。前者と混同しないこと")
    print(f"  3 相分配 cyt/mono/clu = {r['f_cyt']:.4f} / {r['f_mono']:.4f}"
          f" / {r['f_clu']:.4f}   （実測 0.296/0.507/0.198）")
    print(f"  細胞質の勾配      = {r['cyto_var']:.2e}"
          f"   （Cycle 1 は 15%。単量体の結合が一様なので勾配が立たない）")
    es = [K.steady(k, Par)["enr_clu"] for k in (51, 101, 201, 401)]
    print("  格子収束（クラスター濃縮）n=51..401: "
          + " -> ".join(f"{e:.4f}" for e in es)
          + f"   幅 {max(abs(np.diff(es)))/es[-1]:.2%}")
    print(f"  k_grow（校正値）  = {r['k_grow']:.4e}")
    print(f"  well-mixed 推定   = {K.k_grow_wellmixed():.4e}"
          f"   （比 {r['k_grow']/K.k_grow_wellmixed():.2f}）")
    print(f"  定常残差 / 収支誤差 = {r['steady_residual']:.1e} / {r['budget_err']:.1e}")
    print("\n  Cycle 1 との対比")
    e1 = M.solve(201, M.Par, T=20.0)["enrichment"]
    print(f"    Cycle 1（単量体のみ、rho={M.Par.rho}）の極濃縮 = {e1:.4f}")
    print(f"    Cycle 2 の膜全体の極濃縮                      = {r['enrichment']:.4f}")
    print(f"    Cycle 1 の天井 1/a = {1/a:.3f}"
          f"（膜拡散があるため到達不能。クラスターにはこの天井が無い）")

    # ---------------------------------------------------------- 時間スケール
    h("3. 時間スケール（REPORT §4.2）★ Cycle 1 から質的に変わった")
    rc = K.solve(101, Par, T=1e5, ic="cytoplasm", n_out=200)
    e, t = rc["enr_clu_t"], rc["t"]
    ok = np.isfinite(e) & (t > 0)
    e, t = e[ok], t[ok]
    ess = e[-1]
    print("  発現開始（全分子が細胞質）からの立ち上がり")
    for frac in (0.5, 0.9, 0.99):
        k = int(np.argmax(np.abs(e - ess) / ess < (1 - frac)))
        print(f"    クラスター濃縮が定常の {frac:.0%} に達する時刻 = {t[k]:8.1f} s"
              f"   （世代時間 {GEN:.0f} s の {t[k]/GEN:.2f} 倍）")
    A = rc["g"]["A"]
    fclu_t = (rc["Nt"] * A[:, None]).sum(0) / rc["N0"]
    print("\n    時刻ごとのスナップショット")
    print(f"    {'t (s)':>9} {'f_clu':>8} {'クラスター濃縮':>14} {'膜全体':>9}")
    for tt in (1.0, 100.0, GEN, 1e4, 1e5):
        i = int(np.argmin(np.abs(rc["t"] - tt)))
        print(f"    {rc['t'][i]:9.0f} {fclu_t[i]:8.4f} {rc['enr_clu_t'][i]:14.3f}"
              f" {rc['enr_t'][i]:9.3f}")
    rc1 = M.solve(201, M.Par, T=3600.0, ic="cytoplasm", n_out=300)
    e1t, t1 = rc1["enr_t"], rc1["t"]
    ok1 = np.isfinite(e1t) & (t1 > 0)
    e1t, t1 = e1t[ok1], t1[ok1]
    k1 = int(np.argmax(np.abs(e1t - e1t[-1]) / e1t[-1] < 0.01))
    print(f"\n  Cycle 1 は定常の 1% 以内に {t1[k1]:.3f} s で入る"
          f"（世代時間の 1/{GEN/t1[k1]:.0f}）→ 準静的に扱えた")
    print(f"  Cycle 2 は {t[int(np.argmax(np.abs(e-ess)/ess < 0.01))]:.0f} s"
          f" → 世代時間と同程度。★ 局在は分裂との競争になる")

    # ---------------------------------------------------- 校正とエネルギー
    h("4. 校正とエネルギー（REPORT §4.3）★ Cycle 2 の主結果")
    dG = M.monomer_curvature_energy()
    print(f"  単量体 1 個の曲率ミスマッチ dG = {dG:.5f} kT"
          f"   （lp=100 nm, L=30 nm, R={Par.R*1000:.0f} nm）")
    print(f"\n  Cycle 1: 単量体 1 個に ln rho = {np.log(M.Par.rho):.4f} kT を要求")
    print(f"           = dG の {np.log(M.Par.rho)/dG:.0f} 倍  → 物理的に正当化できない")
    Q, _ = K.required_cluster_size(Par.rho_n)
    print(f"  Cycle 2: クラスターに ln rho_n = {np.log(Par.rho_n):.4f} kT を要求")
    print(f"           = {Q:.0f} 分子ぶん  → 実測の平均クラスターサイズ 47 分子の"
          f" {Q/47:.1f} 倍")
    print(f"           （最大クラスター 1198 分子。分布の中に収まる）")
    print("\n  校正の経路（Cycle 1 と同じ手順）")
    print(f"    B. subtilis 幾何（R={R_BSUB}, L={L_BSUB}）で極割合 66% に合わせる")
    pb = variant(R=R_BSUB, L=L_BSUB)
    rn_b = K.calibrate_rho_n(0.66, par=pb)
    print(f"      -> rho_n = {rn_b:.4f}   (ln = {np.log(rn_b):.4f} kT)")
    for lab, expo in [("1/R²（曲率エネルギーの形。採用）", 2), ("1/(2R)（Cycle 1 の形）", 1)]:
        rn = float(np.exp(np.log(rn_b) * (R_BSUB / Par.R) ** expo))
        rr = K.steady(101, variant(rho_n=rn))
        print(f"      移植 {lab:32} rho_n = {rn:.4f}"
              f"  -> 極割合 {rr['clu_pole_frac']:.4f}, 濃縮 {rr['enr_clu']:.4f}")
    print(f"    採用値 rho_n = {Par.rho_n}")
    print("\n  lp 感度（必要クラスターサイズ Q）")
    for lp in (30, 100, 300, 1000):
        Ql, dGl = K.required_cluster_size(Par.rho_n, lp=float(lp))
        print(f"    lp={lp:5d} nm -> dG={dGl:.5f} kT, Q={Ql:6.0f} 分子"
              f"   (Cycle 1 のギャップは {np.log(M.Par.rho)/dGl:.0f} 倍)")

    # -------------------------------------------------------------- 感度
    h("5. 感度（REPORT §4.4）")
    base = K.steady(101, Par)
    print(f"  baseline: クラスター濃縮 {base['enr_clu']:.4f}, "
          f"極割合 {base['clu_pole_frac']:.4f}")
    print("  （以下はすべて n=101。§2 の n=201 とは格子分だけ違う。"
          "比較は相対値で見ること）\n")
    specs = [("rho_n", [Par.rho_n / 2, Par.rho_n, Par.rho_n * 2]),
             ("p", [0.3, 0.5, 0.8]),
             ("k_off_n", [0.0054, 0.0115, 0.0136]),
             ("f_nm", [0.0, 0.5, 1.0]),
             ("f_clu", [0.10, 0.198, 0.50]),
             ("D_m", [0.01, 0.1, 1.0]),
             ("k_nuc", [1e-6, 1e-5, 1e-4]),
             ("N", [169.0, 1690.0, 169000.0])]
    print(f"  {'パラメータ':>10}  値 -> クラスター濃縮")
    for name, vals in specs:
        es = []
        for v in vals:
            kw = {name: v}
            if name == "f_clu":      # 分配比は和を保つ必要がある
                kw = {"f_clu": v, "f_mono": Par.f_mono + Par.f_clu - v}
            es.append(K.steady(101, variant(**kw))["enr_clu"])
        span = max(es) - min(es)
        print(f"  {name:>10}  " + " | ".join(f"{v:g}->{x:.4f}"
                                             for v, x in zip(vals, es))
              + f"    幅 {span:.4f}")
    print("\n  ※ N は k_grow を f_clu に校正する限り厳密に効かない（V11、"
          "スケーリング対称性）")

    # -------------------------------------------------------------- FRAP
    h("6. FRAP と動力学（REPORT §4.5）★ Cycle 1 の 3 桁のずれが解消")
    rb = K.solve(101, Par, T=3e3, ic="bleach", n_out=300)
    cap = rb["g"]["A_capL"]
    mono = (rb["Mt"] * cap[:, None]).sum(0)
    clu = (rb["Nt"] * cap[:, None]).sum(0)
    tot = mono + clu
    amp_c = (clu[-1] - clu[0]) / (tot[-1] - tot[0])
    t_fast, t_slow = _half(rb["t"], mono), _half(rb["t"], clu)
    print("  片極の膜シグナル（単量体+クラスター）を漂白したときの回復")
    print(f"    速相 T1/2 = {t_fast*1000:8.1f} ms   振幅 {100*(1-amp_c):.1f}%  （単量体）")
    print(f"    遅相 T1/2 = {t_slow:8.1f} s    振幅 {100*amp_c:.1f}%  （クラスター）")
    print(f"    ★ 二相になるのは Cycle 2 の予測。単一指数でフィットしてはいけない")
    print(f"\n    実測（単一指数フィット）: 野生型 {FRAP_WT} s / dminJ {FRAP_DMINJ} s")
    print(f"    遅相とのずれ: {t_slow/FRAP_WT:.1f} 〜 {t_slow/FRAP_DMINJ:.1f} 倍（モデルが遅い）")
    print(f"    Cycle 1（単量体のみ）: {CYCLE1_FRAP*1000:.1f} ms → "
          f"{FRAP_WT/CYCLE1_FRAP:.0f} 〜 {FRAP_DMINJ/CYCLE1_FRAP:.0f} 倍 速すぎた")
    print(f"    ★ ずれは 3 桁から 1 桁未満へ縮み、向きが逆転した")

    # ------------------------------------------------------------ 持続摂動
    h("7. 持続摂動と利得（REPORT §4.6）★ Cycle 1 の「利得 ≤ 1」が破れた")
    print(f"  {'theta':>6} {'左右比(無摂動/塞いだ極)':>22} {'局所平衡 1/(1-θ)':>18}"
          f" {'利得':>8}")
    ths = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    gains = []
    for th in ths:
        a_ = K.sustained_asymmetry(th, n=201)
        asym = 1.0 / a_
        ideal = 1.0 / (1.0 - th)
        gains.append(asym / ideal)
        print(f"  {th:6.2f} {asym:22.3f} {ideal:18.3f} {asym/ideal:8.3f}")
    print(f"\n  ★ 利得が 1 を超える領域がある（最大 {max(gains):.2f}）。"
          "Cycle 1 は全域で 1 未満だった")
    print("  Cycle 1 で必要だった被覆率との比較（目標の左右比を出すのに要る theta）")
    print(f"  {'目標':>6} {'Cycle 1':>10} {'Cycle 2':>10}")
    for target in (2, 3, 5, 10):
        th1 = brentq(lambda x: M.asymmetry(x) - target, 0.01, 0.985, xtol=1e-4)
        try:
            th2 = brentq(lambda x: 1.0 / K.sustained_asymmetry(x) - target,
                         0.01, 0.99, xtol=1e-3)
            s2 = f"{th2:10.3f}"
        except ValueError:
            s2 = "     届かず"
        print(f"  {target:6d} {th1:10.3f} {s2}")

    # -------------------------------------------------------- 双安定性
    h("8. 双安定性とラッチ（REPORT §4.7）★ 次に測るべき量")
    print("  成長の次数 p を振る（p 以外は同じ。k_grow は毎回 f_clu に再校正）")
    print(f"  {'p':>5} {'臨界の種':>10} {'非対称枝の左右比':>18} {'双安定':>8}")
    for row in K.bistability_scan([0.5, 0.8, 1.0, 1.1, 1.2, 1.5], n=101):
        print(f"  {row[0]:5.2f} {row[1]:10.4g} {row[2]:18.6g} {str(row[3]):>8}")
    print("\n  PopZ を 600 s かけて外したあと非対称が残るか（ラッチ試験）")
    print(f"  {'p':>5} {'theta':>6} {'外す直前':>12} {'最終':>14}")
    for p in (0.5, 1.2):
        for th in (0.5, 0.9):
            held, fin = K.latch_test(theta=th, t_release=600.0,
                                     par=variant(p=p), T=1e5)
            print(f"  {p:5.2f} {th:6.2f} {1/held:12.3f} {1/fin:14.4g}")
    print("  ※ 最終が 1 なら固定されない（PopZ の持続発現が要る）")
    print("\n  発現量 N の扱いによる分岐（k_grow を p=2 の校正値に固定した場合）")
    kg2 = K._kg(variant(p=2.0))
    print(f"  k_grow = {kg2:.4e} に固定")
    print(f"  {'N':>9} {'臨界の種':>10} {'非対称枝の左右比':>18}")
    for N in (1690.0, 5000.0, 16900.0, 169000.0):
        sv, aa = K.critical_seed(variant(p=2.0, N=N, k_grow=kg2), n=71)
        print(f"  {N:9.0f} {sv:10.4g} {aa:18.6g}")
    print("  ※ 発現量には最適な窓がある（低すぎると双安定にならず、"
          "高すぎると両極が飽和して対称に戻る）")

    # ------------------------------------------------------ Cycle 1 との比較
    h("9. Cycle 1 との比較（REPORT §6）")
    rows = [
        ("極濃縮（膜全体）", f"{e1:.2f}", f"{r['enrichment']:.2f}"),
        ("極濃縮（クラスター）", "—", f"{r['enr_clu']:.2f}"),
        ("要求される結合エネルギー",
         f"{np.log(M.Par.rho):.2f} kT / 単量体1個",
         f"{np.log(Par.rho_n):.2f} kT / クラスター"),
        ("単量体エネルギーとの比",
         f"{np.log(M.Par.rho)/dG:.0f} 倍（正当化不能）",
         f"{Q:.0f} 分子ぶん（実測47の{Q/47:.1f}倍）"),
        ("局在の確立時間", f"{t1[k1]:.2f} s", f"{t[int(np.argmax(np.abs(e-ess)/ess<0.01))]:.0f} s"),
        ("FRAP 半回復", f"{CYCLE1_FRAP*1000:.0f} ms（1300倍速い）",
         f"二相 {t_fast*1000:.0f} ms + {t_slow:.0f} s（遅相が{t_slow/FRAP_WT:.1f}倍遅い）"),
        ("持続摂動の利得", "≤ 1（増幅なし）", f"最大 {max(gains):.2f}（増幅あり）"),
        ("一過的摂動の記憶", "無し", "p>1 なら有り"),
        ("発現量の効果", "厳密にゼロ", "校正次第（V11）"),
    ]
    w = max(len(x[0]) for x in rows)
    print(f"  {'':{w}}  {'Cycle 1':>32}  {'Cycle 2':>34}")
    for k_, v1, v2 in rows:
        print(f"  {k_:{w}}  {v1:>32}  {v2:>34}")

    print("\n" + "=" * 72)
    print("完了。REPORT_CYCLE2.md / WIKI_CYCLE2.md の数値と突き合わせること。")
    print("=" * 72)


if __name__ == "__main__":
    main()
