"""
DivIVA 局在モデル — Cycle 1（単量体のみ・クラスターなし）

    M_(n-1) <-> M_n <-> M_(n+1)     膜上の拡散（距離は弧長 ds）
    C_(n-1) <-> C_n <-> C_(n+1)     細胞質の拡散（距離は軸 dz）
    M_n <-> C_n                     結合・解離

細胞を軸 z で等間隔にスライスし、各スライスが膜区画 M_n と細胞質区画 C_n を持つ。

    python model.py     # 検証だけ走る（本計算の前に必ず通すこと）

=============================================================================
  設計判断と、その根拠（すべて数値検証済み）
=============================================================================

[1] 細胞質は「軸 z」で切る。弧長 s ではない。
    極付近では細胞輪郭が軸にほぼ垂直なので、弧長を軸座標だと思って断面積を
    割り当てると先端セルの体積が潰れる（W_min が 75 倍小さくなり計算が発散）。
    z で切れば先端セル体積 = πR·dz² で、質量誤差 1e-13、計算 0.2 秒。

[2] 面積重み付きの「セル平均」を使う。セル中心の点の値は使わない。
    キャップ/円柱の接合部は幾何的な不連続。セル中心で判定すると格子を変える
    たび接合部が飛び、収束が非単調になる（実測 3.9% 誤差）。
      セル中心方式 n=51..801: 1.904 -> 1.936 -> 1.901 -> 1.897 -> 1.863（非単調）
      セル平均方式 n=51..801: 1.9218 -> 1.9239 -> 1.9248 -> 1.9252 -> 1.9252（単調）

[3] 膜どうしの伝導度には弧長 ds を使う。dz ではない。
    先端では ds/dz = 7.65。dz を使うと膜拡散を 7.7 倍過大評価する。
    （ただし k_off が速いため答えへの影響は +0.3%。正しくやるが慌てなくてよい）

[4] A/V は「細胞質側の結合項だけ」に掛ける。単なる単位換算。
      C は 分子/µm³、M は 分子/µm²、flux は 分子/(µm²·s)
      -> dC/dt には A/V が要る。dM/dt には要らない。拡散演算子にも要らない。
    間違えると分子数が 1690 -> 5786 に増える。極濃縮は「それらしい」値が
    出てしまい結果からは気づけないので、質量保存の検算が唯一の防波堤。

[5] 細胞質は一様ではない。約 15% の勾配がある。
    極で優先的に結合 -> 膜上を拡散して側壁へ -> 解離 -> 細胞質を通って極へ戻る、
    という定常循環が生じるため。rho=1 にすると勾配はぴったり 0% になり、
    膜拡散が原因であることが確認できる（A/V の発散は無関係）。

[6] BDF（陰解法）必須。最も硬いのは細胞質拡散 D_c/dz² ≈ 7800 /s（n=101）。
    陽解法なら 10 秒の積分に約 8 万ステップ必要。
    rtol は 1e-9 で十分（1e-6 と 1e-12 で答えは同一）。

[8] 局所摂動（PopZ の占有）の境界も面積重みで入れる。[2] と同じ理由。
    セル中心判定だと格子を変えるたびブロック境界が飛び、収束が非単調になる。
      theta=0.94 での左右比 n=101..801:
        セル中心   8.96 -> 10.60 -> 10.04 -> 10.55（非単調）
        面積重み   9.50 -> 10.16 -> 10.26 -> 10.44（単調）
    ⚠️ ただし theta が 1 に近いと左右比は急峻になり、格子依存性が残る:
        theta=0.56 (2x)  n=201..801 で 0.27% の幅
        theta=0.85 (5x)  同           1.1%
        theta=0.94 (10x) 同           2.8%
    theta を 2 桁で引用するぶんには n=201 で足りるが、
    10 倍以上の非対称を論じるときは n=401 以上を使うこと。

[7] 格子は n=101 で十分。収束値から 0.1%、0.2 秒。
    n=401 まで上げても変わらない（プロファイルの図を描くときだけ）。
    n を上げすぎると nfine 不足・条件数悪化・計算時間爆発で損しかない。

=============================================================================
  パラメータの独立性
=============================================================================
    k_off  独立・実測（SPT 滞在時間 0.11 s）            影響: 中（拡散長経由）
    D_m    独立・実測（SPT「低速移動」集団）             影響: 大
    f_cyt  実測だが揺れる（撮影速度で 15.7〜29.6%）      影響: 小
    k_on   ✗ 独立でない。k_off と f_cyt から導出          影響: 小
    rho    独立・★これが求める量                        影響: 大

  固定されるのは k_on の「面積積分」:
      ∫ k_on dA = k_off · V · (1 - f_cyt) / f_cyt
  これは rho に依らない。rho は総量ではなく「極と側壁への配分」だけを決める。
  そのため f_cyt が 3.6 倍振れても極濃縮は 5% しか動かない（細胞質勾配経由の
  二次効果のみ）。一番不確かな量が一番効かない構造。

  なお k_on と k_off を同時にλ倍しても定常は不変にならない。D_m が実測値で
  固定されているため、拡散長 sqrt(D_m/k_off) が変わるから。縮退していない。

=============================================================================
  既知の限界
=============================================================================
  - クラスターを含まない。単量体だけでは実測の極濃縮に届かない
    （必要な rho は 8.76 で、単量体の曲率エネルギーの 231 倍。正当化できない）。
  - N=1690 は B. subtilis の天然発現量。大腸菌の異種発現なら 10〜100 倍に
    なりうる。Par.N はスキャン変数として扱うこと。
  - 幾何と D_c は大腸菌だが、DivIVA の速度定数は B. subtilis 由来。
    大腸菌に DivIVA は無いので置き換える先が存在しない。
    特に k_off,n（クラスター）は大腸菌に MinJ が無いため
    野生型 0.0054 /s ではなく dminJ の 0.0136 /s が対応する可能性が高い。
  - 隔壁（セプタム）と PopZ は本ファイルでは扱わない。追加方法は末尾を参照。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


# =============================================================================
class Par:
    """既定パラメータ。

    上書きは `variant(rho=2.0)` を使う（クラス属性を直接書き換えないこと。
    他の計算に漏れる）。
    """

    # =====================================================================
    #  ★ 大腸菌（E. coli）を対象とした設定
    #
    #  幾何と D_c は大腸菌の値。それ以外は B. subtilis の DivIVA 測定値で、
    #  大腸菌に DivIVA は存在しないため置き換える先がない。
    # =====================================================================

    # --- 幾何：大腸菌（指数増殖期の代表値。★自分の株で実測して置き換えること）
    R = 0.4            # µm   細胞半径（幅 0.8 µm）
    L = 3.0            # µm   細胞長
    #   参考: B. subtilis のモデル細胞は R=0.425, L=2.8（Feddersen 2021）
    #         大腸菌は増殖条件で L が 2〜4 µm、幅 0.65〜1.0 µm と大きく振れる。
    #         極キャップ面積比 a = 2R/L は L=2.5/3.0/4.0 で 0.320/0.267/0.200 と
    #         動き、これは答えに効くので実測が要る。

    # --- 実測パラメータ
    D_c = 3.0          # µm²/s 大腸菌細胞質（Elowitz 1999 の GFP 7.7 を融合サイズ補正）
    # ↓ 以下 4 つは B. subtilis の DivIVA。大腸菌での対応値は存在しない
    D_m = 0.1          # µm²/s 膜上拡散（SPT「低速移動」集団）
    k_off = 9.0        # 1/s   解離（SPT 滞在時間 0.11 s）
    N = 1690.0         # 分子/細胞（B. subtilis 天然値。大腸菌では要スキャン）
    f_cyt = 0.30       # 定常での細胞質割合（SPT「高速移動」集団）

    # --- 求める量
    rho = 8.76         # 極/側壁の結合能比
    #   B. subtilis 幾何で極濃縮 2.2 倍に校正すると rho = 7.70（ln rho = 2.042 kT）。
    #   rho は分子の性質なので ln rho = alpha/(2R) の関係で大腸菌へ移植する:
    #     ln rho_Ec = 2.042 * (0.425/0.4) = 2.170  ->  rho = 8.76
    #   この値での大腸菌の極濃縮 2.43 倍は【予測】であり、Wet で検証できる。

    # --- 数値
    rtol, atol = 1e-9, 1e-11


def variant(**kw):
    """Par を一部だけ上書きした新しいパラメータクラスを返す。

        p = variant(rho=2.0, L=4.0)
        solve(101, p)

    Par 自体を書き換えないので、スキャン中に他の計算へ漏れない。
    """
    unknown = set(kw) - {k for k in vars(Par) if not k.startswith("_")}
    if unknown:
        raise ValueError(f"Par に無い属性: {sorted(unknown)}")
    return type("ParVariant", (Par,), kw)


# =============================================================================
def geometry(n, par=Par, nfine=None):
    """面積重み付きセル平均された幾何量を返す（設計判断 [1][2]）。

    戻り値の各配列は長さ n:
        z      セル中心の軸座標 [µm]
        V      セル体積 [µm³]
        A      セル膜面積 [µm²]
        S      セル弧長 [µm]（膜の伝導度用、設計判断 [3]）
        A_cap  そのセルのうち極キャップぶんの膜面積 [µm²]
        A_capL / A_capR  同上を左右の極に分けたもの（局所摂動の面積重み用）
        w      面積重み付きセル平均した結合重み（側壁で 1、極で rho）
        rc     セル中心での細胞半径 [µm]
    """
    R, L = par.R, par.L
    Lcyl = L - 2 * R
    if Lcyl <= 0:
        raise ValueError(f"L ({L}) は 2R ({2*R}) より大きい必要がある")
    if nfine is None:
        nfine = max(20001, 40 * n)          # セルあたり 40 点以上を確保

    ze = np.linspace(0, L, n + 1)
    zf = np.linspace(0, L, nfine)
    dzf = np.gradient(zf)

    r = np.full_like(zf, R)
    a = zf < R
    r[a] = np.sqrt(np.maximum(R**2 - (zf[a] - R) ** 2, 0))
    b = zf > R + Lcyl
    r[b] = np.sqrt(np.maximum(R**2 - (zf[b] - R - Lcyl) ** 2, 0))

    dsdz = np.sqrt(1 + np.gradient(r, zf) ** 2)   # 極で R/r に発散
    band = 2 * np.pi * r * dsdz                   # dz あたりの膜面積
    is_cap = (zf < R) | (zf > R + Lcyl)
    H = np.where(is_cap, 1 / R, 1 / (2 * R))      # 平均曲率: 極 1/R、側壁 1/2R
    w = par.rho ** ((H - 1 / (2 * R)) / (1 / R - 1 / (2 * R)))
    # ⚠️ H を輪郭からではなく step で置いている。スフェロシリンダーなら厳密だが、
    #    隔壁（くびれ）を入れるときはここを輪郭から計算し直す必要がある。

    i = np.clip(np.searchsorted(ze, zf) - 1, 0, n - 1)
    acc = lambda f: np.bincount(i, f * dzf, minlength=n)

    A = acc(band)
    return dict(
        n=n, dz=L / n, z=0.5 * (ze[:-1] + ze[1:]),
        V=acc(np.pi * r**2),
        A=A,
        S=acc(dsdz),
        A_cap=acc(band * is_cap),
        A_capL=acc(band * (zf < R)),
        A_capR=acc(band * (zf > R + Lcyl)),
        w=acc(band * w) / A,          # ★ 面積重み付きセル平均（設計判断 [2]）
        rc=np.interp(0.5 * (ze[:-1] + ze[1:]), zf, r),
    )


def operators(g, par=Par):
    """細胞質と膜、それぞれの拡散演算子（保存形・有限体積）を返す。"""
    n, dz = g["n"], g["dz"]
    rf = 0.5 * (g["rc"][:-1] + g["rc"][1:])
    G_c = par.D_c * np.pi * rf**2 / dz                                   # 距離 dz
    G_m = par.D_m * 2 * np.pi * rf / (0.5 * (g["S"][:-1] + g["S"][1:]))  # 距離 ds

    def lap(G, W):
        di = np.zeros(n)
        di[:-1] -= G
        di[1:] -= G
        return diags([G / W[1:], di / W, G / W[:-1]], [-1, 0, 1], format="csr")

    return lap(G_c, g["V"]), lap(G_m, g["A"])


def k_on_of(g, par=Par, block=None):
    """k_on(z) を返す [µm/s]。

    定常での細胞質割合が f_cyt になるよう校正する。k_on は独立パラメータでは
    なく、固定されるのは面積積分 ∫k_on dA = k_off·V·(1-f_cyt)/f_cyt のほう。

    block: 長さ n の配列。結合能をその係数で局所的に下げる（PopZ の占有を
           模す。1 - theta を渡す）。None なら摂動なし。
    """
    w = g["w"] if block is None else g["w"] * block
    return (w * par.k_off * (1 - par.f_cyt) * g["V"].sum()
            / (par.f_cyt * (w * g["A"]).sum()))


# =============================================================================
def solve(n=101, par=Par, T=20.0, ic="uniform", n_out=60, block=None):
    """時間発展を解く。

    ic:
        'uniform'   膜に一様に結合した状態から（既定）
        'cytoplasm' 全分子が細胞質（発現開始を模す）
        'one_pole'  片極だけに偏在した状態から（対称性の回復を見る）
        'bleach'    定常に達したあと片極の膜だけ漂白（FRAP と同じ操作）
    block:
        k_on_of() に渡す。片極を持続的に塞ぐ摂動（PopZ の占有）。

    戻り値の主なキー:
        enrichment  定常の極濃縮（極の膜密度 / 全体平均）
        enr_t       その時間変化
        f_cyt       定常の細胞質割合
        cyto_var    細胞質濃度の相対的なばらつき（1 - min/max）
        mass_err    質量保存の相対誤差。★これが大きければ結果を信用しない
    """
    g = geometry(n, par)
    L_c, L_m = operators(g, par)
    V, A = g["V"], g["A"]
    k_on = k_on_of(g, par, block)

    if ic == "cytoplasm":
        C0, M0 = np.full(n, par.N / V.sum()), np.zeros(n)
    elif ic == "uniform":
        M0 = np.full(n, (1 - par.f_cyt) * par.N / A.sum())
        C0 = np.full(n, par.f_cyt * par.N / V.sum())
    elif ic == "bleach":
        # まず定常まで解き、片極の膜だけをゼロにする（FRAP の漂白に対応）
        base = solve(n, par, T=5.0, ic="uniform", n_out=3, block=block)
        M0, C0 = base["M"].copy(), base["C"].copy()
        M0[g["z"] < par.R] = 0.0
    elif ic == "one_pole":
        M0 = np.zeros(n)
        M0[: max(2, n // 20)] = 1.0
        M0 *= (1 - par.f_cyt) * par.N / (M0 * A).sum()
        C0 = np.full(n, par.f_cyt * par.N / V.sum())
    else:
        raise ValueError(f"unknown ic: {ic!r}")

    def rhs(t, y):
        C, M = y[:n], y[n:]
        flux = k_on * C - par.k_off * M          # 分子/(µm²·s)
        return np.concatenate([L_c @ C - flux * A / V,   # ★ A/V は ここだけ [4]
                               L_m @ M + flux])

    # 立ち上がりが ~0.1 秒なので対数刻みで記録する
    t_eval = np.clip(np.unique(np.r_[0.0, np.logspace(-4, np.log10(T), n_out)]), 0.0, T)
    sol = solve_ivp(rhs, (0, T), np.concatenate([C0, M0]), method="BDF",
                    t_eval=t_eval, rtol=par.rtol, atol=par.atol)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    Ct, Mt = sol.y[:n, :], sol.y[n:, :]
    cap_frac = g["A_cap"].sum() / A.sum()
    tot_m = (Mt * A[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        enr_t = np.where(tot_m > 1e-12,
                         (Mt * g["A_cap"][:, None]).sum(0) / np.maximum(tot_m, 1e-300)
                         / cap_frac, np.nan)
    mass_t = (Ct * V[:, None]).sum(0) + (Mt * A[:, None]).sum(0)

    # 'bleach' は漂白で分子を取り除くので総量が N と一致しない。
    # 質量保存は「初期の総量」を基準に検算する。
    N0 = mass_t[0]
    return dict(g=g, par=par, t=sol.t, C=Ct[:, -1], M=Mt[:, -1], Ct=Ct, Mt=Mt,
                k_on=k_on, enrichment=enr_t[-1], enr_t=enr_t, N0=N0, ic=ic,
                f_cyt=(Ct[:, -1] * V).sum() / N0,
                mass_err=np.abs(mass_t - N0).max() / N0,
                cyto_var=1 - Ct[:, -1].min() / Ct[:, -1].max())


# ----------------------------------------------------------------- 便利関数
def enrichment(rho=None, n=101, par=Par, **kw):
    """定常の極濃縮だけを返すショートカット。rho スキャン用。"""
    p = par if rho is None else type("P", (par,), {"rho": rho})
    return solve(n, p, T=20.0, **kw)["enrichment"]


def analytic(par=Par, n=801):
    """膜拡散を無視した局所平衡の解 rho/(rho*a + 1 - a) と、極キャップ面積比 a。

    数値解はこれより必ず小さくなる（膜拡散がパターンをならすため）。
    """
    g = geometry(n, par)
    a = g["A_cap"].sum() / g["A"].sum()
    return par.rho / (par.rho * a + 1 - a), a


def asymmetry(theta, par=Par, n=201, T=20.0):
    """片極の結合能を (1-theta) 倍して持続的に押さえたときの左右比を返す。

    theta は PopZ が極膜を覆う割合に対応する。
    「良い極（無摂動側）/ 塞いだ極」を返すので 1 以上。
    """
    g = geometry(n, par)

    # ★ セル中心の判定ではなく「そのセルのうち左キャップが占める面積割合」で
    #    重みをつける（設計判断 [2] と同じ理由）。セル中心判定だと格子を変える
    #    たびブロック境界が飛び、収束が非単調になる:
    #      セル中心方式 n=201/401/801: 0.9353 -> 0.9397 -> 0.9356（非単調）
    #      面積重み方式 n=201/401/801: 0.9367 -> 0.9369 -> 0.9370（単調）
    fL = g["A_capL"] / g["A"]
    block = 1.0 - theta * fL

    r = solve(n, par, T=T, ic="cytoplasm", n_out=5, block=block)
    M = r["M"]
    return (M * g["A_capR"]).sum() / (M * g["A_capL"]).sum()


def pole_asymmetry_trace(par=Par, n=201, T=20.0, n_out=200):
    """片極に全分子を置いた状態からの左右比の時間変化 (t, asym) を返す。

    asym = (左 - 右)/(左 + 右)。1 から始まり 0 へ向かう。
    """
    r = solve(n, par, T=T, ic="one_pole", n_out=n_out)
    g = r["g"]
    left = g["z"] < par.R
    right = g["z"] > par.L - par.R
    Ml = (r["Mt"] * g["A_cap"][:, None])[left].sum(0)
    Mr = (r["Mt"] * g["A_cap"][:, None])[right].sum(0)
    return r["t"], (Ml - Mr) / (Ml + Mr)


def rise_time(r, frac=0.632):
    """定常値への偏差が (1-frac) まで減る時刻。増加・減少どちらの初期条件でも動く。

    frac=0.5 にすると FRAP の「半回復時間」に相当する。
    ic='cytoplasm' は t=0 で膜が空なので極濃縮が未定義。有限値だけ使う。
    """
    e, t = np.asarray(r["enr_t"], float), np.asarray(r["t"], float)
    ok = np.isfinite(e)
    e, t = e[ok], t[ok]
    if len(e) < 3:
        return np.nan
    d = np.abs(e - e[-1])
    if d[0] <= 0:
        return np.nan
    target = (1.0 - frac) * d[0]
    below = np.nonzero(d <= target)[0]
    if len(below) == 0:
        return np.nan
    k = below[0]
    if k == 0:
        return t[0]
    f = (d[k - 1] - target) / max(d[k - 1] - d[k], 1e-300)
    return t[k - 1] + f * (t[k] - t[k - 1])


def monomer_curvature_energy(par=Par, lp=100.0, length=30.0):
    """単量体が極/側壁の曲率差から得られるエネルギー [kT]。

    worm-like chain（Kratky-Porod）の標準形:
        dG = 1/2 · kT · lp · L · (dC)²
    dC = 1/R を使う（球面ではどの向きでも 1/R、円筒では軸に沿えば 0 のため）。
    lp, length は nm。par.R は µm なので nm に直す。

    ⚠️ lp = 100 nm はコイルドコイルのオーダー推定で、DivIVA の実測値ではない。
       10 倍外れても結論（2 桁のギャップ）は変わらない。
    """
    R_nm = par.R * 1000.0
    return 0.5 * lp * length / R_nm**2


# =============================================================================
def validate(par=Par, verbose=True):
    """本計算の前に必ず通すこと。False が返ったら結果を信用しない。"""
    out, ok = [], True

    def say(s):
        out.append(s)

    say("=" * 68)
    say("VALIDATION")
    say("=" * 68)

    g = geometry(401, par)
    Lcyl = par.L - 2 * par.R
    V_ex = np.pi * par.R**2 * Lcyl + 4 / 3 * np.pi * par.R**3
    A_ex = 2 * np.pi * par.R * Lcyl + 4 * np.pi * par.R**2
    for lab, got, exp in [("細胞体積", g["V"].sum(), V_ex),
                          ("膜面積", g["A"].sum(), A_ex),
                          ("極キャップ面積比", g["A_cap"].sum() / g["A"].sum(),
                           2 * par.R / par.L)]:   # a = 4πR²/A = 2R/L（幾何から）
        e = abs(got - exp) / exp
        ok &= e < 5e-3
        say(f"  V1 {lab:16} {got:9.4f}  (解析 {exp:.4f})  誤差 {e:.1e} "
            f"{'OK' if e < 5e-3 else '★NG'}")

    L_c, L_m = operators(g, par)
    for lab, Lp in [("細胞質", L_c), ("膜", L_m)]:
        m = np.abs(Lp @ np.ones(401)).max()
        ok &= m < 1e-9
        say(f"  V2 {lab}ラプラシアン @ 定数 = {m:.1e} {'OK' if m < 1e-9 else '★NG'}")

    r = solve(101, par, T=20.0)
    ok &= r["mass_err"] < 1e-10
    say(f"  V3 質量保存 誤差 = {r['mass_err']:.1e} "
        f"{'OK' if r['mass_err'] < 1e-10 else '★NG（A/V の使い方を疑え）'}")

    r1 = solve(101, type("P", (par,), {"rho": 1.0}), T=20.0)
    good = abs(r1["enrichment"] - 1.0) < 1e-4 and r1["cyto_var"] < 1e-6
    ok &= good
    say(f"  V4 rho=1: 極濃縮 = {r1['enrichment']:.6f}（期待 1.0）, "
        f"細胞質変動 = {r1['cyto_var']:.2e}（期待 0）{' OK' if good else ' ★NG'}")

    ana, _ = analytic(par)
    ok &= r["enrichment"] < ana
    say(f"  V5 極濃縮 {r['enrichment']:.4f} < 解析解 {ana:.4f} "
        f"{'OK（膜拡散でなまるため）' if r['enrichment'] < ana else '★NG'}")

    es = [solve(k, par, T=20.0)["enrichment"] for k in (51, 101, 201, 401)]
    d = max(abs(np.diff(es))) / es[-1]
    ok &= d < 0.01
    say("  V6 格子収束 n=51..401: " + " -> ".join(f"{e:.4f}" for e in es)
        + f"  最大変化 {d:.2%} {'OK' if d < 0.01 else '★NG'}")

    say("=" * 68)
    say("すべて合格" if ok else "★ 不合格あり。本計算に進まないこと")
    say("=" * 68)

    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)


# =============================================================================
# 将来の拡張（本ファイルでは意図的に外してある）
#
#   隔壁（セプタム）: geometry() の r(zf) にくびれを入れる。
#       r -= depth * R * np.exp(-((zf - L/2)**2) / (2 * width**2))
#       ★ 曲率 H も step ではなく輪郭から計算し直す必要がある。
#
#   PopZ による空間占有: 既に block 引数で入っている。
#       block = np.where(g["z"] < par.R, 1 - theta, 1.0) を solve() に渡す。
#       ⚠️ 「V から球の断面積を引く」のは誤り。C は濃度なので体積を減らすと
#          濃度が上がり、PopZ 極ほど結合が増えてしまう（符号が逆になる）。
#
#   クラスター状態: 3 番目の場 n(s) を足し、曲率を k_on ではなく
#       クラスターの安定性 k_off_cluster に入れる（単量体は曲率を感知できないため）。
#           dn/dt = (k_nuc + k_grow · n^p) · m - k_off_n(s) · n
#       ★ 合成・切断を入れると閉じた系でなくなるので、質量保存の検算を
#          収支検算に置き換えること（唯一の防波堤なので必ず先に）。
