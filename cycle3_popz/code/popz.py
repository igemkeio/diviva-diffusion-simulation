"""
DivIVA + PopZ モデル — Cycle 3（S1 空間占有 と S2 局所分解の比較）

    S1 空間占有 : PopZ が極の膜を物理的に占め、DivIVA の結合を妨げる
    S2 局所分解 : PopZ が極にプロテアーゼを呼び、そこで DivIVA を切る

    python popz.py              # 検証だけ走る（本計算の前に必ず通すこと）
    python popz.py --summary    # 主要な数値を再計算して印字

`cluster.py`（Cycle 2）の場・幾何・演算子・定常ソルバをそのまま使い、
PopZ の占有プロファイル u(z) を一つ足して、それを 2 通りに効かせる。

=============================================================================
  Cycle 3 の設計判断
=============================================================================

[P1] ★ 質量保存を「収支検算」に置き換えた。これを先にやってから分解を入れた。

     Cycle 2 の唯一の防波堤は質量保存（閉じた系）だった。分解は分子を系から
     取り除くので、総量は減るのが正しく、質量保存は常に落ちる。
     そこで先に収支の恒等式

         d/dt ∫(C·V + M·A + N·A) = mu·(N_tot - 総量) - ∫r_deg dA

     を検算として実装し、**分解をゼロにした状態で既存の質量保存と同じ桁
     （1e-13 級）が出ることを確認してから** 分解を有効にした。
     V_B1（瞬時・解析）と V_B2（累積・軌道）の 2 本で見ている。
     ⚠️ この順序を逆にしてはいけない。A/V の取り違えのような
       「結果からは気づけない間違い」を捕まえる手段が無くなる。

[P2] ★ mu > 0 を既定にした。S2 には選択の余地がない。

     分解があって合成が無いと、定常解は自明解（全部ゼロ）しかない。
     S2 の定常状態を論じるには合成項が必須である。Cycle 2 の [C3] に従い、
     合成は細胞質への一様項 mu·N/V で入れて希釈と厳密に釣り合わせる。
     このとき **分解がなければ定常の総量は厳密に N になる**（V_B3 で確認）。
     分解があると総量は N - D/mu まで下がり、この下がりぶんが
     判定基準 2「足場として使える量が残るか」の本体になる。

     既定 mu = ln2/1500 = 4.62e-4 /s（世代時間 1500 s）。
     ⚠️ Cycle 2 の公表値は mu = 0 のものである。
       当初ここに「差はクラスター極濃縮で 1% 未満」と書いていたが、
       **実測は 3.11% で誤りだった**（n=210/420 で mu=0: 2.5131 →
       mu=ln2/1500: 2.4350）。格子の揃え直し（[P12]、+0.52%）と合わせると、
       Cycle 2 の公表値 2.500 は Cycle 3 の既定で **2.4350**（-2.6%）になる。
       結論を変える大きさではないが、数値を引用するときは
       **どちらの設定か明記すること**。

[P3] ★ 定常方程式の冗長性は「閉じた系のときだけ」解消する。

     Cycle 2 の steady() は最後の 1 本を「総量 = N」の拘束で置き換えていた。
     これは閉じた系で Σ(V·dC + A·dM + A·dN) ≡ 0 だったからである。
     mu > 0 または分解があると、この和は mu·(N - 総量) - D になって
     恒等的にゼロではない。つまり **方程式はもう冗長でない** ので、
     拘束で置き換えてはいけない（置き換えると総量を誤って固定してしまう）。
     closed = (mu == 0 かつ 分解なし) のときだけ従来どおりにする。

[P4] ★ PopZ は静的な占有プロファイルとして入れる。その正当性は p で決まる。

     PopZ は先に誘導され、DivIVA が効く頃には定常に達しているという前提。
     ⚠️ Cycle 2 で DivIVA の局在確立に 1399 s（世代時間の 0.93 倍）かかると
       分かっているので、この前提は自明でない。
     本ファイルの答え: **p ≤ 1 なら定常は一意なので PopZ の立ち上がり方は
     定常に影響しない（過渡だけ）。p > 1 では経路が結果を決めるので
     立ち上がり時間と先行時間が本質的になる。**
     後者を latch() の 3 段階積分（先行 → PopZ on → 解放）で扱う。
     PopZ 自身の時定数の実測値は要らない形に問題を落としてある。

[P5] ★ 占有プロファイルは面積重みで入れる（Cycle 1 の設計判断 [8] を継承）。

     セル中心判定だと格子を変えるたび境界が飛び、収束が非単調になる。
     弧長は先端で ds/dz が発散するので **数値積分せず解析形で与える**:
         左極   s(z) = R·arccos((R - z)/R)
         円筒   s(z) = Rπ/2 + (z - R)
         右極   s(z) = Rπ/2 + L_cyl + R·arcsin((z - R - L_cyl)/R)

[P6] ★ 体積は減らさない。PopZ が場所を取るからといって V から引くのは誤り。
     C は濃度なので体積を減らすと濃度が上がり、PopZ のある極ほど結合が
     増える。符号が逆になる（`model.py` 末尾と同じ理由）。

[P7] ★ 占有で結合能は「再配分される」のか「失われる」のか、両方走らせる。

     block_mode = "redistribute" : ∫k_on dA を保つ（Cycle 1/2 の規約）。
                                   塞いだぶんが他へ回るので f_cyt は変わらない。
     block_mode = "lose"         : k_on を局所的に下げるだけ。∫k_on dA は減り、
                                   結合できない分子は細胞質に留まる。
     物理としては "lose" のほうが素直だが、Cycle 2 の必要被覆率
     35% / 65% / 81% は "redistribute" で出した数字なので、
     **両方報告して差を示す**（片方だけで数字を出さない）。

[P8] ★ S1 が何を塞ぐのかにも分岐がある。

     Cycle 2 の block は k_on（単量体の結合）だけを塞いでいた。
     しかし PopZ が膜を占めれば **クラスターが使える面積も奪われる** はずで、
     非対称を担っているのはクラスターのほうである。
     block_grow = True で成長項にも (1 - u) を掛ける。
     既定は False（Cycle 2 の規約。回帰テストのため）。両方報告する。

[P9] ★ S2 の分解は既定で単量体 M だけを切る。
     TEV は配列を認識するので、会合体の内側の分子にアクセスできるかは別問題。
     cut_target = "MN" でクラスターも切れる。両方報告する。
     ⚠️ 切られた分子は系から消える（C にも M にも戻らない）。

[P10] ★ プロテアーゼの処理能力に上限項は入れず、**必要処理量を出力する**。
     ClpXP の細胞全体の処理能力（数分子/s）は `PARAMETERS.md` §5.1 のとおり
     未確認なので、上限をモデルに固定で埋め込むと未確認の数字が結論に
     化けてしまう。代わりに各動作点で必要な切断流量 [分子/s] を報告し、
     あとで実測値と突き合わせられる形にした。

[P13] ★ 分解の空間重みは phi（PopZ の存在）で、占有の強さ theta ではない。

     当初の実装は切断速度を k_cut·u（u = theta·phi）としていた。これだと
     **theta = 0 で分解項が消えるので、「占有はせず分解だけする」純粋な S2 が
     原理的に作れない**。S1 と S2 を独立に振れないので、手順書 §6 の比較が
     そもそも成立しなかった。
     物理的にも、PopZ がプロテアーゼを呼ぶかどうかは PopZ が「居るか」
     （phi）で決まり、DivIVA の結合をどれだけ妨げるか（theta）とは別の性質である。
     ⚠️ 旧実装は非単調な副作用も生んでいた: theta=1 では塞いだ極に単量体が
       来ないので切る相手が居らず、theta=0.5 より総量が削れない
       （mass_frac 0.277 → 0.446）。占有と分解が交絡していた。

     修正後:  S1 単独 = (theta>0, k_cut=0)
              S2 単独 = (theta=0, k_cut>0)
              併用     = 両方 > 0

[P11] ★ 収支検算の分母は「足し合わせた項の絶対値和」にする。

     収支の左辺は結合・解離・拡散の大きな項（~5e4 µm³·分子/s）が打ち消し合って
     1e-5 級の正味になる。打ち消しは **4.4e15 倍** に達する。
     分母を出入りの流量（mu·N ~ 0.8）に取ると、閾値 1e-12 は
     「Σ|項| に対して 1.5e-17」＝機械精度より下を要求することになり、
     実装が正しくても必ず落ちる（実測 1.2e-11 ＝ eps·Σ|項| = 1.2e-11 の床）。
     ⚠️ ここで「落ちるから閾値を緩める」のは禁じ手なので、代わりに
       **検出力が保たれることを実験で確かめた**:
         正常                        2.3e-16（丸め誤差の床）
         A/V と V/A を取り違えた版   5.0e-05（= 11 桁の余裕で捕まる）
     防波堤としての機能は落ちていない。

[P12] ★ 格子収束は極キャップ境界がセル境界に乗る n で測る。

     ⚠️ Cycle 2 の格子収束の記述（クラスター極濃縮 2.4614→2.4780→2.5001→2.5043、
       「幅 0.88%」）は**物理ではなく報告指標の揺れだった**。
     境界 z=R, z=L-R がセル境界に乗らないと、その境界セルの面積を
     極と側壁へ按分する割合が n ごとに不規則に変わり、A_capL で積分した
     報告量（lr_clu, enr_clu）が非単調に揺れる。解の誤差ではない。

       theta=0.98, n=101/201/401/801 : 27.12→25.81→25.63→25.24（5.1%→0.7%→1.6%、
                                       401→801 で悪化＝非単調）
       theta=0.98, n=105/210/420/840 : 25.63→25.17→25.06→25.03（収束比 4.01/3.95）

     条件は n·R/L と n·(L-R)/L がともに整数。R=0.4, L=3.0 なら **n は 15 の倍数**。
     揃えると設計どおりの 2 次精度（収束比 4）で単調に収束する。
     PopZ なしの Cycle 2 定常は n=105〜840 で **enr_clu = 2.5131 が一定**になる。
     本ファイルの既定格子は n=210 とした。

     ⚠️ したがって Cycle 2 の公表値には按分由来の偏りがある（§summary で補正）:
       クラスター極濃縮 2.500(n=201) -> 2.5131   極割合 0.6608 -> 0.6701
       膜全体の極濃縮   1.415(n=201) -> 1.4260

=============================================================================
  判定に使う量（手順書 §6）
=============================================================================
  lr_clu       良い極 / 塞いだ極 のクラスター分子数の比（≥ 1 が目標）
  scaffold_R   良い極のクラスター分子数 [分子]。★判定基準 2 の本体
  scaffold_rel 同じものを PopZ なしの基準値で割った比
  mass_frac    細胞あたり総量 / N（分解でどれだけ削ったか）
  cut_flux     必要なプロテアーゼ処理量 [分子/s]（設計判断 [P10]）

  ★ 左右比だけで勝敗を決めないこと。目的は「DivIVA を消すこと」ではなく
    「片極に足場を作ること」である。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, bmat

import model
import cluster
from model import geometry, operators
from cluster import fractions, geometry2, k_off_n_of, _pow_p, derive, _kg


# =============================================================================
class Par(cluster.Par):
    """Cycle 3 の既定パラメータ。cluster.Par に PopZ の分だけ足す。"""

    # --- 合成と希釈（設計判断 [P2]）★ Cycle 2 と違って既定で入れる
    mu = np.log(2) / 1500.0     # 1/s  世代時間 1500 s

    # --- PopZ の幾何
    popz_mode = "cap"      # "cap" | "arc"
    #   "cap": 左極キャップ全体を一様に覆う（Cycle 2 の block と同一。回帰用）
    #   "arc": 左極先端からの弧長 r_popz 以内を覆う（凝集体の半径を使う）
    r_popz = 0.25          # µm  PopZ 凝集体の半径（150〜350 nm でスキャン）⚠️ 仮定
    theta = 0.0            # 覆った領域で結合能を落とす割合（0 = PopZ なし）

    # --- S1 空間占有の規約（設計判断 [P7][P8]）
    block_mode = "redistribute"   # "redistribute" | "lose"
    block_grow = False            # True なら成長項にも (1-u) を掛ける

    # --- S2 局所分解（設計判断 [P9]）
    k_cut = 0.0            # 1/s  占有率 1 のところでの切断速度 ⚠️ 実測値なし
    cut_target = "M"       # "M" | "MN"
    k_cut_n_rel = 1.0      # cut_target="MN" のときのクラスター側の相対速度

    # --- 数値
    T = 3000.0
    rtol = 1e-8


def variant(**kw):
    """モジュール既定の Par を基底にした派生。"""
    return derive(Par, **kw)


# =============================================================================
#  PopZ の占有プロファイル
# =============================================================================
def _fine(par, nfine):
    """細かい輪郭。model.geometry と同じ式で、弧長だけ解析形で足す（[P5]）。"""
    R, L = par.R, par.L
    Lcyl = L - 2 * R
    if Lcyl <= 0:
        raise ValueError(f"L ({L}) は 2R ({2*R}) より大きい必要がある")
    zf = np.linspace(0, L, nfine)
    dzf = np.gradient(zf)

    r = np.full_like(zf, R)
    a = zf < R
    r[a] = np.sqrt(np.maximum(R**2 - (zf[a] - R) ** 2, 0))
    b = zf > R + Lcyl
    r[b] = np.sqrt(np.maximum(R**2 - (zf[b] - R - Lcyl) ** 2, 0))

    dsdz = np.sqrt(1 + np.gradient(r, zf) ** 2)
    band = 2 * np.pi * r * dsdz

    # 左極の先端から測った弧長（先端で dsdz が発散するので積分しない）
    s = np.empty_like(zf)
    s[a] = R * np.arccos(np.clip((R - zf[a]) / R, -1.0, 1.0))
    mid = (~a) & (~b)
    s[mid] = R * np.pi / 2 + (zf[mid] - R)
    s[b] = (R * np.pi / 2 + Lcyl
            + R * np.arcsin(np.clip((zf[b] - R - Lcyl) / R, -1.0, 1.0)))
    return dict(zf=zf, dzf=dzf, r=r, dsdz=dsdz, band=band, s=s,
                is_cap=a | b, is_capL=a, Lcyl=Lcyl)


def popz_profile(n, par=Par, nfine=None):
    """PopZ の占有プロファイルを返す。

        phi : そのセルの膜面積のうち PopZ が覆う割合（面積重み、設計判断 [P5]）
        u   : theta * phi（結合能を落とす割合、分解速度の重み）

    戻り値にはプロファイルの面積（覆った膜面積）と、それが極キャップの
    どれだけを占めるかも入れる。★ theta と r_popz の二重性を見るために必要。
    """
    if nfine is None:
        nfine = max(20001, 40 * n)
    g = geometry2(n, par)
    A = g["A"]

    if par.popz_mode == "cap":
        phi = g["A_capL"] / A
        area = g["A_capL"].sum()
    elif par.popz_mode == "arc":
        fi = _fine(par, nfine)
        cov = (fi["s"] <= par.r_popz).astype(float)
        ze = np.linspace(0, par.L, n + 1)
        i = np.clip(np.searchsorted(ze, fi["zf"]) - 1, 0, n - 1)
        acc = np.bincount(i, fi["band"] * cov * fi["dzf"], minlength=n)
        phi = acc / A
        area = acc.sum()
    else:
        raise ValueError(f"popz_mode は 'cap' か 'arc': {par.popz_mode!r}")

    capL = g["A_capL"].sum()
    return dict(phi=phi, u=par.theta * phi, area=area,
                cap_cover=area / capL, capL=capL, g=g,
                # 結合能の重みで測った実効被覆率。Cycle 2 の theta と比べる量
                theta_eff=(par.theta * phi * g["A_capL"]).sum() / capL)


def popz_cap_cover_analytic(par=Par):
    """弧長 r_popz の球冠が極キャップのどれだけを覆うか（解析解）。

        覆う面積 = 2πR²(1 - cos(r_popz/R)),  極キャップ = 2πR²
        -> 被覆率 = 1 - cos(r_popz/R)

    ⚠️ r_popz > Rπ/2 では円筒部に回り込むのでこの式は使えない。
    """
    al = par.r_popz / par.R
    if al > np.pi / 2:
        return np.nan
    return 1.0 - np.cos(al)


# =============================================================================
#  組み立て
# =============================================================================
def _k_on_of(g, par, u):
    """PopZ 占有込みの k_on(z)（設計判断 [P7]）。"""
    f_cyt, f_mono, _ = fractions(par)
    b = 1.0 - u
    if par.block_mode == "redistribute":
        w = g["w"] * b
        return (w * par.k_off * f_mono * g["V"].sum()
                / (f_cyt * (w * g["A"]).sum()))
    if par.block_mode == "lose":
        w = g["w"]
        k_free = (w * par.k_off * f_mono * g["V"].sum()
                  / (f_cyt * (w * g["A"]).sum()))
        return k_free * b
    raise ValueError(f"block_mode は 'redistribute' か 'lose': {par.block_mode!r}")


def _assemble(n, par):
    """幾何・演算子・速度定数・RHS・ヤコビアン・収支をまとめて作る。"""
    f_cyt, f_mono, f_clu = fractions(par)
    pz = popz_profile(n, par)
    g = pz["g"]
    u = pz["u"]
    L_c, L_m = operators(g, par)
    V, A = g["V"], g["A"]
    k_off_n = k_off_n_of(g, par)
    k_grow = _kg(par)              # ★ PopZ なしの基準状態で校正された値
    p, eps, f_nm, mu = par.p, par.eps_p, par.f_nm, par.mu

    k_on = _k_on_of(g, par, u)
    src = mu * par.N / V.sum()     # 細胞質への一様合成（設計判断 [P2]）

    bg = (1.0 - u) if par.block_grow else np.ones(n)   # 設計判断 [P8]

    # ★ 分解の空間重みは phi（PopZ が居るか）であって u = theta*phi ではない
    #   （設計判断 [P13]）。u を使うと theta=0 で分解が消え、純粋な S2 が作れない。
    phi_c = pz["phi"]
    if par.cut_target == "M":
        kcM, kcN = par.k_cut * phi_c, np.zeros(n)
    elif par.cut_target == "MN":
        kcM = par.k_cut * phi_c
        kcN = par.k_cut * par.k_cut_n_rel * phi_c
    else:
        raise ValueError(f"cut_target は 'M' か 'MN': {par.cut_target!r}")

    def rhs(y):
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        flux = k_on * C - par.k_off * M
        G = (par.k_nuc + k_grow * _pow_p(Nf, p, eps)) * M * bg
        rel = k_off_n * Nf
        dC = L_c @ C + (-flux + (1 - f_nm) * rel) * A / V - mu * C + src
        dM = L_m @ M + flux - G + f_nm * rel - mu * M - kcM * M
        dN = G - rel - mu * Nf - kcN * Nf
        return np.concatenate([dC, dM, dN])

    absLc, absLm = abs(L_c), abs(L_m)

    def rhs_mag(y):
        """各式の項の大きさ（無次元化の尺度。設計判断 [C9]）。"""
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        AV = A / V
        b = k_on * C + par.k_off * M
        G = (par.k_nuc + k_grow * _pow_p(Nf, p, eps)) * M * bg
        rel = k_off_n * Nf
        mC = absLc @ C + (b + (1 - f_nm) * rel) * AV + mu * C + src
        mM = absLm @ M + b + G + f_nm * rel + mu * M + kcM * M
        mN = G + rel + mu * Nf + kcN * Nf
        return np.concatenate([mC, mM, mN])

    def jac(y):
        M, Nf = y[n:2 * n], y[2 * n:]
        AV = A / V
        phi = _pow_p(Nf, p, eps)
        dphi = p * Nf * (Nf * Nf + eps * eps) ** (0.5 * p - 1.0)
        kg_phi = (par.k_nuc + k_grow * phi) * bg
        dG_dN = k_grow * dphi * M * bg

        dCC = L_c - diags(k_on * AV + mu)
        dCM = diags(par.k_off * AV)
        dCN = diags((1 - f_nm) * k_off_n * AV)
        dMC = diags(k_on)
        dMM = L_m - diags(par.k_off + kg_phi + mu + kcM)
        dMN = diags(f_nm * k_off_n - dG_dN)
        dNM = diags(kg_phi)
        dNN = diags(dG_dN - k_off_n - mu - kcN)
        return bmat([[dCC, dCM, dCN],
                     [dMC, dMM, dMN],
                     [None, dNM, dNN]], format="csr")

    def deg_flux(y):
        """∫r_deg dA [分子/s]。切断された分子が系から出る速度（設計判断 [P10]）。"""
        M, Nf = y[n:2 * n], y[2 * n:]
        return float(((kcM * M + kcN * Nf) * A).sum())

    def total(y):
        return float((y[:n] * V).sum() + (y[n:2 * n] * A).sum()
                     + (y[2 * n:] * A).sum())

    def budget(y):
        """収支検算（設計判断 [P1]）。

            d/dt ∫(C·V + M·A + N·A) = mu·(N - 総量) - ∫r_deg dA

        左辺を RHS から、右辺を独立に作って比べる。相対誤差を返す。

        ★ 尺度は「足し合わせた項の絶対値和」にする（設計判断 [P11]）。
          左辺は結合・解離・拡散の大きな項（~1e4 µm³·分子/s）が打ち消し合って
          1e-5 級の正味になる。打ち消しは 1e15 倍に達するので、出入りの流量
          （mu·N ~ 0.8）を分母にすると機械精度より下を要求してしまい、
          実装が正しくても必ず落ちる。
          分母を Σ|項| にすると、正常時 2e-16（＝丸め誤差の床）に対し
          A/V と V/A を取り違えたバグは 5e-5 を返す。
          **閾値 1e-12 のまま防波堤としての検出力は保たれている。**
        """
        d = rhs(y)
        contrib = np.concatenate([d[:n] * V, d[n:2 * n] * A, d[2 * n:] * A])
        lhs = float(contrib.sum())
        rhs_ = mu * (par.N - total(y)) - deg_flux(y)
        sc = float(np.abs(contrib).sum()) + abs(mu * par.N) + abs(deg_flux(y))
        return abs(lhs - rhs_) / max(sc, 1e-300), lhs, rhs_

    closed = (mu == 0.0) and (par.k_cut == 0.0)

    return dict(g=g, V=V, A=A, u=u, pz=pz, rhs=rhs, jac=jac, rhs_mag=rhs_mag,
                deg_flux=deg_flux, budget=budget, total=total, closed=closed,
                k_on=k_on, k_off_n=k_off_n, k_grow=k_grow, bg=bg,
                kcM=kcM, kcN=kcN, fr=(f_cyt, f_mono, f_clu))


def initial(n, par=Par, ic="uniform", seed=10.0):
    """初期条件。cluster.initial と同じだが PopZ 込みの定常を使える。"""
    if ic == "steady_free":
        # PopZ なしの定常（DivIVA が先に立ち上がっている状態）
        r = steady(n, variant_from(par, theta=0.0, k_cut=0.0))
        return np.concatenate([r["C"], r["M"], r["Nf"]])
    return cluster.initial(n, par, ic, None, seed=seed)


def variant_from(par, **kw):
    """par を基底に一部上書き（cluster.derive の薄いラッパ）。"""
    return derive(par, **kw)


# =============================================================================
#  報告
# =============================================================================
def _report(par, asm, ys, ts, ic):
    """判定に使う量をまとめる（手順書 §6）。"""
    g, V, A = asm["g"], asm["V"], asm["A"]
    n = g["n"]
    Ct, Mt, Nt = ys[:n, :], ys[n:2 * n, :], ys[2 * n:, :]
    C, M, Nf = Ct[:, -1], Mt[:, -1], Nt[:, -1]
    cap_frac = g["A_cap"].sum() / A.sum()

    memb_t = Mt + Nt
    tot_m = (memb_t * A[:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        enr_t = np.where(tot_m > 1e-12,
                         (memb_t * g["A_cap"][:, None]).sum(0)
                         / np.maximum(tot_m, 1e-300) / cap_frac, np.nan)
        tot_n = (Nt * A[:, None]).sum(0)
        enr_clu_t = np.where(tot_n > 1e-12,
                             (Nt * g["A_cap"][:, None]).sum(0)
                             / np.maximum(tot_n, 1e-300) / cap_frac, np.nan)

    mass_t = ((Ct * V[:, None]).sum(0) + (Mt * A[:, None]).sum(0)
              + (Nt * A[:, None]).sum(0))

    # 左 = PopZ を置いた極、右 = 無摂動の極
    nL = float((Nf * g["A_capL"]).sum())
    nR = float((Nf * g["A_capR"]).sum())
    mL = float((M * g["A_capL"]).sum())
    mR = float((M * g["A_capR"]).sum())
    nLt = (Nt * g["A_capL"][:, None]).sum(0)
    nRt = (Nt * g["A_capR"][:, None]).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lr_t = nRt / np.maximum(nLt, 1e-300)

    bud, d_lhs, d_rhs = asm["budget"](ys[:, -1])
    n_int = (Nf * A).sum()
    N_tot = mass_t[-1]

    return dict(
        g=g, par=par, t=ts, C=C, M=M, Nf=Nf, Ct=Ct, Mt=Mt, Nt=Nt, ic=ic,
        u=asm["u"], pz=asm["pz"], k_on=asm["k_on"], k_off_n=asm["k_off_n"],
        k_grow=asm["k_grow"],
        # --- 判定量
        lr_clu=nR / max(nL, 1e-300), lr_t=lr_t,
        lr_mono=mR / max(mL, 1e-300),
        scaffold_R=nR, scaffold_L=nL, mono_R=mR, mono_L=mL,
        mass_frac=N_tot / par.N,
        cut_flux=asm["deg_flux"](ys[:, -1]),
        # --- Cycle 2 と比べる量
        enrichment=enr_t[-1], enr_t=enr_t,
        enr_clu=enr_clu_t[-1], enr_clu_t=enr_clu_t,
        clu_pole_frac=(Nf * g["A_cap"]).sum() / max(n_int, 1e-300),
        f_cyt=(C * V).sum() / N_tot, f_mono=(M * A).sum() / N_tot,
        f_clu=n_int / N_tot,
        asym_clu=nL / max(nR, 1e-300),   # cluster.py と同じ向き（左/右）
        # --- 検算
        mass_t=mass_t, N_tot=N_tot,
        mass_err=np.abs(mass_t - mass_t[0]).max() / mass_t[0],
        budget_err=bud, budget_lhs=d_lhs, budget_rhs=d_rhs,
        residual=float(np.max(np.abs(asm["rhs"](ys[:, -1]))
                              / np.maximum(asm["rhs_mag"](ys[:, -1]), 1e-300))),
    )


# =============================================================================
#  ソルバ
# =============================================================================
def solve(n=101, par=Par, T=None, ic="uniform", n_out=80, seed=10.0, y0=None):
    """時間発展を解く。"""
    if T is None:
        T = par.T
    asm = _assemble(n, par)
    y0 = (initial(n, par, ic, seed=seed) if y0 is None
          else np.asarray(y0, float))

    atol_vec = par.atol_rel * np.concatenate([
        np.full(n, par.N / asm["V"].sum()),
        np.full(n, par.N / asm["A"].sum()),
        np.full(n, par.N / asm["A"].sum())])

    t_eval = np.clip(np.unique(np.r_[0.0, np.logspace(-4, np.log10(T), n_out)]),
                     0.0, T)
    sol = solve_ivp(lambda t, y: asm["rhs"](y), (0, T), y0, method="BDF",
                    t_eval=t_eval, rtol=par.rtol, atol=atol_vec,
                    jac=lambda t, y: asm["jac"](y))
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return _report(par, asm, sol.y, sol.t, ic)


def steady(n=101, par=Par, ic="uniform", y0=None, tol=1e-7, maxit=40,
           seed_T=None, seed=10.0):
    """定常状態を直接解く（Cycle 2 の [C8][C9] を継承、冗長性の扱いは [P3]）。"""
    from scipy.sparse.linalg import spsolve

    asm = _assemble(n, par)
    V, A = asm["V"], asm["A"]
    rhs, jac, rhs_mag = asm["rhs"], asm["jac"], asm["rhs_mag"]
    closed = asm["closed"]

    if par.mu == 0.0 and not closed:
        raise ValueError(
            "mu = 0 で分解を入れると定常解は自明解（全部ゼロ）しかない。"
            "S2 を定常で論じるには合成項（mu > 0）が必要（設計判断 [P2]）")

    floor = 1e-14 * par.N / A.sum()
    scale = max(par.k_off_n, 1e-12) * par.N / A.sum()
    mass_row = np.concatenate([V, A, A]) / par.N * scale

    def F(v):
        r = rhs(v)
        if closed:
            mass = (v[:n] * V).sum() + (v[n:2 * n] * A).sum() + (v[2 * n:] * A).sum()
            r[-1] = (mass / par.N - 1.0) * scale
        return r

    def J(v):
        Jm = jac(v)
        if not closed:
            return Jm
        Jl = Jm.tolil()
        Jl[Jm.shape[0] - 1, :] = mass_row
        return Jl.tocsr()

    def relres(v):
        return float(np.max(np.abs(rhs(v)) / np.maximum(rhs_mag(v), 1e-300)))

    def newton(y_start):
        y = np.maximum(np.asarray(y_start, float), floor)
        best_y, best, stall = y.copy(), relres(y), 0
        for _ in range(maxit):
            if best < tol:
                break
            dy = spsolve(J(y), -F(y))
            if not np.all(np.isfinite(dy)):
                break
            neg = dy < 0
            a = 1.0
            if np.any(neg):
                a = min(1.0, 0.9 * np.min(y[neg] / -dy[neg]))
            y = np.maximum(y + a * dy, floor)
            r = relres(y)
            if r < best:
                best, best_y, stall = r, y.copy(), 0
            else:
                stall += 1
                if stall >= 3:
                    break
        return best_y, best

    y, res = None, np.inf
    starts = []
    if y0 is not None:
        starts.append(lambda: np.asarray(y0, float))
    T0 = seed_T if seed_T is not None else par.T
    for k in range(3):
        Tk = T0 * 10.0**k
        starts.append(lambda Tk=Tk: np.concatenate(
            [(sd := solve(n, par, T=Tk, ic=ic, n_out=3, seed=seed))["C"],
             sd["M"], sd["Nf"]]))
    for mk in starts:
        yk, rk = newton(mk())
        if rk < res:
            y, res = yk, rk
        if res < tol:
            break

    if res > 1e-5:
        raise RuntimeError(f"steady() 収束せず: 無次元残差 {res:.3e}（許容 1e-5）")
    if np.any(y <= 0):
        raise RuntimeError("steady() が非正の濃度を返した")

    out = _report(par, asm, y[:, None], np.array([np.inf]), ic)
    out["steady_residual"] = res
    return out


# =============================================================================
#  収支検算（設計判断 [P1]）
# =============================================================================
def budget_trajectory(n=51, par=Par, T=None, ic="uniform"):
    """累積の収支検算。分解量と合成量を状態に足して軌道上で閉じるかを見る。

        y_ext = [C, M, N, D, S],   dD/dt = ∫r_deg dA,  dS/dt = mu·(N - 総量)

    すると **総量(t) + D(t) - S(t) = 総量(0)** が厳密に成り立つはずである。
    これが §1.1 の収支検算の軌道版。瞬時版（_assemble の budget）が
    代数的な取り違えを捕まえ、こちらが積分の取り違えを捕まえる。

    戻り値: (相対誤差, 最終の総量/N, 累積分解量 D, 累積合成量 S)
    """
    if T is None:
        T = par.T
    asm = _assemble(n, par)
    V, A = asm["V"], asm["A"]
    y0 = np.r_[initial(n, par, ic), 0.0, 0.0]

    def f(t, ye):
        y = ye[:3 * n]
        d = asm["rhs"](y)
        return np.r_[d, asm["deg_flux"](y), par.mu * (par.N - asm["total"](y))]

    atol_vec = par.atol_rel * np.concatenate([
        np.full(n, par.N / V.sum()), np.full(n, par.N / A.sum()),
        np.full(n, par.N / A.sum()), [par.N, par.N]])
    sol = solve_ivp(f, (0, T), y0, method="BDF", rtol=par.rtol, atol=atol_vec,
                    t_eval=np.linspace(0, T, 40))
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    tot = np.array([asm["total"](sol.y[:3 * n, k]) for k in range(sol.y.shape[1])])
    D, S = sol.y[3 * n], sol.y[3 * n + 1]
    resid = tot + D - S - tot[0]
    scale = max(tot[0], D.max(), abs(S).max())
    return float(np.abs(resid).max() / scale), tot[-1] / par.N, D[-1], S[-1]


# =============================================================================
#  S1 / S2 の実験
# =============================================================================
def sustained(par=Par, n=210, y0=None):
    """PopZ を出し続けたときの定常。判定に使う量を dict で返す。"""
    return steady(n, par, y0=y0)


def baseline(par=Par, n=210):
    """PopZ なしの基準状態（theta=0, k_cut=0）。"""
    return steady(n, derive(par, theta=0.0, k_cut=0.0))


def latch(par=Par, n=210, t_pre=0.0, t_on=600.0, T_after=1e5, ic="cytoplasm"):
    """3 段階積分（設計判断 [C10][P4]）。

        段 0: PopZ なしで t_pre だけ走らせる（DivIVA の先行時間）
        段 1: PopZ を on にして t_on だけ走らせる
        段 2: PopZ を外して T_after だけ走らせる

    t_pre = 0 なら「PopZ が先、DivIVA が後」（本プロジェクトの基本戦略）。
    t_pre > 0 なら DivIVA が先に局在を作ってしまった場合。

    ⚠️ RHS の中で時刻を見て切り替えてはいけない（不連続で BDF が止まる）。
    各段を別の積分として解き、終状態を次の初期値にする。

    戻り値: (段 1 の終わりの左右比, 最終の左右比, 最終の report)
    """
    free = derive(par, theta=0.0, k_cut=0.0)
    y = initial(n, par, ic)
    if t_pre > 0:
        r0 = solve(n, free, T=t_pre, ic=ic, n_out=4)
        y = np.concatenate([r0["C"], r0["M"], r0["Nf"]])
    r1 = solve(n, par, T=t_on, n_out=4, y0=y)
    y1 = np.concatenate([r1["C"], r1["M"], r1["Nf"]])
    r2 = solve(n, free, T=T_after, n_out=40, y0=y1)
    return r1["lr_clu"], r2["lr_clu"], r2


def theta_for_ratio(target, par=Par, n=210, lo=0.0, hi=0.995):
    """目標の左右比に必要な被覆率 theta を返す（S1）。届かなければ nan。"""
    from scipy.optimize import brentq

    warm = {"y": None}

    def f(th):
        r = steady(n, derive(par, theta=th), y0=warm["y"])
        warm["y"] = np.concatenate([r["C"], r["M"], r["Nf"]])
        return r["lr_clu"] - target

    if f(hi) < 0:
        return np.nan
    if f(lo) > 0:
        return lo
    return brentq(f, lo, hi, rtol=1e-5)


def kcut_for_ratio(target, par=Par, n=210, lo=0.0, hi=1e3):
    """目標の左右比に必要な切断速度 k_cut を返す（S2）。届かなければ nan。"""
    from scipy.optimize import brentq

    warm = {"y": None}

    def f(kc):
        r = steady(n, derive(par, k_cut=kc), y0=warm["y"])
        warm["y"] = np.concatenate([r["C"], r["M"], r["Nf"]])
        return r["lr_clu"] - target

    v = max(lo, 1e-6)
    if f(v) > 0:
        return v
    while v < hi:
        if f(v) > 0:
            return brentq(f, v / 10, v, rtol=1e-4)
        v *= 10.0
    return np.nan


# =============================================================================
def validate(par=Par, verbose=True):
    """本計算の前に必ず通すこと。Cycle 2 の 13 項目は cluster.validate で通す。"""
    out, ok = [], True

    def say(s):
        out.append(s)

    def chk(cond, label, detail):
        nonlocal ok
        ok &= bool(cond)
        say(f"  {label} {detail} {'OK' if cond else 'NG'}")

    say("=" * 78)
    say("VALIDATION - Cycle 3 (PopZ)")
    say("=" * 78)

    # --- V0: Cycle 2 の検証を先に通す
    say("\n  -- V0 cycle 2 validation first (13 items) --")
    chk(cluster.validate(cluster.Par, verbose=False), "V0", "cluster.validate()")

    # --- VP0: 細かい輪郭が model.geometry と一致するか（弧長の下地）
    n = 201
    g = geometry2(n, par)
    # ⚠️ nfine は model.geometry と同じ規則で作ること。別の値を渡すと
    #    先端セル（dsdz が発散する）の離散化が変わり、1% 級で食い違う。
    fi = _fine(par, max(20001, 40 * n))
    ze = np.linspace(0, par.L, n + 1)
    i = np.clip(np.searchsorted(ze, fi["zf"]) - 1, 0, n - 1)
    A_mine = np.bincount(i, fi["band"] * fi["dzf"], minlength=n)
    e = np.abs(A_mine - g["A"]).max() / g["A"].max()
    chk(e < 1e-12, "VP0", f"fine contour A matches model.geometry: rel {e:.1e}")

    # --- VP1: 弧長の端点と、arc プロファイルが cap プロファイルに戻るか
    s_tot = fi["s"][-1]
    s_ex = par.R * np.pi + (par.L - 2 * par.R)
    # ⚠️ 閾値は 1e-12 にできない。右端で arcsin の引数が 1-2.2e-16 になり、
    #    arcsin は 1 で微分が発散するので、丸め誤差が sqrt(eps) ~ 1e-8 級に
    #    増幅される。これは実装の誤りではなく解析形の原理的な限界である
    #    （左極の arccos も同じ。PopZ は左極なので実害はない）。
    chk(abs(s_tot - s_ex) / s_ex < 1e-7, "VP1",
        f"total contour arc length {s_tot:.9f} vs analytic {s_ex:.9f} "
        f"(rel {abs(s_tot-s_ex)/s_ex:.1e}, sqrt(eps) 限界)")
    pa = popz_profile(n, derive(par, popz_mode="arc",
                                r_popz=par.R * np.pi / 2, theta=1.0))
    pc = popz_profile(n, derive(par, popz_mode="cap", theta=1.0))
    e = np.abs(pa["phi"] - pc["phi"]).max()
    chk(e < 1e-9, "VP1",
        f"arc(r=R*pi/2) profile equals cap profile: max diff {e:.1e}")

    # --- VP2: 覆った面積が球冠の解析解と一致するか
    for rp in (0.15, 0.25, 0.35):
        pz = popz_profile(401, derive(par, popz_mode="arc", r_popz=rp, theta=1.0))
        ana = popz_cap_cover_analytic(derive(par, r_popz=rp))
        e = abs(pz["cap_cover"] - ana) / ana
        chk(e < 2e-3, "VP2",
            f"r_popz={rp*1000:.0f}nm covers {pz['cap_cover']:.4f} of pole cap "
            f"(analytic {ana:.4f}, rel {e:.1e})")

    # --- VB1: ★ 収支検算（瞬時・解析）。分解ゼロで既存の質量保存と同じ桁か
    say("\n  -- VB 収支検算（設計判断 [P1]）--")
    p_closed = derive(par, mu=0.0, theta=0.0, k_cut=0.0)
    asm = _assemble(101, p_closed)
    y = initial(101, p_closed, "uniform") * (1 + 0.05 * np.sin(np.arange(303)))
    # ★ VB1a〜d はすべて同じ尺度（Σ|項|）で判定する。
    #   以前は VB1a だけ rhs_mag ベースの別尺度で、甘すぎた（2.7e-19 が出た）。
    e, lhs0, _ = asm["budget"](y)
    chk(e < 1e-12, "VB1a",
        f"closed system (mu=0, no cut): d(total)/dt = {lhs0:.3e}, rel {e:.1e}")

    p_mu = derive(par, theta=0.0, k_cut=0.0)
    asm = _assemble(101, p_mu)
    y = initial(101, p_mu, "uniform") * (1 + 0.05 * np.sin(np.arange(303)))
    e = asm["budget"](y)[0]
    chk(e < 1e-12, "VB1b", f"mu>0, no cut: budget identity rel err {e:.1e}")

    p_cut = derive(par, popz_mode="cap", theta=0.8, k_cut=0.01)
    asm = _assemble(101, p_cut)
    y = initial(101, p_cut, "uniform") * (1 + 0.05 * np.sin(np.arange(303)))
    e = asm["budget"](y)[0]
    chk(e < 1e-12, "VB1c", f"mu>0 + cut (M): budget identity rel err {e:.1e}")

    p_cut2 = derive(p_cut, cut_target="MN")
    asm = _assemble(101, p_cut2)
    e = asm["budget"](y)[0]
    chk(e < 1e-12, "VB1d", f"mu>0 + cut (M+N): budget identity rel err {e:.1e}")

    # --- VB2: 収支検算（軌道・累積）
    e0 = budget_trajectory(51, derive(par, mu=0.0, theta=0.0, k_cut=0.0), T=3000.0)
    chk(e0[0] < 1e-9, "VB2a",
        f"trajectory budget, closed: rel {e0[0]:.1e} (total/N {e0[1]:.6f})")
    e1 = budget_trajectory(51, p_mu, T=3000.0)
    chk(e1[0] < 1e-9, "VB2b",
        f"trajectory budget, mu>0: rel {e1[0]:.1e} (total/N {e1[1]:.6f})")
    e2 = budget_trajectory(51, p_cut, T=3000.0)
    chk(e2[0] < 1e-9, "VB2c",
        f"trajectory budget, cut on: rel {e2[0]:.1e} "
        f"(total/N {e2[1]:.4f}, cut {e2[2]:.1f} molecules)")
    chk(e2[2] > 0.0, "VB2d",
        f"cut actually removes molecules: D = {e2[2]:.2f} (total/N {e2[1]:.4f})")

    # --- VB3: mu > 0 かつ分解なしなら定常の総量は厳密に N（設計判断 [P2]）
    rs = steady(101, p_mu)
    e = abs(rs["mass_frac"] - 1.0)
    chk(e < 1e-8, "VB3",
        f"mu>0, no cut: steady total/N = {rs['mass_frac']:.10f} (dev {e:.1e})")

    # --- VR1: ★ PopZ ゼロで Cycle 2 に戻るか（最重要の回帰テスト）
    say("\n  -- VR cycle 2 への回帰 --")
    p0 = derive(par, mu=0.0, theta=0.0, k_cut=0.0)
    r4 = steady(101, p0)
    r2 = cluster.steady(101, cluster.Par)
    for key, lab in (("enr_clu", "cluster enrichment"),
                     ("clu_pole_frac", "pole fraction"),
                     ("enrichment", "membrane enrichment"),
                     ("f_clu", "f_clu")):
        rel = abs(r4[key] - r2[key]) / abs(r2[key])
        chk(rel < 1e-8, "VR1", f"{lab}: {r4[key]:.8f} vs cycle2 {r2[key]:.8f} "
                               f"(rel {rel:.1e})")

    # --- VR2: theta=0 なら block_mode / block_grow に依らない
    a = steady(101, derive(par, block_mode="lose"))["enr_clu"]
    b = steady(101, derive(par, block_mode="redistribute"))["enr_clu"]
    c = steady(101, derive(par, block_grow=True))["enr_clu"]
    rel = max(abs(a - b), abs(a - c)) / b
    chk(rel < 1e-10, "VR2",
        f"theta=0: conventions agree (rel {rel:.1e})")

    # --- VR3: theta>0 を Cycle 2 の block と突き合わせる（規約 redistribute）
    th = 0.9
    g2 = geometry2(201, cluster.Par)
    blk = 1.0 - th * (g2["A_capL"] / g2["A"])
    r2b = cluster.steady(201, cluster.Par, block=blk)
    r4b = steady(201, derive(par, mu=0.0, popz_mode="cap", theta=th))
    rel = abs(r4b["asym_clu"] - r2b["asym_clu"]) / r2b["asym_clu"]
    chk(rel < 1e-6, "VR3",
        f"theta={th} matches cycle2 block: {1/r4b['asym_clu']:.4f}x vs "
        f"{1/r2b['asym_clu']:.4f}x (rel {rel:.1e})")

    # --- VJ: 解析ヤコビアンと差分（分解項・成長阻害込み）
    say("\n  -- VJ / VG 数値 --")
    nj = 21
    for lab, pp in (("cut M", p_cut), ("cut M+N, block_grow",
                                       derive(p_cut, cut_target="MN",
                                              block_grow=True,
                                              block_mode="lose"))):
        asm = _assemble(nj, pp)
        yj = initial(nj, pp, "uniform") * (1 + 0.05 * np.sin(np.arange(3 * nj)))
        Ja = asm["jac"](yj).toarray()
        f0 = asm["rhs"](yj)
        Jn = np.zeros_like(Ja)
        for j in range(3 * nj):
            hj = 1e-6 * max(abs(yj[j]), 1e-6)
            yp = yj.copy()
            yp[j] += hj
            Jn[:, j] = (asm["rhs"](yp) - f0) / hj
        rel = np.abs(Ja - Jn).max() / np.abs(Jn).max()
        chk(rel < 1e-6, "VJ", f"analytic jac vs FD ({lab}): rel {rel:.1e}")

    # --- VS: steady() と長時間積分の一致（分解あり）
    rs = steady(101, p_cut)
    rt = solve(101, p_cut, T=1e5, n_out=5)
    rel = abs(rt["lr_clu"] - rs["lr_clu"]) / rs["lr_clu"]
    chk(rel < 1e-3, "VS",
        f"steady {rs['lr_clu']:.4f}x vs integration T=1e5 {rt['lr_clu']:.4f}x "
        f"(rel {rel:.1e}, residual {rt['residual']:.1e})")

    # --- VG: 格子収束（設計判断 [P12]）
    #
    # ★ 極キャップの境界 z=R と z=L-R が**セル境界に乗る n** を使う。
    #   乗らない n だと、境界セルの面積を極と側壁へ按分する割合が n ごとに
    #   不規則に変わり、A_capL で積分した報告量（lr_clu, enr_clu）が
    #   非単調に揺れる。これは解の誤差ではなく**報告指標の誤差**である。
    #
    #   実測: theta=0.98 で n=101/201/401/801 を使うと 27.12→25.81→25.63→25.24
    #   と変化が 5.08%→0.71%→1.57% で非単調（401→801 で悪化する）。
    #   境界を乗せた n=105/210/420/840 では 25.63→25.17→25.06→25.03 となり、
    #   収束比 4.01 / 3.95（2次精度の理論値 4）で単調に収束する。
    #
    #   条件: n·R/L と n·(L-R)/L がともに整数。R=0.4, L=3.0 なら n は 15 の倍数。
    ngs = (105, 210, 420)
    for k in ngs:
        al = max(abs(k * par.R / par.L - round(k * par.R / par.L)),
                 abs(k * (par.L - par.R) / par.L
                     - round(k * (par.L - par.R) / par.L)))
        chk(al < 1e-12, "VG0", f"n={k} puts pole boundary on a cell edge")

    for th in (0.8, 0.98):
        pp = derive(par, popz_mode="cap", theta=th)
        es = [steady(k, pp)["lr_clu"] for k in ngs]
        d = np.diff(es)
        rate = abs(d[0] / d[1]) if abs(d[1]) > 0 else np.inf
        rel = abs(d[-1]) / es[-1]
        # 単調であること・2次精度であること・最後の変化が小さいこと
        chk(d[0] * d[1] > 0, "VGa",
            f"theta={th} monotone: " + " -> ".join(f"{e:.4f}" for e in es))
        chk(2.5 < rate < 6.0, "VGb",
            f"theta={th} convergence rate {rate:.2f} (2nd order = 4)")
        chk(rel < 0.01, "VGc",
            f"theta={th} last change {rel:.2%}, "
            f"Richardson {es[-1] + d[-1]/3:.4f}")

    # --- VT: 2 段階積分が十分長い持続摂動と一致するか（設計判断 [C10]）
    pp = derive(par, popz_mode="cap", theta=0.9)
    held = solve(101, pp, T=3000.0, ic="uniform", n_out=4)
    r1, _, _ = latch(pp, n=101, t_on=3000.0, T_after=1.0, ic="uniform")
    rel = abs(r1 - held["lr_clu"]) / held["lr_clu"]
    chk(rel < 1e-4, "VT",
        f"staged integration matches sustained at t_release: "
        f"{r1:.4f} vs {held['lr_clu']:.4f} (rel {rel:.1e})")

    # --- VP3: p<=1 では定常が経路に依らない（設計判断 [P4] の根拠）
    pp = derive(par, popz_mode="cap", theta=0.9, p=0.5)
    a = steady(101, pp, ic="uniform")["lr_clu"]
    b = steady(101, pp, ic="cytoplasm")["lr_clu"]
    c = steady(101, pp, ic="seed_pole", seed=100.0)["lr_clu"]
    rel = max(abs(a - b), abs(a - c)) / a
    chk(rel < 1e-5, "VP3",
        f"p=0.5: steady independent of path ({a:.4f}/{b:.4f}/{c:.4f}, "
        f"rel {rel:.1e})")

    say("=" * 78)
    say("all passed" if ok else "FAILURES PRESENT - do not proceed")
    say("=" * 78)

    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys

    ok = validate()
    sys.exit(0 if ok else 1)
