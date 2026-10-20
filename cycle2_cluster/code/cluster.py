"""
DivIVA 局在モデル — Cycle 2（単量体 + クラスター）

    C_(n-1) <-> C_n <-> C_(n+1)     細胞質の拡散（距離は軸 dz）
    M_(n-1) <-> M_n <-> M_(n+1)     膜上単量体の拡散（距離は弧長 ds）
    N_n                             クラスター（不動。拡散なし）

    C <-> M                         結合・解離          k_on(z), k_off
    M  -> N                         核形成 + 成長       (k_nuc + k_grow·N^p)·M
    N  -> M, C                      クラスターの崩壊    k_off_n(z)、分岐比 f_nm

    python cluster.py       # 検証だけ走る（本計算の前に必ず通すこと）

`model.py`（Cycle 1）の geometry / operators をそのまま使う。
幾何・保存形・面積重み・BDF に関する設計判断 [1]〜[8] はすべて継承している。
ここでは **Cycle 2 で新たに加わった判断だけ** を記す。

=============================================================================
  Cycle 2 の設計判断
=============================================================================

[C1] ★ 曲率は k_on ではなく k_off_n（クラスターの寿命）に入れる。
     これが Cycle 1 との唯一の本質的な違いであり、Cycle 2 の存在理由そのもの。

     Cycle 1 は極濃縮 2.43 倍を出すのに ln rho = 2.17 kT を単量体 1 個に
     要求した。単量体の曲率ミスマッチは 0.0094 kT なので 231 倍であり、
     物理的に正当化できない（`model.py` 既知の限界）。
     Lenarcic 2009 [3] / Ramamurthi 2009 [4] の主張も「単量体は曲率を
     感知できない、感知するのは会合体である」というものだった。

     そこで本ファイルでは既定で **rho = 1（単量体は曲率に無関心）** とし、
     曲率をクラスターの安定性だけに入れる:

         k_off_n(z) = k_off_n0 / w_n(z),    w_n = 極で rho_n、側壁で 1

     クラスターは Q 分子の集合体なので、必要な ln rho_n を Q で割ったものが
     「分子あたりに要求される曲率エネルギー」になる。Cycle 1 の 231 倍が
     Q 倍だけ緩む、というのが Cycle 2 の定量的な主張である。
     required_cluster_size() がその Q を逆算する。

[C2] ★ クラスターは動かない（D_n = 0）。拡散演算子を持たない。
     実測の上限は 2e-5 µm²/s（拘束半径 97 nm / 寿命 128.6 s）で、
     単量体の D_m = 0.1 µm²/s より 4 桁小さい。
     これは近似ではなく **パターンの鋭さの源** である。Cycle 1 で極濃縮に
     天井（1/a = 3.75）があったのは膜拡散がパターンをならすからで、
     クラスターにはその制約がかからない。

     ⚠️ その代わり n の場には拡散による平滑化が一切ないので、
        格子依存性は Cycle 1 より強く出る。V6 の格子収束を必ず見ること。

[C3] ★ 閉じた系を保つ。合成・分解は入れない（mu = 0 が既定）。
     `model.py` 末尾が警告しているとおり、合成・切断を入れると質量保存が
     使えなくなり、唯一の防波堤を失う。Cycle 2 では A/V の取り扱いが
     Cycle 1 より複雑（クラスターの崩壊が C と M に分岐する）なので、
     質量保存は Cycle 1 以上に重要である。

     希釈 mu を入れるときは、細胞質への一様な合成 mu·N/V で厳密に釣り合わせる。
     こうすると d(総量)/dt = mu·(N - 総量) となり、総量 = N から始める限り
     質量保存は厳密に成り立ったままになる。検算を殺さずに希釈を入れられる。

[C4] ★ N^p は p < 1 で N=0 に特異点を持つ。平滑化しないと BDF が止まる。
     p = 0.5 のとき d(N^p)/dN = 0.5·N^(-0.5) は N→0 で発散する。
     ic='cytoplasm'（膜もクラスターも空）から始めると初手で詰む。

         N^p  ->  (N² + eps²)^(p/2),   eps = 1e-6

     導関数は高々 300 程度に収まり、N ~ 45 での値のずれは (eps/N)² ~ 1e-15。
     負に振れた N も |N| として扱うので、BDF の行き過ぎで発散しない。

[C5] ★ k_on の校正の目標が Cycle 1 と違う。
     Cycle 1:  ∫k_on dA = k_off·V·(1 - f_cyt)/f_cyt     （膜 = 単量体だけ）
     Cycle 2:  ∫k_on dA = k_off·V· f_mono   /f_cyt      （膜 = 単量体 + クラスター）

     分配比は Giacomelli 2022 [1] Fig.2C の 3 集団
     （細胞質 29.6% / 膜上単量体 50.7% / 拘束 19.8%）に対応させる。
     Cycle 1 の (1-f_cyt) = 0.70 をそのまま使うと単量体が 40% 過剰になる。

[C6] ★ 時間スケールが 2 桁伸びる。T = 20 s では定常に全く届かない。
     Cycle 1 の律速は k_off = 9 /s（緩和 0.1 s 級）だったが、
     Cycle 2 では k_off_n = 0.0054 /s（緩和 185 s 級）が加わる。
     さらに成長項があるため実効的な緩和はそれより遅い。既定は T = 3000 s。
     ⚠️ Cycle 1 の感覚で T を縮めると「定常」でない値を定常として報告する。
        rise_time() で立ち上がりを必ず確認すること。

[C7] ★ 解析ヤコビアンを自分で書いて渡す。差分近似に任せない。
     未知数が n（Cycle 1）から 3n に増えたので、密ヤコビアンの差分近似は
     9 倍重い。結合構造は既知（C-M-N は同じセル内、拡散は隣接のみ）で、
     N^p の微分も閉じた形で書けるので、疎行列として組み立てて BDF に渡す。
     定常ソルバ [C8] のニュートン法にはそもそも必須である。
     差分と一致することは V8 で毎回確認している（相対差 2e-10）。

=============================================================================
  パラメータの独立性（Cycle 1 の表に追加される分）
=============================================================================
    k_off_n  独立・実測（FRAP T1/2 = 128.6 s）        影響: 大
    p        ✗ 実測なし。物理的議論のみ（縁で成長）     影響: ★最大
    k_grow   ✗ 独立でない。クラスター収支から導出       影響: 中
    k_nuc    ✗ 実測なし。N=0 を吸収状態にしないための値  影響: 小
    f_nm     ✗ 原理的に決まらない（FRAP は総脱出速度のみ）影響: 小〜中
    rho_n    独立・★これが求める量                     影響: 大

  k_grow は k_on と同じく従属変数である。固定されるのはクラスター収支:
      ∫(k_nuc + k_grow·N^p)·M dA = ∫k_off_n·N dA
  これを well-mixed で解くと k_grow ≈ 3.21e-4 が出る（PARAMETERS §4）。
  calibrate_k_grow() は同じ量を PDE で解き直す。両者が一致するかどうかが
  「well-mixed 近似がこの系で使えるか」の検証そのものになる（V8）。

  p は実測から一切制約されていないのに結論を左右する。既定 0.5 は
  「2 次元ラティスは縁で成長し、周長 ∝ sqrt(N)」という議論だけに立っている。
  ★ p < 1 では成長が飽和するので定常は一意（自発的対称性の破れは起きない）。
  ★ p > 1 では双安定になりうる。bistability_scan() で境界を探す。
    これは本プロジェクトの中心的な問い（PopZ の一過的な誘導だけで
    非対称を固定できるか）に直結する。

=============================================================================
  Cycle 2 で出た結論（summary() で再計算できる）
=============================================================================

(1) ★ 曲率エネルギーのギャップが閉じた。これが Cycle 2 の存在理由。
    Cycle 1: 単量体 1 個に ln rho = 2.17 kT を要求 → 曲率ミスマッチ
             0.0094 kT の 231 倍。正当化できない。
    Cycle 2: クラスターに ln rho_n = 0.87 kT を要求 → 92 分子ぶん。
             実測の平均クラスターサイズ 47 分子・最大 1198 分子
             （PARAMETERS §3）の分布の中に収まる。
    「単量体は曲率を感知できない、感知するのは会合体である」という
    Lenarcic 2009 [3] / Ramamurthi 2009 [4] の主張と定量的に整合した。

(2) 極濃縮の天井が外れた。クラスターは動かないので膜拡散でならされない。
    校正後の大腸菌での【予測】: クラスター分子の 66% が極、
    クラスターの極濃縮 2.48、膜全体（単量体 + クラスター）では 1.41。
    ⚠️ 単量体が 3 分の 2 を占めるため、蛍光で見える総量の極濃縮は
       クラスターの濃縮よりずっと低く出る。Wet と比べるときは
       どちらを測っているのかを必ず区別すること。

(3) ★ PopZ は「出し続ける」必要がある — ただし p ≤ 1 のとき。
    p = 0.5（既定）:
      持続摂動  theta=0.5/0.9/0.98 で 左右比 3.0 / 16.3 / 26.9 倍。
                Cycle 1 は theta=0.94 でようやく 10 倍だったので、
                クラスターが非対称を増幅している。
      一過的摂動 600 s かけて外すと左右比は厳密に 1 へ戻る。固定されない。
    p = 1.2（双安定）:
      theta=0.5 を 600 s かけて外すだけで左右比 5800 倍が永久に残る。

(4) ★ したがって次に測るべきはクラスター成長の次数 p である。
    p はこのモデルで唯一「実測の裏付けが全く無いのに結論を反転させる」量。
    p ≤ 1 なら PopZ の持続発現が要る。p > 1 なら一過的な誘導で足りる。
    実験計画（誘導タイミング・プロモーター強度）が丸ごと変わる。

(5) 発現量 N の扱いには分岐がある。
    k_grow を f_clu に校正し続ける限り、N は構造的に効かない（V11、
    厳密なスケーリング対称性）。Cycle 1 と同じ構造である。
    k_grow を分子定数として固定すると N は効き、p=2 では N=1690 で単安定、
    N≈5000 から双安定に変わる。さらに上げると両極が飽和して再び対称に戻る
    （＝発現量に最適な窓がある）。どちらを採るかは仮定であって
    データでは決まらない。

=============================================================================
  既知の限界
=============================================================================
  - クラスターのサイズ分布を持たない。N は「クラスターに属する分子の面密度」
    であって「クラスターの個数」ではない。平均サイズ 47 分子・最大 1198 分子
    という広い分布（PARAMETERS §3）は表現できていない。
    required_cluster_size() が返す Q は「分布の代表値としての Q」であり、
    分布を持つモデルなら必要な Q はこれより小さくなりうる。
  - 校正の目標値 66%（極・隔壁のクラスター割合）には隔壁が入っている。
    本ファイルは隔壁を持たないので、66% をすべて極に帰しているぶん
    rho_n を過大評価している。PARAMETERS §9-1 の最優先の確認事項。
  - 52.27%（PALM）と 19.8%（SPT 拘束画分）が 2.6 倍食い違う問題は未解決。
    f_clu = 0.198 は後者を採っている。前者を採ると k_grow が上がる。
    par.f_clu をスキャンして両端を出すこと。
  - PopZ・TEV は本ファイルでは扱わない（Cycle 3）。ただし空間占有は
    block 引数で既に入れられる。末尾を参照。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, identity, bmat

import model
from model import geometry, operators


# =============================================================================
class Par(model.Par):
    """Cycle 2 の既定パラメータ。model.Par を継承して追加分だけ書く。

    上書きは `variant(rho_n=20.0)` を使う（クラス属性を直接書き換えないこと）。
    """

    # --- 分配比（Giacomelli 2022 [1] Fig.2C、20 ms 撮影の 3 集団）
    f_cyt = 0.296      # 細胞質          （高速移動）
    f_mono = 0.507     # 膜上の単量体    （低速移動）
    f_clu = 0.198      # クラスター      （拘束）
    #   ⚠️ 撮影速度依存が大きい。5 ms 撮影では 15.7 / 29.3 / 55.0%（3 倍振れる）。
    #      PALM のクラスター割合 52.27% とも 2.6 倍食い違う（PARAMETERS §3）。
    #      f_clu は感度を必ず見ること。

    # --- クラスターの実測パラメータ（すべて B. subtilis）
    k_off_n = 0.0054   # 1/s  Feddersen 2021 [2] Tab.2、DivIVA-GFP の FRAP
    #   T1/2 = 128.64 s -> ln2/T1/2 = 0.0054。範囲 0.0054〜0.0115。
    #   ★ 大腸菌には MinJ が無いので dminJ の 0.0136 /s のほうが対応する
    #     可能性が高い（PARAMETERS §2）。要スキャン。
    D_n = 0.0          # µm²/s クラスターは不動（実測上限 2e-5、設計判断 [C2]）
    k_off_n_anchor = "pole"   # k_off_n をどこの値とみなすか（設計判断 [C12]）
    #   "pole"     : 実測 0.0054 /s は【極】のクラスターの値。側壁は rho_n 倍速い
    #   "sidewall" : 実測は側壁の値で、極は rho_n 分の 1 に遅くなる
    #   ★ FRAP は DivIVA が居る場所、すなわち極で測られている。したがって
    #     "pole" が素直な読み方である。既定はこちら。
    #     定常状態はどちらでもほぼ変わらない（k_grow が校正で吸収する）が、
    #     時間スケールは rho_n 倍ちがう。REPORT で両方報告すること。

    # --- 仮定（実測なし）⚠️
    p = 0.5            # 成長の次数。★ 最も危険な仮定。PARAMETERS §5
    k_nuc = 1e-5       # 1/s  核形成。N=0 を吸収状態にしないためだけの小さな値
    f_nm = 0.5         # クラスター崩壊の分岐比。1 なら全部 M へ、0 なら全部 C へ
    #   ⚠️ FRAP の T1/2 は総脱出速度しか測っていないので内訳は原理的に不明。
    #      0 と 1 で振って両端を出すこと。

    # --- 曲率（Cycle 2 の主役）
    rho = 1.0          # ★ 単量体は曲率を感知しない（設計判断 [C1]）
    rho_n = 2.370      # 極/側壁のクラスター寿命比。★これが求める量
    #   B. subtilis 幾何（R=0.425, L=2.8）で「クラスター分子の 66% が極」に
    #   合わせると rho_n = 2.147（ln = 0.764 kT）。これを大腸菌へ移植する。
    #   移植の指数は Cycle 2 の議論（ln rho_n = Q·dG_mono、dG_mono ∝ 1/R²）に
    #   合わせて 1/R² を採る:
    #     ln rho_n_Ec = 0.764 × (0.425/0.400)² = 0.8627  ->  rho_n = 2.370
    #   ⚠️ Cycle 1 と同じ 1/(2R) の形で移植すると 2.25。差は 5% で、
    #      他の不確かさに比べて無視できる。Cycle 1 では移植の形が効いたが、
    #      Cycle 2 では効かない。
    #   なお大腸菌幾何で直接 66% に合わせてもほぼ同じ値になる。
    #   k_off_n_anchor を "sidewall" にしても 2.377 とほとんど動かない
    #   （定常は k_off_n の一様倍率に鈍い。設計判断 [C12]）。
    #   この rho_n での大腸菌の極割合 66.1% / クラスター濃縮 2.48 は【予測】。

    # --- 成長（既定は切ってある。設計判断 [C3]）
    mu = 0.0           # 1/s  希釈。世代時間 1500 s なら ln2/1500 = 4.6e-4

    # --- 従属パラメータ（None なら実行時に校正して埋める）
    k_grow = None      # クラスター収支から導出。k_grow_wellmixed() 参照

    # --- 数値
    T = 3000.0         # s  既定の積分時間（設計判断 [C6]）
    eps_p = 1e-6       # N^p の平滑化（設計判断 [C4]）
    atol_rel = 1e-11   # 各場の代表値に対する相対 atol（設計判断 [C11]）
    rtol = 1e-8        # ★ Cycle 1 の 1e-9 より緩い（設計判断 [C11]）


def derive(base, **kw):
    """base を一部だけ上書きした新しいパラメータクラスを返す。

    ★ 関数の内部で派生を作るときは必ず引数の par を基底にすること。
      モジュール既定の Par を基底にすると、呼び出し側が渡した設定が
      黙って無視される（実際に一度これを踏んだ）。
    """
    known = {k for k in dir(base) if not k.startswith("_")}
    unknown = set(kw) - known
    if unknown:
        raise ValueError(f"{base.__name__} に無い属性: {sorted(unknown)}")
    return type("ParVariant", (base,), kw)


def variant(**kw):
    """モジュール既定の Par を基底にした派生。スキャンの入口用。"""
    return derive(Par, **kw)


def fractions(par=Par):
    """3 相の分配比を正規化して (f_cyt, f_mono, f_clu) で返す。

    ⚠️ 原典 [1] Fig.2C の 3 集団は 29.6 + 50.7 + 19.8 = 100.1% で、
       丸めのぶん 0.1% だけ 1 を超える。Par には原典の値をそのまま書いておき
       （書き換えると出典と照合できなくなる）、ここで正規化して使う。
       ずれが 2% を超えるのは打ち間違いなので、そのときは止める。
    """
    f = np.array([par.f_cyt, par.f_mono, par.f_clu], float)
    if np.any(f < 0):
        raise ValueError(f"分配比が負: {f}")
    s = f.sum()
    if abs(s - 1.0) > 0.02:
        raise ValueError(f"f_cyt + f_mono + f_clu = {s:.4f}。1 から離れすぎ")
    return tuple(f / s)


# =============================================================================
#  幾何と重み
# =============================================================================
def geometry2(n, par=Par):
    """Cycle 1 の geometry() に、クラスター用の曲率重み w_n を足して返す。

    w  : 単量体の結合重み（par.rho。既定 1.0 = 曲率に無関心）
    w_n: クラスターの安定性重み（par.rho_n。極で rho_n、側壁で 1）

    どちらも `model.geometry` の面積重み付きセル平均をそのまま使う
    （設計判断 [2]）。rho だけ差し替えて 2 回呼ぶ。
    """
    g = geometry(n, par)
    g["w_n"] = geometry(n, type("P", (par,), {"rho": par.rho_n}))["w"]
    return g


def k_on_of(g, par=Par, block=None):
    """k_on(z) [µm/s]。定常の細胞質割合が f_cyt になるよう校正（設計判断 [C5]）。

    固定されるのは面積積分のほう:
        ∫k_on dA = k_off · V · f_mono / f_cyt
    Cycle 1 の (1 - f_cyt) が f_mono に変わっている点に注意。

    block: 長さ n の配列。結合能を局所的に下げる（PopZ の空間占有）。
           ⚠️ Cycle 1 と同じく、ブロックしても ∫k_on dA は保たれる。
              つまり「塞いだぶんの結合能が他へ再配分される」という約束。
              f_cyt を固定したまま配分だけを動かすための規約である。
    """
    f_cyt, f_mono, _ = fractions(par)
    w = g["w"] if block is None else g["w"] * block
    return (w * par.k_off * f_mono * g["V"].sum()
            / (f_cyt * (w * g["A"]).sum()))


def k_off_n_of(g, par=Par):
    """k_off_n(z) [1/s]。曲率はここに入る（設計判断 [C1]）。

        k_off_n(z) = base / w_n(z),   w_n は極で rho_n、側壁で 1

    [C12] ★ base をどこに合わせるかで時間スケールが rho_n 倍変わる。
      実測の k_off_n = 0.0054 /s は Feddersen 2021 の FRAP だが、
      FRAP は DivIVA が居る場所（＝極）で測られている。したがって
      「極で 0.0054 /s」と読むのが素直で、側壁はそれより rho_n 倍速い。
      par.k_off_n_anchor = "pole" がこの読み方（既定）。
      "sidewall" にすると実測値を側壁の値とみなし、極は rho_n 分の 1 に遅くなる。

      ⚠️ 定常状態はどちらでもほぼ変わらない。k_off_n を一様に λ 倍しても
        k_grow の校正が λ 倍で吸収するためで、厳密に不変でないのは
        k_nuc が固定だからだが、その寄与は成長項の 1% 未満である。
        変わるのは時間スケールで、こちらは rho_n 倍ちがう。
    """
    if par.k_off_n_anchor == "pole":
        base = par.k_off_n * par.rho_n
    elif par.k_off_n_anchor == "sidewall":
        base = par.k_off_n
    else:
        raise ValueError(f"k_off_n_anchor は 'pole' か 'sidewall': "
                         f"{par.k_off_n_anchor!r}")
    return base / g["w_n"]


# =============================================================================
#  k_grow の校正
# =============================================================================
def k_grow_wellmixed(par=Par):
    """well-mixed 近似でのクラスター収支から k_grow を出す（PARAMETERS §4 の再現）。

        ∫(k_nuc + k_grow·N^p)·M dA = ∫k_off_n·N dA

    空間依存を無視し、面密度を細胞全体の平均で置くと

        (k_nuc + k_grow·N̄^p)·M̄·A = k_off_n·N̄·A
        k_eff := k_off_n · f_clu / f_mono
        k_grow = (k_eff - k_nuc) / N̄^p,    N̄ = f_clu·N/A

    既定値で k_eff = 0.00211 /s、N̄ = 44.4 /µm²、k_grow = 3.14e-4 となり、
    PARAMETERS の 3.21e-4（分配比 0.5/0.2、B. subtilis の面積で計算）と
    3 % 以内で一致する。
    """
    _, f_mono, f_clu = fractions(par)
    g = geometry(401, par)
    n_bar = f_clu * par.N / g["A"].sum()
    k_eff = par.k_off_n * f_clu / f_mono
    if f_clu == 0.0:
        return 0.0
    return (k_eff - par.k_nuc) / n_bar**par.p


# 校正結果のキャッシュ。同じパラメータで何度も brentq を回さないため。
_KG_CACHE = {}


def _kg_key(par):
    """k_grow の校正結果を決めるパラメータの組。"""
    return (par.R, par.L, par.N, par.D_c, par.D_m, par.k_off, par.k_off_n,
            par.rho, par.rho_n, par.p, par.k_nuc, par.f_nm, par.mu,
            par.f_cyt, par.f_mono, par.f_clu, par.k_off_n_anchor)


def _kg(par):
    """k_grow を返す。None なら f_clu に合わせて PDE で校正する。

    ★ k_grow は独立パラメータではない。固定されるのはクラスター収支のほうで、
      実測のクラスター割合 f_clu = 19.8% を再現するよう決まる。
      Cycle 1 で ∫k_on dA が f_cyt に固定されていたのと同じ構造である。

    ⚠️ well-mixed の推定値（k_grow_wellmixed）は rho_n = 1 でしか使えない。
       rho_n > 1 では極の k_off_n が rho_n 分の 1 に下がるので、同じ k_grow だと
       極がシンクになってクラスターが暴走する（rho_n=30 で f_clu が
       0.198 -> 0.889 まで走り、緩和に 1e5 秒かかる）。
       この食い違いの大きさ自体が「この系で well-mixed 近似は使えない」
       という結果であり、V8 で毎回確認している。
    """
    if par.k_grow is not None:
        return par.k_grow
    key = _kg_key(par)
    if key not in _KG_CACHE:
        _KG_CACHE[key] = calibrate_k_grow(par=par)
    return _KG_CACHE[key]


# =============================================================================
#  ソルバ
# =============================================================================
def _pow_p(x, p, eps):
    """N^p の平滑化（設計判断 [C4]）。(N² + eps²)^(p/2)。"""
    return (x * x + eps * eps) ** (0.5 * p)


def _sparsity(n):
    """3n × 3n の疎ヤコビアン構造。

    いまは解析ヤコビアン（_assemble の jac）を直接渡しているので本体では
    使っていない。構造だけ必要になったとき（サイズ分布版へ拡張するときなど）
    のために残してある。

        C 行: C は三重対角、M と N は対角
        M 行: C 対角、M 三重対角、N 対角
        N 行: M 対角、N 対角（C とは直接つながらない）
    """
    tri = diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])
    I = identity(n)
    return bmat([[tri, I, I],
                 [I, tri, I],
                 [None, I, I]], format="csr")


def _assemble(n, par, block=None):
    """幾何・演算子・速度定数・RHS をまとめて作る。solve() と steady() で共有する。

    戻り値の rhs(y, blocked) は block の有無を引数で切り替える。
    時間依存の block（一過的な PopZ）はここで k_on を 2 通り作っておき、
    呼ぶ側が時刻で選ぶ。校正済みの k_on を途中で作り直さないための作り。
    """
    f_cyt, f_mono, f_clu = fractions(par)
    g = geometry2(n, par)
    L_c, L_m = operators(g, par)
    V, A = g["V"], g["A"]
    k_off_n = k_off_n_of(g, par)
    k_grow = _kg(par)
    p, eps, f_nm, mu = par.p, par.eps_p, par.f_nm, par.mu

    k_on_free = k_on_of(g, par, None)
    k_on_blk = k_on_free if block is None else k_on_of(g, par, block)
    src = mu * par.N / V.sum()      # 細胞質への一様合成（設計判断 [C3]）

    def rhs(y, blocked=False):
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        flux = k_on * C - par.k_off * M                     # 分子/(µm²·s)  C -> M
        G = (par.k_nuc + k_grow * _pow_p(Nf, p, eps)) * M   # M -> N
        rel = k_off_n * Nf                                  # N -> M, C
        dC = L_c @ C + (-flux + (1 - f_nm) * rel) * A / V - mu * C + src
        dM = L_m @ M + flux - G + f_nm * rel - mu * M
        dN = G - rel - mu * Nf
        return np.concatenate([dC, dM, dN])
        # ★ A/V は細胞質側の結合項だけ（設計判断 [4]）。
        #   クラスター崩壊の (1-f_nm)·rel も「膜面あたり」の量なので A/V が要る。
        #   dM, dN には要らない。拡散演算子にも要らない。

    absLc, absLm = abs(L_c), abs(L_m)

    def rhs_mag(y, blocked=False):
        """各式の「項の大きさ」を返す。残差を無次元化するための尺度。

        ★ 絶対残差では収束を判定できない。拡散項は D_c/dz²·C ~ 1e7 の桁で、
          結合項は ~1e3、クラスター項は ~1e-1 と 8 桁散らばっている。
          絶対値で閾値を切ると、細胞質の式だけ見ているのと変わらない。
          各式の項の絶対値の和で割った無次元残差で判定する。
        """
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        AV = A / V
        b = k_on * C + par.k_off * M
        G = (par.k_nuc + k_grow * _pow_p(Nf, p, eps)) * M
        rel = k_off_n * Nf
        mC = absLc @ C + (b + (1 - f_nm) * rel) * AV + mu * C + src
        mM = absLm @ M + b + G + f_nm * rel + mu * M
        mN = G + rel + mu * Nf
        return np.concatenate([mC, mM, mN])

    def jac(y, blocked=False):
        """解析ヤコビアン（疎）。差分近似より速く、ニュートン法に必要。

        phi(N) = (N² + eps²)^(p/2) （[C4] の平滑化）、
        phi'(N) = p·N·(N² + eps²)^(p/2 - 1)。
        """
        M, Nf = y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        AV = A / V
        phi = _pow_p(Nf, p, eps)
        dphi = p * Nf * (Nf * Nf + eps * eps) ** (0.5 * p - 1.0)
        kg_phi = par.k_nuc + k_grow * phi          # G / M

        dCC = L_c - diags(k_on * AV + mu)
        dCM = diags(par.k_off * AV)
        dCN = diags((1 - f_nm) * k_off_n * AV)
        dMC = diags(k_on)
        dMM = L_m - diags(par.k_off + kg_phi + mu)
        dMN = diags(f_nm * k_off_n - k_grow * dphi * M)
        dNM = diags(kg_phi)
        dNN = diags(k_grow * dphi * M - k_off_n - mu)
        return bmat([[dCC, dCM, dCN],
                     [dMC, dMM, dMN],
                     [None, dNM, dNN]], format="csr")

    return dict(g=g, V=V, A=A, rhs=rhs, jac=jac, rhs_mag=rhs_mag,
                k_on=k_on_free, k_on_blk=k_on_blk,
                k_off_n=k_off_n, k_grow=k_grow,
                fr=(f_cyt, f_mono, f_clu))


def initial(n, par=Par, ic="uniform", block=None, seed=10.0):
    """初期条件 y0 を返す。"""
    g = geometry2(n, par)
    V, A = g["V"], g["A"]
    f_cyt, f_mono, f_clu = fractions(par)

    def split(fc, fm, fn):
        return (np.full(n, fc * par.N / V.sum()),
                np.full(n, fm * par.N / A.sum()),
                np.full(n, fn * par.N / A.sum()))

    if ic == "uniform":
        C0, M0, N0f = split(f_cyt, f_mono, f_clu)
    elif ic == "cytoplasm":
        C0, M0, N0f = np.full(n, par.N / V.sum()), np.zeros(n), np.zeros(n)
    elif ic == "bleach":
        base = solve(n, par, ic="uniform", n_out=3, block=block)
        C0, M0, N0f = base["C"].copy(), base["M"].copy(), base["Nf"].copy()
        cap = g["z"] < par.R
        M0[cap] = 0.0
        N0f[cap] = 0.0
    elif ic == "one_pole":
        C0, M0, _ = split(f_cyt, f_mono, 0.0)
        N0f = np.zeros(n)
        N0f[: max(2, n // 20)] = 1.0
        N0f *= f_clu * par.N / (N0f * A).sum()
    elif ic == "seed_pole":
        # 一様分配に、片極だけクラスターを seed 倍に積んだ状態（双安定性の探索用）
        # ★ seed の大きさは結果を変える。双安定な系では「臨界の種」より
        #   大きくないと非対称な枝に落ちない。固定値で「双安定でない」と
        #   結論しないこと（p=2 で実際にこれを踏んだ）。critical_seed() を使う。
        C0, M0, N0f = split(f_cyt, f_mono, f_clu)
        N0f = N0f * (1.0 + (seed - 1.0) * (g["A_capL"] / A))
        tot = (C0 * V).sum() + (M0 * A).sum() + (N0f * A).sum()
        sc = par.N / tot
        C0, M0, N0f = C0 * sc, M0 * sc, N0f * sc
    else:
        raise ValueError(f"unknown ic: {ic!r}")
    return np.concatenate([C0, M0, N0f])


def _report(par, asm, ys, ts, ic):
    """時系列 ys（形 3n × len(ts)）から診断量をまとめる。solve/steady で共有。"""
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
    N_tot0 = mass_t[0]

    # クラスター収支（設計判断 [C3]。閉じた系でなくなったときの防波堤）
    G_ss = (par.k_nuc + asm["k_grow"] * _pow_p(Nf, par.p, par.eps_p)) * M
    infl, outfl = (G_ss * A).sum(), (asm["k_off_n"] * Nf * A).sum()
    budget_err = abs(infl - outfl) / max(outfl, 1e-300)

    n_int = (Nf * A).sum()
    return dict(
        g=g, par=par, t=ts, C=C, M=M, Nf=Nf, Ct=Ct, Mt=Mt, Nt=Nt,
        k_on=asm["k_on"], k_off_n=asm["k_off_n"], k_grow=asm["k_grow"],
        ic=ic, N0=N_tot0,
        enrichment=enr_t[-1], enr_t=enr_t,
        enr_clu=enr_clu_t[-1], enr_clu_t=enr_clu_t,
        clu_pole_frac=(Nf * g["A_cap"]).sum() / max(n_int, 1e-300),
        f_cyt=(C * V).sum() / N_tot0,
        f_mono=(M * A).sum() / N_tot0,
        f_clu=n_int / N_tot0,
        mass_err=np.abs(mass_t - N_tot0).max() / N_tot0,
        budget_err=budget_err,
        cyto_var=1 - C.min() / C.max(),
        asym_clu=_lr(Nf, g),
        residual=float(np.max(np.abs(asm["rhs"](ys[:, -1]))
                              / np.maximum(asm["rhs_mag"](ys[:, -1]), 1e-300))),
    )


def solve(n=101, par=Par, T=None, ic="uniform", n_out=80,
          block=None, seed=10.0, y0=None):
    """時間発展を解く。

    ic:
        'uniform'    3 相が分配比どおりに一様に分かれた状態から（既定）
        'cytoplasm'  全分子が細胞質（発現開始を模す）
        'one_pole'   クラスターが片極だけにある状態から（対称性の回復を見る）
        'bleach'     定常に達したあと片極の膜・クラスターを漂白（FRAP）
        'seed_pole'  一様分配 + 片極にクラスターの種（双安定性の探索用）
    block:
        長さ n の配列。k_on を局所的に下げる（PopZ の空間占有）。
    y0:
        初期状態を直接与える（ic を無視）。摂動を途中で外す実験に使う。

    [C10] ★ 摂動を「時刻で切り替える」実装にしてはいけない。
      RHS の中で t を見て block を外すと RHS が不連続になり、BDF が
      その時刻で刻み幅を潰して事実上止まる（実際に踏んだ）。
      摂動をかけた積分 → その終状態を y0 にして摂動なしで再開、と
      2 段階に分けること。latch_test() がその形になっている。

    戻り値の主なキー:
        enrichment    定常の極濃縮（(M+N) の極膜密度 / 全体平均）
        enr_clu       クラスターだけの極濃縮
        clu_pole_frac 全クラスター分子のうち極にいる割合（校正の目標 0.66）
        f_cyt/f_mono/f_clu  定常の 3 相分配
        mass_err      質量保存の相対誤差。★これが大きければ結果を信用しない
        budget_err    クラスター収支 ∫G = ∫k_off_n·N の相対誤差
        residual      最終時刻での |dy/dt| の最大値（定常に達したかの目安）
    """
    if T is None:
        T = par.T
    asm = _assemble(n, par, block)
    y0 = (initial(n, par, ic, block, seed=seed) if y0 is None
          else np.asarray(y0, float))
    rhs = asm["rhs"]

    jac = asm["jac"]
    blk = block is not None

    def f(t, y):
        return rhs(y, blocked=blk)

    def fj(t, y):
        return jac(y, blocked=blk)

    t_eval = np.clip(np.unique(np.r_[0.0, np.logspace(-4, np.log10(T), n_out)]),
                     0.0, T)
    # [C11] ★ 長時間積分では許容誤差を Cycle 1 より緩める。
    #   Cycle 1 は「rtol は 1e-9 で十分（1e-6 と 1e-12 で答えは同一）」と
    #   書いているが、それは T=20 s の話である。Cycle 2 は定常まで 1e5 秒
    #   積むので、定常に落ち着いたあと BDF が誤差推定を満たせなくなり、
    #   刻み幅を潰して事実上止まる。T=3e4 までの実測:
    #     rtol=1e-9  nfev=95874 njev=7900  38 秒
    #     rtol=1e-8  nfev=  970 njev=   2  0.1 秒
    #   精度が落ちていないことは V9（steady() との一致）で毎回確認している。
    #   atol も場ごとの大きさに合わせてベクトルで与える。スカラーの 1e-11 は
    #   C（~1e3 分子/µm³）の丸め誤差の床 1e-10 より小さく、意味がない。
    atol_vec = par.atol_rel * np.concatenate([
        np.full(n, par.N / asm["V"].sum()),     # C: 分子/µm³
        np.full(n, par.N / asm["A"].sum()),     # M: 分子/µm²
        np.full(n, par.N / asm["A"].sum())])    # N: 分子/µm²
    sol = solve_ivp(f, (0, T), y0, method="BDF", t_eval=t_eval,
                    rtol=par.rtol, atol=atol_vec, jac=fj)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return _report(par, asm, sol.y, sol.t, ic)


def steady(n=101, par=Par, ic="uniform", block=None, y0=None,
           tol=1e-7, maxit=40, seed_T=None, seed=10.0):
    """定常状態を直接解く（長時間積分をしない）。

    ★ なぜ要るか: rho_n が大きいと極の k_off_n が rho_n 分の 1 まで下がり、
      緩和時間が 1/k_off_n × rho_n ~ 1e5 秒まで伸びる（設計判断 [C6]）。
      スキャンや校正のたびに 1e5 秒を BDF で積むのは非現実的なうえ、
      「定常に見えるが届いていない」値を報告してしまう（実際に一度踏んだ:
      T=3000 s で f_clu=0.855 と出たが、真の定常は 0.889 だった）。

    [C8] ★ 短い時間積分で助走をつけてから、正値性を保つニュートン法で仕上げる。
      助走なしだと収束しない。実測した挙動:
        初期値（一様）から直接ニュートン → N を 0 に押しやって停滞
          （刻み幅が正値性の制約で 1 回あたり 10 分の 1 ずつしか進まない）
        seed_T = 3000 s の積分で助走  → 無次元残差
          1.5e-1 → 2.1e-2 → 3.9e-4 → 1.4e-7 → 2.3e-11（4 回で二次収束）
      助走は 0.1 秒で済むので、これを既定にする。
      既製の root は使わない。hybr は濃度のままだと負の濃度に落ち、
      対数変数 y=exp(u) にすると exp がオーバーフローする（両方試した）。

    [C9] ★ 収束判定は無次元残差で行う。絶対残差では判定できない。
      拡散項は D_c/dz²·C ~ 1e7、結合項は ~1e3、クラスター項は ~1e-1 と
      8 桁散らばっている。絶対値で閾値を切ると細胞質の式しか見ていないのと
      同じになる。各式をその項の絶対値の和で割った比で測る（rhs_mag）。
      丸め誤差の床は条件数次第で 1e-10〜1e-8 まで動くので、tol = 1e-7 と
      それより緩く取る。厳しくすると「もう下がらない残差」を追って
      助走をやり直し続け、校正が何分もかかるようになる（実際に踏んだ）。

    ⚠️ 閉じた系なので定常方程式は 1 本ぶん冗長（Σ(V·dC + A·dM + A·dN) ≡ 0）。
      そのまま解くとヤコビアンが特異になる。最後の 1 本を「総量 = N」の
      拘束で置き換えて階数を回復させる。

    戻り値は solve() と同じ形。t は [inf]。
    ★ 結果は必ず steady_residual と mass_err で確認すること。
      solve() の長時間積分と一致するかは V11 で検証している。
    """
    from scipy.sparse.linalg import spsolve

    asm = _assemble(n, par, block)
    V, A = asm["V"], asm["A"]
    rhs, jac, rhs_mag = asm["rhs"], asm["jac"], asm["rhs_mag"]
    blocked = block is not None

    floor = 1e-14 * par.N / A.sum()

    # 拘束の大きさを RHS と揃える（桁が違うとニュートン法が拘束を無視する）
    scale = max(par.k_off_n, 1e-12) * par.N / A.sum()
    mass_row = np.concatenate([V, A, A]) / par.N * scale

    def F(v):
        r = rhs(v, blocked=blocked)
        mass = (v[:n] * V).sum() + (v[n:2 * n] * A).sum() + (v[2 * n:] * A).sum()
        r[-1] = (mass / par.N - 1.0) * scale
        return r

    def relres(v):
        """無次元残差（設計判断 [C9]）。"""
        r = np.abs(rhs(v, blocked=blocked))
        return float(np.max(r / np.maximum(rhs_mag(v, blocked=blocked), 1e-300)))

    def newton(y_start):
        """正値性を保つ減衰ニュートン。(解, 無次元残差) を返す。"""
        y = np.maximum(np.asarray(y_start, float), floor)
        best_y, best, stall = y.copy(), relres(y), 0
        for _ in range(maxit):
            if best < tol:
                break
            dy = spsolve(_with_mass_row(jac(y, blocked=blocked), mass_row), -F(y))
            if not np.all(np.isfinite(dy)):
                break
            neg = dy < 0              # 正値性: 元の 10% を下回らないところまで
            a = 1.0
            if np.any(neg):
                a = min(1.0, 0.9 * np.min(y[neg] / -dy[neg]))
            y = np.maximum(y + a * dy, floor)
            r = relres(y)
            if r < best:
                best, best_y, stall = r, y.copy(), 0
            else:
                stall += 1
                if stall >= 3:        # 丸め誤差の床に着いた
                    break
        return best_y, best

    # ★ 助走 → ニュートン。届かなければ助走を 10 倍ずつ伸ばして再試行
    #   （設計判断 [C8]）。y0 を与えられた場合はまずそれを試す。
    y, res = (None, np.inf)
    starts = []
    if y0 is not None:
        starts.append(("与えられた y0", lambda: np.asarray(y0, float)))
    T0 = seed_T if seed_T is not None else par.T
    for k in range(3):
        Tk = T0 * 10.0**k
        starts.append((f"助走 {Tk:.3g} s", lambda Tk=Tk: np.concatenate(
            [(sd := solve(n, par, T=Tk, ic=ic, n_out=3, block=block,
                          seed=seed))["C"], sd["M"], sd["Nf"]])))
    for _label, mk in starts:
        yk, rk = newton(mk())
        if rk < res:
            y, res = yk, rk
        if res < tol:
            break

    if res > 1e-5:
        raise RuntimeError(f"steady() 収束せず: 無次元残差 {res:.3e}（許容 1e-5）。"
                           " seed_T を伸ばすか y0 を与えること")

    if np.any(y <= 0):
        raise RuntimeError("steady() が非正の濃度を返した")

    out = _report(par, asm, y[:, None], np.array([np.inf]), ic)
    out["steady_residual"] = res
    return out


def _with_mass_row(Jm, mass_row):
    """ヤコビアンの最後の 1 行を質量拘束で置き換える（冗長性の解消）。"""
    Jl = Jm.tolil()
    Jl[Jm.shape[0] - 1, :] = mass_row
    return Jl.tocsr()


def _lr(f, g):
    """左右の極キャップでの比（左/右）。1 なら対称。"""
    l, r = (f * g["A_capL"]).sum(), (f * g["A_capR"]).sum()
    return l / max(r, 1e-300)


# =============================================================================
#  校正と解析
# =============================================================================
def rho_n_ceiling(par=Par, hi=1e6):
    """k_grow の校正が成り立つ rho_n の上限を返す。

    ★ モデルに実在する制約であって、数値上の都合ではない。
      極では k_off_n が rho_n 分の 1 まで下がるので、rho_n を上げていくと
      やがて **核形成だけで** 実測のクラスター割合 f_clu を超えてしまう。
      その状態では成長項 k_grow を 0 にしても f_clu を実測に合わせられない。
      おおよその目安は N/M ~ k_nuc·rho_n/k_off_n が f_clu/f_mono を超えるとき、
      つまり rho_n > (f_clu/f_mono)·(k_off_n/k_nuc)。既定値では ~211。

    ⚠️ この上限は k_nuc（実測なし・PARAMETERS §5）に比例する。
      k_nuc を小さくすれば上限は上がるので、上限そのものを物理的な結論として
      引用しないこと。「rho_n をいくらでも大きくはできない」という定性的な
      事実と、校正が上限に近づいていないかの確認に使う。
    """
    from scipy.optimize import brentq

    target = fractions(par)[2]

    def g(v):
        # k_grow = 0（核形成だけ）での f_clu が target を超えるか
        return steady(101, derive(par, rho_n=v, k_grow=0.0))["f_clu"] - target

    if g(1.0) > 0:
        raise RuntimeError("rho_n=1 でも核形成だけで f_clu が実測を超える。"
                           " k_nuc が大きすぎる")
    v = 2.0
    while v < hi and g(v) < 0:
        v *= 2.0
    if v >= hi:
        return hi
    return brentq(g, v / 2, v, rtol=1e-6)


def calibrate_rho_n(target=0.66, n=101, par=Par, lo=1.0, hi=None):
    """「クラスター分子のうち極にいる割合」が target になる rho_n を返す。

    target = 0.66 は Feddersen 2021 [2] の PALM（極・隔壁のクラスター割合）。
    ⚠️ この 66% には隔壁が入っている。本モデルは隔壁を持たないので、
       66% をすべて極に帰すと rho_n を過大評価する（既知の限界）。
       PARAMETERS §9-1 の最優先の確認事項。

    ★ rho_n を変えるたび k_grow も校正し直される（_kg 参照）。
      f_clu = 19.8% は実測なので、rho_n をどう動かしても保たれるべき量である。
    """
    from scipy.optimize import brentq

    if hi is None:
        hi = min(100.0, 0.9 * rho_n_ceiling(par))
    warm = {"y": None}

    def f(v):
        r = steady(n, derive(par, rho_n=v), y0=warm["y"])
        warm["y"] = np.concatenate([r["C"], r["M"], r["Nf"]])
        return r["clu_pole_frac"] - target

    flo, fhi = f(lo), f(hi)
    if flo > 0:
        raise RuntimeError(f"rho_n={lo} でも極の割合が {target} を超える")
    if fhi < 0:
        raise RuntimeError(f"rho_n={hi:.3g}（校正可能な上限）でも極の割合が "
                           f"{target} に届かない")
    return brentq(f, lo, hi, rtol=1e-8)


def calibrate_k_grow(n=101, par=Par, hi0=None):
    """定常のクラスター割合が実測の f_clu になる k_grow を求める。

    f_clu(k_grow) は単調増加（k_grow=0 では核形成だけでほぼ 0、大きくすると
    すべてクラスターへ）なので brentq で一意に決まる。

    ★ 時間発展ではなく steady() を使う。rho_n が大きいと緩和時間が 1e5 秒級に
      なるので、積分で校正すると 1 回 2 分かかるうえ「定常に届いていない値」で
      校正してしまう（実際にこれを踏んだ）。

    ★ 直前の解を次の初期値に渡す（継続法）。毎回ゼロから助走を取り直すと
      brentq の 1 回ごとに BDF 積分が走り、校正が数分かかる。
    """
    from scipy.optimize import brentq

    target = fractions(par)[2]
    if target == 0.0:
        return 0.0
    warm = {"y": None}

    def f(v):
        r = steady(n, derive(par, k_grow=v), y0=warm["y"])
        warm["y"] = np.concatenate([r["C"], r["M"], r["Nf"]])
        return r["f_clu"] - target

    lo = 0.0
    if f(lo) > 0:
        raise RuntimeError("k_grow=0 でも f_clu が目標を超える。k_nuc が大きすぎる")
    hi = hi0 if hi0 is not None else max(k_grow_wellmixed(par), 1e-9)
    for _ in range(60):
        if f(hi) > 0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("f_clu の目標を上回る k_grow が見つからない")
    return brentq(f, lo, hi, rtol=1e-10)


def required_cluster_size(rho_n=None, par=Par, lp=100.0, length=30.0):
    """必要な ln rho_n を、単量体 1 個の曲率エネルギーで割った分子数 Q を返す。

    クラスターが Q 分子の剛体的な集合体で、各分子が独立に曲率ミスマッチ
    エネルギーを稼ぐなら
        ln rho_n = Q · dG_mono
    なので Q = ln(rho_n) / dG_mono。

    これが Cycle 2 の定量的な主張そのもの:
      Cycle 1 は単量体 1 個に 231 分子ぶんのエネルギーを要求していた（破綻）。
      Cycle 2 は Q 分子の集合体に同じ量を要求する。実測の平均クラスターサイズ
      47 分子（PARAMETERS §3）と比べて Q が小さければ、機構は成立する。

    ⚠️ 各分子が独立に稼ぐという仮定は上限側に甘い（協同性があればもっと少ない
       分子数で済むし、内部歪みがあればもっと必要）。オーダーの議論に留めること。
    """
    if rho_n is None:
        rho_n = par.rho_n
    dG = model.monomer_curvature_energy(par, lp=lp, length=length)
    return np.log(rho_n) / dG, dG


def analytic_clu(par=Par, n=801):
    """膜拡散もクラスターの再配置もない局所平衡での、クラスターの極濃縮の上限。

    N ∝ 1/k_off_n ∝ w_n の局所平衡なので、形は Cycle 1 と同じ:
        rho_n / (rho_n·a + 1 - a)
    ★ ただし Cycle 1 と違い、クラスターは動かないので数値解がこれを
       下回るとは限らない（単量体 M の側の平滑化を経由するだけ）。
    """
    g = geometry(n, par)
    a = g["A_cap"].sum() / g["A"].sum()
    return par.rho_n / (par.rho_n * a + 1 - a), a


def critical_seed(par=Par, n=101, seeds=(2, 5, 10, 30, 100, 300, 1000),
                  rel=0.01):
    """非対称な枝に落とすのに必要な「片極の種」の大きさを返す。

    seed = 片極のクラスター密度を一様値の何倍に積むか。
    双安定でなければ（どの種でも対称に戻れば）np.inf を返す。

    ★ 固定の種で「双安定でない」と結論してはいけない。双安定な系には
      臨界の種があり、それより小さい種は対称な枝に吸い込まれる。
      p=2 を seed=10 だけで見て「単安定」と誤判定した（実際は違った）。

    戻り値: (臨界の種, その種での左右比)。見つからなければ (inf, 1.0)。
    """
    try:
        base = steady(n, par, ic="uniform")["asym_clu"]
    except RuntimeError:
        return np.inf, np.nan
    for sv in seeds:
        try:
            a = steady(n, par, ic="seed_pole", seed=float(sv))["asym_clu"]
        except RuntimeError:
            continue
        if abs(a - base) > rel * max(1.0, abs(base)):
            return float(sv), a
    return np.inf, base


def bistability_scan(p_values, n=101, par=Par, seeds=(2, 5, 10, 30, 100, 300, 1000)):
    """p を振って、初期条件の違いが定常に残るか（双安定か）を調べる。

    各 p について critical_seed() で種を振り、非対称な枝が存在するかを見る。

    ★ 本プロジェクトの中心的な問いに直結する:
      単安定なら PopZ は「効かせ続ける」必要がある（持続的な摂動）。
      双安定なら「一度傾ければ固定される」（一過的な誘導で足りる）。

    ⚠️ 「見つからなかった = 双安定でない」の証明にはならない。種の列を
      広げること、境界の近くでは latch_test() と時間発展でも確かめること。

    戻り値: 行ごとに (p, 臨界の種, 非対称な枝での左右比, 双安定か)
    """
    rows = []
    for p in p_values:
        sv, a = critical_seed(derive(par, p=p), n=n, seeds=seeds)
        rows.append((p, sv, a, np.isfinite(sv)))
    return rows


def latch_test(theta=0.9, t_release=600.0, n=101, par=Par, T=None):
    """PopZ を一過的にかけて外したあと、非対称が残るかを調べる。

    theta:     左極の結合能を (1-theta) 倍に落とす（PopZ の空間占有）
    t_release: この時刻に PopZ を外す
    T:         外したあとさらに積む時間

    ★ 本プロジェクトの実験計画に直結する:
      最終比が 1 に戻る  → PopZ は出し続けないといけない（持続発現）
      最終比が残る       → 一過的な誘導で足りる（ラッチ）

    実装は 2 段階（設計判断 [C10]）。摂動をかけて t_release まで積み、
    その終状態を初期値にして摂動なしで積み直す。RHS の中で時刻を見て
    切り替えると不連続になり BDF が止まる。

    戻り値: (外す直前の左右比, 最終の左右比)
    左右比は「塞いだ左極 / 無摂動の右極」。1 未満なら左が負けている。
    """
    if T is None:
        T = par.T
    g = geometry2(n, par)
    block = 1.0 - theta * (g["A_capL"] / g["A"])   # 面積重み（設計判断 [8]）
    held = solve(n, par, T=t_release, ic="uniform", n_out=4, block=block)
    y = np.concatenate([held["C"], held["M"], held["Nf"]])
    rel = solve(n, par, T=T, n_out=40, y0=y)
    return held["asym_clu"], rel["asym_clu"]


def sustained_asymmetry(theta=0.9, n=101, par=Par):
    """PopZ をかけ続けたときの定常の左右比（塞いだ左 / 無摂動の右）。"""
    g = geometry2(n, par)
    block = 1.0 - theta * (g["A_capL"] / g["A"])
    return steady(n, par, block=block)["asym_clu"]


def rise_time(r, frac=0.632, key="enr_clu_t"):
    """定常値への偏差が (1-frac) まで減る時刻。model.rise_time のクラスター版。"""
    e = np.asarray(r[key], float)
    t = np.asarray(r["t"], float)
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


# =============================================================================
def summary(par=Par, n=101, verbose=True):
    """冒頭「Cycle 2 で出た結論」の数値をすべて再計算して印字する。

    ⚠️ 冒頭の記述と食い違ったら、記述のほうが古い。この出力が唯一の正。
    実行 1 分ほど。
    """
    out = []

    def say(s):
        out.append(s)

    say("=" * 74)
    say("Cycle 2 SUMMARY")
    say("=" * 74)

    r = steady(n, par)
    Q, dG = required_cluster_size(par.rho_n, par)
    say("\n(1) 曲率エネルギーのギャップ")
    say(f"    単量体 1 個の曲率ミスマッチ        {dG:.5f} kT")
    say(f"    Cycle 1 が単量体に要求した ln rho  {np.log(model.Par.rho):.4f} kT"
        f"  = 単量体の {np.log(model.Par.rho)/dG:.0f} 倍")
    say(f"    Cycle 2 がクラスターに要求 ln rho_n {np.log(par.rho_n):.4f} kT"
        f"  = {Q:.0f} 分子ぶん")
    say(f"    実測の平均クラスターサイズ 47 分子との比 {Q/47:.1f}"
        f"（最大 1198 分子）")

    say("\n(2) 大腸菌での【予測】（rho_n を B. subtilis から移植）")
    say(f"    クラスター分子のうち極にいる割合   {r['clu_pole_frac']:.4f}")
    say(f"    クラスターの極濃縮                 {r['enr_clu']:.4f}")
    say(f"    膜全体（単量体 + クラスター）      {r['enrichment']:.4f}")
    say(f"    3 相分配 cyt/mono/clu              "
        f"{r['f_cyt']:.3f} / {r['f_mono']:.3f} / {r['f_clu']:.3f}")
    say(f"    k_grow（f_clu に校正）             {r['k_grow']:.4e}")

    say("\n(3) PopZ の摂動（p = %.2f）" % par.p)
    say(f"    {'theta':>7} {'持続: 左右比':>14} {'一過的(600s)後':>16}")
    for th in (0.5, 0.9, 0.98):
        a = sustained_asymmetry(th, n=n, par=par)
        _, f_ = latch_test(theta=th, t_release=600.0, n=n, par=par, T=1e5)
        say(f"    {th:7.2f} {1/a:13.1f}x {f_:16.6g}")
    say("    ※ 一過的の列が 1 なら固定されない（PopZ の持続発現が要る）")

    say("\n(4) 成長の次数 p と双安定性")
    say(f"    {'p':>5} {'臨界の種':>10} {'非対称枝の左右比':>18} {'双安定':>8}")
    for row in bistability_scan([0.5, 0.8, 1.0, 1.1, 1.2, 1.5], n=71, par=par):
        say(f"    {row[0]:5.2f} {row[1]:10.4g} {row[2]:18.6g} {str(row[3]):>8}")

    say("\n(5) 発現量 N")
    e_lo = steady(n, derive(par, N=169.0))["enr_clu"]
    e_hi = steady(n, derive(par, N=169000.0))["enr_clu"]
    say(f"    k_grow を校正し続けると N=169 と N=169000 で "
        f"{e_lo:.8f} / {e_hi:.8f}（不変）")
    kg2 = _kg(derive(par, p=2.0))
    say(f"    k_grow を p=2 の校正値 {kg2:.3e} に固定して N を振る:")
    for N in (1690.0, 5000.0, 16900.0, 169000.0):
        sv, a = critical_seed(derive(par, p=2.0, N=N, k_grow=kg2), n=71)
        say(f"      N={N:8.0f}  臨界の種 {sv:8.4g}  非対称枝 左右比 {a:12.6g}")

    say("=" * 74)
    if verbose:
        print("\n".join(out))
    return "\n".join(out)


def validate(par=Par, verbose=True):
    """本計算の前に必ず通すこと。False が返ったら結果を信用しない。

    実行 30 秒ほど。Cycle 1 の validate() も中で呼ぶ（V0）。
    """
    out, ok = [], True

    def say(s):
        out.append(s)

    def chk(cond, label, detail):
        nonlocal ok
        ok &= bool(cond)
        say(f"  {label} {detail} {'OK' if cond else '★NG'}")

    say("=" * 76)
    say("VALIDATION — Cycle 2（クラスター）")
    say("=" * 76)

    fr = fractions(par)
    say(f"  分配比（原典）  {par.f_cyt}/{par.f_mono}/{par.f_clu}"
        f"  和 = {par.f_cyt + par.f_mono + par.f_clu:.3f}（丸めで 1 を超える）")
    say(f"  分配比（正規化）{fr[0]:.4f}/{fr[1]:.4f}/{fr[2]:.4f}")

    # --- V0: 幾何・演算子・保存形は Cycle 1 のものをそのまま使う
    say("\n  -- V0 Cycle 1 の検証を先に通す（幾何・演算子・保存形） --")
    chk(model.validate(model.Par, verbose=False), "V0", "model.validate()")

    # --- V1: クラスターの曲率重み w_n の端点
    g = geometry2(401, par)
    w_side, w_pole = g["w_n"][401 // 2], g["w_n"][0]
    chk(abs(w_side - 1.0) < 1e-6, "V1", f"側壁の w_n = {w_side:.6f}（期待 1.0）")
    chk(w_pole > 0.9 * par.rho_n, "V1",
        f"極の w_n = {w_pole:.4f}（rho_n = {par.rho_n} に近いこと）")

    # --- V2: 質量保存。唯一の防波堤（設計判断 [C3]）
    r = solve(101, par, ic="uniform")
    chk(r["mass_err"] < 1e-9, "V2",
        f"質量保存 誤差 = {r['mass_err']:.1e}"
        + ("" if r["mass_err"] < 1e-9 else "（A/V の使い方と f_nm の分岐を疑え）"))

    # --- V3: クラスター収支 ∫G dA = ∫k_off_n·N dA（定常で）
    rs = steady(101, par)
    chk(rs["budget_err"] < 1e-6, "V3",
        f"クラスター収支 誤差 = {rs['budget_err']:.1e}"
        f"（定常残差 {rs['steady_residual']:.1e}）")

    # --- V4: 希釈 mu を入れても総量が N のまま（設計判断 [C3]）
    rmu = solve(101, variant(mu=np.log(2) / 1500), ic="uniform")
    chk(rmu["mass_err"] < 1e-8, "V4",
        f"mu = ln2/1500 でも質量保存 誤差 = {rmu['mass_err']:.1e}")

    # --- V5: rho = rho_n = 1 なら完全一様
    a_cap = g["A_cap"].sum() / g["A"].sum()
    r1 = steady(101, variant(rho=1.0, rho_n=1.0))
    good = (abs(r1["enrichment"] - 1.0) < 1e-5
            and abs(r1["enr_clu"] - 1.0) < 1e-5
            and abs(r1["clu_pole_frac"] - a_cap) < 1e-5
            and r1["cyto_var"] < 1e-6)
    chk(good, "V5",
        f"rho=rho_n=1: 膜濃縮 {r1['enrichment']:.6f}, クラスター濃縮 "
        f"{r1['enr_clu']:.6f}, 極割合 {r1['clu_pole_frac']:.6f}"
        f"（期待は面積比 {a_cap:.6f}）, 細胞質変動 {r1['cyto_var']:.1e}")

    # --- V6: 格子収束。クラスターは拡散しないので Cycle 1 より厳しい [C2]
    es = [steady(k, par)["enr_clu"] for k in (51, 101, 201, 401)]
    d = max(abs(np.diff(es))) / es[-1]
    chk(d < 0.02, "V6",
        "格子収束 n=51..401: " + " -> ".join(f"{e:.4f}" for e in es)
        + f"  最大変化 {d:.2%}")

    # --- V7: ★ Cycle 1 への回帰。クラスターを殺すと Cycle 1 に一致するか
    p1 = variant(k_nuc=0.0, k_grow=0.0, rho=model.Par.rho, rho_n=1.0,
                 f_cyt=model.Par.f_cyt, f_mono=1 - model.Par.f_cyt, f_clu=0.0)
    r7 = solve(101, p1, T=20.0, ic="uniform")
    e1 = model.solve(101, model.Par, T=20.0)["enrichment"]
    rel7 = abs(r7["enrichment"] - e1) / e1
    chk(rel7 < 1e-6, "V7",
        f"Cycle 1 回帰: {r7['enrichment']:.8f} vs {e1:.8f}（相対差 {rel7:.1e}）")

    # --- V8: 解析ヤコビアンを差分と突き合わせる（[C7]。ニュートン法の前提）
    nj = 21
    asm = _assemble(nj, par)
    yj = initial(nj, par, "uniform") * (1 + 0.05 * np.sin(np.arange(3 * nj)))
    Ja = asm["jac"](yj).toarray()
    f0 = asm["rhs"](yj)
    Jn = np.zeros_like(Ja)
    for j in range(3 * nj):
        hj = 1e-6 * max(abs(yj[j]), 1e-6)
        yp = yj.copy()
        yp[j] += hj
        Jn[:, j] = (asm["rhs"](yp) - f0) / hj
    rel8 = np.abs(Ja - Jn).max() / np.abs(Jn).max()
    chk(rel8 < 1e-6, "V8", f"解析ヤコビアン vs 差分 相対差 = {rel8:.1e}")

    # --- V9: ★ steady() と長時間積分が一致するか（[C8] の裏取り）
    rt = solve(101, par, T=1e5, ic="uniform", n_out=5)
    rel9 = abs(rt["enr_clu"] - rs["enr_clu"]) / rs["enr_clu"]
    chk(rel9 < 1e-4, "V9",
        f"steady() {rs['enr_clu']:.6f} vs 積分 T=1e5 {rt['enr_clu']:.6f}"
        f"（相対差 {rel9:.1e}、積分側の残差 {rt['residual']:.1e}）")

    # --- V10: well-mixed の k_grow は rho_n=1 でのみ正しい（[C5] / _kg）
    kg_wm = k_grow_wellmixed(par)
    kg_1 = calibrate_k_grow(par=variant(rho_n=1.0))
    rel10 = abs(kg_wm - kg_1) / kg_1
    chk(rel10 < 1e-3, "V10",
        f"rho_n=1: well-mixed {kg_wm:.6e} vs PDE {kg_1:.6e}"
        f"（相対差 {rel10:.1e}）")
    kg_n = _kg(par)
    say(f"       ⚠️ rho_n={par.rho_n}（anchor={par.k_off_n_anchor}）では "
        f"PDE 校正 {kg_n:.4e} = well-mixed の {kg_n/kg_wm:.2f} 倍。"
        f"曲率を k_off_n に入れた時点で well-mixed は使えない")
    say(f"       PARAMETERS §4 の記載値 3.21e-4 と well-mixed の比 = "
        f"{kg_wm/3.21e-4:.3f}")

    # --- V11: ★ k_grow を f_clu に校正する限り N は構造的に効かない
    #     (C,M,N) -> λ(C,M,N)、k_grow -> λ^(-p)·k_grow で方程式が厳密に不変。
    #     Cycle 1 で N が 10 桁まで効かなかったのと同じ構造がここでも成り立つ。
    e_lo = steady(101, variant(N=169.0))["enr_clu"]
    e_hi = steady(101, variant(N=169000.0))["enr_clu"]
    rel11 = abs(e_lo - e_hi) / e_hi
    chk(rel11 < 1e-6, "V11",
        f"N を 1000 倍振ってもクラスター濃縮は不変: "
        f"{e_lo:.8f} vs {e_hi:.8f}（相対差 {rel11:.1e}）")
    say("       ★ ただし k_grow を分子定数として固定すると N は効く。"
        "どちらを採るかは仮定であって、データでは決まらない")

    # --- V12: p < 1 は単安定、p > 1 は双安定（成長の次数が対称性を決める）
    mono = critical_seed(derive(par, p=0.5), n=71)[0]
    bist = critical_seed(derive(par, p=1.2), n=71)[0]
    chk(not np.isfinite(mono) and np.isfinite(bist), "V12",
        f"p=0.5 の臨界の種 = {mono}（単安定を期待）, "
        f"p=1.2 = {bist}（双安定を期待）")

    # --- V13: 校正した rho_n が「校正可能な上限」から十分離れているか
    ceil = rho_n_ceiling(par)
    hit_cap = ceil >= 1e6 - 1
    chk(par.rho_n < 0.2 * ceil, "V13",
        f"rho_n = {par.rho_n} < 校正可能な上限 "
        + ("（anchor='pole' では極の k_off_n が rho_n に依らないので上限は"
           "実質的に無い。探索上限 1e6 まで到達せず）"
           if hit_cap else
           f"{ceil:.0f}（その {par.rho_n/ceil:.1%}）")
        + "。核形成だけで f_clu を超える手前にいること")

    say("=" * 76)
    say("すべて合格" if ok else "★ 不合格あり。本計算に進まないこと")
    say("=" * 76)

    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys

    ok = validate()
    if ok and "--summary" in sys.argv:
        print()
        summary()
    sys.exit(0 if ok else 1)


# =============================================================================
# 将来の拡張（本ファイルでは意図的に外してある）
#
#   PopZ の空間占有（Cycle 3 S1）: block 引数で既に入る。
#       g = geometry2(n); block = 1 - theta * (g["A_capL"] / g["A"])
#       solve(n, par, block=block)
#       ⚠️ 面積重みで入れること（設計判断 [8]）。セル中心判定は非単調になる。
#       ⚠️ 「V から球の断面積を引く」のは誤り（model.py 末尾と同じ理由）。
#
#   PopZ による局所分解（Cycle 3 S2）: 片極だけ k_off_n を上げる。
#       k_off_n_of() の戻り値に (1 + gamma * A_capL/A) を掛ける。
#       ★ 分解は分子を系から取り除くので閉じた系でなくなる。
#         質量保存の検算（V2）が効かなくなるので、先に収支検算（V3）を
#         「流入 = 流出 + 分解」の形に書き換えてから実装すること。
#         防波堤を外したまま進まないこと。
#
#   クラスターのサイズ分布: N を単一の場ではなくサイズ別の場の列にする。
#       未知数が 3n から (2+K)n に増えるので _sparsity() を書き直す。
#       平均 47 / 最大 1198 分子という広い分布（PARAMETERS §3）を
#       表現したいときに必要。
#
#   隔壁（セプタム）: model.py 末尾を参照。geometry() の r(zf) にくびれ。
#       ★ 校正の目標 66% は「極 + 隔壁」なので、隔壁を入れて初めて
#         この値と正面から比較できるようになる。
