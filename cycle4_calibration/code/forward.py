"""前向きモデル — モデルの面密度から、広視野顕微鏡の線プロファイルを合成する。

    Cycle 5 で新設。これが無いと dry と wet は比べられない。
    モデルが出すのは面密度 C(z) [分子/µm³], M(z), N(z) [分子/µm²]、
    wet が出すのは投影された線プロファイル [ADU]。別の量である。

    python forward.py       # 検証だけ走る（本計算の前に必ず通すこと）

=============================================================================
  Cycle 5 の設計判断（前向きモデル）
=============================================================================

[F1] ★ x 方向の積分は解析形で行う。格子で刻まない。

     膜の投影は細胞の縁で 1/sqrt(1-(x/r)²) の可積分特異点を持つ。
     格子で刻むと縁の画素だけ値が発散し、線幅を変えるたび結果が飛ぶ。
     画素 [x1,x2] への寄与を閉じた形で積分する:

       細胞質 (濃度 c):  c·[ x·sqrt(r²-x²) + r²·arcsin(x/r) ]
       膜     (面密度 σ): σ·2·sqrt(1+r'²)·r·[ arcsin(x/r) ]

     ⚠️ 面積重みの話（Cycle 1 設計判断 [2]）と同じ理由。境界を点で判定すると
       格子依存が非単調になる。

[F2] ★ 一様な膜タンパクは、幅方向に全部足すと長軸に沿って完全に平らになる。
     しかし**細い線で中心だけ見ると、極のほうが明るく出る。**

     全幅: キャップの膜面積は dz あたり 2πR（アルキメデス）、円筒も 2πR。
           だから全幅で積分すると z に依らず 2πRσ で平ら。
     中心線 (x=0): 投影密度は σ·2·sqrt(1+r'²) = 2σ·R/r(z)。
           キャップでは r < R なので **1 より大きい倍率で明るくなる**
           （先端では発散し、PSF と線幅で丸められる）。
           全幅で平らなのは、キャップが細いぶん幅方向の積分範囲が狭いから。

     ⚠️ ここは一度逆に予想して検定を落とした。細い線では
       **一様な膜タンパクでも極が明るい**。これが「膜タンパクの投影
       アーティファクト」の正体で、線幅が効く。wet の線幅（3 px）を
       必ず合わせること。

[F3] ★ 細胞質 null が幾何の検定になる。

     遊離 GFP / mCherry は一様な細胞質種なので、投影は R/L だけで決まる。
     実測の end_mid（mCherry 0.524 / GFP 0.581）を再現できるかで
     幾何と PSF の合否が決まり、合わせにいけば R が逆算できる。
     `NEXT_CYCLE3_WET.md` §0.2 が「R は実測できない」とした理由
     （線形な撮像と一様マーカーが無い）は、null では両方満たされている。

[F4] ★ 指標は wet の metrics.py と同じ定義・同じ窓で計算する。
     POLE (0.15,0.35)+(0.65,0.85) / MID (0.40,0.60) / END (0,0.15)+(0.85,1)、
     背景は線の外側 12% の中央値、負はクリップ、5 点移動平均。
     ⚠️ 定義を 1 つでも変えると wet の数字と比べられなくなる。

[F5] ★ 画素化と PSF の順序: 投影 → PSF 畳み込み → 画素化 → 線幅で平均。
     逆にすると PSF が画素より小さいとき形が変わる。

[F6] 焦点深度は無視する（全投影）。
     NA 0.75 の焦点深度は約 1.2 µm で細胞径 0.8 µm より大きいので、
     広視野では細胞全体が実質的に投影される。
     ⚠️ 共焦点のデータには使えない（GFP_REPORT の注意と同じ）。
"""

import numpy as np

PX_UM = 0.1625          # µm/px（wet と共通。確定値）
TAIL = 0.12             # 背景に使う線の端の割合（wet の metrics.py と同じ）
POLE = [(0.15, 0.35), (0.65, 0.85)]
MID = (0.40, 0.60)
END = [(0.0, 0.15), (0.85, 1.0)]
HALF_A, HALF_B = (0.0, 0.35), (0.65, 1.0)


# =============================================================================
#  幾何
# =============================================================================
def radius(z, R, L):
    """スフェロシリンダーの半径 r(z) と dr/dz。z は 0..L。外側は 0。"""
    z = np.asarray(z, float)
    Lcyl = L - 2 * R
    r = np.zeros_like(z)
    dr = np.zeros_like(z)
    cap_l = (z >= 0) & (z < R)
    cyl = (z >= R) & (z <= R + Lcyl)
    cap_r = (z > R + Lcyl) & (z <= L)
    for m, c in ((cap_l, R), (cap_r, R + Lcyl)):
        u = z[m] - c
        r[m] = np.sqrt(np.maximum(R * R - u * u, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            dr[m] = np.where(r[m] > 1e-12, -u / np.maximum(r[m], 1e-12), 0.0)
    r[cyl] = R
    dr[cyl] = 0.0
    return r, dr


def _ix_cyt(x, r):
    """∫ 2·sqrt(r²-x²) dx の原始関数（設計判断 [F1]）。"""
    xc = np.clip(x, -r, r)
    return xc * np.sqrt(np.maximum(r * r - xc * xc, 0.0)) + r * r * np.arcsin(
        np.divide(xc, r, out=np.zeros_like(xc), where=r > 1e-12))


def _ix_mem(x, r):
    """∫ dx/sqrt(1-(x/r)²) の原始関数（設計判断 [F1]）。

    ⚠️ ここに前置きの 2（表裏の膜）を入れないこと。呼ぶ側の
      w = 2·sqrt(1+r'²) が持っている。一度 2 を二重に掛けて
      F1（投影の全積分 = 膜面積）が 2 倍で落ちた。
    """
    xc = np.clip(x, -r, r)
    return r * np.arcsin(
        np.divide(xc, r, out=np.zeros_like(xc), where=r > 1e-12))


# =============================================================================
#  投影
# =============================================================================
def project(z_grid, x_edges, par_R, par_L, c_of_z=None, s_of_z=None):
    """投影像 [分子 / (µm in z) / (µm in x)] を (len(z_grid), len(x_edges)-1) で返す。

    c_of_z: 細胞質の濃度 [分子/µm³]（z_grid 上の配列 or None）
    s_of_z: 膜の面密度 [分子/µm²]（同上）
    x 方向は画素区間で解析積分する（設計判断 [F1]）。
    """
    r, dr = radius(z_grid, par_R, par_L)
    r = r[:, None]
    dx = np.diff(x_edges)[None, :]
    x1 = x_edges[None, :-1]
    x2 = x_edges[None, 1:]
    out = np.zeros((len(z_grid), len(x_edges) - 1))
    if c_of_z is not None:
        c = np.asarray(c_of_z, float)[:, None]
        out += c * (_ix_cyt(x2, r) - _ix_cyt(x1, r)) / dx
    if s_of_z is not None:
        s = np.asarray(s_of_z, float)[:, None]
        w = (2.0 * np.sqrt(1.0 + dr * dr))[:, None]
        out += s * w * (_ix_mem(x2, r) - _ix_mem(x1, r)) / dx
    return out


def _gauss_kernel(sigma_px, trunc=4.0):
    if sigma_px <= 0:
        return np.array([1.0])
    rad = max(1, int(np.ceil(trunc * sigma_px)))
    t = np.arange(-rad, rad + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma_px) ** 2)
    return k / k.sum()


def line_profile(C=None, M=None, N=None, z_model=None, R=0.4, L=3.0,
                 psf_fwhm_um=0.40, line_w_px=3, margin_um=4.0,
                 nz_fine=12, offset=0.0, noise=0.0, rng=None):
    """モデルの場から、wet と同じ形の線プロファイルを返す。

    C, M, N: z_model 上の濃度・面密度。None は 0 扱い。
             観測される蛍光 = 細胞質 C + 膜 (M + N)。
    psf_fwhm_um: PSF の半値全幅。sigma = FWHM / 2.355
    line_w_px:   ImageJ の線幅（w 本の平行線を平均。w=3 なら -1,0,+1 px）
    margin_um:   細胞の外に取る余白（背景を測るため。wet の線も余白がある）
    offset:      カメラオフセット相当の定数（背景推定の検定用）
    noise:       画素ノイズの SD（0 なら無し）

    戻り値 dict: pos_um（線上の位置、細胞の左端が 0）, val, z（0..1 に正規化）
    """
    # ★ 画素格子は細胞の中心に対して厳密に対称にする。
    #   そうしないと左右の窓に入る点がずれ、対称な入力でも asym が 0 にならない
    #   （実測 1.9e-2。これは wet でも起きるが、検定では切り分けたい）。
    npx = int(round((L + 2 * margin_um) / PX_UM))
    pos = L / 2 + (np.arange(npx) - (npx - 1) / 2.0) * PX_UM

    # --- z 方向は画素内をさらに細かく刻んで平均（設計判断 [F5] の前段）
    sub = (np.arange(nz_fine) + 0.5) / nz_fine - 0.5
    zf = (pos[:, None] + sub[None, :] * PX_UM).ravel()
    inside = (zf >= 0) & (zf <= L)

    def interp(f):
        if f is None:
            return None
        v = np.zeros_like(zf)
        v[inside] = np.interp(zf[inside], z_model, f)
        return v

    # --- x 方向は画素を十分広く取る（PSF の裾が入るように）
    nx = int(np.ceil((2 * R + 6 * psf_fwhm_um) / PX_UM / 2)) * 2 + 1
    xe = (np.arange(nx + 1) - nx / 2.0) * PX_UM

    mem = None
    if M is not None or N is not None:
        mem = (interp(M) if M is not None else 0.0) + \
              (interp(N) if N is not None else 0.0)
    img = project(zf, xe, R, L, c_of_z=interp(C), s_of_z=mem)
    img = img.reshape(npx, nz_fine, nx).mean(axis=1)        # z 画素内で平均

    # --- PSF（分離可能ガウス）を z と x に掛ける（設計判断 [F5]）
    sig_px = psf_fwhm_um / 2.3548 / PX_UM
    k = _gauss_kernel(sig_px)
    if len(k) > 1:
        pad = len(k) // 2
        img = np.apply_along_axis(
            lambda v: np.convolve(np.r_[np.zeros(pad), v, np.zeros(pad)],
                                  k, mode="same")[pad:-pad], 0, img)
        img = np.apply_along_axis(
            lambda v: np.convolve(np.r_[np.zeros(pad), v, np.zeros(pad)],
                                  k, mode="same")[pad:-pad], 1, img)

    # --- 線幅 w の平均（ImageJ と同じく w 本の平行線）
    c0 = nx // 2
    off = np.arange(line_w_px) - (line_w_px - 1) // 2
    val = img[:, c0 + off].mean(axis=1) + offset
    if noise > 0:
        rng = np.random.default_rng() if rng is None else rng
        val = val + rng.normal(0.0, noise, size=val.shape)
    return dict(pos_um=pos, val=val, z=pos / L, L=L, R=R)


# =============================================================================
#  wet と同じ指標（設計判断 [F4]。anl/scripts/metrics.py と同一定義）
# =============================================================================
def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _region(z, lo, hi):
    return (z >= lo) & (z <= hi)


def shape_metrics(v, z, cell_len_um=None):
    """wet の metrics.shape_metrics と同じ定義。v は生の線プロファイル。"""
    v = np.asarray(v, float)
    n = len(v)
    k = max(2, int(round(n * TAIL)))
    t = np.r_[v[:k], v[-k:]]
    bg, sd = np.median(t), _mad(t)
    s = np.clip(v - bg, 0, None)
    sm = np.convolve(s, np.ones(5) / 5, mode="same")
    inc = _region(z, 0, 1)
    pa = np.r_[s[_region(z, *POLE[0])], s[_region(z, *POLE[1])]]
    mm = s[_region(z, *MID)]
    en = np.r_[s[_region(z, *END[0])], s[_region(z, *END[1])]]
    IA = s[_region(z, *HALF_A)].mean()
    IB = s[_region(z, *HALF_B)].mean()
    si = np.sort(s[inc])[::-1]
    kk = max(1, int(round(0.2 * len(si))))
    out = dict(
        bg=bg, noise=sd,
        pole_mid=pa.mean() / mm.mean() if mm.mean() > 0 else np.nan,
        end_mid=en.mean() / mm.mean() if mm.mean() > 0 else np.nan,
        endA_mid=s[_region(z, *END[0])].mean() / mm.mean() if mm.mean() > 0 else np.nan,
        endB_mid=s[_region(z, *END[1])].mean() / mm.mean() if mm.mean() > 0 else np.nan,
        asym=(IA - IB) / (IA + IB) if IA + IB > 0 else np.nan,
        f20=si[:kk].sum() / si.sum() if si.sum() > 0 else np.nan,
        snr_prof=sm.max() / sd if sd > 0 else np.inf,
        total=s[inc].sum(),
    )
    return out


def metrics_of(C=None, M=None, N=None, z_model=None, **kw):
    """モデルの場 → 線プロファイル → wet と同じ指標。"""
    pr = line_profile(C=C, M=M, N=N, z_model=z_model, **kw)
    m = shape_metrics(pr["val"], pr["z"])
    m["profile"] = pr
    return m


def uniform_metrics(kind="cyt", R=0.4, L=3.0, **kw):
    """一様な種（細胞質 or 膜）の指標。wet の null に対応する基準線。"""
    zm = np.linspace(0, L, 2001)
    one = np.ones_like(zm)
    if kind == "cyt":
        return metrics_of(C=one, z_model=zm, R=R, L=L, **kw)
    if kind == "mem":
        return metrics_of(M=one, z_model=zm, R=R, L=L, **kw)
    raise ValueError(kind)


# =============================================================================
def validate(verbose=True):
    """本計算の前に必ず通すこと。"""
    out, ok = [], True

    def say(s):
        out.append(s)

    def chk(cond, label, detail):
        nonlocal ok
        ok &= bool(cond)
        say(f"  {label} {detail} {'OK' if cond else '★NG'}")

    say("=" * 76)
    say("VALIDATION — Cycle 5 前向きモデル")
    say("=" * 76)

    R, L = 0.4, 3.0

    # --- F1: 幾何の整合。投影を全幅・全長で積分すると体積・面積になる
    zf = np.linspace(0, L, 4001)
    xe = np.linspace(-R, R, 2001)
    one = np.ones_like(zf)
    Pc = project(zf, xe, R, L, c_of_z=one)
    Pm = project(zf, xe, R, L, s_of_z=one)
    dz = zf[1] - zf[0]
    dx = np.diff(xe)[0]
    V_num, A_num = Pc.sum() * dz * dx, Pm.sum() * dz * dx
    Lcyl = L - 2 * R
    V_ex = np.pi * R**2 * Lcyl + 4 / 3 * np.pi * R**3
    A_ex = 2 * np.pi * R * Lcyl + 4 * np.pi * R**2
    for lab, got, exp in (("体積", V_num, V_ex), ("膜面積", A_num, A_ex)):
        e = abs(got - exp) / exp
        chk(e < 2e-3, "F1", f"投影の全積分 = {lab} {got:.4f}（解析 {exp:.4f}、"
                            f"誤差 {e:.1e}）")

    # --- F2: 一様な膜は全幅で足すと z に平ら、全幅で足した細胞質は r² に比例
    Pm_line = Pm.sum(axis=1) * dx
    mid = (zf > R) & (zf < L - R)
    cap = (zf > 0.05 * R) & (zf < 0.95 * R)
    flat = abs(Pm_line[cap].mean() / Pm_line[mid].mean() - 1.0)
    chk(flat < 5e-3, "F2",
        f"一様な膜を全幅で積分: キャップ/円筒 = "
        f"{Pm_line[cap].mean()/Pm_line[mid].mean():.5f}（期待 1）")
    Pc_line = Pc.sum(axis=1) * dx
    r_, _ = radius(zf, R, L)
    rel = np.abs(Pc_line - np.pi * r_**2).max() / (np.pi * R**2)
    chk(rel < 2e-3, "F2", f"一様な細胞質を全幅で積分 = π·r(z)²（最大差 {rel:.1e}）")

    # --- F3: PSF なし・線幅 1 の極限で、細胞質の中心線は 2·sqrt(r²) に比例
    pr = line_profile(C=np.ones(2001), z_model=np.linspace(0, L, 2001),
                      R=R, L=L, psf_fwhm_um=0.0, line_w_px=1, nz_fine=24)
    ins = (pr["pos_um"] > R) & (pr["pos_um"] < L - R)
    chk(abs(pr["val"][ins].mean() / (2 * R) - 1.0) < 2e-2, "F3",
        f"PSF 無し・幅 1 px の円筒部 = {pr['val'][ins].mean():.4f}"
        f"（期待 2R = {2*R:.4f}）")

    # --- F4: 線の外は厳密に 0（背景推定が offset を正しく拾う）
    pr2 = line_profile(C=np.ones(2001), z_model=np.linspace(0, L, 2001),
                       R=R, L=L, psf_fwhm_um=0.40, offset=100.0)
    m = shape_metrics(pr2["val"], pr2["z"])
    chk(abs(m["bg"] - 100.0) < 1e-6, "F4",
        f"背景推定 = {m['bg']:.6f}（与えた offset 100）")

    # --- F5: 指標が offset と総量のスケールに不変（比なので）
    pr3 = line_profile(C=np.ones(2001) * 7.0, z_model=np.linspace(0, L, 2001),
                       R=R, L=L, psf_fwhm_um=0.40, offset=-50.0)
    m3 = shape_metrics(pr3["val"], pr3["z"])
    d = max(abs(m3[k] - m[k]) for k in ("end_mid", "pole_mid", "f20"))
    chk(d < 1e-9, "F5", f"指標がスケール・オフセットに不変（最大差 {d:.1e}）")

    # --- F6: 左右対称な入力では asym = 0
    chk(abs(m["asym"]) < 1e-9, "F6", f"対称な細胞質の asym = {m['asym']:.2e}")

    # --- F7: ★ 細い線ほど一様な膜の端/中央が高く出る（設計判断 [F2]）
    e3 = uniform_metrics("mem", R, L, psf_fwhm_um=0.40, line_w_px=3)["end_mid"]
    e9 = uniform_metrics("mem", R, L, psf_fwhm_um=0.40, line_w_px=9)["end_mid"]
    chk(e3 > e9, "F7", f"一様な膜の end_mid: 幅 3 px {e3:.4f} > 幅 9 px {e9:.4f}"
                       f"（中心線は極で 2σR/r に増える。全幅なら平ら）")

    # --- F8: ★ 一様な膜 > 一様な細胞質（膜タンパクの基準線は細胞質より高い）
    c3 = uniform_metrics("cyt", R, L, psf_fwhm_um=0.40, line_w_px=3)["end_mid"]
    chk(e3 > c3, "F8",
        f"一様な膜 {e3:.4f} > 一様な細胞質 {c3:.4f} "
        f"→ 細胞質 null を膜タンパクの基準にすると濃縮を過大評価する")

    # --- F9: 格子（nz_fine）収束
    vals = [uniform_metrics("cyt", R, L, psf_fwhm_um=0.40, nz_fine=k)["end_mid"]
            for k in (4, 8, 16, 32)]
    d9 = max(abs(np.diff(vals))) / vals[-1]
    chk(d9 < 2e-3, "F9", "nz_fine 収束 4..32: "
        + " -> ".join(f"{v:.5f}" for v in vals) + f"  最大変化 {d9:.2%}")

    say("=" * 76)
    say("すべて合格" if ok else "★ 不合格あり。本計算に進まないこと")
    say("=" * 76)
    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
