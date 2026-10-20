"""
DivIVA 局在モデル — Cycle 2（単量体 + クラスター）

★ これはプロジェクト正本 `claude/cluster.py` の作業用コピーである。
  コードは正本と同一（docstring だけ要点に絞ってある）。
  設計判断 [C1]〜[C12] の全文は正本を見ること。

    [C1]  曲率は k_on ではなく k_off_n（クラスターの寿命）に入れる
    [C2]  クラスターは動かない（D_n = 0）
    [C3]  閉じた系を保つ（mu = 0 が既定）。質量保存が唯一の防波堤
    [C4]  N^p は p<1 で特異点を持つので平滑化する
    [C5]  k_on の校正の目標は f_mono（膜 = 単量体 + クラスター）
    [C6]  時間スケールが 2 桁伸びる（既定 T = 3000 s）
    [C7]  解析ヤコビアンを自分で書いて渡す
    [C8]  定常は短い助走 + 正値性を保つニュートン法
    [C9]  収束判定は無次元残差
    [C10] 一過的摂動は 2 段階の積分で実装する
    [C11] rtol = 1e-8（Cycle 1 の 1e-9 より緩い）
    [C12] k_off_n の実測値は極の値とみなす（anchor = "pole"）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, identity, bmat

import model
from model import geometry, operators


# =============================================================================
class Par(model.Par):
    """Cycle 2 の既定パラメータ。"""

    # --- 分配比（Giacomelli 2022 [1] Fig.2C）
    f_cyt = 0.296      # 細胞質
    f_mono = 0.507     # 膜上の単量体
    f_clu = 0.198      # クラスター

    # --- クラスターの実測パラメータ（すべて B. subtilis）
    k_off_n = 0.0054   # 1/s  Feddersen 2021 [2] Tab.2（T1/2 = 128.64 s）
    D_n = 0.0          # µm²/s クラスターは不動
    k_off_n_anchor = "pole"   # 実測値をどこの値とみなすか（設計判断 [C12]）

    # --- 仮定（実測なし）
    p = 0.5            # 成長の次数。最も危険な仮定
    k_nuc = 1e-5       # 1/s  核形成
    f_nm = 0.5         # クラスター崩壊の分岐比

    # --- 曲率（Cycle 2 の主役）
    rho = 1.0          # 単量体は曲率を感知しない
    rho_n = 2.370      # 極/側壁のクラスター寿命比

    # --- 成長（既定は切ってある。設計判断 [C3]）
    mu = 0.0           # 1/s  希釈

    # --- 従属パラメータ
    k_grow = None      # クラスター収支から導出

    # --- 数値
    T = 3000.0
    eps_p = 1e-6
    atol_rel = 1e-11
    rtol = 1e-8


def derive(base, **kw):
    """base を一部だけ上書きした新しいパラメータクラスを返す。"""
    known = {k for k in dir(base) if not k.startswith("_")}
    unknown = set(kw) - known
    if unknown:
        raise ValueError(f"{base.__name__} に無い属性: {sorted(unknown)}")
    return type("ParVariant", (base,), kw)


def variant(**kw):
    return derive(Par, **kw)


def fractions(par=Par):
    """3 相の分配比を正規化して (f_cyt, f_mono, f_clu) で返す。"""
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
    """Cycle 1 の geometry() に、クラスター用の曲率重み w_n を足して返す。"""
    g = geometry(n, par)
    g["w_n"] = geometry(n, type("P", (par,), {"rho": par.rho_n}))["w"]
    return g


def k_on_of(g, par=Par, block=None):
    """k_on(z) [µm/s]。∫k_on dA = k_off·V·f_mono/f_cyt に校正（[C5]）。

    block を与えても ∫k_on dA は保たれる（＝塞いだぶんは他へ再配分される）。
    """
    f_cyt, f_mono, _ = fractions(par)
    w = g["w"] if block is None else g["w"] * block
    return (w * par.k_off * f_mono * g["V"].sum()
            / (f_cyt * (w * g["A"]).sum()))


def k_off_n_of(g, par=Par):
    """k_off_n(z) [1/s]。曲率はここに入る（設計判断 [C1][C12]）。"""
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
    """well-mixed 近似でのクラスター収支から k_grow を出す。"""
    _, f_mono, f_clu = fractions(par)
    g = geometry(401, par)
    n_bar = f_clu * par.N / g["A"].sum()
    k_eff = par.k_off_n * f_clu / f_mono
    if f_clu == 0.0:
        return 0.0
    return (k_eff - par.k_nuc) / n_bar**par.p


_KG_CACHE = {}


def _kg_key(par):
    return (par.R, par.L, par.N, par.D_c, par.D_m, par.k_off, par.k_off_n,
            par.rho, par.rho_n, par.p, par.k_nuc, par.f_nm, par.mu,
            par.f_cyt, par.f_mono, par.f_clu, par.k_off_n_anchor)


def _kg(par):
    """k_grow を返す。None なら f_clu に合わせて PDE で校正する。"""
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
    """N^p の平滑化（設計判断 [C4]）。"""
    return (x * x + eps * eps) ** (0.5 * p)


def _sparsity(n):
    tri = diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])
    I = identity(n)
    return bmat([[tri, I, I],
                 [I, tri, I],
                 [None, I, I]], format="csr")


def _assemble(n, par, block=None):
    """幾何・演算子・速度定数・RHS をまとめて作る。"""
    f_cyt, f_mono, f_clu = fractions(par)
    g = geometry2(n, par)
    L_c, L_m = operators(g, par)
    V, A = g["V"], g["A"]
    k_off_n = k_off_n_of(g, par)
    k_grow = _kg(par)
    p, eps, f_nm, mu = par.p, par.eps_p, par.f_nm, par.mu

    k_on_free = k_on_of(g, par, None)
    k_on_blk = k_on_free if block is None else k_on_of(g, par, block)
    src = mu * par.N / V.sum()

    def rhs(y, blocked=False):
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        flux = k_on * C - par.k_off * M
        G = (par.k_nuc + k_grow * _pow_p(Nf, p, eps)) * M
        rel = k_off_n * Nf
        dC = L_c @ C + (-flux + (1 - f_nm) * rel) * A / V - mu * C + src
        dM = L_m @ M + flux - G + f_nm * rel - mu * M
        dN = G - rel - mu * Nf
        return np.concatenate([dC, dM, dN])

    absLc, absLm = abs(L_c), abs(L_m)

    def rhs_mag(y, blocked=False):
        """各式の「項の大きさ」（無次元化の尺度、設計判断 [C9]）。"""
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
        """解析ヤコビアン（疎）。"""
        M, Nf = y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        AV = A / V
        phi = _pow_p(Nf, p, eps)
        dphi = p * Nf * (Nf * Nf + eps * eps) ** (0.5 * p - 1.0)
        kg_phi = par.k_nuc + k_grow * phi

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
        C0, M0, N0f = split(f_cyt, f_mono, f_clu)
        N0f = N0f * (1.0 + (seed - 1.0) * (g["A_capL"] / A))
        tot = (C0 * V).sum() + (M0 * A).sum() + (N0f * A).sum()
        sc = par.N / tot
        C0, M0, N0f = C0 * sc, M0 * sc, N0f * sc
    else:
        raise ValueError(f"unknown ic: {ic!r}")
    return np.concatenate([C0, M0, N0f])


def _report(par, asm, ys, ts, ic):
    """時系列から診断量をまとめる。"""
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
    """時間発展を解く。"""
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
    atol_vec = par.atol_rel * np.concatenate([
        np.full(n, par.N / asm["V"].sum()),
        np.full(n, par.N / asm["A"].sum()),
        np.full(n, par.N / asm["A"].sum())])
    sol = solve_ivp(f, (0, T), y0, method="BDF", t_eval=t_eval,
                    rtol=par.rtol, atol=atol_vec, jac=fj)
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return _report(par, asm, sol.y, sol.t, ic)


def steady(n=101, par=Par, ic="uniform", block=None, y0=None,
           tol=1e-7, maxit=40, seed_T=None, seed=10.0):
    """定常状態を直接解く（設計判断 [C8][C9]）。"""
    from scipy.sparse.linalg import spsolve

    asm = _assemble(n, par, block)
    V, A = asm["V"], asm["A"]
    rhs, jac, rhs_mag = asm["rhs"], asm["jac"], asm["rhs_mag"]
    blocked = block is not None

    floor = 1e-14 * par.N / A.sum()

    scale = max(par.k_off_n, 1e-12) * par.N / A.sum()
    mass_row = np.concatenate([V, A, A]) / par.N * scale

    def F(v):
        r = rhs(v, blocked=blocked)
        mass = (v[:n] * V).sum() + (v[n:2 * n] * A).sum() + (v[2 * n:] * A).sum()
        r[-1] = (mass / par.N - 1.0) * scale
        return r

    def relres(v):
        r = np.abs(rhs(v, blocked=blocked))
        return float(np.max(r / np.maximum(rhs_mag(v, blocked=blocked), 1e-300)))

    def newton(y_start):
        y = np.maximum(np.asarray(y_start, float), floor)
        best_y, best, stall = y.copy(), relres(y), 0
        for _ in range(maxit):
            if best < tol:
                break
            dy = spsolve(_with_mass_row(jac(y, blocked=blocked), mass_row), -F(y))
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

    y, res = (None, np.inf)
    starts = []
    if y0 is not None:
        starts.append(("given y0", lambda: np.asarray(y0, float)))
    T0 = seed_T if seed_T is not None else par.T
    for k in range(3):
        Tk = T0 * 10.0**k
        starts.append((f"seed {Tk:.3g} s", lambda Tk=Tk: np.concatenate(
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
    Jl = Jm.tolil()
    Jl[Jm.shape[0] - 1, :] = mass_row
    return Jl.tocsr()


def _lr(f, g):
    """左右の極キャップでの比（左/右）。"""
    l, r = (f * g["A_capL"]).sum(), (f * g["A_capR"]).sum()
    return l / max(r, 1e-300)


# =============================================================================
#  校正と解析
# =============================================================================
def rho_n_ceiling(par=Par, hi=1e6):
    """k_grow の校正が成り立つ rho_n の上限。"""
    from scipy.optimize import brentq

    target = fractions(par)[2]

    def g(v):
        return steady(101, derive(par, rho_n=v, k_grow=0.0))["f_clu"] - target

    if g(1.0) > 0:
        raise RuntimeError("rho_n=1 でも核形成だけで f_clu が実測を超える")
    v = 2.0
    while v < hi and g(v) < 0:
        v *= 2.0
    if v >= hi:
        return hi
    return brentq(g, v / 2, v, rtol=1e-6)


def calibrate_rho_n(target=0.66, n=101, par=Par, lo=1.0, hi=None):
    """クラスター分子の極割合が target になる rho_n を返す。"""
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
        raise RuntimeError(f"rho_n={hi:.3g} でも極の割合が {target} に届かない")
    return brentq(f, lo, hi, rtol=1e-8)


def calibrate_k_grow(n=101, par=Par, hi0=None):
    """定常のクラスター割合が実測の f_clu になる k_grow を求める。"""
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
        raise RuntimeError("k_grow=0 でも f_clu が目標を超える")
    hi = hi0 if hi0 is not None else max(k_grow_wellmixed(par), 1e-9)
    for _ in range(60):
        if f(hi) > 0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("f_clu の目標を上回る k_grow が見つからない")
    return brentq(f, lo, hi, rtol=1e-10)


def required_cluster_size(rho_n=None, par=Par, lp=100.0, length=30.0):
    """必要な ln rho_n を単量体 1 個の曲率エネルギーで割った分子数 Q。"""
    if rho_n is None:
        rho_n = par.rho_n
    dG = model.monomer_curvature_energy(par, lp=lp, length=length)
    return np.log(rho_n) / dG, dG


def analytic_clu(par=Par, n=801):
    """局所平衡でのクラスター極濃縮の上限。"""
    g = geometry(n, par)
    a = g["A_cap"].sum() / g["A"].sum()
    return par.rho_n / (par.rho_n * a + 1 - a), a


def critical_seed(par=Par, n=101, seeds=(2, 5, 10, 30, 100, 300, 1000),
                  rel=0.01):
    """非対称な枝に落とすのに必要な片極の種の大きさ。"""
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
    """p を振って双安定かを調べる。"""
    rows = []
    for p in p_values:
        sv, a = critical_seed(derive(par, p=p), n=n, seeds=seeds)
        rows.append((p, sv, a, np.isfinite(sv)))
    return rows


def latch_test(theta=0.9, t_release=600.0, n=101, par=Par, T=None):
    """PopZ を一過的にかけて外したあと、非対称が残るか（[C10] の 2 段階）。"""
    if T is None:
        T = par.T
    g = geometry2(n, par)
    block = 1.0 - theta * (g["A_capL"] / g["A"])
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
    """定常値への偏差が (1-frac) まで減る時刻。"""
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
def validate(par=Par, verbose=True):
    """本計算の前に必ず通すこと（13 項目）。"""
    out, ok = [], True

    def say(s):
        out.append(s)

    def chk(cond, label, detail):
        nonlocal ok
        ok &= bool(cond)
        say(f"  {label} {detail} {'OK' if cond else 'NG'}")

    say("=" * 76)
    say("VALIDATION - Cycle 2 (cluster)")
    say("=" * 76)

    fr = fractions(par)
    say(f"  fractions (source)  {par.f_cyt}/{par.f_mono}/{par.f_clu}"
        f"  sum = {par.f_cyt + par.f_mono + par.f_clu:.3f}")
    say(f"  fractions (norm.)   {fr[0]:.4f}/{fr[1]:.4f}/{fr[2]:.4f}")

    say("\n  -- V0 cycle 1 validation first --")
    chk(model.validate(model.Par, verbose=False), "V0", "model.validate()")

    g = geometry2(401, par)
    w_side, w_pole = g["w_n"][401 // 2], g["w_n"][0]
    chk(abs(w_side - 1.0) < 1e-6, "V1", f"sidewall w_n = {w_side:.6f} (exp 1.0)")
    chk(w_pole > 0.9 * par.rho_n, "V1",
        f"pole w_n = {w_pole:.4f} (exp near rho_n = {par.rho_n})")

    r = solve(101, par, ic="uniform")
    chk(r["mass_err"] < 1e-9, "V2", f"mass conservation err = {r['mass_err']:.1e}")

    rs = steady(101, par)
    chk(rs["budget_err"] < 1e-6, "V3",
        f"cluster budget err = {rs['budget_err']:.1e}"
        f" (steady residual {rs['steady_residual']:.1e})")

    rmu = solve(101, variant(mu=np.log(2) / 1500), ic="uniform")
    chk(rmu["mass_err"] < 1e-8, "V4",
        f"mu = ln2/1500 mass err = {rmu['mass_err']:.1e}")

    a_cap = g["A_cap"].sum() / g["A"].sum()
    r1 = steady(101, variant(rho=1.0, rho_n=1.0))
    good = (abs(r1["enrichment"] - 1.0) < 1e-5
            and abs(r1["enr_clu"] - 1.0) < 1e-5
            and abs(r1["clu_pole_frac"] - a_cap) < 1e-5
            and r1["cyto_var"] < 1e-6)
    chk(good, "V5",
        f"rho=rho_n=1: memb {r1['enrichment']:.6f}, clu "
        f"{r1['enr_clu']:.6f}, pole frac {r1['clu_pole_frac']:.6f}"
        f" (exp {a_cap:.6f}), cyto_var {r1['cyto_var']:.1e}")

    es = [steady(k, par)["enr_clu"] for k in (51, 101, 201, 401)]
    d = max(abs(np.diff(es))) / es[-1]
    chk(d < 0.02, "V6",
        "grid n=51..401: " + " -> ".join(f"{e:.4f}" for e in es)
        + f"  max change {d:.2%}")

    p1 = variant(k_nuc=0.0, k_grow=0.0, rho=model.Par.rho, rho_n=1.0,
                 f_cyt=model.Par.f_cyt, f_mono=1 - model.Par.f_cyt, f_clu=0.0)
    r7 = solve(101, p1, T=20.0, ic="uniform")
    e1 = model.solve(101, model.Par, T=20.0)["enrichment"]
    rel7 = abs(r7["enrichment"] - e1) / e1
    chk(rel7 < 1e-6, "V7",
        f"cycle 1 regression: {r7['enrichment']:.8f} vs {e1:.8f} (rel {rel7:.1e})")

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
    chk(rel8 < 1e-6, "V8", f"analytic jac vs FD rel = {rel8:.1e}")

    rt = solve(101, par, T=1e5, ic="uniform", n_out=5)
    rel9 = abs(rt["enr_clu"] - rs["enr_clu"]) / rs["enr_clu"]
    chk(rel9 < 1e-4, "V9",
        f"steady() {rs['enr_clu']:.6f} vs integration T=1e5 {rt['enr_clu']:.6f}"
        f" (rel {rel9:.1e}, residual {rt['residual']:.1e})")

    kg_wm = k_grow_wellmixed(par)
    kg_1 = calibrate_k_grow(par=variant(rho_n=1.0))
    rel10 = abs(kg_wm - kg_1) / kg_1
    chk(rel10 < 1e-3, "V10",
        f"rho_n=1: well-mixed {kg_wm:.6e} vs PDE {kg_1:.6e} (rel {rel10:.1e})")
    kg_n = _kg(par)
    say(f"       note rho_n={par.rho_n} (anchor={par.k_off_n_anchor}): "
        f"PDE {kg_n:.4e} = {kg_n/kg_wm:.2f}x well-mixed")

    e_lo = steady(101, variant(N=169.0))["enr_clu"]
    e_hi = steady(101, variant(N=169000.0))["enr_clu"]
    rel11 = abs(e_lo - e_hi) / e_hi
    chk(rel11 < 1e-6, "V11",
        f"N x1000 invariance: {e_lo:.8f} vs {e_hi:.8f} (rel {rel11:.1e})")

    mono = critical_seed(derive(par, p=0.5), n=71)[0]
    bist = critical_seed(derive(par, p=1.2), n=71)[0]
    chk(not np.isfinite(mono) and np.isfinite(bist), "V12",
        f"p=0.5 critical seed = {mono} (exp monostable), "
        f"p=1.2 = {bist} (exp bistable)")

    ceil = rho_n_ceiling(par)
    hit_cap = ceil >= 1e6 - 1
    chk(par.rho_n < 0.2 * ceil, "V13",
        f"rho_n = {par.rho_n} < calibratable ceiling "
        + ("(anchor='pole': effectively no ceiling, search hit 1e6)"
           if hit_cap else f"{ceil:.0f} ({par.rho_n/ceil:.1%})"))

    say("=" * 76)
    say("all passed" if ok else "FAILURES PRESENT - do not proceed")
    say("=" * 76)

    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys

    ok = validate()
    sys.exit(0 if ok else 1)
