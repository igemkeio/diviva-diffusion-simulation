"""DivIVA 局在モデル — Cycle 2（単量体 + クラスター）

⚠️ これは Cycle 5 の計算用のローカル複製。コードはプロジェクトの `cluster.py` と
   同一で、長い解説だけ省いてある。転記の正しさは validate() で検定する。
   設計判断 [C1]〜[C12] の原本はプロジェクトの `cluster.py` を見ること。
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, identity, bmat

import model
from model import geometry, operators


class Par(model.Par):
    # --- 分配比（Giacomelli 2022 Fig.2C）
    f_cyt = 0.296
    f_mono = 0.507
    f_clu = 0.198
    # --- クラスターの実測パラメータ（B. subtilis）
    k_off_n = 0.0054   # 1/s
    D_n = 0.0
    k_off_n_anchor = "pole"
    # --- 仮定（実測なし）
    p = 0.5            # 成長の次数。★ 最も危険な仮定
    k_nuc = 1e-5
    f_nm = 0.5
    # --- 曲率
    rho = 1.0
    rho_n = 2.370
    # --- 成長
    mu = 0.0
    # --- 従属
    k_grow = None
    # --- 数値
    T = 3000.0
    eps_p = 1e-6
    atol_rel = 1e-11
    rtol = 1e-8


def derive(base, **kw):
    known = {k for k in dir(base) if not k.startswith("_")}
    unknown = set(kw) - known
    if unknown:
        raise ValueError(f"{base.__name__} に無い属性: {sorted(unknown)}")
    return type("ParVariant", (base,), kw)


def variant(**kw):
    return derive(Par, **kw)


def fractions(par=Par):
    f = np.array([par.f_cyt, par.f_mono, par.f_clu], float)
    if np.any(f < 0):
        raise ValueError(f"分配比が負: {f}")
    s = f.sum()
    if abs(s - 1.0) > 0.02:
        raise ValueError(f"f_cyt + f_mono + f_clu = {s:.4f}。1 から離れすぎ")
    return tuple(f / s)


def geometry2(n, par=Par):
    g = geometry(n, par)
    g["w_n"] = geometry(n, type("P", (par,), {"rho": par.rho_n}))["w"]
    return g


def k_on_of(g, par=Par, block=None):
    f_cyt, f_mono, _ = fractions(par)
    w = g["w"] if block is None else g["w"] * block
    return (w * par.k_off * f_mono * g["V"].sum()
            / (f_cyt * (w * g["A"]).sum()))


def k_off_n_of(g, par=Par):
    if par.k_off_n_anchor == "pole":
        base = par.k_off_n * par.rho_n
    elif par.k_off_n_anchor == "sidewall":
        base = par.k_off_n
    else:
        raise ValueError(f"k_off_n_anchor は 'pole' か 'sidewall': "
                         f"{par.k_off_n_anchor!r}")
    return base / g["w_n"]


def k_grow_wellmixed(par=Par):
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
    if par.k_grow is not None:
        return par.k_grow
    key = _kg_key(par)
    if key not in _KG_CACHE:
        _KG_CACHE[key] = calibrate_k_grow(par=par)
    return _KG_CACHE[key]


def _pow_p(x, p, eps):
    """N^p の平滑化（設計判断 [C4]）。"""
    return (x * x + eps * eps) ** (0.5 * p)


def _assemble(n, par, block=None, grow_block=None):
    """幾何・演算子・速度定数・RHS。

    grow_block: Cycle 5 で追加。長さ n の配列で成長項にも掛ける
                （PopZ がクラスターの成長も妨げる規約。popz.py [P8] 相当）。
                None なら成長は妨げない（Cycle 2 の規約）。
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
    gb = np.ones(n) if grow_block is None else np.asarray(grow_block, float)
    src = mu * par.N / V.sum()

    def rhs(y, blocked=False):
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        gbk = gb if blocked else np.ones(n)
        flux = k_on * C - par.k_off * M
        G = (par.k_nuc + gbk * k_grow * _pow_p(Nf, p, eps)) * M
        rel = k_off_n * Nf
        dC = L_c @ C + (-flux + (1 - f_nm) * rel) * A / V - mu * C + src
        dM = L_m @ M + flux - G + f_nm * rel - mu * M
        dN = G - rel - mu * Nf
        return np.concatenate([dC, dM, dN])

    absLc, absLm = abs(L_c), abs(L_m)

    def rhs_mag(y, blocked=False):
        C, M, Nf = y[:n], y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        gbk = gb if blocked else np.ones(n)
        AV = A / V
        b = k_on * C + par.k_off * M
        G = (par.k_nuc + gbk * k_grow * _pow_p(Nf, p, eps)) * M
        rel = k_off_n * Nf
        mC = absLc @ C + (b + (1 - f_nm) * rel) * AV + mu * C + src
        mM = absLm @ M + b + G + f_nm * rel + mu * M
        mN = G + rel + mu * Nf
        return np.concatenate([mC, mM, mN])

    def jac(y, blocked=False):
        M, Nf = y[n:2 * n], y[2 * n:]
        k_on = k_on_blk if blocked else k_on_free
        gbk = gb if blocked else np.ones(n)
        AV = A / V
        phi = _pow_p(Nf, p, eps)
        dphi = p * Nf * (Nf * Nf + eps * eps) ** (0.5 * p - 1.0)
        kg_phi = par.k_nuc + gbk * k_grow * phi

        dCC = L_c - diags(k_on * AV + mu)
        dCM = diags(par.k_off * AV)
        dCN = diags((1 - f_nm) * k_off_n * AV)
        dMC = diags(k_on)
        dMM = L_m - diags(par.k_off + kg_phi + mu)
        dMN = diags(f_nm * k_off_n - gbk * k_grow * dphi * M)
        dNM = diags(kg_phi)
        dNN = diags(gbk * k_grow * dphi * M - k_off_n - mu)
        return bmat([[dCC, dCM, dCN],
                     [dMC, dMM, dMN],
                     [None, dNM, dNN]], format="csr")

    return dict(g=g, V=V, A=A, rhs=rhs, jac=jac, rhs_mag=rhs_mag,
                k_on=k_on_free, k_on_blk=k_on_blk,
                k_off_n=k_off_n, k_grow=k_grow,
                fr=(f_cyt, f_mono, f_clu))


def initial(n, par=Par, ic="uniform", block=None, seed=10.0):
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
          block=None, seed=10.0, y0=None, grow_block=None):
    if T is None:
        T = par.T
    asm = _assemble(n, par, block, grow_block)
    y0 = (initial(n, par, ic, block, seed=seed) if y0 is None
          else np.asarray(y0, float))
    rhs, jac = asm["rhs"], asm["jac"]
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
           tol=1e-7, maxit=40, seed_T=None, seed=10.0, grow_block=None):
    from scipy.sparse.linalg import spsolve

    asm = _assemble(n, par, block, grow_block)
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
        starts.append(("与えられた y0", lambda: np.asarray(y0, float)))
    T0 = seed_T if seed_T is not None else par.T
    for k in range(3):
        Tk = T0 * 10.0**k
        starts.append((f"助走 {Tk:.3g} s", lambda Tk=Tk: np.concatenate(
            [(sd := solve(n, par, T=Tk, ic=ic, n_out=3, block=block,
                          seed=seed, grow_block=grow_block))["C"],
             sd["M"], sd["Nf"]])))
    for _label, mk in starts:
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


def _with_mass_row(Jm, mass_row):
    Jl = Jm.tolil()
    Jl[Jm.shape[0] - 1, :] = mass_row
    return Jl.tocsr()


def _lr(f, g):
    l, r = (f * g["A_capL"]).sum(), (f * g["A_capR"]).sum()
    return l / max(r, 1e-300)


def calibrate_k_grow(n=101, par=Par, hi0=None):
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


def calibrate_rho_n(target=0.66, n=101, par=Par, lo=1.0, hi=100.0):
    """クラスター分子の極割合が target になる rho_n。"""
    from scipy.optimize import brentq
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


def required_cluster_size(rho_n=None, par=Par, lp=100.0, length=30.0):
    if rho_n is None:
        rho_n = par.rho_n
    dG = model.monomer_curvature_energy(par, lp=lp, length=length)
    return np.log(rho_n) / dG, dG


def analytic_clu(par=Par, n=801):
    g = geometry(n, par)
    a = g["A_cap"].sum() / g["A"].sum()
    return par.rho_n / (par.rho_n * a + 1 - a), a


def critical_seed(par=Par, n=101, seeds=(2, 5, 10, 30, 100, 300, 1000),
                  rel=0.01):
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


def block_of(theta, n=101, par=Par):
    """左極の占有（面積重み。設計判断 [8]）。"""
    g = geometry2(n, par)
    return 1.0 - theta * (g["A_capL"] / g["A"])


def sustained_asymmetry(theta=0.9, n=101, par=Par, grow=False):
    """PopZ をかけ続けたときの定常の左右比（塞いだ左 / 無摂動の右）。"""
    b = block_of(theta, n, par)
    gb = b if grow else None
    return steady(n, par, block=b, grow_block=gb)["asym_clu"]


def latch_test(theta=0.9, t_release=600.0, n=101, par=Par, T=None, grow=False):
    if T is None:
        T = par.T
    b = block_of(theta, n, par)
    gb = b if grow else None
    held = solve(n, par, T=t_release, ic="uniform", n_out=4, block=b, grow_block=gb)
    y = np.concatenate([held["C"], held["M"], held["Nf"]])
    rel = solve(n, par, T=T, n_out=40, y0=y)
    return held["asym_clu"], rel["asym_clu"]


def validate(par=Par, verbose=True):
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

    chk(model.validate(model.Par, verbose=False), "V0", "model.validate()")

    g = geometry2(401, par)
    w_side, w_pole = g["w_n"][401 // 2], g["w_n"][0]
    chk(abs(w_side - 1.0) < 1e-6, "V1", f"側壁の w_n = {w_side:.6f}")
    chk(w_pole > 0.9 * par.rho_n, "V1", f"極の w_n = {w_pole:.4f}")

    r = solve(101, par, ic="uniform")
    chk(r["mass_err"] < 1e-9, "V2", f"質量保存 誤差 = {r['mass_err']:.1e}")

    rs = steady(101, par)
    chk(rs["budget_err"] < 1e-6, "V3",
        f"クラスター収支 誤差 = {rs['budget_err']:.1e}"
        f"（定常残差 {rs['steady_residual']:.1e}）")

    rmu = solve(101, variant(mu=np.log(2) / 1500), ic="uniform")
    chk(rmu["mass_err"] < 1e-8, "V4", f"mu>0 でも質量保存 = {rmu['mass_err']:.1e}")

    a_cap = g["A_cap"].sum() / g["A"].sum()
    r1 = steady(101, variant(rho=1.0, rho_n=1.0))
    good = (abs(r1["enrichment"] - 1.0) < 1e-5
            and abs(r1["enr_clu"] - 1.0) < 1e-5
            and abs(r1["clu_pole_frac"] - a_cap) < 1e-5
            and r1["cyto_var"] < 1e-6)
    chk(good, "V5", f"rho=rho_n=1: 濃縮 {r1['enrichment']:.6f}/"
                    f"{r1['enr_clu']:.6f}, 極割合 {r1['clu_pole_frac']:.6f}"
                    f"（面積比 {a_cap:.6f}）")

    es = [steady(k, par)["enr_clu"] for k in (51, 101, 201, 401)]
    d = max(abs(np.diff(es))) / es[-1]
    chk(d < 0.02, "V6", "格子収束 n=51..401: "
        + " -> ".join(f"{e:.4f}" for e in es) + f"  最大変化 {d:.2%}")

    p1 = variant(k_nuc=0.0, k_grow=0.0, rho=model.Par.rho, rho_n=1.0,
                 f_cyt=model.Par.f_cyt, f_mono=1 - model.Par.f_cyt, f_clu=0.0)
    r7 = solve(101, p1, T=20.0, ic="uniform")
    e1 = model.solve(101, model.Par, T=20.0)["enrichment"]
    rel7 = abs(r7["enrichment"] - e1) / e1
    chk(rel7 < 1e-6, "V7", f"Cycle 1 回帰: 相対差 {rel7:.1e}")

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
    chk(rel8 < 1e-6, "V8", f"解析ヤコビアン vs 差分 = {rel8:.1e}")

    rt = solve(101, par, T=1e5, ic="uniform", n_out=5)
    rel9 = abs(rt["enr_clu"] - rs["enr_clu"]) / rs["enr_clu"]
    chk(rel9 < 1e-4, "V9", f"steady() {rs['enr_clu']:.6f} vs 積分 "
                           f"{rt['enr_clu']:.6f}（相対差 {rel9:.1e}）")

    kg_wm = k_grow_wellmixed(par)
    kg_1 = calibrate_k_grow(par=variant(rho_n=1.0))
    rel10 = abs(kg_wm - kg_1) / kg_1
    chk(rel10 < 1e-3, "V10", f"rho_n=1: well-mixed vs PDE 相対差 {rel10:.1e}")

    e_lo = steady(101, variant(N=169.0))["enr_clu"]
    e_hi = steady(101, variant(N=169000.0))["enr_clu"]
    rel11 = abs(e_lo - e_hi) / e_hi
    chk(rel11 < 1e-6, "V11", f"N を 1000 倍振っても不変（相対差 {rel11:.1e}）")

    mono = critical_seed(derive(par, p=0.5), n=71)[0]
    bist = critical_seed(derive(par, p=1.2), n=71)[0]
    chk(not np.isfinite(mono) and np.isfinite(bist), "V12",
        f"p=0.5 単安定 {mono}, p=1.2 双安定 {bist}")

    say("=" * 76)
    say("すべて合格" if ok else "★ 不合格あり。本計算に進まないこと")
    say("=" * 76)
    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
