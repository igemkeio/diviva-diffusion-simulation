"""
DivIVA 局在モデル — Cycle 1（単量体のみ・クラスターなし）
（プロジェクトの model.py を復元したもの。数値確認用）
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags


class Par:
    # --- 幾何：大腸菌
    R = 0.4            # µm
    L = 3.0            # µm

    # --- 実測パラメータ
    D_c = 3.0          # µm²/s
    D_m = 0.1          # µm²/s
    k_off = 9.0        # 1/s
    N = 1690.0         # 分子/細胞
    f_cyt = 0.30

    # --- 求める量
    rho = 8.76

    # --- 数値
    rtol, atol = 1e-9, 1e-11


def variant(**kw):
    unknown = set(kw) - {k for k in vars(Par) if not k.startswith("_")}
    if unknown:
        raise ValueError(f"Par に無い属性: {sorted(unknown)}")
    return type("ParVariant", (Par,), kw)


def geometry(n, par=Par, nfine=None):
    R, L = par.R, par.L
    Lcyl = L - 2 * R
    if Lcyl <= 0:
        raise ValueError(f"L ({L}) は 2R ({2*R}) より大きい必要がある")
    if nfine is None:
        nfine = max(20001, 40 * n)

    ze = np.linspace(0, L, n + 1)
    zf = np.linspace(0, L, nfine)
    dzf = np.gradient(zf)

    r = np.full_like(zf, R)
    a = zf < R
    r[a] = np.sqrt(np.maximum(R**2 - (zf[a] - R) ** 2, 0))
    b = zf > R + Lcyl
    r[b] = np.sqrt(np.maximum(R**2 - (zf[b] - R - Lcyl) ** 2, 0))

    dsdz = np.sqrt(1 + np.gradient(r, zf) ** 2)
    band = 2 * np.pi * r * dsdz
    is_cap = (zf < R) | (zf > R + Lcyl)
    H = np.where(is_cap, 1 / R, 1 / (2 * R))
    w = par.rho ** ((H - 1 / (2 * R)) / (1 / R - 1 / (2 * R)))

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
        w=acc(band * w) / A,
        rc=np.interp(0.5 * (ze[:-1] + ze[1:]), zf, r),
    )


def operators(g, par=Par):
    n, dz = g["n"], g["dz"]
    rf = 0.5 * (g["rc"][:-1] + g["rc"][1:])
    G_c = par.D_c * np.pi * rf**2 / dz
    G_m = par.D_m * 2 * np.pi * rf / (0.5 * (g["S"][:-1] + g["S"][1:]))

    def lap(G, W):
        di = np.zeros(n)
        di[:-1] -= G
        di[1:] -= G
        return diags([G / W[1:], di / W, G / W[:-1]], [-1, 0, 1], format="csr")

    return lap(G_c, g["V"]), lap(G_m, g["A"])


def k_on_of(g, par=Par, block=None):
    w = g["w"] if block is None else g["w"] * block
    return (w * par.k_off * (1 - par.f_cyt) * g["V"].sum()
            / (par.f_cyt * (w * g["A"]).sum()))


def solve(n=101, par=Par, T=20.0, ic="uniform", n_out=60, block=None):
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
        flux = k_on * C - par.k_off * M
        return np.concatenate([L_c @ C - flux * A / V,
                               L_m @ M + flux])

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

    N0 = mass_t[0]
    return dict(g=g, par=par, t=sol.t, C=Ct[:, -1], M=Mt[:, -1], Ct=Ct, Mt=Mt,
                k_on=k_on, enrichment=enr_t[-1], enr_t=enr_t, N0=N0, ic=ic,
                f_cyt=(Ct[:, -1] * V).sum() / N0,
                mass_err=np.abs(mass_t - N0).max() / N0,
                cyto_var=1 - Ct[:, -1].min() / Ct[:, -1].max())


def enrichment(rho=None, n=101, par=Par, **kw):
    p = par if rho is None else type("P", (par,), {"rho": rho})
    return solve(n, p, T=20.0, **kw)["enrichment"]


def analytic(par=Par, n=801):
    g = geometry(n, par)
    a = g["A_cap"].sum() / g["A"].sum()
    return par.rho / (par.rho * a + 1 - a), a


def asymmetry(theta, par=Par, n=201, T=20.0):
    g = geometry(n, par)
    fL = g["A_capL"] / g["A"]
    block = 1.0 - theta * fL
    r = solve(n, par, T=T, ic="cytoplasm", n_out=5, block=block)
    M = r["M"]
    return (M * g["A_capR"]).sum() / (M * g["A_capL"]).sum()


def pole_asymmetry_trace(par=Par, n=201, T=20.0, n_out=200):
    r = solve(n, par, T=T, ic="one_pole", n_out=n_out)
    g = r["g"]
    left = g["z"] < par.R
    right = g["z"] > par.L - par.R
    Ml = (r["Mt"] * g["A_cap"][:, None])[left].sum(0)
    Mr = (r["Mt"] * g["A_cap"][:, None])[right].sum(0)
    return r["t"], (Ml - Mr) / (Ml + Mr)


def rise_time(r, frac=0.632):
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
    R_nm = par.R * 1000.0
    return 0.5 * lp * length / R_nm**2


def validate(par=Par, verbose=True):
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
                           2 * par.R / par.L)]:
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
        f"{'OK' if r['mass_err'] < 1e-10 else '★NG'}")

    r1 = solve(101, type("P", (par,), {"rho": 1.0}), T=20.0)
    good = abs(r1["enrichment"] - 1.0) < 1e-4 and r1["cyto_var"] < 1e-6
    ok &= good
    say(f"  V4 rho=1: 極濃縮 = {r1['enrichment']:.6f}, "
        f"細胞質変動 = {r1['cyto_var']:.2e}{' OK' if good else ' ★NG'}")

    ana, _ = analytic(par)
    ok &= r["enrichment"] < ana
    say(f"  V5 極濃縮 {r['enrichment']:.4f} < 解析解 {ana:.4f} "
        f"{'OK' if r['enrichment'] < ana else '★NG'}")

    es = [solve(k, par, T=20.0)["enrichment"] for k in (51, 101, 201, 401)]
    d = max(abs(np.diff(es))) / es[-1]
    ok &= d < 0.01
    say("  V6 格子収束 n=51..401: " + " -> ".join(f"{e:.4f}" for e in es)
        + f"  最大変化 {d:.2%} {'OK' if d < 0.01 else '★NG'}")

    say("=" * 68)
    say("すべて合格" if ok else "★ 不合格あり")
    say("=" * 68)

    if verbose:
        print("\n".join(out))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if validate() else 1)
