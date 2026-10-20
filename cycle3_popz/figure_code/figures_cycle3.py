"""
Cycle 3 の図を生成する — S1 空間占有 と S2 局所分解のどちらを採るか。

    python figures_cycle3.py          # 全部（約 3 分）
    python figures_cycle3.py 2        # 図2 だけ

    fig1_timecourse.png   図1  時間発展（強さ別に何本も引く）
    fig2_strength.png     図2  機構の強さへの依存 + 酵素の実現範囲
    fig3_verdict.png      図3  判定図（1 枚で結論を見せる）

⚠️ 数値は `results_cycle3.py` の出力が唯一の正である。食い違ったらそちらを見ること。

配色について
------------
`figures2.py` の検証済み配色をそのまま引き継ぐ（緑・青緑・紫は P 型 D 型色覚で
既存色と区別がつかないので使わない）。Cycle 3 で増えた区別は「戦略が 2 つある」
ことなので、

    S1 空間占有 → 紺 #1f4e79（Cycle 1/2 の膜の色）
    S2 局所分解 → 赤 #c1666b（Cycle 1/2 の細胞質の色）

と **色相を戦略に、明度を機構の強さに**割り当てた。
この 2 色は figures2.py で全ペア CVD 合格（最悪 ΔE 11.1）を確認済みである。
琥珀 #e8a33d は酵素の推定帯だけに使う（背景とのコントラストが 2.1:1 と低いので
**必ず直接ラベルを添える**）。

フォント
--------
⚠️ Cycle 1/2 の図は DejaVu Sans（英語ラベル）だが、本サイクルの図は
プロジェクトの文書に合わせて日本語ラベルにし、Noto Sans CJK JP を使う。
iGEM wiki 用に英語が要るなら LANG = "en" に変えること。
"""

# === repo 構成に合わせたパス解決（figure_code/ から実行する）=========================
#   code/ のモデル本体を import し、図は ../figures/ に出す。
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
_sys.path.insert(0, _os.path.join(_ROOT, "code"))
_FIGS = _os.path.join(_ROOT, "figures"); _os.makedirs(_FIGS, exist_ok=True)
_os.chdir(_FIGS)
# ==============================================================================


import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import popz
from cluster import derive

LANG = "ja"          # "ja" | "en"（`python figures_cycle3.py --en 5` でも切替）

# 英語版はファイル名に _en が付く。ラベルはすべてこの辞書を通す。
LAB = {}


def T(k):
    """ラベルを引く。英語が未登録なら日本語にフォールバックして気づけるようにする。"""
    d = LAB.get(k)
    if d is None:
        raise KeyError(f"ラベル未登録: {k}")
    return d.get(LANG, d["ja"])


def _fn(name):
    return name if LANG == "ja" else name.replace(".png", "_en.png")


def _style():
    plt.rcParams["font.family"] = ("Noto Sans CJK JP" if LANG == "ja"
                                   else "DejaVu Sans")


plt.rcParams.update({
    "font.family": "Noto Sans CJK JP",
    "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 10, "axes.titlesize": 11, "figure.dpi": 150,
    "axes.linewidth": 0.9, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "axes.unicode_minus": False,
})

# --- figures2.py と同じ配色
S1C = "#1f4e79"       # S1 空間占有（紺）
S2C = "#c1666b"       # S2 局所分解（赤）
ENZ = "#e8a33d"       # 酵素の推定帯（琥珀。直接ラベル必須）
GREY = "#8a8a8a"
P = popz.Par
NG = 210              # 極境界がセル境界に乗る格子（設計判断 [P12]）
GEN = 1500.0          # s, 一世代
T_END = 9000.0        # s, 6 世代

# --- 機構の強さ（何本線を引くか）
TH = (0.3, 0.5, 0.7, 0.9)                 # S1 の被覆率
KC = (0.003, 0.01, 0.03, 0.1)             # S2 の切断速度 [1/s]

# 明度ランプ（色相は戦略、明度は強さ）
S1R = ["#9db9d4", "#5b85b0", "#2f6394", "#17395a"]
S2R = ["#e3b4b7", "#d2888c", "#c1666b", "#8e484c"]


# =============================================================================
#  酵素の実現しうる切断速度 ⚠️ 推定
# =============================================================================
#   k_cut = (k_cat/K_M) * [TEV]_local
#
#   k_cat/K_M = 5e3 M^-1 s^-1   ... PARAMETERS.md §5.1 ⚠️ 原典未確認
#   PopZ 微小区画を半径 r_popz の半球とみなす（r_popz = 150〜350 nm、§5.1 の
#   スキャン範囲）。V = (2/3)pi r^3 = 0.0071〜0.090 µm³
#   そこへ集まる TEV を 100〜10000 分子と置く ⚠️ 根拠なし
#
#   1 分子/µm³ = 1.6605e-9 M
#
#   ⚠️ この帯は「原典未確認の速度定数」に「根拠のない分子数」を掛けたもので、
#     幅が 3 桁ある。**値そのものを主張してはいけない。**
#     それでも図に描く意味は、モデルが必要とする窓（§4.3）が
#     この帯の下端にしかないことを見せる点にある。
KCAT_KM = 5.0e3
MOL_PER_UM3_TO_M = 1.6605e-9


def _kcut_from(n_tev, r_popz):
    V = 2.0 / 3.0 * np.pi * r_popz ** 3
    return KCAT_KM * (n_tev / V) * MOL_PER_UM3_TO_M


ENZ_LO = _kcut_from(100, 0.35)        # 0.0092 /s
ENZ_HI = _kcut_from(10000, 0.15)      # 11.8  /s
ENZ_MID = _kcut_from(1000, 0.25)      # 0.254 /s

# モデルが必要とする窓（results_cycle3.py §5。cut_target="MN"）
NEED_LO, NEED_HI = 0.002429, 0.01279   # 左右比 2 倍 〜 10 倍


def _save(fig, name):
    fig.savefig(name, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}", flush=True)


def _trace(par, T=T_END, n_out=90):
    """時間発展を (世代, 局在比, 足場, 総量/N) で返す。"""
    r = popz.solve(NG, par, T=T, ic="cytoplasm", n_out=n_out)
    g, Nt = r["g"], r["Nt"]
    nL = (Nt * g["A_capL"][:, None]).sum(0)
    nR = (Nt * g["A_capR"][:, None]).sum(0)
    # クラスターがまだ無い時刻の比は意味を持たないので隠す
    ok = (nL + nR) > 1e-3
    lr = np.where(ok, r["lr_t"], np.nan)
    return r["t"] / GEN, lr, nR, r["mass_t"] / par.N


def _label_end(ax, x, y, txt, color, dx=0.08, dy=1.0, yabs=None, **kw):
    """線の右端に直接ラベルを置く（凡例だけに頼らない）。

    yabs を与えるとラベルだけその高さに置き、線と引き出し線で結ぶ。
    下端で値が詰まって重なるときに使う。
    """
    i = np.max(np.where(np.isfinite(y))[0])
    if yabs is None:
        ax.annotate(txt, (x[i], y[i] * dy), xytext=(x[i] + dx, y[i] * dy),
                    color=color, fontsize=8.5, va="center", **kw)
    else:
        ax.annotate(txt, (x[i], y[i]), xytext=(x[i] + dx, yabs),
                    color=color, fontsize=8.5, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.7,
                                    shrinkA=0, shrinkB=2), **kw)


# ------------------------------------------------------------------- 図1
def fig1():
    """時間発展。強さ別に何本も引く（ユーザー要求の中心）。"""
    base = popz.baseline(n=NG)
    b_scaf = base["scaffold_R"]

    runs = []
    for th, c in zip(TH, S1R):
        runs.append(("S1", f"θ={th}", c, _trace(derive(P, theta=th))))
    for kc, c in zip(KC, S2R):
        runs.append(("S2", f"k={kc}", c,
                     _trace(derive(P, theta=0.0, k_cut=kc, cut_target="MN"))))
    # 単量体のみを切る場合（左右比が立たないことを 1 本だけ示す）
    mono = _trace(derive(P, theta=0.0, k_cut=0.01, cut_target="M"))

    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.0))

    for a in ax:
        a.axvline(1.0, color=GREY, lw=0.8, ls=":", zorder=0)
        a.set_xlabel("時間 [世代]（1 世代 = 1500 s）")
        a.set_xlim(0, T_END / GEN)

    # (a) 局在比
    for kind, lab, c, (t, lr, _, _) in runs:
        ax[0].plot(t, lr, color=c, lw=1.8)
        _label_end(ax[0], t, lr, lab, c)
    ax[0].plot(mono[0], mono[1], color=GREY, lw=1.4, ls="--")
    _label_end(ax[0], mono[0], mono[1], "単量体のみ\nk=0.01", GREY)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("局在比（良い極 / 塞いだ極のクラスター分子数）")
    ax[0].set_title("(a) 左右比は S2 でも立つ", loc="left")
    ax[0].set_xlim(0, T_END / GEN + 1.4)

    # (b) 足場 ★ 判定の核心
    YABS = {"k=0.01": 24.0, "k=0.03": 11.0, "k=0.1": -2.0}   # 下端で詰まるので高さをずらす
    for kind, lab, c, (t, _, sc, _) in runs:
        ax[1].plot(t, sc, color=c, lw=1.8)
        _label_end(ax[1], t, sc, lab, c, yabs=YABS.get(lab))
    ax[1].axhline(b_scaf, color=GREY, lw=1.2, ls="-")
    ax[1].annotate(f"PopZ なしの基準 {b_scaf:.0f} 分子", (3.2, b_scaf),
                   xytext=(3.2, b_scaf + 4), color=GREY, fontsize=8.5)
    ax[1].set_ylabel("良い極の足場 [クラスター分子数]")
    ax[1].set_title("(b) ★ 足場は S1 で増え、S2 で減る", loc="left")
    ax[1].set_xlim(0, T_END / GEN + 1.4)
    ax[1].set_ylim(-8, 175)

    # (c) 総量
    for kind, lab, c, (t, _, _, m) in runs:
        ax[2].plot(t, m, color=c, lw=1.8)
    ax[2].plot(mono[0], mono[3], color=GREY, lw=1.4, ls="--")
    ax[2].axhline(1.0, color=GREY, lw=1.2)
    ax[2].annotate("S1 は 4 本すべてここ（総量を削らない）", (0.2, 1.0),
                   xytext=(0.3, 0.92), color=S1C, fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color=S1C, lw=0.9))
    ax[2].annotate("S2（上から k=0.003 / 0.01 / 0.03 / 0.1）", (1.3, 0.70),
                   color=S2C, fontsize=8.5)
    ax[2].annotate("単量体のみ k=0.01", (6.0, 0.37), xytext=(3.4, 0.30),
                   color=GREY, fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax[2].set_ylabel("細胞あたり DivIVA 総量 / N")
    ax[2].set_title("(c) S2 は細胞全体を削る", loc="left")
    ax[2].set_ylim(0, 1.08)

    fig.text(0.5, -0.04,
             "PopZ を先に発現させておき、t=0 で DivIVA を誘導（全分子が細胞質から出発）。"
             "S2 はクラスターも切れる場合（S2 に有利な側）を実線で示す。",
             ha="center", fontsize=8.5, color="#444444")
    _save(fig, "fig1_timecourse.png")


# ------------------------------------------------------------------- 図2
def fig2():
    """機構の強さへの依存と、酵素が実現しうる範囲。"""
    base = popz.baseline(n=NG)
    b_scaf = base["scaffold_R"]
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    ths = np.concatenate([np.linspace(0.0, 0.9, 19), [0.93, 0.95, 0.97, 0.98]])
    lr1, sc1 = [], []
    w = y0
    for th in ths:
        r = popz.steady(NG, derive(P, theta=th), y0=w)
        w = np.concatenate([r["C"], r["M"], r["Nf"]])
        lr1.append(r["lr_clu"])
        sc1.append(r["scaffold_R"] / b_scaf)

    kcs = np.logspace(-4, 1.3, 28)
    lr2, sc2 = [], []
    w = y0
    for kc in kcs:
        r = popz.steady(NG, derive(P, theta=0.0, k_cut=kc, cut_target="MN"), y0=w)
        w = np.concatenate([r["C"], r["M"], r["Nf"]])
        lr2.append(r["lr_clu"])
        sc2.append(r["scaffold_R"] / b_scaf)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3))
    YL = (3e-5, 3e4)

    # (a) S1
    ax[0].plot(ths, lr1, color=S1C, lw=2.2)
    ax[0].plot(ths, sc1, color=S1C, lw=2.0, ls="--")
    ax[0].annotate("局在比", (0.84, lr1[-5]), color=S1C, fontsize=9,
                   fontweight="bold")
    ax[0].annotate("足場（基準比）", (0.50, sc1[10] * 1.5), color=S1C, fontsize=9)
    ax[0].axhline(1.0, color="k", lw=0.8)
    ax[0].annotate("基準と同じ", (0.02, 2.6), color=GREY, fontsize=8)
    ax[0].annotate("S1 の足場は θ をどれだけ上げても\n基準を下回らない（最大 1.37 倍）。\n"
                   "下げる仕組みが無いため、\nこの軸には行き場がない。",
                   (0.07, 1e-3), color=S1C, fontsize=9, va="center")
    ax[0].set_yscale("log")
    ax[0].set_ylim(*YL)
    ax[0].set_xlim(0, 1.0)
    ax[0].set_xlabel("S1 の強さ: 極の実効被覆率 θ")
    ax[0].set_ylabel("定常での値（無次元）")
    ax[0].set_title("(a) S1 空間占有 — 強くしても足場は減らない", loc="left")
    for tg, th in ((2, 0.350), (10, 0.818)):
        ax[0].plot([th], [tg], "o", color=S1C, ms=5)
        ax[0].annotate(f"{tg}倍 → θ={th:.2f}", (th, tg), xytext=(th - 0.30, tg * 2.2),
                       color=S1C, fontsize=8,
                       arrowprops=dict(arrowstyle="->", color=S1C, lw=0.8))

    # (b) S2
    ax[1].axvspan(ENZ_LO, ENZ_HI, color=ENZ, alpha=0.22, lw=0, zorder=0)
    ax[1].axvline(ENZ_MID, color=ENZ, lw=1.4, ls="-", zorder=1)
    ax[1].plot(kcs, lr2, color=S2C, lw=2.2, zorder=3)
    ax[1].plot(kcs, sc2, color=S2C, lw=2.0, ls="--", zorder=3)
    ax[1].annotate("局在比", (2.0, 2e3), color=S2C, fontsize=9, fontweight="bold")
    ax[1].annotate("足場（基準比）", (1.2e-2, 4e-3), color=S2C, fontsize=9)
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(*YL)
    ax[1].set_xlim(1e-4, 2e1)
    ax[1].set_xlabel("S2 の強さ: 切断速度 $k_{cut}$ [1/s]")
    ax[1].set_title("(b) S2 局所分解 — 強くすると足場が消える", loc="left")

    # 酵素帯のラベル（琥珀は直接ラベル必須）
    ax[1].annotate("TEV が実現しうる範囲（推定）", (np.sqrt(ENZ_LO * ENZ_HI), 1.1e4),
                   color="#9a6a1a", fontsize=8.5, ha="center")
    ax[1].annotate("中心的な想定\n0.25 /s", (ENZ_MID, 1.1e-3), xytext=(1.1, 7e-4),
                   color="#9a6a1a", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="#9a6a1a", lw=0.8))
    ov = (np.log10(NEED_HI) - np.log10(ENZ_LO)) / (np.log10(ENZ_HI) - np.log10(ENZ_LO))
    ax[1].annotate(f"必要な窓は帯の下端\n{ov*100:.0f}% にしか重ならない",
                   (2e-4, 2e-2), color=S1C, fontsize=9, va="center")

    # モデルが必要とする窓
    ax[1].plot([NEED_LO, NEED_HI], [1.3e-4, 1.3e-4], color=S1C, lw=4,
               solid_capstyle="butt", zorder=4)
    ax[1].annotate("モデルが必要とする窓\n（左右比 2〜10 倍）",
                   (np.sqrt(NEED_LO * NEED_HI), 1.3e-4), xytext=(1.3e-4, 1.1e-3),
                   color=S1C, fontsize=8,
                   arrowprops=dict(arrowstyle="->", color=S1C, lw=0.8))

    fig.text(0.5, -0.05,
             "どちらも実線が局在比、破線が足場（PopZ なしの基準に対する比）。"
             "横の黒線 1.0 が基準。S2 は左右比を上げると必ず足場が下がる。\n"
             "※ 琥珀の帯は $k_{cat}/K_M = 5\\times10^{3}\\ \\mathrm{M^{-1}s^{-1}}$（原典未確認）に"
             "TEV 100〜10000 分子・半径 150〜350 nm の区画を仮定して作った概算で、幅が 3 桁ある。",
             ha="center", fontsize=8.5, color="#444444")
    _save(fig, "fig2_strength.png")


# ------------------------------------------------------------------- 図3
def fig3():
    """判定図 — 1 枚で結論を見せる。"""
    base = popz.baseline(n=NG)
    b_scaf = base["scaffold_R"]
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    def scan(vals, mk):
        lr, sc, w = [], [], y0
        for v in vals:
            r = popz.steady(NG, mk(v), y0=w)
            w = np.concatenate([r["C"], r["M"], r["Nf"]])
            lr.append(r["lr_clu"])
            sc.append(r["scaffold_R"] / b_scaf)
        return np.array(lr), np.array(sc)

    ths = np.concatenate([np.linspace(0.02, 0.9, 23), [0.93, 0.95, 0.97, 0.98]])
    lr1, sc1 = scan(ths, lambda v: derive(P, theta=v))
    kcs = np.logspace(-3.4, 1.0, 26)
    lr2, sc2 = scan(kcs, lambda v: derive(P, theta=0.0, k_cut=v, cut_target="MN"))
    lr2m, sc2m = scan(np.logspace(-1, 1.6, 18),
                      lambda v: derive(P, theta=0.0, k_cut=v, cut_target="M"))

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axhspan(3e-5, 1.0, color="#f6f0ee", lw=0, zorder=0)
    ax.axhline(1.0, color="k", lw=0.9, zorder=1)

    ax.plot(lr1, sc1, color=S1C, lw=2.6, zorder=4)
    ax.plot(lr2, sc2, color=S2C, lw=2.6, zorder=4)
    ax.plot(lr2m, sc2m, color=GREY, lw=1.6, ls="--", zorder=3)

    # 目標の左右比での到達点
    for tg in (2, 5, 10):
        for lr, sc, c in ((lr1, sc1, S1C), (lr2, sc2, S2C)):
            v = np.interp(np.log(tg), np.log(lr), sc)
            ax.plot([tg], [v], "o", color=c, ms=7, zorder=5,
                    mec="white", mew=1.2)
        ax.axvline(tg, color=GREY, lw=0.7, ls=":", zorder=0)

    ax.annotate("S1 空間占有\n（塞いだ極の分子が良い極へ移る）",
                (lr1[-1], sc1[-1]), xytext=(160, 1.75),
                color=S1C, fontsize=10, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=S1C, lw=1.0))
    ax.annotate("S2 局所分解\nクラスターも切れる場合", (330, 4e-3),
                color=S2C, fontsize=10, fontweight="bold", ha="center")
    ax.annotate("S2 単量体のみ\n（左右比すら立たない）", (3.4, 7e-5),
                xytext=(14, 2.6e-4), color=GREY, fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8))
    ax.annotate("足場が基準より増える", (1.22, 1.45), color=S1C, fontsize=8.5)
    ax.annotate("足場が基準より減る", (1.22, 0.62), color=S2C, fontsize=8.5)

    for tg, x in ((2, 2), (5, 5), (10, 10)):
        ax.annotate(f"{tg}倍", (x, 2.4), color=GREY, fontsize=8, ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.1, 2e3)
    ax.set_ylim(3e-5, 3.0)
    ax.set_xticks([2, 5, 10, 100, 1000])
    ax.set_xticklabels(["2", "5", "10", "100", "1000"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("達成した局在比 [倍]（良い極 / 塞いだ極）→ 右が強い非対称")
    ax.set_ylabel("良い極に残る足場（PopZ なしの基準に対する比）→ 上が良い")
    ax.set_title("同じ左右比を、どれだけの足場を犠牲にして買うか", loc="left")
    fig.text(0.5, -0.02,
             "★ 目的は「DivIVA を消すこと」ではなく「片極に足場を作ること」なので、"
             "右上に行く戦略が勝つ。\n丸は左右比 2 / 5 / 10 倍の到達点。",
             ha="center", fontsize=8.5, color="#444444")
    _save(fig, "fig3_verdict.png")


# ------------------------------------------------------------------- 図4
# 実測の物差し（PARAMETERS.md §3、Feddersen 2021 の PALM）
CLU_MEAN = 47.0       # 分子/クラスター（平均）
CLPXP_LO, CLPXP_HI = 1.0, 10.0   # ClpXP 細胞全体の処理能力「数分子/s」⚠️未確認


def fig4():
    """良い極の『絶対量』で条件を比べる。

    足場も処理量も発現量 N に**厳密に比例する**（検証 V11 のスケーリング対称性、
    results_cycle3.py §9）。したがって絶対量は N=1690 での値 × (N/1690) で読める。
    この性質を使って「目標の絶対量を得るのに必要な発現量」まで出せる。
    """
    base = popz.baseline(n=NG)
    b_scaf = base["scaffold_R"]
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    def scan(vals, mk):
        lr, sc, cf, w = [], [], [], y0
        for v in vals:
            r = popz.steady(NG, mk(v), y0=w)
            w = np.concatenate([r["C"], r["M"], r["Nf"]])
            lr.append(r["lr_clu"])
            sc.append(r["scaffold_R"])
            cf.append(r["cut_flux"])
        return np.array(lr), np.array(sc), np.array(cf)

    ths = np.concatenate([np.linspace(0.02, 0.9, 23), [0.93, 0.95, 0.97, 0.98]])
    lr1, sc1, cf1 = scan(ths, lambda v: derive(P, theta=v))
    kcs = np.logspace(-3.4, -0.6, 24)
    lr2, sc2, cf2 = scan(kcs, lambda v: derive(P, theta=0.0, k_cut=v,
                                               cut_target="MN"))

    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.3))

    # (a) 絶対分子数
    a = ax[0]
    a.axhline(b_scaf, color=GREY, lw=1.2)
    a.axhline(CLU_MEAN, color=GREY, lw=1.2, ls="--")
    a.annotate(f"PopZ なしの基準 {b_scaf:.0f} 分子", (4.5, b_scaf * 0.80),
               color=GREY, fontsize=8.5)
    a.annotate(f"平均クラスター 1 個分 {CLU_MEAN:.0f} 分子", (4.5, CLU_MEAN * 1.10),
               color=GREY, fontsize=8.5)
    a.plot(lr1, sc1, color=S1C, lw=2.6)
    a.plot(lr2, sc2, color=S2C, lw=2.6)
    # S2 が 47 分子を割る左右比
    i = np.max(np.where(sc2 >= CLU_MEAN)[0])
    xc = np.interp(CLU_MEAN, sc2[::-1], lr2[::-1])
    a.plot([xc], [CLU_MEAN], "o", color=S2C, ms=7, mec="white", mew=1.2, zorder=5)
    a.annotate(f"S2 はここで\nクラスター 1 個分を割る\n（左右比 {xc:.1f} 倍）",
               (xc, CLU_MEAN), xytext=(1.5, 7.0), color=S2C, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=S2C, lw=0.9))
    a.annotate("S1 空間占有", (16, 175), color=S1C, fontsize=10, fontweight="bold")
    a.annotate("S2 局所分解", (16, 7.5), color=S2C, fontsize=10, fontweight="bold")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(1.1, 40); a.set_ylim(3, 400)
    a.set_xlabel("達成した局在比")
    a.set_ylabel("良い極の足場 [クラスター分子数]")
    a.set_title("(a) 絶対量で見る（発現量 N = 1690）", loc="left")

    # (b) 必要な発現量
    b = ax[1]
    b.axhspan(1690 * 10, 1690 * 100, color=GREY, alpha=0.13, lw=0)
    b.annotate("大腸菌の異種発現で\n到達しうる範囲（10〜100 倍）", (1.25, 4.5e4),
               color=GREY, fontsize=8.5)
    b.axhline(1690, color=GREY, lw=1.2)
    b.annotate("実測の発現量 1690 分子/細胞", (1.25, 1950), color=GREY, fontsize=8.5)
    b.plot(lr1, b_scaf / (sc1 / P.N), color=S1C, lw=2.6)
    b.plot(lr2, b_scaf / (sc2 / P.N), color=S2C, lw=2.6)
    b.annotate("S1 は実測の発現量より\n少なくて足りる", (1.25, 700),
               color=S1C, fontsize=8.5)
    b.annotate("S2", (16, 2.2e4), color=S2C, fontsize=10, fontweight="bold")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlim(1.1, 40); b.set_ylim(3e2, 2e5)
    b.set_xlabel("達成した局在比")
    b.set_ylabel("必要な発現量 N [分子/細胞]")
    b.set_title("(b) 足場を基準の 109 分子に揃えるには", loc="left")

    # (c) そのとき要るプロテアーゼ処理量
    c = ax[2]
    c.axhspan(CLPXP_LO, CLPXP_HI, color=ENZ, alpha=0.22, lw=0)
    c.annotate("ClpXP の細胞全体の処理能力\n「数分子/s」（推定）", (1.25, 3.0),
               color="#9a6a1a", fontsize=8.5)
    lamN = b_scaf / (sc2 / P.N) / P.N
    c.plot(lr2, cf2 * lamN, color=S2C, lw=2.6)
    c.plot(lr1, np.zeros_like(lr1), color=S1C, lw=2.6)
    c.annotate("S1 は 0（切らない）", (1.3, 0.35), color=S1C, fontsize=9)
    c.annotate("S2", (16, 5.2), color=S2C, fontsize=10, fontweight="bold")
    c.set_xscale("log")
    c.set_xlim(1.1, 40); c.set_ylim(-0.4, 9)
    c.set_xlabel("達成した局在比")
    c.set_ylabel("必要なプロテアーゼ処理量 [分子/s]")
    c.set_title("(c) (b) の発現量で払う代償", loc="left")

    for a_ in ax:
        a_.set_xticks([2, 3, 5, 10, 20, 40])
        a_.set_xticklabels(["2", "3", "5", "10", "20", "40"])
        a_.xaxis.set_minor_formatter(NullFormatter())
        a_.set_xlabel("達成した局在比 [倍]")

    fig.text(0.5, -0.045,
             "足場も処理量も発現量 N に厳密に比例する（スケーリング対称性、"
             "results_cycle3.py §9）ので、(b)(c) は (a) から外挿で出せる。\n"
             "★ S2 は発現量を上げれば絶対量を取り戻せるが、"
             "そのぶんプロテアーゼの処理量も比例して増え、ClpXP の推定能力に当たる。",
             ha="center", fontsize=8.5, color="#444444")
    _save(fig, "fig4_absolute.png")



# ------------------------------------------------------------------- 図5
LAB.update({
    "f5title": {"ja": "欲しいものと、そのために払うもの",
                "en": "What you want, and what it costs"},
    "f5x": {"ja": "目標の局在比 [倍]（良い極 / 塞いだ極）",
            "en": "Target pole-to-pole ratio"},
    "f5ya": {"ja": "常時分解される DivIVA の割合 [%]",
             "en": "Fraction of DivIVA continuously destroyed [%]"},
    "f5yb": {"ja": "必要な発現量（実測 1690 分子/細胞 に対する比）",
             "en": "Required expression level (relative to measured 1690/cell)"},
    "f5a": {"ja": "(a) 払う代償 — 細胞の DivIVA をどれだけ壊すか",
            "en": "(a) The cost - how much DivIVA must be destroyed"},
    "f5b": {"ja": "(b) 同じ足場（109 分子）を良い極に置くのに要る発現量",
            "en": "(b) Expression needed for the same scaffold (109 molecules)"},
    "s1": {"ja": "S1 空間占有", "en": "S1 Steric occlusion"},
    "s2": {"ja": "S2 局所分解\n（クラスターも切れる場合）",
           "en": "S2 Local degradation\n(clusters also cleaved)"},
    "s2m": {"ja": "S2 単量体のみ", "en": "S2 monomers only"},
    "f5s1a": {"ja": "S1 は一切壊さない（0%）",
              "en": "S1 destroys nothing (0%)"},
    "f5s1b": {"ja": "S1 は実測より少ない発現量で足りる",
              "en": "S1 needs less than the measured level"},
    "f5meas": {"ja": "実測の発現量", "en": "measured level"},
    "f5note": {
        "ja": "★ 横軸は「設計目標」であって、モデルの操作量（θ や $k_{cut}$）ではない。"
              "単位の違う 2 つの機構を同じ軸で比べるために、操作量を消してある。\n"
              "(a) の割合は発現量 N に依らない（スケーリング対称性）。"
              "N を上げても薄まらず、壊す分子数が比例して増えるだけである。",
        "en": "The x axis is the design goal, not a model knob ($\\theta$ or $k_{cut}$). "
              "Eliminating the knob is what makes the two mechanisms comparable.\n"
              "The fraction in (a) is independent of expression level N: "
              "raising N does not dilute it, it only destroys proportionally more."},
})


def fig5():
    """★ 代償図 — 目標を横軸に置いて、2 戦略を同じ単位で比べる。

    S1 の θ（無次元 0〜1）と S2 の k_cut（1/s、桁で振れる）は単位が違うので
    直接は比べられない。そこで**操作量を消して**、設計目標（左右比）を横軸に、
    代償（壊す割合・必要発現量）を縦軸に置く。これで 1 枚に乗る。
    """
    base = popz.baseline(n=NG)
    b_scaf = base["scaffold_R"]
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    def scan(vals, mk):
        lr, ms, sc, w = [], [], [], y0
        for v in vals:
            r = popz.steady(NG, mk(v), y0=w)
            w = np.concatenate([r["C"], r["M"], r["Nf"]])
            lr.append(r["lr_clu"])
            ms.append(1.0 - r["mass_frac"])
            sc.append(r["scaffold_R"] / P.N)
        return np.array(lr), np.array(ms), np.array(sc)

    ths = np.concatenate([np.linspace(0.05, 0.9, 20), [0.93, 0.95, 0.97]])
    lr1, ms1, sc1 = scan(ths, lambda v: derive(P, theta=v))
    kcs = np.logspace(-3.2, -1.1, 22)
    lr2, ms2, sc2 = scan(kcs, lambda v: derive(P, theta=0.0, k_cut=v,
                                               cut_target="MN"))
    kms = np.logspace(0.0, 1.4, 10)
    lrm, msm, _ = scan(kms, lambda v: derive(P, theta=0.0, k_cut=v,
                                             cut_target="M"))

    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.4))
    XL = (1.4, 25)

    # (a) 壊す割合
    a = ax[0]
    a.plot(lrm, msm * 100, color=GREY, lw=1.8, ls="--")
    a.plot(lr2, ms2 * 100, color=S2C, lw=2.8)
    a.plot(lr1, ms1 * 100, color=S1C, lw=3.2)
    a.annotate(T("f5s1a"), (3.0, 4.0), color=S1C, fontsize=9.5,
               fontweight="bold")
    a.annotate(T("s2"), (11.5, 52), color=S2C, fontsize=9.5,
               fontweight="bold", ha="center")
    a.annotate(T("s2m"), (6.0, 93), color=GREY, fontsize=9)
    for tg in (2, 5, 10):
        v = np.interp(np.log(tg), np.log(lr2), ms2) * 100
        a.plot([tg], [v], "o", color=S2C, ms=6.5, mec="white", mew=1.2, zorder=5)
        a.annotate(f"{v:.0f}%", (tg, v), xytext=(tg * 0.80, v + 7),
                   color=S2C, fontsize=9, fontweight="bold")
        a.plot([tg], [0], "o", color=S1C, ms=6.5, mec="white", mew=1.2, zorder=5)
    a.set_ylim(-5, 104)
    a.set_ylabel(T("f5ya"))
    a.set_title(T("f5a"), loc="left")

    # (b) 必要な発現量
    b = ax[1]
    b.axhline(1.0, color=GREY, lw=1.2)
    b.annotate(T("f5meas"), (1.5, 1.06), color=GREY, fontsize=9)
    b.plot(lr2, b_scaf / sc2 / P.N, color=S2C, lw=2.8)
    b.plot(lr1, b_scaf / sc1 / P.N, color=S1C, lw=3.2)
    b.annotate(T("s1"), (11.0, 0.62), color=S1C, fontsize=9.5,
               fontweight="bold", ha="center")
    b.annotate(T("f5s1b"), (1.5, 0.45), color=S1C, fontsize=9)
    b.annotate(T("s2"), (3.1, 21.0), color=S2C, fontsize=9.5,
               fontweight="bold", ha="center")
    for tg in (2, 5, 10):
        v = np.interp(np.log(tg), np.log(lr2), b_scaf / sc2 / P.N)
        b.plot([tg], [v], "o", color=S2C, ms=6.5, mec="white", mew=1.2, zorder=5)
        b.annotate(f"x{v:.1f}", (tg, v), xytext=(tg * 0.76, v * 1.35),
                   color=S2C, fontsize=9, fontweight="bold")
        v1 = np.interp(np.log(tg), np.log(lr1), b_scaf / sc1 / P.N)
        b.plot([tg], [v1], "o", color=S1C, ms=6.5, mec="white", mew=1.2, zorder=5)
    b.set_yscale("log")
    b.set_ylim(0.35, 40)
    b.set_yticks([0.5, 1, 2, 5, 10, 20])
    b.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    b.yaxis.set_minor_formatter(NullFormatter())
    b.set_ylabel(T("f5yb"))
    b.set_title(T("f5b"), loc="left")

    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlim(*XL)
        a_.set_xticks([2, 3, 5, 10, 20])
        a_.set_xticklabels(["2", "3", "5", "10", "20"])
        a_.xaxis.set_minor_formatter(NullFormatter())
        a_.set_xlabel(T("f5x"))

    fig.suptitle(T("f5title"), x=0.09, ha="left", fontsize=12.5, y=1.02)
    fig.text(0.5, -0.07, T("f5note"), ha="center", fontsize=8.5,
             color="#444444")
    _save(fig, _fn("fig5_cost.png"))



# ------------------------------------------------------------------- 図6
LAB.update({
    "f6title": {"ja": "核形成の壁 — $p>1$ なら誘導の順序が結果を決める",
                "en": "The nucleation barrier - for $p>1$, induction order decides"},
    "f6a": {"ja": "(a) PopZ 抜きでも、$p>1$ では出発点が行き先を決める",
            "en": "(a) The starting point decides when $p>1$"},
    "f6b": {"ja": "(b) PopZ を 600 s かけて外す（$p$ = 1.2）",
            "en": "(b) A 600 s PopZ pulse, then release ($p$ = 1.2)"},
    "f6c": {"ja": "(c) 素からクラスター相に入れるか（$f_{clu}>0.1$）",
            "en": "(c) Can clusters form de novo? ($f_{clu}>0.1$)"},
    "f6yc": {"ja": "クラスター成長の次数 $p$", "en": "Growth exponent $p$"},
    "f6ok": {"ja": "届く", "en": "reached"},
    "f6ng": {"ja": "届かない（核形成の壁）", "en": "blocked (nucleation barrier)"},
    "f6seed": {"ja": "大きい種から", "en": "from a large seed"},
    "f6basin": {"ja": "種が大きいほど\n高い $p$ まで届く",
                "en": "larger seed,\nhigher $p$"},
    "f6xa": {"ja": "クラスター成長の次数 $p$", "en": "Growth exponent $p$"},
    "f6ya": {"ja": "定常のクラスター割合 $f_{clu}$",
             "en": "Steady-state cluster fraction $f_{clu}$"},
    "f6xb": {"ja": "時間 [世代]（1 世代 = 1500 s）",
             "en": "Time [generations] (1 generation = 1500 s)"},
    "f6yb": {"ja": "局在比（良い極 / 塞いだ極）",
             "en": "Pole-to-pole ratio"},
    "f6xc": {"ja": "核形成速度 $k_{nuc}$ [1/s]", "en": "Nucleation rate $k_{nuc}$ [1/s]"},
    "f6pre": {"ja": "クラスター込みの初期値から", "en": "from a pre-clustered state"},
    "f6den": {"ja": "素から（全分子が細胞質）", "en": "de novo (all cytoplasmic)"},
    "f6gap": {"ja": "二相に分かれる", "en": "two phases"},
    "f6ord1": {"ja": "DivIVA が先\n（逆順）", "en": "DivIVA first\n(reversed order)"},
    "f6ord2": {"ja": "PopZ が先\n（本プロジェクトの基本戦略）",
               "en": "PopZ first\n(the project's plan)"},
    "f6rel": {"ja": "PopZ を外す", "en": "PopZ released"},
    "f6stay": {"ja": "残る（足場 1377 分子）", "en": "latches (1377 molecules)"},
    "f6back": {"ja": "対称へ戻るどころか\nクラスターが一切立たない（足場 0.6 分子）",
               "en": "not just symmetric - no clusters at all\n(0.6 molecules)"},
    "f6win": {"ja": "$p>1$ が成立する窓", "en": "window where $p>1$ works"},
    "f6cal": {"ja": "校正の上限\n（これ以上は $k_{grow}>0$ の解が無い）",
              "en": "calibration limit\n(no $k_{grow}>0$ solution above)"},
    "f6def": {"ja": "既定", "en": "default"},
    "f6note": {
        "ja": "★ Cycle 2 の「$p>1$ なら一過的誘導でラッチが残る」は (b) の上の線、"
              "つまり DivIVA が先に局在している場合の結果だった。\n"
              "本プロジェクトの「PopZ を先に発現」では下の線になる。"
              "先行時間 $\\Delta t$ を 10 世代まで伸ばしても (b) の下の線は変わらない。",
        "en": "Cycle 2's \"a transient pulse latches when $p>1$\" is the upper curve "
              "in (b) - the case where DivIVA is already localized.\n"
              "The project's plan (PopZ first) gives the lower curve. Extending the "
              "lead time to 10 generations does not change it."},
})

CLU_C = "#e8a33d"     # クラスター相が生きている（figures2.py の CLU）
CLU_R = ["#f0c88a", "#e8a33d", "#c07f1f", "#8a5a13"]


def _staged(par, t_on=600.0, T_after=6e4, ic="cytoplasm"):
    """3 段階積分の軌跡をつなげて (世代, 局在比, 足場) で返す。

    ⚠️ RHS の中で時刻を見て切り替えてはいけない（設計判断 [C10]）。
    各段を別の積分として解き、終状態を次の初期値にする。
    """
    free = derive(par, theta=0.0, k_cut=0.0)
    y = popz.initial(NG, par, ic)
    r1 = popz.solve(NG, par, T=t_on, n_out=50, y0=y)
    y1 = np.concatenate([r1["C"], r1["M"], r1["Nf"]])
    r2 = popz.solve(NG, free, T=T_after, n_out=70, y0=y1)
    t = np.concatenate([r1["t"], t_on + r2["t"]]) / GEN
    out = []
    for r in (r1, r2):
        g, Nt = r["g"], r["Nt"]
        nL = (Nt * g["A_capL"][:, None]).sum(0)
        nR = (Nt * g["A_capR"][:, None]).sum(0)
        ok = (nL + nR) > 1e-3
        out.append((np.where(ok, r["lr_t"], np.nan), nR))
    return (t, np.concatenate([out[0][0], out[1][0]]),
            np.concatenate([out[0][1], out[1][1]]))


def fig6():
    """★ 核形成の壁と誘導順序 — Cycle 3 のもう一つの主結果。"""
    fig, ax = plt.subplots(1, 3, figsize=(14.6, 4.3))

    # (a) 核形成の壁 — 出発点（初期のクラスター量）で行き先が変わる
    #
    # ⚠️ 「p>1 で二相」という単純な話ではない。クラスター相の引き込み域が
    #   p とともに狭まり、必要な種が大きくなる。素から（種ゼロ）は p≈1.05 で
    #   届かなくなるが、大きい種なら p=1.4 でも届く。
    ps = [0.5, 0.7, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5]
    cur = {"cytoplasm": [], "uniform": [], "seed": []}
    for p in ps:
        q = derive(P, p=p)
        cur["cytoplasm"].append(popz.steady(NG, q, ic="cytoplasm")["f_clu"])
        cur["uniform"].append(popz.steady(NG, q, ic="uniform")["f_clu"])
        cur["seed"].append(popz.steady(NG, q, ic="seed_pole",
                                       seed=100.0)["f_clu"])
    # ⚠️ uniform（中間の種）は 2 本の間に入るだけで線が交錯するので描かない。
    a = ax[0]
    a.plot(ps, cur["seed"], color=CLU_C, lw=2.8, marker="o", ms=5)
    a.plot(ps, cur["cytoplasm"], color=GREY, lw=2.4, marker="o", ms=5,
           ls="--")
    a.annotate(T("f6seed"), (1.47, 1.35), color="#9a6a1a", fontsize=9,
               fontweight="bold", ha="right")
    a.annotate(T("f6den"), (0.50, 0.075), color=GREY, fontsize=9)
    a.annotate(T("f6basin"), (0.95, 0.0045), color="#9a6a1a", fontsize=8.5,
               ha="center")
    a.annotate("", xy=(1.44, 0.0012), xytext=(1.12, 0.0022),
               arrowprops=dict(arrowstyle="->", color="#9a6a1a", lw=1.0))
    a.axvline(1.0, color=GREY, lw=0.8, ls=":")
    a.set_yscale("log")
    a.set_ylim(3e-4, 2.5)
    a.set_xlim(0.45, 1.55)
    a.set_xlabel(T("f6xa"))
    a.set_ylabel(T("f6ya"))
    a.set_title(T("f6a"), loc="left")

    # (b) ラッチ
    b = ax[1]
    for ic, c, lw, lab, ann in (
            ("steady_free", CLU_C, 2.8, T("f6ord1"), T("f6stay")),
            ("cytoplasm", GREY, 2.4, T("f6ord2"), T("f6back"))):
        t, lr, sc = _staged(derive(P, p=1.2, theta=0.9), ic=ic)
        b.plot(t, lr, color=c, lw=lw)
    b.axvline(600.0 / GEN, color=GREY, lw=0.9, ls=":")
    b.annotate(T("f6rel"), (600.0 / GEN * 1.15, 1.2e5), color=GREY,
               fontsize=8.5, va="top", ha="left")
    b.annotate(T("f6ord1"), (28, 4.5e4), color="#9a6a1a", fontsize=9.5,
               fontweight="bold", ha="right")
    b.annotate(T("f6stay"), (28, 1.1e4), color="#9a6a1a", fontsize=8.5,
               ha="right")
    b.annotate(T("f6ord2"), (28, 7.5), color=GREY, fontsize=9.5,
               fontweight="bold", ha="right")
    b.annotate(T("f6back"), (28, 1.9), color=GREY, fontsize=8.5, ha="right",
               va="bottom")
    b.set_yscale("log")
    b.set_xscale("log")
    b.set_xlim(0.02, 40)
    b.set_ylim(0.8, 1e6)
    b.set_xlabel(T("f6xb"))
    b.set_ylabel(T("f6yb"))
    b.set_title(T("f6b"), loc="left")

    # (c) 素からクラスター相に入れるか — 相図（線より読みやすい）
    #
    # ⚠️ 双安定域では f_clu が非単調に跳ねる（上枝へ乗る）ので、
    #   値そのものを線で描くと読めない。「届いたかどうか」だけを示す。
    kns = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3.5e-3]
    pps = [1.0, 1.05, 1.1, 1.2, 1.3, 1.4]
    c = ax[2]
    c.axvspan(1e-3, 3.874e-3, color=CLU_C, alpha=0.18, lw=0, zorder=0)
    ok_x, ok_y, ng_x, ng_y = [], [], [], []
    for p in pps:
        for kn in kns:
            try:
                f = popz.steady(NG, derive(P, p=p, k_nuc=kn),
                                ic="cytoplasm")["f_clu"]
            except Exception:
                continue
            (ok_x if f > 0.1 else ng_x).append(kn)
            (ok_y if f > 0.1 else ng_y).append(p)
    c.scatter(ng_x, ng_y, s=58, facecolors="white", edgecolors=GREY,
              linewidths=1.4, zorder=3)
    c.scatter(ok_x, ok_y, s=58, color=CLU_C, edgecolors="white",
              linewidths=1.0, zorder=4)
    c.axvline(3.874e-3, color=GREY, lw=1.2, ls="--")
    c.annotate(T("f6cal"), (4.9e-3, 1.23), color=GREY, fontsize=8,
               ha="center", va="center", rotation=90)
    c.axvline(P.k_nuc, color=GREY, lw=0.9, ls=":")
    c.annotate(T("f6def"), (P.k_nuc * 1.3, 1.43), color=GREY, fontsize=8.5)
    c.annotate(T("f6win"), (1.9e-3, 1.44), color="#9a6a1a", fontsize=9,
               ha="center")
    c.scatter([2.2e-5], [1.33], s=58, color=CLU_C, edgecolors="white",
              linewidths=1.0, zorder=4)
    c.annotate(T("f6ok"), (3.2e-5, 1.33), color="#9a6a1a", fontsize=8.5,
               va="center")
    c.scatter([2.2e-5], [1.26], s=58, facecolors="white", edgecolors=GREY,
              linewidths=1.4, zorder=3)
    c.annotate(T("f6ng"), (3.2e-5, 1.26), color=GREY, fontsize=8.5,
               va="center")
    c.set_xscale("log")
    c.set_xlim(6e-6, 9e-3)
    c.set_ylim(0.96, 1.50)
    c.set_xlabel(T("f6xc"))
    c.set_ylabel(T("f6yc"))
    c.set_title(T("f6c"), loc="left")

    fig.suptitle(T("f6title"), x=0.065, ha="left", fontsize=12.5, y=1.03)
    fig.text(0.5, -0.07, T("f6note"), ha="center", fontsize=8.5,
             color="#444444")
    _save(fig, _fn("fig6_order.png"))



# ------------------------------------------------------------------- 図7
LAB.update({
    "f7title": {"ja": "同じ非対称を、足場をどれだけ残して達成できるか",
                "en": "How much scaffold survives at a given asymmetry"},
    "f7x": {"ja": "達成された左右比（優勢極 / 劣勢極）",
            "en": "Pole-to-pole ratio achieved (dominant / suppressed)"},
    "f7y": {"ja": "優勢極の足場 [クラスター分子数]",
            "en": "Scaffold at the dominant pole [molecules in clusters]"},
    "f7s1": {"ja": "S1 空間占有", "en": "S1 Steric occlusion"},
    "f7s2m": {"ja": "S2 局所分解", "en": "S2 Local degradation"},
    "f7unit": {"ja": "分子", "en": "molecules"},
})


def fig7():
    """★ 結果 第2部 の 1 枚 — 左右比 vs 優勢極の絶対分子数。

    ⚠️ 文字は表題・凡例・軸だけに絞ってある（wiki 本文側で説明するため）。
      クラスターも切れる場合の曲線と PopZ なしの基準線は、
      詳細節で使う別の図（fig4 / fig5）に任せる。
    """
    base = popz.baseline(n=NG)
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    def scan(vals, mk):
        lr, sc, w = [], [], y0
        for v in vals:
            r = popz.steady(NG, mk(v), y0=w)
            w = np.concatenate([r["C"], r["M"], r["Nf"]])
            lr.append(r["lr_clu"])
            sc.append(r["scaffold_R"])
        return np.array(lr), np.array(sc)

    ths = np.concatenate([np.linspace(0.02, 0.72, 20), [0.75]])
    lr1, sc1 = scan(ths, lambda v: derive(P, theta=v))
    kcm = np.logspace(-3.2, 1.0, 26)
    lr2, sc2 = scan(kcm, lambda v: derive(P, theta=0.0, k_cut=v,
                                          cut_target="M"))

    # 判定点（本文の表と同じ刻み）。
    # ⚠️ 2.2 は極濃縮（校正の目標値）であって左右比ではないので使わない。
    TGS = (2.0, 3.0)

    # ⚠️ 両曲線とも lr は k_cut / theta に対して単調増加。足場は桁で動くので
    #   log-log で補間する（線形補間だと下側の曲線が大きくずれる）。
    def _at(tg, lr, sc):
        return float(np.exp(np.interp(np.log(tg), np.log(lr), np.log(sc))))

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for tg in TGS:
        ax.axvline(tg, color=GREY, lw=1.1, ls=":", zorder=1)
    ax.plot(lr2, sc2, color=S2C, lw=3.0, zorder=4, label=T("f7s2m"))
    ax.plot(lr1, sc1, color=S1C, lw=3.0, zorder=4, label=T("f7s1"))

    for tg in TGS:
        for lr, sc, c, dy in ((lr1, sc1, S1C, 2.8), (lr2, sc2, S2C, 3.4)):
            v = _at(tg, lr, sc)
            ax.plot([tg], [v], "o", color=c, ms=8, mec="white", mew=1.5,
                    zorder=6)
            ax.annotate(f"{v:,.0f} {T('f7unit')}" if v >= 1
                        else f"{v:.3f} {T('f7unit')}",
                        (tg, v), xytext=(tg, v * dy), color=c,
                        fontsize=10.5, fontweight="bold", ha="center")

    ax.set_yscale("log")
    ax.set_xlim(0.97, 5.15)
    ax.set_ylim(1e-3, 3000)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel(T("f7x"))
    ax.set_ylabel(T("f7y"))
    ax.set_title(T("f7title"), loc="left", fontsize=12)

    h, l = ax.get_legend_handles_labels()
    ax.legend([h[1], h[0]], [l[1], l[0]], loc="center right", frameon=False,
              fontsize=10, handlelength=2.4, borderaxespad=1.2)
    _save(fig, _fn("fig7_scaffold.png"))



# ------------------------------------------------------------------- 図8
LAB.update({
    "f8title": {"ja": "クラスターも切れると仮定した場合",
                "en": "If clusters were also cleavable"},
    "f8s2n": {"ja": "S2 局所分解（クラスターも切断）",
              "en": "S2 Local degradation (clusters too)"},
    "f8s2m": {"ja": "S2 局所分解（単量体のみ）",
              "en": "S2 Local degradation (monomers only)"},
})


def fig8():
    """図7 に「クラスターも切れる場合」を足した版（結果 2-2 用）。

    体裁は図7 と揃える（文字は表題・凡例・軸と判定点の値だけ）。
    クラスターに関わる線なので色は琥珀（figures2.py の CLU）を使う。
    """
    base = popz.baseline(n=NG)
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    def scan(vals, mk):
        lr, sc, w = [], [], y0
        for v in vals:
            r = popz.steady(NG, mk(v), y0=w)
            w = np.concatenate([r["C"], r["M"], r["Nf"]])
            lr.append(r["lr_clu"])
            sc.append(r["scaffold_R"])
        return np.array(lr), np.array(sc)

    ths = np.concatenate([np.linspace(0.02, 0.72, 20), [0.75]])
    lr1, sc1 = scan(ths, lambda v: derive(P, theta=v))
    kcm = np.logspace(-3.2, 1.0, 26)
    lr2, sc2 = scan(kcm, lambda v: derive(P, theta=0.0, k_cut=v,
                                          cut_target="M"))
    kcn = np.logspace(-3.4, -1.0, 22)
    lr3, sc3 = scan(kcn, lambda v: derive(P, theta=0.0, k_cut=v,
                                          cut_target="MN"))

    TGS = (2.0, 3.0)

    def _at(tg, lr, sc):
        return float(np.exp(np.interp(np.log(tg), np.log(lr), np.log(sc))))

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    for tg in TGS:
        ax.axvline(tg, color=GREY, lw=1.1, ls=":", zorder=1)
    ax.plot(lr2, sc2, color=S2C, lw=3.0, zorder=4, label=T("f8s2m"))
    ax.plot(lr3, sc3, color=CLU_C, lw=3.0, zorder=5, label=T("f8s2n"))
    ax.plot(lr1, sc1, color=S1C, lw=3.0, zorder=4, label=T("f7s1"))

    for tg in TGS:
        for lr, sc, c, dy in ((lr1, sc1, S1C, 2.8),
                              (lr3, sc3, CLU_C, 0.32),
                              (lr2, sc2, S2C, 3.4)):
            v = _at(tg, lr, sc)
            ax.plot([tg], [v], "o", color=c, ms=8, mec="white", mew=1.5,
                    zorder=6)
            ax.annotate(f"{v:,.0f} {T('f7unit')}" if v >= 1
                        else f"{v:.3f} {T('f7unit')}",
                        (tg, v), xytext=(tg, v * dy), color=c,
                        fontsize=10.5, fontweight="bold", ha="center")

    ax.set_yscale("log")
    ax.set_xlim(0.97, 5.15)
    ax.set_ylim(1e-3, 3000)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel(T("f7x"))
    ax.set_ylabel(T("f7y"))
    ax.set_title(T("f8title"), loc="left", fontsize=12)

    h, l = ax.get_legend_handles_labels()
    ax.legend([h[2], h[1], h[0]], [l[2], l[1], l[0]], loc="center right",
              frameon=False, fontsize=9.5, handlelength=2.4,
              borderaxespad=1.0)
    _save(fig, _fn("fig8_cleavable.png"))



# ------------------------------------------------------------------- 図9
LAB.update({
    "f9title": {"ja": "被覆率が局在比を決める",
                "en": "Coverage sets the localization ratio"},
    "f9x": {"ja": "極の実効被覆率 $\\theta$（PopZ 極で結合が妨げられる割合）",
            "en": "Effective polar coverage $\\theta$"},
    "f9y": {"ja": "局在比（優勢極 / 劣勢極）",
            "en": "Pole-to-pole ratio"},
    "f9loc": {"ja": "極の結合能の比 $1/(1-\\theta)$",
              "en": "ratio of binding capacity $1/(1-\\theta)$"},
    "f9mod": {"ja": "モデルが出す局在比", "en": "model"},
})


def fig9():
    """★ 結果 3 用 — 実効被覆率 theta と局在比の対応。

    ⚠️ 文字は表題・凡例・軸・判定点だけに絞る（図7 と同じ方針）。
    """
    base = popz.baseline(n=NG)
    y0 = np.concatenate([base["C"], base["M"], base["Nf"]])

    ths = np.concatenate([np.linspace(0.0, 0.9, 28), [0.93, 0.95, 0.97, 0.98]])
    lr, w = [], y0
    for v in ths:
        r = popz.steady(NG, derive(P, theta=v), y0=w)
        w = np.concatenate([r["C"], r["M"], r["Nf"]])
        lr.append(r["lr_clu"])
    lr = np.array(lr)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    loc = 1.0 / (1.0 - np.clip(ths, 0, 0.985))
    ax.plot(ths, loc, color=GREY, lw=1.8, ls="--", zorder=2, label=T("f9loc"))
    ax.plot(ths, lr, color=S1C, lw=3.0, zorder=4, label=T("f9mod"))

    for tg in (2.0, 3.0):
        th = float(np.interp(tg, lr, ths))
        ax.plot([th, th], [1.0, tg], color=GREY, lw=1.1, ls=":", zorder=1)
        ax.plot([0.0, th], [tg, tg], color=GREY, lw=1.1, ls=":", zorder=1)
        ax.plot([th], [tg], "o", color=S1C, ms=8, mec="white", mew=1.5,
                zorder=6)
        ax.annotate(f"$\\theta$ = {th:.2f}", (th, tg),
                    xytext=(th - 0.02, tg * 1.30), color=S1C, fontsize=10.5,
                    fontweight="bold", va="center", ha="right")

    ax.set_yscale("log")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(1.0, 32)
    ax.set_yticks([1, 2, 3, 5, 10, 20, 30])
    ax.set_yticklabels(["1", "2", "3", "5", "10", "20", "30"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(T("f9x"))
    ax.set_ylabel(T("f9y"))
    ax.set_title(T("f9title"), loc="left", fontsize=12)
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[1], h[0]], [l[1], l[0]], loc="upper left", frameon=False,
              fontsize=9.5, handlelength=2.4, borderaxespad=1.0)
    _save(fig, _fn("fig9_theta.png"))


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6,
        7: fig7, 8: fig8, 9: fig9}

if __name__ == "__main__":
    args = list(sys.argv[1:])
    if "--en" in args:
        LANG = "en"
        args.remove("--en")
    if "--both" in args:
        args.remove("--both")
        want = [int(a) for a in args] or sorted(FIGS)
        for lg in ("ja", "en"):
            LANG = lg
            _style()
            print(f"  [{lg}]")
            for i in want:
                FIGS[i]()
        sys.exit(0)
    _style()
    want = [int(a) for a in args] or sorted(FIGS)
    print(f"  酵素帯: {ENZ_LO:.4g} 〜 {ENZ_HI:.4g} /s（中心 {ENZ_MID:.3g}）")
    print(f"  必要な窓: {NEED_LO:.4g} 〜 {NEED_HI:.4g} /s")
    for i in want:
        FIGS[i]()
