"""Wet（anl/）の線プロファイルを読み、Cycle 5 のモデル比較用にそろえる。

    wet 側の生成物を読むだけで、wet の解析はやり直さない。
    ⚠️ 指標の定義は forward.py（= anl/scripts/metrics.py）と同一。
"""

import os
import numpy as np
import pandas as pd

WET = os.environ.get(
    "WET_DIR", "/mnt/user-data/uploads/Igem数値解析/anl")
PX_UM = 0.1625


def load():
    """profiles_w3 と cells_integrated を結合して返す。"""
    pr = pd.read_csv(os.path.join(WET, "extracted", "profiles_w3.csv"))
    ci = pd.read_csv(os.path.join(WET, "results", "cells_integrated.csv"))
    return pr, ci


def mean_profile(pr, ci, condition, channel="divIVA_or_fp",
                 zlo=-0.25, zhi=1.25, nz=141, min_snr=None, fold=False):
    """条件ごとの平均プロファイル（背景を引き、中央の平均で規格化）。

    各細胞で
      1. 線の外側 12% の中央値を背景として引く
      2. z = (線上の位置 − z0_px)/L_px（DIC 由来の細胞長で規格化）
      3. 中央 z 0.40–0.60 の平均で割る
      4. 共通の z 格子へ内挿
    したうえで細胞間の中央値を取る。

    fold=True なら z と 1−z を折り返して平均（左右対称化）。
    """
    sel = ci[(ci.condition == condition) & (ci.cell_in_line == True)]
    if min_snr is not None:
        sel = sel[sel.snr_prof >= min_snr]
    uids = set(sel.cell_uid)
    sub = pr[pr.cell_uid.isin(uids)]
    info = sel.set_index("cell_uid")[["z0_px", "L_px"]].to_dict("index")

    zg = np.linspace(zlo, zhi, nz)
    rows = []
    for uid, gdf in sub.groupby("cell_uid", sort=False):
        v = gdf[channel].to_numpy(float)
        if not np.isfinite(v).all() or len(v) < 20:
            continue
        k = max(2, int(round(len(v) * 0.12)))
        bg = np.median(np.r_[v[:k], v[-k:]])
        s = np.clip(v - bg, 0, None)
        d = info[uid]
        z = (np.arange(len(v)) - d["z0_px"]) / d["L_px"]
        mid = (z >= 0.40) & (z <= 0.60)
        if mid.sum() < 3 or s[mid].mean() <= 0:
            continue
        rows.append(np.interp(zg, z, s / s[mid].mean(), left=np.nan, right=np.nan))
    P = np.array(rows)
    if fold and len(P):
        Pf = np.interp(zg, zg, np.nanmedian(P, axis=0))
        back = np.interp(1.0 - zg, zg, Pf, left=np.nan, right=np.nan)
        med = np.nanmean(np.vstack([Pf, back]), axis=0)
    else:
        med = np.nanmedian(P, axis=0) if len(P) else np.full(nz, np.nan)
    return dict(z=zg, med=med, n=len(P), all=P,
                lo=np.nanpercentile(P, 25, axis=0) if len(P) else None,
                hi=np.nanpercentile(P, 75, axis=0) if len(P) else None)


def cond_summary(ci):
    """条件ごとの n・細胞長・SNR・主要指標の中央値。"""
    g = ci[ci.cell_in_line == True].groupby("condition")
    return pd.DataFrame(dict(
        n=g.size(),
        L_um=g.cell_len_um.median(),
        snr=g.snr_prof.median(),
        end_mid=g.end_mid.median(),
        pole_mid=g.pole_mid.median(),
        f20=g.f20.median(),
    )).sort_values("n", ascending=False)
