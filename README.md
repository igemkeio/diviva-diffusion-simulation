# model-2 — DivIVA / PopZ 局在モデル

大腸菌の両極に局在する DivIVA を、片極に局在する PopZ で非対称化できるかを調べた
反応拡散モデルと、その wet 校正のコード一式である。

Wiki の [Model ページ](#) に対応する。4 つの DBTL サイクルがそのままディレクトリに
なっており、**各サイクルは独立に実行できる**。後のサイクルは前のサイクルのコードを
コピーして改変しているので、`model.py` や `cluster.py` は複数のサイクルに同名で
存在するが、中身は異なる（サイクルごとに固定したスナップショットである）。

| サイクル | 問い | 結論 | 主なスクリプト |
| --- | --- | --- | --- |
| **1** 単量体 2 場 | 曲率を単量体の膜結合速度に入れて両極局在を説明できるか | できない（エネルギー 231 倍不足、FRAP 2〜3 桁、天井 1/a） | `cycle1_monomer/code/model.py` |
| **2** クラスター場 | 曲率をクラスターの寿命に移すと 3 つの破綻が同時に解消するか | する（要求 92 分子ぶん、FRAP 二相） | `cycle2_cluster/code/cluster.py` |
| **3** PopZ | 空間占有（S1）と局所分解（S2）のどちらを採るか | S1。S2 は足場を削って左右比を買う | `cycle3_popz/code/popz.py` |
| **4** wet 校正 | 実測でモデルを較正しても判定は動かないか | 動かない。一致度 72%、$p \le 1$ が確定 | `cycle4_calibration/code/forward.py` |

## ディレクトリ構成

```
model-2/
├── cycle1_monomer/        code/  figure_code/  figures/  outputs/
├── cycle2_cluster/        code/  figure_code/  figures/  outputs/
├── cycle3_popz/           code/  figure_code/  figures/  outputs/
├── cycle4_calibration/    code/  figure_code/  figures/  outputs/  data/
└── wiki_figures/roadmap/  Wiki の DBTL ロードマップ図（HTML → PNG）
```

| | 中身 |
| --- | --- |
| `code/` | モデル本体・検証・数値計算。**図は作らない** |
| `figure_code/` | Wiki に載せた図を作るスクリプト。`code/` を import し、図は `figures/` に出す |
| `outputs/` | `code/` の実行ログ。本文の数値はここから引いている |
| `figures/` | 生成物。再生成できるので `.gitignore` で追跡していない |

`figure_code/` のスクリプトは**自分のファイル位置からパスを解決する**ので、どこから
実行しても `code/` を import し、図を `figures/` に出す。

## なぜこの構成か

モデルは段階的に複雑になる。各サイクルは**前のサイクルを特別な場合として含む**ように
作ってあり、それを回帰テストで保証している。

- Cycle 2 でクラスターの生成速度をゼロにすると Cycle 1 の答えを有効数字 10 桁で再現する
- Cycle 3 で PopZ をゼロにすると Cycle 2 の答えに相対差 0（厳密一致）で戻る

この性質があるので、**サイクル間の結果の違いを「足した場の有無だけ」に帰せる**。
コードを 1 本にまとめず、サイクルごとに凍結してあるのはこのためである。

## インストール

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 以上。`tifffile` は Cycle 4 で生画像を読むときだけ必要である。

## 実行

`code/` に入って実行する。図を作るスクリプトだけは `figure_code/` にあり、
出力先は各サイクルの `figures/` である。

### Cycle 1 — 単量体 2 場モデル

```bash
cd cycle1_monomer/code
python model.py                        # 検証 6 項目（約 20 秒）★ 先にこれを通す
python cycle1_numbers.py               # 本文の数値をすべて再計算（約 1.5 分）
python ../figure_code/figures_cycle1.py   # 図 A〜K（約 2 分）。末尾に A C で個別指定も可
```

`model.py` が落ちたら以降の数字は信用しない。とくに **V3（質量保存、$7.9\times10^{-14}$）**と
**V4（$\rho=1$ で極濃縮がちょうど 1）**が通ることが前提である。
出力例は `outputs/cycle1_out.txt`。

### Cycle 2 — クラスター場

```bash
cd cycle2_cluster/code
python cluster.py                      # 検証 13 項目（V7 = Cycle 1 への回帰テストを含む）
python results2.py                     # 本文の数値
python ../figure_code/figures2.py      # 図 1〜8。末尾に 3 5 で個別指定も可
python ../figure_code/figures_notion.py   # ★ Wiki に貼った c2fig1〜6（figures2.py の後に実行）
python ../figure_code/fig_sideview.py  # 横から見た細胞の静止画と GIF
```

⚠️ **`figures_notion.py` は `figures2.py` の出力を流用する**（6 枚中 5 枚は改名コピー、
自前で描くのは `c2fig5_bistability.png` だけ）。順番を逆にすると落ちる。
出力例は `outputs/results2_out.txt`、`outputs/figures2_out.txt`。

### Cycle 3 — PopZ（S1 空間占有 / S2 局所分解）

```bash
cd cycle3_popz/code
python popz.py                         # 検証 36 項目（約 4 分）★ VR1 が通ることが前提
python results_cycle3.py               # 判定に使う数値
python ../figure_code/figures_cycle3.py     # 図 1〜9（_en 付きは英語版）
python ../figure_code/fig_occupancy_gif.py  # θ = 0.659（左右比 5 倍）の時間発展 GIF
```

出力例は `outputs/results_cycle3_out.txt`、`outputs/validate_cycle3_out.txt`。

### Cycle 4 — wet データでの校正

```bash
cd cycle4_calibration/code
python forward.py                      # 前向きモデル（投影 → PSF → 画素）の自己診断
python cycle5.py                       # 本文の数値をまとめて再計算 ⚠️ 下の「既知の問題」を参照
python final_p.py                      # 成長の次数 p の推定（α = 1.076 ± 0.061）
python s1s2_curves.py                  # fig_s1s2.py が読む figures/_s1s2_curves.json を作る
python ../figure_code/fig_model_vs_image.py   # 模擬画像と実画像の比較
python ../figure_code/fig_agreement.py        # 較正前後の一致度（48% → 72%）
```

データは `cycle4_calibration/data/` にある。パスはファイル位置から解決するので
作業ディレクトリは問わない。

## Wiki の図とスクリプトの対応

★ は Wiki（Model ページ / DBTL ページ）に貼った図である。同じスクリプトが
貼らなかった図も作る。

| 図 | スクリプト |
| --- | --- |
| ★ `figE_energy.png`、★ `figK_total_sideview.png/.gif`、`figA`〜`figJ` | `cycle1_monomer/figure_code/figures_cycle1.py` |
| ★ `c2fig1_energy.png`、`c2fig2`〜`c2fig6` | `cycle2_cluster/figure_code/figures_notion.py`（＋ `figures2.py`） |
| ★ `c2figK_total_sideview.png/.gif` | `cycle2_cluster/figure_code/fig_sideview.py` |
| ★ `fig8_cleavable_en.png`、★ `fig9_theta_en.png`、`fig1`〜`fig7` | `cycle3_popz/figure_code/figures_cycle3.py` |
| ★ `c3figK_occupancy.png/.gif` | `cycle3_popz/figure_code/fig_occupancy_gif.py` |
| ★ `cal_fig6_model_vs_image.png` | `cycle4_calibration/figure_code/fig_model_vs_image.py` |
| ★ `cal_fig_agreement.png` | `〃/fig_agreement.py` |
| ★ `cal_fig_alpha.png` | `〃/fig_alpha_fit.py` |
| ★ `cal_fig_baseline.png` | `〃/fig_baseline_membrane.py` |
| ★ `cal_fig5_iptg_expression.png` | `〃/fig_iptg_expression.py` |
| ★ `cal_fig7_membrane_baseline.png` | `〃/fig_membrane_baseline.py` |
| ★ `cal_fig2_validation.png` | `〃/fig_validation.py` |
| `cal_fig_s1_theta.png`、`cal_fig_s1s2_scaffold.png` | `〃/fig_s1s2.py`（先に `code/s1s2_curves.py`） |
| ★ `fig_roadmap.png` / `fig_roadmap_ja.png` | `wiki_figures/roadmap/`（`shot.py`） |

## データ

`cycle4_calibration/data/` に解析済みの表と、図に使う生画像を 2 枚だけ置いてある。
撮像と細胞ごとの輝度抽出は別リポジトリ（[ecoli-imaging](https://github.com/igemkeio/ecoli-imaging)）
で行っており、ここにあるのはその出力である。出どころと md5 は `data/SOURCES.md` を参照。

| ファイル | 中身 |
| --- | --- |
| `cells_final_w5.csv` | 細胞ごとの線プロファイル（線幅 5 px）と指標。1643 細胞 |
| `cells_integrated.csv` | ROI の座標・細胞長・SNR など、切り出しに要る情報 |
| `cells_profiles_wide_w3.csv` | 同じものを線幅 3 px で取ったもの（線幅依存の検証用） |
| `pool_features.csv` | 手選び前の母集団の特徴量（選択バイアスの確認用） |
| `1010_mc_Div/5/5aa.tif` | DivIVA-mCherry（IPTG 10 µM）の生画像 |
| `1010_FPnull/mche/mche4.tif` | 遊離 mCherry（局在しない対照）の生画像 |

⚠️ **`cells_final_w5.csv` の `end_mid` 列は max(poleA, poleB) である。**
両極の平均が要るときは `poleA_mid` と `poleB_mid` から取り直すこと。
本文の数値はすべて取り直した値を使っている。

## 自分の系に合わせるには

| 変えるもの | どこ |
| --- | --- |
| 細胞の幾何（$R$, $L$） | 各サイクルの `model.py` の `Par` |
| 速度定数 | 同上（Cycle 2 以降は `cluster.py` の `Par`） |
| PopZ の被覆（$\theta$, $r_{popz}$） | `popz.py` の `Par.theta` / `Par.r_popz` |
| 画素サイズ・PSF | `forward.py` の `PX_UM` と、各スクリプトの `PSF` |

$\rho$, $\rho_n$, $k_{on}$, $k_{grow}$ は**独立なパラメータではない**。
実測の分配比と極割合から毎回逆算される従属量なので、直接書き換えてはいけない。

## 結果が妥当かの確認

どのサイクルも、結果を出す前に検証を通す設計になっている。

| | Cycle 1 | Cycle 2 | Cycle 3 |
| --- | --- | --- | --- |
| 項目数 | 6 | 13 | 36 |
| 質量保存 / 収支 | $7.9\times10^{-14}$ | $6.1\times10^{-13}$ | $2.0\times10^{-16}$ |
| 前サイクルへの回帰 | — | $1.5\times10^{-11}$ | 相対差 0 |

## 既知の問題

- **`cycle5.py` はこのリポジトリのデータだけでは動かない。**`wetdata.py` が
  `anl/extracted/profiles_w3.csv`（1 画素 1 行の long 形式）を読むが、`data/` に
  入っているのは 1 細胞 1 行の wide 形式（`cells_profiles_wide_w3.csv`）である。
  `WET_DIR` に元の `anl/` を指すか、long 形式の csv を `data/` に足す必要がある
- **`cycle4_calibration/outputs/` が空である。**Cycle 4 の数値は標準出力にしか
  残っていない。`python cycle5.py > ../outputs/results5_out.txt` のように保存すること
- `fit_R_short_axis.py` は末尾の保存処理の括弧が壊れていたので直した（2026-10-21）。
  印字される短軸フィットの値（R と PSF）は変わらない

## 既知の制約

- **隔壁（セプタム）を幾何に持たない。**校正の目標値「クラスター分子の 66% が極・隔壁」を
  すべて極に帰しているので、極の効果を過大に評価している
- **クラスターのサイズ分布を持たない。**面密度として扱っているので、実画像にある
  粒状の構造は再現できない
- **速度定数はすべて *B. subtilis* 由来。**大腸菌に DivIVA は存在せず、置き換える先が無い
- **wet 側に独立な繰り返しが無い。**条件はすべて IPTG 濃度の違いであって反復ではない

## ライセンス

MIT。
