# DBTL ロードマップ図

Wiki の Model ページに貼った、4 つの DBTL サイクルと wet の流れの図。
HTML で組んで Chromium のスクリーンショットで PNG にしている（matplotlib では
文字組みが詰められないため）。

```
roadmap.html      英語版
roadmap_ja.html   日本語版（確認用）
shot.py           HTML → PNG（Playwright + Chromium、device_scale_factor=2）
```

## 作り直し方

```bash
pip install playwright && playwright install chromium
python shot.py fig_roadmap.png                       # 英語版（roadmap.html）
python shot.py fig_roadmap_ja.png roadmap_ja.html    # 日本語版
```

出力は 4480 × 約 1670 px（CSS 2240 px の 2 倍）。

- 上段 = DRY（Cycle 1〜4）、下段 = WET。レーンの左端は Cycle 3 の左端に合わせてある
- 3 つ目の wet（発現・撮影・計測）が Cycle 4 の真下に来る位置関係が時系列を表している
- 4 つ目（次の設計）は破線＝まだ実施していない。Cycle 4 の右辺から折れ矢印で入る

日本語版は IPAPGothic を使う。フォントが無い環境では `font-family` を手元のゴシックに
差し替えること。
