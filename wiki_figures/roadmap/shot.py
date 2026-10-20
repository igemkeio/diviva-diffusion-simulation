from playwright.sync_api import sync_playwright
import sys
from pathlib import Path

# 使い方: python shot.py [出力PNG] [HTML名]（HTML は shot.py と同じディレクトリから読む。既定は roadmap.html）
HTML = Path(__file__).resolve().parent / (sys.argv[2] if len(sys.argv)>2 else 'roadmap.html')
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width':2240,'height':900}, device_scale_factor=2)
    pg.goto(HTML.as_uri())
    pg.wait_for_timeout(400)
    h = pg.evaluate("document.body.scrollHeight")
    pg.set_viewport_size({'width':2240,'height':int(h)+8})
    pg.wait_for_timeout(200)
    pg.screenshot(path=sys.argv[1] if len(sys.argv)>1 else 'fig_roadmap.png')
    b.close()
    print('height', h)
