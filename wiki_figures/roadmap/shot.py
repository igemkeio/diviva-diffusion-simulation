from playwright.sync_api import sync_playwright
import sys
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={'width':2240,'height':900}, device_scale_factor=2)
    pg.goto('file:///home/claude/roadmap/roadmap_ja.html')
    pg.wait_for_timeout(400)
    h = pg.evaluate("document.body.scrollHeight")
    pg.set_viewport_size({'width':2240,'height':int(h)+8})
    pg.wait_for_timeout(200)
    pg.screenshot(path=sys.argv[1] if len(sys.argv)>1 else 'fig_roadmap.png')
    b.close()
    print('height', h)
