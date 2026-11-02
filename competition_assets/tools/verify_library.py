"""Check native PNG integrity, real-source quotas, links and gallery interactions."""
from pathlib import Path
import collections,hashlib,json
from PIL import Image
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
m=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'));rows=m['candidates'];report={'counts':m['counts'],'sources':dict(collections.Counter(r['source_type'] for r in rows)),'failures':[]}
assert 200<=m['counts']['raw_candidates']<=400 and 40<=m['counts']['curated']<=60 and 15<=m['counts']['core']<=25
selected=[r for r in rows if r.get('topic')];assert len({r['sha256'] for r in selected})==len(selected)
assert any(r.get('core_rank') and r.get('topic')=='05_hazard_warning' and r['visual_review'].get('visible_critical') for r in selected),'Core must include a visually verified red camera alert'
for r in rows:
    p=ROOT/r['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    with Image.open(p) as im:im.load()
    if r in selected:
        assert r['source_type']=='REAL_PCIE_LIVE' and not r['exclusions']
        for k in ('originals','ppt_16x9','focus_16x9','project_book'):
            p=ROOT/r['exports'][k]
            with Image.open(p) as im:
                im.load();assert im.size==((1240,615) if k=='project_book' else (1280,720))
            if k in ('originals','ppt_16x9'):assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
with sync_playwright() as pw:
    browser=pw.chromium.launch(channel='chrome',headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1040});errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((ROOT/'index.html').as_uri());page.wait_for_selector('article');assert page.locator('article').count()==m['counts']['core']
    page.select_option('#tier','curated');assert page.locator('article').count()==m['counts']['curated']
    page.select_option('#topic','03_multicamera_detection');assert page.locator('article').count()==12
    page.select_option('#tier','all');page.select_option('#topic','');assert page.locator('article').count()==m['counts']['raw_candidates']
    page.locator('#search').fill('FAST-Q4');assert page.locator('article').count()>0
    page.locator('#search').fill('');page.select_option('#tier','core');page.locator('article img').first.click();assert page.locator('dialog').is_visible();page.locator('dialog button').click()
    page.wait_for_function("[...document.querySelectorAll('article img')].slice(0,3).every(x=>x.complete && x.naturalWidth>0)")
    page.screenshot(path=str(ROOT/'evidence/gallery_preview.png'),full_page=True)
    report['gallery']={'core_filter':True,'curated_filter':True,'all_candidates_filter':True,'topic_filter':True,'search':True,'image_dialog':True,'page_errors':errors};assert not errors
    browser.close()
report['all_checks_passed']=True
(ROOT/'evidence/library_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))