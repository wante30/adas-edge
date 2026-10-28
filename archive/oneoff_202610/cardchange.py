from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
s=s.replace('from q2_navigation_reader import Q2NavigationReader','from q2_navigation_reader import CARD_ROI, Q2NavigationReader')
s=s.replace('''        self.last_nav_thumb = None
        self.nav_ocr_skipped = 0''','''        self.last_nav_thumb = None
        self.nav_ocr_skipped = 0
        self.navigation_last_force_ns = 0''')
old='''            thumb = cv2.resize(cv2.cvtColor(navigation_image, cv2.COLOR_BGR2GRAY),
                               (32, 18), interpolation=cv2.INTER_AREA)
            current_nav = self.navigation.current(int(timestamp_ns))
            changed = True
            if self.last_nav_thumb is not None and current_nav.valid:
                diff = np.abs(thumb.astype(np.int16) - self.last_nav_thumb.astype(np.int16)).mean()
                changed = diff > 5.0
            self.last_nav_thumb = thumb
            self.navigation_last_update_ns = int(timestamp_ns)'''
new='''            x0, y0, x1, y1 = CARD_ROI
            card = navigation_image[y0:y1, x0:x1]
            thumb = cv2.resize(cv2.cvtColor(card, cv2.COLOR_BGR2GRAY),
                               (96, 45), interpolation=cv2.INTER_AREA)
            current_nav = self.navigation.current(int(timestamp_ns))
            changed = True
            if self.last_nav_thumb is not None and current_nav.valid:
                diff = np.abs(thumb.astype(np.int16) - self.last_nav_thumb.astype(np.int16)).mean()
                changed = diff > 3.0
            # Force a periodic full re-check even if the card looks static.
            force = timestamp_ns - self.navigation_last_force_ns >= 10_000_000_000
            if force:
                changed = True
                self.navigation_last_force_ns = int(timestamp_ns)
            self.last_nav_thumb = thumb
            self.navigation_last_update_ns = int(timestamp_ns)'''
if old not in s: raise SystemExit('thumb block')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('card roi change patched')
