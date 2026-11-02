"""Deduplicate unchanged Q2 instruction cards, retaining actual OCR freshness."""
import cv2
import numpy as np

class NavigationGate:
    def __init__(self):
        self.reference=None
        self.submitted_ns=0

    def wants(self,image,timestamp_ns,valid,ocr_timestamp_ns):
        # CARD_ROI is defined for 960x540; source Q2 is 640x360.
        h,w=image.shape[:2]
        x0,y0,x1,y1=[int(round(v*s)) for v,s in zip((45,105,460,300),(w/960,h/540,w/960,h/540))]
        card=image[y0:y1,x0:x1]
        thumb=cv2.resize(cv2.cvtColor(card,cv2.COLOR_BGR2GRAY),(96,45),interpolation=cv2.INTER_AREA)
        changed=self.reference is None
        if self.reference is not None:
            diff=np.abs(thumb.astype(np.int16)-self.reference.astype(np.int16))
            # Local patches catch a changed distance digit hidden by a global mean.
            changed=float(diff.mean())>.5 or float(diff.reshape(9,5,12,8).mean(axis=(1,3)).max())>3
        fresh=valid and isinstance(ocr_timestamp_ns,int) and 0<=timestamp_ns-ocr_timestamp_ns<1_000_000_000
        return changed or not fresh

    def submitted(self,image,timestamp_ns):
        h,w=image.shape[:2]
        x0,y0,x1,y1=[int(round(v*s)) for v,s in zip((45,105,460,300),(w/960,h/540,w/960,h/540))]
        self.reference=cv2.resize(cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY),(96,45),interpolation=cv2.INTER_AREA)
        self.submitted_ns=timestamp_ns
