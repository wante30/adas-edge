import json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"ai"))
from q2_navigation_reader import Q2NavigationReader as R
for i in (0,3,8,15,17,18):
 im=cv2.imread(str(ROOT/f"evidence/ocr_corpus/{i}.png"));x,y,w,h=R._locate_card(im);card=im[y:y+h,x:x+w];arrow=R._arrow_in_card(card)
 if not arrow:continue
 box=arrow[3];left=int(box[0]+box[2]+5);top=max(0,int(box[1]-12));right=min(card.shape[1],left+170);bottom=min(card.shape[0],int(box[1]+box[3]+12))
 cv2.imwrite(str(ROOT/f"evidence/crop_{i}.png"),card[top:bottom,left:right]);print(i,(x,y,w,h),box,(left,top,right,bottom))
