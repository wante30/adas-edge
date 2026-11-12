"""Full-video stage-two QAT inference, unmodified masks, H.264 and audit logs."""
import argparse
import csv
from fractions import Fraction
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import torch

from . import CLASSES
from .dataset_adapter import file_hash, letterbox, read_json, restore, write_json

ROOT=Path(__file__).resolve().parents[3]
# RGB: required colors, independent overlapping predictions.
COLORS=np.array([(0,220,60),(255,40,40),(0,230,255),(235,40,235)],np.uint8)
LABELS=['可行驶区域','实线','虚线','斑马线区域']
VIDEO_NAMES=['road_seg_overlay.mp4','road_seg_comparison.mp4','road_seg_fp32_int8_comparison.mp4']


def probe(path,count=False):
    command=['ffprobe','-v','error']
    if count:command+=['-count_frames']
    command+=['-show_streams','-show_format','-of','json',str(path)]
    return json.loads(subprocess.check_output(command,encoding='utf-8'))


def sessions(model_root):
    model_root=Path(model_root).resolve();search=read_json(model_root/'exports/search.json')
    if search['selected']!='qat':raise ValueError('Selected final model is not QAT')
    selected=next(v for v in search['variants'] if v['name']=='qat')
    files=dict(fp32=model_root/'exports/model.fp32.onnx',int8=model_root/'exports/model.int8.qdq.onnx',
               fp32_checkpoint=model_root/'checkpoints/candidate-plus/best.pt',
               qat_checkpoint=model_root/'checkpoints/candidate-plus-qat/qat.best.pt')
    hashes={k:file_hash(v) for k,v in files.items()}
    if hashes['int8']!=selected['sha256'] or hashes['int8']!=file_hash(model_root/'exports/qat.int8.qdq.onnx'):
        raise ValueError('QAT ONNX identity mismatch')
    if hashes['fp32_checkpoint']!=search['checkpoint_sha256']:
        raise ValueError('Wrong FP32 checkpoint')
    ck=torch.load(files['qat_checkpoint'],map_location='cpu',weights_only=False)
    if ck['fp32_checkpoint_sha256']!=hashes['fp32_checkpoint']:
        raise ValueError('QAT checkpoint did not originate from selected stage-two FP32')
    options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
    result={k:ort.InferenceSession(str(files[k]),options,providers=['CPUExecutionProvider']) for k in ('fp32','int8')}
    for runtime in result.values():
        if runtime.get_inputs()[0].shape!=[1,3,192,320] or runtime.get_outputs()[0].shape!=[1,4,192,320]:
            raise ValueError('Unexpected model shape')
    evidence=dict(paths={k:str(v) for k,v in files.items()},sha256=hashes,selected='qat',qat_epoch=ck['epoch'],
        input=[1,3,192,320],color='RGB',normalization='uint8 / 255',letterbox='bilinear image resize, fill 114',
        postprocess='sigmoid; undo letterbox and bilinear restore probabilities; threshold >=0.5 independently',
        threshold=.5,temporal_smoothing=False,mask_cleanup=False,ort_version=ort.__version__,
        backend='ONNX Runtime CPUExecutionProvider, 4 intra-op/1 inter-op threads for both FP32 and QAT',
        fp32_reference='Stage-two optimized FP32 baseline before QAT; comparison includes QAT fine-tuning as well as quantization')
    return result,evidence


@lru_cache(maxsize=8)
def font(size):
    for path in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf']:
        if Path(path).exists():return ImageFont.truetype(path,size)
    return ImageFont.load_default()


@lru_cache(maxsize=32)
def banner(width,title,legend=True):
    height=86 if legend else 42
    im=Image.new('RGB',(width,height),(12,15,20));draw=ImageDraw.Draw(im);f=font(24)
    draw.text((14,5),title,font=f,fill='white')
    if legend:
        for c,label in enumerate(LABELS):
            x=14+c*(width//4)
            draw.rectangle((x,49,x+25,72),fill=tuple(map(int,COLORS[c])))
            draw.text((x+34,43),label,font=f,fill='white')
    return np.asarray(im)


def overlay(rgb,masks):
    color=rgb.copy()
    for c in range(4):color[masks[c]]=COLORS[c]
    return cv2.addWeighted(rgb,.55,color,.45,0)


def canvases(rgb,fp,iq,index,fps):
    h,w=rgb.shape[:2];overlay_i=overlay(rgb,iq);overlay_f=overlay(rgb,fp)
    title=f'第二阶段 Candidate-plus INT8 QAT | 帧 {index+1} | {index/fps:.2f}s | 原始逐帧输出'
    top=banner(w,title);overlay_i[:len(top)]=top
    # All panels retain the source 16:9 aspect; whole MP4 remains source W x H.
    tw,th=w//3,h//2
    grid=np.zeros_like(rgb)
    panels=[(rgb,'原视频'),(overlay_i,'INT8 QAT 四类叠加')]
    for c in range(4):
        mask=np.zeros_like(rgb);mask[iq[c]]=COLORS[c]
        panels.append((mask,LABELS[c]+' 独立 Mask'))
    for j,(panel,label) in enumerate(panels):
        header=banner(tw,label,False)
        scale=min(tw/w,(th-len(header))/h)
        pw,ph=round(w*scale),round(h*scale)
        tile=np.zeros((th,tw,3),np.uint8)
        x,y=(tw-pw)//2,len(header)+(th-len(header)-ph)//2
        tile[y:y+ph,x:x+pw]=cv2.resize(panel,(pw,ph),interpolation=cv2.INTER_AREA if j<2 else cv2.INTER_NEAREST)
        tile[:len(header)]=header
        grid[j//3*th:(j//3+1)*th,j%3*tw:(j%3+1)*tw]=tile
    diff=np.zeros_like(rgb)
    for c in range(4):diff[fp[c]!=iq[c]]=COLORS[c]
    compare=np.zeros_like(rgb);cw,ch=w//2,h//2
    for j,(panel,label) in enumerate([(rgb,'原视频'),(overlay_f,'第二阶段 FP32'),
                                     (overlay_i,'最终 INT8 QAT'),(diff,'FP32 / QAT 阈值分歧（非真值错误）')]):
        tile=cv2.resize(panel,(cw,ch),interpolation=cv2.INTER_AREA)
        header=banner(cw,label,False);tile[:len(header)]=header
        compare[j//2*ch:(j//2+1)*ch,j%2*cw:(j%2+1)*cw]=tile
    return [overlay_i,grid,compare]


class Encoder:
    def __init__(self,path,source,width,height,fps,frames):
        self.path=Path(path);self.pending=self.path.with_name(self.path.stem+'.encoding.mp4')
        if self.path.exists() or self.pending.exists():raise ValueError(f'video exists: {path}; preserve it and use a new output directory')
        self.log=self.path.with_suffix('.ffmpeg.log').open('wb')
        command=['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
            '-video_size',f'{width}x{height}','-framerate',fps,'-i','pipe:0','-i',str(source),
            '-map','0:v:0','-map','1:a:0?','-frames:v',str(frames),'-c:v','libx264','-preset','veryfast',
            '-crf','18','-threads','2','-pix_fmt','yuv420p','-profile:v','high','-level:v','4.2',
            '-c:a','copy','-movflags','+faststart',str(self.pending)]
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=self.log)
        self.write_seconds=0.

    def write(self,rgb):
        start=time.perf_counter();self.process.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.write_seconds+=time.perf_counter()-start

    def finish(self):
        start=time.perf_counter();self.process.stdin.close();code=self.process.wait();self.log.close()
        if code:raise RuntimeError(f'ffmpeg failed for {self.path}, exit={code}; inspect encoder log')
        os.replace(self.pending,self.path)
        return dict(pipe_write_backpressure_seconds=self.write_seconds,encoder_flush_seconds=time.perf_counter()-start)

    def abort(self):
        if self.process.poll() is None:
            try:self.process.stdin.close()
            except OSError:pass
            self.process.terminate();self.process.wait()
        self.log.close()


def statistics(values):
    a=np.asarray(values)
    return dict(mean_ms=float(a.mean()),p50_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),
                sum_seconds=float(a.sum()/1000))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model-root',default=str(ROOT/'outputs/fpga_road_seg_stage2_qat'))
    a=p.parse_args();source=Path(a.video).resolve();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True);(out/'frames').mkdir(exist_ok=True);(out/'cache').mkdir(exist_ok=True)
    if any((out/n).exists() for n in VIDEO_NAMES):raise ValueError('Completed video outputs already exist; no overwrite')
    torch.set_num_threads(2);cv2.setNumThreads(2)
    start=time.perf_counter();meta=probe(source,True);video=next(s for s in meta['streams'] if s['codec_type']=='video')
    w,h=video['width'],video['height'];n=int(video['nb_read_frames']);fps_text=video['avg_frame_rate'];fps=float(Fraction(fps_text))
    if w%6 or h%4:raise ValueError('Comparison grid currently requires source width divisible by 6 and height by 4')
    if video['avg_frame_rate']!=video['r_frame_rate']:raise ValueError('Variable-rate source requires explicit timestamp-preserving path')
    if shutil.disk_usage(out).free < n*2*4*192*320*4+2_000_000_000:raise OSError('Insufficient output/cache disk space')
    runtimes,evidence=sessions(a.model_root)
    evidence.update(source=str(source),source_sha256=file_hash(source),source_probe=meta,
                    colors_rgb={CLASSES[c]:COLORS[c].tolist() for c in range(4)},
                    native_size=[w,h],frames=n,fps=fps_text,alpha=.45,
                    overlay_precedence=list(CLASSES),note='Overlapping independent channels; later colors take display precedence only')
    write_json(out/'model_and_video.json',evidence)
    print(f'verified final QAT={evidence["sha256"]["int8"]} video={w}x{h} frames={n} fps={fps_text}',flush=True)
    identity=dict(source_sha256=evidence['source_sha256'],model_hashes=evidence['sha256'],shape=[n,2,4,192,320],dtype='float32')
    cache=out/'cache/probabilities.npy';cache_done=out/'cache/complete.json'
    if cache_done.exists():
        if read_json(cache_done)!=identity:raise ValueError('Cached model/video identity mismatch')
        probabilities=np.lib.format.open_memmap(cache,mode='r')
        inference_rows=list(csv.DictReader((out/'inference_per_frame.csv').open(encoding='utf-8')))
        print('reusing all-frame inference cache with exact model/video hashes',flush=True)
    else:
        if cache.exists():raise ValueError('Incomplete inference cache preserved; use a new output directory')
        probabilities=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(n,2,4,192,320))
        cap=cv2.VideoCapture(str(source));inference_rows=[]
        try:
            with (out/'inference_per_frame.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['frame','time_seconds','decode_ms','preprocess_ms','fp32_model_ms','int8_model_ms','probability_cache_ms'])
                writer.writeheader()
                for i in range(n):
                    t=time.perf_counter();ok,bgr=cap.read();decode=(time.perf_counter()-t)*1000
                    if not ok:raise RuntimeError(f'Premature source decode failure at frame {i}/{n}')
                    if bgr.shape[:2]!=(h,w):raise ValueError('Source frame size changed')
                    t=time.perf_counter();rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);x,g=letterbox(rgb);inputs={'rgb':x[None].numpy()};pre=(time.perf_counter()-t)*1000
                    if i==0:
                        for runtime in runtimes.values():
                            for _ in range(20):runtime.run(None,inputs)
                        write_json(out/'geometry.json',g)
                    timing={};logits=[]
                    for name in ('fp32','int8'):
                        t=time.perf_counter();result=runtimes[name].run(None,inputs)[0];timing[name]=(time.perf_counter()-t)*1000
                        if not np.isfinite(result).all():raise RuntimeError('Nonfinite model output')
                        logits.append(result)
                    t=time.perf_counter()
                    for j,result in enumerate(logits):probabilities[i,j]=torch.from_numpy(result).sigmoid().numpy()[0]
                    row=dict(frame=i,time_seconds=i/fps,decode_ms=decode,preprocess_ms=pre,fp32_model_ms=timing['fp32'],
                             int8_model_ms=timing['int8'],probability_cache_ms=(time.perf_counter()-t)*1000)
                    writer.writerow(row);inference_rows.append(row)
                    if (i+1)%100==0:stream.flush();print(f'inferred={i+1}/{n}',flush=True)
                if cap.read()[0]:raise RuntimeError('Decoded frames exceed independent ffprobe frame count')
        finally:cap.release()
        probabilities.flush();write_json(cache_done,identity)
    # Rendering is independent from inference; float32 caches avoid rounding or
    # model reruns if encoding fails. Every source frame is decoded again in order.
    g=read_json(out/'geometry.json');cap=cv2.VideoCapture(str(source));encoders=[]
    stage_start=time.perf_counter();render_rows=[];frame_metrics=[];previous=None
    best_candidates={};regular=set(np.linspace(0,n-1,12,dtype=int).tolist())
    try:
        for name in VIDEO_NAMES:
            encoders.append(Encoder(out/name,source,w,h,fps_text,n))
        with (out/'render_per_frame.csv').open('w',encoding='utf-8',newline='') as stream:
            fields=['frame','time_seconds','decode_ms','restore_and_analysis_ms','visualization_ms','encoder_pipe_write_ms']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i in range(n):
                t=time.perf_counter();ok,bgr=cap.read();decode=(time.perf_counter()-t)*1000
                if not ok:raise RuntimeError(f'Video render decode failure at frame {i}')
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);t=time.perf_counter()
                masks=[restore(probabilities[i,j],g)>=.5 for j in (0,1)]
                fp,iq=masks;area=iq.sum((1,2));different=(fp!=iq).sum((1,2))
                # These are inspection triggers, not ground-truth accuracy.
                bottom=int(iq[3,int(h*.82):].sum());upper=int(iq[3,:int(h*.4)].sum())
                light=(rgb.min(2)>180)&((rgb.max(2).astype(np.int16)-rgb.min(2))<50)
                light_overlap=int((iq[3]&light).sum())
                changed=np.zeros(4) if previous is None else (previous!=iq).sum((1,2))/(w*h)
                record=dict(frame=i,time_seconds=i/fps,area_pixels=area.tolist(),fp32_int8_disagreement_pixels=different.tolist(),
                            crosswalk_bottom_roi_pixels=bottom,crosswalk_upper_roi_pixels=upper,
                            crosswalk_bright_low_saturation_overlap_pixels=light_overlap,
                            temporal_changed_fraction=changed.tolist())
                frame_metrics.append(record)
                scores=dict(crosswalk_area=int(area[3]),dashboard_suspect=bottom,upper_scene_suspect=upper,
                            bright_surface_suspect=light_overlap,quantization_difference=int(different.sum()),
                            lane_change=float(changed[1]+changed[2]))
                reasons=[]
                for reason,score in scores.items():
                    if reason not in best_candidates or score>best_candidates[reason]['score']:
                        best_candidates[reason]=dict(frame=i,score=score);reasons.append(reason)
                analysis=(time.perf_counter()-t)*1000;t=time.perf_counter()
                frames=canvases(rgb,fp,iq,i,fps);visual=(time.perf_counter()-t)*1000
                for reason in reasons:
                    Image.fromarray(frames[2]).save(out/'frames'/f'candidate_{reason}.jpg',quality=92)
                if i in regular:
                    Image.fromarray(frames[2]).save(out/'frames'/f'frame_{i:05d}_{i/fps:06.2f}s.jpg',quality=92)
                t=time.perf_counter()
                for encoder,frame in zip(encoders,frames):encoder.write(frame)
                write_ms=(time.perf_counter()-t)*1000
                row=dict(frame=i,time_seconds=i/fps,decode_ms=decode,restore_and_analysis_ms=analysis,
                         visualization_ms=visual,encoder_pipe_write_ms=write_ms)
                writer.writerow(row);render_rows.append(row);previous=iq
                if (i+1)%100==0:stream.flush();print(f'rendered_encoded={i+1}/{n}',flush=True)
        enc_timing={e.path.name:e.finish() for e in encoders}
    except BaseException:
        for encoder in encoders:encoder.abort()
        raise
    finally:cap.release()
    render_seconds=time.perf_counter()-stage_start
    write_json(out/'frame_analysis.json',dict(frames=frame_metrics,inspection_candidates=best_candidates,
        caution='ROI/brightness and frame-to-frame changes are heuristic review triggers, not confirmed false positives or accuracy; motion/occlusion can cause changes'))
    # Decode every encoded output, rather than trusting container frame counts.
    outputs={}
    for name in VIDEO_NAMES:
        info=probe(out/name,True);s=next(v for v in info['streams'] if v['codec_type']=='video')
        if s['codec_name']!='h264' or s['pix_fmt']!='yuv420p' or int(s['nb_read_frames'])!=n or s['width']!=w or s['height']!=h or Fraction(s['avg_frame_rate'])!=Fraction(fps_text):
            raise RuntimeError(f'Output codec/frame-count/FPS/dimensions mismatch: {name}')
        outputs[name]=dict(path=str(out/name),sha256=file_hash(out/name),probe=info)
    timings={key:statistics([float(row[key]) for row in inference_rows]) for key in ('decode_ms','preprocess_ms','fp32_model_ms','int8_model_ms','probability_cache_ms')}
    timings.update({key:statistics([float(row[key]) for row in render_rows]) for key in ('restore_and_analysis_ms','visualization_ms','encoder_pipe_write_ms')})
    elapsed=time.perf_counter()-start
    summary=dict(evidence=evidence,frames=n,timing=timings,encoder_timing=enc_timing,outputs=outputs,
        processing_wall_seconds=elapsed,end_to_end_video_frames_per_second=n/elapsed,
        rendering_and_encoding_wall_seconds=render_seconds,rendering_and_encoding_frames_per_second=n/render_seconds,
        measurement_note='Model timings are synchronous ORT session.run only. Pipe timing includes transfer and encoder backpressure, not isolated encoder compute. Wall time includes warmup, cache, visualizations, encoding, full output decode verification. No video ground truth; do not interpret disagreement as accuracy.',
        candidates=best_candidates)
    write_json(out/'summary.json',summary)
    report=f'''# 第二阶段 Candidate-plus 全视频逐帧验证

源视频：{source}。完整 {n} 帧，{w}×{h}，{fps_text} FPS，保持顺序，无跳帧。
输出三段 H.264 / yuv420p MP4，逐段完整解码检查帧数、FPS 和分辨率一致，原音频复制保留。

- road_seg_overlay.mp4：最终 INT8 QAT 四类半透明叠加。
- road_seg_comparison.mp4：原视频、QAT 叠加、四类独立 Mask，共六面板；每面板等比缩放，整体保持原视频分辨率。
- road_seg_fp32_int8_comparison.mp4：原视频、第二阶段 FP32、最终 QAT、阈值差异；差异不是有真值的错误。

绿色=可行驶区域；红色=实线；青色=虚线；洋红=斑马线区域。独立输出可重叠，显示时后者覆盖前者。
RGB、320×192 letterbox（填 114）、/255、sigmoid、去 padding 后双线性还原概率、≥0.5。
无时序平滑、形态学、补线、ROI 屏蔽或额外规则修饰。Mask 边缘放大是训练配套几何还原。

## 模型身份

INT8 ONNX：{evidence['paths']['int8']}

INT8 SHA256：{evidence['sha256']['int8']}

QAT checkpoint：{evidence['paths']['qat_checkpoint']}

QAT checkpoint SHA256：{evidence['sha256']['qat_checkpoint']}，最佳第 {evidence['qat_epoch']} 轮。

FP32 ONNX：{evidence['paths']['fp32']}

FP32 SHA256：{evidence['sha256']['fp32']}。

后端：{evidence['backend']}；不是 RK3566/FPGA 延迟。FP32 是第二阶段优化后、QAT 前基线，差异包含 QAT 微调，不是固定相同权重下的纯量化误差。

## 实测耗时

FP32 模型 mean/p50/p95：{timings['fp32_model_ms']['mean_ms']:.3f}/{timings['fp32_model_ms']['p50_ms']:.3f}/{timings['fp32_model_ms']['p95_ms']:.3f} ms。
INT8 QAT 模型 mean/p50/p95：{timings['int8_model_ms']['mean_ms']:.3f}/{timings['int8_model_ms']['p50_ms']:.3f}/{timings['int8_model_ms']['p95_ms']:.3f} ms。
预处理 mean：{timings['preprocess_ms']['mean_ms']:.3f} ms；还原/分析 mean：{timings['restore_and_analysis_ms']['mean_ms']:.3f} ms；
可视化 mean：{timings['visualization_ms']['mean_ms']:.3f} ms；三路编码管道写入/背压 mean：{timings['encoder_pipe_write_ms']['mean_ms']:.3f} ms。
编码排空时间逐路记录 summary.json。管道时间不是独立测得的编码计算时间，编码器与生产帧重叠执行。
全流程 {elapsed:.2f} s，{n/elapsed:.2f} 源帧/s（含两模型、三路视频、缓存和完整输出解码核验）；渲染/编码阶段 {n/render_seconds:.2f} 源帧/s。
逐帧原始耗时见 inference_per_frame.csv、render_per_frame.csv；原始 float32 概率保存在 cache/probabilities.npy，可独立复核。

## 效果检查

frames/ 保存时间均匀样本以及斑马线面积、底部/上部区域、亮色表面重叠、实虚线变化和量化分歧最大的待检查截图。
这些自动触发不能直接认定仪表台/白车误检或车道闪烁，需要结合画面人工核对。frame_analysis.json 保留完整逐帧统计。
视频没有密集真值，不输出 IoU/F1 或把亮色重叠当作误检率。斑马线监督语义为整个过街区域，含条纹间隙。
人工检查结论会另存 VISUAL_REVIEW.md，区分已观察现象与无法确认的情况。
'''
    (out/'README_RESULT.md').write_text(report,encoding='utf-8')
    print(f'complete frames={n} elapsed={elapsed:.2f}s fps={n/elapsed:.2f} output={out}',flush=True)


if __name__=='__main__':main()
