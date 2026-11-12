"""Plateau-triggered lightweight drivable distillation, with unchanged inference graph."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .dataset_adapter import read_json, write_json, file_hash
from .train import load_checkpoint, selection_score

ROOT=Path(__file__).resolve().parents[3]


def maybe_refine(output, manifest):
    out=Path(output);marker=out/'refinement.json'
    if marker.exists():return read_json(marker)
    report=read_json(out/'checkpoints/candidate-plus/report.json')
    history=report['history'];last=history[-1]
    gain=last['selection_score']-history[max(0,len(history)-4)]['selection_score']
    crosswalk=last['validation']['domains']['rlmd']['classes']['crosswalk']['iou']
    teacher=ROOT/'outputs/fpga_road_seg_final/checkpoints/twin-small/best.pt'
    if len(history)<10 or gain>=.004 or crosswalk>=.5 or not teacher.exists():
        result=dict(triggered=False,reason='No sustained plateau below crosswalk IoU 0.5, or teacher unavailable',last_three_interval_score_gain=gain)
        write_json(marker,result);return result
    base=out/'checkpoints/candidate-plus/best.pt'
    if not base.exists():
        result=dict(triggered=False,reason='No eligible mixed checkpoint to initialize distillation')
        write_json(marker,result);return result
    retained=out/'checkpoints/mixed_pre_distillation';retained.mkdir(parents=True,exist_ok=True)
    if not (retained/'best.pt').exists():shutil.copyfile(base,retained/'best.pt')
    cfg=report['config'].copy()
    cfg.update(lr=.00005,epochs=3,min_epochs=3,patience=2,epoch_samples=32000,distillation_weight=.2)
    config=out/'distillation_config.json';write_json(config,cfg)
    target=out/'checkpoints/candidate-plus-distillation'
    command=[sys.executable,'-u','-m','experiments.training.fpga_road_seg.train','--manifest',str(manifest),
        '--config',str(config),'--output',str(target),'--amp','--workers','4','--device','cuda','--teacher',str(teacher)]
    last_checkpoint=target/'last.pt'
    command+=['--resume',str(last_checkpoint)] if last_checkpoint.exists() else ['--init',str(retained/'best.pt')]
    print('plateau -> bounded drivable distillation from existing Twin Small, no architecture change',flush=True)
    code=subprocess.run(command,cwd=ROOT).returncode
    result=dict(triggered=True,teacher_sha256=file_hash(teacher),last_three_interval_score_gain=gain,exit_code=code,
        method='Existing trained Twin Small drivable logits; only genuinely supervised BDD pixels. Real RLMD crosswalk/solid/dashed loss continues. Max 3 epochs; same inference model/MACs.',selected=False)
    if code==0 and (target/'best.pt').exists():
        _,previous=load_checkpoint(retained/'best.pt');_,candidate=load_checkpoint(target/'best.pt')
        original_baseline=previous['baseline_domains']
        old_domains=previous['history'][previous['epoch']-1]['validation']['domains']
        new_domains=candidate['history'][candidate['epoch']-1]['validation']['domains']
        old_score,_=selection_score(old_domains,original_baseline,.025)
        new_score,eligible=selection_score(new_domains,original_baseline,.025)
        result.update(previous_score=old_score,candidate_score=new_score,bdd_guard_passed=eligible)
        if eligible and new_score>old_score+.001:
            shutil.copyfile(target/'best.pt',base)
            result['selected']=True
    if code:
        # Keep prior valid checkpoint available, but do not hide a failed refinement.
        result['reason']='Distillation failed; original mixed best retained, inspect run.log; rerun distillation explicitly to recover'
    write_json(marker,result);return result
