"""Publish only small measured reports to the existing authorized PR branch."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
REPORTS = 'experiments/training/fpga_road_seg/reports/stage2_results'
BRANCH = 'feature/issue15-fpga-road-seg-20261112'


def publish(output):
    out=Path(output)
    env=os.environ.copy();env['HTTPS_PROXY']='http://127.0.0.1:7890'
    def run(*args):
        result=subprocess.run(list(args),cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode:raise RuntimeError(f'{args[0]} failed: {result.stderr[-2000:]}')
        return result.stdout
    branch=run('git','branch','--show-current').strip()
    if branch!=BRANCH:raise RuntimeError('Refuse to publish results from another branch')
    staged=run('git','diff','--cached','--name-only').splitlines()
    if any(not p.startswith(REPORTS+'/') for p in staged):
        raise RuntimeError('Unrelated staged changes exist; reports left ready for manual review')
    for path in (ROOT/REPORTS).rglob('*'):
        if path.is_file() and (path.stat().st_size>5_000_000 or path.suffix.lower() in ('.pt','.onnx','.zip','.mp4','.npz')):
            raise ValueError('Large/model/dataset artifact in report publication directory')
    run('git','add','-f','--',REPORTS)
    if run('git','diff','--cached','--name-only').strip():
        run('git','commit','-m','results(issue15): publish measured stage-two RLMD and PTQ comparisons')
    run('git','-c','http.proxy=http://127.0.0.1:7890','push','origin',BRANCH)
    record=json.loads(run('gh','pr','view','16','--repo','wante30/adas-edge','--json','body,headRefName'))
    if record['headRefName']!=BRANCH:raise RuntimeError('PR head mismatch')
    marker='<!-- issue15-stage2-results -->'
    body=record['body'].split(marker)[0].split('<!-- issue15-stage2-running -->')[0].rstrip()
    summary=(out/'README_RESULT.md').read_text(encoding='utf-8')
    table='\n'.join(line for line in summary.splitlines() if line.startswith('|'))
    body+='\n\n'+marker+'\n## 第二阶段：真实 RLMD 混合微调\n\n'+table+'\n\n'
    body+='继续训练第一阶段 Candidate-plus best.pt，网络未改。原结果保留。RLMD dense crosswalk 与 BDD 三项分别评估；缺失监督屏蔽。\n\n'
    body+='数据来源、许可限制、分组划分、量化误差与硬件依赖见 `'+REPORTS+'/README_RESULT.md` 和 `FPGA_HANDOFF.md`。权重、数据和视频仅保留本地 outputs。\n'
    bodyfile=out/'pr16_body.md';bodyfile.write_text(body,encoding='utf-8')
    run('gh','pr','edit','16','--repo','wante30/adas-edge','--body-file',str(bodyfile))
    (out/'publication.json').write_text(json.dumps(dict(commit=run('git','rev-parse','HEAD').strip(),pr='https://github.com/wante30/adas-edge/pull/16'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    publish(sys.argv[1])
