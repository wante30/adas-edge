"""Run an old-vector regression and new 480x640 reference RTL comparisons."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--iverilog',required=True,type=Path)
    ap.add_argument('--vvp',required=True,type=Path)
    ap.add_argument('--jobs',type=int,default=4)
    args=ap.parse_args();root=args.root.resolve()
    compile_result=subprocess.run([str(args.iverilog),'-g2012','-s','tb_avm_legacy_contract','-o','logs/avm_legacy.vvp',
                                   'rtl/avm_4ch_640_stream.v','rtl/tb_avm_legacy_contract.v'],cwd=root,capture_output=True,text=True,check=True)
    (root/'logs/rtl_compile.log').write_text(compile_result.stdout+compile_result.stderr,encoding='utf-8')
    jobs=[]
    for frame in (0,24,49,74,99):
        base=f'4ch/golden/frame_{frame:03d}'
        jobs.append((f'new_frame_{frame:03d}','4ch/lut/lut_readmemh96.hex',
                     [f'{base}/inputs/slot_{i}_{role}.hex' for i,role in enumerate(('front','left','right','rear'))],
                     f'{base}/expected_rgb565.hex'))
    old='reference/legacy_case/4ch'
    jobs.append(('original_archive_frame_000',f'{old}/lut/lut_readmemh96.hex',
                 [f'{old}/golden/frame_000/inputs/slot_{i}_{role}.hex' for i,role in enumerate(('front_right','side_left','side_right','rear_center'))],
                 f'{old}/golden/frame_000/expected_rgb565.hex'))
    jobs.append(('source_diagnostic','diagnostic/lut/lut_readmemh96.hex',jobs[0][2],'diagnostic/expected_rgb565.hex'))
    def run(job):
        name,lut,images,expected=job
        command=[str(args.vvp),'logs/avm_legacy.vvp',f'+LUT={lut}']
        command += [f'+SRC{i}={path}' for i,path in enumerate(images)]
        command += [f'+EXPECTED={expected}']
        result=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=600)
        log=result.stdout+result.stderr
        (root/'logs'/f'rtl_{name}.log').write_text(log,encoding='utf-8')
        m=re.search(r'PASS: legacy contract 480x640 pixels=(\d+) mismatches=(\d+) cycles=(\d+)',log)
        if result.returncode or m is None or 'FAIL' in log or 'MISMATCH' in log:
            raise AssertionError(f'{name}: {log}')
        row=dict(case=name,pixels=int(m[1]),mismatches=int(m[2]),cycles=int(m[3]))
        assert row['pixels']==307200 and row['mismatches']==0
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=max(1,min(args.jobs,4))) as pool:
        for future in as_completed([pool.submit(run,job) for job in jobs]):
            rows.append(future.result())
            print(json.dumps(rows[-1]),flush=True)
            (root/'logs/rtl_progress.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    v=json.loads((root/'validation.json').read_text(encoding='utf-8'))
    v['status']='LEGACY_CONTRACT_PC_AND_REFERENCE_RTL_PASS'
    v['rtl']=sorted(rows,key=lambda row:row['case'])
    v['board_verified']=False
    (root/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ALL LEGACY/NEW/DIAGNOSTIC RTL CASES PASS',flush=True)


if __name__=='__main__':main()
