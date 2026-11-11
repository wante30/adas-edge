param(
    [int]$ExistingPipelinePid = 0,
    [string]$Python = 'C:\Users\JingyuWang\miniconda3\envs\yolo\python.exe',
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$segRepo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location -LiteralPath $segRepo
$segOutput = Join-Path $segRepo 'outputs\fpga_road_seg_final'
New-Item -ItemType Directory -Path $segOutput -Force | Out-Null
$segLog = Join-Path $segOutput 'supervisor.log'
if ($ExistingPipelinePid -gt 0) {
    "$(Get-Date -Format o) Waiting for pipeline PID $ExistingPipelinePid" | Add-Content -LiteralPath $segLog -Encoding utf8
    Wait-Process -Id $ExistingPipelinePid -ErrorAction SilentlyContinue
}
for ($segAttempt = 1; $segAttempt -le 3; $segAttempt++) {
    "$(Get-Date -Format o) Recovery/delivery attempt $segAttempt" | Add-Content -LiteralPath $segLog -Encoding utf8
    & $Python -u -m experiments.training.fpga_road_seg.run_pipeline *>> $segLog
    & $Python -u -m experiments.training.fpga_road_seg.run_pipeline --postprocess-only *>> $segLog
    if ($LASTEXITCODE -eq 0 -and $Publish) {
        & $Python -u -m experiments.training.fpga_road_seg.publish_results --output $segOutput --push *>> $segLog
    }
    if ($LASTEXITCODE -eq 0) {
        "$(Get-Date -Format o) Delivery completed" | Add-Content -LiteralPath $segLog -Encoding utf8
        exit 0
    }
    Start-Sleep -Seconds 30
}
throw 'Persistent pipeline exhausted recovery attempts; inspect supervisor.log and run.log'
