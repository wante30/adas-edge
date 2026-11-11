param([Parameter(Mandatory=$true)][int]$SupervisorPid)
$ErrorActionPreference = 'Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class SegPowerRequest {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
try {
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: display may turn off normally.
    $segPowerState = [SegPowerRequest]::SetThreadExecutionState(2147483649)
    if ($segPowerState -eq 0) { throw 'Unable to request system awake state' }
    Wait-Process -Id $SupervisorPid -ErrorAction SilentlyContinue
} finally {
    [void][SegPowerRequest]::SetThreadExecutionState(2147483648)
}
