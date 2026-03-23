param(
    [double]$Days = 7,
    [int]$PollSeconds = 300,
    [int]$MaxTrades = 100,
    [string]$Query = "",
    [switch]$PreventSleep
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"
$main = Join-Path $root "main.py"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}

if (-not (Test-Path $main)) {
    throw "main.py not found at $main"
}

if ([string]::IsNullOrWhiteSpace($Query)) {
    $Query = Read-Host "Enter the trading objective"
}

Write-Host "Continuous session ready:"
Write-Host " - Days: $Days"
Write-Host " - PollSeconds: $PollSeconds"
Write-Host " - MaxTrades: $MaxTrades"
Write-Host " - Query: $Query"
$confirm = Read-Host "Start session? [y/N]"
if ($confirm -notin @("y", "Y", "yes", "YES", "Yes")) {
    throw "Continuous session cancelled."
}

if ($PreventSleep) {
    Add-Type -Namespace Win32 -Name NativeMethods -MemberDefinition @"
using System;
using System.Runtime.InteropServices;
public static class NativeMethods {
  [DllImport("kernel32.dll", SetLastError=true)]
  public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
    $ES_CONTINUOUS = 0x80000000
    $ES_SYSTEM_REQUIRED = 0x00000001
    $ES_AWAYMODE_REQUIRED = 0x00000040
    [void][Win32.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_AWAYMODE_REQUIRED)
}

try {
    & $python $main `
        --continuous-days $Days `
        --poll-seconds $PollSeconds `
        --max-trades $MaxTrades `
        --query $Query
}
finally {
    if ($PreventSleep) {
        $ES_CONTINUOUS = 0x80000000
        [void][Win32.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS)
    }
}
