param(
    [string]$TaskName = "CryptoAgentContinuous",
    [double]$Days = 7,
    [int]$PollSeconds = 300,
    [int]$MaxTrades = 100,
    [switch]$PreventSleep
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $root "run_continuous_agent.ps1"
$refreshRunner = Join-Path $root "run_refresh_holdings.ps1"
if (-not (Test-Path $runner)) {
    throw "Runner script not found: $runner"
}
if (-not (Test-Path $refreshRunner)) {
    throw "Refresh runner script not found: $refreshRunner"
}

$ps = "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
$preventSleepArg = ""
if ($PreventSleep) {
    $preventSleepArg = " -PreventSleep"
}

$taskArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Days $Days -PollSeconds $PollSeconds -MaxTrades $MaxTrades$preventSleepArg"
$refreshTaskArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$refreshRunner`""

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650)

$action = New-ScheduledTaskAction -Execute $ps -Argument $taskArgs
$refreshAction = New-ScheduledTaskAction -Execute $ps -Argument $refreshTaskArgs

# Run on system startup (survives logout)
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$startupPrincipal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

Register-ScheduledTask `
    -TaskName "${TaskName}-Startup" `
    -Action $action `
    -Trigger $startupTrigger `
    -Settings $settings `
    -Principal $startupPrincipal `
    -Force | Out-Null

# Run at current user logon as backup trigger
$logonTrigger = New-ScheduledTaskTrigger -AtLogOn
$logonPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType InteractiveToken -RunLevel Highest

Register-ScheduledTask `
    -TaskName "${TaskName}-Logon" `
    -Action $action `
    -Trigger $logonTrigger `
    -Settings $settings `
    -Principal $logonPrincipal `
    -Force | Out-Null

$dailyTrigger = New-ScheduledTaskTrigger -Daily -At 3am

Register-ScheduledTask `
    -TaskName "${TaskName}-NightlyHoldingsRefresh" `
    -Action $refreshAction `
    -Trigger $dailyTrigger `
    -Settings $settings `
    -Principal $startupPrincipal `
    -Force | Out-Null

Write-Host "Scheduled tasks created:"
Write-Host " - ${TaskName}-Startup (SYSTEM, AtStartup)"
Write-Host " - ${TaskName}-Logon (User, AtLogOn)"
Write-Host " - ${TaskName}-NightlyHoldingsRefresh (SYSTEM, Daily 3:00 AM)"
Write-Host ""
Write-Host "Use these commands:"
Write-Host "Start-ScheduledTask -TaskName '${TaskName}-Startup'"
Write-Host "Get-ScheduledTask -TaskName '${TaskName}-*'"
