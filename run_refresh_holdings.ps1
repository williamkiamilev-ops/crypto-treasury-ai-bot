param()

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "venv\Scripts\python.exe"
$script = Join-Path $root "refresh_holdings.py"

if (-not (Test-Path $python)) {
    throw "Python venv not found at $python"
}

if (-not (Test-Path $script)) {
    throw "refresh_holdings.py not found at $script"
}

& $python $script
