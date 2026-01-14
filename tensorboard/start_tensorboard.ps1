param(
    [string]$Run = "",
    [int]$Port = 6006,
    [switch]$List,
    [switch]$All,
    [switch]$NoOpen
)

function Escape-BashSingleQuote([string]$value) {
    return $value -replace "'", "'\\''"
}

$tbRoot = Resolve-Path $PSScriptRoot
$available = Get-ChildItem -Path $tbRoot -Directory -Filter "PPO_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -ExpandProperty Name

if ($List) {
    if (-not $available) {
        Write-Host "No PPO_* folders found under $tbRoot"
        exit 0
    }
    Write-Host "Available PPO runs:"
    $available | ForEach-Object { Write-Host " - $_" }
    exit 0
}

if (-not $All -and -not $Run) {
    $latest = $available | Select-Object -First 1
    if (-not $latest) {
        Write-Error "No PPO_* folders found under $tbRoot"
        exit 1
    }
    $Run = $latest
}

$repoRoot = Resolve-Path (Join-Path $tbRoot "..")
$repoRoot = $repoRoot.Path
$repoRootEsc = Escape-BashSingleQuote $repoRoot
$repoRootWsl = & wsl -e bash -lc "wslpath -a '$repoRootEsc'"
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($repoRootWsl)) {
    Write-Error "Failed to convert repo path to WSL path"
    exit 1
}

if ($All) {
    $logDir = "tensorboard"
} else {
    $logDir = "tensorboard/$Run"
}
$logDirEsc = Escape-BashSingleQuote $logDir
$repoRootWslEsc = Escape-BashSingleQuote ($repoRootWsl.Trim())

Write-Host "Starting TensorBoard for $Run on port $Port"
& wsl -e bash -lc "cd '$repoRootWslEsc' && source venv/bin/activate && tensorboard --logdir '$logDirEsc' --port $Port"

if (-not $NoOpen) {
    Start-Process "http://localhost:$Port"
}
