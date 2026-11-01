$ErrorActionPreference = 'Stop'

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$DatasetDir = Join-Path $ProjectRoot 'outputs/dataset/style-review-v1'
$ServerScript = Join-Path $ProjectRoot 'runtime/nino_style_review_server.py'
$PythonExe = Join-Path $ProjectRoot '.venv-irodori-gpu/Scripts/python.exe'
$Port = 8766
$Url = "http://127.0.0.1:$Port/"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Project venv Python was not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $ServerScript -PathType Leaf)) {
    throw "Review server script was not found: $ServerScript"
}
$ManifestPath = Join-Path $DatasetDir 'manifest.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Review manifest was not found: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$ExpectedDataset = [string]$Manifest.dataset_id
$ExpectedFingerprint = [string]$Manifest.collection_fingerprint
if (-not $ExpectedDataset -or -not $ExpectedFingerprint) {
    throw 'Manifest is missing dataset_id or collection_fingerprint.'
}

try {
    $Health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get -TimeoutSec 2
    if ($Health.status -eq 'ready' -and
        $Health.dataset_id -eq $ExpectedDataset -and
        $Health.collection_fingerprint -eq $ExpectedFingerprint) {
        Write-Output "Review server already ready: $Url"
        exit 0
    }
    $Observed = "$($Health.dataset_id) / $($Health.collection_fingerprint) / $($Health.status)"
    throw "Port $Port is occupied by a different service ($Observed). No process was stopped."
} catch {
    if ($_.Exception.Message -like "Port $Port is occupied*") { throw }
    $Probe = New-Object System.Net.Sockets.TcpClient
    try {
        $Connect = $Probe.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($Connect.AsyncWaitHandle.WaitOne(500) -and $Probe.Connected) {
            throw "Port $Port is occupied by a service that did not return the expected health response. No process was stopped."
        }
    } finally {
        $Probe.Dispose()
    }
}

Write-Output "Starting review server in this terminal: $Url"
Write-Output 'Stop the server with Ctrl+C.'
Push-Location -LiteralPath $ProjectRoot
try {
    & $PythonExe -u $ServerScript --dataset-dir $DatasetDir --port ([string]$Port)
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
