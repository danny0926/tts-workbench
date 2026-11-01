param([string]$Destination = '')
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Destination) { $Destination = Join-Path $RepoRoot 'vendor/irodori-tts' }
$PinnedCommit = '59be938c99d48b5653ca5aa3fdd0cddaa56940e5'
if (-not (Test-Path -LiteralPath $Destination)) {
    & git clone --no-checkout 'https://github.com/Aratako/Irodori-TTS.git' $Destination
    if ($LASTEXITCODE -ne 0) { throw 'Upstream clone failed.' }
} else {
    $Changes = & git -C $Destination status --porcelain
    if ($LASTEXITCODE -ne 0 -or $Changes) { throw 'Existing upstream checkout is invalid or has local changes.' }
    $Remote = & git -C $Destination remote get-url origin
    if ($LASTEXITCODE -ne 0 -or $Remote -notin @('https://github.com/Aratako/Irodori-TTS.git', 'https://github.com/Aratako/Irodori-TTS')) {
        throw 'Existing checkout has a different upstream origin.'
    }
}
& git -C $Destination fetch origin $PinnedCommit
if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream fetch failed.' }
& git -C $Destination checkout --detach $PinnedCommit
if ($LASTEXITCODE -ne 0) { throw 'Pinned upstream checkout failed.' }
$Effective = & git -C $Destination rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $Effective -ne $PinnedCommit) { throw 'Pinned upstream verification failed.' }
Write-Output "Irodori upstream ready: $Effective"
