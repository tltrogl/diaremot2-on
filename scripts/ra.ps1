param(
  [Parameter(Position=0, Mandatory=$true)] [string]$InputPath,
  [Parameter(ValueFromRemainingArguments=$true)] [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repoRoot '.balls\Scripts\python.exe'
if (-not (Test-Path $python)) { Write-Error "Venv missing: .\.balls"; exit 1 }
$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot 'src'))

$in = (Resolve-Path $InputPath).Path
$inParent = Split-Path -Parent $in
$stem = [System.IO.Path]::GetFileNameWithoutExtension($in)
$out = Join-Path (Join-Path $inParent 'outs') $stem
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $python -m diaremot.cli run -i $in -o $out @Rest
exit $LASTEXITCODE

