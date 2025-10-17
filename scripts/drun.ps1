param(
  [Parameter(Position=0, Mandatory=$true)] [string]$InputPath,
  [Parameter(Position=1, Mandatory=$false)] [string]$OutDir,
  [Parameter(ValueFromRemainingArguments=$true)] [string[]]$Rest
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $repoRoot '.balls\Scripts\python.exe'))) {
  Write-Error "Virtualenv not found at .\.balls. Expected .\.balls\Scripts\python.exe"
  exit 1
}

$in = (Resolve-Path $InputPath).Path
if (-not $OutDir -or $OutDir -eq '') {
  $inParent = Split-Path -Parent $in
  $stem = [System.IO.Path]::GetFileNameWithoutExtension($in)
  $OutDir = Join-Path (Join-Path $inParent 'outs') $stem
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot 'src'))
$python = Join-Path $repoRoot '.balls\Scripts\python.exe'

& $python -m diaremot.cli run -i $in -o $OutDir @Rest
exit $LASTEXITCODE
