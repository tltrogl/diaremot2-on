param(
  [Parameter(Position=0, Mandatory=$true)] [string]$InputPath,
  [Parameter(Position=1, Mandatory=$true)] [string]$OutDir,
  [Parameter(ValueFromRemainingArguments=$true)] [string[]]$Rest
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not (Test-Path (Join-Path $repoRoot '.balls\Scripts\python.exe'))) {
  Write-Error "Virtualenv not found at .\.balls. Expected .\.balls\Scripts\python.exe"
  exit 1
}

$env:PYTHONPATH = (Resolve-Path (Join-Path $repoRoot 'src'))
$python = Join-Path $repoRoot '.balls\Scripts\python.exe'

& $python -m diaremot.cli run -i $InputPath -o $OutDir @Rest
exit $LASTEXITCODE
