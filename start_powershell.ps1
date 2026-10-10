$ErrorActionPreference = "Stop"
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $AppRoot

# Um proxy temporário de WSL/VPN pode apontar para 127.0.0.1 depois que ele
# já foi encerrado. Remova-o apenas do processo que iniciará o DubStudio.
foreach ($name in "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY") {
  Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}

$PythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCommand) {
  Write-Host "Python não foi encontrado. Instale Python 3.11 ou 3.12." -ForegroundColor Red
  exit 1
}

$PythonVersion = & $PythonCommand.Source -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
if ($PythonVersion -notin @("3.11", "3.12")) {
  Write-Host "Este Studio precisa de Python 3.11 ou 3.12. Encontrado: $PythonVersion" -ForegroundColor Red
  exit 1
}
if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
  try {
    $ExistingStudio = Invoke-RestMethod -Uri "http://127.0.0.1:8765/api/health" -TimeoutSec 3
    if ($ExistingStudio.version -like "3.*") { Start-Process "http://127.0.0.1:8765"; exit 0 }
  } catch {}
  Write-Host "A porta 8765 está ocupada. Encerre a versão antiga do Studio antes de abrir a nova." -ForegroundColor Yellow
  exit 1
}
$RuntimePath = Join-Path $AppRoot "runtime"
New-Item -ItemType Directory -Path $RuntimePath -Force | Out-Null
$LogStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Process = Start-Process -FilePath $PythonCommand.Source -ArgumentList @("-u", "server.py", "--host", "127.0.0.1", "--port", "8765") -WorkingDirectory $AppRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RuntimePath "studio-$LogStamp.stdout.log") -RedirectStandardError (Join-Path $RuntimePath "studio-$LogStamp.stderr.log") -PassThru
Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:8765"
if ($Process.HasExited) { Write-Host "Falha ao abrir o Studio. Consulte os logs em runtime." -ForegroundColor Red; exit 1 }
Write-Host "DubStudio Colab iniciado. PID: $($Process.Id)" -ForegroundColor Cyan
Write-Host "Para encerrar: Stop-Process -Id $($Process.Id)"
