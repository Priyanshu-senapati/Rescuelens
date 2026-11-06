# Starts a training script as its own process (keeps running if this terminal or session closes).
#   powershell -ExecutionPolicy Bypass -File training\run_local.ps1 -Script train_classifier.py -Quick 0
# Output: training\<script>.log (progress) and training\<script>.log.err (warnings / errors). Dataset: ..\data
param([string]$Script = "train_classifier.py", [string]$Quick = "0", [string]$EvalOnly = "0")
$root = Split-Path $PSScriptRoot -Parent
$env:DATA_ROOT = Join-Path (Split-Path $root -Parent) "data"
if (-not $env:EXPORT_DIR) { $env:EXPORT_DIR = "models" }   # override to keep a second run separate
$env:CACHE_DIR = "training/.cache"
$env:QUICK = $Quick
$env:EVAL_ONLY = $EvalOnly
$log = "training\" + [IO.Path]::GetFileNameWithoutExtension($Script) + "_" + $env:EXPORT_DIR + ".log"
$p = Start-Process -FilePath "$root\training\.venv\Scripts\python.exe" -ArgumentList "-u", "training\$Script" `
    -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden -PassThru
Write-Output "started $Script (pid $($p.Id)) -> $log"
